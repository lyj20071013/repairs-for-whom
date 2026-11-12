"""漂移样本的逐条分类（2026-11-08 写；第 6 篇；登记补充五第 5 条；只作描述，不进任何判定）。

为什么：第一步动作的 TV 把文字回复合成一类（或按逐字相同比较），看不出 π1 的文字回复在"做法"上有没有变。
这里把每个第一步样本归到一个事先写死的类别，再在类别分布上算 TV，与 π0 自己两次抽样（4B 与 4B#2）的噪声底比较。

类别（写死；补充五第 5 条）：
  工具调用（按工具名确定，不调用 27B）：
    T1 调用查询类工具（不改数据库，含 calculate）   T2 调用写操作工具（firstbad.WRITE_TOOLS）   T3 调用 transfer_to_human_agents
  文字回复（27B 判，只看回复本身）：
    A 索取身份验证信息（邮箱、姓名 + 邮编、用户 id）
    B 索取或澄清办理所需的其他信息（订单号、预订号、要改的商品 / 航班 / 乘客、付款方式、原因等）
    C 列出具体操作的细节，请顾客确认是否执行（yes / no）
    D 只告知信息或结果（转述查询结果、说明政策或选项），不向顾客索取任何东西
    E 拒绝或说明无法办理
    F 结束对话、告别、问"还有别的需要吗"，或说明将转接人工
    G 其他
  一条回复做了几件事时，按"它要顾客下一步做什么"定，优先级 C > A > B > E > D > F > G。

做法（事先写定）：
  - 盲判：发给 27B 的只有回复文字，不含模型名、位置、对话；相同的文字只判一次（温度 0），各样本按文字取结果。
  - 打乱顺序：唯一文字按 sha256("DRIFTCLS|" + 文字) 排序后发送。
  - 截断：drift.jsonl 里的文字是小写、去多余空白、整条截到 300 个字符（canon[:300]）的；长度达到 300 的记为"被截断"。
    报告各模型被截断的比例，TV 按"全部样本""只用未截断样本""只用截断样本"三组分别算。
  - TV：每个位置上两个模型的类别分布的总变差距离，在位置上平均；噪声底 = TV(4B, 4B#2)，与每个 π1 在同一批位置上算。
    另报众数类别变化率与各模型的类别占比。分组的 TV 只用两边各有 ≥ 3 个该组样本的位置。
  - 27B 的回答不是 A–G 中的一个字母时记为 "?"，单独计数。

用法（14b2，tau2_pilot 下，τ² 的 venv；27B 在 8003；可以与 π1 续跑同时跑）：
  python classify_drift.py classify --drift pool/drift.jsonl --port 8003 --out pool/drift_cls.jsonl
  python classify_drift.py report   --drift pool/drift.jsonl --cls pool/drift_cls.jsonl --design pool/design_run.jsonl --out pool/report_drift_cls.md
正常日志：classify 先打印"唯一文字 n 个（已判 m 个），待判 k 个"，每 500 个一行"完成 x/k（无效回答 e）"，最后 "== 完成"。中断后重跑同一条命令只补没判的。
"""
import argparse
import collections
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDEFG"
PROMPT = """Below is one message that a customer-service agent (retail or airline) sent to a customer. The text is lowercased and may be cut off at 300 characters.

=== Message ===
{text}
=== End ===

Classify what this message mainly does. Categories:
A: asks the customer for identity-verification information (email, name and zip code, or user id)
B: asks the customer for, or to clarify, other information needed to proceed (order or reservation id, which item / flight / passenger to change, payment method, reason, etc.)
C: lists the details of a specific action and asks the customer to confirm whether to proceed (yes / no)
D: only gives information or results (reports lookup results, explains policy or options) without asking the customer for anything
E: refuses, or says the request cannot be done
F: ends the conversation, says goodbye, asks whether there is anything else, or says the customer will be transferred to a human agent
G: none of the above

If the message does several things, decide by what it asks the customer to do next, with priority C > A > B > E > D > F > G.
Answer with exactly one letter."""


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_names():
    sys.path.insert(0, str(HERE))
    from firstbad import WRITE_TOOLS
    return set().union(*WRITE_TOOLS.values())


def tool_cat(canon, writes):
    name = canon[2:].split("(", 1)[0].strip()
    if name == "transfer_to_human_agents":
        return "T3"
    return "T2" if name in writes else "T1"


def ask(port, name, text):
    body = {"model": name, "messages": [{"role": "user", "content": PROMPT.format(text=text)}], "temperature": 0.0,
            "max_tokens": 4, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())
            raw = ((out["choices"][0].get("message") or {}).get("content") or "").strip()
            c = raw[:1].upper()
            return (c if c in LETTERS else "?"), raw[:20]
        except Exception as e:                       # 服务出错：重试，三次都失败就跳过（下次重跑再判）
            err = repr(e)[:200]
            time.sleep(5)
    raise RuntimeError(err)


def cmd_classify(a):
    texts = sorted({r["canon"][2:] for r in jl(a.drift) if r["canon"].startswith("X:")}, key=lambda t: h("DRIFTCLS|" + t))
    out = Path(a.out)
    done = {r["h"] for r in jl(out)} if out.exists() else set()
    todo = [t for t in texts if h(t) not in done]
    print(f"唯一文字 {len(texts)} 个（已判 {len(done)} 个），待判 {len(todo)} 个；并发 {a.workers}", flush=True)
    if a.dry_run:
        for t in todo[: a.dry_run]:
            print("----\n" + PROMPT.format(text=t))
        return 0
    n = bad = fail = 0
    with ThreadPoolExecutor(a.workers) as ex, open(out, "a", encoding="utf-8") as f:
        futs = {ex.submit(ask, a.port, a.name, t): t for t in todo}
        for fu in as_completed(futs):
            t = futs[fu]
            try:
                c, raw = fu.result()
            except Exception as e:
                fail += 1
                continue
            f.write(json.dumps({"h": h(t), "cat": c, "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            bad += c == "?"
            if n % 500 == 0 or n == len(todo) - fail:
                print(f"  {time.strftime('%H:%M:%S')} 完成 {n}/{len(todo)}（无效回答 {bad}，请求失败 {fail}）", flush=True)
    print("== 完成" if fail == 0 else f"== 有 {fail} 个请求失败（重跑同一条命令只补这些）")
    return 0 if fail == 0 else 1


def tv(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))


def mean(v):
    return sum(v) / len(v) if v else None


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def cmd_report(a):
    writes = write_names()
    cls = {r["h"]: r["cat"] for r in jl(a.cls)}
    fold = {d["pos_id"]: d.get("fold") for d in jl(a.design)} if a.design else {}
    S = collections.defaultdict(lambda: collections.defaultdict(list))      # (模型, 次) -> 位置 -> [(类别, 是否截断)]
    miss = 0
    for r in jl(a.drift):
        c = r["canon"]
        if c.startswith("T:"):
            cat, tr = tool_cat(c, writes), False
        elif c.startswith("X:"):
            cat = cls.get(h(c[2:]))
            tr = len(c) >= 300
            if cat is None:
                miss += 1
                continue
        else:
            cat, tr = "INVALID", False
        S[(r["model"], r["rep"])][r["pos_id"]].append((cat, tr))
    L = ["# 漂移样本的逐条分类（classify_drift.py；登记补充五第 5 条；只作描述）", "",
         f"没有分类结果的文字样本 {miss} 个（应为 0）。类别：T1 查询类调用、T2 写操作调用、T3 转接人工调用；文字 A 身份验证、B 其他信息、"
         f"C 请确认、D 只告知、E 拒绝、F 结束 / 转人工、G 其他、? 无效回答。", "",
         "## 各模型的类别占比与截断比例", "", "| 模型 | 样本 | " + " | ".join(["T1", "T2", "T3"] + list(LETTERS) + ["?"]) + " | 文字中被截断 |",
         "|---|---:|" + "---:|" * 12]
    for key in sorted(S):
        xs = [x for v in S[key].values() for x in v]
        cnt = collections.Counter(c for c, _ in xs)
        txt = [t for c, t in xs if c in LETTERS or c == "?"]
        L.append(f"| {key[0]}#{key[1]} | {len(xs)} | " + " | ".join(f3(cnt[c] / len(xs)) for c in ["T1", "T2", "T3"] + list(LETTERS) + ["?"])
                 + f" | {f3(sum(txt) / len(txt) if txt else None)} |")
    base, noise = S.get((a.pi0, "1"), {}), S.get((a.pi0, "2"), {})
    L += ["", "## TV（类别分布，位置上平均）：π1 与 π0，噪声底 = π0 两次抽样，同一批位置", "",
          "| 比较 | 组 | 位置 | TV(π1, π0) | 噪声底 TV(π0#2, π0) | 众数类别变化率 | 噪声底 |", "|---|---|---:|---:|---:|---:|---:|"]
    res = {}
    for m in [k for k in sorted(S) if k[0] != a.pi0]:
        o = S[m]
        for grp, pick, mn in (("全部样本", lambda xs: xs, 1), ("只用未截断", lambda xs: [x for x in xs if not x[1]], 3),
                              ("只用截断", lambda xs: [x for x in xs if x[1]], 3)):
            t1, t0, m1, m0, n = [], [], [], [], 0
            for p in o:
                if p not in base or p not in noise:
                    continue
                b, z, q = [c for c, _ in pick(base[p])], [c for c, _ in pick(noise[p])], [c for c, _ in pick(o[p])]
                if min(len(b), len(z), len(q)) < mn:
                    continue
                n += 1
                t1.append(tv(q, b)); t0.append(tv(z, b))
                mode = lambda xs: collections.Counter(xs).most_common(1)[0][0]
                m1.append(float(mode(q) != mode(b))); m0.append(float(mode(z) != mode(b)))
            res[f"{m[0]}#{m[1]}|{grp}"] = {"n": n, "tv": mean(t1), "noise": mean(t0), "mode": mean(m1), "mode_noise": mean(m0)}
            L.append(f"| {m[0]}#{m[1]} 对 {a.pi0} | {grp} | {n} | {f3(mean(t1))} | {f3(mean(t0))} | {f3(mean(m1))} | {f3(mean(m0))} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(str(a.out).rsplit(".", 1)[0] + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("--drift", default="pool/drift.jsonl")
    c.add_argument("--port", type=int, default=8003)
    c.add_argument("--name", default="Qwen3.8-27B-FP8")
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--dry-run", type=int, default=0, help="只打印前 N 个提示，不调用 27B")
    c.add_argument("--out", default="pool/drift_cls.jsonl")
    r = sub.add_parser("report")
    r.add_argument("--drift", default="pool/drift.jsonl")
    r.add_argument("--cls", default="pool/drift_cls.jsonl")
    r.add_argument("--design", default="pool/design_run.jsonl")
    r.add_argument("--pi0", default="4B")
    r.add_argument("--out", default="pool/report_drift_cls.md")
    a = ap.parse_args()
    return {"classify": cmd_classify, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
