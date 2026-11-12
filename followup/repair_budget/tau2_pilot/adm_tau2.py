"""τ² 候选的可执行性判定（2026-11-05 写；第 6 篇 τ² 主实验准备第 1 步）。
从 train_tau2.py 的 export 里抽出来（原来的 adm_fn、missing_args、judge 等），供 design 之后的标注、export、分析共用。

候选 c 在位置 t 可执行，当且仅当（登记_τ²探针 补充四第 2 节）：
  1. 参数可得：c 的每个标量参数值（字符串、数字；不含布尔与空值）出现在该位置之前部署者能看到的文字里，
     或该领域的工具 schema 里；
  2. 写操作已获确认：c 含写操作（firstbad.WRITE_TOOLS）时，27B（温度 0，关思考）判断用户已明确同意，答 YES。
  只有文字、没有工具调用的候选总是可执行。

规则版本（--rule）：
  v1  探针原样（默认；探针的标签与敏感性分析都用它）。"能看到的文字" = 前缀里的消息、工具调用参数、工具结果
      （不含系统提示）；比较 = 不区分大小写的子串。
  v2  候选规则，是否用于主实验由主实验登记决定（在探针 448 个训练候选里看到的 v1 误判，只看了参数，没看任何续跑）：
      a) 能看到的文字加上系统提示里的领域政策（例：airline 政策里写着当前日期 2024-05-15）；
      b) 自由文本参数不查：transfer_to_human_agents.summary（v1 下 9 个 4B 提议因此被判不可执行）；
         calculate.expression 只查其中的数字；
      c) 归一化：去掉值开头的 "#"（订单号），数字按数值比较（67 与 67.0），
         匹配要求两边不是字母数字（v1 的子串匹配会让 "1" 命中任何含 1 的文字）。
      其余（日期格式不同、计算出的金额等）v1 / v2 相同：判为参数不可得。
  v3  = v2 + 先验证身份（11-05 加；用户提议）：retail 政策要求对话开始时先用 email 或姓名 + 邮编找到用户 id，
      "This has to be done even when the user already provides the user id"，验证之后才能提供订单、商品、资料信息。
      所以 retail 里除 AUTH_EXEMPT 以外的工具调用，要求前缀里已有一次成功的 find_user_id_by_email / find_user_id_by_name_zip
      （工具结果不是错误），否则判 why = not_authenticated。airline 的政策没有身份验证这一步，v3 与 v2 相同。
      判定顺序：参数可得 → 身份验证 → 写操作已确认（没验证身份的写操作不再问 27B）。

用法（tau2_pilot/tau2-bench 下，τ² 的 venv；写操作要 27B 服务 8003）：
  python ../adm_tau2.py annotate --design ../probe/design_eval_ret.jsonl ../probe/design_eval_air.jsonl \
      --judge-port 8003 --rule v1 --out ../probe/adm_eval_v1.jsonl
  python ../adm_tau2.py compare --a ../probe/samples_s4b.jsonl.admissible.jsonl --b ../probe/adm_train_v1.jsonl
输出（每个候选一行）：{"pos_id","cand_id","kinds","tools","ok","why","missing"?,"judge"?,"rule","judged_by"}
  why ∈ null / args_missing / not_authenticated（v3）/ not_confirmed；judged_by ∈ none（不是写操作或参数不可得）/ 27b / cache。
正常日志：annotate 每 20 个位置一行"[i/N] … 候选 n、可执行 m、27B 调用 j"，最后一行统计与"== 完成"。
"""
import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULES = ("v1", "v2", "v3")


def jl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def vllm_chat(port, served, msgs, tools, temp, seed, n=1, tool_choice=None, max_tokens=2048, timeout=300):
    import urllib.request
    body = {"model": served, "messages": msgs, "temperature": temp, "seed": seed, "n": n, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["choices"]


def sseed(*xs):
    import hashlib
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


WRITE_NAMES = None


def write_names():
    global WRITE_NAMES
    if WRITE_NAMES is None:
        sys.path.insert(0, str(HERE))
        from firstbad import WRITE_TOOLS
        WRITE_NAMES = set().union(*WRITE_TOOLS.values())
    return WRITE_NAMES


def scalars(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from scalars(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from scalars(v)
    elif x is None or isinstance(x, bool):
        return
    else:
        yield str(x)


def ctx_text(msgs):
    """litellm 格式的对话（不含系统提示）里部署者能看到的全部文字：消息、工具调用参数、工具结果。"""
    parts = []
    for m in msgs:
        if m.get("role") == "system":
            continue
        parts.append(str(m.get("content") or ""))
        for tc in (m.get("tool_calls") or []):
            f = tc.get("function", tc)
            parts.append(json.dumps(f.get("arguments"), ensure_ascii=False) if not isinstance(f.get("arguments"), str)
                         else f["arguments"])
    return "\n".join(parts)


def call_list(msg):
    """候选 / 目标消息里的 [(工具名, 参数 dict)]；兼容 τ² 的 dict（name/arguments）与 litellm 的 function 格式。"""
    out = []
    for tc in (msg.get("tool_calls") or []):
        f = tc.get("function", tc)
        args = f.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        out.append((f.get("name") or tc.get("name"), args or {}))
    return out


def missing_args(calls, ctx, schema_text):
    """v1（探针原样）。"""
    low_ctx, low_sch = ctx.lower(), schema_text.lower()
    return [v for _, args in calls for v in scalars(args) if v.lower() not in low_ctx and v.lower() not in low_sch]


# ---------------------------------------------------------------- v2（候选规则；见文件头）
FREE_TEXT = {("transfer_to_human_agents", "summary")}
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")


def _num_set(text):
    out = set()
    for m in NUM.finditer(text):
        try:
            out.add(round(float(m.group()), 6))
        except ValueError:
            pass
    return out


def _found_v2(v, low_text, nums):
    s = v.strip().lstrip("#").lower()
    if not s:
        return True
    try:
        if round(float(s), 6) in nums:
            return True
    except ValueError:
        pass
    return re.search(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])", low_text) is not None


def missing_args_v2(calls, ctx, schema_text):
    text = (ctx + "\n" + schema_text).lower()
    nums = _num_set(text)
    miss = []
    for name, args in calls:
        for k, v in (args or {}).items():
            if (name, k) in FREE_TEXT:
                continue
            if name == "calculate" and k == "expression":
                miss += [x for x in NUM.findall(str(v)) if not _found_v2(x, text, nums)]
                continue
            miss += [x for x in scalars(v) if not _found_v2(x, text, nums)]
    return miss


AUTH_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
AUTH_EXEMPT = AUTH_TOOLS | {"transfer_to_human_agents", "calculate"}
AUTH_DOMAINS = {"retail"}


def authenticated(prefix):
    """τ² 消息列表里是否已有一次成功的身份验证调用（按工具调用 id 找到对应的工具结果，error 为假）。"""
    names = {}
    for m in prefix:
        for tc in (getattr(m, "tool_calls", None) or []):
            names[tc.id] = tc.name
        if getattr(m, "role", None) == "tool" and names.get(getattr(m, "id", None)) in AUTH_TOOLS \
                and not getattr(m, "error", False):
            return True
    return False


def transcript(msgs, n=14, cut=400):
    lines = []
    for m in [m for m in msgs if m.get("role") != "system"][-n:]:
        r = m.get("role")
        c = str(m.get("content") or "").strip()
        if r == "user":
            lines.append(f"USER: {c[:1500]}")
        elif r == "assistant":
            if c:
                lines.append(f"AGENT: {c[:1500]}")
            for name, args in call_list(m):
                lines.append(f"AGENT CALLS {name}({json.dumps(args, ensure_ascii=False)[:600]})")
        elif r == "tool":
            lines.append(f"TOOL RESULT: {c[:cut]}")
    return "\n".join(lines)


JUDGE_CONFIRM = ("Below is a conversation between a customer-service agent and a user, followed by an action the agent is "
                 "about to take. The agent's policy requires that, before any action that changes the database, the agent "
                 "lists the action details and obtains the user's explicit confirmation (yes).\n\n=== Conversation ===\n{conv}\n\n"
                 "=== Action the agent is about to take ===\n{act}\n\nBased only on the conversation above, has the user "
                 "already explicitly confirmed this specific action with these details? Answer with exactly YES or NO.")


def judge(port, served, prompt, seed):
    c = vllm_chat(port, served, [{"role": "user", "content": prompt}], None, 0.0, seed, max_tokens=4)[0]
    t = ((c.get("message") or {}).get("content") or "").strip().upper()
    return t.startswith("YES"), t[:20]


def act_str(calls):
    return "; ".join(f"{n}({json.dumps(g, ensure_ascii=False)})" for n, g in calls)


def load_cache(paths):
    """以前的判定（*.admissible.jsonl 或本脚本的输出）：{(pos_id, cand_id): 27B 的原始回答}。只复用 27B 的回答。"""
    out = {}
    for p in paths or []:
        for r in jl(p):
            if r.get("judge") is not None:
                out[(r["pos_id"], r["cand_id"])] = r["judge"]
    return out


class Admissibility:
    """check(d, c) -> 一行判定；d 是设计文件的一行，c 是其中一个候选。需要在 tau2-bench 目录下运行（d["src"] 是相对路径）。"""

    def __init__(self, rule="v1", judge_port=0, judge_name="Qwen3.8-27B-FP8", cache=None):
        assert rule in RULES, rule
        sys.path.insert(0, str(HERE))
        import pool_tau2 as P
        from tau2.data_model.simulation import Results
        self.P, self.Results = P, Results
        self.rule, self.port, self.name = rule, judge_port, judge_name
        self.cache = cache or {}
        self.srcs, self.schema, self.policy, self.ctx, self.rows, self.auth = {}, {}, {}, {}, {}, {}
        self.n_judge = self.n_cache = 0

    def R(self, src):
        if src not in self.srcs:
            self.srcs[src] = self.Results.load(Path(src))
        return self.srcs[src]

    def _env(self, dom, task, prefix):
        P = self.P
        return P.build_text_orchestrator(P.make_cfg(dom, "4B", "4B", 10), P.with_history(task, prefix), seed=0).environment

    def ctx_of(self, d):
        if d["pos_id"] not in self.ctx:
            from tau2.utils.llm_utils import to_litellm_messages
            Rs = self.R(d["src"])
            s_ = next(x for x in Rs.simulations if x.id == d["sim_id"])
            task_ = next(t for t in Rs.tasks if t.id == d["task_id"])
            prefix = [m.model_copy(deep=True) for m in s_.messages[: d["t"]]]
            lm = to_litellm_messages(prefix)
            dom = d["domain"]
            if dom not in self.schema:
                env = self._env(dom, task_, prefix)
                self.schema[dom] = json.dumps([t.openai_schema for t in env.get_tools()], ensure_ascii=False)
                self.policy[dom] = env.get_policy()
            self.ctx[d["pos_id"]] = (lm, ctx_text(lm), self.schema[dom])
            self.auth[d["pos_id"]] = authenticated(prefix)
        return self.ctx[d["pos_id"]]

    def check(self, d, c):
        k = (d["pos_id"], c["cand_id"])
        if k in self.rows:
            return self.rows[k]
        calls = call_list(c["msg"] or {})
        row = {"pos_id": d["pos_id"], "cand_id": c["cand_id"], "kinds": c["kinds"],
               "tools": [n for n, _ in calls], "ok": True, "why": None, "rule": self.rule, "judged_by": "none"}
        if calls:
            lm, ctx, sch = self.ctx_of(d)
            if self.rule == "v1":
                miss = missing_args(calls, ctx, sch)
            else:
                miss = missing_args_v2(calls, ctx + "\n" + self.policy[d["domain"]], sch)
            if miss:
                row.update(ok=False, why="args_missing", missing=miss[:8])
            elif self.rule == "v3" and d["domain"] in AUTH_DOMAINS and not self.auth[d["pos_id"]] \
                    and any(n not in AUTH_EXEMPT for n, _ in calls):
                row.update(ok=False, why="not_authenticated")
            elif any(n in write_names() for n, _ in calls):
                if k in self.cache:
                    raw = self.cache[k]
                    yes = str(raw).strip().upper().startswith("YES")
                    row["judged_by"] = "cache"
                    self.n_cache += 1
                else:
                    if not self.port:
                        raise RuntimeError(f"{k} 是写操作，需要 27B（--judge-port）或 --cache 里的判定")
                    yes, raw = judge(self.port, self.name, JUDGE_CONFIRM.format(conv=transcript(lm), act=act_str(calls)),
                                     sseed("JW", d["pos_id"], c["cand_id"]))
                    row["judged_by"] = "27b"
                    self.n_judge += 1
                row["judge"] = raw
                if not yes:
                    row.update(ok=False, why="not_confirmed")
        self.rows[k] = row
        return row


def load_verdicts(paths):
    """{(pos_id, cand_id): ok}；annotate 的输出或 export 写的 *.admissible.jsonl 都可以。"""
    out = {}
    for p in paths:
        for r in jl(p):
            out[(r["pos_id"], r["cand_id"])] = bool(r["ok"])
    return out


def cmd_annotate(a):
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING", filter=lambda r: r["function"] not in ("get_response_cost", "get_cost"))
    design = [d for f in a.design for d in jl(f)]
    A = Admissibility(a.rule, a.judge_port, a.judge_name, load_cache(a.cache))
    out = Path(a.out)
    done = {}
    if out.exists():
        for r in jl(out):
            done[(r["pos_id"], r["cand_id"])] = r
        A.rows.update(done)
    n_new, n_err = 0, 0
    with open(out, "a", encoding="utf-8") as f:
        for i, d in enumerate(design):
            for c in d["cands"]:
                if c["cand_id"] == "free" or not c.get("msg"):
                    continue
                if not a.all_cands and not c.get("restore_ok"):
                    continue
                k = (d["pos_id"], c["cand_id"])
                if k in done:
                    continue
                try:
                    row = A.check(d, c)
                except Exception as e:
                    n_err += 1
                    log(f"  出错 {k}：{repr(e)[:300]}")
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_new += 1
            if (i + 1) % 20 == 0 or i + 1 == len(design):
                log(f"[{i + 1}/{len(design)}] 新判 {n_new} 个候选；27B 调用 {A.n_judge}、用缓存 {A.n_cache}；出错 {n_err}")
    rows = [A.rows[(d["pos_id"], c["cand_id"])] for d in design for c in d["cands"]
            if (d["pos_id"], c["cand_id"]) in A.rows]
    by = collections.Counter(("+".join(r["kinds"]), r["why"] or "ok") for r in rows)
    log(f"规则 {a.rule}；位置 {len(design)}；候选 {len(rows)}（可执行 {sum(r['ok'] for r in rows)}）")
    log("按来源：" + "；".join(f"{k}/{w} {n}" for (k, w), n in sorted(by.items())))
    print("== 完成" if n_err == 0 else f"== 有 {n_err} 个出错，见上（重跑同一条命令只补没判过的）")
    return 0 if n_err == 0 else 1


def cmd_compare(a):
    A = {(r["pos_id"], r["cand_id"]): r for r in jl(a.a)}
    B = {(r["pos_id"], r["cand_id"]): r for r in jl(a.b)}
    both = sorted(set(A) & set(B))
    print(f"A {len(A)} 行，B {len(B)} 行，共同 {len(both)}；只在 A {len(set(A) - set(B))}，只在 B {len(set(B) - set(A))}")
    diff = [k for k in both if (A[k]["ok"], A[k]["why"]) != (B[k]["ok"], B[k]["why"])]
    jdiff = [k for k in both if A[k].get("judge") is not None and B[k].get("judge") is not None
             and str(A[k]["judge"]).upper().startswith("YES") != str(B[k]["judge"]).upper().startswith("YES")]
    print(f"判定（ok, why）不同 {len(diff)} 个；27B 回答不同 {len(jdiff)} 个")
    for k in diff[: a.show]:
        print("  ", k, A[k]["kinds"], A[k]["tools"], "A:", A[k]["why"], A[k].get("missing"), A[k].get("judge"),
              "| B:", B[k]["why"], B[k].get("missing"), B[k].get("judge"))
    c = collections.Counter((A[k]["why"] or "ok", B[k]["why"] or "ok") for k in both)
    print("（A, B）组合：", dict(c))
    return 0 if not diff and (a.allow_only_a or len(set(A) - set(B)) == 0) else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("annotate")
    n.add_argument("--design", nargs="+", required=True)
    n.add_argument("--rule", choices=RULES, default="v1")
    n.add_argument("--judge-port", type=int, default=0)
    n.add_argument("--judge-name", default="Qwen3.8-27B-FP8")
    n.add_argument("--cache", nargs="*", default=[], help="复用以前 27B 的回答（*.admissible.jsonl 或 annotate 的输出）")
    n.add_argument("--all-cands", action="store_true", help="也判 restore_ok 为假的候选（默认跳过，它们不续跑）")
    n.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--show", type=int, default=30)
    c.add_argument("--allow-only-a", action="store_true", help="A 有、B 没有的行不算不一致")
    a = ap.parse_args()
    return {"annotate": cmd_annotate, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
