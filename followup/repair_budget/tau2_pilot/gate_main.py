"""主实验串联脚本（chain_main.sh）用的核对与闸门（2026-11-07 写；第 6 篇）。都只用 CPU，不输出任何 reward 读数。

子命令（tau2_pilot 下运行）：
  complete  π0（或任一执行者）的续跑是否齐全：设计文件里每个可恢复分支 × k 次都要有一行 reward 不为空。
            只打印个数（齐全 / 出错 / 没跑），退出码 0 = 齐全，1 = 不齐。
  taskmix   核对任务重名 bug 的修正已生效：pool_tau2.py 按 (领域, 任务 id) 取任务；ro_pi0 的受影响行已挪出（有 *.taskmix_removed.jsonl）。
  driftfix  漂移样本里受任务重名影响的位置（同 fix_taskmix.py 的认定）挪到 drift.jsonl.taskmix_removed.jsonl（只做一次），
            之后 run_main.sh drift0 会补跑这些位置。drift.jsonl 不存在时什么也不做。
  behavior  行为检查的闸门（登记第 4.4 节）：每折、每个领域，π1 与 π0 在同一批（位置, 候选）上比较
            问题终止率（出错 / 工具错误过多 / 跑满步数）与重复循环率（同一调用连续 ≥ 3 次的对话），
            都 ≤ π0 + 0.05 才通过。退出码 0 = 通过，1 = 不通过，2 = 数据不全。
  terms     （11-10 加；登记补充六，只作描述）各执行者续跑的终止原因分布（USER_STOP / AGENT_STOP / TOO_MANY_ERRORS / MAX_STEPS /
            出错 / 其他）与"重试过的续跑"比例；π0 另按"同一批（位置, 候选）"与每个 π1 对齐。不读 reward 数值（只看是否为空）。
            python gate_main.py terms --rollouts pool/ro_pi0.jsonl pool/ro_pi1_A.jsonl pool/ro_pi1_B.jsonl pool/ro_pi1s2_A.jsonl
"""
import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def expected(design_paths, k):
    exp = set()
    for p in design_paths:
        for d in jl(p):
            for c in d["cands"]:
                if c.get("restore_ok"):
                    exp |= {(d["pos_id"], c["cand_id"], i) for i in range(k)}
    return exp


def cmd_complete(a):
    exp = expected(a.design, a.k)
    got, err, why = set(), set(), collections.defaultdict(set)
    for p in a.rollouts:
        if not Path(p).exists():
            continue
        for r in jl(p):
            if r["executor"] != a.executor or r.get("temp") == 0:
                continue
            key = (r["pos_id"], r["cand_id"], r["k"])
            if key not in exp:
                continue
            if r.get("score") == "orig_task" and r.get("reward") is not None:
                got.add(key)
            else:                                     # 出错的行（reward 为空；出错行没有 score 字段）
                err.add(key)
                why[key].add(str(r.get("termination"))[:60])
    err -= got
    miss = exp - got - err
    print(f"{a.executor}：应有 {len(exp)} 次续跑；齐全 {len(got)}，出错（reward 为空）{len(err)}，没跑 {len(miss)}")
    if len(got) == len(exp):
        return 0
    cw = {k for k in err if any("ContextWindow" in w for w in why[k])}
    for k in sorted(err - got):
        print(f"  缺 {k}：{sorted(why[k])}")
    if a.cw_ok and not miss and cw == err:
        # 11-08（补充四："π1 也按同样处理"）：缺的都是上下文超长的确定性失败 → 按第 7 节按可用次数算
        print(f"  缺的 {len(err)} 次都是上下文超长（ContextWindowExceeded），按登记第 7 节与补充四按可用次数算")
        return 0
    return 1


def cmd_taskmix(a):
    src = (HERE / "pool_tau2.py").read_text(encoding="utf-8")
    ok1 = "def tasks_by_domain" in src and '_CTX["tasks"] = tasks_by_domain' in src
    ok2 = Path(a.rollouts + ".taskmix_removed.jsonl").exists()
    print(f"pool_tau2.py 按 (领域, 任务 id) 取任务：{'是' if ok1 else '否'}；ro_pi0 的受影响行已挪出：{'是' if ok2 else '否'}")
    return 0 if ok1 and ok2 else 1


def cmd_driftfix(a):
    drift = Path(a.drift)
    removed = Path(str(drift) + ".taskmix_removed.jsonl")
    if not drift.exists():
        print("没有 drift.jsonl：不用处理（之后 drift0 会全部重做）")
        return 0
    if removed.exists():
        print("已处理过（有 drift.jsonl.taskmix_removed.jsonl），不重复")
        return 0
    sys.path.insert(0, str(HERE))
    from fix_taskmix import affected_positions
    _, _, aff = affected_positions(jl(a.design))
    rows = drift.read_text(encoding="utf-8").splitlines()
    keep = [l for l in rows if l.strip() and json.loads(l)["pos_id"] not in aff]
    drop = [l for l in rows if l.strip() and json.loads(l)["pos_id"] in aff]
    removed.write_text("".join(l + "\n" for l in drop), encoding="utf-8")
    drift.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    print(f"受影响位置 {len(aff)} 个；漂移样本挪出 {len(drop)} 行，保留 {len(keep)} 行")
    return 0


def cmd_behavior(a):
    sys.path.insert(0, str(HERE))
    from pi1_diag import conv_stats, read_msgs
    dom = {}
    pairs = {}
    for f, p in a.check:
        for d in jl(p):
            dom[d["pos_id"]] = d["domain"]
            for c in d["cands"]:
                if c.get("restore_ok"):
                    pairs[(d["pos_id"], c["cand_id"])] = f
    best = {}                                     # 同一 (执行者, 位置, 候选, k) 有多行时（出错后重跑），取 reward 不为空的那行，否则取最后一行
    msgs = {}
    for p in a.rollouts:
        for r in jl(p):
            if (r["pos_id"], r["cand_id"]) in pairs and r.get("score") == "orig_task" and r.get("temp") != 0:
                key = (r["executor"], r["pos_id"], r["cand_id"], r["k"])
                if r.get("reward") is not None or key not in best or best[key].get("reward") is None:
                    best[key] = r
        mp = Path(str(p) + ".msgs.jsonl.gz")
        if mp.exists():
            for m in read_msgs(str(mp)):
                if (m["pos_id"], m["cand_id"]) in pairs:
                    msgs[(m["executor"], m["pos_id"], m["cand_id"], m["k"])] = m["msgs"]
    rows = collections.defaultdict(list)
    for (ex, _, _, _), r in best.items():
        rows[ex].append(r)
    pi1 = dict(x.split("=", 1) for x in a.pi1)
    ok_all, short = True, False
    for f, m1 in sorted(pi1.items()):
        for dm in ("retail", "airline"):
            res = {}
            for ex in (a.pi0, m1):
                x = [r for r in rows[ex] if pairs[(r["pos_id"], r["cand_id"])] == f and dom[r["pos_id"]] == dm]
                if ex == m1:
                    have = {(r["pos_id"], r["cand_id"]) for r in x}
                else:
                    have1 = {(r["pos_id"], r["cand_id"]) for r in rows[m1] if pairs[(r["pos_id"], r["cand_id"])] == f and dom[r["pos_id"]] == dm}
                    x = [r for r in x if (r["pos_id"], r["cand_id"]) in have1]
                prob = sum(1 for r in x if r.get("reward") is None or "TOO_MANY" in r["termination"] or "MAX_STEPS" in r["termination"])
                cs = [conv_stats(msgs[k]) for r in x if (k := (ex, r["pos_id"], r["cand_id"], r["k"])) in msgs]
                loops = sum(c["max_repeat"] >= 3 for c in cs)
                res[ex] = (len(x), prob / max(len(x), 1), len(cs), loops / max(len(cs), 1))
            n1 = res[m1][0]
            if n1 == 0:
                print(f"{f} 折 {dm}：π1 没有续跑（这一折这个领域没有检查位置？）")
                continue
            if res[m1][2] < 0.9 * n1 or res[a.pi0][2] < 0.9 * res[a.pi0][0]:
                short = True
            p0, l0 = res[a.pi0][1], res[a.pi0][3]
            p1, l1 = res[m1][1], res[m1][3]
            ok = p1 <= p0 + 0.05 + 1e-9 and l1 <= l0 + 0.05 + 1e-9
            ok_all &= ok
            print(f"{f} 折 {dm}：续跑 π0 {res[a.pi0][0]} / π1 {n1}；问题终止率 π0 {p0:.3f} → π1 {p1:.3f}；"
                  f"重复循环率 π0 {l0:.3f} → π1 {l1:.3f}（消息记录 {res[a.pi0][2]} / {res[m1][2]}）→ {'通过' if ok else '不通过'}")
    if short:
        print("[数据不全] 有消息记录的续跑少于 90%")
        return 2
    print("== 行为检查通过" if ok_all else "== 行为检查不通过")
    return 0 if ok_all else 1


def term_cat(r):
    t = str(r.get("termination"))
    if r.get("reward") is None or t.startswith("error"):
        return "出错"
    for k, name in (("USER_STOP", "USER_STOP"), ("AGENT_STOP", "AGENT_STOP"), ("TOO_MANY_ERRORS", "TOO_MANY_ERRORS"), ("MAX_STEPS", "MAX_STEPS")):
        if k in t.upper():
            return name
    return "其他"


TERMS = ["USER_STOP", "AGENT_STOP", "TOO_MANY_ERRORS", "MAX_STEPS", "出错", "其他"]


def cmd_terms(a):
    """补充六（描述，不读 reward 数值）：各执行者的终止原因分布、重试过的续跑比例。
    每个 (执行者, 位置, 候选, k) 取最终一行（有 reward 的那行；都没有则取最后一行）；
    "重试过" = 最终行 attempts > 1，或同一键之前有出错行（--retry-errors 重跑过）。
    两种口径：全部（各执行者自己的全部续跑）；同一批（π0 只取该 π1 跑过的（位置, 候选））。"""
    dom = {}
    for p in a.design:
        for d in jl(p):
            dom[d["pos_id"]] = d["domain"]
    final, nrows = {}, collections.Counter()
    for p in a.rollouts:
        if not Path(p).exists():
            print(f"[提示] 找不到 {p}，跳过")
            continue
        for r in jl(p):
            if r.get("temp") == 0 or r["pos_id"] not in dom:
                continue
            if r.get("reward") is not None and r.get("score") != "orig_task":
                continue
            key = (r["executor"], r["pos_id"], r["cand_id"], r["k"])
            nrows[key] += 1
            if key not in final or final[key].get("reward") is None:      # 已有 reward 的行不被后面的行替换
                final[key] = r
    by = collections.defaultdict(list)
    for key, r in final.items():
        by[key[0]].append((key, r))
    execs = [a.pi0] + [m for m in a.pi1 if m != a.pi0]
    out = {}

    def row(label, items):
        n = len(items)
        cnt = collections.Counter(term_cat(r) for _, r in items)
        retried = sum(1 for k, r in items if (r.get("attempts") or 1) > 1 or nrows[k] > 1)
        att = sum(1 for _, r in items if (r.get("attempts") or 1) > 1)
        out[label] = {"n": n, **{t: cnt[t] for t in TERMS}, "retried": retried, "attempts_gt1": att}
        if n == 0:
            return f"| {label} | 0 |" + " — |" * (len(TERMS) + 2)
        return (f"| {label} | {n} | " + " | ".join(f"{cnt[t] / n:.3f}" for t in TERMS)
                + f" | {retried / n:.3f} | {att / n:.3f} |")

    L = ["# 终止原因与重试（gate_main.py terms；登记补充六；只作描述，不读 reward 数值）", "",
         "| 执行者（口径，领域） | 续跑 | " + " | ".join(TERMS) + " | 重试过 | 其中 attempts > 1 |",
         "|---|---:|" + "---:|" * (len(TERMS) + 2)]
    for ex in execs:
        for dm in ("全部领域", "retail", "airline"):
            items = [(k, r) for k, r in by.get(ex, []) if dm == "全部领域" or dom[k[1]] == dm]
            L.append(row(f"{ex}（全部，{dm}）", items))
    for m in execs[1:]:
        pc = {(k[1], k[2]) for k, _ in by.get(m, [])}
        for dm in ("全部领域", "retail", "airline"):
            items = [(k, r) for k, r in by.get(a.pi0, []) if (k[1], k[2]) in pc and (dm == "全部领域" or dom[k[1]] == dm)]
            L.append(row(f"{a.pi0}（同一批 = {m} 跑过的，{dm}）", items))
    L.append("")
    L.append("说明：终止原因按最终一行；出错 = 重跑后 reward 仍为空（如上下文超长）。重试过 = 该次续跑内部的基础设施重试（attempts > 1）"
             "或被 --retry-errors 整次重跑过。")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(str(a.out).rsplit(".", 1)[0] + ".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("complete")
    c.add_argument("--design", nargs="+", required=True)
    c.add_argument("--rollouts", nargs="+", required=True)
    c.add_argument("--executor", default="4B")
    c.add_argument("--k", type=int, default=6)
    c.add_argument("--cw-ok", action="store_true", help="缺的都是上下文超长的出错时也算通过（补充四）")
    t = sub.add_parser("taskmix")
    t.add_argument("--rollouts", default="pool/ro_pi0.jsonl")
    d = sub.add_parser("driftfix")
    d.add_argument("--design", default="pool/design_run.jsonl")
    d.add_argument("--drift", default="pool/drift.jsonl")
    b = sub.add_parser("behavior")
    b.add_argument("--check", nargs="+", required=True, type=lambda s: tuple(s.split("=", 1)), help="折=检查设计文件，如 A=pool/design_pi1check_A.jsonl")
    b.add_argument("--rollouts", nargs="+", required=True)
    b.add_argument("--pi0", default="4B")
    b.add_argument("--pi1", nargs="+", required=True, help="折=模型名")
    tm = sub.add_parser("terms")
    tm.add_argument("--design", nargs="+", default=["pool/design_run.jsonl"])
    tm.add_argument("--rollouts", nargs="+", required=True)
    tm.add_argument("--pi0", default="4B")
    tm.add_argument("--pi1", nargs="+", default=["4B-A", "4B-B", "4B-A2"], help="模型名")
    tm.add_argument("--out", default="pool/report_terms.md")
    a = ap.parse_args()
    return {"complete": cmd_complete, "taskmix": cmd_taskmix, "driftfix": cmd_driftfix, "behavior": cmd_behavior,
            "terms": cmd_terms}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
