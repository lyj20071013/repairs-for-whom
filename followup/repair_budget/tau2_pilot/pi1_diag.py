"""π1 行为诊断（2026-11-01 写；只用 CPU）。**不读 reward**：只看终止原因、对话长度、工具调用与出错、重复循环，用来判断 π1 是否被训坏。
用法（tau2_pilot 下）：python pi1_diag.py --rollouts probe/ro_eval_ret.jsonl probe/ro_eval_air.jsonl --models 4B 4B-S1 [--same-pos]
  --same-pos：只比较两个模型都有续跑的 (位置, 候选)（π1 还没跑完时用，避免位置构成不同）。
写操作工具名取自 firstbad.WRITE_TOOLS。读 .msgs.jsonl.gz 时遇到写了一半的块就停在那里（之前的记录照用）。
"""
import argparse
import collections
import gzip
import json
import statistics as st
from pathlib import Path

from firstbad import WRITE_TOOLS
from gzrobust import read_records

WRITES = set().union(*WRITE_TOOLS.values())


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def read_msgs(path):
    if not Path(path).exists():
        return []
    recs, ok = read_records(path)                # 坏块可能在中间：按块头分段读，能读的都读出来
    if not ok:
        print(f"（{path} 有写了一半的块，已跳过坏的部分，读出 {len(recs)} 条）")
    return recs


def conv_stats(msgs):
    calls = [c for m in msgs if m["role"] == "assistant" for c in (m.get("tool_calls") or [])]
    err_ids = {m.get("id") for m in msgs if m["role"] == "tool" and m.get("error")}
    sig = [(c["name"], json.dumps(c.get("arguments"), sort_keys=True)) for c in calls]
    run, best = 1, 1 if sig else 0
    for i in range(1, len(sig)):
        run = run + 1 if sig[i] == sig[i - 1] else 1
        best = max(best, run)
    texts = [len(m.get("content") or "") for m in msgs if m["role"] == "assistant"]
    # 11-01 加：不带文字的工具调用（不含续跑的第一条，那是被塞进去的候选），以及紧接工具报错之后的调用
    silent = after_err = after_err_silent = n_tc = 0
    for i, m in enumerate(msgs):
        if i == 0 or m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        emp = not (m.get("content") or "").strip()
        n_tc += 1
        silent += emp
        prev = msgs[i - 1]
        if prev["role"] == "tool" and (prev.get("error") or str(prev.get("content", "")).startswith("Error")):
            after_err += 1
            after_err_silent += emp
    return {"n_tc_msgs": n_tc, "n_silent": silent, "n_after_err": after_err, "n_after_err_silent": after_err_silent,
            "n_calls": len(calls), "n_err": sum(1 for c in calls if c.get("id") in err_ids),
            "n_write": sum(1 for c in calls if c["name"] in WRITES),
            "n_write_err": sum(1 for c in calls if c["name"] in WRITES and c.get("id") in err_ids),
            "max_repeat": best, "text_len": st.median(texts) if texts else 0,
            "err_names": collections.Counter(c["name"] for c in calls if c.get("id") in err_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["4B", "4B-S1"])
    ap.add_argument("--same-pos", action="store_true")
    a = ap.parse_args()
    rows, msgs = [], {}
    for p in a.rollouts:
        dom = "air" if "_air" in p else "ret"
        for r in jl(p):
            if r["executor"] in a.models:
                r["_dom"] = dom
                rows.append(r)
        for m in read_msgs(str(p) + ".msgs.jsonl.gz"):
            msgs[(m["executor"], m["pos_id"], m["cand_id"], m["k"], m.get("temp", 0.7))] = m["msgs"]
    if a.same_pos:
        have = collections.defaultdict(set)
        for r in rows:
            have[r["executor"]].add((r["pos_id"], r["cand_id"]))
        common = set.intersection(*have.values()) if have else set()
        rows = [r for r in rows if (r["pos_id"], r["cand_id"]) in common]
        print(f"只看两边都有的 (位置, 候选)：{len(common)} 组")
    for dom in ("ret", "air"):
        for ex in a.models:
            x = [r for r in rows if r["executor"] == ex and r["_dom"] == dom]
            if not x:
                continue
            ok = [r for r in x if r.get("reward") is not None]
            term = collections.Counter((r["termination"] if r.get("reward") is not None else "error")[:40].replace("TerminationReason.", "")
                                       for r in x)
            cs = [conv_stats(msgs[k]) for r in x
                  if (k := (ex, r["pos_id"], r["cand_id"], r["k"], r["temp"])) in msgs]
            agg = lambda f: round(st.mean(f(c) for c in cs), 2) if cs else None
            loops = sum(c["max_repeat"] >= 3 for c in cs)
            names = collections.Counter()
            for c in cs:
                names.update(c["err_names"])
            print(f"\n== {dom} {ex}：续跑 {len(x)}（出错 {len(x) - len(ok)}）；有消息记录 {len(cs)}")
            print(f"   终止：{dict(term)}")
            if ok:
                print(f"   耗时中位 {st.median([r['secs'] for r in ok]):.0f} 秒；整段消息数中位 {st.median([r['n_msgs'] for r in ok])}")
            print(f"   每段：工具调用 {agg(lambda c: c['n_calls'])}、出错 {agg(lambda c: c['n_err'])}、写操作 {agg(lambda c: c['n_write'])}、"
                  f"写操作出错 {agg(lambda c: c['n_write_err'])}；同一调用连续重复 ≥ 3 次的对话 {loops}/{len(cs)}；文字长度中位 {agg(lambda c: c['text_len'])}")
            print(f"   出错最多的工具：{names.most_common(5)}")
            prob = sum(1 for r in x if r.get("reward") is None or "TOO_MANY" in r["termination"] or "MAX_STEPS" in r["termination"])
            n_tc = sum(c["n_tc_msgs"] for c in cs)
            n_ae = sum(c["n_after_err"] for c in cs)
            print(f"   【闸门】问题终止率 {prob}/{len(x)} = {prob / max(len(x), 1):.3f}；重复循环率 {loops}/{len(cs)} = {loops / max(len(cs), 1):.3f}")
            print(f"   【描述】不带文字的工具调用 {sum(c['n_silent'] for c in cs)}/{n_tc} = {sum(c['n_silent'] for c in cs) / max(n_tc, 1):.2f}"
                  f"（π0 约 0.10）；紧接报错之后的调用中不带文字 {sum(c['n_after_err_silent'] for c in cs)}/{n_ae}")
            # 出问题的续跑集中在哪些位置
            badpos = collections.Counter(r["pos_id"] for r in x if r.get("reward") is None or "TOO_MANY" in r["termination"]
                                         or "MAX_STEPS" in r["termination"])
            if badpos:
                print(f"   出问题（出错 / 工具错误过多 / 跑满步数）的位置：{badpos.most_common(8)}（共 {len(badpos)} 个位置）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
