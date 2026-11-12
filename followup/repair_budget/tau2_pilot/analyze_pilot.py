"""τ²-bench 先导分析（Claude 在工作区 / 本地运行，只用 CPU；2026-10-29 写）。
读 tau2-bench/data/simulations/<标签>/results.json 与 logs/branch_*.jsonl，输出 Q2–Q5 的读数，
并把失败对话导出成可读的 markdown，供人工标注"用户导致的失败"（Q3）。
用法：python analyze_pilot.py <tau2_pilot 目录> [--dump 20]
"""
import argparse
import collections
import json
import random
import statistics as st
from datetime import datetime
from pathlib import Path

RUNS = ["p_4b_air_local", "p_4b_tel_local", "p_9b_air_local", "p_9b_tel_local", "p_4b_air_flash", "p_4b_tel_flash",
        "p2_4b_tel_q27", "p2_4b_air_q27"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def user_loop(msgs, n=3):
    """用户连续 n 次说同一句话（去空白后完全相同）。"""
    last, run = None, 0
    for m in msgs:
        if m.get("role") != "user" or m.get("tool_calls"):
            continue
        c = (m.get("content") or "").strip()
        run = run + 1 if c and c == last else 1
        last = c
        if run >= n:
            return True
    return False


def tok(s, who):
    u = s.get(f"{who}_usage")
    if not u:
        return None                                  # 本地 vLLM 时 tau2 可能不记录用量；token 改从 vLLM 日志估
    return (u.get("prompt_tokens") or 0), (u.get("completion_tokens") or 0)


def ts(x):
    return datetime.fromisoformat(x) if x else None


def summarize(tag, d):
    sims = d["simulations"]
    infra = [s for s in sims if "infra" in str(s.get("termination_reason")).lower()]
    ev = [s for s in sims if s not in infra and s.get("reward_info")]
    rew = [s["reward_info"]["reward"] for s in ev]
    succ = sum(r == 1.0 for r in rew)
    term = collections.Counter(str(s.get("termination_reason")) for s in sims)
    loops = sum(user_loop(s.get("messages") or []) for s in ev)
    nm = [len(s.get("messages") or []) for s in ev]
    dur = [s.get("duration") or 0 for s in ev]
    starts = [ts(s.get("start_time")) for s in sims if s.get("start_time")]
    ends = [ts(s.get("end_time")) for s in sims if s.get("end_time")]
    wall = (max(ends) - min(starts)).total_seconds() if starts and ends else float("nan")
    at = [x for x in (tok(s, "agent") for s in ev) if x is not None]
    # 每个任务的成功率（多次试次）
    by = collections.defaultdict(list)
    for s in ev:
        by[s["task_id"]].append(s["reward_info"]["reward"] == 1.0)
    mixed = sum(0 < sum(v) < len(v) for v in by.values() if len(v) > 1)
    return dict(tag=tag, n=len(sims), infra=len(infra), evaluated=len(ev), success=succ,
                rate=succ / len(ev) if ev else float("nan"), terms=dict(term), user_loops=loops,
                msgs_med=st.median(nm) if nm else None, msgs_max=max(nm) if nm else None,
                dur_med=st.median(dur) if dur else None, wall=wall, per_conv_wall=wall / max(len(sims), 1),
                agent_prompt_tok=st.mean(a for a, _ in at) if at else None,
                agent_out_tok=st.mean(b for _, b in at) if at else None,
                tasks=len(by), tasks_mixed=mixed, by_task=by)


def dump(sims, path, k, seed=0):
    fails = [s for s in sims if s.get("reward_info") and s["reward_info"]["reward"] == 0.0
             and "infra" not in str(s.get("termination_reason")).lower()]
    random.Random(seed).shuffle(fails)
    out = []
    for s in fails[:k]:
        out.append(f"\n## 任务 {s['task_id']}  试次 {s.get('trial')}  终止 {s.get('termination_reason')}  "
                   f"消息 {len(s.get('messages') or [])}\n")
        ri = s["reward_info"]
        out.append(f"判分：{json.dumps(ri.get('reward_breakdown'), ensure_ascii=False)}；"
                   f"DB {json.dumps((ri.get('db_check') or {}).get('db_match'))}\n")
        for i, m in enumerate(s.get("messages") or []):
            c = (m.get("content") or "").replace("\n", " ")
            tc = m.get("tool_calls")
            t = (" ⟶ " + "; ".join(f"{x.get('name')}({json.dumps(x.get('arguments'), ensure_ascii=False)})"
                                   for x in tc)) if tc else ""
            out.append(f"{i:3d} **{m.get('role')}** {c[:400]}{t[:400]}")
        out.append("\n标注：用户导致失败？ [ ]   部署者导致？ [ ]   备注：\n")
    Path(path).write_text("\n".join(out), encoding="utf-8")
    return len(fails[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--dump", type=int, default=20)
    a = ap.parse_args()
    root = Path(a.root)
    S = {}
    lines = ["# τ²-bench 先导读数\n", "| 运行 | 对话 | 基础设施错误 | 评分 | 成功 | 成功率 | 用户重复≥3 | 消息中位 / 最大 | 单次对话时长中位（秒） | 墙钟 / 对话（秒） | 部署者 token（入 / 出，平均） | 任务内结果不一 |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|"]
    for tag in RUNS:
        p = root / "tau2-bench" / "data" / "simulations" / tag / "results.json"
        if not p.exists():
            lines.append(f"| {tag} | （没有结果文件） |")
            continue
        d = load(p)
        s = summarize(tag, d)
        S[tag] = (s, d)
        tokcol = (f"{s['agent_prompt_tok']:.0f} / {s['agent_out_tok']:.0f}"
                  if s['agent_prompt_tok'] is not None else "未记录")
        lines.append(f"| {tag} | {s['n']} | {s['infra']} | {s['evaluated']} | {s['success']} | {s['rate']:.2f} | "
                     f"{s['user_loops']} | {s['msgs_med']} / {s['msgs_max']} | {s['dur_med']:.0f} | "
                     f"{s['per_conv_wall']:.1f} | {tokcol} | {s['tasks_mixed']}/{s['tasks']} |")
    lines.append("\n终止原因：")
    for tag, (s, _) in S.items():
        lines.append(f"- {tag}：{s['terms']}")
    # Q3：同一批任务上本地用户与 flash 用户的成功率（4B，只用本地的第 1 次试次对齐）
    pairs = [(f"p_4b_{d}_local", f"p_4b_{d}_flash") for d in ("air", "tel")] + \
            [(f"p2_4b_{d}_q27", f"p_4b_{d}_flash") for d in ("air", "tel")] + [("p2_4b_tel_q27", "p_4b_tel_local")]
    for la, fa in pairs:
        dom = la
        L, F = S.get(la), S.get(fa)
        if L and F:
            l0 = {s["task_id"]: s["reward_info"]["reward"] == 1.0 for s in L[1]["simulations"]
                  if s.get("reward_info") and s.get("trial") == 0}
            f0 = {s["task_id"]: s["reward_info"]["reward"] == 1.0 for s in F[1]["simulations"] if s.get("reward_info")}
            common = sorted(set(l0) & set(f0))
            if common:
                lines.append(f"\nQ3 {la} 对 {fa}（试次 0，共同任务 {len(common)}）：前者成功 {sum(l0[t] for t in common)}，"
                             f"后者成功 {sum(f0[t] for t in common)}；两者不一致的任务 {sum(l0[t] != f0[t] for t in common)}")
    # Q5：分叉续跑
    for f in sorted((root / "logs").glob("branch_*.jsonl")):
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        ok = [r for r in rows if r.get("reward") is not None]
        grp = collections.defaultdict(list)
        for r in ok:
            grp[(r["task_id"], r["trial"], r["pos"])].append(r["reward"] == 1.0)
        means = [sum(v) / len(v) for v in grp.values()]
        lines.append(f"\nQ5 {f.name}：续跑 {len(rows)}（出错 {len(rows) - len(ok)}）；位置 {len(grp)}；"
                     f"续跑成功率 {sum(r['reward'] == 1.0 for r in ok) / max(len(ok), 1):.2f}；"
                     f"0 < 位置成功率 < 1 的位置 {sum(0 < m < 1 for m in means)}/{len(means)}；"
                     f"单次续跑耗时中位 {st.median([r['secs'] for r in ok]) if ok else float('nan'):.0f} 秒")
    out = root / "logs" / "pilot_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for tag in ("p_4b_air_local", "p_4b_tel_local", "p_4b_air_flash", "p_4b_tel_flash", "p2_4b_tel_q27", "p2_4b_air_q27"):
        if tag in S:
            k = dump(S[tag][1]["simulations"], root / "logs" / f"fails_{tag}.md", a.dump)
            print(f"导出 {k} 条失败对话 → logs/fails_{tag}.md")


if __name__ == "__main__":
    main()
