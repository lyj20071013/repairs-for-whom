"""探针闸门"π1 变好"（2026-11-01 写；登记补充三第 3 节第 4 条）。只用 CPU。
**只读自由选择（cand_id = free）分支的 reward**，不读任何修复候选的 reward（D1 在闸门通过之前不算）。

读数：评价位置上自由接手的成功率 π1 − π0。每个位置先取各自 k 次的平均，再对位置取平均（两个领域合并；各领域另报）；
区间 = 按任务重抽 2,000 次的 90% 区间（同一任务的位置一起抽）。只用两边都有读数的位置。reward 为空（出错）的续跑去掉并报告个数。
判定（事先写定）：
  通过        点估计 ≥ +0.05 且区间下界 > 0
  变坏        点估计 ≤ −0.05
  未证实变好  其余（交用户决定）
用法（tau2_pilot 下）：python gate_free.py --rollouts probe/ro_eval_ret.jsonl probe/ro_eval_air.jsonl --pi0 4B --pi1 4B-S4
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def free_rates(paths, executor):
    """{pos_id: (domain, task_id, [reward…])}；同一 (位置, k, 温度) 多行时取 reward 不为空的最后一行（同 train_tau2.rates_from_rollouts）。"""
    R, n_err = {}, 0
    for p in paths:
        dom = "airline" if "_air" in str(p) else "retail"
        for r in jl(p):
            if r["executor"] != executor or r["cand_id"] != "free" or r.get("score") != "orig_task":
                continue
            key = (r["pos_id"], r["k"], r["temp"])
            if r.get("reward") is not None or key not in R:
                R[key] = (dom, r)
    out = {}
    for (pos, k, temp), (dom, r) in sorted(R.items()):
        if r.get("reward") is None:
            n_err += 1
            continue
        out.setdefault(pos, (dom, str(r["task_id"]), []))[2].append(float(r["reward"]))
    return out, n_err


def boot(pos_list, d, n_boot, seed):
    tasks = sorted({(d[p][0], d[p][1]) for p in pos_list})
    by_task = {t: [p for p in pos_list if (d[p][0], d[p][1]) == t] for t in tasks}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(tasks), len(tasks))
        ps = [p for i in pick for p in by_task[tasks[i]]]
        vals.append(np.mean([d[p][2] for p in ps]))
    return np.percentile(vals, [5, 95])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--pi0", default="4B")
    ap.add_argument("--pi1", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r0, e0 = free_rates(a.rollouts, a.pi0)
    r1, e1 = free_rates(a.rollouts, a.pi1)
    common = sorted(set(r0) & set(r1))
    d = {p: (r0[p][0], r0[p][1], float(np.mean(r1[p][2])) - float(np.mean(r0[p][2]))) for p in common}
    seed = int(hashlib.sha256(b"GATE_FREE").hexdigest()[:8], 16)
    rep = {"pi0": a.pi0, "pi1": a.pi1, "n_pos": len(common), "n_err_pi0": e0, "n_err_pi1": e1,
           "n_pos_only_pi0": len(set(r0) - set(r1)), "n_pos_only_pi1": len(set(r1) - set(r0))}
    print(f"自由接手（只读 free 分支）：两边都有的位置 {len(common)} 个（只有 π0 的 {rep['n_pos_only_pi0']}、只有 π1 的 {rep['n_pos_only_pi1']}）；"
          f"去掉出错的续跑 π0 {e0} 次、π1 {e1} 次")
    for name, ps in [("retail", [p for p in common if d[p][0] == "retail"]),
                     ("airline", [p for p in common if d[p][0] == "airline"]), ("合并", common)]:
        if not ps:
            continue
        m0 = np.mean([np.mean(r0[p][2]) for p in ps])
        m1 = np.mean([np.mean(r1[p][2]) for p in ps])
        lo, hi = boot(ps, d, a.n_boot, seed)
        rep[name] = {"n_pos": len(ps), "n_tasks": len({(d[p][0], d[p][1]) for p in ps}), "pi0": m0, "pi1": m1, "diff": m1 - m0,
                     "ci90": [lo, hi]}
        print(f"  {name:<7} 位置 {len(ps):>3}（任务 {rep[name]['n_tasks']}）：π0 {m0:.3f}  π1 {m1:.3f}  "
              f"π1 − π0 {m1 - m0:+.3f}  90% 区间 [{lo:+.3f}, {hi:+.3f}]")
    g = rep.get("合并")
    if not g:
        print("== 没有可比的位置")
        return 2
    diff, lo = g["diff"], g["ci90"][0]
    verdict = "通过" if (diff >= 0.05 and lo > 0) else ("变坏" if diff <= -0.05 else "未证实变好")
    rep["verdict"] = verdict
    print(f"== 判定（合并）：{verdict}（规则：点估计 ≥ +0.05 且下界 > 0 → 通过；≤ −0.05 → 变坏；其余 → 未证实变好）")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"通过": 0, "未证实变好": 3, "变坏": 1}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
