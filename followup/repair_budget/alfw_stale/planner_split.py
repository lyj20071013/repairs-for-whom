"""A 层补充分析（2026-11-06 写；探索性，事后分析）：
1) 点过期率按"π0 的最好修复是不是规划器动作"分组；
2) 补算 V、D_V = V(全部重新验证) − V(只查位置)、D_V(标签) = V(全部重新验证) − V(只查标签)，与 τ² 主实验同定义（k_c = 5）；成本比。
只读 full.jsonl（S_s11、S_s12 全分支）与 free_b.jsonl（B 的自由选择）；定义沿用 stale_analyze.py。"""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
import stale_alfw as SA
import stale_analyze as AN
from rt_common import jl_read

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
SA.setup("real", str(OUT))
D, ev, pos = SA.load_design()
SA.DESIGN[0] = D
res = defaultdict(lambda: defaultdict(dict))
for stage in ("full", "free_b"):
    for r in jl_read(OUT / f"{stage}.jsonl"):
        key = r["branch"] if r["branch"] != "free" else r["pos_id"] + "|free"
        res[r["model"]][key][r["i"]] = bool(r["won"])
Bfree = {p["pos_id"]: v for p in pos if (v := AN.lst(res["B"].get(p["pos_id"] + "|free"))) is not None}
trajs = sorted({p["traj_id"] for p in pos})
TH = AN.TH
mean = AN.mean

def boot(pp, keys, fn, n=2000):
    by = defaultdict(list)
    for k in keys:
        by[pp[k]["traj"]].append(k)
    T = sorted(by)
    rng = random.Random(0)
    est = fn(keys)
    vals = []
    for _ in range(n):
        ks = [k for t in (rng.choice(T) for _ in T) for k in by[t]]
        v = fn(ks)
        if v is not None and not math.isnan(v):
            vals.append(v)
    vals.sort()
    return est, vals[int(.05 * len(vals))], vals[min(len(vals) - 1, int(.95 * len(vals)))], len(keys), len(T)

def f(t):
    e, lo, hi, n, nt = t
    return f"{e:.3f} [{lo:.3f}, {hi:.3f}]（{n} 个位置，{nt} 条轨迹）"

out = {}
lines = ["# A 层补充分析：按规划器分组的点过期率；V 与 D_V（2026-11-06，事后、探索性）", ""]
for m in ("S_s11", "S_s12"):
    pp = AN.per_position(D, ev, pos, res, m, Bfree)
    w0 = [k for k, x in pp.items() if x.get("G0") is not None and x["G0"] >= TH and "G1" in x]
    grp = {"全部": w0, "最好修复是规划器动作": [k for k in w0 if pp[k]["planner_b0"]],
           "最好修复不是规划器动作": [k for k in w0 if not pp[k]["planner_b0"]]}
    rate = lambda ks: mean(pp[k]["G1"] < TH for k in ks) if ks else float("nan")
    lines += [f"## {m}", "", "### 1. 点过期率（G，全分支；条件 G_π0 ≥ 0.20）", "", "| 组 | 点过期率 [90% 区间] |", "|---|---|"]
    o = {}
    for g, ks in grp.items():
        o[g] = boot(pp, ks, rate)
        lines.append(f"| {g} | {f(o[g])} |")
    pl, npl = set(grp["最好修复是规划器动作"]), set(grp["最好修复不是规划器动作"])
    def diff(ks):
        a = [k for k in ks if k in pl]; b = [k for k in ks if k in npl]
        return rate(a) - rate(b) if a and b else float("nan")
    o["差（规划器 − 非规划器）"] = boot(pp, w0, diff)
    lines += [f"| 差（规划器 − 非规划器） | {f(o['差（规划器 − 非规划器）'])} |", ""]
    # 所有位置里规划器为最好修复的比例（参照）
    allk = [k for k, x in pp.items() if "G1" in x]
    lines.append(f"参照：全部 {len(allk)} 个位置里 π0 最好修复是规划器动作的比例 {mean(pp[k]['planner_b0'] for k in allk):.2f}；"
                 f"π0 值得修的位置里 {len(pl) / len(w0):.2f}。")
    lines.append("")
    # 2. V 与 D_V
    rows, lam = AN.refresh(pp, {})
    def V(ks, meth):
        return AN.refresh_summary(rows, ks, 5, meth)["V"] if ks else float("nan")
    def per_pos_v(k, meth):
        return mean(kp * g for kp, g, c in rows[k][5][meth])
    dv = lambda ks: mean(per_pos_v(k, "full") - per_pos_v(k, "point") for k in ks) if ks else float("nan")
    dvl = lambda ks: mean(per_pos_v(k, "full") - per_pos_v(k, "point_label") for k in ks) if ks else float("nan")
    lam_pp = {k: pp[k] for k in lam}
    o["D_V"] = boot(lam_pp, lam, dv)
    o["D_V_label"] = boot(lam_pp, lam, dvl)
    cost_full = mean(rows[k][5]["full"][0][2] for k in lam); cost_pt = mean(rows[k][5]["point"][0][2] for k in lam)
    lines += ["### 2. V 与 D_V（k_c = 5，标签集 = G_π0（全部 10 次）≥ 0.20 的位置；与 τ² 主实验同定义）", "",
              "| 刷新方式 | V | Q | 保留比例 | 每位置续跑数 |", "|---|---:|---:|---:|---:|"]
    names = {"reuse": "复用", "point": "只查位置", "point_label": "只查标签（查位置 + 查标签）", "full": "全部重新验证"}
    for meth in ("reuse", "point", "point_label", "full"):
        s = AN.refresh_summary(rows, lam, 5, meth)
        lines.append(f"| {names[meth]} | {s['V']:+.3f} | {s['Q']:+.3f} | {s['kept_frac']:.2f} | {s['cost']:.1f} |")
    lines += ["", f"- **D_V = V(全部重新验证) − V(只查位置)**：{f(o['D_V'])}",
              f"- **D_V(标签) = V(全部重新验证) − V(只查标签)**：{f(o['D_V_label'])}",
              f"- 成本比（全部重新验证 / 只查位置）：{cost_full / cost_pt:.1f} 倍（{cost_full:.1f} / {cost_pt:.1f} 次续跑）", ""]
    out[m] = {k: list(v) for k, v in o.items()}
    out[m]["cost_ratio"] = cost_full / cost_pt
Path("planner_split.md").write_text("\n".join(lines), encoding="utf-8")
Path("planner_split.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n".join(lines))
