"""第 6 篇实验 B 的分析（CPU）：ALFWorld 上"不用规划器标签"训练的后继 S-np。
登记：D:\\Paper\\repair_budget\\登记_ALFWorld无规划器后继_20261110.md（第 5 节的读数与判定在这里逐条实现）。

读数的定义与事后分析 A（posthoc_noplanner.py）逐行相同：γ = 0.20（含等号，EPS = 1e-9）；每一半 5 次（0–4 / 5–9）；
并列打破 sha256("STALE-TIE|位置|动作")；区间 = 按轨迹重抽 2,000 次（种子 0）的 90% 百分位区间；单侧 90% 界 = 第 10 / 90 百分位。

输入（只读）：data/design_69.json、data/evals_q7_69.jsonl（π0 的分支续跑）、out_stale/free_b.jsonl（π0 的自由选择）、
  out_stale/full.jsonl（S_s11、S_s12，对照）、out_stale/drift.jsonl（B、B#2、S_s11、S_s12 的第一步）、
  --np-dir（默认 out_np/）里的 full.jsonl 与 drift.jsonl（S-np_s11、S-np_s12）。
用法：python np_analyze.py [--np-dir out_np] [--models S-np_s11,S-np_s12] [--ref S_s11,S_s12]
输出：<np-dir>/report_np.md、report_np.json
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TH, EPS, NB = 0.20, 1e-9, 2000
HALVES = (tuple(range(0, 5)), tuple(range(5, 10)))
SPL = ((HALVES[0], HALVES[1]), (HALVES[1], HALVES[0]))
ALL = tuple(range(10))
MARGIN = 0.05


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


ap = argparse.ArgumentParser()
ap.add_argument("--np-dir", default=str(ROOT / "out_np"))
ap.add_argument("--models", default="S-np_s11,S-np_s12")
ap.add_argument("--ref", default="S_s11,S_s12", help="同种子对照（顺序与 --models 一一对应）")
A = ap.parse_args()
NPD = Path(A.np_dir)
MODELS = A.models.split(",")
REFS = A.ref.split(",")

D = json.loads((ROOT / "data/design_69.json").read_text(encoding="utf-8"))
BR = D["branches"]
ev = {r["branch_id"]: [bool(x) for x in r["results"]] for r in jl(ROOT / "data/evals_q7_69.jsonl") if r["condition"] == "q7_T07"}
res = defaultdict(lambda: defaultdict(dict))
srcs = [ROOT / "out_stale/full.jsonl", ROOT / "out_stale/free_b.jsonl", NPD / "full.jsonl"]
for path in srcs:
    if not path.exists():
        continue
    for r in jl(path):
        key = r["branch"] if r["branch"] != "free" else r["pos_id"] + "|free"
        res[r["model"]][key][r["i"]] = bool(r["won"])
POS = [{"pos_id": f"{t['traj_id']}|t{p['t']}", "traj": t["traj_id"], "orig": p["orig"], "alts": list(p["alts"])}
       for t in D["trajectories"] for p in t["positions"]]
SETS = {
    "无规划器流水线（主）": lambda b: bool(BR[b].get("proposers")) or bool(BR[b].get("random")),
    "原集合": lambda b: True,
    "去掉全部规划器动作": lambda b: not BR[b].get("planner"),
}
MAIN = "无规划器流水线（主）"


def lst(d):
    return [d[i] for i in ALL] if d and all(i in d for i in ALL) else None


def tie(pos, b):
    return int(hashlib.sha256(f"STALE-TIE|{pos}|{BR[b]['action']}".encode()).hexdigest()[:12], 16)


def p_on(arr, ks):
    return mean(arr[i] for i in ks)


def best(pos, rr, cands, ks):
    return max(cands, key=lambda c: (p_on(rr[c], ks), tie(pos, c)))


def completeness(model):
    R = res[model]
    need = [b for p in POS for b in [p["orig"]] + p["alts"]]
    miss_b = sum(1 for b in need if lst(R.get(b)) is None)
    miss_f = sum(1 for p in POS if lst(R.get(p["pos_id"] + "|free")) is None)
    return {"branches_missing": miss_b, "free_missing": miss_f, "branches_needed": len(need)}


def records(model, keep):
    R = res[model]
    out = {}
    for p in POS:
        pid = p["pos_id"]
        C = [b for b in p["alts"] if keep(b)]
        f0, f1 = lst(res["B"].get(pid + "|free")), lst(R.get(pid + "|free"))
        if not C or f0 is None or f1 is None or any(lst(R.get(b)) is None for b in C):
            continue
        r0 = {b: ev[b] for b in C}
        r1 = {b: lst(R[b]) for b in C}
        x = {"traj": p["traj"], "n": len(C), "f0": p_on(f0, ALL), "f1": p_on(f1, ALL)}
        G = lambda rr, fr: mean(p_on(rr[best(pid, rr, C, s)], e) - p_on(fr, e) for s, e in SPL)  # noqa: E731
        Gh = lambda rr, fr, h: max(p_on(rr[c], h) for c in C) - p_on(fr, h)  # noqa: E731
        x["G0"], x["G1"] = G(r0, f0), G(r1, f1)
        x["Gh0"], x["Gh1"] = [Gh(r0, f0, h) for h in HALVES], [Gh(r1, f1, h) for h in HALVES]
        a0 = best(pid, r0, C, ALL)
        x["a0_planner"] = bool(BR[a0].get("planner"))
        x["G0_all"] = p_on(r0[a0], ALL) - p_on(f0, ALL)
        b0h = best(pid, r0, C, HALVES[0])
        x["L"] = mean(p_on(r1[best(pid, r1, C, s)], e) - p_on(r1[b0h], e) for s, e in SPL)
        x["L_a0"] = mean(p_on(r1[best(pid, r1, C, s)], e) - p_on(r1[a0], e) for s, e in SPL)
        x["same_best"] = best(pid, r1, C, ALL) == a0
        x["same_h1"] = best(pid, r1, C, HALVES[0]) == best(pid, r1, C, HALVES[1])
        x["same_h0"] = best(pid, r0, C, HALVES[0]) == best(pid, r0, C, HALVES[1])
        p0 = p_on(r0[a0], ALL)
        ref = defaultdict(list)
        for s, e in SPL:
            fS, fE = p_on(f1, s), p_on(f1, e)
            g = p_on(r1[a0], e) - fE
            ref["reuse"].append((1, g, 0))
            ref["point"].append((p0 - fS >= TH - EPS, g, 5))
            ref["label"].append((p_on(r1[a0], s) - fS >= TH - EPS, g, 10))
            bb = best(pid, r1, C, s)
            ref["full"].append((p_on(r1[bb], s) - fS >= TH - EPS, p_on(r1[bb], e) - fE, (len(C) + 1) * 5))
        x["ref"] = dict(ref)
        x["v"] = {m: mean(k * g for k, g, c in v) for m, v in ref.items()}
        out[pid] = x
    return out


def boot(groups, keys, fn, seed=0):
    """groups：{位置: 轨迹}；fn(位置列表) -> 数。按轨迹整体重抽。返回 [估计, 5%, 95%, 10%, 90%, n, 轨迹数]。"""
    by = defaultdict(list)
    for k in keys:
        by[groups[k]].append(k)
    T = sorted(by)
    rng = random.Random(seed)
    est, vals = fn(keys), []
    for _ in range(NB):
        ks = [k for t in (rng.choice(T) for _ in T) for k in by[t]]
        v = fn(ks)
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return [est, None, None, None, None, len(keys), len(T)]
    q = lambda a: vals[min(len(vals) - 1, int(a * len(vals)))]  # noqa: E731
    return [est, q(.05), q(.95), q(.10), q(.90), len(keys), len(T)]


def f(t, sign=True):
    e, lo, hi = t[0], t[1], t[2]
    s = "+" if sign else ""
    return f"{e:{s}.3f} [{lo:{s}.3f}, {hi:{s}.3f}]" if lo is not None else f"{e:{s}.3f}"


def h2(pp, q, which):
    n0 = n1 = d = 0
    for k in q:
        x = pp[k]
        for i, j in ((0, 1), (1, 0)):
            if x["Gh0"][i] >= TH - EPS:
                d += 1
                n0 += x["Gh0"][j] < TH - EPS
                n1 += x["Gh1"][j] < TH - EPS
    if not d:
        return float("nan")
    return {"N0": n0 / d, "S1": n1 / d, "delta": (n1 - n0) / d}[which]


def readouts(model, keep):
    pp = records(model, keep)
    ks = sorted(pp)
    tr = {k: pp[k]["traj"] for k in ks}
    w0 = [k for k in ks if pp[k]["G0"] >= TH - EPS]
    both = [k for k in w0 if pp[k]["G1"] >= TH - EPS]
    lam = [k for k in ks if pp[k]["G0_all"] >= TH - EPS]
    hk = [k for k in ks if max(pp[k]["Gh0"]) >= TH - EPS]
    o = {"n_pos": len(ks), "n_w0": len(w0), "n_both": len(both), "n_lam": len(lam), "n_half_worth": len(hk),
         "mean_cands": mean(pp[k]["n"] for k in ks)}
    o["R1_free"] = boot(tr, ks, lambda q: mean(pp[k]["f1"] - pp[k]["f0"] for k in q))
    o["free"] = [mean(pp[k]["f0"] for k in ks), mean(pp[k]["f1"] for k in ks)]
    o["point_stale"] = boot(tr, w0, lambda q: mean(pp[k]["G1"] < TH - EPS for k in q) if q else float("nan"))
    for w in ("N0", "S1", "delta"):
        o[w] = boot(tr, hk, lambda q, w=w: h2(pp, q, w))
    o["L_w0"] = boot(tr, w0, lambda q: mean(pp[k]["L"] for k in q))
    o["L_a0_w0"] = boot(tr, w0, lambda q: mean(pp[k]["L_a0"] for k in q))
    o["L_both"] = boot(tr, both, lambda q: mean(pp[k]["L"] for k in q)) if both else [float("nan")] + [None] * 4 + [0, 0]
    o["same_best_both"] = mean(pp[k]["same_best"] for k in both)
    o["same_h1_both"] = mean(pp[k]["same_h1"] for k in both)
    o["same_h0_both"] = mean(pp[k]["same_h0"] for k in both)
    o["a0_planner_frac_w0"] = mean(pp[k]["a0_planner"] for k in w0)
    tab = {}
    for meth in ("reuse", "point", "label", "full"):
        rows = [r for k in lam for r in pp[k]["ref"][meth]]
        kept = [g for kp, g, c in rows if kp]
        tab[meth] = {"V": mean(kp * g for kp, g, c in rows), "Q": mean(kept), "kept": len(kept) / len(rows) if rows else float("nan"),
                     "cost": mean(c for _, _, c in rows)}
    o["refresh"] = tab
    o["D_V"] = boot(tr, lam, lambda q: mean(pp[k]["v"]["full"] - pp[k]["v"]["point"] for k in q))
    o["D_V_label"] = boot(tr, lam, lambda q: mean(pp[k]["v"]["full"] - pp[k]["v"]["label"] for k in q))
    o["cost_ratio"] = tab["full"]["cost"] / tab["point"]["cost"]
    return o, pp, w0, lam


def paired(m, r, keep):
    """S-np − 同种子 S（同一批位置、同一 π0 标签集合）；按轨迹配对重抽。"""
    _, pa, w0, lam = readouts(m, keep)
    pb = records(r, keep)
    ks = sorted(set(pa) & set(pb))
    tr = {k: pa[k]["traj"] for k in ks}
    w = [k for k in w0 if k in pb]
    la = [k for k in lam if k in pb]
    return {
        "free_gain": boot(tr, ks, lambda q: mean((pa[k]["f1"] - pb[k]["f1"]) for k in q)),
        "point_stale": boot(tr, w, lambda q: mean((pa[k]["G1"] < TH - EPS) - (pb[k]["G1"] < TH - EPS) for k in q)),
        "L_w0": boot(tr, w, lambda q: mean(pa[k]["L"] - pb[k]["L"] for k in q)),
        "D_V": boot(tr, la, lambda q: mean((pa[k]["v"]["full"] - pa[k]["v"]["point"]) - (pb[k]["v"]["full"] - pb[k]["v"]["point"]) for k in q)),
    }


def drift_planner():
    """第一步动作恰是该位置规划器动作的比例（每位置 20 个样本，位置平均）。"""
    plan = {}
    for p in POS:
        pl = [BR[b]["action"] for b in p["alts"] if BR[b].get("planner")]
        if pl:
            plan[p["pos_id"]] = pl[0]
    rows = jl(ROOT / "out_stale/drift.jsonl") + (jl(NPD / "drift.jsonl") if (NPD / "drift.jsonl").exists() else [])
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "@" in r["model"] or r["pos_id"] not in plan:
            continue
        by[r["model"]][r["pos_id"]].append(r["cmd"] == plan[r["pos_id"]])
    out = {}
    for m in ["B", "B#2", "A_s11", "S_s11", "S_s12", "S-plan_s11", "U_s11"] + MODELS:
        if m in by:
            out[m] = {"frac": mean(mean(v) for v in by[m].values()), "n_pos": len(by[m])}
    return out


def tv_vs_B(model):
    rows = jl(ROOT / "out_stale/drift.jsonl") + (jl(NPD / "drift.jsonl") if (NPD / "drift.jsonl").exists() else [])
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["model"]][r["pos_id"]].append(r["cmd"])
    if model not in by:
        return None

    def tv(a, b):
        ca, cb = defaultdict(float), defaultdict(float)
        for x in a:
            ca[x] += 1 / len(a)
        for x in b:
            cb[x] += 1 / len(b)
        return 0.5 * sum(abs(ca[k] - cb[k]) for k in set(ca) | set(cb))
    ps = sorted(set(by[model]) & set(by["B"]))
    return mean(tv(by[model][p], by["B"][p]) for p in ps)


def verdict_H1(t):
    """登记 5.3：单侧。q10 = 单侧 90% 下界，q90 = 单侧 90% 上界。"""
    q10, q90 = t[3], t[4]
    if q10 is None:
        return "无法判定"
    if q10 > 0:
        return "小的动作过期（下界 > 0，上界 < +0.05）" if q90 < MARGIN else "动作过期（下界 > 0）"
    if q90 < MARGIN:
        return "不过期"
    return "不确定"


def verdict_H1_twosided(t):
    """A 层登记的写法（对照）：双侧 90% 区间完全落在 ±0.05 内 → 不过期；下界 > 0 → 过期；其余不确定。"""
    lo, hi = t[1], t[2]
    if lo is None:
        return "无法判定"
    if lo > 0:
        return "过期"
    if lo >= -MARGIN and hi <= MARGIN:
        return "不过期"
    return "不确定"


def main():
    out = {"complete": {m: completeness(m) for m in MODELS}, "models": {}, "paired": {}, "sets": {}}
    L = ["# 实验 B：ALFWorld 上不用规划器标签训练的后继（np_analyze.py；登记 `登记_ALFWorld无规划器后继_20261110.md`）", "",
         "γ = 0.20（含等号）；每一半 5 次；区间 = 按轨迹重抽 2,000 次的 90% 区间；单侧 90% 界 = 第 10 / 90 百分位。", "",
         "## 0. 齐全核对", ""]
    for m, c in out["complete"].items():
        L.append(f"- {m}：缺分支 {c['branches_missing']}/{c['branches_needed']}，缺自由选择 {c['free_missing']}/180")
    L += ["", f"## 1. 主设定（{MAIN}）：登记的判定", ""]
    changed_any = {}
    for m in MODELS:
        o = readouts(m, SETS[MAIN])[0]
        out["models"][m] = o
        r1 = o["R1_free"][3] is not None and o["R1_free"][3] > 0
        h2c = o["delta"][3] is not None and o["delta"][3] > 0
        changed = r1 or h2c
        changed_any[m] = changed
        h1 = verdict_H1(o["L_w0"])
        o["H1_twosided"] = verdict_H1_twosided(o["L_w0"])
        dv_up = o["D_V"][4]
        h3 = ("重新验证动作不划算" if dv_up is not None and dv_up < MARGIN else "不确定") if changed else "π1 变化不足，只作描述"
        o["verdict"] = {"R1": r1, "H2": h2c, "changed": changed, "H1": h1 if changed else f"{h1}（π1 变化不足，只作描述）", "H3": h3}
        L += [f"### {m}", "",
              f"- 位置 {o['n_pos']}（平均候选 {o['mean_cands']:.2f}）；π0 值得修 {o['n_w0']}；两边都值得修 {o['n_both']}；标签集 {o['n_lam']}",
              f"- **R1** 自由选择 π0 {o['free'][0]:.3f} → π1 {o['free'][1]:.3f}，差 {f(o['R1_free'])}，单侧下界 {o['R1_free'][3]:+.3f} → {'确认' if r1 else '未确认'}",
              f"- **H2** 点过期率 {f(o['point_stale'], False)}；N0 {o['N0'][0]:.3f}，S1 {o['S1'][0]:.3f}，Δ {f(o['delta'])}，单侧下界 {o['delta'][3]:+.3f} → {'确认' if h2c else '未确认'}",
              f"- 保护条款：π1 {'确实变了' if changed else '变化不足（R1、H2 都未确认）'}",
              f"- **H1（主）** L {f(o['L_w0'])}，单侧上界 {o['L_w0'][4]:+.3f}、下界 {o['L_w0'][3]:+.3f} → **{o['verdict']['H1']}**（A 层的双侧写法：{o['H1_twosided']}）；L（a0 用全部 10 次）{f(o['L_a0_w0'])}",
              f"- **H3** D_V {f(o['D_V'])}，单侧上界 {dv_up:+.3f} → **{h3}**；D_V(标签) {f(o['D_V_label'])}；成本比 {o['cost_ratio']:.1f}",
              f"- L_both {f(o['L_both'])}（{o['n_both']} 个；选择效应，只描述）；最好候选相同（两边都值得修）{o['same_best_both']:.3f}，对照 π1 两半 {o['same_h1_both']:.3f}、π0 两半 {o['same_h0_both']:.3f}",
              f"- π0 值得修的位置里最好候选是规划器动作（模型也提议过）的比例 {o['a0_planner_frac_w0']:.3f}",
              "", "| 刷新方式 | V | Q | 保留比例 | 每位置续跑数 |", "|---|---:|---:|---:|---:|"]
        for meth, nm in (("reuse", "复用"), ("point", "只查位置"), ("label", "只查标签"), ("full", "全部重新验证")):
            t = o["refresh"][meth]
            L.append(f"| {nm} | {t['V']:+.3f} | {t['Q']:+.3f} | {t['kept']:.2f} | {t['cost']:.1f} |")
        L.append("")
    seeds = [out["models"][m]["verdict"]["H1"] for m in MODELS]
    if all(changed_any.values()):
        if all(s == "不过期" for s in seeds):
            overall = "不过期（两个种子的单侧上界都 < +0.05）"
        elif any(s.startswith("动作过期") for s in seeds):
            overall = "动作过期（至少一个种子的下界 > 0 且上界 ≥ +0.05）"
        elif any(s.startswith("小的动作过期") for s in seeds):
            overall = "小的动作过期（至少一个种子的下界 > 0，都 < +0.05）"
        else:
            overall = "不确定"
    else:
        overall = "至少一个种子变化不足；该种子的 H1 只作描述"
    out["H1_overall"] = overall
    L += [f"**H1 合并判定**：{overall}", "", "## 2. 与同种子 S 的对照（描述；同一批位置、同一 π0 标签；S-np − S）", ""]
    for m, r in zip(MODELS, REFS):
        if r not in res:
            continue
        pr = paired(m, r, SETS[MAIN])
        out["paired"][f"{m}-{r}"] = pr
        L.append(f"- {m} − {r}：自由选择 {f(pr['free_gain'])}；点过期率 {f(pr['point_stale'])}；L {f(pr['L_w0'])}；D_V {f(pr['D_V'])}")
    L += ["", "## 3. 其他候选集合（描述）", "", "| 集合 | 模型 | π0 值得修 | 点过期 | Δ | L | D_V | 成本比 |", "|---|---|---:|---|---|---|---|---:|"]
    for sname, keep in SETS.items():
        for m in MODELS + [r for r in REFS if r in res]:
            o = readouts(m, keep)[0]
            out["sets"][f"{sname}|{m}"] = {k: o[k] for k in ("n_w0", "point_stale", "delta", "L_w0", "D_V", "cost_ratio", "N0", "S1")}
            L.append(f"| {sname} | {m} | {o['n_w0']} | {f(o['point_stale'], False)} | {f(o['delta'])} | {f(o['L_w0'])} | {f(o['D_V'])} | {o['cost_ratio']:.1f} |")
    dp = drift_planner()
    out["first_action_is_planner"] = dp
    out["tv_vs_B"] = {m: tv_vs_B(m) for m in MODELS + REFS + ["B#2"]}
    L += ["", "## 4. 第一步动作（描述）", "", "| 模型 | 第一步恰是规划器动作的比例 | 与 B 的 TV |", "|---|---:|---:|"]
    for m, v in dp.items():
        tv = tv_vs_B(m)
        L.append(f"| {m} | {v['frac']:.3f} | {tv:.3f} |" if tv is not None else f"| {m} | {v['frac']:.3f} | — |")
    NPD.mkdir(parents=True, exist_ok=True)
    (NPD / "report_np.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (NPD / "report_np.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


main()
