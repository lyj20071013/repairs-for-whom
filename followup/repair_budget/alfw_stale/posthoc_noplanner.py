"""事后分析 A（2026-11-10；探索性）：ALFWorld 去掉规划器候选后重算点过期、同口径噪声对照、L、V / Q / D_V。
定义见 `事后分析_A与B_定义_20261110.md`。只读已有文件，不依赖 rt_common（并列打破用 sha256，与原脚本不同，"原集合"一栏用来核对复现）。

用法：python posthoc_noplanner.py [alfw_stale 目录]   → 写 out_stale/posthoc_noplanner.md / .json
"""
import hashlib, json, os, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
TH, EPS, NB = 0.20, float(os.environ.get("EPS", 1e-9)), int(os.environ.get("NB", 2000))
SALT = os.environ.get("TIE_SALT", "")          # 并列打破的盐：空 = 主运行；其他值用来看结论对并列打破的敏感程度
SUF = f"_salt{SALT}" if SALT else ""
HALVES = (tuple(range(0, 5)), tuple(range(5, 10)))
SPL = ((HALVES[0], HALVES[1]), (HALVES[1], HALVES[0]))
ALL = tuple(range(10))


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


D = json.loads((ROOT / "data/design_69.json").read_text(encoding="utf-8"))
BR = D["branches"]
ev = {r["branch_id"]: [bool(x) for x in r["results"]] for r in jl(ROOT / "data/evals_q7_69.jsonl") if r["condition"] == "q7_T07"}
res = defaultdict(lambda: defaultdict(dict))
for f in ("full", "free_b"):
    for r in jl(ROOT / "out_stale" / f"{f}.jsonl"):
        key = r["branch"] if r["branch"] != "free" else r["pos_id"] + "|free"
        res[r["model"]][key][r["i"]] = bool(r["won"])
POS = [{"pos_id": f"{t['traj_id']}|t{p['t']}", "traj": t["traj_id"], "alts": list(p["alts"])} for t in D["trajectories"] for p in t["positions"]]

SETS = {
    "原集合": lambda b: True,
    "无规划器流水线（主）": lambda b: bool(BR[b].get("proposers")) or bool(BR[b].get("random")),
    "去掉全部规划器动作": lambda b: not BR[b].get("planner"),
}


def lst(d):
    return [d[i] for i in ALL] if d and all(i in d for i in ALL) else None


def tie(pos, b):
    return int(hashlib.sha256(f"STALE-TIE{SALT}|{pos}|{BR[b]['action']}".encode()).hexdigest()[:12], 16)


def p_on(arr, ks):
    return mean(arr[i] for i in ks)


def best(pos, rr, cands, ks):
    return max(cands, key=lambda c: (p_on(rr[c], ks), tie(pos, c)))


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
        x = {"traj": p["traj"], "n": len(C)}
        G = lambda rr, fr: mean(p_on(rr[best(pid, rr, C, s)], e) - p_on(fr, e) for s, e in SPL)
        Gh = lambda rr, fr, h: max(p_on(rr[c], h) for c in C) - p_on(fr, h)
        x["G0"], x["G1"] = G(r0, f0), G(r1, f1)
        x["Gh0"], x["Gh1"] = [Gh(r0, f0, h) for h in HALVES], [Gh(r1, f1, h) for h in HALVES]
        a0 = best(pid, r0, C, ALL)
        x["a0_planner"] = bool(BR[a0].get("planner"))
        x["G0_all"] = p_on(r0[a0], ALL) - p_on(f0, ALL)
        b0h = best(pid, r0, C, HALVES[0])                       # 原脚本的 L：π0 用前 5 次选
        x["L"] = mean(p_on(r1[best(pid, r1, C, s)], e) - p_on(r1[b0h], e) for s, e in SPL)
        x["L_a0"] = mean(p_on(r1[best(pid, r1, C, s)], e) - p_on(r1[a0], e) for s, e in SPL)   # 与论文第 2 章一致：a0 用全部 10 次
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


def boot(pp, keys, fn):
    by = defaultdict(list)
    for k in keys:
        by[pp[k]["traj"]].append(k)
    T = sorted(by)
    rng = random.Random(0)
    est, vals = fn(keys), []
    for _ in range(NB):
        ks = [k for t in (rng.choice(T) for _ in T) for k in by[t]]
        v = fn(ks)
        if v == v:
            vals.append(v)
    vals.sort()
    return [est, vals[int(.05 * len(vals))], vals[min(len(vals) - 1, int(.95 * len(vals)))], len(keys), len(T)] if vals else [est, None, None, len(keys), len(T)]


def f(t, sign=False):
    e, lo, hi, n, nt = t
    s = "+" if sign else ""
    return f"{e:{s}.3f} [{lo:{s}.3f}, {hi:{s}.3f}]" if lo is not None else f"{e:{s}.3f}"


out, lines = {}, ["# 事后分析 A：ALFWorld 去掉规划器候选（2026-11-10；事后、探索性）", "",
                  "定义见 `事后分析_A与B_定义_20261110.md`。γ = 0.20；每一半 5 次；区间 = 按轨迹重抽 2,000 次的 90% 区间。", ""]
for sname, keep in SETS.items():
    for m in ("S_s11", "S_s12"):
        pp = records(m, keep)
        ks = sorted(pp)
        w0 = [k for k in ks if pp[k]["G0"] >= TH - EPS]
        both = [k for k in w0 if pp[k]["G1"] >= TH - EPS]
        lam = [k for k in ks if pp[k]["G0_all"] >= TH - EPS]
        o = {"n_pos": len(ks), "n_w0": len(w0), "n_both": len(both), "n_lam": len(lam), "mean_cands": mean(pp[k]["n"] for k in ks)}
        o["point_stale"] = boot(pp, w0, lambda q: mean(pp[k]["G1"] < TH - EPS for k in q) if q else float("nan"))

        def h2(q, which):
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
        hk = [k for k in ks if max(pp[k]["Gh0"]) >= TH - EPS]
        o["n_half_worth"] = len(hk)
        for w in ("N0", "S1", "delta"):
            o[w] = boot(pp, hk, lambda q, w=w: h2(q, w))
        o["L_w0"] = boot(pp, w0, lambda q: mean(pp[k]["L"] for k in q))
        o["L_a0_w0"] = boot(pp, w0, lambda q: mean(pp[k]["L_a0"] for k in q))
        o["L_both"] = boot(pp, both, lambda q: mean(pp[k]["L"] for k in q))
        o["same_best_both"] = mean(pp[k]["same_best"] for k in both)
        o["same_h1_both"] = mean(pp[k]["same_h1"] for k in both)
        o["same_h0_both"] = mean(pp[k]["same_h0"] for k in both)
        o["a0_planner_frac_w0"] = mean(pp[k]["a0_planner"] for k in w0)
        o["free"] = [mean(mean(res["B"][k + "|free"].values()) for k in ks), mean(mean(res[m][k + "|free"].values()) for k in ks)]
        tab = {}
        for meth in ("reuse", "point", "label", "full"):
            rows = [r for k in lam for r in pp[k]["ref"][meth]]
            kept = [g for kp, g, c in rows if kp]
            tab[meth] = {"V": mean(kp * g for kp, g, c in rows), "Q": mean(kept), "kept": len(kept) / len(rows), "cost": mean(c for _, _, c in rows)}
        o["refresh"] = tab
        o["D_V"] = boot(pp, lam, lambda q: mean(pp[k]["v"]["full"] - pp[k]["v"]["point"] for k in q))
        o["D_V_label"] = boot(pp, lam, lambda q: mean(pp[k]["v"]["full"] - pp[k]["v"]["label"] for k in q))
        o["cost_ratio"] = tab["full"]["cost"] / tab["point"]["cost"]
        out[f"{sname}|{m}"] = o
        lines += [f"## {sname} — {m}", "",
                  f"- 位置 {o['n_pos']}（平均候选 {o['mean_cands']:.1f} 个）；π0 值得修（交叉拟合）{o['n_w0']}；两边都值得修 {o['n_both']}；自由选择 π0 {o['free'][0]:.3f} → π1 {o['free'][1]:.3f}",
                  f"- **点过期率** {f(o['point_stale'])}",
                  f"- 同口径噪声对照（一半内部的 G；{o['n_half_worth']} 个位置）：N0 {f(o['N0'])}，S1 {f(o['S1'])}，**Δ {f(o['delta'], True)}**",
                  f"- L（π0 值得修的位置；原脚本定义）{f(o['L_w0'], True)}；a0 用全部 10 次选 {f(o['L_a0_w0'], True)}；L_both {f(o['L_both'], True)}（{o['n_both']} 个）",
                  f"- 最好候选相同的比例（两边都值得修）{o['same_best_both']:.3f}；对照：π1 两半 {o['same_h1_both']:.3f}，π0 两半 {o['same_h0_both']:.3f}",
                  f"- π0 值得修的位置里，最好候选是规划器动作的比例 {o['a0_planner_frac_w0']:.3f}",
                  f"- 标签集（G0 全部 10 次 ≥ γ）{o['n_lam']} 个：",
                  "", "| 刷新方式 | V | Q | 保留比例 | 每位置续跑数 |", "|---|---:|---:|---:|---:|"]
        for meth, nm in (("reuse", "复用"), ("point", "只查位置"), ("label", "只查标签"), ("full", "全部重新验证")):
            t = tab[meth]
            lines.append(f"| {nm} | {t['V']:+.3f} | {t['Q']:+.3f} | {t['kept']:.2f} | {t['cost']:.1f} |")
        lines += ["", f"- **D_V** {f(o['D_V'], True)}；D_V(标签) {f(o['D_V_label'], True)}；成本比 {o['cost_ratio']:.1f}", ""]
if not SALT:
    (ROOT / "out_stale/posthoc_noplanner.md").write_text("\n".join(lines), encoding="utf-8")
(Path(os.environ.get("SALT_DIR", ROOT / "out_stale")) / f"posthoc_noplanner{SUF}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
if not SALT:
    print("\n".join(lines))
