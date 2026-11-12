"""把"训练前的自己"当作替代评估者的选择损失（定义见 定义_训练前的自己当评估者_20261110.md）。CPU。"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("root", nargs="?", default=".")
_ap.add_argument("--succ", default="S_s11,S_s12", help="两个后继的模型名（同一配方的两个种子）")
_ap.add_argument("--extra", default="", help="额外的续跑文件（逗号分隔），如 out_np/full.jsonl")
_ap.add_argument("--out", default="self_eval", help="输出文件前缀")
_A = _ap.parse_args()
ROOT = Path(_A.root)
SUCC = _A.succ.split(",")
assert len(SUCC) == 2
EXEC = ["pi0"] + SUCC
NSPLIT, NB = 50, 2000
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
D = json.loads((ROOT / "data/design_69.json").read_text(encoding="utf-8"))
R = defaultdict(dict)       # R[executor][branch] = np.array(10)
G = {}
for r in jl(ROOT / "data/evals_q7_69.jsonl"):
    if r["condition"] == "q7_T07": R["pi0"][r["branch_id"]] = np.array(r["results"], float)
    elif r["condition"] == "q7_greedy": G[r["branch_id"]] = float(r["results"][0])
tmp = defaultdict(lambda: defaultdict(dict))
for f in ["out_stale/full.jsonl", "out_stale/free_b.jsonl"] + [x for x in _A.extra.split(",") if x]:
    for r in jl(f if Path(f).is_absolute() else ROOT / f):
        key = r["branch"] if r["branch"] != "free" else r["pos_id"] + "|free"
        tmp[r["model"]][key][r["i"]] = float(r["won"])
for m, d in tmp.items():
    for b, x in d.items():
        if len(x) == 10: R[m][b] = np.array([x[i] for i in range(10)])
for b, x in R.get("B", {}).items():
    if b.endswith("|free"): R["pi0"][b] = x
TR = []
for t in D["trajectories"]:
    c = []
    for pi, p in enumerate(t["positions"]):
        for b in p["alts"]: c.append((b, p["orig"], pi, f"{t['traj_id']}|t{p['t']}|free"))
    TR.append({"id": t["traj_id"], "cands": c})
rng = np.random.default_rng(0)
SPL = [rng.permutation(10) for _ in range(NSPLIT)]
SUB = [rng.permutation(10)[:5] for _ in range(NSPLIT)]

def est(ex, cands, idx, base="orig"):
    """每个候选的净增益估计（用 idx 这几次）。"""
    return np.array([R[ex][b][idx].mean() - R[ex][o if base == "orig" else fr][idx].mean() for b, o, pi, fr in cands])

def pick(scores):
    return np.flatnonzero(scores >= scores.max() - 1e-12)

def run(e0, base="orig"):
    out = defaultdict(list)
    for T in TR:
        c = T["cands"]; pos = np.array([x[2] for x in c])
        acc = defaultdict(list)
        for s in range(NSPLIT):
            sel, ev = SPL[s][:5], SPL[s][5:]
            val = est(e0, c, ev, base)
            m = pick(est(e0, c, sel, base)); acc["matched"].append(val[m].mean())
            acc["random"].append(val.mean())
            for name, ex in [(x, x) for x in EXEC]:
                if ex == e0: continue
                k = pick(est(ex, c, sel, base)); acc[name].append(val[k].mean())
                if name == "pi0":
                    # 同一位置的比例（按并列集合的期望）
                    acc["same_pos"].append(np.mean([[pos[i] == pos[j] for j in k] for i in m]))
                    acc["same_cand"].append(np.mean([[i == j for j in k] for i in m]))
                    # 位置内：把 π0 的选择限制在匹配选择的位置（取匹配集合的第一个位置的期望）
                    w = []
                    for i in m:
                        ii = np.flatnonzero(pos == pos[i]); s0 = est("pi0", [c[q] for q in ii], sel, base)
                        w.append(val[ii[pick(s0)]].mean())
                    acc["pi0_within_matched_pos"].append(np.mean(w))
                    # 只换位置：位置用 π0 选的，位置内动作用 π1（选择那一半）选
                    w = []
                    for j in k:
                        jj = np.flatnonzero(pos == pos[j]); s1 = est(e0, [c[q] for q in jj], sel, base)
                        w.append(val[jj[pick(s1)]].mean())
                    acc["pi0_pos_e0_action"].append(np.mean(w))
            g = np.array([G[b] - G[o] for b, o, _, _ in c]); acc["pi0_greedy"].append(val[pick(g)].mean())
            for name in EXEC:
                if name == e0: continue
                k = pick(est(name, c, sel, base)); vx = est(name, c, ev, base)
                dd = [(val[i] - val[j], vx[i] - vx[j]) for i in m for j in k if i != j]
                if dd:
                    acc["dD_" + name].append(np.mean([x for x, _ in dd])); acc["dE_" + name].append(np.mean([y for _, y in dd]))
                    acc["w_" + name].append(len(dd) / (len(m) * len(k)))
        allv = est(e0, c, np.arange(10), base); bi = pick(allv)[0]
        for name in EXEC:
            if name != e0: out["sat_" + name].append(float(R[name][c[bi][1]].mean()))
        for name in EXEC:
            if name != e0:
                out["contested_" + name].append(float(np.sum(acc.pop("w_" + name, [0])) / NSPLIT))
                for q in ("dD_", "dE_"):
                    v = acc.pop(q + name, []); out[q + name].append(float(np.mean(v)) if v else float("nan"))
        for k2, v in acc.items(): out[k2].append(float(np.mean(v)))
    return {k: np.array(v, float) for k, v in out.items()}

def ci(x):
    r = random.Random(0); n = len(x); bs = sorted(float(np.mean([x[r.randrange(n)] for _ in range(n)])) for _ in range(NB))
    return float(x.mean()), bs[int(.05 * NB)], bs[int(.95 * NB)]
f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
res = {}
L = ["# 训练前的自己当评估者：选择损失（self_eval.py；事后、探索性）", ""]
for base in ("orig", "free"):
    L += [f"## 净增益的参照 = {'原动作（第 4 篇口径）' if base == 'orig' else '自由选择（第 6 篇口径，描述）'}", "",
          "| 部署执行者 | 匹配选择的净增益 | 随机的损失 | π0 当评估者的损失 | 另一种子的损失 | π0 贪心的损失 | π0 限制在匹配位置内的损失 | 位置用 π0、动作用自己 的损失 | 同位置 | 同候选 |", "|---|---|---|---|---|---|---|---|---:|---:|"]
    for e0 in EXEC:
        o = run(e0, base); m = o["matched"]
        row = {"matched": ci(m), "random": ci(m - o["random"]), "greedy": ci(m - o["pi0_greedy"])}
        if e0 != "pi0":
            other = SUCC[1] if e0 == SUCC[0] else SUCC[0]
            row.update(pi0=ci(m - o["pi0"]), seed=ci(m - o[other]), within=ci(m - o["pi0_within_matched_pos"]), posonly=ci(m - o["pi0_pos_e0_action"]),
                       same_pos=float(o["same_pos"].mean()), same_cand=float(o["same_cand"].mean()))
            L.append(f"| {e0} | {f(row['matched'])} | {f(row['random'])} | **{f(row['pi0'])}** | {f(row['seed'])} | {f(row['greedy'])} | {f(row['within'])} | {f(row['posonly'])} | {row['same_pos']:.2f} | {row['same_cand']:.2f} |")
        else:
            row.update(s11=ci(m - o[SUCC[0]]), s12=ci(m - o[SUCC[1]]))
            L.append(f"| π0（核对） | {f(row['matched'])} | {f(row['random'])} | — | S11 {f(row['s11'])}；S12 {f(row['s12'])} | {f(row['greedy'])} | — | — | — | — |")
        res[f"{base}|{e0}"] = row
    L.append("")
    L += ["| 部署执行者 | 评估者 | 损失 | 占（匹配 − 随机）的比例 | 评估者在部署者最好修复的位置已饱和（原动作成功率 ≥ 0.9 / < 0.5）的轨迹数 | 未饱和层的损失 | 选择不同的比例 | Δ_部署 | Δ_评估 |", "|---|---|---|---:|---|---|---:|---|---|"]
    for e0 in EXEC:
        o = run(e0, base); m = o["matched"]
        for ev_ in EXEC:
            if ev_ == e0: continue
            sat = o["sat_" + ev_]; lo = sat < 0.5; hi = sat >= 0.9
            los = m - o[ev_]
            ok = ~np.isnan(o["dD_" + ev_])
            row = {"loss": ci(los), "frac": float(los.mean() / (m - o["random"]).mean()), "n_sat": int(hi.sum()), "n_unsat": int(lo.sum()),
                   "loss_unsat": ci(los[lo]) if lo.sum() > 1 else None, "contested": float(o["contested_" + ev_].mean()),
                   "dD": ci(o["dD_" + ev_][ok]), "dE": ci(o["dE_" + ev_][ok])}
            res[f"{base}|{e0}<-{ev_}"] = row
            L.append(f"| {e0} | {ev_} | {f(row['loss'])} | {row['frac']:.2f} | {row['n_sat']} / {row['n_unsat']}（共 60） | {f(row['loss_unsat']) if row['loss_unsat'] else '—'} | {row['contested']:.2f} | {f(row['dD'])} | {f(row['dE'])} |")
    L.append("")
Path(_A.out + ".md").write_text("\n".join(L) + "\n", encoding="utf-8"); Path(_A.out + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n".join(L))
