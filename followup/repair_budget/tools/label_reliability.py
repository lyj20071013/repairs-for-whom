"""事后分析 B（2026-11-10；探索性）：同一策略（π0）下"值得修"标签的不可重复率 N0(h, γ)。
定义见 `事后分析_A与B_定义_20261110.md`。只读已有文件，只用 π0。

用法：python label_reliability.py <repair_budget 目录>   → 写 label_reliability.md / .json（在该目录下）
"""
import hashlib, json, math, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
EPS, NDRAW, NB = 1e-9, 200, 2000
GAMMAS = (0.10, 0.20, 0.30, 0.40, 0.50)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


def load_alfw(keep_planner_only):
    A = ROOT / "alfw_stale"
    D = json.loads((A / "data/design_69.json").read_text(encoding="utf-8"))
    BR = D["branches"]
    ev = {r["branch_id"]: [float(x) for x in r["results"]] for r in jl(A / "data/evals_q7_69.jsonl") if r["condition"] == "q7_T07"}
    fr = defaultdict(dict)
    for r in jl(A / "out_stale/free_b.jsonl"):
        if r["model"] == "B":
            fr[r["pos_id"]][r["i"]] = float(r["won"])
    out = []
    for t in D["trajectories"]:
        for p in t["positions"]:
            pid = f"{t['traj_id']}|t{p['t']}"
            C = [b for b in p["alts"] if keep_planner_only or BR[b].get("proposers") or BR[b].get("random")]
            if C and len(fr[pid]) == 10:
                out.append({"cl": t["traj_id"], "cands": [ev[b] for b in C], "free": [fr[pid][i] for i in range(10)]})
    return out, 5


def load_tau2():
    T = ROOT / "tau2_pilot/pool"
    rec = json.loads((T / "pi0_records.json").read_text(encoding="utf-8"))
    best = {}
    for r in jl(T / "ro_pi0.jsonl"):
        if r["executor"] != "4B" or r.get("score") != "orig_task" or r["temp"] == 0:
            continue
        key = (r["pos_id"], r["cand_id"], r["k"])
        if r.get("reward") is not None or key not in best:
            best[key] = r
    R = defaultdict(lambda: defaultdict(dict))
    for (pos, c, k), r in best.items():
        if r.get("reward") is not None:
            R[pos][c][k] = float(r["reward"])
    out, skipped = [], 0
    for pos, x in rec.items():
        reps = x.get("reps_v3") or []
        rows = [R[pos].get(c, {}) for c in reps]
        f = R[pos].get("free", {})
        if not reps:
            continue
        if any(len(a) < 6 for a in rows) or len(f) < 6:
            skipped += 1                                    # 缺续跑的位置（补充四）：曲线里不用
            continue
        out.append({"cl": f"{x['domain']}:{x['task_id']}", "cands": [[a[k] for k in range(6)] for a in rows], "free": [f[k] for k in range(6)]})
    return out, 3, skipped


def n0(data, hmax, h, g, draws, rng):
    """返回 (N0, 选中的（位置, 方向）数 / 抽样次数, 选中里 G 恰为最小可能值的比例, 逐位置 (分子, 分母, 簇))。"""
    H = (list(range(0, hmax)), list(range(hmax, 2 * hmax)))
    gmin = math.ceil(g * h - EPS) / h
    num = den = atmin = 0
    per = [[0, 0] for _ in data]
    for _ in range(draws):
        ia, ib = (sorted(rng.sample(H[0], h)), sorted(rng.sample(H[1], h))) if h < hmax else H
        for n, d in enumerate(data):
            G = [max(mean(c[i] for i in idx) for c in d["cands"]) - mean(d["free"][i] for i in idx) for idx in (ia, ib)]
            for a, b in ((0, 1), (1, 0)):
                if G[a] >= g - EPS:
                    den += 1; per[n][1] += 1
                    atmin += abs(G[a] - gmin) < 1e-6
                    if G[b] < g - EPS:
                        num += 1; per[n][0] += 1
    return (num / den if den else float("nan")), den / draws, (atmin / den if den else float("nan")), per


def boot(data, per):
    by = defaultdict(list)
    for d, p in zip(data, per):
        by[d["cl"]].append(p)
    T = sorted(by)
    rng = random.Random(0)
    vals = []
    for _ in range(NB):
        nn = dd = 0
        for t in (rng.choice(T) for _ in T):
            for a, b in by[t]:
                nn += a; dd += b
        if dd:
            vals.append(nn / dd)
    vals.sort()
    return vals[int(.05 * len(vals))], vals[min(len(vals) - 1, int(.95 * len(vals)))]


envs = {}
envs["ALFWorld（原集合）"] = load_alfw(True)
envs["ALFWorld（无规划器流水线）"] = load_alfw(False)
d, hm, sk = load_tau2()
envs["τ²（v3 可执行修复候选）"] = (d, hm)
out = {"tau2_skipped_incomplete": sk}
lines = ["# 事后分析 B：同一策略下标签的不可重复率 N0(h, γ)（2026-11-10；事后、探索性；只用 π0）", "",
         "N0 = P(另一半内部的 G < γ | 这一半内部的 G ≥ γ)。h = 每一半每个分支的续跑次数；h 没取满时是 200 次随机抽取的平均。",
         f"格子里：N0（选中的位置 × 方向数；选中里 G 恰为超过门槛的最小可能值的比例）。h 取满的格子另给按轨迹 / 任务重抽的 90% 区间。τ² 有 {sk} 个位置缺续跑，不用。", ""]
for name, (data, hmax) in envs.items():
    lines += [f"## {name}（{len(data)} 个位置，{len({x['cl'] for x in data})} 个簇）", "", "| h \\ γ | " + " | ".join(f"{g:.2f}" for g in GAMMAS) + " |", "|---|" + "---|" * len(GAMMAS)]
    o = {}
    for h in range(1, hmax + 1):
        row = []
        for g in GAMMAS:
            v, n, am, per = n0(data, hmax, h, g, 1 if h == hmax else NDRAW, random.Random(0))
            cell = {"N0": v, "n_sel": n, "at_min": am}
            s = f"{v:.3f}（{n:.0f}；{am:.2f}）"
            if h == hmax:
                cell["ci"] = boot(data, per)
                s = f"**{v:.3f}** [{cell['ci'][0]:.3f}, {cell['ci'][1]:.3f}]（{n:.0f}；{am:.2f}）"
            o[f"h{h}_g{g:.2f}"] = cell
            row.append(s)
        lines.append(f"| {h} | " + " | ".join(row) + " |")
    out[name] = o
    lines.append("")

# ---- 追加（看过上面的曲线之后加的，定义文件里没有）：一半选出的标签在另一半上的平均增益（诚实估计；并列的最好候选取平均）
def honest(data, hmax, h, g):
    H = (list(range(hmax))[:h], list(range(hmax, 2 * hmax))[:h])
    v = []
    for d in data:
        for a, b in ((0, 1), (1, 0)):
            pa = [mean(c[i] for i in H[a]) for c in d["cands"]]
            if max(pa) - mean(d["free"][i] for i in H[a]) >= g - EPS:
                bs = [k for k, x in enumerate(pa) if abs(x - max(pa)) < 1e-9]
                v.append(mean(mean(d["cands"][k][i] for i in H[b]) for k in bs) - mean(d["free"][i] for i in H[b]))
    return mean(v), len(v)


lines += ["## 追加：一半选出的标签，在另一半上的平均增益（看过曲线后加的）", "",
          "| 环境 | π0 自由选择成功率 | h | γ = 0.20 | γ = 0.50 |", "|---|---:|---:|---|---|"]
out["honest_gain"] = {}
for name, (data, hmax) in envs.items():
    fr = mean(mean(d["free"]) for d in data)
    for h in sorted({3, hmax}):
        cells = [honest(data, hmax, h, g) for g in (0.20, 0.50)]
        out["honest_gain"][f"{name}|h{h}"] = {"free": fr, "g0.20": cells[0], "g0.50": cells[1]}
        lines.append(f"| {name} | {fr:.3f} | {h} | " + " | ".join(f"{m:+.3f}（{n} 个）" for m, n in cells) + " |")
lines.append("")
(ROOT / "label_reliability.md").write_text("\n".join(lines), encoding="utf-8")
(ROOT / "label_reliability.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n".join(lines))
