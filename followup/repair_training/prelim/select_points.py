"""第 5 篇（方向 B）事前分析之二：样本数相同、由打标签者挑点时，训练目标质量差多少。

探索性；只读 repair_transfer 的 out_ds_alfw（上一篇已分析过的数据），不改动它，结果只用于设计与功效估计。
2026-10-28 17:06 写（北京时间；原标 17:10，换算有误，17:38 更正）；运行前没有看过任何新数据。

与 label_targets.py 的区别：label_targets.py 让每个决策点都进训练集（打标签者只决定"改什么"）；
这里每个打标签者用自己的续跑估计每个决策点的最佳候选净收益 ĝ = max_a ŝ(a)，在全部决策点上按 ĝ 取前 N 个
（N 各组相同，并列随机打破），每个入选点上取它的 argmax 候选作目标（打标签者同时决定"改什么"和"在哪里改"）。
零续跑的打标签者没有 ĝ：规划器不验证（U）= 随机 N 个点 + 规划器动作；随机（R）= 随机 N 个点 + 随机候选。
另加"每条轨迹一个位置"规则（N = 轨迹数）：每条轨迹在早 / 中 / 晚三个点里取 ĝ 最大的一个（与上一篇的选择问题一致）。

训练集质量 = 入选样本的目标在 7B 留出 5 次续跑上的净收益的平均（交叉拟合，与 label_targets.py 相同）；
另报"训练集总净收益 / 全部决策点数"，即把没入选的点记 0 后的平均，便于与"全部入选"版比较。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from label_targets import load, make_value  # noqa: E402

LAB = ("q7", "q14", "l8", "g7", "g14", "mix", "planner", "random")


def scores(d, V, ev, k, rng, n14):
    """一次重复：返回每个决策点 {ev7: 各候选的 7B 留出净收益, sel: {L: 各候选的估计}, pl, traj}。"""
    br = d["branches"]
    pts = []
    pm = rng.permutation(10)
    A, B = pm[:k], pm[5:]
    for ti, t in enumerate(d["trajectories"]):
        for p in t["positions"]:
            alts, o = p["alts"], p["orig"]
            pl = np.array([bool(br[a].get("planner")) for a in alts])
            if not pl.any():
                continue
            i14 = rng.permutation(n14)[:k]
            il8 = rng.permutation(10)[:k]
            s7 = np.array([V(a, "q7", A) - V(o, "q7", A) for a in alts])
            s14 = np.array([V(a, "q14", i14) - V(o, "q14", i14) for a in alts])
            sel = {"q7": s7, "q14": s14, "mix": (s7 + s14) / 2,
                   "l8": np.array([V(a, "l8", il8) - V(o, "l8", il8) for a in alts]),
                   "g7": np.array([V(a, "q7", greedy=True) - V(o, "q7", greedy=True) for a in alts]),
                   "g14": np.array([V(a, "q14", greedy=True) - V(o, "q14", greedy=True) for a in alts])}
            pts.append({"ev7": np.array([V(a, "q7", B) - V(o, "q7", B) for a in alts]),
                        "sel": sel, "pl": pl, "traj": ti})
    return pts


def choose(pts, L, N, rule, rng):
    """返回 [(决策点下标, 目标候选的 7B 留出净收益)]，共 N 个（rule='topN'）或每条轨迹一个（rule='per_traj'）。"""
    n = len(pts)
    tgt, ghat = np.zeros(n), np.zeros(n)
    for i, p in enumerate(pts):
        if L == "planner":
            ids = np.flatnonzero(p["pl"])
            tgt[i], ghat[i] = p["ev7"][ids].mean(), rng.random()
        elif L == "random":
            j = rng.integers(len(p["ev7"]))
            tgt[i], ghat[i] = p["ev7"][j], rng.random()
        else:
            s = p["sel"][L]
            ids = np.flatnonzero(s >= s.max() - 1e-12)
            tgt[i] = p["ev7"][ids].mean()                 # 并列：在并列者上平均（期望意义的随机打破）
            ghat[i] = s.max() + 1e-6 * rng.random()        # 点之间并列：随机打破
    if rule == "all":
        return np.arange(n), tgt
    if rule == "topN":
        keep = np.argsort(-ghat)[:N]
        return keep, tgt[keep]
    keep = []                                              # per_traj
    for ti in sorted({p["traj"] for p in pts}):
        idx = [i for i, p in enumerate(pts) if p["traj"] == ti]
        keep.append(idx[int(np.argmax(ghat[idx]))])
    keep = np.array(keep)
    return keep, tgt[keep]


def rname(r, f):
    return {"all": "全部入选（原草案）", "per_traj": "每条轨迹 1 个点"}.get(r) or f"前 {f:.0%} 个点"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d, ev = load("alfw", a.data)
    V = make_value("alfw", d, ev)
    n14 = len(next(v for (c, _), v in ev.items() if c == "q14_T07"))
    rng = np.random.default_rng(a.seed)
    fracs = (1.0, 2 / 3, 1 / 2, 1 / 3)
    rules = [("all", 1.0)] + [("topN", f) for f in fracs[1:]] + [("per_traj", None)]
    res = {(r, f, L): [] for r, f in rules for L in LAB}
    ovl = {(r, f, L): [] for r, f in rules for L in LAB}
    npts = None
    for _ in range(a.reps):
        pts = scores(d, V, ev, a.k, rng, n14)
        npts = len(pts)
        for r, f in rules:
            N = int(round(f * npts)) if f else None
            ref = None
            for L in LAB:
                keep, v = choose(pts, L, N, r, rng)
                res[(r, f, L)].append((v.mean(), v.sum() / npts, len(v)))
                if L == "q7":
                    ref = set(keep.tolist())
                ovl[(r, f, L)].append(len(ref & set(keep.tolist())) / len(keep))
    out = [f"### ALFWorld 6.9，k = {a.k}，决策点 {npts} 个（有规划器动作的点），重复 {a.reps} 次，种子 {a.seed}", "",
           "每格：入选样本目标的 7B 留出净收益平均（括号内：与 7B 自选点集的重合比例）。S − X 等差值在表下。", ""]
    head = "| 规则 | N | " + " | ".join(LAB) + " |"
    out += [head, "|" + "---|" * (len(LAB) + 2)]
    for r, f in rules:
        name = rname(r, f)
        Nn = int(np.mean([x[2] for x in res[(r, f, 'q7')]]))
        cells = [f"{np.mean([x[0] for x in res[(r, f, L)]]):+.3f} ({np.mean(ovl[(r, f, L)]):.2f})" for L in LAB]
        out.append(f"| {name} | {Nn} | " + " | ".join(cells) + " |")
    out += ["", "| 规则 | S − X（14B） | S − G（7B 贪心） | S − U（规划器不验证） | S − R（随机） | (S − X)/(S − R) | S − Llama |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for r, f in rules:
        name = rname(r, f)
        m = {L: np.mean([x[0] for x in res[(r, f, L)]]) for L in LAB}
        out.append(f"| {name} | {m['q7'] - m['q14']:+.3f} | {m['q7'] - m['g7']:+.3f} | {m['q7'] - m['planner']:+.3f} | "
                   f"{m['q7'] - m['random']:+.3f} | {(m['q7'] - m['q14']) / (m['q7'] - m['random']):.2f} | {m['q7'] - m['l8']:+.3f} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
