"""第 5 篇事前分析之三（探索性；只读 6.9 的旧数据）：位置（7B 筛选 / 随机）× 动作（7B argmax / 规划器）四格的目标质量。
2026-10-28 19:52 写（北京时间）；没有任何新数据。规则同 select_points.py：前 50% 按 7B 的 ĝ（k 次续跑），交叉拟合用另外 5 次评。
"""
import sys
import numpy as np
from label_targets import load, make_value

def main(data, k=3, reps=300, seed=0):
    d, ev = load("alfw", data); V = make_value("alfw", d, ev); br = d["branches"]
    rng = np.random.default_rng(seed)
    cells = {c: [] for c in ("S", "S-plan", "S-rand", "U")}
    agree = {c: [] for c in ("top", "rand")}
    for _ in range(reps):
        pm = rng.permutation(10); A, B = pm[:k], pm[5:]
        pts = []
        for t in d["trajectories"]:
            for p in t["positions"]:
                alts, o = p["alts"], p["orig"]
                pl = np.array([bool(br[a].get("planner")) for a in alts])
                if not pl.any():
                    continue
                s = np.array([V(a, "q7", A) - V(o, "q7", A) for a in alts])
                e = np.array([V(a, "q7", B) - V(o, "q7", B) for a in alts])
                ids = np.flatnonzero(s >= s.max() - 1e-12)
                pts.append({"ghat": s.max() + 1e-6 * rng.random(), "t7": e[ids].mean(), "tp": e[pl].mean(),
                            "same": float(pl[ids].mean())})
        n = len(pts); N = n // 2
        top = sorted(range(n), key=lambda i: -pts[i]["ghat"])[:N]
        rnd = rng.permutation(n)[:N]
        cells["S"].append(np.mean([pts[i]["t7"] for i in top]))
        cells["S-plan"].append(np.mean([pts[i]["tp"] for i in top]))
        cells["S-rand"].append(np.mean([pts[i]["t7"] for i in rnd]))
        cells["U"].append(np.mean([pts[i]["tp"] for i in rnd]))
        agree["top"].append(np.mean([pts[i]["same"] for i in top]))
        agree["rand"].append(np.mean([pts[i]["same"] for i in rnd]))
    m = {c: float(np.mean(v)) for c, v in cells.items()}
    print(f"k = {k}，{n} 个决策点，前 50% = {N}，重复 {reps} 次")
    print("| 格 | 位置 | 动作 | 目标在 7B 上的净收益 |\n|---|---|---|---:|")
    for c, (a, b) in {"S": ("7B 筛选", "7B argmax"), "S-plan": ("7B 筛选", "规划器"),
                      "S-rand": ("随机", "7B argmax"), "U": ("随机", "规划器")}.items():
        print(f"| {c} | {a} | {b} | {m[c]:+.3f} |")
    print(f"7B argmax 恰是规划器动作的比例：筛选的点 {np.mean(agree['top']):.2f}，随机点 {np.mean(agree['rand']):.2f}")
    print(f"挑点（固定 7B 动作）S − S-rand = {m['S'] - m['S-rand']:+.3f}；挑点（固定规划器）S-plan − U = {m['S-plan'] - m['U']:+.3f}")
    print(f"排序（筛选的点）S − S-plan = {m['S'] - m['S-plan']:+.3f}；排序（随机点）S-rand − U = {m['S-rand'] - m['U']:+.3f}")
    print(f"交互 (S − S-plan) − (S-rand − U) = {(m['S'] - m['S-plan']) - (m['S-rand'] - m['U']):+.3f}")

if __name__ == "__main__":
    for k in (3, 5):
        main(sys.argv[1], k=k); print()
