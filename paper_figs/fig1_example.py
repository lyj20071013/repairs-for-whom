"""figure 1：Same one ALFWorld On the failed trajectories，One step by one for each candidate 7B Yeah 14B Net proceeds（10-27 Simplified version：Scatter + circle each executors choice）。

Data Read From repair_transfer（Dont write anything there）：
    out_alfw_b/main_design.json、main_rollouts.jsonl     7B（Deployment，Each branch 30 runs）
    out_ladder_alfw/q14.jsonl、qflash.jsonl                14B / Qwen3.8-flash（Each branch 10 runs）
Read Reuse repair_transfer/consequence.py of load()，Same code as the other numbers in the paper。

    python fig1_example.py --rt D:/Paper/repair_transfer --out D:/Paper/overleaf/figures/fig1_example.pdf
    python fig1_example.py --rt ... --list      # List the differences between the three on all trajectories，to make changing examples easier

Example Selection（recorded here for transparency）：c263 yes 30 among these trajectories“by 14B The best fixes are to be delivered 7B Maximum loss”One of the previous ones，
and flash succeeds on every branch（Net proceeds are all 0），Show them at the same time“At what point?”Disagreements and loss of resolution of the strong executors。
This is an example of a figure to be noted，For summary results, see main text。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

EXECUTORS = [  # (Neutrons, Legend, Colours, Tag)；Colour is through CVD Validation of the three colours（Third Color Contrast < 3:1，Filled with shapes and legends）
    ("7B", "Qwen2.5-7B (deployed)", "#2a78d6", "o"),
    ("14B", "Qwen2.5-14B", "#eb6834", "^"),
    ("flash", "Qwen3.8-flash", "#1baf7a", "s"),
]
TEXT = "#0b0b0b"
MUTED = "#6b6a66"
GRID = "#e4e3df"


def load(rt):
    sys.path.insert(0, str(rt))
    import consequence  # noqa: E402  (repair_transfer Modules，Read-only)
    trajs, br, _ = consequence.load("alfw", rt)
    return trajs, br


def rate(br, b, ex):
    x = br[b]
    if x["det"] is not None:
        return float(x["det"]), 0
    s = np.concatenate(x[ex]).astype(float)
    return float(s.mean()), len(s)


def table(trajs, br, tid):
    t = next(x for x in trajs if x["id"] == tid)
    rows = []
    for p in t["positions"]:
        p0 = {ex: rate(br, p["orig"], ex)[0] for ex, *_ in EXECUTORS}
        for a in p["alts"]:
            rows.append({"t": p["t"], "action": a.split("|")[-1], "planner": bool(br[a]["expert"]),
                         **{ex: rate(br, a, ex)[0] - p0[ex] for ex, *_ in EXECUTORS},
                         **{f"p0_{ex}": p0[ex] for ex, *_ in EXECUTORS}})
    return rows


def list_all(trajs, br):
    print("traj  best7  7B-value-of-14B-pick  7B-value-of-flash-pick  flash-range")
    for t in trajs:
        rows = table(trajs, br, t["id"])
        g = {ex: np.array([r[ex] for r in rows]) for ex, *_ in EXECUTORS}
        print(f"{t['id']}  {g['7B'].max():+.2f}  {g['7B'][g['14B'].argmax()]:+.2f}  "
              f"{g['7B'][g['flash'].argmax()]:+.2f}  {np.ptp(g['flash']):.2f}")


def plot(rows, out, png=False):
    """Simplified version（10-27）：One point for each candidate，Cross-axis = For deployment 7B Net proceeds，Axes = Yeah 14B Net proceeds；
    Colours / Shapes = Fix Location；circle the repair each executor would select。flash It worked on every branch of this trajectory（Net proceeds are all 0），No painting，state in the figure caption。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
    })
    steps = sorted({r["t"] for r in rows})
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    mks = ["o", "^", "s"]
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.plot([-0.5, 1], [-0.5, 1], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    x7 = np.array([r["7B"] for r in rows])
    x14 = np.array([r["14B"] for r in rows])
    for s_, c, m in zip(steps, cols, mks):
        idx = [i for i, r in enumerate(rows) if r["t"] == s_]
        r0 = rows[idx[0]]
        ax.scatter(x7[idx], x14[idx], s=16, marker=m, color=c, edgecolors="white", linewidths=0.5, zorder=3,
                   label=f"repair at step {s_} (no-repair success: 7B {r0['p0_7B']:.2f}, 14B {r0['p0_14B']:.2f})")
    i7, i14 = int(x7.argmax()), int(x14.argmax())
    for i, who, off, ha in ((i7, "7B's choice", (-6, 13), "right"), (i14, "14B's choice", (6, -2), "left")):
        ax.scatter([x7[i]], [x14[i]], s=62, marker="o", facecolors="none", edgecolors=TEXT, linewidths=0.8,
                   zorder=4)
        r = rows[i]
        ax.annotate(f"{who}: {r['action']} at step {r['t']}\n7B {r['7B']:+.2f}, 14B {r['14B']:+.2f}",
                    (x7[i], x14[i]), xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=6.0,
                    color=TEXT, arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=0, shrinkB=4))
    ax.set_xlim(-0.12, 0.85)
    ax.set_ylim(-0.5, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_xlabel("net gain for the deployed Qwen2.5-7B agent")
    ax.set_ylabel("net gain for Qwen2.5-14B")
    ax.legend(loc="lower right", frameon=False, fontsize=5.6, handletextpad=0.1, borderaxespad=0.1,
              labelspacing=0.25)
    fig.tight_layout(pad=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    if png:                                   # Preview for，Default not to write，to avoid Overleaf Multiple files in folders
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Write {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", required=True, help="repair_transfer directory（Read-only）")
    ap.add_argument("--out", default="fig1_example.pdf")
    ap.add_argument("--traj", default="c263")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--png", action="store_true", help="Save another one PNG Preview")
    a = ap.parse_args()
    trajs, br = load(Path(a.rt))
    if a.list:
        list_all(trajs, br)
        return 0
    rows = table(trajs, br, a.traj)
    for r in rows:
        print(f"t={r['t']:2d} {r['action']:34s} 7B {r['7B']:+.2f} 14B {r['14B']:+.2f} flash {r['flash']:+.2f}"
              + ("  (planner)" if r["planner"] else ""))
    plot(rows, Path(a.out), a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
