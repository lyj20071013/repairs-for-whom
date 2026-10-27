"""figure 3：(a) More for the replacement. Can you make it back?？(b) Preference check for independent renewal。

(a) Do Not Filter ALFWorld 7B Ji!：By each of the executives k Well pick the next fix，the selected repairs value for the deployed executor 7B Net proceeds（90% trajectory bootstrap interval）；
    7B The choice is cross-compatibility（5 Selection 5 Commentary，therefore limited to k = 5）。data：out_612/analysis_612.json of C['7B']（The same code counts all three lines）。
(b) 7B and 14B cases with disagreement 37 trajectories（a = 7B the original best candidate，b = 14B the original best candidate）and 15 Article Contrast（both originally have the same best candidate，
    b = 7B The second best）：Just a new run（Each branch per executor 30 runs）Yes Δ_e = R_e(r_a) − R_e(r_b)。data：out_rev612/analysis_rev.json。

    python fig3_curves.py --rt D:/Paper/repair_transfer --out D:/Paper/overleaf/figures/fig3_curves.pdf
"""
import argparse
import json
from pathlib import Path

import numpy as np

import style

NAMES = {"q7": "Qwen2.5-7B (deployed)", "l8": "Llama-3.1-8B", "q14": "Qwen2.5-14B"}
COLORS = {"q7": style.BLUE, "l8": style.AQUA, "q14": style.ORANGE}
MARKS = {"q7": "o", "l8": "s", "q14": "^"}
GX = 0.62          # The place of greedy on the cross-axis（On the logarithmic axis k = 1 Left）


def load(rt):
    rt = Path(rt)
    c = json.loads((rt / "out_612" / "analysis_612.json").read_text(encoding="utf-8"))["C"]["7B"]["rows"]
    curves = {}
    for r in c:
        if r["var"] != "all":
            continue
        k = GX if r["k"] == "greedy" else float(r["k"])
        curves.setdefault(r["selector"], []).append((k, r["net"]["mean"], *r["net"]["ci"]))
    rev = json.loads((rt / "out_rev612" / "analysis_rev.json").read_text(encoding="utf-8"))["pairs"]["q14"]
    return curves, rev


def panel_budget(ax, curves):
    style.clean(ax, grid_axis="y")
    ax.set_xscale("log")
    for sel in ("q7", "l8", "q14"):
        pts = sorted(curves[sel])
        g = [p for p in pts if p[0] == GX]
        s = [p for p in pts if p[0] != GX]
        x = [p[0] for p in s]
        ax.fill_between(x, [p[2] for p in s], [p[3] for p in s], color=COLORS[sel], alpha=0.14, lw=0, zorder=1)
        ax.plot(x, [p[1] for p in s], color=COLORS[sel], lw=1.2, marker=MARKS[sel], ms=3.0, mec="white", mew=0.4,
                zorder=3)
        for gx, gm, glo, ghi in g:
            ax.errorbar([gx], [gm], yerr=[[gm - glo], [ghi - gm]], fmt="D", color=COLORS[sel], ms=2.8, mec="white",
                        mew=0.4, elinewidth=0.7, capsize=0, zorder=3)
        lx, ly = s[-1][0], s[-1][1]
        off = {"q7": (3, 4), "l8": (3, -1), "q14": (-2, -7)}[sel]
        ha = {"q7": "left", "l8": "left", "q14": "right"}[sel]
        ax.annotate(NAMES[sel].replace(" (deployed)", "\n(deployed)"), (lx, ly), xytext=off,
                    textcoords="offset points", fontsize=5.6, color=style.TEXT, ha=ha, va="center")
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([GX, 1, 2, 3, 5, 10, 20]))
    ax.xaxis.set_major_formatter(FixedFormatter(["G", "1", "2", "3", "5", "10", "20"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.5, 34)
    ax.set_ylim(0.3, 0.92)
    ax.set_xlabel("rollouts per candidate (G: greedy)")
    ax.set_ylabel("net gain for the 7B agent")
    ax.set_title("(a) More substitute rollouts", fontsize=6.8, pad=3)


def panel_reversal(ax, rev):
    style.clean(ax, grid_axis=None)
    ax.axhline(0, color=style.GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=style.GRID, lw=0.6, zorder=0)
    ax.plot([-1, 1], [-1, 1], color=style.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    rng = np.random.default_rng(0)                 # Just to stagger the overlap point a little bit（Without prejudice to any number）
    for key, col, mk, lab in (("control", style.BLUE, "s", "agreed (control, 15)"),
                              ("cases", style.ORANGE, "o", "contested (37)")):
        rows = rev[key]["rows"]
        x = np.array([r["d7_net"] for r in rows]) + rng.uniform(-0.012, 0.012, len(rows))
        y = np.array([r["do_net"] for r in rows]) + rng.uniform(-0.012, 0.012, len(rows))
        ax.scatter(x, y, s=7, marker=mk, color=col, alpha=0.55, lw=0, zorder=2)
        mx, my = rev[key]["d7_net"], rev[key]["do_net"]
        ax.errorbar([mx["mean"]], [my["mean"]],
                    xerr=[[mx["mean"] - mx["ci"][0]], [mx["ci"][1] - mx["mean"]]],
                    yerr=[[my["mean"] - my["ci"][0]], [my["ci"][1] - my["mean"]]],
                    fmt=mk, color=col, ms=4.2, mec="white", mew=0.6, elinewidth=1.0, capsize=0, zorder=4, label=lab)
    ax.set_xlim(-0.6, 1.2)
    ax.set_ylim(-0.92, 0.75)
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.fill_between([0, 1.2], [-0.92, -0.92], [0, 0], color=style.ORANGE, alpha=0.06, lw=0, zorder=0)
    ax.set_xlabel(r"$\Delta_{\mathrm{7B}}=R_{\mathrm{7B}}(r_a)-R_{\mathrm{7B}}(r_b)$")
    ax.set_ylabel(r"$\Delta_{\mathrm{14B}}$")
    ax.legend(loc="upper left", frameon=False, fontsize=5.4, handletextpad=0.2, borderaxespad=0.1,
              markerscale=0.8)
    ax.set_title("(b) Fresh rollouts, 7B vs. 14B", fontsize=6.8, pad=3)


def plot(curves, rev, out, png=False):
    plt = style.setup()
    plt.rcParams["mathtext.fontset"] = "stix"
    fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN, 1.78), gridspec_kw={"width_ratios": [1.08, 1]})
    panel_budget(axes[0], curves)
    panel_reversal(axes[1], rev)
    fig.tight_layout(pad=0.2, w_pad=0.8)
    style.save(fig, out, png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", required=True)
    ap.add_argument("--out", default="fig3_curves.pdf")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    curves, rev = load(a.rt)
    for sel, pts in curves.items():
        print(sel, [(p[0], round(p[1], 3)) for p in sorted(pts)])
    for key in ("cases", "control"):
        print(key, rev[key]["n"], {k: round(rev[key][k]["mean"], 3) for k in ("d7_net", "do_net", "I_net")})
    plot(curves, rev, a.out, a.png)


if __name__ == "__main__":
    main()
