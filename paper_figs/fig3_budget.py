"""figure 3：The selection of losses is subject to change in the budget for deployment renewal——all candidates × k runs vs Well use the environment signal first and then well see × k runs。

Data Read From repair_transfer/out_method/method_gw.json（Development Set）and method_alfw.json（Leave），Read-only。
The loss zone is a trajectories bootstrap 90% interval（method_cascade.py，Sample evaluation that does not coincide with selection）。

    python fig3_budget.py --rt D:/Paper/repair_transfer --out D:/Paper/overleaf/figures/fig3_budget.pdf
"""
import argparse
import json
import re
from pathlib import Path

import style


def load(rt, env):
    m = json.loads((Path(rt) / "out_method" / f"method_{env}.json").read_text(encoding="utf-8"))
    rows = m["rows"]

    def series(prefix):
        pts = []
        for k, v in rows.items():
            mm = re.fullmatch(prefix + r" × (\d+) runs", k)
            if mm:
                pts.append((v["budget"], *v["loss"]))
        return sorted(pts)
    full = [p for p in series("all candidates")]
    scr = series("screening m=1")
    greedy = rows["greedy 1 runs"]
    env_only = rows["use environment signals only：oracle + slack（Zero budget）"]
    return {"full": full, "screen": scr, "greedy": (greedy["budget"], *greedy["loss"]),
            "env": env_only["loss"], "n": m["n_traj"]}


def panel(ax, d, title, env_label_below=False):
    style.clean(ax, grid_axis="y")
    ax.set_xscale("log")
    for key, col, mk, lab in (("full", style.BLUE, "o", "all candidates"), ("screen", style.ORANGE, "s", "screened")):
        pts = [p for p in d[key] if p[1] > 0 or p[0] < max(q[0] for q in d[key])]   # Reference itself（The loss is constant 0）No painting
        x = [p[0] for p in pts]
        ax.fill_between(x, [p[2] for p in pts], [p[3] for p in pts], color=col, alpha=0.15, lw=0, zorder=1)
        ax.plot(x, [p[1] for p in pts], color=col, lw=1.2, marker=mk, ms=3.0, mec="white", mew=0.4, zorder=3,
                label=lab)
    gx, gl, glo, ghi = d["greedy"]
    ax.errorbar([gx], [gl], yerr=[[gl - glo], [ghi - gl]], fmt="D", color=style.TEXT, ms=3.2, mec="white", mew=0.4,
                elinewidth=0.7, capsize=0, zorder=4)
    ax.annotate("greedy", (gx, gl), xytext=(4, 0), textcoords="offset points", fontsize=5.8, color=style.TEXT,
                va="center")
    e = d["env"][0]
    ax.axhline(e, color=style.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.03, e - 0.008 if env_label_below else e + 0.006, "env. signal only",
            transform=ax.get_yaxis_transform(), ha="left", va="top" if env_label_below else "bottom",
            fontsize=5.6, color=style.MUTED)
    ax.axhline(0, color=style.MUTED, lw=0.6, zorder=1)
    ax.set_title(title, fontsize=7, pad=3)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300]))
    ax.xaxis.set_major_formatter(FixedFormatter(["10", "30", "100", "300"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(5, 400)


def plot(gw, alfw, out, png=False):
    plt = style.setup()
    fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN, 1.75), sharey=True)
    panel(axes[0], gw, "Gridworld (design)", env_label_below=True)
    panel(axes[1], alfw, "ALFWorld (fixed settings)")
    axes[0].set_ylabel("selection loss")
    axes[0].set_ylim(-0.02, 0.4)
    h, lab = axes[0].get_legend_handles_labels()
    fig.supxlabel("deployed-agent rollouts per trajectory (log scale)", fontsize=7, y=0.02)
    fig.tight_layout(pad=0.2, w_pad=0.6, rect=(0, 0.04, 1, 0.9))
    fig.legend(h, ["all candidates $\\times$ k rollouts", "screened candidates $\\times$ k rollouts"], loc="upper center",
               ncol=2, frameon=False, handlelength=1.6, bbox_to_anchor=(0.5, 1.0))
    style.save(fig, out, png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", required=True)
    ap.add_argument("--out", default="fig3_budget.pdf")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    gw, alfw = load(a.rt, "gw"), load(a.rt, "alfw")
    for name, d in (("gw", gw), ("alfw", alfw)):
        print(name, "screen", [(p[0], round(p[1], 3)) for p in d["screen"]], "greedy", d["greedy"][:2],
              "env", round(d["env"][0], 3))
    plot(gw, alfw, a.out, a.png)


if __name__ == "__main__":
    main()
