"""Plot paired selection losses from the original archived reports and primary tests."""
import argparse
import json
import re
from pathlib import Path

import style

NUM = r"([+-]?\d+\.\d+)"


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def mc(d):
    return float(d["mean"]), float(d["ci"][0]), float(d["ci"][1])


def ladder(rt):
    a = (Path(rt) / "out_ladder_alfw" / "ladder_report.md").read_text(encoding="utf-8")
    g = (Path(rt) / "out_ladder_gw" / "ladder_report.md").read_text(encoding="utf-8")
    out = {}
    for ex in ("q14", "qflash"):
        m = re.search(rf"\| 7B 10 次 − {ex} 10 次 \| {NUM} \| \[{NUM}, {NUM}\] \| \[{NUM}, {NUM}\] \|", a)
        out[("alfw", ex)] = (float(m.group(1)), float(m.group(4)), float(m.group(5)))
        m = re.search(rf"\| 7B 10 次 − {ex} 10 次 \| {NUM} \[{NUM}, {NUM}\]（校正 \[{NUM}, {NUM}\]）", g)
        out[("gw", ex)] = (float(m.group(1)), float(m.group(4)), float(m.group(5)))
    return out


def rows(rt):
    """Return grouped estimates and confidence intervals in plotting order."""
    rt = Path(rt)
    a7 = jl(rt / "out_ds_alfw" / "analysis.json")["tests"]
    t10 = jl(rt / "out_ds_alfw_l8" / "tests_610.json")
    ws = jl(rt / "out_ws_summary" / "tests_ws.json")
    lad = ladder(rt)
    R = []
    g = "Unscreened ALFWorld (60), deployed Qwen2.5-7B"
    R += [(g, "greedy replay", *mc(a7["P1"]), False),
          (g, "Qwen2.5-14B rollouts", *mc(a7["P2"]), False),
          (g, "Llama-3.1-8B rollouts", *mc(t10["H1"]), False)]
    g = "Unscreened ALFWorld (60), deployed Llama-3.1-8B"
    R += [(g, "greedy replay", *mc(t10["H3"]), False),
          (g, "Qwen2.5-7B rollouts", *mc(t10["H2"]), False)]
    g = "Unscreened WebShop (60), deployed Qwen2.5-7B; reward"
    R.append((g, "greedy replay", *mc(ws["P1w"]), True))
    if ws.get("P2w"):                       # Preserve group ordering and reserve a separate row for each group heading.
        R.append((g, "Qwen2.5-14B rollouts", *mc(ws["P2w"]), True))
    R.append((g, "Llama-3.1-8B rollouts", *mc(ws["H1w"]), True))
    if ws.get("H3w") or ws.get("H2w"):
        g = "Unscreened WebShop (60), deployed Llama-3.1-8B; reward"
        if ws.get("H3w"):
            R.append((g, "greedy replay", *mc(ws["H3w"]), True))
        if ws.get("H2w"):
            R.append((g, "Qwen2.5-7B rollouts", *mc(ws["H2w"]), True))
    g = "Screened ALFWorld (30), deployed Qwen2.5-7B"
    R += [(g, "Qwen2.5-14B rollouts", *lad[("alfw", "q14")], False),
          (g, "Qwen3.8-flash rollouts", *lad[("alfw", "qflash")], False)]
    g = "Screened gridworld (34), deployed Qwen2.5-7B"
    R += [(g, "Qwen2.5-14B rollouts", *lad[("gw", "q14")], False),
          (g, "Qwen3.8-flash rollouts", *lad[("gw", "qflash")], False)]
    return R


PRIMARY = ("Unscreened",)


def plot(R, out, png=False):
    from matplotlib.transforms import blended_transform_factory
    plt = style.setup()
    groups = []
    for r in R:
        if r[0] not in groups:
            groups.append(r[0])
    # Preserve group ordering and reserve a separate row for each group heading.
    y, heads, yv = {}, [], 0.0
    for gi, gname in enumerate(groups):
        if gi and gname.startswith("Screened") and not groups[gi - 1].startswith("Screened"):
            yv += 0.45
        heads.append((yv, gname))
        yv += 0.9
        for i, r in enumerate(R):
            if r[0] == gname:
                y[i] = yv
                yv += 0.8
        yv += 0.2
    n_lines = yv
    fig, ax = plt.subplots(figsize=(style.COLUMN, 0.118 * n_lines + 0.42))
    fig.subplots_adjust(left=0.37, right=0.97, top=0.995, bottom=0.42 / (0.118 * n_lines + 0.42))
    style.clean(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=style.MUTED, lw=0.6, zorder=1)
    for i, (gname, lab, est, lo, hi, is_ws) in enumerate(R):
        col = style.BLUE if "greedy" in lab else style.ORANGE
        mk = "D" if "greedy" in lab else "o"
        ax.errorbar([est], [y[i]], xerr=[[est - lo], [hi - est]], fmt=mk, color=col,
                    ms=3.4 if mk == "o" else 3.0, mec="white", mew=0.4, elinewidth=1.0, capsize=0, zorder=3)
        ax.text(hi + 0.015, y[i], f"{est:.2f}", va="center", ha="left", fontsize=5.8, color=style.TEXT)
    ax.set_yticks([y[i] for i in range(len(R))])
    ax.set_yticklabels([r[1] for r in R], fontsize=6.0)
    ax.tick_params(axis="y", length=0, pad=2)
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for yh, gname in heads:
        ax.text(0.005, yh, gname, transform=tr, ha="left", va="center", fontsize=6.2, fontweight="bold",
                color=style.TEXT, zorder=5, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    ax.set_ylim(n_lines - 0.15, -0.55)
    ax.set_xlim(-0.04, 0.8)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("deployment value lost (own selection $-$ substitute)")
    style.save(fig, out, png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", required=True)
    ap.add_argument("--out", default="fig2_forest.pdf")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    R = rows(a.rt)
    for r in R:
        print(f"{r[0]:42s} {r[1]:26s} {r[2]:+.3f} [{r[3]:+.3f}, {r[4]:+.3f}]")
    plot(R, a.out, a.png)


if __name__ == "__main__":
    main()
