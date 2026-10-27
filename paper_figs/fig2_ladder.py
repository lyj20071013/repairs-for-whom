"""figure 2：The stronger enforcers are not better assessors。

Cross-axis：Average success rate of the executor on all random branches；Axes：By the executor 10 evaluate the repair selected from these continuations under the deployed executor 7B when，Losses relative to matching selection
（Aligning deviations 90% interval）。Data Read From repair_transfer STEVE REPORT ON THE（Read-only）：
    out_ladder_gw/ladder_report.md、out_ladder_alfw/ladder_report.md

    python fig2_ladder.py --rt D:/Paper/repair_transfer --out D:/Paper/overleaf/figures/fig2_ladder.pdf
"""
import argparse
import re
from pathlib import Path

import style

NUM = r"([+-]?\d+\.\d+)"


def parse(rt):
    """Back {environment: [(executor, success rate, loss, lower bound, upper bound)]}。"""
    out = {}
    # gridworld：| q14 Sample 10 runs | +0.595 | +0.067 [-0.003, +0.101]（Correct [+0.017, +0.120]） |
    g = (Path(rt) / "out_ladder_gw" / "ladder_report.md").read_text(encoding="utf-8")
    a = (Path(rt) / "out_ladder_alfw" / "ladder_report.md").read_text(encoding="utf-8")
    pat_g = {"7B": r"\| 7B Deployment 10 runs \|", "14B": r"\| q14 Sample 10 runs \|", "flash": r"\| qflash Sample 10 runs \|"}
    pat_a = {"7B": r"\| 7B Deployment 10 runs \|", "14B": r"\| q14 continuation 10 runs \|", "flash": r"\| qflash continuation 10 runs \|"}
    rows = []
    for ex, p in pat_g.items():
        m = re.search(p + rf" {NUM} \| {NUM} \[{NUM}, {NUM}\]（Correct \[{NUM}, {NUM}\]）", g)
        rows.append((ex, float(m.group(2)), float(m.group(5)), float(m.group(6))))
    out["gw"] = rows
    rows = []
    for ex, p in pat_a.items():
        m = re.search(p + rf" {NUM} \| {NUM} \| \[{NUM}, {NUM}\] \| \[{NUM}, {NUM}\] \|", a)
        rows.append((ex, float(m.group(2)), float(m.group(5)), float(m.group(6))))
    out["alfw"] = rows
    succ = {}
    for env, txt in (("gw", g), ("alfw", a)):
        s = {"7B": None}
        for m in re.finditer(r"Average success rate for random branches [^：]*：7B (\d\.\d+)；(q14|qflash) (\d\.\d+)", txt):
            s["7B"] = float(m.group(1))
            s["14B" if m.group(2) == "q14" else "flash"] = float(m.group(3))
        succ[env] = s
    return {env: [(ex, succ[env][ex], l, lo, hi) for ex, l, lo, hi in rows] for env, rows in out.items()}


def plot(data, out, png=False):
    plt = style.setup()
    fig, ax = plt.subplots(figsize=(style.COLUMN, 1.95))
    style.clean(ax)
    ax.axhline(0, color=style.MUTED, lw=0.6, zorder=1)
    spec = {"gw": ("Gridworld", style.BLUE, "o"), "alfw": ("ALFWorld", style.ORANGE, "s")}
    name = {"7B": "7B (deployed)", "14B": "14B", "flash": "Qwen3.8-flash"}
    for env, rows in data.items():
        lab, col, mk = spec[env]
        x = [r[1] for r in rows]
        y = [r[2] for r in rows]
        ax.plot(x, y, color=col, lw=1.2, zorder=2)
        for ex, s, l, lo, hi in rows:
            ax.errorbar([s], [l], yerr=[[l - lo], [hi - l]], fmt=mk, color=col, ms=4.2, mec="white", mew=0.5,
                        elinewidth=0.8, capsize=0, zorder=3)
        ax.text(x[-1] + 0.02, y[-1], lab, color=style.TEXT, fontsize=6.5, va="center", ha="left")
        # Executors name is on the left side of the point（7B the two points nearly overlap，only in gridworld Point below point once）
        for ex, s, l, lo, hi in rows:
            if ex == "7B":
                if env == "gw":
                    ax.annotate(name[ex], (s, l), xytext=(-2, -9), textcoords="offset points", ha="center",
                                va="top", fontsize=5.8, color=style.MUTED)
                continue
            off, ha = ((5, -5), "left") if (env, ex) == ("gw", "14B") else ((-6, 0), "right")
            ax.annotate(name[ex], (s, l), xytext=off, textcoords="offset points", ha=ha, va="center",
                        fontsize=5.8, color=style.MUTED)
    ax.set_xlim(0.15, 1.08)
    ax.set_ylim(-0.12, 0.8)
    ax.set_xlabel("executor's success rate from the repaired states")
    ax.set_ylabel("loss when its choice\nis given to the 7B agent")
    fig.tight_layout(pad=0.2)
    style.save(fig, out, png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", required=True)
    ap.add_argument("--out", default="fig2_ladder.pdf")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    data = parse(a.rt)
    for env, rows in data.items():
        for r in rows:
            print(env, r)
    plot(data, a.out, a.png)


if __name__ == "__main__":
    main()
