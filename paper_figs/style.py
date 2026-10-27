"""Common Style for Papers：Fonts、Colours（passed CVD Verify the first three categories of colours）、Width。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3df"
COLUMN = 3.25          # ICML Single Column Width（Inches）


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.edgecolor": MUTED, "xtick.color": TEXT, "ytick.color": TEXT,
    })
    return plt


def clean(ax, grid_axis="y"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, out, png=False):
    from pathlib import Path
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    if png:
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Write {out}")
