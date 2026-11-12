"""第 5 篇主实验的粗略功效（正态近似）。参数都是假设，先导之后用实测值替换。2026-10-28 写。

差值方差 = 2·σ_seed²·(1−ρ_seed)/S + τ²/P + 2·p(1−p)/(P·m·S)              （接管读数）
         = 2·σ_seed²·(1−ρ_seed)/S + 2·p(1−p)·(1−ρ_game)/(E·S)              （端到端读数）
δ = 0.29·Γ：S − X 的每决策点目标差（0.147）占 S − R（0.500）的比例（prelim/label_targets_out.md，ALFWorld k = 3），
假设训练增益与目标平均质量成线性。功效 = Φ(δ/SE − 1.96)（单侧 0.025，即 Holm m = 2 的第一步）。
"""
from math import sqrt
from statistics import NormalDist
N = NormalDist()
RATIO = 0.147 / 0.500


def takeover(G, sig, rho, S, P=1000, m=4, p=0.25, tau=0.2):
    se = sqrt(2 * sig**2 * (1 - rho) / S + tau**2 / P + 2 * p * (1 - p) / (P * m * S))
    d = RATIO * G
    return se, d, N.cdf(d / se - 1.96)


def e2e(G, sig, rho, S, E=1000, p=0.68, rho_game=0.5):
    se = sqrt(2 * sig**2 * (1 - rho) / S + 2 * p * (1 - p) * (1 - rho_game) / (E * S))
    d = RATIO * G
    return se, d, N.cdf(d / se - 1.96)


if __name__ == "__main__":
    print("## 接管成功率（P = 1000、m = 4、p = 0.25、τ = 0.2）\n")
    print("| Γ（S − R） | 种子 SD | 组间种子相关 | 种子数 | 差的 SE | δ（S − X） | 功效 |\n|---:|---:|---:|---:|---:|---:|---:|")
    for G in (0.10, 0.20, 0.30):
        for sig in (0.01, 0.02, 0.03):
            for rho in (0.0, 0.7):
                for S in (3, 5):
                    se, d, pw = takeover(G, sig, rho, S)
                    print(f"| {G:.2f} | {sig:.2f} | {rho:.1f} | {S} | {se:.4f} | {d:.3f} | {pw:.2f} |")
    print("\n## 端到端成功率（每模型 E = 1000 局、p = 0.68、同局配对相关 0.5）\n")
    print("| Γ_e2e（S − R） | 种子 SD | 组间种子相关 | 种子数 | 差的 SE | δ（S − X） | 功效 |\n|---:|---:|---:|---:|---:|---:|---:|")
    for G in (0.04, 0.06, 0.10):
        for sig in (0.01, 0.015):
            for rho in (0.0, 0.7):
                for S in (3, 5):
                    se, d, pw = e2e(G, sig, rho, S)
                    print(f"| {G:.2f} | {sig:.3f} | {rho:.1f} | {S} | {se:.4f} | {d:.3f} | {pw:.2f} |")
