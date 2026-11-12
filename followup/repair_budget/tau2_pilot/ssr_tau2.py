"""盲态样本量重估（SSR）规则的模拟（2026-11-05 写；主实验准备第 3 步；只用 CPU）。
读数：D_V = V(全部重新验证) − V(只查位置)，π0 值得修位置上的逐位置配对差的均值；单侧非劣效：90% 区间上界 < +0.05 → "重新验证动作不划算"。
规则：池先跑 N1 = 400 个位置；π1 跑完、看 D_V 均值之前，只用逐位置配对差的方差与按任务成簇的标准误算出上界半宽 h = 1.645·SE；
      h > 0.04 → 扩到 N2 = 600（预留的 200 个位置），否则停在 400。最后在实际样本上判定。
模型：
  - π0 值得修的比例 q0 ~ Beta(1 + 17, 1 + 43)（探针 60 个评价位置，v2 去随机候选）；每个位置独立地以 q0 值得修。
  - 池任务 G = 51 个有位置的任务（建池模拟：每任务 20 次、上限 6 条失败、每条 3 个位置），位置均匀多项分配到任务。
  - 逐位置配对差 d = μ + s·σ·(√ICC·u_任务 + √(1−ICC)·e)：σ = 0.200（探针 17 个位置的经验标准差），
    e 从探针的 17 个配对差（中心化、标准化）里有放回抽（保留"大多为 0"的形状），u ~ N(0, 1)。
  - 标准误：按任务成簇的稳健标准误（CR1）。判定用正态近似（实际分析用按任务成簇重抽 2,000 次）。
"""
import argparse, json, math, random

EMP = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.333, -0.333, 0.0, 0.0, 0.0, -0.167, -0.167, 0.167, 0.333]


def cr_se(d, g):
    n = len(d)
    if n < 3:
        return float("inf")
    m = sum(d) / n
    by = {}
    for x, t in zip(d, g):
        by[t] = by.get(t, 0.0) + (x - m)
    G = len(by)
    if G < 2:
        return float("inf")
    return math.sqrt(G / (G - 1) * sum(v * v for v in by.values())) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--G", type=int, default=51)
    ap.add_argument("--n1", type=int, default=400)
    ap.add_argument("--n2", type=int, default=600)
    ap.add_argument("--hmax", type=float, default=0.04)
    ap.add_argument("--bound", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m0 = sum(EMP) / len(EMP)
    sd0 = math.sqrt(sum((x - m0) ** 2 for x in EMP) / (len(EMP) - 1))
    E = [(x - m0) / sd0 for x in EMP]
    rng = random.Random(0)
    rows = []
    for icc in (0.0, 0.3):
        for sm in (1.0, 1.5, 2.0):
            for mu in (0.0, 0.02, 0.039, 0.05):
                ext = succ = s400 = s600 = 0
                hw = []
                for _ in range(a.sims):
                    q0 = rng.betavariate(18, 44)
                    u = [rng.gauss(0, 1) for _ in range(a.G)]
                    d, g = [], []
                    def add(n):
                        for _ in range(n):
                            if rng.random() < q0:
                                t = rng.randrange(a.G)
                                e = math.sqrt(icc) * u[t] + math.sqrt(1 - icc) * rng.choice(E)
                                d.append(mu + sm * sd0 * e); g.append(t)
                    add(a.n1)
                    se1 = cr_se(d, g); h1 = 1.645 * se1; hw.append(h1)
                    up1 = sum(d) / len(d) + h1 if d else float("inf")
                    s400 += up1 < a.bound
                    if h1 > a.hmax:
                        ext += 1
                        add(a.n2 - a.n1)
                    se = cr_se(d, g); up = sum(d) / len(d) + 1.645 * se
                    succ += up < a.bound
                    if h1 <= a.hmax:
                        s600 += up1 < a.bound      # 不扩时 600 档与 400 档相同；下面另算固定 600
                hw.sort()
                rows.append(dict(icc=icc, sm=sm, mu=mu, p_ext=ext / a.sims, p_succ=succ / a.sims, p_succ_fixed400=s400 / a.sims,
                                 h400_med=hw[len(hw) // 2]))
    # 固定 600 的成功率（对照）
    for r in rows:
        ok = 0
        for _ in range(a.sims // 2):
            q0 = rng.betavariate(18, 44); u = [rng.gauss(0, 1) for _ in range(a.G)]; d, g = [], []
            for _ in range(a.n2):
                if rng.random() < q0:
                    t = rng.randrange(a.G)
                    d.append(r["mu"] + r["sm"] * sd0 * (math.sqrt(r["icc"]) * u[t] + math.sqrt(1 - r["icc"]) * rng.choice(E))); g.append(t)
            ok += sum(d) / len(d) + 1.645 * cr_se(d, g) < a.bound
        r["p_succ_fixed600"] = ok / (a.sims // 2)
    L = ["# 盲态样本量重估规则的模拟（ssr_tau2.py）", "",
         f"规则：先 {a.n1} 个位置；上界半宽 > {a.hmax} → 扩到 {a.n2}。判定：90% 上界 < +{a.bound}（单侧非劣效）。μ = D_V 的真值（0.05 = 界上，此行的成功率 = 第一类错误率；0.039 = 探针点估计）。", "",
         "| ICC | 方差放大 | 400 时半宽中位 | 触发扩展的概率 | μ | 判定成功率：规则 | 固定 400 | 固定 600 |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['icc']} | {r['sm']} | {r['h400_med']:.3f} | {r['p_ext']:.2f} | {r['mu']} | {r['p_succ']:.2f} | {r['p_succ_fixed400']:.2f} | {r['p_succ_fixed600']:.2f} |")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(rows, open(a.out.rsplit(".", 1)[0] + ".json", "w"), ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
