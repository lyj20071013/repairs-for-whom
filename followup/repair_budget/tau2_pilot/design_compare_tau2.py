"""τ² 主实验：现设计（池 / 训练按任务对半）与按任务两折交叉的功效与卡时比较（2026-11-05 写；只用 CPU）。
读数：D_V（配对差，单侧：90% 上界 < +0.05）；L_w0（π0 值得修位置上的平均 L，同样报单侧上界半宽）。
模型同 ssr_tau2.py：q0 ~ Beta(18, 44)；位置均匀分到 G 个有位置的任务；逐位置的 (D_V 差, L) 成对地从探针 17 个
π0 值得修位置里重抽（保持两者相关），中心化后乘方差倍数，再加任务效应（ICC）；按任务成簇的稳健标准误，
临界值用 t 分布（自由度 G − 1）。
"""
import json, math, random, sys

D = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.333, -0.333, 0.0, 0.0, 0.0, -0.167, -0.167, 0.167, 0.333]
L = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.167, -0.167, 0.0, 0.0]


def tq(df, z=1.6449):
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def cr(d, g):
    n = len(d); m = sum(d) / n; by = {}
    for x, t in zip(d, g):
        by[t] = by.get(t, 0.0) + (x - m)
    G = len(by)
    return m, math.sqrt(G / (G - 1) * sum(v * v for v in by.values())) / n, G


def sim(G, N, icc, sm, mu, rng, n_sims):
    md, ml = sum(D) / len(D), sum(L) / len(L)
    sd_d = math.sqrt(sum((x - md) ** 2 for x in D) / 16); sd_l = math.sqrt(sum((x - ml) ** 2 for x in L) / 16)
    hd, hl, sd_, sl_, n0s = [], [], 0, 0, []
    for _ in range(n_sims):
        q0 = rng.betavariate(18, 44)
        ud = [rng.gauss(0, 1) for _ in range(G)]; ul = [rng.gauss(0, 1) for _ in range(G)]
        dd, ll, gg = [], [], []
        for _ in range(N):
            if rng.random() < q0:
                t = rng.randrange(G); j = rng.randrange(17)
                dd.append(mu + sm * (math.sqrt(icc) * sd_d * ud[t] + math.sqrt(1 - icc) * (D[j] - md)))
                ll.append(sm * (math.sqrt(icc) * sd_l * ul[t] + math.sqrt(1 - icc) * (L[j] - ml)))
                gg.append(t)
        n0s.append(len(dd))
        m, se, Gh = cr(dd, gg); h = tq(Gh - 1) * se; hd.append(h); sd_ += m + h < 0.05
        m, se, Gh = cr(ll, gg); h = tq(Gh - 1) * se; hl.append(h); sl_ += m + h < 0.05
    q = lambda v: sorted(v)[len(v) // 2]
    return dict(n0=q(n0s), hD=q(hd), hL=q(hl), pD=sd_ / n_sims, pL=sl_ / n_sims)


def cost(N, design, B=3.48, rep=2.07, q0=0.29, train=500):
    rate = 360.0
    pi0 = N * B * 6 / rate
    pi1 = (N * 6 + (q0 + 0.15 * (1 - q0)) * N * rep * 6) / rate          # 自由选择全跑 + 值得修的跑全部修复候选 + 15% 核对
    fixed = 3 + 2.5 + 5 + 1                                                # 设计与判定、训练与质检、端到端、4B 锚对话
    if design == "cur":
        collect = 4.5
        seed2 = q0 * N * 2 * 6 / rate
        label = train * 15.4 / rate                                         # 训练位置自适应 3 + 3
        return dict(collect=collect, pi0=pi0, pi1=pi1 + seed2, label=label, fixed=fixed, total=collect + pi0 + pi1 + seed2 + label + fixed)
    collect = 2.2 if N <= 700 else 3.4
    return dict(collect=collect, pi0=pi0, pi1=pi1, label=0.0, fixed=fixed, total=collect + pi0 + pi1 + fixed)


def main():
    rng = random.Random(0)
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    designs = [("现设计 池 400 / 训练 500", "cur", 51, 400, 90), ("现设计 池 600 / 训练 500", "cur", 51, 600, 90),
               ("两折 共 400", "cv", 102, 400, 36), ("两折 共 600", "cv", 102, 600, 54), ("两折 共 800", "cv", 102, 800, 72)]
    out, Lines = [], ["| 方案 | 簇数 | 每个 π1 的标签 | π0 值得修（中位） | ICC | 方差 | D_V 上界半宽 | P(D_V 判不划算 \\| 0) | P(… \\| +0.02) | L_w0 上界半宽 | P(L_w0 上界 < 0.05 \\| 0) | 卡时 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, kind, G, N, lab in designs:
        c = cost(N, kind)
        for icc in (0.0, 0.3):
            for sm in (1.0, 1.5):
                r0 = sim(G, N, icc, sm, 0.0, rng, n_sims)
                r2 = sim(G, N, icc, sm, 0.02, rng, n_sims)
                row = dict(name=name, G=G, N=N, labels=lab, icc=icc, sm=sm, **r0, pD02=r2["pD"], cost=c)
                out.append(row)
                Lines.append(f"| {name} | {G} | 约 {lab} | {r0['n0']} | {icc} | {sm} | {r0['hD']:.3f} | {r0['pD']:.2f} | {r2['pD']:.2f} | {r0['hL']:.3f} | {r0['pL']:.2f} | {c['total']:.0f} |")
    Lines += ["", "卡时分解（每卡时 360 次续跑；π0 每位置 3.48 个分支，含原动作；π1 只跑修复候选与自由选择）："]
    for name, kind, G, N, lab in designs:
        c = cost(N, kind)
        Lines.append(f"- {name}：收集 {c['collect']:.1f}、π0 {c['pi0']:.1f}、π1 {c['pi1']:.1f}、训练位置打标签 {c['label']:.1f}、其他 {c['fixed']:.1f} → {c['total']:.0f}")
    open("probe/design_compare.md", "w", encoding="utf-8").write("\n".join(Lines) + "\n")
    json.dump(out, open("probe/design_compare.json", "w"), ensure_ascii=False, indent=1)
    print("\n".join(Lines))


if __name__ == "__main__":
    main()
