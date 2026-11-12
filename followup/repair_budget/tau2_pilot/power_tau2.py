"""τ² 主实验的功效计算（2026-11-05 写；主实验准备第 3 步；只用 CPU）。
主读数：D = Q(全部重新验证) − Q(只查位置)（等价界 ±0.05）；配套 L_w0 = π0 值得修的位置上的平均 L，
分解为 点过期部分 与 两边都值得修部分。L_both 只作描述（报预期的位置数）。
依据：探针 60 个评价位置的逐位置记录（probe_analyze.position_record，候选按 --adm 过滤），π0 = 4B，π1 = 4B-S4r，k = 6。
模拟：
  1. π0 值得修的比例 q0 ~ Beta(1 + 探针值得修数, 1 + 其余)；池 N 个位置 → 值得修位置数 n0 ~ Bin(N, q0)。
  2. 从探针的 π0 值得修位置里有放回抽 n0 个（独立位置模型），算 D、L_w0 → 抽样分布的标准差 SE。
  3. 任务成簇：池任务 T = 62；探针里 π0 值得修的位置落在 56% 的任务上，所以 m = n0 /（0.56 T）为每个"有值得修位置的任务"
     平均的位置数，设计效应 deff = 1 + (m − 1)·ICC（m < 1 时取 1），ICC ∈ {0, 0.3}；SE × √deff。
     探针里按任务成簇与按位置独立重抽的 SE 几乎相同（D 0.084 / 0.084，L_w0 0.020 / 0.022），ICC 估不出来，所以只作情景。
  3b. 逐位置方差的放大倍数 s ∈ {1, 2}：探针的 13–17 个值得修位置里 L 大多恰好为 0（π1 与 π0 选中同一个候选），
     经验方差很可能偏小；主实验候选更多时会更大。s = 2 是保守情景。
  4. 判定的功效（正态近似，90% 区间 = 估计 ± 1.645 SE）：
     等价（区间整个在 ±0.05 内）、非劣（区间上界 < +0.05），分别在真值 = 0 与真值 = 探针点估计下算。
局限：逐位置的方差来自探针的 π1（26 个标签）；主实验的 π1 不同。k 固定为 6（与探针相同）。
"""
import argparse, json, math, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_analyze as PA

THR = 0.20


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def stat(w0):
    """w0：π0 值得修的位置记录。返回 D、L_w0、L_stale、L_both、两边都值得修的个数。"""
    q = {}
    for k in ("point", "full"):
        q[k] = mean([v for r in w0 for keep, v in r["refresh"][k] if keep and v is not None])
    D = None if q["point"] is None or q["full"] is None else q["full"] - q["point"]
    # V：每个 π0 标签平均留下的价值（丢掉的标签记 0）；D_V = V(全部重新验证) − V(只查位置)，逐位置配对
    def V(r, k):
        v = [(val if keep else 0.0) for keep, val in r["refresh"][k] if val is not None]
        return sum(v) / len(v) if v else None
    dv = [V(r, "full") - V(r, "point") for r in w0 if V(r, "full") is not None and V(r, "point") is not None]
    both = [r for r in w0 if r["G_pi1"] >= THR - 1e-9]
    stale = [r for r in w0 if r["G_pi1"] < THR - 1e-9]
    return dict(D=D, D_V=sum(dv) / len(dv) if dv else None, L_w0=mean([r["L"] for r in w0]), L_stale=mean([r["L"] for r in stale]),
                L_both=mean([r["L"] for r in both]), n_both=len(both), share_stale=len(stale) / len(w0) if w0 else None)


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def p_equiv(mu, se, b=0.05, z=1.645):
    # P(est − z se > −b 且 est + z se < b)，est ~ N(mu, se)
    lo, hi = -b + z * se, b - z * se
    return max(0.0, phi((hi - mu) / se) - phi((lo - mu) / se)) if hi > lo else 0.0


def p_noninf(mu, se, b=0.05, z=1.645):
    return phi((b - z * se - mu) / se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", nargs="+", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--adm", nargs="+", required=True)
    ap.add_argument("--pi1", default="4B-S4r")
    ap.add_argument("--N", nargs="+", type=int, default=[260, 400, 600])
    ap.add_argument("--T", type=int, default=62)
    ap.add_argument("--icc", nargs="+", type=float, default=[0.0, 0.3])
    ap.add_argument("--sdmult", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    PA.ADM = {(r["pos_id"], r["cand_id"]): bool(r["ok"]) for p in a.adm for r in jl(p)}
    R = PA.load_rates(a.rollouts, {"4B", a.pi1})
    recs = [r for r in (PA.position_record(d, R, "4B", a.pi1) for p in a.design for d in jl(p)) if r]
    w0 = [r for r in recs if r["G_pi0"] >= THR - 1e-9]
    n_pos, n_w0 = len(recs), len(w0)
    n_both = sum(r["G_pi1"] >= THR - 1e-9 for r in w0)
    obs = stat(w0)
    rng = random.Random(a.seed)
    out = {"n_pos": n_pos, "n_w0": n_w0, "n_both": n_both, "obs": obs, "rows": []}
    L = [f"# τ² 主实验功效计算（power_tau2.py；{a.adm}）", "",
         f"探针：{n_pos} 个位置，π0 值得修 {n_w0}，两边都值得修 {n_both}；观测 D = {obs['D']:+.3f}，D_V = {obs['D_V']:+.3f}，L_w0 = {obs['L_w0']:+.3f}"
         f"（点过期部分 {obs['L_stale']:+.3f}，两边都值得修部分 {obs['L_both'] if obs['L_both'] is None else round(obs['L_both'], 3)}）", "",
         "| N | π0 值得修（中位 [10%, 90%]） | 两边都值得修 | ICC | 方差放大 | D 半宽（中位 / 80% 分位） | L_w0 半宽 | P(等价 \\| D 真值 0) | P(等价 \\| 真值 = 探针) | P(非劣 \\| 0) | P(非劣 \\| 探针) | D_V 半宽 | P(D_V 等价 \\| 0) | P(D_V 等价 \\| 探针) |",
         "|---:|---|---|---:|---:|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for N in a.N:
        sims = []
        for _ in range(a.sims):
            q0 = rng.betavariate(1 + n_w0, 1 + n_pos - n_w0)
            qb = rng.betavariate(1 + n_both, 1 + n_pos - n_both)
            n0 = sum(rng.random() < q0 for _ in range(N))
            nb = sum(rng.random() < qb for _ in range(N))
            smp = [rng.choice(w0) for _ in range(max(n0, 2))]
            sims.append((n0, nb, stat(smp)))
        n0s = sorted(s[0] for s in sims)
        nbs = sorted(s[1] for s in sims)
        q = lambda v, f: v[int(f * (len(v) - 1))]
        # 抽样分布的 SE：按 n0 分组太细，改用"SE ∝ 1/√n0"：先在 n0 的中位数上估计 SE，再对每次模拟按 √(n_ref/n0) 缩放
        n_ref = q(n0s, 0.5)
        ref = [stat([rng.choice(w0) for _ in range(n_ref)]) for _ in range(a.sims)]
        def sd(key):
            v = [s[key] for s in ref if s[key] is not None]
            m = sum(v) / len(v)
            return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        seD, seL, seV = sd("D"), sd("L_w0"), sd("D_V")
        for icc, sm in [(i, m) for m in a.sdmult for i in a.icc]:
            hwD, hwL, pe0, peo, pn0, pno, hwV, pv0, pvo = [], [], [], [], [], [], [], [], []
            for n0, nb, _ in sims:
                n0 = max(n0, 2)
                m = n0 / (0.56 * a.T)
                deff = 1 + (m - 1) * icc if m > 1 else 1.0
                s_D = sm * seD * math.sqrt(n_ref / n0) * math.sqrt(deff)
                s_L = sm * seL * math.sqrt(n_ref / n0) * math.sqrt(deff)
                hwD.append(1.645 * s_D); hwL.append(1.645 * s_L)
                s_V = sm * seV * math.sqrt(n_ref / n0) * math.sqrt(deff)
                hwV.append(1.645 * s_V); pv0.append(p_equiv(0, s_V)); pvo.append(p_equiv(obs["D_V"], s_V))
                pe0.append(p_equiv(0, s_D)); peo.append(p_equiv(obs["D"], s_D))
                pn0.append(p_noninf(0, s_D)); pno.append(p_noninf(obs["D"], s_D))
            hwD.sort(); hwL.sort(); hwV.sort()
            row = dict(N=N, icc=icc, sdmult=sm, n0=[q(n0s, .1), q(n0s, .5), q(n0s, .9)], nboth=[q(nbs, .1), q(nbs, .5), q(nbs, .9)],
                       hwD=[q(hwD, .5), q(hwD, .8)], hwL=[q(hwL, .5), q(hwL, .8)],
                       hwV=[q(hwV, .5), q(hwV, .8)], pV_equiv0=mean(pv0), pV_equiv_obs=mean(pvo),
                       p_equiv0=mean(pe0), p_equiv_obs=mean(peo), p_noninf0=mean(pn0), p_noninf_obs=mean(pno))
            out["rows"].append(row)
            L.append(f"| {N} | {row['n0'][1]} [{row['n0'][0]}, {row['n0'][2]}] | {row['nboth'][1]} [{row['nboth'][0]}, {row['nboth'][2]}] | {icc} | {sm} "
                     f"| {row['hwD'][0]:.3f} / {row['hwD'][1]:.3f} | {row['hwL'][0]:.3f} / {row['hwL'][1]:.3f} "
                     f"| {row['p_equiv0']:.2f} | {row['p_equiv_obs']:.2f} | {row['p_noninf0']:.2f} | {row['p_noninf_obs']:.2f} | {row['hwV'][0]:.3f} / {row['hwV'][1]:.3f} | {row['pV_equiv0']:.2f} | {row['pV_equiv_obs']:.2f} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(str(a.out).rsplit(".", 1)[0] + ".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
