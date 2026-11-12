"""τ² 主实验的分析（2026-11-05 写；第 6 篇；读数定义见 `登记_τ²主实验_20261105.md` 第 6 节；只用 CPU）。

用法（tau2_pilot 下）：
  python main_analyze.py --design pool/design_run.jsonl --adm-v2 pool/adm_v2.jsonl --adm-v3 pool/adm_v3.jsonl \
      --w0 pool/w0.json --rollouts pool/ro_pi0.jsonl pool/ro_pi1_A.jsonl pool/ro_pi1_B.jsonl pool/ro_pi1s2_A.jsonl \
      --pi0 4B --pi1 A=4B-A B=4B-B --pi1-seed2 A=4B-A2 --drift pool/drift.jsonl \
      --e2e-pi0 <π0 端到端 results.json …> --e2e-pi1 A=<…> B=<…> --out pool/report_main.md
  探针核对（--probe）：设计文件里没有 fold 的位置都当作 A 折，π1 = --pi1 A=…；不需要 --w0。

主要输出（第 6 节）：R1（自由接手、端到端）；H2（交叉拟合的点过期率 = 主估计；同一算法的检验 Δ = S1 − N0）；
H1（D_V = V(全部重新验证) − V(只查位置) 的逐位置配对均值、单侧 90% 上界、非劣效判定、成本比、保护条款）；
次要：L_w0 与分解、L_both；描述：刷新的 V / Q / 保留比例、最好候选换了的比例、按来源、阈值 0.10 / 0.30、v2 敏感性、
两折各自、种子 2、核对样本、去掉反复报错的位置、漂移。
区间：按任务成簇的稳健标准误（CR1），t 分布（自由度 = 簇数 − 1）；另报按任务成簇、按折分层重抽 2,000 次的百分位区间（D_V、H2、R1）。
"""
import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path

H = [(0, 1, 2), (3, 4, 5)]
ALLK = (0, 1, 2, 3, 4, 5)
SPLITS = [(H[0], H[1]), (H[1], H[0])]
EPS = 1e-9


def sseed(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


# ---------------------------------------------------------------- t 分布（不依赖 scipy）
def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    hh = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(c) > 1e-300 else 1e300
        hh *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(c) > 1e-300 else 1e300
        de = d * c
        hh *= de
        if abs(de - 1) < 1e-12:
            break
    return hh


def _ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    p = 0.5 * _ibeta(df / 2, 0.5, x)
    return 1 - p if t > 0 else p


def t_q(p, df):
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------- 按任务成簇的均值与比率
def cr_mean(vals, clusters):
    """均值、CR1 标准误、簇数。"""
    pairs = [(v, c) for v, c in zip(vals, clusters) if v is not None]
    n = len(pairs)
    if n == 0:
        return None, None, 0
    m = sum(v for v, _ in pairs) / n
    by = collections.defaultdict(float)
    for v, c in pairs:
        by[c] += v - m
    G = len(by)
    if G < 2:
        return m, None, G
    return m, math.sqrt(G / (G - 1) * sum(x * x for x in by.values())) / n, G


def cr_ratio(nums, dens, clusters):
    """Σnum / Σden，线性化的 CR1 标准误。"""
    sn, sd = sum(nums), sum(dens)
    if sd <= 0:
        return None, None, 0
    r = sn / sd
    by = collections.defaultdict(float)
    for a, b, c in zip(nums, dens, clusters):
        by[c] += a - r * b
    G = len([c for c in by])
    if G < 2:
        return r, None, G
    return r, math.sqrt(G / (G - 1) * sum(x * x for x in by.values())) / sd, G


def ci_t(m, se, G, two=True, level=0.90):
    if m is None or se is None or G < 2:
        return None
    q = t_q(1 - (1 - level) / 2 if two else level, G - 1)
    return (m - q * se, m + q * se)


# ---------------------------------------------------------------- 读数据
def load_rates(paths, models):
    best = {}
    for p in paths:
        for r in jl(p):
            if r["executor"] not in models or r.get("score") != "orig_task" or r["temp"] == 0:
                continue
            key = (r["executor"], r["pos_id"], r["cand_id"], r["k"])
            if r.get("reward") is not None or key not in best:
                best[key] = r
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (m, pos, cand, k), r in best.items():
        if r.get("reward") is not None:
            out[(m, pos)][cand][k] = float(r["reward"])
    return out


def p_on(rr, c, ks):
    v = [rr.get(c, {}).get(k) for k in ks]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def argbest(rr, cands, ks, pos):
    sc = [(p_on(rr, c, ks), sseed("TIE", pos, c), c) for c in cands]
    sc = [x for x in sc if x[0] is not None]
    return max(sc)[2] if sc else None


def has_all(rr, cands):
    return all(len(rr.get(c, {})) > 0 for c in cands) and len(rr.get("free", {})) > 0


# ---------------------------------------------------------------- 逐位置记录
def record(d, reps, r0, r1, thr):
    pos = d["pos_id"]
    rec = {"pos": pos, "cluster": f"{d['domain']}:{d['task_id']}", "fold": d.get("fold", "A"), "domain": d["domain"],
           "n_rep": len(reps), "free0": p_on(r0, "free", ALLK), "free1": p_on(r1, "free", ALLK) if r1 else None}
    if not reps or "free" not in r0:
        return rec
    kinds = {c["cand_id"]: "+".join(c["kinds"]) for c in d["cands"]}
    ctype = {c["cand_id"]: ("带工具调用" if (c.get("msg") or {}).get("tool_calls") else "纯文字") for c in d["cands"]}   # 补充二

    def G(rr):
        g = []
        for s, e in SPLITS:
            b = argbest(rr, reps, s, pos)
            pb, pf = p_on(rr, b, e), p_on(rr, "free", e)
            if b is not None and pb is not None and pf is not None:
                g.append(pb - pf)
        return mean(g)

    def Gh(rr, hh):
        v = [p_on(rr, c, hh) for c in reps]
        v = [x for x in v if x is not None]
        pf = p_on(rr, "free", hh)
        return None if not v or pf is None else max(v) - pf

    rec["G0"] = G(r0)
    rec["Gh0"] = [Gh(r0, hh) for hh in H]
    rec["best0"] = argbest(r0, reps, ALLK, pos)
    rec["best0_kind"] = kinds.get(rec["best0"])
    rec["best0_type"] = ctype.get(rec["best0"])
    rec["bestH0"] = [argbest(r0, reps, hh, pos) for hh in H]
    rec["p0_best0"] = p_on(r0, rec["best0"], ALLK)
    rec["full1"] = bool(r1) and has_all(r1, reps)
    if r1 and "free" in r1:                      # 复用标签的价值（种子 2 也能算）
        rec["reuse_val1"] = None if p_on(r1, rec["best0"], ALLK) is None else p_on(r1, rec["best0"], ALLK) - p_on(r1, "free", ALLK)
    if not rec["full1"]:
        return rec
    rec["G1"] = G(r1)
    rec["Gh1"] = [Gh(r1, hh) for hh in H]
    rec["best1"] = argbest(r1, reps, ALLK, pos)
    rec["bestH1"] = [argbest(r1, reps, hh, pos) for hh in H]
    Ls = []
    for s, e in SPLITS:
        b1 = argbest(r1, reps, s, pos)
        x, y = p_on(r1, b1, e), p_on(r1, rec["best0"], e)
        if x is not None and y is not None:
            Ls.append(x - y)
    rec["L"] = mean(Ls)
    a0, p0 = rec["best0"], rec["p0_best0"]
    ref = {"reuse": [], "point": [], "pointlabel": [], "label": [], "full": []}
    for s, e in SPLITS:
        fS, fE = p_on(r1, "free", s), p_on(r1, "free", e)
        val = lambda act: None if p_on(r1, act, e) is None or fE is None else p_on(r1, act, e) - fE
        ref["reuse"].append((True, val(a0)))
        keep_pt = p0 is not None and fS is not None and p0 - fS >= thr - EPS
        ref["point"].append((keep_pt, val(a0)))
        p1a0 = p_on(r1, a0, s)
        ref["pointlabel"].append((keep_pt and p1a0 is not None and p1a0 - fS >= thr - EPS, val(a0)))
        ref["label"].append((p1a0 is not None and fS is not None and p1a0 - fS >= thr - EPS, val(a0)))   # 只查标签：π1 跑 a0 与 free
        b1 = argbest(r1, reps, s, pos)
        keep_f = b1 is not None and fS is not None and p_on(r1, b1, s) - fS >= thr - EPS
        ref["full"].append((keep_f, val(b1)))
    rec["refresh"] = ref

    def V(k):
        v = [(x if keep else 0.0) for keep, x in ref[k] if x is not None]
        return mean(v) if len(v) == len(ref[k]) else None   # 两个切分都要有值，才能逐位置配对
    rec["V"] = {k: V(k) for k in ref}
    rec["d"] = None if rec["V"]["full"] is None or rec["V"]["point"] is None else rec["V"]["full"] - rec["V"]["point"]
    rec["d_label"] = None if rec["V"]["full"] is None or rec["V"]["label"] is None else rec["V"]["full"] - rec["V"]["label"]
    return rec


# ---------------------------------------------------------------- 汇总
def summarize(recs, thr, boot=0, seed=0):
    worth = lambda g: g is not None and g >= thr - EPS
    S = {"n_pos": len(recs)}
    # R1：自由接手
    fr = [r for r in recs if r["free0"] is not None and r["free1"] is not None]
    m, se, G = cr_mean([r["free1"] - r["free0"] for r in fr], [r["cluster"] for r in fr])
    S["R1"] = {"n": len(fr), "free0": mean([r["free0"] for r in fr]), "free1": mean([r["free1"] for r in fr]), "diff": m, "se": se,
               "G": G, "ci90": ci_t(m, se, G), "lower1": None if se is None else m - t_q(0.90, G - 1) * se}
    S["R1"]["confirmed"] = S["R1"]["lower1"] is not None and S["R1"]["lower1"] > 0
    # w0 与 π1 读数
    w0 = [r for r in recs if worth(r.get("G0"))]
    w0f = [r for r in w0 if r.get("full1")]
    S["n_w0"], S["n_w0_with_pi1"] = len(w0), len(w0f)
    S["frac_w0"] = len(w0) / len(recs) if recs else None
    both = [r for r in w0f if worth(r.get("G1"))]
    stale = [r for r in w0f if not worth(r.get("G1"))]
    S["n_both"], S["frac_both"] = len(both), len(both) / len(recs) if recs else None
    # H2 主估计（交叉拟合）
    m, se, G = cr_mean([float(not worth(r["G1"])) for r in w0f], [r["cluster"] for r in w0f])
    S["H2_main"] = {"point_stale": m, "se": se, "G": G, "ci90": ci_t(m, se, G)}
    # H2 检验（同一算法：每一半内部的 G）
    nums1, nums0, dens, cls = [], [], [], []
    miss = 0
    for r in recs:
        if r.get("Gh0") is None:
            continue
        cond = [worth(g) for g in r["Gh0"]]
        if not any(cond):
            continue
        if not r.get("full1"):
            miss += 1
            continue
        n0 = n1 = dd = 0
        for i, j in ((0, 1), (1, 0)):
            if cond[i] and r["Gh0"][j] is not None and r["Gh1"][j] is not None:
                dd += 1
                n0 += not worth(r["Gh0"][j])
                n1 += not worth(r["Gh1"][j])
        if dd:
            nums0.append(n0); nums1.append(n1); dens.append(dd); cls.append(r["cluster"])
    N0 = sum(nums0) / sum(dens) if dens else None
    S1 = sum(nums1) / sum(dens) if dens else None
    dl, se, G = cr_ratio([a - b for a, b in zip(nums1, nums0)], dens, cls)
    lower = None if se is None else dl - t_q(0.90, G - 1) * se
    S["H2_test"] = {"N0": N0, "S1": S1, "delta": dl, "se": se, "G": G, "lower1": lower, "n_pos": len(dens),
                    "n_missing_pi1": miss, "pass": lower is not None and lower > 0}
    # H1：D_V
    dv = [r for r in w0f if r.get("d") is not None]
    m, se, G = cr_mean([r["d"] for r in dv], [r["cluster"] for r in dv])
    up = None if se is None else m + t_q(0.90, G - 1) * se
    S["H1"] = {"D_V": m, "se": se, "G": G, "n": len(dv), "upper1": up, "ci90": ci_t(m, se, G),
               "noninferior": up is not None and up < 0.05,
               "cost_ratio": mean([r["n_rep"] + 1 for r in dv]),
               "V": {k: mean([r["V"][k] for r in dv]) for k in ("reuse", "point", "pointlabel", "label", "full")}}
    m2, se2, G2 = cr_mean([r["d_label"] for r in dv], [r["cluster"] for r in dv])
    S["H1"]["D_V_label"] = {"D": m2, "ci90": ci_t(m2, se2, G2), "upper1": None if se2 is None else m2 + t_q(0.90, G2 - 1) * se2,
                            "lower1": None if se2 is None else m2 - t_q(0.90, G2 - 1) * se2}
    S["H1"]["lower1"] = None if se is None else S["H1"]["D_V"] - t_q(0.90, G - 1) * se
    S["H1"]["noninferior_0.03"] = up is not None and up < 0.03          # 补充一：+0.03 下的判定（敏感性）
    S["H1"]["note_0.03_0.05"] = up is not None and 0.03 <= up < 0.05     # 上界在 +0.03 至 +0.05 之间要加注明
    S["H1"]["matrix"] = matrix(S["H1"]["upper1"], S["H1"]["lower1"], S["H1"]["D_V_label"]["upper1"], S["H1"]["D_V_label"]["lower1"])
    S["changed"] = S["H2_test"]["pass"] or S["R1"]["confirmed"]
    if not S["changed"]:
        S["H1"]["verdict"] = "π1 变化太小，不具判别力（D_V 只作描述）"
    elif S["H1"]["noninferior"]:
        S["H1"]["verdict"] = "重新验证动作不划算（单侧 90% 上界 < +0.05）"
    else:
        S["H1"]["verdict"] = "判不出（报告估计与区间）"
    # Q 与保留比例（描述）
    for k in ("reuse", "point", "pointlabel", "label", "full"):
        kept = [v for r in w0f for keep, v in r["refresh"][k] if keep and v is not None]
        tot = sum(len(r["refresh"][k]) for r in w0f)
        S[f"Q_{k}"] = mean(kept)
        S[f"keep_{k}"] = len(kept) / tot if tot else None
    # 次要：L_w0 与分解
    lw = [r for r in w0f if r.get("L") is not None]
    m, se, G = cr_mean([r["L"] for r in lw], [r["cluster"] for r in lw])
    S["L_w0"] = {"L": m, "se": se, "G": G, "ci90": ci_t(m, se, G),
                 "share_stale": len(stale) / len(w0f) if w0f else None, "L_stale": mean([r["L"] for r in stale]),
                 "share_both": len(both) / len(w0f) if w0f else None, "L_both": mean([r["L"] for r in both]),
                 "n_both": len(both), "both_insufficient": len(both) < 15}
    # 最好候选换了（两边都值得修）与对照
    S["best_changed_both"] = mean([float(r["best1"] != r["best0"]) for r in both])
    S["split_half_pi1_both"] = mean([float(r["bestH1"][0] != r["bestH1"][1]) for r in both])
    S["split_half_pi0_both"] = mean([float(r["bestH0"][0] != r["bestH0"][1]) for r in both])
    # 按 π0 最好候选的来源
    src = collections.defaultdict(list)
    for r in w0f:
        src[r.get("best0_kind")].append(r)
    S["by_source"] = {k: {"n_w0": len(v), "L": mean([r["L"] for r in v]),
                          "n_both": sum(worth(r["G1"]) for r in v),
                          "best_changed_both": mean([float(r["best1"] != r["best0"]) for r in v if worth(r["G1"])])}
                      for k, v in src.items()}
    # 补充二：按 π0 最好修复 a0 的类型（纯文字 / 带工具调用）分开：点过期率（交叉拟合）、L、最好候选换了的比例
    typ = collections.defaultdict(list)
    for r in w0f:
        typ[r.get("best0_type")].append(r)
    S["by_type"] = {}
    for k, v in typ.items():
        vb = [r for r in v if worth(r["G1"])]
        m_, se_, G_ = cr_mean([float(not worth(r["G1"])) for r in v], [r["cluster"] for r in v])
        S["by_type"][k] = {"n_w0": len(v), "point_stale": m_, "ci90": ci_t(m_, se_, G_), "L": mean([r["L"] for r in v]),
                           "n_both": len(vb), "best_changed_both": mean([float(r["best1"] != r["best0"]) for r in vb])}
    if boot:
        S["boot"] = bootstrap(recs, thr, boot, seed)
    return S


def state(upper, lower, bound=0.05):
    """小 = 单侧 90% 上界 < +0.05；大 = 单侧 90% 下界 > 0 且不是"小"；其余 = 不确定。"""
    if upper is None or lower is None:
        return "不确定"
    if upper < bound:
        return "小"
    if lower > 0:
        return "大"
    return "不确定"


MATRIX = {  # (D_V 的状态, D_V(标签) 的状态) -> 解读（补充一第 2 条，事先写定）
    ("小", "小"): "只查位置就够",
    ("大", "小"): "动作没过期，但只查位置会误丢标签：推荐只查标签",
    ("大", "大"): "动作过期，需要重新选",
    ("小", "大"): "异常，需检查（比较 V(只查位置) 与 V(只查标签) 及两者的保留比例）",
    ("小", "不确定"): "只查位置就够（主判定）；重新选动作的价值不确定",
    ("大", "不确定"): "只查位置不够；差别来自误丢标签还是动作过期，分不清",
    ("不确定", "小"): "判不出；重新选动作没有明显价值（只查标签已接近全部重新验证）",
    ("不确定", "大"): "判不出；有动作过期的迹象",
    ("不确定", "不确定"): "判不出",
}


def matrix(up1, lo1, up2, lo2):
    a, b = state(up1, lo1), state(up2, lo2)
    return {"D_V": a, "D_V_label": b, "reading": MATRIX[(a, b)]}


def bootstrap(recs, thr, n, seed):
    """按任务成簇、按折分层重抽：D_V、H2 主估计、Δ、R1 的 90% 百分位区间。"""
    rng = random.Random(seed)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        by[r["fold"]][r["cluster"]].append(r)
    keys = {"D_V": ("H1", "D_V"), "point_stale": ("H2_main", "point_stale"), "delta": ("H2_test", "delta"), "R1": ("R1", "diff")}
    vals = collections.defaultdict(list)
    for _ in range(n):
        smp = []
        for f, cl in by.items():
            names = sorted(cl)
            for c in (rng.choice(names) for _ in names):
                smp.extend(cl[c])
        s = summarize(smp, thr)
        for k, (a, b) in keys.items():
            if s[a][b] is not None:
                vals[k].append(s[a][b])
    out = {}
    for k, v in vals.items():
        v.sort()
        out[k] = (v[int(0.05 * len(v))], v[int(0.95 * len(v)) - 1]) if len(v) > 20 else None
    return out


# ---------------------------------------------------------------- 漂移与端到端
def drift(path, pi0, pi1s, fold_of):
    D = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in jl(path):
        D[(r["model"], r["rep"])][r["pos_id"]].append(r["canon"])

    def tv(a, b):
        ca, cb = collections.Counter(a), collections.Counter(b)
        return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))
    base = D.get((pi0, "1"), {})
    out = {}
    for name, (model, rep), fold in [("noise_pi0#2", (pi0, "2"), None)] + [(f"{f}:{m}", (m, "1"), f) for f, m in pi1s]:
        o = D.get((model, rep), {})
        ps = [p for p in base if p in o and base[p] and o[p] and (fold is None or fold_of.get(p) == fold)]
        if ps:
            out[name] = {"n_pos": len(ps), "tv": mean([tv(base[p], o[p]) for p in ps]),
                         "noise_tv_same_pos": mean([tv(base[p], D[(pi0, "2")][p]) for p in ps if D.get((pi0, "2"), {}).get(p)]),
                         "mode_changed": mean([float(collections.Counter(base[p]).most_common(1)[0][0] != collections.Counter(o[p]).most_common(1)[0][0]) for p in ps]),
                         "tool_call": mean([float(x.startswith("T:")) for p in ps for x in o[p]]),
                         "invalid": mean([float(x == "INVALID") for p in ps for x in o[p]])}
    return out


def e2e(pi0_files, pi1_files, folds):
    def rewards(files):
        out = collections.defaultdict(list)
        for f in files:
            R = json.loads(Path(f).read_text(encoding="utf-8"))
            dom = R["info"]["environment_info"]["domain_name"] if "info" in R else None
            for s in R["simulations"]:
                if s.get("reward_info") is None:
                    continue
                out[f"{dom}:{s['task_id']}"].append(float(s["reward_info"]["reward"]))
        return out
    r0 = rewards(pi0_files)
    res = {}
    allv, allc = [], []
    for fold, files in pi1_files:
        r1 = rewards(files)
        tasks = [t for t in r1 if folds.get(t) == fold and t in r0]
        d = [mean(r1[t]) - mean(r0[t]) for t in tasks]
        m, se, G = cr_mean(d, tasks)
        res[fold] = {"n_tasks": len(tasks), "pi0": mean([mean(r0[t]) for t in tasks]), "pi1": mean([mean(r1[t]) for t in tasks]),
                     "diff": m, "ci90": ci_t(m, se, G), "n_dialogs_pi1": sum(len(r1[t]) for t in tasks)}
        allv += d; allc += tasks
    m, se, G = cr_mean(allv, allc)
    res["pooled"] = {"diff": m, "ci90": ci_t(m, se, G), "n_tasks": len(allc)}
    return res


# ---------------------------------------------------------------- 主程序
def build(design, A, R, pi0, pi1, thr, rule):
    recs = []
    for d in design:
        reps = [c["cand_id"] for c in d["cands"] if c.get("restore_ok") and "free" not in c["kinds"] and c["kinds"] != ["orig"]
                and A[rule].get((d["pos_id"], c["cand_id"]), False)]
        f = d.get("fold", "A")
        r0 = R.get((pi0, d["pos_id"]), {})
        r1 = R.get((pi1[f], d["pos_id"]), {}) if f in pi1 else {}
        recs.append(record(d, reps, r0, r1, thr))
    return recs


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, (tuple, list)) and len(x) == 2 and all(isinstance(y, (int, float)) for y in x):
        return f"[{x[0]:+.{nd}f}, {x[1]:+.{nd}f}]"
    if isinstance(x, float):
        return f"{x:+.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", nargs="+", required=True)
    ap.add_argument("--adm-v2", nargs="+", required=True)
    ap.add_argument("--adm-v3", nargs="+", required=True)
    ap.add_argument("--w0", default=None, help="main_pool.py w0 写的 w0.json（核对 π1 跑过的位置与事先定的一致）")
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--pi0", default="4B")
    ap.add_argument("--pi1", nargs="+", required=True, help="折=模型名，如 A=4B-A B=4B-B")
    ap.add_argument("--pi1-seed2", nargs="*", default=[], help="折=模型名，如 A=4B-A2")
    ap.add_argument("--rule", choices=["v2", "v3"], default="v3")
    ap.add_argument("--thr", type=float, default=0.20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--drift", default=None)
    ap.add_argument("--e2e-pi0", nargs="*", default=[])
    ap.add_argument("--e2e-pi1", nargs="*", default=[], help="折=results.json[,results.json]")
    ap.add_argument("--folds", default="pool/folds.json")
    ap.add_argument("--probe", action="store_true", help="探针核对：没有 fold 的位置当作 A 折")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    design = [d for p in a.design for d in jl(p)]
    if not a.probe and any("fold" not in d for d in design):
        raise SystemExit("[停下] 设计文件里有位置没有 fold（用 main_pool.py branches 写的 design_run.jsonl）")
    A = {"v2": {}, "v3": {}}
    for k, files in (("v2", a.adm_v2), ("v3", a.adm_v3)):
        for p in files:
            for r in jl(p):
                A[k][(r["pos_id"], r["cand_id"])] = bool(r["ok"])
    pi1 = dict(x.split("=", 1) for x in a.pi1)
    s2 = dict(x.split("=", 1) for x in a.pi1_seed2)
    R = load_rates(a.rollouts, {a.pi0, *pi1.values(), *s2.values()})
    recs = build(design, A, R, a.pi0, pi1, a.thr, a.rule)
    res = {"rule": a.rule, "thr": a.thr, "pi0": a.pi0, "pi1": pi1}
    # 核对：π1 跑全部修复候选的位置 ⊇ 事先定的集合
    if a.w0:
        W = json.loads(Path(a.w0).read_text(encoding="utf-8"))
        have = {r["pos"] for r in recs if r.get("full1")}
        need = set(W["pi1_full"]) | set(W["check"])
        res["w0_check"] = {"pi1_full_planned": len(need), "missing": sorted(need - have)[:20], "n_missing": len(need - have)}
        pre = set(W["w0"][a.rule])
        now = {r["pos"] for r in recs if r.get("G0") is not None and r["G0"] >= a.thr - EPS}
        res["w0_check"]["w0_same_as_planned"] = pre == now
        loops = set(W.get("loop_positions", []))
        check = set(W.get("check", []))
    else:
        loops, check = set(), set()
    res["all"] = summarize(recs, a.thr, boot=a.boot)
    for f in sorted({r["fold"] for r in recs}):
        res[f"fold_{f}"] = summarize([r for r in recs if r["fold"] == f], a.thr)
    for dom in sorted({r["domain"] for r in recs}):
        res[f"domain_{dom}"] = summarize([r for r in recs if r["domain"] == dom], a.thr)
    # 敏感性
    other = "v2" if a.rule == "v3" else "v3"
    res[f"sens_{other}"] = summarize(build(design, A, R, a.pi0, pi1, a.thr, other), a.thr)
    for t in (0.10, 0.50):   # 11-05：k = 6 时 G 只取 1/6 的整数倍，0.30 与 0.20 等价（都 = 1/3），改报 0.10（1/6）与 0.50（3/6）
        res[f"sens_thr_{t}"] = summarize(build(design, A, R, a.pi0, pi1, t, a.rule), t)
    if loops:
        res["sens_no_loops"] = summarize([r for r in recs if r["pos"] not in loops], a.thr)
    # 核对样本：π0 不值得修的位置上 π1 值得修的比例
    ck = [r for r in recs if r["pos"] in check and r.get("full1") and r.get("G1") is not None]
    res["check_sample"] = {"n": len(ck), "frac_pi1_worth": mean([float(r["G1"] >= a.thr - EPS) for r in ck])}
    # 种子 2：复用标签的点过期（π0 的最好修复在 π1 下是否仍比 free 好 ≥ 阈值），两个种子在同一批位置上并列
    for f, m2 in s2.items():
        r2 = build(design, A, R, a.pi0, {f: m2}, a.thr, a.rule)
        by1 = {r["pos"]: r for r in recs}
        ps = [r for r in r2 if r["fold"] == f and r.get("G0") is not None and r["G0"] >= a.thr - EPS
              and r.get("reuse_val1") is not None and by1[r["pos"]].get("reuse_val1") is not None]
        res[f"seed2_{f}"] = {"n": len(ps),
                             "reuse_stale_seed1": mean([float(by1[r["pos"]]["reuse_val1"] < a.thr - EPS) for r in ps]),
                             "reuse_stale_seed2": mean([float(r["reuse_val1"] < a.thr - EPS) for r in ps]),
                             "reuse_value_seed1": mean([by1[r["pos"]]["reuse_val1"] for r in ps]),
                             "reuse_value_seed2": mean([r["reuse_val1"] for r in ps]),
                             "free_seed1": mean([by1[r["pos"]]["free1"] for r in ps]), "free_seed2": mean([r["free1"] for r in ps])}
    if a.drift:
        fold_of = {d["pos_id"]: d.get("fold", "A") for d in design}
        res["drift"] = drift(a.drift, a.pi0, [(f, m) for f, m in pi1.items()] + [(f, m) for f, m in s2.items()], fold_of)
    if a.e2e_pi0:
        folds = json.loads(Path(a.folds).read_text(encoding="utf-8"))
        res["e2e"] = e2e(a.e2e_pi0, [(x.split("=", 1)[0], x.split("=", 1)[1].split(",")) for x in a.e2e_pi1], folds)
    Path(str(a.out).rsplit(".", 1)[0] + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    Path(a.out).write_text(report(res), encoding="utf-8")
    print(report(res))
    return 0


def report(res):
    S = res["all"]
    L = [f"# τ² 主实验报告（main_analyze.py；修复候选 {res['rule']}；阈值 {res['thr']}；π0 = {res['pi0']}，π1 = {res['pi1']}）", ""]
    if "w0_check" in res:
        L.append(f"核对：事先定的 π1 全部修复候选位置 {res['w0_check']['pi1_full_planned']} 个，缺 {res['w0_check']['n_missing']}；"
                 f"π0 值得修的位置与事先写定的一致：{fmt(res['w0_check']['w0_same_as_planned'])}")
    L += [f"位置 {S['n_pos']}；π0 值得修 {S['n_w0']}（{fmt(S['frac_w0'])}，有 π1 读数 {S['n_w0_with_pi1']}）；两边都值得修 {S['n_both']}（{fmt(S['frac_both'])}）", "",
          "## R1 训练让策略变好", f"- 自由接手 π0 {fmt(S['R1']['free0'])} → π1 {fmt(S['R1']['free1'])}，差 {fmt(S['R1']['diff'])} "
          f"90% 区间 {fmt(S['R1']['ci90'])}，单侧下界 {fmt(S['R1']['lower1'])}（{S['R1']['G']} 个任务）→ 得到确认：{fmt(S['R1']['confirmed'])}"]
    if "e2e" in res:
        e = res["e2e"]
        L.append(f"- 端到端（按折配对）：合并差 {fmt(e['pooled']['diff'])} {fmt(e['pooled']['ci90'])}；" +
                 "；".join(f"{f} 折 π0 {fmt(v['pi0'])} → π1 {fmt(v['pi1'])}（{v['n_tasks']} 个任务）" for f, v in e.items() if f != "pooled"))
    t = S["H2_test"]
    L += ["", "## H2 点过期",
          f"- 主估计（交叉拟合）：{fmt(S['H2_main']['point_stale'])} 90% 区间 {fmt(S['H2_main']['ci90'])}",
          f"- 检验（每一半内部的 G）：噪声底 N0 {fmt(t['N0'])}，对照 S1 {fmt(t['S1'])}，Δ {fmt(t['delta'])}，单侧 90% 下界 {fmt(t['lower1'])}"
          f"（{t['n_pos']} 个位置、{t['G']} 个任务；缺 π1 读数 {t['n_missing_pi1']}）→ {'点过期高于噪声' if t['pass'] else '未能与噪声区分'}"]
    h1 = S["H1"]
    L += ["", "## H1 刷新：D_V = V(全部重新验证) − V(只查位置)（主读数）",
          f"- D_V {fmt(h1['D_V'])}（{h1['n']} 个位置、{h1['G']} 个任务），90% 区间 {fmt(h1['ci90'])}，单侧 90% 上界 {fmt(h1['upper1'])}",
          f"- 成本比（全部重新验证 / 只查位置，每位置分支数）：{fmt(h1['cost_ratio'], 2)}",
          f"- 敏感性：判定线 +0.03 下{'判"不划算"' if h1['noninferior_0.03'] else '判不出'}"
          + ("；**上界在 +0.03 至 +0.05 之间：注明合成数据中 π1 与 π0 相同时 D_V 约为 −0.02**" if h1["note_0.03_0.05"] else ""),
          f"- π1 确实变了（第 6.4A 节）：{fmt(S['changed'])} → **{h1['verdict']}**",
          f"- V：复用 {fmt(h1['V']['reuse'])}，只查位置 {fmt(h1['V']['point'])}，查位置 + 查标签 {fmt(h1['V']['pointlabel'])}，只查标签 {fmt(h1['V']['label'])}，全部重新验证 {fmt(h1['V']['full'])}",
          f"- 对照：D_V(标签) = V(全部重新验证) − V(只查标签) {fmt(h1['D_V_label']['D'])} {fmt(h1['D_V_label']['ci90'])}，单侧上界 {fmt(h1['D_V_label']['upper1'])}，单侧下界 {fmt(h1['D_V_label']['lower1'])}",
          f"- 解读表（补充一第 2 条）：D_V {h1['matrix']['D_V']}、D_V(标签) {h1['matrix']['D_V_label']} → {h1['matrix']['reading']}"
          + ("" if S["changed"] else "（π1 变化太小，只作描述）"),
          f"- Q（保留标签的平均收益）/ 保留比例：" + "；".join(f"{k} {fmt(S['Q_' + k])} / {fmt(S['keep_' + k])}" for k in ("reuse", "point", "pointlabel", "label", "full"))]
    if "boot" in S:
        L.append(f"- 对照：按任务成簇、按折分层重抽的 90% 区间：{ {k: fmt(v) for k, v in S['boot'].items()} }")
    lw = S["L_w0"]
    L += ["", "## 次要：动作过期",
          f"- L_w0 {fmt(lw['L'])} 90% 区间 {fmt(lw['ci90'])}；分解：点过期部分 {fmt(lw['share_stale'])} × {fmt(lw['L_stale'])} + 两边都值得修 {fmt(lw['share_both'])} × {fmt(lw['L_both'])}",
          f"- L_both（描述）{fmt(lw['L_both'])}，{lw['n_both']} 个位置" + ("（< 15：不足，预期内）" if lw["both_insufficient"] else ""),
          f"- 最好候选换了的比例（两边都值得修）{fmt(S['best_changed_both'])}；对照：π1 两半 {fmt(S['split_half_pi1_both'])}，π0 两半 {fmt(S['split_half_pi0_both'])}",
          "- 按 π0 最好候选的来源：" + "；".join(f"{k} n={v['n_w0']} L {fmt(v['L'])} 两边都值得修 {v['n_both']}" for k, v in S["by_source"].items()),
          "- 按 π0 最好修复的类型（补充二）：" + "；".join(f"{k} n={v['n_w0']} 点过期 {fmt(v['point_stale'])} {fmt(v['ci90'])} L {fmt(v['L'])} 两边都值得修 {v['n_both']}、其中最好候选换了 {fmt(v['best_changed_both'])}" for k, v in S["by_type"].items()),
          "", "## 两折、领域与敏感性（描述）", "| 子集 | 位置 | π0 值得修 | 两边 | R1 | 点过期（主） | Δ 下界 | D_V | D_V 上界 | L_w0 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in res.items():
        if isinstance(v, dict) and "H1" in v and k != "all":
            L.append(f"| {k} | {v['n_pos']} | {v['n_w0']} | {v['n_both']} | {fmt(v['R1']['diff'])} | {fmt(v['H2_main']['point_stale'])} | "
                     f"{fmt(v['H2_test']['lower1'])} | {fmt(v['H1']['D_V'])} | {fmt(v['H1']['upper1'])} | {fmt(v['L_w0']['L'])} |")
    L += ["", f"- 核对样本：{res['check_sample']}"]
    for k, v in res.items():
        if k.startswith("seed2_"):
            L.append(f"- 种子 2（{k[6:]} 折，{v['n']} 个位置）：复用标签的点过期 {fmt(v['reuse_stale_seed1'])} / {fmt(v['reuse_stale_seed2'])}；"
                     f"复用的价值 {fmt(v['reuse_value_seed1'])} / {fmt(v['reuse_value_seed2'])}；自由接手 {fmt(v['free_seed1'])} / {fmt(v['free_seed2'])}")
    if "drift" in res:
        L.append("- 漂移：" + "；".join(f"{k} TV {fmt(v['tv'])}（同位置噪声底 {fmt(v['noise_tv_same_pos'])}），众数变化 {fmt(v['mode_changed'])}" for k, v in res["drift"].items()))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
