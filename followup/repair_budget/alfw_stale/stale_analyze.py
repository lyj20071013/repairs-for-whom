"""第 6 篇 A 层的分析（只用 CPU）：读 stale_alfw.py 的输出，按登记（第 5 节与补充一 8.4）算读数，写 report_stale.md、analysis_stale.json。

用法：python stale_analyze.py [--out 输出目录] [--profile mock]
没有数据的读数会写"（还没有数据）"，可以在优先级 1 跑完后先看一次。区间：按轨迹重抽 2,000 次的 90% 百分位区间。
"""
import argparse
import json
import math
import os
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stale_alfw as SA  # noqa: E402
from rt_common import jdump, jl_read  # noqa: E402

TH = 0.20
N_BOOT, SEED_BOOT = 2000, 0
KC = [1, 2, 3, 5]


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


def nanmean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return mean(x)


# ---------------------------------------------------------------- 读数据
def load_all(out):
    D, ev, pos = SA.load_design()
    SA.DESIGN[0] = D
    res = defaultdict(lambda: defaultdict(dict))          # res[模型][分支或 位置|free][i] = 成功
    for stage in ("full", "two", "free_b"):
        for r in jl_read(Path(out) / f"{stage}.jsonl"):
            key = r["branch"] if r["branch"] != "free" else r["pos_id"] + "|free"
            res[r["model"]][key][r["i"]] = bool(r["won"])
    rev = defaultdict(lambda: defaultdict(dict))
    for r in jl_read(Path(out) / "reverse.jsonl"):
        rev[r["model"]][r["branch"]][r["i"]] = bool(r["won"])
    drift = defaultdict(lambda: defaultdict(list))
    for r in jl_read(Path(out) / "drift.jsonl"):
        a = r["cmd"] if r["kind"] in ("valid", "alias") else "<bad>"
        drift[r["model"]][r["pos_id"]].append((r["i"], a))
    return D, ev, pos, res, rev, drift


def lst(d, n=SA.K):
    return [d[i] for i in range(n)] if d and all(i in d for i in range(n)) else None


# ---------------------------------------------------------------- 每个位置的量
def per_position(D, ev, pos, res, model, Bfree):
    """返回 {位置: dict}；只含数据完整的位置。"""
    R = res.get(model, {})
    out = {}
    for p in pos:
        alts = p["alts"]
        r0 = {b: ev[b] for b in alts}
        b0_all = SA.best_alt(p, {b: mean(r0[b]) for b in alts})
        fr1 = lst(R.get(p["pos_id"] + "|free"))
        fr0 = Bfree.get(p["pos_id"])
        full = all(lst(R.get(b)) is not None for b in alts)
        x = {"traj": p["traj_id"], "p": p, "b0_all": b0_all, "planner_b0": bool(D["branches"][b0_all].get("planner")),
             "n_alts": len(alts)}
        # π0 的 G（交叉拟合）
        if fr0 is not None:
            g = []
            for s, e in ((slice(0, 5), slice(5, 10)), (slice(5, 10), slice(0, 5))):
                b = SA.best_alt(p, {c: mean(r0[c][s]) for c in alts})
                g.append(mean(r0[b][e]))
            x["G0"] = mean(g) - mean(fr0)
            x["G0_all"] = mean(r0[b0_all]) - mean(fr0)
        # π1 的两分支量（G~）
        rb = lst(R.get(b0_all))
        if fr1 is not None and rb is not None:
            x["Gt1"] = mean(rb) - mean(fr1)
            x["free1"] = mean(fr1)
        if fr0 is not None:
            x["free0"] = mean(fr0)
        if full and fr1 is not None:
            r1 = {b: lst(R[b]) for b in alts}
            L, g1, ctrl = [], [], []
            for s, e in ((slice(0, 5), slice(5, 10)), (slice(5, 10), slice(0, 5))):
                b1 = SA.best_alt(p, {c: mean(r1[c][s]) for c in alts})
                b0 = SA.best_alt(p, {c: mean(r0[c][0:5]) for c in alts})
                L.append(mean(r1[b1][e]) - mean(r1[b0][e]))
                g1.append(mean(r1[b1][e]))
                ctrl.append(mean(r1[b1][e]) - mean(mean(r1[c][e]) for c in alts))
            x["L"] = mean(L)
            for k0 in (3, 10):
                b0k = SA.best_alt(p, {c: mean(r0[c][0:k0]) for c in alts})
                x[f"L_k0_{k0}"] = mean([mean(r1[SA.best_alt(p, {c: mean(r1[c][s]) for c in alts})][e]) - mean(r1[b0k][e])
                                        for s, e in ((slice(0, 5), slice(5, 10)), (slice(5, 10), slice(0, 5)))])
            x["G1"] = mean(g1) - mean(fr1)
            x["ctrl"] = mean(ctrl)
            b1_all = SA.best_alt(p, {c: mean(r1[c]) for c in alts})
            x["agree"] = b1_all == b0_all
            best_v = max(mean(r1[c]) for c in alts)
            x["r5_near"] = mean(r1[b0_all]) >= best_v - 0.10
            x["r5_best"] = b1_all == b0_all
            x["Gt_minus_G"] = x["Gt1"] - x["G1"]
            x["r1"], x["fr1"], x["r0"] = r1, fr1, r0
        out[p["pos_id"]] = x
    return out


def refresh(pp, drift_tv, th=TH):
    """补充一 8.4 R4：四种刷新方式 + 按漂移挑。返回 {kc: {方式: (价值, 精确率, 丢掉的真增益, 每位置成本)}} 的逐位置值。"""
    lam = [k for k, x in pp.items() if "r1" in x and x.get("G0_all", -1) >= th]
    if not lam:
        return None, []
    tvs = [drift_tv.get(k) for k in lam if drift_tv.get(k) is not None]
    med = st.median(tvs) if tvs else None
    rows = {}
    for k in lam:
        x = pp[k]
        r1, fr1, r0 = x["r1"], x["fr1"], x["r0"]
        b0 = x["b0_all"]
        p0 = mean(r0[b0])
        row = {}
        for kc in KC:
            acc = defaultdict(list)
            for s, e in ((range(0, 5), range(5, 10)), (range(5, 10), range(0, 5))):
                sel = lambda arr: mean(arr[i] for i in list(s)[:kc])            # noqa: E731
                ev_ = lambda arr: mean(arr[i] for i in e)                          # noqa: E731
                f_sel, f_ev = sel(fr1), ev_(fr1)
                # 复用
                g_true = ev_(r1[b0]) - f_ev
                acc["reuse"].append((1, g_true, 0))
                # 只查位置
                keep = p0 - f_sel >= th
                acc["point"].append((keep, g_true, kc))
                # 查位置 + 查标签
                keep = sel(r1[b0]) - f_sel >= th
                acc["point_label"].append((keep, g_true, 2 * kc))
                # 全部重新验证
                bb = SA.best_alt(x["p"], {c: sel(r1[c]) for c in r1})
                keep_f = sel(r1[bb]) - f_sel >= th
                acc["full"].append((keep_f, ev_(r1[bb]) - f_ev, (x["n_alts"] + 1) * kc))
                # 按漂移挑
                if med is not None and drift_tv.get(k) is not None:
                    if drift_tv[k] >= med:
                        acc["drift"].append((keep_f, ev_(r1[bb]) - f_ev, (x["n_alts"] + 1) * kc))
                    else:
                        acc["drift"].append((1, g_true, 0))
            row[kc] = {m: [(float(kp), g, c) for kp, g, c in v] for m, v in acc.items()}
        rows[k] = row
    return rows, lam


def refresh_summary(rows, keys, kc, method):
    """补充二（10-30）：主读数改为"保留的标签的平均真增益" Q；另报精确率、召回率、补充一定义的总价值 V（描述）、成本。"""
    vals, kept_g, prec_num, drop, cost, useful, useful_kept = [], [], 0, [], [], 0, 0
    for k in keys:
        for (kp, g, c), (_, g0, _) in zip(rows[k][kc].get(method, []), rows[k][kc]["reuse"]):
            vals.append(kp * g)
            cost.append(c)
            useful += g0 >= TH
            if kp:
                kept_g.append(g)
                prec_num += g >= TH
                useful_kept += g >= TH
            else:
                drop.append(max(0.0, g))
    n = len(vals)
    return {"Q": mean(kept_g), "prec": prec_num / len(kept_g) if kept_g else float("nan"),
            "recall": useful_kept / useful if useful else float("nan"), "kept_frac": len(kept_g) / n if n else float("nan"),
            "V": mean(vals), "dropped_gain": sum(drop) / n if n else float("nan"), "cost": mean(cost)}


# ---------------------------------------------------------------- 漂移
def tv(a, b):
    ca, cb = Counter(a), Counter(b)
    na, nb = len(a), len(b)
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def mode(a):
    c = Counter(a)
    return max(sorted(c), key=lambda k: c[k])


def drift_table(drift):
    B = {p: [a for i, a in sorted(v)] for p, v in drift.get("B", {}).items()}
    out = {}
    for m, d in drift.items():
        if m == "B":
            continue
        tvs, ch, per, col, colB = [], [], {}, [], []
        B2 = {p: [a for i, a in sorted(v)] for p, v in drift.get("B#2", {}).items()}
        for p, v in d.items():
            if p not in B:
                continue
            a = [x for i, x in sorted(v)]
            per[p] = tv(a, B[p])
            tvs.append(per[p])
            ch.append(mode(a) != mode(B[p]))
            col.append(mean(x == y for x in a for y in B[p]))            # 两边各抽一个、相同的概率（独立样本，无偏）
            if p in B2:
                colB.append(mean(x == y for x in B2[p] for y in B[p]))
        out[m] = {"tv": mean(tvs), "greedy_change": mean(ch), "n_pos": len(tvs), "per_pos": per,
                  "same_action_prob": mean(col), "same_action_prob_BB": mean(colB) if colB else float("nan")}
    return out


# ---------------------------------------------------------------- 区间
def boot(pp_list, fn, trajs, n=N_BOOT):
    """pp_list：一个或多个 {位置: x}；fn(按位置的列表们) -> 数。按轨迹重抽。"""
    by_t = defaultdict(list)
    for k, x in pp_list[0].items():
        by_t[x["traj"]].append(k)
    rng = random.Random(SEED_BOOT)
    est = fn([list(pp_list[0].keys())])
    vals = []
    T = sorted(by_t)
    for _ in range(n):
        ks = [k for t in (rng.choice(T) for _ in T) for k in by_t[t]]
        v = fn([ks])
        if v is not None and not math.isnan(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return est, None, None
    return est, vals[int(0.05 * len(vals))], vals[min(len(vals) - 1, int(0.95 * len(vals)))]


def traj_mean(pp, ks, field):
    by = defaultdict(list)
    for k in ks:
        v = pp[k].get(field)
        if v is not None:
            by[pp[k]["traj"]].append(v)
    return mean(mean(v) for v in by.values()) if by else float("nan")


def fmt(t, nd=3):
    e, lo, hi = t
    if e is None or (isinstance(e, float) and math.isnan(e)):
        return "（还没有数据）"
    if lo is None:
        return f"{e:+.{nd}f}"
    return f"{e:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for q in range(i, j + 1):
                r[o[q]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    a, b = rk(x), rk(y)
    ma, mb = mean(a), mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else float("nan")


# ---------------------------------------------------------------- 主函数
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("STALE_OUT", "/root/autodl-tmp/stale_runs/out_stale"))
    ap.add_argument("--profile", default="real", choices=["real", "mock"])
    a = ap.parse_args()
    SA.setup(a.profile, a.out)
    D, ev, pos, res, rev, drift = load_all(a.out)
    Bfree = {}
    for p in pos:
        v = lst(res.get("B", {}).get(p["pos_id"] + "|free"))
        if v is not None:
            Bfree[p["pos_id"]] = v
    dt = drift_table(drift)
    trajs = sorted({p["traj_id"] for p in pos})
    models = [m for m in res if m != "B"]
    full_models = [m for m in models if sum(1 for b in res[m] if not b.endswith("|free")) > len(pos)]
    A = {"time": time.strftime("%Y-%m-%d %H:%M"), "n_pos": len(pos), "B_free_n": len(Bfree), "models": {}}
    L = [f"# A 层（ALFWorld）标签过期：结果（{A['time']} 生成；登记 `登记_ALFWorld过期_20261030.md` 的读法）", "",
         f"6.9 池：{len(trajs)} 条轨迹、{len(pos)} 个位置；B 的自由选择已有 {len(Bfree)} 个位置。区间 = 按轨迹重抽 {N_BOOT} 次的 90% 区间。", ""]
    # 漂移
    L += ["## 漂移（登记 5.4；与 B 的第一步动作分布的总变差距离，各 20 个样本）", "",
          "第一步相同的概率 = 两边各抽一个样本、动作相同的概率（独立样本）；B 与 B 自身（B#2）的值是参照。", "",
          "| 模型 | TV | 贪心变化率 | 第一步相同的概率 | 位置数 |", "|---|---:|---:|---:|---:|"]
    for m in sorted(dt, key=lambda k: (k != "B#2", k)):
        d = dt[m]
        name = "B#2（B 自身的噪声底）" if m == "B#2" else m
        L.append(f"| {name} | {d['tv']:.3f} | {d['greedy_change']:.3f} | {d['same_action_prob']:.3f} | {d['n_pos']} |")
        A["models"].setdefault(m, {})["drift"] = {k: v for k, v in d.items() if k != "per_pos"}
    L.append("")
    # 每个模型
    tk = {}
    tk_path = SA.STATE["rt_out"] / "eval_rollouts.jsonl" if SA.STATE["rt_out"] else None
    if tk_path and tk_path.exists():
        acc = defaultdict(list)
        for r in jl_read(tk_path):
            if "|tk|" in r["rid"]:
                acc[r["model"]].append(bool(r["won"]))
        tk = {m: mean(v) for m, v in acc.items()}
    r6 = []
    L += ["## 每个 π1 的读数", ""]
    for m in sorted(models):
        pp = per_position(D, ev, pos, res, m, Bfree)
        ks = list(pp)
        info = {}
        is_full = m in full_models
        cond = [k for k in ks if pp[k].get("G0") is not None and pp[k]["G0"] >= TH]

        def rate(field):
            def f(kss):
                kk = [k for k in kss[0] if pp[k].get("G0") is not None and pp[k]["G0"] >= TH and pp[k].get(field) is not None]
                return mean(pp[k][field] < TH for k in kk) if kk else float("nan")
            return f
        info["point_stale_Gt"] = boot([pp], rate("Gt1"), trajs)
        info["free_diff"] = boot([pp], lambda kss: nanmean(pp[k].get("free1", float("nan")) - pp[k].get("free0", float("nan"))
                                                           for k in kss[0] if "free1" in pp[k] and "free0" in pp[k]), trajs)
        L += [f"### {m}（{'全分支' if is_full else '两分支'}）", "",
              f"- 自由选择成功率 π1 − π0：{fmt(info['free_diff'])}",
              f"- 点过期率（用 G̃，条件 G_π0 ≥ {TH}；{len(cond)} 个位置）：{fmt(info['point_stale_Gt'])}"]
        if is_full:
            info["L"] = boot([pp], lambda kss: traj_mean(pp, [k for k in kss[0] if "L" in pp[k]], "L"), trajs)
            info["L_k0_3"] = boot([pp], lambda kss: traj_mean(pp, [k for k in kss[0] if "L" in pp[k]], "L_k0_3"), trajs)
            info["L_k0_10"] = boot([pp], lambda kss: traj_mean(pp, [k for k in kss[0] if "L" in pp[k]], "L_k0_10"), trajs)
            info["ctrl"] = boot([pp], lambda kss: traj_mean(pp, [k for k in kss[0] if "L" in pp[k]], "ctrl"), trajs)
            info["point_stale_G"] = boot([pp], rate("G1"), trajs)
            both = lambda kss: nanmean(pp[k]["L"] for k in kss[0] if "L" in pp[k] and pp[k].get("G0", -1) >= TH and pp[k]["G1"] >= TH)  # noqa: E731
            info["action_stale"] = boot([pp], both, trajs)
            info["agree"] = boot([pp], lambda kss: nanmean(float(pp[k]["agree"]) for k in kss[0] if "agree" in pp[k]), trajs)
            info["r5_near"] = boot([pp], lambda kss: nanmean(float(pp[k]["r5_near"]) for k in kss[0] if "r5_near" in pp[k]), trajs)
            info["Gt_minus_G"] = boot([pp], lambda kss: nanmean(pp[k]["Gt_minus_G"] for k in kss[0] if "Gt_minus_G" in pp[k]), trajs)
            r7 = lambda kss: traj_mean(pp, [k for k in kss[0] if "L" in pp[k] and not pp[k]["planner_b0"]], "L")  # noqa: E731
            info["L_r7"] = boot([pp], r7, trajs)
            n7 = sum(1 for k in ks if "L" in pp[k] and not pp[k]["planner_b0"])
            e, lo, hi = info["L"]
            verdict = ("还没有数据" if lo is None else "不过期（90% 区间在 ±0.05 内）" if lo >= -0.05 and hi <= 0.05
                       else "过期（下界 > 0）" if lo > 0 else "不确定")
            info["H1"] = verdict
            L += [f"- **过期损失 L**（π0 用前 5 次选；登记 5.1）：{fmt(info['L'])} → **H1：{verdict}**",
                  f"  - π0 用 3 次选：{fmt(info['L_k0_3'])}；用全部 10 次：{fmt(info['L_k0_10'])}",
                  f"  - 阳性对照（π1 自选 − 随机候选）：{fmt(info['ctrl'])}；刻度锚：第 4 篇同池 7B 自选 − 按 14B 选 = +0.255",
                  f"- 点过期率（用 G，全分支）：{fmt(info['point_stale_G'])}；G̃ − G 平均 {fmt(info['Gt_minus_G'])}",
                  f"- 动作过期（两边都值得修的位置上的 L）：{fmt(info['action_stale'])}",
                  f"- 两边最好修复相同的比例：{fmt(info['agree'])}；R5：π0 的最好修复在 π1 下是最好或差 < 0.10 的比例 {fmt(info['r5_near'])}",
                  f"- R7（π0 的最好修复不是规划器动作的 {n7} 个位置）的 L：{fmt(info['L_r7'])}"]
            # R4
            rows, lam = refresh(pp, dt.get(m, {}).get("per_pos", {}))
            if rows:
                L += ["", f"  **R4 刷新策略**（标签集 {len(lam)} 个位置；主读数 Q = 保留下来的标签在 π1 下的平均真增益，补充二）", "",
                      "  | k_c | 方式 | Q | 精确率 | 召回率 | 保留比例 | 总价值 V（描述） | 每位置续跑数 |",
                      "  |---:|---|---:|---:|---:|---:|---:|---:|"]
                info["R4"] = {}
                for kc in KC:
                    for meth in ("reuse", "point", "point_label", "full", "drift"):
                        s = refresh_summary(rows, lam, kc, meth)
                        info["R4"][f"{kc}_{meth}"] = s
                        L.append(f"  | {kc} | {meth} | {s['Q']:+.3f} | {s['prec']:.2f} | {s['recall']:.2f} | {s['kept_frac']:.2f} | "
                                 f"{s['V']:+.3f} | {s['cost']:.1f} |")

                def rec(kss):
                    ks2 = [k for k in kss[0] if k in rows]
                    if not ks2:
                        return float("nan")
                    vr = refresh_summary(rows, ks2, 5, "reuse")["Q"]
                    vp = refresh_summary(rows, ks2, 5, "point")["Q"]
                    vf = refresh_summary(rows, ks2, 5, "full")["Q"]
                    return (vp - vr) / (vf - vr) if abs(vf - vr) > 1e-9 else float("nan")

                def qdiff(a_, b_):
                    def f(kss):
                        ks2 = [k for k in kss[0] if k in rows]
                        return refresh_summary(rows, ks2, 5, a_)["Q"] - refresh_summary(rows, ks2, 5, b_)["Q"] if ks2 else float("nan")
                    return f
                ppl = {k: pp[k] for k in lam}
                info["R4_recovery"] = boot([ppl], rec, trajs)
                info["R4_Q_point_minus_reuse"] = boot([ppl], qdiff("point", "reuse"), trajs)
                info["R4_Q_full_minus_reuse"] = boot([ppl], qdiff("full", "reuse"), trajs)
                info["R4_Q_full_minus_point"] = boot([ppl], qdiff("full", "point"), trajs)
                L += [f"\n  - k_c = 5 的 Q 差：只查位置 − 复用 {fmt(info['R4_Q_point_minus_reuse'])}；全部重新验证 − 复用 "
                      f"{fmt(info['R4_Q_full_minus_reuse'])}；全部重新验证 − 只查位置 {fmt(info['R4_Q_full_minus_point'])}",
                      f"  - **恢复比**（Q；只查位置的提升 / 全部重新验证的提升；分母接近 0 时不稳定，以 Q 差为准）：{fmt(info['R4_recovery'])}"]
                for th2 in (0.10, 0.30):
                    rows2, lam2 = refresh(pp, dt.get(m, {}).get("per_pos", {}), th=th2)
                    if rows2:
                        vr = refresh_summary(rows2, lam2, 5, "reuse")["Q"]
                        vp = refresh_summary(rows2, lam2, 5, "point")["Q"]
                        vf = refresh_summary(rows2, lam2, 5, "full")["Q"]
                        L.append(f"  - 阈值 {th2}（敏感性，描述；Q）：复用 {vr:+.3f}、只查位置 {vp:+.3f}、全部重新验证 {vf:+.3f}")
            # 反转检查
            if m in rev:
                rp = SA.P(f"reverse_positions_{m}.json")
                if rp.exists():
                    sup = []
                    for d in json.loads(rp.read_text()):
                        a1, a0 = lst(rev[m].get(d["b_pi1"]), SA.K_REV), lst(rev[m].get(d["b_pi0"]), SA.K_REV)
                        if a1 and a0:
                            sup.append(1.0 if mean(a1) > mean(a0) else 0.5 if mean(a1) == mean(a0) else 0.0)
                    info["reverse_support_pi1"] = mean(sup) if sup else None
                    L.append(f"- 反转检查（{len(sup)} 个位置，各 30 次新续跑）：支持 π1 偏好的比例 {mean(sup):.2f}" if sup else "- 反转检查：还没有数据")
        L.append("")
        v6 = info["point_stale_Gt"][0]
        if m in tk and "B" in tk and v6 is not None and not math.isnan(v6):
            r6.append((m, tk[m] - tk["B"], info["point_stale_Gt"][0]))
        A["models"].setdefault(m, {}).update({k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()})
    # R6
    L += ["## R6 跨模型剂量（描述）：点过期率（G̃）对 P0 接管增益", ""]
    if len(r6) >= 3:
        L += ["| 模型 | P0 接管增益（相对 B） | 点过期率 |", "|---|---:|---:|"] + \
             [f"| {m} | {g:+.3f} | {s:.3f} |" for m, g, s in sorted(r6, key=lambda z: z[1])]
        rho = spearman([z[1] for z in r6], [z[2] for z in r6])
        A["R6_spearman"] = rho
        L.append(f"\nSpearman = {rho:+.2f}（{len(r6)} 个模型）")
    else:
        L.append("（模型不够，至少要 3 个有两分支或全分支数据的模型）")
    out = Path(a.out)
    (out / "report_stale.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    jdump(out / "analysis_stale.json", A)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
