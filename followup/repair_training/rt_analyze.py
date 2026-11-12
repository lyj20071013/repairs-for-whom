"""先导 P0 的分析（论文方案.md 5.7 统计、5.8 判定）。只读结果文件，写 analysis.json 与 report.md。

    python rt.py --stage analyze

- 每组先在种子之间平均每局（每个接管点）的结果，再与另一组配对相减。
- 区间：重抽评测局（端到端）/ 接管轨迹（接管；同一轨迹的 3 个点一起抽）2,000 次的百分位区间；90% 下界 = 第 5 百分位。
  先导的区间把种子当作固定（只反映评测局的抽样）；种子 SD 单独报告。
- 判定（5.8）：通过 = S − B 端到端 ≥ +0.03 且下界 > 0，并且 S − R 接管 ≥ +0.10 且下界 > 0。
- 主实验功效（5.8）：δ = Γ_接管 × 0.234；种子 SD 取 S 三个种子的极差，组间相关取估计值与 0.3 中较小者；
  P = 1,000、m = 4；接管上的组 × 点交互 SD τ 取估计值与 0.2 中较大者（5.11 实现补充）。
"""
import math
import time
from collections import defaultdict
from statistics import NormalDist

import numpy as np

import rt_common as C
from rt_common import CFG, P, jdump, jl_read, jload, log

REF_SX = 0.234            # 论文方案.md 3.2：前 50% 规则下 S − X 的目标质量差（k = 3）
GROUPS = ("S", "U", "R", "DPO-S")


def group_of(model):
    return "B" if model == "B" else model.rsplit("_s", 1)[0]


# ---------------------------------------------------------------- 数据整理
def tables(rows):
    """-> e2e[model][game] = 0/1；tk[model][pid] = m 次的均值；tk_traj[pid] = traj_id；src[game]。"""
    e2e, tkr, tk_traj, src = defaultdict(dict), defaultdict(lambda: defaultdict(list)), {}, {}
    for r in rows:
        if "|e2e|" in r["rid"]:
            e2e[r["model"]][r["gamefile"]] = float(r["won"])
            src[r["gamefile"]] = r.get("src")
        elif "|tk|" in r["rid"]:
            tkr[r["model"]][r["pid"]].append(float(r["won"]))
            tk_traj[r["pid"]] = r["traj_id"]
    tk = {m: {p: float(np.mean(v)) for p, v in d.items()} for m, d in tkr.items()}
    return dict(e2e), tk, tk_traj, src


def group_values(tab, models, keys):
    """组 -> 每个键（局 / 点）上种子平均的值（数组，顺序同 keys）。只用该组全部种子都有结果的键。"""
    by = defaultdict(list)
    for m in models:
        by[group_of(m)].append(m)
    out = {}
    for g, ms in by.items():
        out[g] = np.array([np.mean([tab[m][k] for m in ms]) for k in keys])
    return out, dict(by)


def boot_diff(a, b, clusters, n_boot, seed):
    """配对差 mean(a − b)，按 clusters 整群重抽。返回 (估计, 第 5 百分位, 第 95 百分位, 单侧 p)。"""
    d = a - b
    cl = np.asarray(clusters)
    uniq, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    reps = sums[idx].sum(1) / cnts[idx].sum(1)
    est = float(d.mean())
    p = (np.sum(reps <= 0) + 1) / (n_boot + 1)
    return est, float(np.percentile(reps, 5)), float(np.percentile(reps, 95)), float(p)


# ---------------------------------------------------------------- 核心（selftest 用合成数据检验已知答案）
CONTRASTS = [("S", "B"), ("S", "R"), ("S", "U"), ("U", "B"), ("R", "B"), ("DPO-S", "B"), ("DPO-S", "S"),
             # 5.12 增补：锚样本对照与"挑点 / 排序"
             ("S", "A"), ("A", "B"), ("U", "A"), ("R", "A"), ("S-plan", "A"), ("S-plan", "U"), ("S", "S-plan")]


def analyze_core(rows, n_boot=None, seed=None):
    n_boot = n_boot or CFG["N_BOOT"]
    seed = CFG["SEED_BOOT"] if seed is None else seed
    e2e, tk, tk_traj, src = tables(rows)
    models = sorted(set(e2e) | set(tk), key=lambda m: (m != "B", m))
    games = sorted(set.intersection(*[set(e2e[m]) for m in models])) if e2e else []
    pts = sorted(set.intersection(*[set(tk[m]) for m in models])) if tk else []
    ge, by = group_values(e2e, models, games)
    gt, _ = group_values(tk, models, pts)
    res = {"models": models, "groups": by, "n_games": len(games), "n_points": len(pts),
           "n_traj": len({tk_traj[p] for p in pts}),
           "level": {"e2e": {g: float(v.mean()) for g, v in ge.items()}, "tk": {g: float(v.mean()) for g, v in gt.items()}},
           "model_level": {"e2e": {m: float(np.mean([e2e[m][k] for k in games])) for m in models},
                           "tk": {m: float(np.mean([tk[m][p] for p in pts])) for m in models}},
           "diff": {}}
    for a, b in CONTRASTS:
        if a in ge and b in ge:
            res["diff"][f"{a}-{b}"] = {
                "e2e": boot_diff(ge[a], ge[b], games, n_boot, seed),
                "tk": boot_diff(gt[a], gt[b], [tk_traj[p] for p in pts], n_boot, seed + 1)}
    # 种子：S 的组均值 SD 与极差；同种子组间相关（只有 2 个种子，很粗）
    seeds = {}
    for rd in ("e2e", "tk"):
        ml = res["model_level"][rd]
        s_vals = [ml[m] for m in by.get("S", [])]
        corr = {}
        for g1, g2 in (("S", "U"), ("S", "R"), ("U", "R")):
            com = sorted({m.rsplit('_s', 1)[1] for m in by.get(g1, [])} & {m.rsplit('_s', 1)[1] for m in by.get(g2, [])})
            if len(com) == 2:
                x = [ml[f"{g1}_s{s}"] for s in com]
                y = [ml[f"{g2}_s{s}"] for s in com]
                vx, vy = (x[0] - x[1]) ** 2 / 2, (y[0] - y[1]) ** 2 / 2
                vd = ((x[0] - y[0]) - (x[1] - y[1])) ** 2 / 2
                corr[f"{g1},{g2}"] = None if vx + vy == 0 else 1 - vd / (vx + vy)
        cv = [v for v in corr.values() if v is not None]
        seeds[rd] = {"S_values": s_vals, "S_sd": float(np.std(s_vals, ddof=1)) if len(s_vals) > 1 else None,
                     "S_range": float(max(s_vals) - min(s_vals)) if s_vals else None, "corr": corr,
                     "corr_mean": float(np.mean(cv)) if cv else None}
    res["seeds"] = seeds
    # 接管上的组 × 点交互 SD（S − B 的逐点差，扣掉二项抽样方差）
    if "S" in gt and "B" in gt and pts:
        m, nS = CFG["TK_M"], len(by["S"])
        d = gt["S"] - gt["B"]

        def binv(vals):
            v = np.clip(vals, 0, 1)
            return v * (1 - v) * m / max(m - 1, 1) / m
        samp = np.mean([np.mean(binv(np.array([tk[mm][p] for mm in by["S"]]))) / nS + float(binv(np.array([tk["B"][p]]))[0])
                        for p in pts])
        res["tau_hat"] = float(math.sqrt(max(0.0, np.var(d, ddof=1) - samp))) if len(pts) > 1 else None
        res["p_tk"] = float(np.mean(list(res["level"]["tk"].values())))
    return res


def decide(res):
    d = res["diff"]
    if "S-B" not in d or "S-R" not in d:
        return {"verdict": "不适用（缺 S、B 或 R 组；冒烟只检查流程）", "pass_e2e": None, "pass_tk": None,
                "S-B_e2e": d.get("S-B", {}).get("e2e"), "S-R_tk": d.get("S-R", {}).get("tk")}
    se, sr = d["S-B"]["e2e"], d["S-R"]["tk"]
    pe = se[0] >= 0.03 and se[1] > 0
    pt = sr[0] >= 0.10 and sr[1] > 0
    # 5.13：判定是"是否达到主实验的预设推进条件"（成本门槛），不是关于"修复 SFT 有没有效果"的科学结论；
    # 不通过时附上分解（相对 A、相对 U），供另行登记时参考
    if pe and pt:
        v = "达到主实验的预设推进条件（两项都满足；还要看功效与成本）"
    elif pt:
        v = "只达到接管一项：主实验可推进，端到端只作描述，论文不声称部署收益（还要看功效与成本）"
    else:
        v = "未达到主实验的预设推进条件（S − R 接管 < +0.10 或下界 ≤ 0）。这不等于修复 SFT 没有效果：见下面相对 A、U 的分解；后续另行登记"
    brk = {k: d[k]["tk"] for k in ("S-A", "R-A", "U-A", "S-U", "S-plan-U") if k in d}
    return {"S-B_e2e": se, "S-R_tk": sr, "pass_e2e": pe, "pass_tk": pt, "verdict": v, "breakdown_tk": brk}


def power(delta, sig, rho, S, P_=1000, m=4, p=0.25, tau=0.2):
    se = math.sqrt(2 * sig ** 2 * (1 - rho) / S + tau ** 2 / P_ + 2 * p * (1 - p) / (P_ * m * S))
    return NormalDist().cdf(delta / se - 1.96), se


def main_plan(res, qbar_S, timing, D, n_branches):
    """Γ、主实验的预期 S − X、所需种子数与卡时（粗估，只用于 5.8 的决定）。

    5.12：Γ 以 A（只用锚样本训练）为基线：Γ = (S − A) / q̄_S；没有 A 组时退回 (S − B) / q̄_S（5.7 原定义），两者都报告。"""
    out = {}
    base = "S-A" if "S-A" in res["diff"] else "S-B"
    sb = res["diff"][base]
    out["Gamma_basis"] = base
    out["qbar_S"] = qbar_S
    out["Gamma_tk"] = sb["tk"][0] / qbar_S if qbar_S else None
    out["Gamma_e2e"] = sb["e2e"][0] / qbar_S if qbar_S else None
    out["Gamma_tk_SB"] = res["diff"]["S-B"]["tk"][0] / qbar_S if qbar_S else None
    st = res["seeds"]["tk"]
    sig = st["S_range"] if st["S_range"] is not None else 0.03
    rho = min(st["corr_mean"], 0.3) if st["corr_mean"] is not None else 0.0
    tau = max(res.get("tau_hat") or 0.0, 0.2)
    p = res.get("p_tk", 0.25)
    out.update(sigma_used=sig, rho_used=rho, tau_used=tau, p_used=p)
    delta = (out["Gamma_tk"] or 0) * REF_SX
    out["delta_SX"] = delta
    rows = []
    for S in (2, 3, 4, 5):
        pw, se = power(delta, sig, rho, S, p=p, tau=tau)
        rows.append({"seeds": S, "power": pw, "se": se})
    out["power_table"] = rows
    need = next((r["seeds"] for r in rows if r["power"] >= 0.8), None)
    out["seeds_needed"] = need
    # 卡时粗估：从本次计时外推（7B 标签每决策点、每次训练、每个模型的评测）
    def hours(prefix):
        return sum(r["wall_s"] for r in timing.get("runs", []) if r["stage"].startswith(prefix)) / 3600
    n_train = max(1, sum(1 for r in timing.get("runs", []) if r["stage"].startswith("train_")))
    n_eval = max(1, len({r["stage"] for r in timing.get("runs", []) if r["stage"].startswith("eval_")}))
    h_label7 = hours("label")
    h_train = hours("train_") / n_train
    h_eval = hours("eval_") / n_eval
    tk_now = CFG["TK_N_TRAJ"] * 3 * CFG["TK_M"]
    scale_eval = (1000 * 4 + 850) / max(1, tk_now + 850)
    S_ = need or 5
    est = {"label7_new_half": h_label7, "label14_all": h_label7 * 2 * (2.0 / 1.02),
           "greedy7_all": h_label7 * 2 * (1.22 / 1.02) / CFG["K_LABEL"],
           "train": 6 * S_ * h_train, "eval": (6 * S_ + 1) * h_eval * scale_eval}      # 主实验：A、S、S-plan、U、X、G + B
    est["total"] = sum(est.values())
    out["main_cost_gpu_h"] = est
    out["main_cost_yuan"] = est["total"] * CFG["PRICE_PER_GPU_HOUR"]
    out["go"] = bool(need is not None and est["total"] <= 150)
    return out


# ---------------------------------------------------------------- 阶段
def qbar_fresh():
    d = jload(P("design.json"))
    ts = jload(P("trainsets.json"))
    res = defaultdict(dict)
    for r in jl_read(P("fresh_q7.jsonl")):
        res[r["bid"]][r["k"]] = float(r["won"])
    orig = {p["pid"]: p["orig"] for p in d["points"]}

    def v(b):
        if d["branches"][b]["deterministic"] is not None:
            return 1.0
        x = res.get(b, {})
        return float(np.mean(list(x.values()))) if len(x) == CFG["N_FRESH"] else None
    out = {}
    for g in ("S", "S-plan"):
        vals = []
        for x in ts["groups"].get(g, []):
            a, o = v(x["bid"]), v(orig[x["pid"]])
            if a is None or o is None:
                vals = None
                break
            vals.append(a - o)
        out[g] = float(np.mean(vals)) if vals else None
    return out


def stage_analyze():
    rows = jl_read(P("eval_rollouts.jsonl"))
    res = analyze_core(rows)
    res["decision"] = decide(res)
    ts = jload(P("trainsets.json"))
    timing = jload(P("timing.json")) if P("timing.json").exists() else {"runs": []}
    qf = qbar_fresh()
    qS = qf["S"]
    res["qbar"] = {"S_fresh": qS, "S-plan_fresh": qf["S-plan"], **{k: v for k, v in ts["stats"]["q_label"].items()}}
    res["main"] = main_plan(res, qS, timing, ts["stats"]["D"], None) if "S-B" in res["diff"] else None
    # 线性检查（探索性）：各组（相对基线）的接管增益 vs 目标质量 q̄；基线 = A（有 A 组时）否则 B
    basis = "A" if "A" in res["level"]["tk"] else "B"
    qmap = {"S": qS, "S-plan": qf["S-plan"], "U": ts["stats"]["q_label"].get("U"), "R": ts["stats"]["q_label"].get("R")}
    pts = [(q, res["level"]["tk"][g] - res["level"]["tk"][basis]) for g, q in qmap.items()
           if q is not None and g in res["level"]["tk"]]
    lin = {"basis": basis, "points": {g: (q, round(res["level"]["tk"][g] - res["level"]["tk"][basis], 4))
                                       for g, q in qmap.items() if q is not None and g in res["level"]["tk"]}}
    if len(pts) >= 3:
        x, y = np.array(pts).T
        slope, icpt = np.polyfit(x, y, 1)
        lin.update(slope=float(slope), intercept=float(icpt))
    res["linearity_Gamma"] = lin
    res["factcheck"] = jload(P("factcheck.json")) if P("factcheck.json").exists() else None
    th = jl_read(P("thoughts.jsonl"))
    res["thoughts"] = {"n": len(th), "template": sum(r["template"] for r in th),
                       "first_try_ok": sum(r["n_tries"] == 1 and not r["template"] for r in th)}
    res["format"] = jload(P("formatcheck.json")) if P("formatcheck.json").exists() else None
    tr = {}
    for g, s in C.wanted_runs():
        f = C.adapter_dir(g, s) / "done.json"
        if f.exists():
            tr[C.model_name(g, s)] = {k: v for k, v in jload(f).items() if k in ("n_used", "n_dropped", "wall_s")}
    res["train"] = tr
    gpu_h = sum(r["wall_s"] for r in timing["runs"] if r.get("gpu_stage")) / 3600
    res["cost"] = {"gpu_h": gpu_h, "yuan": gpu_h * CFG["PRICE_PER_GPU_HOUR"]}
    res["time"] = time.strftime("%Y-%m-%d %H:%M")
    jdump(P("analysis.json"), res)
    P("report.md").write_text(report(res, ts), encoding="utf-8")
    log(f"[analyze] {res['decision']['verdict']} -> {P('report.md')}")
    return 0


def _f(x, nd=3):
    return "—" if x is None else f"{x:+.{nd}f}"


def report(res, ts):
    L = [f"# 先导 P0 结果（{res['time']} 生成；论文方案.md 第 5 节登记的读法）", ""]
    L += [f"评测局 {res['n_games']}；接管点 {res['n_points']}（{res['n_traj']} 条轨迹，每点 m = {CFG['TK_M']}）；"
          f"训练集 D = {ts['stats']['D']} 个决策点，N = {ts['stats']['N']}。", ""]
    L += ["## 各组水平（种子平均）", "", "| 组 | 模型 | 端到端 | 接管 |", "|---|---|---:|---:|"]
    for g, ms in res["groups"].items():
        L.append(f"| {g} | {', '.join(ms)} | {res['level']['e2e'][g]:.3f} | {res['level']['tk'][g]:.3f} |")
    L += ["", "## 配对差（90% 区间；先导把种子当作固定）", "",
          "| 比较 | 端到端 | 区间 | 接管 | 区间 |", "|---|---:|---|---:|---|"]
    for k, v in res["diff"].items():
        e, t = v["e2e"], v["tk"]
        L.append(f"| {k} | {e[0]:+.3f} | [{e[1]:+.3f}, {e[2]:+.3f}] | {t[0]:+.3f} | [{t[1]:+.3f}, {t[2]:+.3f}] |")
    dc = res["decision"]
    L += ["", "## 判定（5.8）", ""]
    if dc["pass_tk"] is not None:
        L += [f"- S − B 端到端 {dc['S-B_e2e'][0]:+.3f}（下界 {dc['S-B_e2e'][1]:+.3f}）：{'满足' if dc['pass_e2e'] else '不满足'} ≥ +0.03 且下界 > 0",
              f"- S − R 接管 {dc['S-R_tk'][0]:+.3f}（下界 {dc['S-R_tk'][1]:+.3f}）：{'满足' if dc['pass_tk'] else '不满足'} ≥ +0.10 且下界 > 0"]
    L += [f"- **{dc['verdict']}**"]
    if dc.get("breakdown_tk"):
        L.append("- 分解（接管，90% 区间）：" + "；".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
                                                    for k, v in dc["breakdown_tk"].items()))
    L.append("")
    mn = res["main"]
    if mn is None:
        return "\n".join(L + ["（缺 S 或 B 组，不计算主实验的校准量）"])
    L += ["## 校准主实验的量", "",
          f"- q̄_S（新鲜续跑）{_f(res['qbar']['S_fresh'])}；标签估计：S（有偏）{_f(res['qbar'].get('S_in_sample'))}、"
          f"U {_f(res['qbar'].get('U'))}、R {_f(res['qbar'].get('R'))}",
          f"- q̄_S-plan（新鲜续跑）{_f(res['qbar'].get('S-plan_fresh'))}",
          f"- Γ_接管 = ({mn['Gamma_basis'].replace('-', ' − ')}) / q̄_S = {_f(mn['Gamma_tk'])}（以 B 为基线：{_f(mn['Gamma_tk_SB'])}）；"
          f"Γ_端到端 = {_f(mn['Gamma_e2e'])}",
          f"- 线性检查（探索性；相对 {res['linearity_Gamma']['basis']} 的接管增益 vs q̄）：{res['linearity_Gamma'].get('points')}；"
          f"斜率 {_f(res['linearity_Gamma'].get('slope'))}、截距 {_f(res['linearity_Gamma'].get('intercept'))}",
          f"- 种子（接管）：S 各种子 {[round(x, 3) for x in res['seeds']['tk']['S_values']]}，SD {_f(res['seeds']['tk']['S_sd'])}，"
          f"极差 {_f(res['seeds']['tk']['S_range'])}；同种子组间相关 {res['seeds']['tk']['corr']}",
          f"- τ̂（接管上的组 × 点交互 SD）{_f(res.get('tau_hat'))}",
          f"- 主实验预期 S − X = Γ × {REF_SX} = {_f(mn['delta_SX'])}；功效（σ = {mn['sigma_used']:.3f}、ρ = {mn['rho_used']:.2f}、"
          f"τ = {mn['tau_used']:.2f}、P = 1,000、m = 4）："
          + "，".join(f"{r['seeds']} 种子 {r['power']:.2f}" for r in mn["power_table"]),
          f"- 主实验卡时粗估 {mn['main_cost_gpu_h']['total']:.0f}（≈ ¥{mn['main_cost_yuan']:.0f}）："
          + "，".join(f"{k} {v:.1f}" for k, v in mn["main_cost_gpu_h"].items() if k != "total"),
          f"- **按 5.8：{'做主实验' if mn['go'] else '不做主实验（先改训练量或读数，另行登记）'}**", ""]
    L += ["## 描述", "",
          f"- Thought 改写：{res['thoughts']}",
          f"- 格式检查：{ {m: round(v, 3) for m, v in (res['format'] or {}).get('rate_first_bad', {}).items()} }",
          "- Thought 事实核查（5.12；无依据断言的比例）：" + (
              f"（Thought 方式 {res['factcheck']['by_group'].get('thought_mode', 'rewrite')}）"
              + "，".join(f"{g} {v['unsupported']:.2f}" for g, v in res["factcheck"]["by_group"].items()
                         if isinstance(v, dict) and v["unsupported"] is not None)
              + (f"；**比 S 高出 > {res['factcheck']['threshold']:.2f} 的组：{res['factcheck']['flag_vs_S']}"
                 "（涉及这些组的比较可能受 Thought 质量影响）**" if res["factcheck"]["flag_vs_S"] else "；没有组超过阈值")
              if res.get("factcheck") else "（没有 factcheck.json）"),
          f"- 训练：{res['train']}",
          f"- 本次先导 GPU 用时 {res['cost']['gpu_h']:.1f} 卡时 ≈ ¥{res['cost']['yuan']:.0f}（¥{CFG['PRICE_PER_GPU_HOUR']}/卡时）", ""]
    return "\n".join(L)
