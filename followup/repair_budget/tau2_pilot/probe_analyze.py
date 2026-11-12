"""τ² 探针的分析（2026-10-31 写；只用 CPU；定义见 登记_τ²探针_20261031.md 第 4–5 节）。
用法（tau2_pilot 下）：
  python probe_analyze.py --design probe/design_eval_ret.jsonl probe/design_eval_air.jsonl \
      --rollouts probe/ro_eval_pi0.jsonl probe/ro_eval_pi1.jsonl --pi0 4B --pi1 4B-S1 --drift probe/drift_eval.jsonl \
      --out probe/report_probe.md
输出：报告（markdown）与同名 .json。
11-05 加 --adm（主实验准备第 2 步，敏感性分析）：给出 adm_tau2.py annotate 的判定文件时，修复候选只用"可执行的"
（判定为 ok）；没有可执行修复候选的位置不进读数（报告里写出去掉了几个）。自由选择与续跑数据不变。
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

THR = 0.20
N_BOOT = 2000
K = 6                         # 每分支续跑次数（--k）；两半 = 前 K/2 次与后 K/2 次
H = [tuple(range(0, K // 2)), tuple(range(K // 2, K))]


def sseed(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_rates(paths, models):
    best = {}
    for p in paths:
        for r in jl(p):
            if r["executor"] not in models or r.get("score") != "orig_task" or r["temp"] == 0:
                continue
            key = (r["executor"], r["pos_id"], r["cand_id"], r["k"])
            if r.get("reward") is not None or key not in best:
                best[key] = r
    out = defaultdict(dict)
    for (m, pos, cand, k), r in best.items():
        if r.get("reward") is not None:
            out[(m, pos)].setdefault(cand, {})[k] = float(r["reward"])
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def p_on(rates, cand, ks):
    v = [rates.get(cand, {}).get(k) for k in ks]
    return mean(v)


def argbest(rates, cands, ks, pos):
    sc = [(p_on(rates, c, ks), sseed("TIE", pos, c), c) for c in cands]
    sc = [x for x in sc if x[0] is not None]
    return max(sc)[2] if sc else None


ADM = None                    # 11-05：{(pos_id, cand_id): ok}；None = 不过滤


def position_record(d, R, pi0, pi1):
    pos = d["pos_id"]
    reps = [c["cand_id"] for c in d["cands"] if c.get("restore_ok") and "free" not in c["kinds"] and c["kinds"] != ["orig"]]
    if ADM is not None:
        miss = [c for c in reps if (pos, c) not in ADM]
        if miss:
            raise SystemExit(f"[停下] --adm 里没有 {pos} 的候选 {miss} 的判定")
        reps = [c for c in reps if ADM[(pos, c)]]
    r0, r1 = R.get((pi0, pos), {}), R.get((pi1, pos), {})
    if not reps or "free" not in r0 or "free" not in r1:
        return None
    allk = tuple(range(K))
    rec = {"pos": pos, "cluster": f"{d['domain']}:{d['task_id']}", "domain": d["domain"], "n_rep": len(reps)}
    for name, rr in [("pi0", r0), ("pi1", r1)]:
        g = []
        for s, e in [(H[0], H[1]), (H[1], H[0])]:
            b = argbest(rr, reps, s, pos)
            pb, pf = p_on(rr, b, e), p_on(rr, "free", e)
            if pb is not None and pf is not None:
                g.append(pb - pf)
        rec[f"G_{name}"] = mean(g)
        rec[f"free_{name}"] = p_on(rr, "free", allk)
        rec[f"best_{name}"] = argbest(rr, reps, allk, pos)
        rec[f"bestH_{name}"] = [argbest(rr, reps, H[0], pos), argbest(rr, reps, H[1], pos)]
    # 过期损失 L：π1 用一半选、π0 用全部 K 次选，都在 π1 的另一半上评价；交换两半取平均
    Ls = []
    for s, e in [(H[0], H[1]), (H[1], H[0])]:
        b1 = argbest(r1, reps, s, pos)
        a, bb = p_on(r1, b1, e), p_on(r1, rec["best_pi0"], e)
        if a is not None and bb is not None:
            Ls.append(a - bb)
    rec["L"] = mean(Ls)
    # 刷新（描述；k_c = 2：一半做决定、另一半评价，交换取平均）
    a0 = rec["best_pi0"]
    p0 = p_on(r0, a0, allk)
    refresh = {"reuse": [], "point": [], "full": []}
    for s, e in [(H[0], H[1]), (H[1], H[0])]:
        fE = p_on(r1, "free", e)
        val = lambda act: (p_on(r1, act, e) - fE) if p_on(r1, act, e) is not None and fE is not None else None
        refresh["reuse"].append((True, val(a0)))
        refresh["point"].append(((p0 or 0) - (p_on(r1, "free", s) or 0) >= THR - 1e-9, val(a0)))
        b1 = argbest(r1, reps, s, pos)
        keep = b1 is not None and (p_on(r1, b1, s) or 0) - (p_on(r1, "free", s) or 0) >= THR - 1e-9
        refresh["full"].append((keep, val(b1)))
    rec["refresh"] = refresh
    return rec


def stats(recs):
    w0 = [r for r in recs if r["G_pi0"] is not None and r["G_pi0"] >= THR - 1e-9]
    both = [r for r in w0 if r["G_pi1"] is not None and r["G_pi1"] >= THR - 1e-9]
    out = {"n_pos": len(recs), "n_worth_pi0": len(w0), "n_both": len(both),
           "free_pi0": mean([r["free_pi0"] for r in recs]), "free_pi1": mean([r["free_pi1"] for r in recs]),
           "G_pi0": mean([r["G_pi0"] for r in recs]), "G_pi1": mean([r["G_pi1"] for r in recs]),
           "point_stale": (len(w0) - len(both)) / len(w0) if w0 else None,
           "L_both": mean([r["L"] for r in both]), "L_all": mean([r["L"] for r in recs]),
           "best_changed_both": mean([float(r["best_pi1"] != r["best_pi0"]) for r in both]),
           "split_half_pi1_both": mean([float(r["bestH_pi1"][0] != r["bestH_pi1"][1]) for r in both]),
           "split_half_pi0_both": mean([float(r["bestH_pi0"][0] != r["bestH_pi0"][1]) for r in both])}
    out["free_diff"] = None if out["free_pi0"] is None else out["free_pi1"] - out["free_pi0"]
    for k in ("reuse", "point", "full"):            # 刷新：保留下来的标签在 π1 下的平均真实收益 Q 与保留比例（只在 π0 的标签集 w0 上）
        kept = [v for r in w0 for keep, v in r["refresh"][k] if keep and v is not None]
        tot = sum(len(r["refresh"][k]) for r in w0)
        out[f"Q_{k}"] = mean(kept)
        out[f"keep_{k}"] = len(kept) / tot if tot else None
    # 11-05（敏感性分析的判定标准）：两个 Q 差、值得修的比例
    out["dQ_point_reuse"] = None if out["Q_point"] is None or out["Q_reuse"] is None else out["Q_point"] - out["Q_reuse"]
    out["dQ_full_point"] = None if out["Q_full"] is None or out["Q_point"] is None else out["Q_full"] - out["Q_point"]
    out["frac_worth_pi0"] = len(w0) / len(recs) if recs else None
    out["frac_both"] = len(both) / len(recs) if recs else None
    return out


def boot(recs, keys, seed=0):
    cl = sorted({r["cluster"] for r in recs})
    by = defaultdict(list)
    for r in recs:
        by[r["cluster"]].append(r)
    rng = random.Random(seed)
    vals = defaultdict(list)
    for _ in range(N_BOOT):
        smp = [r for c in (rng.choice(cl) for _ in cl) for r in by[c]]
        s = stats(smp)
        for k in keys:
            if s[k] is not None:
                vals[k].append(s[k])
    ci = {}
    for k in keys:
        v = sorted(vals[k])
        ci[k] = (v[int(0.05 * len(v))], v[int(0.95 * len(v)) - 1]) if len(v) > 20 else (None, None)
        if len(v) > 20:                              # 11-05：重抽的标准差（作标准误）
            m = sum(v) / len(v)
            ci[k + "_se"] = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    return ci


def drift_stats(path, pi0, pi1):
    D = defaultdict(lambda: defaultdict(list))
    for r in jl(path):
        D[(r["model"], r["rep"])][r["pos_id"]].append(r["canon"])

    def tv(a, b):
        ca, cb = Counter(a), Counter(b)
        return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))
    base = D.get((pi0, "1"), {})
    out = {}
    for key, name in [((pi0, "2"), "noise_pi0#2"), ((pi1, "1"), "pi1")]:
        other = D.get(key, {})
        ps = [p for p in base if p in other and base[p] and other[p]]
        if ps:
            out[name] = {"n_pos": len(ps), "tv": mean([tv(base[p], other[p]) for p in ps]),
                         "mode_changed": mean([float(Counter(base[p]).most_common(1)[0][0] != Counter(other[p]).most_common(1)[0][0]) for p in ps]),
                         "invalid": mean([float(x == "INVALID") for p in ps for x in other[p]]),
                         "tool_call": mean([float(x.startswith("T:")) for p in ps for x in other[p]])}
    return out


def f(x, nd=3):
    return "—" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) and x < 0 else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", nargs="+", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--pi0", default="4B")
    ap.add_argument("--pi1", required=True)
    ap.add_argument("--drift", default=None)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--adm", nargs="*", default=None, help="11-05：adm_tau2.py annotate 的判定文件；只用可执行的修复候选")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global K, H, ADM
    if a.adm:
        ADM = {}
        for p in a.adm:
            for r in jl(p):
                ADM[(r["pos_id"], r["cand_id"])] = bool(r["ok"])
    K = a.k
    H = [tuple(range(0, K // 2)), tuple(range(K // 2, K))]
    design = [d for p in a.design for d in jl(p)]
    R = load_rates(a.rollouts, {a.pi0, a.pi1})
    recs = [r for r in (position_record(d, R, a.pi0, a.pi1) for d in design) if r is not None]
    keys = ["free_diff", "point_stale", "L_both", "L_all", "Q_reuse", "Q_point", "Q_full", "G_pi0", "G_pi1",
            "dQ_point_reuse", "dQ_full_point", "best_changed_both", "frac_worth_pi0", "frac_both"]
    res = {"all": stats(recs), "ci_all": boot(recs, keys)}
    for dom in sorted({r["domain"] for r in recs}):
        sub = [r for r in recs if r["domain"] == dom]
        res[dom] = stats(sub)
    dr = drift_stats(a.drift, a.pi0, a.pi1) if a.drift else {}
    res["drift"] = dr
    s, ci = res["all"], res["ci_all"]
    # 判定（登记第 5 节）
    tv1, tv0 = (dr.get("pi1") or {}).get("tv"), (dr.get("noise_pi0#2") or {}).get("tv")
    changed = ((tv1 is not None and tv0 is not None and tv1 - tv0 >= 0.10)
               or (s["free_diff"] is not None and abs(s["free_diff"]) >= 0.05)
               or (ci["L_both"][0] is not None and ci["L_both"][0] > 0))
    if not changed:
        verdict = "D0 不通过：π1 相对 π0 几乎没变（漂移与自由选择都在阈值内）→ 先加大训练剂量，动作过期的读数不解释"
    elif s["n_both"] < 10:
        verdict = f"D1 不足：两边都值得修的位置只有 {s['n_both']} 个（< 10），动作过期测不出 → 主实验按完整版设计，但先只做 4B 一条链"
    elif s["L_both"] >= 0.10:
        verdict = "D1 明显：L ≥ 0.10 → 做完整的第 6 篇 τ² 主实验（2B、4B 两条链）"
    elif s["L_both"] <= 0.03 and ci["L_both"][1] is not None and ci["L_both"][1] <= 0.10:
        verdict = "D1 约等于 0：L ≤ 0.03 且 90% 上界 ≤ 0.10 → 缩小版主实验，重心放在刷新成本曲线"
    else:
        verdict = "D1 不确定：介于两者之间 → 交用户决定（默认：先做 4B 一条链，看结果再定 2B）"
    res["verdict"] = verdict
    dom_str = "、".join(f"{d} {res[d]['n_pos']}" for d in sorted({r["domain"] for r in recs}))
    if ADM is not None:
        res["adm"] = {"files": a.adm, "n_design": len(design), "n_with_record": len(recs)}
        dom_str += f"；只用可执行的修复候选（{a.adm}），设计文件 {len(design)} 个位置中 {len(recs)} 个有可执行修复候选与读数"
    L = [f"# τ² 探针报告（probe_analyze.py；π0 = {a.pi0}，π1 = {a.pi1}；探索性）", "",
         f"位置 {s['n_pos']} 个（{dom_str}）；"
         f"区间：按任务成簇重抽 {N_BOOT} 次，90%。", "",
         "| 读数 | 全部 | 90% 区间 | " + " | ".join(sorted({r['domain'] for r in recs})) + " |",
         "|---|---:|---|" + "---:|" * len({r['domain'] for r in recs})]
    doms = sorted({r["domain"] for r in recs})
    rows = [("自由选择成功率 π0", "free_pi0"), ("自由选择成功率 π1", "free_pi1"), ("差 π1 − π0", "free_diff"),
            ("平均点价值 G（π0）", "G_pi0"), ("平均点价值 G（π1）", "G_pi1"), ("π0 认为值得修的位置数", "n_worth_pi0"),
            ("两边都值得修的位置数", "n_both"), ("**点过期率**", "point_stale"), ("**动作过期损失 L（两边都值得修）**", "L_both"),
            ("L（全部位置，描述）", "L_all"), ("最好候选换了的比例（两边都值得修）", "best_changed_both"),
            ("  对照：π1 两半各选的不一致率", "split_half_pi1_both"), ("  对照：π0 两半各选的不一致率", "split_half_pi0_both"),
            ("刷新 Q：复用", "Q_reuse"), ("刷新 Q：只查位置", "Q_point"), ("刷新 Q：全部重新验证", "Q_full"),
            ("保留比例：只查位置", "keep_point"), ("保留比例：全部重新验证", "keep_full"),
            ("Q 差：只查位置 − 复用", "dQ_point_reuse"), ("Q 差：全部重新验证 − 只查位置", "dQ_full_point"),
            ("π0 值得修的比例", "frac_worth_pi0"), ("两边都值得修的比例", "frac_both")]
    for name, k in rows:
        c = ci.get(k, (None, None))
        L.append(f"| {name} | {f(s[k])} | " + (f"[{f(c[0])}, {f(c[1])}]" if c[0] is not None else "") + " | "
                 + " | ".join(f(res[d][k]) for d in doms) + " |")
    L += ["", "漂移（第一步动作分布，相对 π0 的第一次抽样）：", ""]
    for k, v in dr.items():
        L.append(f"- {k}：位置 {v['n_pos']}，TV {f(v['tv'])}，众数动作变化率 {f(v['mode_changed'])}，无效输出 {f(v['invalid'])}，工具调用比例 {f(v['tool_call'])}")
    L += ["", f"**判定**：{verdict}", ""]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(str(a.out).rsplit(".", 1)[0] + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
