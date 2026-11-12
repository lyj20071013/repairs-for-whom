"""A 层补充分析（2026-11-06 写；事后、探索性）：不用续跑，能不能预测哪些标签点过期？

预测量（都只用 drift.jsonl：每个位置上每个模型 20 次首动作采样，不需要续跑）：
  p1_a0   = π1 的首动作里等于 a0 的比例（主预测量；假设：π1 自己就会选 a0 → 这个位置不再需要修）
  dp_a0   = p1_a0 − p0_a0（π0 = 基础 7B "B"）
  tv      = π0 与 π1 首动作分布的总变差
  p0_a0   = π0 自己选 a0 的比例（对照：与训练无关的基线）
被预测的量：点过期 = G_π1 < 0.20（S_s11、S_s12 用全分支的 G；另 5 个模型只有两分支，用 G̃ = p̂(a0) − p̂(free)），
条件：G_π0 ≥ 0.20（交叉拟合），与 stale_analyze.py / planner_split.py 相同。
读数：AUC（按轨迹重抽 2,000 次的 90% 区间）；以及"按预测保留标签"的刷新方式（事先定的阈值：p1_a0 < 0.5 才保留），
与只查位置比 V、Q、保留比例（k_c = 5，与 planner_split.py 同定义）。
写这段说明时还没有看过任何预测量与点过期的关系。"""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
import stale_alfw as SA
import stale_analyze as AN
from rt_common import jl_read

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
SA.setup("real", str(OUT))
D, ev, pos = SA.load_design()
SA.DESIGN[0] = D
res = defaultdict(lambda: defaultdict(dict))
for stage in ("full", "two", "free_b"):
    for r in jl_read(OUT / f"{stage}.jsonl"):
        key = r["branch"] if r["branch"] != "free" else r["pos_id"] + "|free"
        res[r["model"]][key][r["i"]] = bool(r["won"])
drift = defaultdict(lambda: defaultdict(list))
for r in jl_read(OUT / "drift.jsonl"):
    drift[r["model"]][r["pos_id"]].append(r["cmd"] if r["kind"] in ("valid", "alias") else "<bad>")
Bfree = {p["pos_id"]: v for p in pos if (v := AN.lst(res["B"].get(p["pos_id"] + "|free"))) is not None}
TH, mean = AN.TH, AN.mean
THR_KEEP = 0.5        # 事先定：π1 选 a0 的频率 < 0.5 才保留标签
MODELS = ["S_s11", "S_s12", "A_s11", "R_s11", "DPO-S_s11", "U_s11", "S-plan_s11"]


def act(b):
    a = D["branches"][b]["action"]
    return a.split(":", 1)[1] if a.startswith("orig:") else a


def auc(score, label):
    """P(score_过期 > score_未过期)，并列算 0.5。"""
    pos_ = [s for s, y in zip(score, label) if y]
    neg = [s for s, y in zip(score, label) if not y]
    if not pos_ or not neg:
        return float("nan")
    return mean((a > b) + 0.5 * (a == b) for a in pos_ for b in neg)


def boot(items, fn, n=2000):
    by = defaultdict(list)
    for it in items:
        by[it["traj"]].append(it)
    T = sorted(by)
    rng = random.Random(0)
    est = fn(items)
    vals = []
    for _ in range(n):
        s = [it for t in (rng.choice(T) for _ in T) for it in by[t]]
        v = fn(s)
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    return est, vals[int(.05 * len(vals))], vals[min(len(vals) - 1, int(.95 * len(vals)))], len(items), len(T)


def f(t):
    e, lo, hi, n, nt = t
    return f"{e:.3f} [{lo:.3f}, {hi:.3f}]（{n} 个位置，{nt} 条轨迹）"


PRED = ["p1_a0", "dp_a0", "tv", "p0_a0"]
lines = ["# A 层补充分析：不用续跑预测点过期（2026-11-06，事后、探索性）", "",
         "预测量只用每个位置 20 次首动作采样；点过期 = G_π1 < 0.20（S_s11、S_s12 用 G；其余模型用 G̃）；条件 G_π0 ≥ 0.20。", ""]
out = {}
pooled = []
for m in MODELS:
    pp = AN.per_position(D, ev, pos, res, m, Bfree)
    full = m in ("S_s11", "S_s12")
    items = []
    for k, x in pp.items():
        if x.get("G0") is None or x["G0"] < TH:
            continue
        g1 = x.get("G1") if full else x.get("Gt1")
        if g1 is None or k not in drift[m] or k not in drift["B"]:
            continue
        a0 = act(x["b0_all"])
        s1, s0 = drift[m][k], drift["B"][k]
        it = {"traj": x["traj"], "pos": k, "stale": g1 < TH, "p1_a0": mean(c == a0 for c in s1),
              "p0_a0": mean(c == a0 for c in s0), "tv": AN.tv(s1, s0), "model": m}
        it["dp_a0"] = it["p1_a0"] - it["p0_a0"]
        items.append(it)
    pooled += items
    o = {"n": len(items), "stale_rate": mean(it["stale"] for it in items)}
    lines += [f"## {m}（{'G' if full else 'G̃'}；点过期率 {o['stale_rate']:.3f}）", "", "| 预测量 | AUC [90% 区间] |", "|---|---|"]
    for p in PRED:
        o[p] = boot(items, lambda s, p=p: auc([it[p] for it in s], [it["stale"] for it in s]))
        lines.append(f"| {p} | {f(o[p])} |")
    # p1_a0 分箱：点过期率
    bins = [(0, 0.0001), (0.0001, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)]
    lines.append("")
    lines.append("p1_a0 分箱的点过期率：" + "；".join(
        f"[{a:.2f},{b:.2f}) {mean(it['stale'] for it in items if a <= it['p1_a0'] < b):.2f}（{sum(a <= it['p1_a0'] < b for it in items)}）"
        for a, b in bins if any(a <= it['p1_a0'] < b for it in items)))
    # 按预测保留（只对全分支模型，与只查位置比 V、Q）
    if full:
        rows, lam = AN.refresh(pp, {})
        d1 = {k: mean(c == act(pp[k]["b0_all"]) for c in drift[m][k]) for k in lam if k in drift[m]}
        lam = [k for k in lam if k in d1]
        def summ(meth):
            vals, kept = [], []
            for k in lam:
                for (kp, g, c), (_, g0, _) in zip(rows[k][5][meth if meth != "pred" else "reuse"], rows[k][5]["reuse"]):
                    if meth == "pred":
                        kp = float(d1[k] < THR_KEEP)
                    vals.append(kp * g)
                    if kp:
                        kept.append(g)
            return mean(vals), mean(kept), len(kept) / len(vals)
        lines += ["", f"按预测保留（p1_a0 < {THR_KEEP}，不用续跑）与其他刷新方式（{len(lam)} 个标签位置，k_c = 5）：", "",
                  "| 方式 | V | Q | 保留比例 | 每位置续跑数 |", "|---|---:|---:|---:|---:|"]
        cost = {"reuse": 0, "pred": 0, "point": 5, "point_label": 10, "full": None}
        for meth, name in (("reuse", "复用"), ("pred", "按预测保留"), ("point", "只查位置"), ("point_label", "只查标签"), ("full", "全部重新验证")):
            V, Q, kf = summ(meth)
            c = cost[meth] if cost[meth] is not None else mean(rows[k][5]["full"][0][2] for k in lam)
            lines.append(f"| {name} | {V:+.3f} | {Q:+.3f} | {kf:.2f} | {c:.1f} |")
            o[f"refresh_{meth}"] = [V, Q, kf, c]
        # 预测保留 vs 只查位置：逐位置配对的 V 差
        def per(k, meth):
            return mean((float(d1[k] < THR_KEEP) if meth == "pred" else kp) * g
                        for (kp, g, c) in rows[k][5][meth if meth != "pred" else "reuse"])
        its = [{"traj": pp[k]["traj"], "k": k} for k in lam]
        o["V_pred_minus_point"] = boot(its, lambda s: mean(per(it["k"], "pred") - per(it["k"], "point") for it in s))
        lines.append(f"\nV(按预测保留) − V(只查位置)：{f(o['V_pred_minus_point'])}")
    lines.append("")
    out[m] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()}
# 七个模型合并（同一位置在不同模型下算不同条目；区间按轨迹重抽，偏窄）
lines += ["## 七个模型合并（描述；同一轨迹的多个模型条目一起重抽）", "", "| 预测量 | AUC [90% 区间] |", "|---|---|"]
out["pooled"] = {}
for p in PRED:
    out["pooled"][p] = list(boot(pooled, lambda s, p=p: auc([it[p] for it in s], [it["stale"] for it in s])))
    lines.append(f"| {p} | {f(tuple(out['pooled'][p]))} |")
Path("stale_predict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
Path("stale_predict.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n".join(lines))
