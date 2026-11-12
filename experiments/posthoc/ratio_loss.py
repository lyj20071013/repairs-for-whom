"""事后分析（2026-11-11）：错配损失占"选修复的价值"的比例。只读各环境 analysis.json 里逐轨迹的行；不调用模型。
比例 = mean(自己 5 次选 − 按别的模型 5 次选) / mean(自己选 − 随机选)；分子分母都是登记过的统计量（P2 类检验与阳性对照）。
区间：按轨迹成对重抽样 10,000 次的 90% 百分位区间（τ² 的登记分析按任务成簇；这里行里没有任务号，按轨迹抽，区间略偏窄）。
另给贪心的比例：mean(自己 5 次选 − 自己贪心) / 同一分母。
用法：python ratio_loss.py <repair_transfer 目录> <τ² 的 analysis.json> [输出前缀]
"""
import json, sys
import numpy as np
R, TAU = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "ratio_loss"
SPEC = [  # 环境, 文件, 行所在的键路径, 部署（执行）者, [(选择者, 损失列)]
    ("ALFWorld", f"{R}/out_ds_alfw/analysis.json", ("net",), "Qwen2.5-7B", [("Qwen2.5-14B", "eq_7B_minus_14B")]),
    ("ALFWorld", f"{R}/out_ds_alfw_l8/analysis.json", ("net",), "Llama-3.1-8B", [("Qwen2.5-7B", "eq_minus_q7")]),
    ("WebShop", f"{R}/out_ws_A/analysis.json", ("reward", "net"), "Qwen2.5-7B", [("Qwen2.5-14B", "eq_minus_q14"), ("Llama-3.1-8B", "eq_minus_l8")]),
    ("WebShop", f"{R}/out_ws_B/analysis.json", ("reward", "net"), "Llama-3.1-8B", [("Qwen2.5-7B", "eq_minus_q7")]),
    ("τ²", TAU, ("net",), "Qwen3.5-2B", [("Qwen3.5-4B", "eq_minus_4B")]),
]
rng = np.random.default_rng(0)
out, L = [], ["| 环境 | 执行者 ← 选择者 | 损失 | 自己选 − 随机选 | 比例 [90%] | 贪心的比例 [90%] | 阳性对照 |", "|---|---|---:|---:|---|---|---|"]
for env, f, path, dep, losses in SPEC:
    a = json.load(open(f, encoding="utf-8"))
    rows = a
    for k in path:
        rows = rows[k]
    rows = rows["rows"]
    col = lambda k: np.array([r[k] for r in rows], float)
    pc = col("matched") - col("random")
    assert abs(pc.mean() - a["tests"]["PC"]["mean"]) < 1e-9, (env, pc.mean(), a["tests"]["PC"]["mean"])
    gr = col("matched") - col("greedy")
    idx = rng.integers(0, len(rows), (10000, len(rows)))

    def ratio(num):
        b = num[idx].mean(1) / pc[idx].mean(1)
        return float(num.mean() / pc.mean()), float(np.quantile(b, .05)), float(np.quantile(b, .95))
    g = ratio(gr)
    for sel, c in losses:
        x = col(c)
        r = ratio(x)
        out.append({"env": env, "executor": dep, "selector": sel, "n": len(rows), "loss": float(x.mean()), "pc": float(pc.mean()),
                    "ratio": r, "greedy_loss": float(gr.mean()), "greedy_ratio": g, "pc_pass": bool(a["tests"]["PC"]["pass"])})
        L.append(f"| {env} | {dep} ← {sel} | {x.mean():+.3f} | {pc.mean():+.3f} | {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}] | {g[0]:.2f} [{g[1]:.2f}, {g[2]:.2f}] | {'通过' if a['tests']['PC']['pass'] else '未通过'} |")
r = np.array([o["ratio"][0] for o in out]); l = np.array([o["loss"] for o in out]); p = np.array([o["pc"] for o in out])
L += ["", f"比例的范围 {r.min():.2f}–{r.max():.2f}（最大 / 最小 = {r.max() / r.min():.1f}）；绝对损失的范围 {l.min():+.3f} 至 {l.max():+.3f}（最大 / 最小 = {l.max() / l.min():.1f}）；分母的范围 {p.min():+.3f} 至 {p.max():+.3f}"]
open(OUT + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
json.dump(out, open(OUT + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n".join(L))
