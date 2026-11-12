"""6.14 附（探索性，零算力）：把 14B 当执行者、在 7B 失败池上读“按别人选、给 14B 用”的损失（P2 的反方向）。

ALFWorld 的同一格已在 6.12 B 里算过（7B 池中按 7B 选、给 14B 用，全部候选，净收益 +0.082 [+0.042, +0.126]）；
这里用同一定义（ds612 的 eq：目标执行者 5 选 / 5 评的交叉拟合 − 另一执行者前 5 次选、在目标执行者全部样本上评价，
重排 50 次取平均；ds_alfw.analyze_core，ks_self = (1, 2, 3, 5, 10)）补 WebShop 的 WS-A 池（主目标奖励，另报成功率），
并先在 ALFWorld 上重算 6.12 的那一格，核对实现。

    python posthoc/rev14.py            读 out_ds_alfw/ 与 out_ws_A/，写 out_rev14/report.md、rev14.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ds_alfw as DS          # noqa: E402
import runner                 # noqa: E402

KS_SELF = (1, 2, 3, 5, 10)
K_EQ = 5
POOLS = {
    "ALFWorld": {"dir": "out_ds_alfw", "batch": "D", "ex": {"q7": 10, "q14": 20, "l8": 10},
                 "proposers": ("q7", "q14", "qflash"), "metrics": ("success",)},
    "WebShop": {"dir": "out_ws_A", "batch": "WS1", "ex": {"q7": 10, "q14": 20, "l8": 10},
                "proposers": ("q7", "l8", "qflash"), "metrics": ("reward", "success")},
}


def arrays(d, t, ev, name, n, batch, metric):
    cands = [(a, p["orig"], i) for i, p in enumerate(t["positions"]) for a in p["alts"]]
    bids = sorted({b for c in cands for b in c[:2]})
    col = {b: j for j, b in enumerate(bids)}
    br = d["branches"]
    key = "rewards" if metric == "reward" else "results"
    dv = (lambda b: br[b]["det_reward"]) if metric == "reward" else (lambda b: br[b]["deterministic"])
    det = np.array([np.nan if br[b]["deterministic"] is None else float(dv(b)) for b in bids])
    rand = np.flatnonzero(np.isnan(det))
    S = np.array([ev[(f"{name}_T07", batch, bids[j])][key][:n] for j in rand], float).reshape(len(rand), n)
    G = det.copy()
    for j in rand:
        G[j] = float(ev[(f"{name}_greedy", batch, bids[j])][key][0])
    return {"cands": cands, "bids": bids, "col": col, "det": det, "rand": rand, "S": S, "G": G,
            "ia": np.array([col[a] for a, _, _ in cands]), "io": np.array([col[o] for _, o, _ in cands])}


def run(pool_name, spec, metric):
    base = ROOT / spec["dir"]
    d = json.loads((base / "design.json").read_text(encoding="utf-8"))
    ev = {n: runner.load_results(base / f"evals_{n}.jsonl") for n in spec["ex"]}
    meta = lambda ti, b: d["branches"][b]
    out = {}
    for var in ("all", "model"):
        dv, empty = (d, []) if var == "all" else DS.restrict_design(d, lambda b: bool(b.get("proposers")))
        TD = [arrays(dv, t, ev["q14"], "q14", spec["ex"]["q14"], spec["batch"], metric) for t in dv["trajectories"]]
        O = {o: [arrays(dv, t, ev[o], o, spec["ex"][o], spec["batch"], metric) for t in dv["trajectories"]]
             for o in ("q7", "l8")}
        rows = DS.analyze_core(TD, O, meta, obj="net", ks_self=KS_SELF, proposers=spec["proposers"])
        pad = [0.0] * len(empty)
        out[var] = {
            "q7→q14": DS.boot([r[f"self_k{K_EQ}"] - r[f"q7_k{K_EQ}"] for r in rows] + pad),
            "l8→q14": DS.boot([r[f"self_k{K_EQ}"] - r[f"l8_k{K_EQ}"] for r in rows] + pad),
            "q14 matched(10)−greedy": DS.boot([r["matched"] - r["greedy"] for r in rows] + pad),
            "q14 PC": DS.boot([r["matched"] - r["random"] for r in rows] + pad),
            "n_traj": len(rows), "n_empty": len(empty)}
    return out


def main():
    res = {}
    for name, spec in POOLS.items():
        for metric in spec["metrics"]:
            res[f"{name}|{metric}"] = run(name, spec, metric)
    f = lambda b: f"{b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]（p = {b['p']:.4f}）"
    L = ["# 6.14 附：14B 当执行者时，按别人选的损失（7B 失败池；探索性）", "",
         "数值 = 选中修复在 14B 上的净收益之差（14B 自己 5 次选、另 10 次评的交叉拟合 − 按另一执行者 5 次选、在 14B 全部 20 次上评价；"
         "90% 轨迹 bootstrap 区间）。ALFWorld 的 7B→14B（全部候选）应与 6.12 B 的 +0.082 [+0.042, +0.126] 相同。", ""]
    for key, r in res.items():
        L += [f"## {key}", "", "| 候选 | 按 7B 选、给 14B 用 | 按 Llama 选、给 14B 用 | 14B 匹配 − 14B 贪心 | 14B 的阳性对照 |",
              "|---|---|---|---|---|"]
        for var in ("all", "model"):
            x = r[var]
            sub = f"，另有 {x['n_empty']} 条没有候选、按 0 计" if x["n_empty"] else ""
            L.append(f"| {'全部' if var == 'all' else '只用模型提出的'}（{x['n_traj']} 条{sub}） | "
                     f"{f(x['q7→q14'])} | {f(x['l8→q14'])} | {f(x['q14 matched(10)−greedy'])} | {f(x['q14 PC'])} |")
        L.append("")
    od = ROOT / "out_rev14"
    od.mkdir(exist_ok=True)
    (od / "rev14.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (od / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
