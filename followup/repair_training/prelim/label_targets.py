"""第 5 篇（方向 B）的事前分析：用上一篇已有的续跑标签，估计"换打标签者"会让训练目标差多少。

探索性，只读 repair_transfer 的输出（不改动它们），结果只用于设计与功效估计，不进入任何登记检验。
2026-10-28 写（北京时间），运行前没有看过任何新数据；用到的数据都是上一篇已公开分析过的。

决策点 = 一条失败轨迹上的一个位置（早 / 中 / 晚）。每个打标签者对该点每个候选（及原动作）用 k 次续跑估净收益，
取最大者作为训练目标（强制取 argmax，不做 "收益 > 0 才入选" 的过滤；并列时在并列者上平均）。
目标质量 = 该目标在部署 7B 的 **另外 5 次** 续跑上的净收益（7B 自己选时用前 k 次选、后 5 次评，交叉拟合）。
其他打标签者：14B（20 次中随机取 k 次）、Llama-8B（10 次中取 k 次）、7B / 14B 贪心（1 次）、
规划器 / 特权动作（不验证）、随机候选、混合（7B 与 14B 各 k 次估计的平均，双倍预算）。

用法：
  python label_targets.py --env alfw --data D:/Paper/repair_transfer/out_ds_alfw --k 3
  python label_targets.py --env ws   --data D:/Paper/repair_transfer/out_ws_A    --k 3
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
LABELERS = ("q7", "q14", "l8", "mix", "g7", "g14", "planner", "random")


def load(env, data):
    data = Path(data)
    d = json.loads((data / "design.json").read_text(encoding="utf-8"))
    field = "results" if env == "alfw" else "rewards"     # ALFWorld：成功（0/1）；WebShop：奖励（主目标）
    ev = {}
    for name in ("q7", "q14", "l8"):
        f = data / f"evals_{name}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            ev[(r["condition"], r["branch_id"])] = np.array(r[field], float)
    return d, ev


def make_value(env, d, ev):
    br = d["branches"]

    def V(b, name, idx=None, greedy=False):
        x = br[b]
        if x.get("deterministic") is not None:
            if env == "ws" and x.get("det_reward") is not None:
                return float(x["det_reward"])
            return float(x["deterministic"])
        if greedy:
            return float(ev[(f"{name}_greedy", b)][0])
        v = ev[(f"{name}_T07", b)]
        return float(v[idx].mean()) if idx is not None else float(v.mean())
    return V


def run(env, data, k, n_rep, seed, only_llm=False):
    d, ev = load(env, data)
    V = make_value(env, d, ev)
    br = d["branches"]
    rng = np.random.default_rng(seed)
    n14 = len(next(v for (c, _), v in ev.items() if c == "q14_T07"))
    per = {L: [] for L in LABELERS}
    agree = {L: [] for L in LABELERS}
    is_pl = {L: [] for L in LABELERS}
    n_pos = 0
    for t in d["trajectories"]:
        for p in t["positions"]:
            alts, o = p["alts"], p["orig"]
            pl = np.array([bool(br[a].get("planner")) for a in alts])
            llm = np.array([len(br[a].get("proposers", [])) > 0 for a in alts])
            sub = (llm & ~pl) if only_llm else np.ones(len(alts), bool)
            if only_llm:
                if not sub.any():
                    continue
            elif not pl.any():                 # 所有组用同一批决策点：要求有规划器 / 特权动作
                continue
            n_pos += 1
            acc = {L: [] for L in LABELERS}
            ag = {L: [] for L in LABELERS}
            ip = {L: [] for L in LABELERS}
            for _ in range(n_rep):
                pm = rng.permutation(10)
                A, B = pm[:k], pm[5:]          # 7B：前 5 次里取 k 次选，后 5 次评
                ev7 = np.array([V(a, "q7", B) - V(o, "q7", B) for a in alts])
                i14 = rng.permutation(n14)[:k]
                il8 = rng.permutation(10)[:k]
                s7 = np.array([V(a, "q7", A) - V(o, "q7", A) for a in alts])
                s14 = np.array([V(a, "q14", i14) - V(o, "q14", i14) for a in alts])
                sel = {"q7": s7, "q14": s14, "mix": (s7 + s14) / 2,
                       "g7": np.array([V(a, "q7", greedy=True) - V(o, "q7", greedy=True) for a in alts]),
                       "g14": np.array([V(a, "q14", greedy=True) - V(o, "q14", greedy=True) for a in alts]),
                       "planner": pl.astype(float), "random": rng.random(len(alts))}
                if any(c == "l8_T07" for c, _ in ev):
                    sel["l8"] = np.array([V(a, "l8", il8) - V(o, "l8", il8) for a in alts])
                ref = None
                for L in LABELERS:
                    if L not in sel or (L == "planner" and only_llm):
                        continue
                    s = np.where(sub, sel[L], -np.inf)
                    ids = np.flatnonzero(s >= s.max() - 1e-12)
                    if L == "q7":
                        ref = set(ids)
                    acc[L].append(ev7[ids].mean())
                    ag[L].append(len(set(ids) & ref) / len(set(ids) | ref))
                    ip[L].append(pl[ids].mean())
            for L in LABELERS:
                per[L].append(np.mean(acc[L]) if acc[L] else np.nan)
                agree[L].append(np.mean(ag[L]) if ag[L] else np.nan)
                is_pl[L].append(np.mean(ip[L]) if ip[L] else np.nan)
    return n_pos, {L: np.array(v) for L, v in per.items()}, agree, is_pl


def report(env, data, k, n_rep, seed, only_llm):
    n, per, agree, is_pl = run(env, data, k, n_rep, seed, only_llm)
    obj = "成功率净收益" if env == "alfw" else "奖励净收益"
    cand = "只用模型提出的候选" if only_llm else "全部候选（限有规划器/特权动作的决策点）"
    out = [f"### {env}，{cand}，k = {k}（决策点 {n} 个；{obj}；重复 {n_rep} 次取平均，种子 {seed}）", "",
           "| 打标签者 | 目标在 7B 上的净收益 | 与 7B 目标相同（Jaccard） | 目标是规划器/特权动作 | 7B − 它（每决策点） | 决策点间 SD | 差 > 0.05 的比例 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for L in LABELERS:
        v = per[L]
        if not np.isfinite(v).any():
            continue
        dl = per["q7"] - v
        out.append(f"| {L} | {np.nanmean(v):+.3f} | {np.nanmean(agree[L]):.2f} | {np.nanmean(is_pl[L]):.2f} | "
                   + ("— | — | — |" if L == "q7" else
                      f"{np.nanmean(dl):+.3f} | {np.nanstd(dl, ddof=1):.3f} | {np.nanmean(np.abs(dl) > 0.05):.2f} |"))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["alfw", "ws"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-llm", action="store_true")
    a = ap.parse_args()
    print(report(a.env, a.data, a.k, a.reps, a.seed, a.only_llm))


if __name__ == "__main__":
    main()
