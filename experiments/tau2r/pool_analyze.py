"""6.14 修订 1–3 的分析：τ² 共享失败状态池（第 6 篇的 pool_tau2.py 产生的数据；部署者 Qwen3.5-2B，执行者 2B / 4B，9B 可选）。

只读第 6 篇的设计文件与续跑文件，不调用模型、不需要 τ²（只要 numpy）；核心统计沿用 ds_alfw.analyze_core（与 ALFWorld、WebShop
同一套代码），区间改为**按任务成簇、按领域分层**的 bootstrap（同一任务的失败轨迹不独立）。

    python tau2r/pool_analyze.py --design <design_*.jsonl …> --ro <ro_*.jsonl …> [--tasks <池任务清单 …>] --out out_tau_pool
    python tau2r/pool_analyze.py --synthetic                已知答案的检验（不读任何数据）

字段（第 6 篇 2026-10-31 的格式说明）：设计文件每个位置一行（pos_id、sim_id、task_id、domain、t、cands[cand_id, kinds, restore_ok…]）；
续跑文件每次续跑一行（pos_id、cand_id、executor、temp、k、reward、score）。只用 score == "orig_task" 且 reward 不为 null 的行
（同一键取最后一行）；reward 是用不带前缀的原任务重判整段对话的分数（τ² 带前缀的判分有偏，见 论文方案.md 6.14 暂停一条）。

对应关系：一条失败对话（sim_id）= 一条轨迹；原动作（kinds 含 orig）是净收益的基准；替代候选 = 其余恢复成功的候选，
**去掉自由选择（free）**（它在不同执行者那里不是同一个动作）。金标下一步填在规划器的字段里（oracle 可行性记 1），
随机工具调用填在随机字段里，4B 的提议记为提出者 4B。
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
N_BOOT = 2000
DEPLOYER = "2B"
EXEC_N = {"2B": 10, "4B": 6, "9B": 6}
TESTS = [("P1t", "matched-greedy", "匹配（2B 5 次选）− 2B 贪心"),
         ("P2t", "eq_minus_4B", "同样 5 次：2B 自己选 − 按 4B 选（都在 2B 上评价）"),
         ]
FAMILY_M = 2          # 6.14 修订 2（10-31）：9B 移出主检验族
# 可选的追加检验（修订 2）：只有 9B 在冻结的池上补跑齐了才算，自成一族、单独 α = 0.05，不参与上面的 Holm
OPTIONAL = [("P3t", "eq_minus_9B", "同样 5 次：2B 自己选 − 按 9B 选（可选追加，单独 α = 0.05）")]
LABEL = {"4B": "Qwen3.5-4B 的提议", "planner": "金标下一步", "random": "随机工具调用"}


def DSmod():
    import ds_alfw as DS
    DS.EXEC_CODE.update({"2B": 21, "4B": 22, "9B": 23})   # 每个执行者自己的随机数流（只改本进程里的表）
    return DS


# ================================================================ 读数据
def jl(paths):
    out = []
    for p in paths:
        for l in Path(p).read_text(encoding="utf-8").splitlines():
            if l.strip():
                out.append(json.loads(l))
    return out


def build(design_rows, ro_rows, pool_tasks=None, adm=None, subset=None):
    """返回 d（ds_alfw 的设计格式，另带每条轨迹的 domain、task）、各执行者的 {bid: [奖励…]}、2B 贪心 {bid: 奖励}、自由选择、核对信息。
    adm：{(pos_id, cand_id): 是否可执行}（第 6 篇 adm_tau2.py 的输出）；给了就只留可执行的替代候选（没有判定的也去掉）。
    subset：只留这些失败对话（sim_id）；修订 3 的 2B 子集。"""
    if subset is not None:
        design_rows = [r for r in design_rows if r["sim_id"] in subset]
    if pool_tasks is not None:
        bad = sorted({(r["domain"], r["task_id"]) for r in design_rows} - pool_tasks)
        if bad:
            raise SystemExit(f"[pool] 设计文件里有不在池任务清单中的任务：{bad[:10]}")
    by_sim = {}
    n_adm_drop = [0]
    for r in design_rows:
        by_sim.setdefault(r["sim_id"], []).append(r)
    d = {"trajectories": [], "branches": {}}
    n_pos_noorig = n_pos_noalt = 0
    for sim_id in sorted(by_sim):
        rows = sorted(by_sim[sim_id], key=lambda r: r["t"])
        tp = []
        for r in rows:
            cs = [c for c in r["cands"] if c.get("restore_ok")]
            orig = [c for c in cs if "orig" in c["kinds"]]
            if len(orig) != 1:
                n_pos_noorig += 1
                continue
            alts = [c for c in cs if "orig" not in c["kinds"] and "free" not in c["kinds"] and "rand" not in c["kinds"]]
            if adm is not None:
                n_adm_drop[0] += sum(1 for c in alts if not adm.get((r["pos_id"], c["cand_id"])))
                alts = [c for c in alts if adm.get((r["pos_id"], c["cand_id"]))]
            if not alts:
                n_pos_noalt += 1
                continue
            ob = f"{r['pos_id']}|{orig[0]['cand_id']}"
            d["branches"][ob] = {"orig": True, "proposers": [], "planner": "gold" in orig[0]["kinds"], "random": False,
                                 "deterministic": None, "oracle_F": 0, "oracle_slack": 0, "flash_rank": None}
            ab = []
            for c in alts:
                b = f"{r['pos_id']}|{c['cand_id']}"
                d["branches"][b] = {"orig": False, "proposers": ["4B"] if "llm" in c["kinds"] else [],
                                    "planner": "gold" in c["kinds"], "random": "rand" in c["kinds"], "deterministic": None,
                                    "oracle_F": int("gold" in c["kinds"]), "oracle_slack": 0, "flash_rank": None,
                                    "kinds": c["kinds"], "tool": bool((c.get("msg") or {}).get("tool_calls"))}
                ab.append(b)
            tp.append({"t": r["t"], "orig": ob, "alts": ab, "pos_id": r["pos_id"]})
        if tp:
            d["trajectories"].append({"traj_id": sim_id, "task_id": rows[0]["task_id"], "domain": rows[0]["domain"],
                                      "positions": tp})
    # 续跑：同一键取最后一行有效的
    last = {}
    n_old = n_null = 0
    for r in ro_rows:
        if r.get("score") != "orig_task":
            n_old += 1
            continue
        if r.get("reward") is None:
            n_null += 1
            continue
        key = (f"{r['pos_id']}|{r['cand_id']}", r["executor"], float(r["temp"]), int(r["k"]))
        last[key] = r
    R = {e: {} for e in EXEC_N}
    G2, free, hist_diff = {}, {e: [] for e in EXEC_N}, [0, 0]
    for (b, e, temp, k), r in last.items():
        if r.get("reward_hist") is not None:
            hist_diff[0] += 1
            hist_diff[1] += float(r["reward_hist"]) != float(r["reward"])
        if r["cand_id"] == "free":
            if temp > 0 and e in free:
                free[e].append((r["pos_id"], float(r["reward"])))
            continue
        if temp == 0:
            if e == DEPLOYER and k == 0:
                G2[b] = float(r["reward"])
            continue
        if e in R:
            R[e].setdefault(b, {})[k] = float(r["reward"])
    S = {e: {b: [v[k] for k in range(EXEC_N[e])] for b, v in R[e].items() if all(k in v for k in range(EXEC_N[e]))}
         for e in EXEC_N}
    info = {"n_traj": len(d["trajectories"]), "n_positions": sum(len(t["positions"]) for t in d["trajectories"]),
            "n_branches": len(d["branches"]), "n_pos_without_orig": n_pos_noorig, "n_pos_without_alts": n_pos_noalt,
            "alts_dropped_not_admissible": n_adm_drop[0] if adm is not None else None,
            "rows_old_scoring_ignored": n_old, "rows_reward_null_ignored": n_null,
            "reward_hist_disagrees": f"{hist_diff[1]}/{hist_diff[0]}",
            "tasks": len({(t["domain"], t["task_id"]) for t in d["trajectories"]}),
            "by_domain": {dm: {"traj": sum(1 for t in d["trajectories"] if t["domain"] == dm),
                               "tasks": len({t["task_id"] for t in d["trajectories"] if t["domain"] == dm})}
                          for dm in sorted({t["domain"] for t in d["trajectories"]})}}
    return d, S, G2, free, info


# ================================================================ 统计
def boot_c(x, clusters, strata, seed=0, n=N_BOOT):
    """按任务成簇、按领域分层的 bootstrap：每个领域内有放回地抽任务（个数不变），取被抽中任务的全部轨迹，平均。"""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    cl = [c for c, o in zip(clusters, ok) if o]
    st = [s for s, o in zip(strata, ok) if o]
    x = x[ok]
    if len(x) == 0:
        return {"mean": float("nan"), "ci": [float("nan")] * 2, "p": float("nan"), "n": 0, "n_clusters": 0}
    rng = np.random.default_rng(seed)
    groups = {}
    for i, (c, s) in enumerate(zip(cl, st)):
        groups.setdefault(s, {}).setdefault(c, []).append(i)
    sums = {s: np.array([x[v].sum() for v in g.values()]) for s, g in groups.items()}
    cnts = {s: np.array([len(v) for v in g.values()]) for s, g in groups.items()}
    m = np.zeros(n)
    for b in range(n):
        tot = num = 0.0
        for s in groups:
            idx = rng.integers(0, len(sums[s]), size=len(sums[s]))
            tot += sums[s][idx].sum()
            num += cnts[s][idx].sum()
        m[b] = tot / num
    return {"mean": float(x.mean()), "ci": [float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))],
            "p": float((np.sum(m <= 0) + 1) / (n + 1)), "n": int(len(x)),
            "n_clusters": int(sum(len(g) for g in groups.values()))}


def arrays(d, t, S_by, G_by, n):
    cands = [(a, p["orig"], i) for i, p in enumerate(t["positions"]) for a in p["alts"]]
    bids = sorted({b for c in cands for b in c[:2]})
    col = {b: j for j, b in enumerate(bids)}
    return {"cands": cands, "bids": bids, "col": col, "det": np.full(len(bids), np.nan), "rand": np.arange(len(bids)),
            "S": np.array([S_by[b][:n] for b in bids], float).reshape(len(bids), n),
            "G": np.array([G_by[b] for b in bids], float),
            "ia": np.array([col[a] for a, _, _ in cands]), "io": np.array([col[o] for _, o, _ in cands])}


def core(DS, d, S, G2, target, others, obj, ks_self):
    n = EXEC_N[target]
    G = G2 if target == DEPLOYER else {b: float(np.mean(v)) for b, v in S[target].items()}
    TD = [arrays(d, t, S[target], G, n) for t in d["trajectories"]]
    O = {o: [arrays(d, t, S[o], {b: float(np.mean(v)) for b, v in S[o].items()}, EXEC_N[o]) for t in d["trajectories"]]
         for o in others}
    meta = lambda ti, b: d["branches"][b]
    return DS.analyze_core(TD, O, meta, obj=obj, ks_self=ks_self, proposers=("4B",))


def complete(d, R):
    return all(b in R for b in d["branches"])


def readouts(DS, d, S, G2, have):
    cl = [(t["domain"], t["task_id"]) for t in d["trajectories"]]
    st = [t["domain"] for t in d["trajectories"]]
    B = lambda x, sub=None: boot_c(x, cl, st) if sub is None else boot_c(
        [v for v, s in zip(x, sub) if s], [c for c, s in zip(cl, sub) if s], [q for q, s in zip(st, sub) if s])
    others = [o for o in ("4B", "9B") if o in have]
    res = {"others": others}
    for obj in ("net", "post"):
        res[obj] = {"rows": core(DS, d, S, G2, DEPLOYER, others, obj, (1, 2, 3, 5))}
    R = res["net"]["rows"]
    diff = lambda a, b: [r[a] - r[b] for r in R]
    tests = {"PC": B(diff("matched", "random")), "S1t": B(diff("matched", "planner_earliest"))}
    tests["PC"]["pass"] = bool(tests["PC"]["ci"][0] > 0.10)
    for tid, stat, lab in TESTS:
        v = DS.stat_values(R, stat)
        if v is not None:
            tests[tid] = dict(B(v), label=lab, stat=stat)
    DS.holm(tests, [t for t, _, _ in TESTS if t in tests], FAMILY_M)
    for tid, stat, lab in OPTIONAL:
        v = DS.stat_values(R, stat)
        if v is not None:
            tests[tid] = dict(B(v), label=lab, stat=stat, optional=True)
            tests[tid]["holm_threshold"], tests[tid]["holm_reject"] = 0.05, bool(tests[tid]["p"] <= 0.05)
    res["tests"] = tests
    # 按领域（描述；登记的规则：某领域阳性对照不过，该领域只报描述）
    per = {}
    for dm in sorted(set(st)):
        sub = [s == dm for s in st]
        x = {"PC": B(diff("matched", "random"), sub)}
        x["PC"]["pass"] = bool(x["PC"]["ci"][0] > 0.10)
        for tid, stat, lab in TESTS + OPTIONAL:
            v = DS.stat_values(R, stat)
            if v is not None:
                x[tid] = B(v, sub)
        per[dm] = x
    res["per_domain"] = per
    # 预算曲线
    curves = {}
    for obj in ("net", "post"):
        RR = res[obj]["rows"]
        c = {f"self_k{k}": B([r[f"self_k{k}"] for r in RR]) for k in (1, 2, 3, 5)}
        for o in others:
            c.update({f"{o}_k{k}": B([r[f"{o}_k{k}"] for r in RR]) for k in (1, 2, 3, 5)})
        for key in ("greedy", "matched", "random", "best_in_sample"):
            c[key] = B([r[key] for r in RR])
        curves[obj] = c
    res["curves"] = curves
    # 来源
    srcs = ["4B", "planner", "random"]
    res["sources"] = {x: {"self": B([r[f"prop_self_{x}"] for r in R]),
                          **{o: B([r[f"prop_{o}_{x}"] for r in R]) for o in others},
                          "best_is": float(np.mean([r["best_is"][x] for r in R]))} for x in srcs}
    res["frac_traj_best_gain_gt_0.10"] = float(np.mean([r["best_in_sample"] > 0.10 for r in R]))
    def sub_readouts(keep):
        """只保留 keep(分支) 为真的替代候选重算；没有候选的轨迹差值记 0（两边都不修复，同 6.9a）。"""
        d2, empty = DS.restrict_design(d, keep)
        for t2 in d2["trajectories"]:
            t0 = next(t for t in d["trajectories"] if t["traj_id"] == t2["traj_id"])
            t2["domain"], t2["task_id"] = t0["domain"], t0["task_id"]
        R2 = core(DS, d2, S, G2, DEPLOYER, others, "net", (1, 2, 3, 5)) if d2["trajectories"] else []
        cl2 = [(t["domain"], t["task_id"]) for t in d2["trajectories"]] + [
            next((t["domain"], t["task_id"]) for t in d["trajectories"] if t["traj_id"] == e) for e in empty]
        st2 = [c[0] for c in cl2]
        pad = [0.0] * len(empty)
        x = {"n_traj": len(d2["trajectories"]), "n_empty": len(empty),
             "matched-greedy": boot_c([r["matched"] - r["greedy"] for r in R2] + pad, cl2, st2),
             "PC": boot_c([r["matched"] - r["random"] for r in R2] + pad, cl2, st2)}
        for o in others:
            x[f"eq_minus_{o}"] = boot_c([r[f"eq_minus_{o}"] for r in R2] + pad, cl2, st2)
        return x

    # 次要：只用 4B 提议的候选
    res["model_only"] = sub_readouts(lambda b: bool(b.get("proposers")))
    # 探索性（修订 4）：只用带工具调用的替代候选 / 只用纯文字的替代候选
    res["by_type"] = {"tool": sub_readouts(lambda b: bool(b.get("tool"))),
                      "text": sub_readouts(lambda b: not b.get("tool"))}
    # 探索性：执行者 × 选择者矩阵（同样 3 次：目标执行者 3 选 3 评 − 按别人 3 次选，在目标执行者全部样本上评价）
    mat = {}
    exs = [DEPLOYER] + others
    for T in exs:
        oth = [o for o in exs if o != T]
        rows = core(DS, d, S, G2, T, oth, "net", (1, 2, 3))
        for o in oth:
            mat[f"{o}→{T}"] = B([r["self_k3"] - r[f"{o}_k3"] for r in rows])
        mat[f"{T} PC"] = B([r["self_k3"] - r["random"] for r in rows])
    res["matrix_k3"] = mat
    for obj in ("net", "post"):
        for r in res[obj]["rows"]:
            r.pop("best_is", None)
    return res


# ================================================================ 报告
def fmt(b):
    return f"{b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]" if b.get("n") else "—"


def free_summary(free, d):
    pos = {p["pos_id"] for t in d["trajectories"] for p in t["positions"]}
    out = {}
    for e, xs in free.items():
        v = [r for p, r in xs if p in pos]
        if v:
            out[e] = {"n": len(v), "mean_reward": float(np.mean(v))}
    return out


def report(res):
    t = res["tests"]
    L = ["# 6.14 修订 1：τ² 共享失败状态池（部署 Qwen3.5-2B；执行者 2B / 4B / 9B；用户 Qwen3.8-27B-FP8）", ""]
    if res.get("synthetic"):
        L += ["> **合成数据**：数字没有任何含义。", ""]
    L += [f"描述：{json.dumps(res['info'], ensure_ascii=False)}", "",
          "数值 = 选中修复在 2B 上的净收益（替代分支 − 原动作分支；奖励为原任务重判的 0/1）；"
          "90% 区间为按任务成簇、按领域分层的 bootstrap（2,000 次）；匹配 = 2B 自己 5 次选、另 5 次评（交叉拟合）。", "",
          "| 读数 | 对比 | 估计 [90%] | 说明 |", "|---|---|---|---|",
          f"| 阳性对照 | 匹配 − 随机 | {fmt(t['PC'])} | {'通过' if t['PC']['pass'] else '未通过（全部读数标为无判别力）'} |"]
    for tid, _, lab in TESTS + OPTIONAL:
        if tid in t:
            L.append(f"| {tid} | {lab} | {fmt(t[tid])} | p = {t[tid]['p']:.4f}；{'单独' if t[tid].get('optional') else 'Holm（m = 2）'}阈值 {t[tid]['holm_threshold']:.4f}，"
                     f"{'确认' if t[tid]['holm_reject'] else '未确认'}；上界 {'<' if t[tid]['ci'][1] < 0.10 else '≥'} 0.10 |")
        else:
            L.append(f"| {tid} | {lab} | {'没有跑（可选）' if tid == 'P3t' else '还没有结果（按 p = 1 计）'} | — |")
    L += [f"| S1t | 匹配 − 最早位置用金标下一步 | {fmt(t['S1t'])} | 次要：{'确认' if t['S1t']['ci'][0] > 0 else '未确认'} |", ""]
    L += ["**按领域**（描述）", "", "| 领域 | 阳性对照 | P1t | P2t | P3t |", "|---|---|---|---|---|"]
    for dm, x in res["per_domain"].items():
        L.append(f"| {dm} | {fmt(x['PC'])}（{'过' if x['PC']['pass'] else '不过'}） | " +
                 " | ".join(fmt(x[k]) if k in x else "—" for k in ("P1t", "P2t", "P3t")) + " |")
    mo = res["model_only"]
    L += ["", f"**只用 4B 提议的候选**（{mo['n_traj']} 条有候选，{mo['n_empty']} 条没有、按 0 计）：匹配 − 贪心 "
              f"{fmt(mo['matched-greedy'])}；阳性对照 {fmt(mo['PC'])}；" +
          "；".join(f"同样 5 次 2B − {o} {fmt(mo[f'eq_minus_{o}'])}" for o in res["others"]), ""]
    for obj, lab in (("net", "净收益"), ("post", "修复后的值")):
        c = res["curves"][obj]
        L += [f"**预算曲线（{lab}；选中修复在 2B 上的价值）**", "", "| 选择方式 | 价值 [90%] |", "|---|---|"]
        for k in (1, 2, 3, 5):
            L.append(f"| 2B {k} 次 | {fmt(c[f'self_k{k}'])} |")
        L += [f"| 2B 贪心 | {fmt(c['greedy'])} |", f"| 随机 | {fmt(c['random'])} |", f"| 样本内最佳（上限） | {fmt(c['best_in_sample'])} |"]
        for o in res["others"]:
            for k in (1, 2, 3, 5):
                L.append(f"| 按 {o} {k} 次 | {fmt(c[f'{o}_k{k}'])} |")
        L.append("")
    L += ["**候选来源**（平均净收益；探索性）", "", "| 来源 | 在 2B 上 | " + " | ".join(f"在 {o} 上" for o in res["others"]) +
          " | 是 2B 最佳候选的比例 |", "|---|---|" + "---|" * len(res["others"]) + "---:|"]
    for typ, lab in (("tool", "只用带工具调用的替代候选"), ("text", "只用纯文字的替代候选")):
        x = res["by_type"][typ]
        L.insert(L.index("**候选来源**（平均净收益；探索性）"),
                 f"**{lab}**（探索性；{x['n_traj']} 条有候选，{x['n_empty']} 条没有、按 0 计）：匹配 − 贪心 {fmt(x['matched-greedy'])}；"
                 f"阳性对照 {fmt(x['PC'])}；" + "；".join(f"同样 5 次 2B − {o} {fmt(x[f'eq_minus_{o}'])}" for o in res["others"]) + "\n")
    for x, v in res["sources"].items():
        L.append(f"| {LABEL[x]} | {fmt(v['self'])} | " + " | ".join(fmt(v[o]) for o in res["others"]) + f" | {v['best_is']:.2f} |")
    L += ["", f"存在净收益 > 0.10 候选的轨迹比例：{res['frac_traj_best_gain_gt_0.10']:.2f}", "",
          "**执行者 × 选择者（探索性；同样 3 次：目标 3 选 3 评 − 按别人 3 次选）**", "", "| 方向 | 损失 [90%] |", "|---|---|"]
    for k, v in res["matrix_k3"].items():
        L.append(f"| {k} | {fmt(v)} |")
    L += ["", f"自由选择（各执行者从同一批位置自己接着做；描述，不参加选择）：{json.dumps(res['free'], ensure_ascii=False)}", ""]
    return "\n".join(L) + "\n"


def analyze(design_rows, ro_rows, out_dir, pool_tasks=None, synthetic=False, adm=None, subset=None):
    DS = DSmod()
    d, S, G2, free, info = build(design_rows, ro_rows, pool_tasks, adm, subset)
    if not d["trajectories"]:
        print(f"[pool] 没有可分析的位置（每个位置都要有一个原动作和至少一个替代候选）：{json.dumps(info, ensure_ascii=False)}")
        return 1
    if not complete(d, S[DEPLOYER]) or not complete(d, G2):
        print(f"[pool] 2B 的续跑没有完成：10 次齐全的分支 {len(S[DEPLOYER])}/{len(d['branches'])}，贪心 {len(G2)}/{len(d['branches'])}")
        return 1
    have = [o for o in ("4B", "9B") if complete(d, S[o])]
    for o in ("4B", "9B"):
        if o not in have:
            print(f"[pool] {o} 的续跑缺失或未完成（{len(S[o])}/{len(d['branches'])}）：与它有关的读数跳过"
                  + ("（P2t 按 p = 1 计）" if o == "4B" else "（可选的 P3t 不报告）"))
    res = readouts(DS, d, S, G2, have)
    res["info"], res["free"], res["synthetic"] = info, free_summary(free, d), synthetic
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(report(res), encoding="utf-8")
    print(report(res))
    return 0


# ================================================================ 已知答案
def synthetic():
    """两个领域、每任务最多 2 条失败；4B 的价值排序与 2B 相反、9B 与 2B 同分布；贪心是噪声。
    预期：P2t 明显 > 0，P3t ≈ 0，P1t > 0，阳性对照 > 0。另检查：自由选择与 free 候选不进入选择；旧判分的行被忽略。"""
    rng = np.random.default_rng(3)
    D, RO = [], []
    for dm, ntask in (("retail", 20), ("airline", 15)):
        for ti in range(ntask):
            for tr in range(1 + ti % 2):
                sim = f"{dm}-{ti}-{tr}"
                for t in (4, 10, 16):
                    pid = f"{sim}_{t}"
                    cands = [{"cand_id": "c0", "kinds": ["orig"], "restore_ok": True},
                             {"cand_id": "free", "kinds": ["free"], "restore_ok": True}]
                    cands += [{"cand_id": f"c{i}", "kinds": [["llm", "gold", "rand", "llm"][i - 1]], "restore_ok": True,
                               "msg": {"role": "assistant", "tool_calls": [{"name": "x"}]} if i == 2 else {"role": "assistant", "content": "t"}}
                              for i in range(1, 5)]
                    D.append({"pos_id": pid, "sim_id": sim, "task_id": str(ti), "domain": dm, "t": t, "cands": cands})
                    for c in cands:
                        p2 = rng.uniform(0.05, 0.4) if c["cand_id"] in ("c0", "free") else rng.uniform(0, 1)
                        p_ = {"2B": p2, "4B": 1 - p2 if c["cand_id"] not in ("c0", "free") else p2, "9B": p2}
                        for e, n in EXEC_N.items():
                            for k in range(n):
                                RO.append({"pos_id": pid, "cand_id": c["cand_id"], "executor": e, "temp": 0.7, "k": k,
                                           "reward": float(rng.random() < p_[e]), "reward_hist": 1.0, "score": "orig_task"})
                        RO.append({"pos_id": pid, "cand_id": c["cand_id"], "executor": "2B", "temp": 0.0, "k": 0,
                                   "reward": float(rng.random() < 0.5), "score": "orig_task"})
                        RO.append({"pos_id": pid, "cand_id": c["cand_id"], "executor": "2B", "temp": 0.7, "k": 0,
                                   "reward": 1.0, "reward_hist": 1.0})          # 旧判分（没有 score）：必须被忽略
    import tempfile
    od = Path(tempfile.mkdtemp())
    assert analyze(D, RO, od, synthetic=True) == 0
    res = json.loads((od / "analysis.json").read_text(encoding="utf-8"))
    t = res["tests"]
    assert t["P2t"]["mean"] > 0.25 and t["P2t"]["holm_reject"], t["P2t"]
    assert abs(t["P3t"]["mean"]) < 0.08 and not t["P3t"]["holm_reject"] and t["P3t"].get("optional"), t["P3t"]
    assert abs(t["P2t"]["holm_threshold"] - 0.025) < 1e-9 or abs(t["P2t"]["holm_threshold"] - 0.05) < 1e-9
    assert t["P1t"]["mean"] > 0.15 and t["PC"]["mean"] > 0.15, (t["P1t"], t["PC"])
    assert res["info"]["rows_old_scoring_ignored"] > 0
    assert res["info"]["n_branches"] == 4 * len(D), res["info"]["n_branches"]   # 每位置 1 原动作 + 3 替代（free、rand 不算，修订 3）
    assert t["P2t"]["n_clusters"] == 35
    # 不跑 9B（修订 2 的默认）：P3t 不出现，P1t、P2t 照常按 m = 2 做 Holm
    od2 = Path(tempfile.mkdtemp())
    assert analyze(D, [r for r in RO if r["executor"] != "9B"], od2, synthetic=True) == 0
    t2 = json.loads((od2 / "analysis.json").read_text(encoding="utf-8"))["tests"]
    assert "P3t" not in t2 and t2["P2t"]["holm_reject"] and t2["P1t"]["holm_reject"], t2
    # 可执行性过滤与子集：把每个位置的 c3（rand）以及 c4 标为不可执行 → 替代候选只剩 c1、c2；子集取前 60 个可用位置
    adm = {(r["pos_id"], c["cand_id"]): c["cand_id"] in ("c1", "c2") for r in D for c in r["cands"]}
    trajs, n_pos, n_tasks = make_subset(D, adm, 60)
    assert n_pos >= 60 and n_tasks == 35 and all(len(t["positions"]) == 3 for t in trajs), (n_pos, n_tasks)
    assert len({(t["domain"], t["task_id"]) for t in trajs}) == len(trajs), "第一轮每个任务只应取 1 条"
    trajs2, n_pos2, _ = make_subset(D, adm, 10 ** 6)
    assert len(trajs2) == len({r["sim_id"] for r in D}), "取满时应包含全部对话（每任务最多 2 条）"
    od3 = Path(tempfile.mkdtemp())
    assert analyze(D, RO, od3, synthetic=True, adm=adm, subset={t["traj_id"] for t in trajs}) == 0
    r3 = json.loads((od3 / "analysis.json").read_text(encoding="utf-8"))
    assert r3["info"]["n_branches"] == 3 * n_pos and r3["info"]["n_traj"] == len(trajs), r3["info"]
    bt = res["by_type"]
    assert bt["tool"]["n_traj"] == len({r["sim_id"] for r in D}) and bt["text"]["n_traj"] == bt["tool"]["n_traj"], bt
    print("[synthetic] == 已知答案全部对上（含不跑 9B、可执行性过滤、2B 子集与按候选类型分组）")
    return 0


def load_adm(paths):
    """adm_tau2.py 的输出：每个候选一行 {pos_id, cand_id, ok, why, rule}；同一候选取最后一行；只接受同一个规则版本。"""
    rows = jl(paths)
    rules = {r.get("rule") for r in rows}
    if len(rules) != 1:
        raise SystemExit(f"[pool] 可执行性文件混了不同的规则版本：{rules}")
    return {(r["pos_id"], r["cand_id"]): bool(r["ok"]) for r in rows}, rules.pop()


def make_subset(design_rows, adm, target, pool_tasks=None):
    """修订 3 的 2B 子集（在任何续跑之前生成）：以失败对话为单位。
    1) 一条对话只算“可用位置”= 有原动作、且至少一个可执行替代候选（v2，不含 free / rand）的位置；没有可用位置的对话不进；
    2) 每个任务内按 sha256("P4SUB|" + sim_id) 排序，任务按 sha256("P4SUB-T|" + 领域 + "|" + 任务) 排序；
    3) 轮流从每个任务取下一条（每任务最多 2 条），整条对话一起进，直到可用位置数 ≥ target（取满或取完为止）。"""
    import hashlib
    h = lambda x: hashlib.sha256(x.encode()).hexdigest()
    d, _, _, _, _ = build(design_rows, [], pool_tasks, adm)
    by_task = {}
    for t in d["trajectories"]:
        by_task.setdefault((t["domain"], t["task_id"]), []).append(t)
    tasks = sorted(by_task, key=lambda k: h(f"P4SUB-T|{k[0]}|{k[1]}"))
    for k in tasks:
        by_task[k].sort(key=lambda t: h("P4SUB|" + t["traj_id"]))
    out, n_pos = [], 0
    for rnd in range(2):
        for k in tasks:
            if n_pos >= target:
                break
            if rnd < len(by_task[k]):
                t = by_task[k][rnd]
                out.append(t)
                n_pos += len(t["positions"])
    return out, n_pos, len(tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", nargs="+")
    ap.add_argument("--ro", nargs="+")
    ap.add_argument("--tasks", nargs="*", help="池任务清单（每行“领域<Tab或空格>任务编号”）；给了就核对设计文件只含池任务")
    ap.add_argument("--out", default=str(ROOT / "out_tau_pool"))
    ap.add_argument("--adm", nargs="*", help="主分析的可执行性文件（修订 3：v3）")
    ap.add_argument("--adm-sens", nargs="*", help="敏感性分析的可执行性文件（修订 3：v2）；结果写在 <out>/sens/")
    ap.add_argument("--subset", help="2B 子集文件（--make-subset 的输出；每行一个 sim_id，# 开头是注释）")
    ap.add_argument("--make-subset", type=int, metavar="N", help="生成 2B 子集：可用位置数达到 N 为止（修订 3：200）；"
                    "需要 --design 与 --adm（v2），写到 --out 指定的文件")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        return synthetic()
    files = lambda xs: sorted({f for x in xs for f in glob.glob(x)})
    pool = None
    if a.tasks:
        pool = {tuple(l.split()[:2]) for p in a.tasks for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()}
    if a.make_subset:
        adm, rule = load_adm(files(a.adm))
        if rule != "v2":
            raise SystemExit(f"[pool] 子集按 v2 的可执行性定（修订 3），这里给的是 {rule}")
        trajs, n_pos, n_tasks = make_subset(jl(files(a.design)), adm, a.make_subset, pool)
        L = [f"# 6.14 修订 3 的 2B 子集：{len(trajs)} 条失败对话、{n_pos} 个可用位置、"
             f"{len({(t['domain'], t['task_id']) for t in trajs})}/{n_tasks} 个任务；规则见 pool_analyze.make_subset"]
        L += [f"{t['traj_id']}\t{t['domain']}\t{t['task_id']}\t" + ",".join(p["pos_id"] for p in t["positions"]) for t in trajs]
        Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
        print(L[0])
        return 0
    subset = None
    if a.subset:
        subset = {l.split("\t")[0] for l in Path(a.subset).read_text(encoding="utf-8").splitlines()
                  if l.strip() and not l.startswith("#")}
    D, RO = jl(files(a.design)), jl(files(a.ro))
    adm = load_adm(files(a.adm))[0] if a.adm else None
    rc = analyze(D, RO, a.out, pool, adm=adm, subset=subset)
    if a.adm_sens:
        rc2 = analyze(D, RO, str(Path(a.out) / "sens"), pool, adm=load_adm(files(a.adm_sens))[0], subset=subset)
        rc = rc or rc2
    return rc


if __name__ == "__main__":
    sys.exit(main())
