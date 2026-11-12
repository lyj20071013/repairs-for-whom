"""事后分析：余量、错配损失与标签可靠性（定义见 `事后分析_余量与可靠性_定义_20261111.md`）。只用 CPU。
用法：python headroom.py <repair_transfer 目录> <repair_budget 目录> [输出前缀]
"""
import collections, json, sys
from pathlib import Path
import numpy as np

RT, RB = Path(sys.argv[1]), Path(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else "headroom"
EPS, NB, TH = 1e-9, 5000, 0.20
NAME = {"q7": "Qwen2.5-7B", "q14": "Qwen2.5-14B", "l8": "Llama-3.1-8B", "2B": "Qwen3.5-2B", "4B": "Qwen3.5-4B"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_ds(d, models, key):
    D = json.loads((RT / d / "design.json").read_text(encoding="utf-8"))
    V = {}
    for m in models:
        V[m] = {r["branch_id"]: [float(x) for x in r[key]] for r in jl(RT / d / f"evals_{m}.jsonl") if r["condition"] == f"{m}_T07"}
        n = max(len(v) for v in V[m].values())
        for b, meta in D["branches"].items():           # 确定性分支（例如买下即结束）：没有续跑，值是已知的常数（与登记的分析相同）
            if meta.get("deterministic") is not None:
                V[m][b] = [float(meta["det_reward"] if key == "rewards" else meta["deterministic"])] * n
    trajs = []
    for t in D["trajectories"]:
        pos = []
        for p in t["positions"]:
            alts = [b for b in p["alts"] if all(b in V[m] for m in models)]          # 含随机候选（与登记的分析相同）
            if alts and all(p["orig"] in V[m] for m in models):
                pos.append({"orig": p["orig"], "alts": alts})
        if pos:
            trajs.append({"id": t["traj_id"], "cl": t["traj_id"], "pos": pos})
    return trajs, V


def load_tau():
    O = RT / "out_tau_pool"
    adm = {(r["pos_id"], r["cand_id"]): bool(r["ok"]) for r in jl(RB / "tau2_pilot/pool/adm_v3.jsonl")}
    best = {}
    for f in (RB / "tau2_pilot/pool/ro_pi0.jsonl", O / "ro_2b.jsonl", O / "ro_2b_fill.jsonl"):
        for r in jl(f):
            if r.get("score") != "orig_task" or r["temp"] == 0 or r["executor"] not in ("2B", "4B"):
                continue
            k = (r["executor"], r["pos_id"], r["cand_id"], r["k"])
            if r.get("reward") is not None or k not in best:
                best[k] = r
    R = collections.defaultdict(dict)
    for (m, pos, c, k), r in best.items():
        if r.get("reward") is not None:
            R[(m, f"{pos}|{c}")][k] = float(r["reward"])
    V = {m: {b: [v[k] for k in sorted(v)] for (mm, b), v in R.items() if mm == m} for m in ("2B", "4B")}
    by = collections.defaultdict(list)
    for d in jl(O / "design_p4sub.jsonl"):
        orig = [c["cand_id"] for c in d["cands"] if "orig" in c["kinds"]]
        alts = [f"{d['pos_id']}|{c['cand_id']}" for c in d["cands"] if c["cand_id"] != "free" and "orig" not in c["kinds"]
                and adm.get((d["pos_id"], c["cand_id"]))]
        alts = [b for b in alts if len(V["2B"].get(b, [])) == 10 and len(V["4B"].get(b, [])) >= 5]
        if orig and alts and len(V["2B"].get(f"{d['pos_id']}|{orig[0]}", [])) == 10:
            by[(d["sim_id"], f"{d['domain']}:{d['task_id']}")].append({"orig": f"{d['pos_id']}|{orig[0]}", "alts": alts})
    return [{"id": s, "cl": cl, "pos": p} for (s, cl), p in by.items()], V


POOLS = [("ALFWorld-7B", lambda: load_ds("out_ds_alfw", ["q7", "q14", "l8"], "results"), "q7", ["q14", "l8"]),
         ("ALFWorld-Llama", lambda: load_ds("out_ds_alfw_l8", ["l8", "q7"], "results"), "l8", ["q7"]),
         ("WebShop-A", lambda: load_ds("out_ws_A", ["q7", "q14", "l8"], "rewards"), "q7", ["q14", "l8"]),
         ("WebShop-B", lambda: load_ds("out_ws_B", ["l8", "q7"], "rewards"), "l8", ["q7"]),
         ("τ²", load_tau, "2B", ["4B"])]


def mean_set(vals, cands):
    m = max(vals[c] for c in cands)
    return [c for c in cands if vals[c] >= m - EPS]


def traj_stats(t, V, dep, sel):
    """两个方向各返回 (h, loss, pc)。"""
    out = []
    n = len(V[dep][t["pos"][0]["orig"]])
    halves = (list(range(n // 2)), list(range(n // 2, n)))
    cands = [(p["orig"], b) for p in t["pos"] for b in p["alts"]]
    oth = {b: np.mean(V[sel][b][:5]) - np.mean(V[sel][o][:5]) for o, b in cands}
    a_o = mean_set(oth, [b for _, b in cands])
    for S, E in (halves, halves[::-1]):
        net = lambda idx: {b: np.mean([V[dep][b][i] for i in idx]) - np.mean([V[dep][o][i] for i in idx]) for o, b in cands}
        nS, nE = net(S), net(E)
        a_s = mean_set(nS, list(nS))
        vs, vo, vr = np.mean([nE[b] for b in a_s]), np.mean([nE[b] for b in a_o]), np.mean(list(nE.values()))
        out.append((max(nS.values()) - np.mean(list(nS.values())), vs - vo, vs - vr))
    return out


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v; r[m] = r[m].mean()
    return r


def spear(x, y):
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


def boot(cl, fn, rng):
    by = collections.defaultdict(list)
    for i, c in enumerate(cl):
        by[c].append(i)
    names = sorted(by)
    est, vals = fn(np.arange(len(cl))), []
    for _ in range(NB):
        idx = np.array([i for j in rng.integers(0, len(names), len(names)) for i in by[names[j]]])
        v = fn(idx)
        if v == v:
            vals.append(v)
    return est, float(np.quantile(vals, .05)), float(np.quantile(vals, .95))


f3 = lambda t, s="": f"{t[0]:{s}.2f} [{t[1]:{s}.2f}, {t[2]:{s}.2f}]"
rng = np.random.default_rng(0)
res, L1, L2, L3 = {}, [], [], []
L1 += ["## 1. 比例表（简化重写；两个方向平均）", "", "| 池 | 执行者 ← 选择者 | 轨迹 | 损失 | 选择的价值 | 比例 [90%] |", "|---|---|---:|---:|---:|---|"]
L2 += ["## 2. 环境内部：余量与损失", "", "| 池 | 选择者 | Spearman(余量, 损失) [90%] | 损失：余量低 / 中 / 高三分之一 | 比例：余量低 / 中 / 高 |", "|---|---|---|---|---|"]
loaded = {}
for pool, loader, dep, sels in POOLS:
    trajs, V = loader()
    loaded[pool] = (trajs, V, dep)
    cl = [t["cl"] for t in trajs]
    for sel in sels:
        st = np.array([traj_stats(t, V, dep, sel) for t in trajs])        # (轨迹, 方向, 3)
        h, loss, pc = st[:, :, 0], st[:, :, 1], st[:, :, 2]
        ratio = boot(cl, lambda i: loss[i].sum() / pc[i].sum(), rng)
        rho = boot(cl, lambda i: np.mean([spear(h[i, d], loss[i, d]) for d in (0, 1)]), rng)
        ter_l, ter_r = [], []
        for g in range(3):
            num = den = 0.0; ls = []
            for d in (0, 1):
                o = np.argsort(h[:, d], kind="mergesort"); k = len(o) // 3
                idx = o[g * k:(g + 1) * k] if g < 2 else o[2 * k:]
                ls.append(loss[idx, d].mean()); num += loss[idx, d].sum(); den += pc[idx, d].sum()
            ter_l.append(float(np.mean(ls))); ter_r.append(float(num / den) if abs(den) > 1e-9 else float("nan"))
        res[f"{pool}|{sel}"] = {"n": len(trajs), "loss": float(loss.mean()), "pc": float(pc.mean()), "ratio": ratio, "rho": rho,
                                "tercile_loss": ter_l, "tercile_ratio": ter_r, "mean_h": float(h.mean())}
        L1.append(f"| {pool} | {NAME[dep]} ← {NAME[sel]} | {len(trajs)} | {loss.mean():+.3f} | {pc.mean():+.3f} | {f3(ratio)} |")
        L2.append(f"| {pool} | {NAME[sel]} | {f3(rho, '+')} | " + " / ".join(f"{x:+.3f}" for x in ter_l) + " | " + " / ".join(f"{x:.2f}" for x in ter_r) + " |")

# ---------- 3. 标签可重复率 ----------
def reliab(trajs, V, m, h, rng, draws):
    pos = [p for t in trajs for p in t["pos"] if all(len(V[m].get(b, [])) >= 2 * h for b in [p["orig"]] + p["alts"])]
    n = min(len(V[m][p["orig"]]) for p in pos) // 2 * 2
    num = den = 0; gains = []
    for _ in range(draws):
        ia = sorted(rng.choice(n // 2, h, replace=False)); ib = sorted(n // 2 + rng.choice(n // 2, h, replace=False))
        for p in pos:
            G, arg = [], []
            for idx in (ia, ib):
                g = {b: np.mean([V[m][b][i] for i in idx]) - np.mean([V[m][p["orig"]][i] for i in idx]) for b in p["alts"]}
                G.append(g); arg.append(mean_set(g, p["alts"]))
            for a, b in ((0, 1), (1, 0)):
                if max(G[a].values()) >= TH - EPS:
                    den += 1; num += max(G[b].values()) < TH - EPS
                    gains.append(np.mean([G[b][c] for c in arg[a]]))
    return (num / den if den else float("nan")), (float(np.mean(gains)) if gains else float("nan")), den / draws / 2, len(pos), \
        float(np.mean([np.mean(V[m][p["orig"]]) for p in pos]))


L3 += ["## 3. 标签可重复率（位置层面；门槛 0.20）", "", "| 池 | 模型 | 位置 | 原动作分支的平均成功率 / 奖励 | N0（每半 3 次） | 保留增益（每半 3 次） | N0（每半取满） | 保留增益（取满） |", "|---|---|---:|---:|---:|---:|---:|---:|"]
for pool, (trajs, V, dep) in loaded.items():
    for m in ([dep, "4B"] if pool == "τ²" else [dep]):
        full = min(len(V[m][t["pos"][0]["orig"]]) for t in trajs) // 2
        a = reliab(trajs, V, m, 3, rng, 1 if full == 3 else 200)
        b = reliab(trajs, V, m, full, rng, 1)
        res[f"reliab|{pool}|{m}"] = {"n_pos": a[3], "orig": a[4], "N0_h3": a[0], "gain_h3": a[1], "N0_full": b[0], "gain_full": b[1], "full_h": full, "n_sel_full": b[2]}
        L3.append(f"| {pool} | {NAME[m]} | {a[3]} | {a[4]:.3f} | {a[0]:.2f} | {a[1]:+.2f} | {b[0]:.2f}（每半 {full} 次） | {b[1]:+.2f} |")

# ---------- 4. 下游训练的可行性（ALFWorld-7B 池） ----------
trajs, V, dep = loaded["ALFWorld-7B"]
pos = [(t["id"], p) for t in trajs for p in t["pos"]]
def labels(m, n):
    out = {}
    for i, (tid, p) in enumerate(pos):
        g = {b: np.mean(V[m][b][:n]) - np.mean(V[m][p["orig"]][:n]) for b in p["alts"]}
        if max(g.values()) >= TH - EPS:
            out[i] = mean_set(g, p["alts"])
    return out
def gain7(i, acts, idx):
    p = pos[i][1]
    return float(np.mean([np.mean([V["q7"][b][k] for k in idx]) - np.mean([V["q7"][p["orig"]][k] for k in idx]) for b in acts]))
own = labels("q7", 10)
# 7B 自己的标签的诚实增益：前 5 次选（且过门槛）、后 5 次评，再对调
g_own = []
for S, E in ((range(5), range(5, 10)), (range(5, 10), range(5))):
    for i, (tid, p) in enumerate(pos):
        g = {b: np.mean([V["q7"][b][k] for k in S]) - np.mean([V["q7"][p["orig"]][k] for k in S]) for b in p["alts"]}
        if max(g.values()) >= TH - EPS:
            g_own.append(gain7(i, mean_set(g, p["alts"]), E))
L4 = ["## 4. 下游训练的可行性（ALFWorld-7B 池；标签 = 选择者自己的续跑里 G ≥ 0.20 的位置及其最好候选）", "",
      f"位置 {len(pos)} 个、轨迹 {len(trajs)} 条。7B 自己（10 次）的标签 {len(own)} 个；在 7B 上的诚实增益（5 选 5 评）{np.mean(g_own):+.3f}，≤ 0 的比例 {np.mean(np.array(g_own) <= EPS):.2f}。", "",
      "| 选择者 | 标签数 | 位置也在 7B 标签里 | 同位置且同动作（占该选择者的标签） | 在 7B 上的平均增益 | 增益 ≤ 0 的比例 | 7B 有标签而它没有的位置 |", "|---|---:|---:|---:|---:|---:|---:|"]
res["train"] = {"n_pos": len(pos), "own": len(own), "own_gain": float(np.mean(g_own))}
for m, n in (("q14", 20), ("q14", 10), ("l8", 10)):
    lab = labels(m, n)
    same_pos = [i for i in lab if i in own]
    same_act = [i for i in same_pos if set(lab[i]) & set(own[i])]
    g = [gain7(i, lab[i], range(10)) for i in lab]
    res["train"][f"{m}_{n}"] = {"n": len(lab), "same_pos": len(same_pos), "same_act": len(same_act), "gain7": float(np.mean(g)), "le0": float(np.mean(np.array(g) <= EPS)), "missed": len(set(own) - set(lab))}
    L4.append(f"| {NAME[m]}（{n} 次） | {len(lab)} | {len(same_pos) / len(lab):.2f} | {len(same_act) / len(lab):.2f} | {np.mean(g):+.3f} | {np.mean(np.array(g) <= EPS):.2f} | {len(set(own) - set(lab))} |")

# ---------- 4b. 追加（看过第 4 项之后加的，定义文件里没有）：其余池上同样的标签重合度 ----------
L4 += ["", "**追加（看过上表之后加的）：其余池**", "", "| 池 | 执行者 ← 选择者 | 执行者自己的标签 | 选择者的标签 | 同位置且同动作 | 选择者标签在执行者上的增益 | 执行者自己标签的诚实增益 |", "|---|---|---:|---:|---:|---:|---:|"]
for pool, (tr, VV, dp) in loaded.items():
    ps = [p for t in tr for p in t["pos"]]
    nD = min(len(VV[dp][p["orig"]]) for p in ps)
    G = lambda m, p, idx: {b: np.mean([VV[m][b][k] for k in idx]) - np.mean([VV[m][p["orig"]][k] for k in idx]) for b in p["alts"]}
    def lab(m, n):
        o = {}
        for i, p in enumerate(ps):
            g = G(m, p, range(n))
            if max(g.values()) >= TH - EPS:
                o[i] = mean_set(g, p["alts"])
        return o
    mine = lab(dp, nD)
    hon = []
    for S, E in ((range(nD // 2), range(nD // 2, nD)), (range(nD // 2, nD), range(nD // 2))):
        for p in ps:
            g = G(dp, p, S)
            if max(g.values()) >= TH - EPS:
                ge = G(dp, p, E); hon.append(np.mean([ge[b] for b in mean_set(g, p["alts"])]))
    for m in [x for x in VV if x != dp]:
        n = min(len(VV[m][p["orig"]]) for p in ps)
        lb = lab(m, min(n, 10))
        sa = [i for i in lb if i in mine and set(lb[i]) & set(mine[i])]
        gg = [np.mean([G(dp, ps[i], range(nD))[b] for b in lb[i]]) for i in lb]
        res[f"overlap|{pool}|{m}"] = {"own": len(mine), "n": len(lb), "same_act": len(sa), "gain": float(np.mean(gg)), "own_gain": float(np.mean(hon))}
        L4.append(f"| {pool} | {NAME[dp]} ← {NAME[m]} | {len(mine)} | {len(lb)} | {len(sa) / len(lb):.2f} | {np.mean(gg):+.3f} | {np.mean(hon):+.3f} |")

L = ["# 事后分析：余量、错配损失与标签可靠性（headroom.py；事后、探索性）", ""] + L1 + [""] + L2 + [""] + L3 + [""] + L4
open(OUT + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
json.dump(res, open(OUT + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n".join(L))
