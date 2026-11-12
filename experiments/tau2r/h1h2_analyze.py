"""第 4 篇：策略差异与选修复损失（τ²）的 H1 / H2 分析。登记：`登记_策略差异与损失_20261111.md`。只用 CPU。
本文件在打开 2B 续跑结果之前冻结（哈希记在登记里）；之后不改。

    python h1h2_analyze.py check   ...   # 只核对结构与齐全（不读 reward 的取值，只数有没有）
    python h1h2_analyze.py report  ...   # 读数与判定，写 <out>.md / <out>.json
    python h1h2_analyze.py selftest      # 合成数据：植入信号时 ρ > 0、无信号时 ρ ≈ 0

量的定义（登记第 3 节）：
  动作键（主）：工具调用 = 规范化的"工具名(参数)"（整串转小写）；文字回复 = 27B 盲判的类别 A–G（classify_drift.py）。
  D_H1 = ½ Σ_s TV(A^s, B^s) − ½ [TV(A^1, A^2) + TV(B^1, B^2)]，A = 4B、B = 2B；每个模型 4 组 × 10 个样本，半 1 = 第 1、2 组，半 2 = 第 3、4 组。
  D_H2 = 1 − 2B 的 40 个样本中动作键等于贪心动作 g 的比例。
  L：分支 = 原动作 + v3 可执行的替代候选。2B 每分支 10 次随机分成 5 选 5 评，重复 50 次（种子 sha256("H1H2|" + 位置)）。
     每次 L_s = 评价折(a*_2B) − 评价折(a*_A)；a*_2B = 选择折均值最高的分支；a*_A：H1 用 4B 全部续跑的均值最高者，H2 用 2B 贪心那 1 次结果最高者。
     并列时取并列分支的平均（等于均匀随机打破平局的期望）。
  检验：Spearman ρ(D, L) > 0；按任务成簇、按领域分层的 bootstrap 10,000 次，p = (#{ρ* ≤ 0} + 1) / (B + 1)；Holm m = 2，单侧 α = 0.05。
"""
import argparse, collections, hashlib, itertools, json, math, sys
from pathlib import Path
import numpy as np

EPS, N_SPLIT, N_BOOT = 1e-9, 50, 10000
MIN_2B, MIN_4B, MIN_HALF = 8, 5, 15


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


# ---------- 动作键 ----------
def key_main(canon, cls):
    if canon.startswith("T:"):
        return "T:" + canon[2:].lower()
    if canon.startswith("X:"):
        return "X:" + cls.get(sha(canon[2:]), "?")
    return "INVALID"


def key_text1(canon, cls):
    return "T:" + canon[2:].lower() if canon.startswith("T:") else ("TEXT" if canon.startswith("X:") else "INVALID")


def key_tool(canon, cls):
    if canon.startswith("T:"):
        return "T:" + ";".join(c.split("(", 1)[0].strip() for c in canon[2:].split(";") if "(" in c)
    return "TEXT" if canon.startswith("X:") else "INVALID"


KEYS = {"main": key_main, "text1": key_text1, "tool": key_tool}


def tv(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))


def entropy(a):
    c = collections.Counter(a)
    return -sum(v / len(a) * math.log(v / len(a)) for v in c.values())


# ---------- 读数据 ----------
def load_first(paths, cls):
    """{(pos, model): {rep: [canon, …]}}"""
    F = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in jl_many(paths):
        F[(r["pos_id"], r["model"])][str(r["rep"])].append(r["canon"])
    return F


def jl_many(paths):
    return [r for p in paths for r in jl(p)]


def load_ro(paths, executor, greedy=False):
    best = {}
    for r in jl_many(paths):
        if r["executor"] != executor or r.get("score") != "orig_task" or (r["temp"] == 0) != greedy:
            continue
        k = (r["pos_id"], r["cand_id"], r["k"])
        if r.get("reward") is not None or k not in best:
            best[k] = r
    R = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (pos, c, k), r in best.items():
        if r.get("reward") is not None:
            R[pos][c][k] = float(r["reward"])
    return R


def build(a):
    cls = {r["h"]: r["cat"] for r in jl_many(a.cls)} if a.cls else {}
    adm = {(r["pos_id"], r["cand_id"]): bool(r["ok"]) for r in jl(a.adm)}
    F = load_first(a.first, cls)
    G = load_first(a.greedy_first, cls) if a.greedy_first else {}
    R2, Rg, R4 = load_ro(a.ro2b, "2B"), load_ro(a.ro2b_greedy, "2B", greedy=True), load_ro(a.ro4b, "4B")
    P, why = [], collections.Counter()
    for d in jl(a.design):
        pid = d["pos_id"]
        br = [c["cand_id"] for c in d["cands"] if c["cand_id"] != "free" and "random" not in c["kinds"]
              and ("orig" in c["kinds"] or adm.get((pid, c["cand_id"]), False))]
        orig = [c["cand_id"] for c in d["cands"] if "orig" in c["kinds"]]
        if not orig or len(br) < 2:
            why["没有原动作或没有 v3 可执行的替代候选"] += 1
            continue
        if any(len(R2[pid].get(b, {})) < MIN_2B for b in br):
            why["2B 续跑不齐"] += 1
            continue
        if any(len(R4[pid].get(b, {})) < MIN_4B for b in br):
            why["4B 续跑不齐"] += 1
            continue
        s2, s4 = F.get((pid, "2B"), {}), F.get((pid, "4B"), {})
        h = lambda s: (s.get("1", []) + s.get("2", []), s.get("3", []) + s.get("4", []))
        if min(map(len, h(s2) + h(s4))) < MIN_HALF:
            why["第一步样本不足"] += 1
            continue
        g = (G.get((pid, "2B"), {}).get("g") or [None])[0]
        P.append({"pos": pid, "cl": f"{d['domain']}:{d['task_id']}", "dom": d["domain"], "br": br,
                  "r2": {b: [R2[pid][b][k] for k in sorted(R2[pid][b])] for b in br},
                  "r4": {b: float(np.mean(list(R4[pid][b].values()))) for b in br},
                  "rg": {b: (list(Rg[pid][b].values()) or [None])[0] if b in Rg[pid] else None for b in br},
                  "f2": h(s2), "f4": h(s4), "g": g})
    return P, cls, why


# ---------- 每个位置的量 ----------
def argmax_set(vals):
    m = max(vals.values())
    return [b for b, v in vals.items() if v >= m - EPS]


def losses(p, sel_idx_fn, eval_idx_fn, a_star, n_split, seed):
    """a_star：打标签者选出的分支集合（并列取平均）。返回平均 L 与选择折上分支均值的极差。"""
    rng = np.random.default_rng(int(sha(seed)[:12], 16))
    Ls, rg = [], []
    for _ in range(n_split):
        sel, ev = {}, {}
        for b in p["br"]:
            x = np.array(p["r2"][b])
            perm = rng.permutation(len(x))
            s, e = sel_idx_fn(perm), eval_idx_fn(perm)
            sel[b], ev[b] = x[s].mean(), x[e].mean()
        own = argmax_set(sel)
        Ls.append(np.mean([ev[b] for b in own]) - np.mean([ev[b] for b in a_star]))
        rg.append(max(sel.values()) - min(sel.values()))
    return float(np.mean(Ls)), float(np.mean(rg))


def block_L(p, a_star, block):
    """可靠性：只用一半续跑（block = 0 → 第 0–4 次，1 → 第 5–9 次），3 选 2 评的全部 10 种划分。"""
    Ls = []
    idx = lambda x: list(range(len(x)))[: len(x) // 2] if block == 0 else list(range(len(x)))[len(x) // 2:]
    n = min(len(idx(p["r2"][b])) for b in p["br"])
    for s in itertools.combinations(range(n), max(1, n * 3 // 5)):
        e = [i for i in range(n) if i not in s]
        if not e:
            continue
        sel = {b: np.mean([p["r2"][b][idx(p["r2"][b])[i]] for i in s]) for b in p["br"]}
        ev = {b: np.mean([p["r2"][b][idx(p["r2"][b])[i]] for i in e]) for b in p["br"]}
        Ls.append(np.mean([ev[b] for b in argmax_set(sel)]) - np.mean([ev[b] for b in a_star]))
    return float(np.mean(Ls))


def position_rows(P, cls):
    rows = []
    for p in P:
        half = lambda perm: perm[: len(perm) // 2]
        rest = lambda perm: perm[len(perm) // 2:]
        a4 = argmax_set(p["r4"])
        r = {"pos": p["pos"], "cl": p["cl"], "dom": p["dom"], "n_br": len(p["br"])}
        r["L_H1"], r["range2B"] = losses(p, half, rest, a4, N_SPLIT, "H1H2|" + p["pos"])
        r["L_H1_a"], r["L_H1_b"] = block_L(p, a4, 0), block_L(p, a4, 1)
        r["tie4B"] = len(a4) / len(p["br"])
        if all(v is not None for v in p["rg"].values()) and p["g"] is not None:
            ag = argmax_set(p["rg"])
            r["L_H2"], _ = losses(p, half, rest, ag, N_SPLIT, "H1H2|" + p["pos"])
            r["L_H2_a"], r["L_H2_b"] = block_L(p, ag, 0), block_L(p, ag, 1)
        for name, kf in KEYS.items():
            k2 = [[kf(c, cls) for c in h] for h in p["f2"]]
            k4 = [[kf(c, cls) for c in h] for h in p["f4"]]
            cross = 0.5 * (tv(k4[0], k2[0]) + tv(k4[1], k2[1]))
            floor = 0.5 * (tv(k4[0], k4[1]) + tv(k2[0], k2[1]))
            r[f"D1_{name}"], r[f"D1raw_{name}"] = cross - floor, cross
            if p["g"] is not None:
                allk = k2[0] + k2[1]
                r[f"D2_{name}"] = 1 - sum(k == kf(p["g"], cls) for k in allk) / len(allk)
        r["H2B"] = entropy([key_main(c, cls) for h in p["f2"] for c in h])
        rows.append(r)
    return rows


# ---------- 统计 ----------
def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y, z=None):
    if len(x) < 4:
        return float("nan")
    rx, ry = rank(x), rank(y)
    if z is not None:
        Z = np.column_stack([np.ones(len(x)), rank(z)])
        rx = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ry = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    sx, sy = rx.std(), ry.std()
    return float(np.corrcoef(rx, ry)[0, 1]) if sx > 0 and sy > 0 else float("nan")


def tercile_gap(d, l):
    o = np.argsort(d, kind="mergesort")
    n = len(o) // 3
    return float(np.mean(np.asarray(l)[o[-n:]]) - np.mean(np.asarray(l)[o[:n]])) if n else float("nan")


def boot(rows, fn, n_boot=N_BOOT, seed=0):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, r in enumerate(rows):
        by[r["dom"]][r["cl"]].append(i)
    rng = np.random.default_rng(seed)
    est, vals = fn(list(range(len(rows)))), []
    for _ in range(n_boot):
        idx = []
        for dom, cl in by.items():
            names = sorted(cl)
            for j in rng.integers(0, len(names), len(names)):
                idx += cl[names[j]]
        v = fn(idx)
        if v == v:
            vals.append(v)
    vals = np.array(vals)
    if not len(vals):                                           # 统计量恒为 nan（例如 D 没有任何变化）
        return {"est": est, "lo": float("nan"), "hi": float("nan"), "p": 1.0, "n": len(rows), "G": sum(len(c) for c in by.values())}
    return {"est": est, "lo": float(np.quantile(vals, .05)), "hi": float(np.quantile(vals, .95)),
            "p": float((np.sum(vals <= 0) + 1) / (len(vals) + 1)), "n": len(rows), "G": sum(len(c) for c in by.values())}


def analyze(rows, n_boot=N_BOOT):
    col = lambda rs, k: np.array([r[k] for r in rs], float)
    out = {}

    def one(tag, rs, dk, lk):
        f = lambda z=None, d=dk: (lambda idx: spearman(col(rs, d)[idx], col(rs, lk)[idx], None if z is None else col(rs, z)[idx]))
        o = {"rho": boot(rs, f(), n_boot)}
        o["tercile_gap"] = boot(rs, lambda idx: tercile_gap(col(rs, dk)[idx], col(rs, lk)[idx]), n_boot)
        o["a_entropy"] = boot(rs, f("H2B"), n_boot)
        o["d_range"] = boot(rs, f("range2B"), n_boot)
        pre = dk.split("_")[0]
        o["b_toolname"] = boot(rs, f(None, f"{pre}_tool"), n_boot)
        o["e_text_one_class"] = boot(rs, f(None, f"{pre}_text1"), n_boot)
        if pre == "D1":
            o["c_no_floor"] = boot(rs, f(None, "D1raw_main"), n_boot)
        o["L_reliability"] = spearman(col(rs, lk + "_a"), col(rs, lk + "_b"))
        o["mean_L"], o["mean_D"] = float(col(rs, lk).mean()), float(col(rs, dk).mean())
        o["share_D_le0"] = float(np.mean(col(rs, dk) <= EPS))
        out[tag] = o
    one("H1", rows, "D1_main", "L_H1")
    r2 = [r for r in rows if "L_H2" in r]
    if len(r2) >= 4:
        one("H2", r2, "D2_main", "L_H2")
    ps = sorted((out[t]["rho"]["p"], t) for t in out)
    m, prev = 2, 0.0
    for i, (p, t) in enumerate(ps):
        adj = max(prev, min(1.0, p * (m - i)))
        out[t]["p_holm"], prev = adj, adj
        out[t]["confirmed"] = bool(adj < 0.05)
    return out


def write_report(out, why, path):
    f = lambda b: f"{b['est']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]"
    L = ["# H1 / H2：位置层面的策略差异与选修复损失（h1h2_analyze.py）", "", f"排除的位置：{dict(why)}", ""]
    for t, o in out.items():
        r = o["rho"]
        L += [f"## {t}（{r['n']} 个位置、{r['G']} 个任务）", "",
              f"- **Spearman ρ(D, L)** {f(r)}；单侧 p = {r['p']:.4f}，Holm 后 {o['p_holm']:.4f} → **{'成立' if o['confirmed'] else '不成立'}**",
              f"- 效应大小（D 最高三分之一 − 最低三分之一的平均 L）：{f(o['tercile_gap'])}",
              f"- 平均 L {o['mean_L']:+.3f}；平均 D {o['mean_D']:+.3f}；D ≤ 0 的位置比例 {o['share_D_le0']:.2f}",
              f"- **L 的可靠性**（两半续跑各自算出的 L 的 Spearman）：{o['L_reliability']:+.3f}" + ("" if o["L_reliability"] >= 0.20 else "　← 低于 0.20：阴性结果不解读"),
              f"- (a) 控制 2B 第一步的熵：{f(o['a_entropy'])}", f"- (b) 只按工具名：{f(o['b_toolname'])}"]
        if "c_no_floor" in o:
            L.append(f"- (c) 不减噪声底：{f(o['c_no_floor'])}")
        L += [f"- (d) 控制 2B 选择折上分支均值的极差：{f(o['d_range'])}", f"- (e) 文字回复合成一类（草稿的原定义）：{f(o['e_text_one_class'])}", ""]
    Path(path + ".md").write_text("\n".join(L), encoding="utf-8")
    Path(path + ".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


def cmd_check(a):
    P, cls, why = build(a)
    n_txt = sum(c.startswith("X:") for p in P for h in p["f2"] + p["f4"] for c in h)
    n_unc = sum(c.startswith("X:") and sha(c[2:]) not in cls for p in P for h in p["f2"] + p["f4"] for c in h)
    print(f"合格位置 {len(P)}，任务 {len({p['cl'] for p in P})}；排除 {dict(why)}")
    print(f"有贪心动作与贪心续跑、可进 H2 的位置 {sum(p['g'] is not None and all(v is not None for v in p['rg'].values()) for p in P)}")
    print(f"文字样本 {n_txt}，其中没有类别的 {n_unc}；每位置分支数 {sorted(collections.Counter(len(p['br']) for p in P).items())}")
    return 0


def cmd_report(a):
    P, cls, why = build(a)
    rows = position_rows(P, cls)
    Path(a.out + ".positions.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    write_report(analyze(rows), why, a.out)
    return 0


def cmd_selftest(a):
    for signal in (0.0, 1.0):
        rng = np.random.default_rng(1)
        rows_P = []
        for i in range(200):
            d = rng.uniform()                                   # 真实差异
            nb = 4
            v4 = rng.uniform(0.1, 0.9, nb)
            v2 = v4.copy()
            if rng.uniform() < signal * d:                      # 差异越大，2B 的最好分支越可能换成 4B 眼里最差的
                v2 = v2[np.argsort(np.argsort(-v4))]            # 逆序
            br = [f"c{j}" for j in range(nb)]
            acts4 = ["T:a({})"] * 20
            acts2 = [("T:b({})" if rng.uniform() < d else "T:a({})") for _ in range(40)]
            rows_P.append({"pos": f"p{i}", "cl": f"retail:{i % 60}", "dom": "retail", "br": br,
                           "r2": {b: list((rng.uniform(size=10) < v2[j]).astype(float)) for j, b in enumerate(br)},
                           "r4": {b: float((rng.uniform(size=6) < v4[j]).mean()) for j, b in enumerate(br)},
                           "rg": {b: float(rng.uniform() < v2[j]) for j, b in enumerate(br)},
                           "f2": (acts2[:20], acts2[20:]), "f4": (acts4, acts4), "g": "T:a({})"})
        out = analyze(position_rows(rows_P, {}), n_boot=400)
        print(f"信号 {signal}: H1 ρ = {out['H1']['rho']['est']:+.3f} (p {out['H1']['rho']['p']:.3f})，L 可靠性 {out['H1']['L_reliability']:+.2f}；"
              f"H2 ρ = {out['H2']['rho']['est']:+.3f}")
        assert (out["H1"]["rho"]["est"] > 0.15) == (signal > 0), "自检失败"
    print("== 自检通过")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "report"):
        s = sub.add_parser(name)
        s.add_argument("--design", required=True)
        s.add_argument("--adm", required=True, help="adm_v3.jsonl")
        s.add_argument("--ro2b", nargs="+", required=True)
        s.add_argument("--ro2b-greedy", nargs="+", required=True)
        s.add_argument("--ro4b", nargs="+", required=True)
        s.add_argument("--first", nargs="+", required=True, help="第一步样本：2B 第 1–4 组、4B 第 1–4 组")
        s.add_argument("--greedy-first", nargs="+", help="2B 温度 0 的第一步（rep = g）")
        s.add_argument("--cls", nargs="+", help="classify_drift.py 的类别文件")
        s.add_argument("--out", default="report_h1h2")
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"check": cmd_check, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
