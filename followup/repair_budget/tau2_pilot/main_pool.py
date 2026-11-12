"""τ² 主实验的建池与分组（2026-11-05 写；第 6 篇；登记 `登记_τ²主实验_20261105.md` 第 3–4 节）。

子命令（都在 tau2_pilot 下运行；select / branches / w0 要 τ² 的 venv，只用 CPU）：
  folds     两折：每个领域内按 sha256("FOLD|MAIN|<领域>|<任务 id>") 排序，前一半为 A。写 pool/folds.json 与 pool/tasks_<折>_<领域>.txt。
  select    从 2B 的对话里选失败（每任务最多 --per-task 条，pool_tau2.select_failures 的随机顺序）、每条早 / 中 / 晚各取 1 个位置
            （pool_tau2.positions，与 design 同一种子），全部候选位置按 sha256("POS800|<位置 id>") 排序取前 --n 个。
            写 pool/positions.jsonl（每行：pos_id, src, sim_id, task_id, domain, t, fold）。
  branches  由设计文件与 v2 判定写出续跑用的设计文件：保留自由选择、"只有原动作来源"的候选（不论判定，第 4 篇要）、v2 可执行的其余候选。
            另写每折一份（export 用）。
  w0        π0 续跑完成后、π1 开跑之前：算 π0 值得修的位置（交叉拟合，v2 / v3 各一次）、一半内部值得修的位置（第 6.3 节检验要用）、
            核对样本、反复报错的位置（只用 π0 的终止原因），并写 π1 的续跑设计文件。不读任何 π1 数据。
正常日志：每个子命令最后一行 "== 完成"；select 打印可得位置数、两折与领域的分布。
"""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
THR = 0.20


def h(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest(), 16)


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def wjl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_ids(p):
    return [x for x in Path(p).read_text(encoding="utf-8").replace(",", " ").split() if x]


# ---------------------------------------------------------------- folds
def cmd_folds(a):
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    out = {}
    for dom, f in (("retail", a.ids_retail), ("airline", a.ids_airline)):
        ids = sorted(read_ids(f), key=lambda t: h("FOLD", "MAIN", dom, t))
        half = len(ids) // 2
        for i, t in enumerate(ids):
            out[f"{dom}:{t}"] = "A" if i < half else "B"
        for fold in "AB":
            Path(a.out_dir, f"tasks_{fold}_{dom}.txt").write_text(
                "".join(t + "\n" for t in ids if out[f"{dom}:{t}"] == fold), encoding="utf-8")
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.out_dir, "folds.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    c = collections.Counter((k.split(":")[0], v) for k, v in out.items())
    print("两折：" + "；".join(f"{d} {f} {n}" for (d, f), n in sorted(c.items())))
    s2 = "A" if h("SEED2", "FOLD", "MAIN") % 2 == 0 else "B"
    print(f"种子 2 的折 = {s2}（登记第 3.2 节写的是 A）")
    print("== 完成" if s2 == "A" else "== 不一致，停下")
    return 0 if s2 == "A" else 1


# ---------------------------------------------------------------- select
def cmd_select(a):
    sys.path.insert(0, str(HERE))
    import pool_tau2 as P
    from tau2.data_model.simulation import Results
    folds = json.loads(Path(a.folds).read_text(encoding="utf-8"))
    rows, n_fail, n_none = [], 0, 0
    for dom, srcs in (("retail", a.src_retail), ("airline", a.src_airline)):
        rl = [(s, Results.load(Path(s))) for s in srcs]
        tasks = {t.id: t for _, R in rl for t in R.tasks}
        fails, n_pool = P.select_failures(rl, 10 ** 9, a.per_task, False)
        n_fail += len(fails)
        print(f"{dom}：符合条件的失败 {n_pool} 条，每任务最多 {a.per_task} 条 → {len(fails)} 条")
        for src, s in fails:
            prng = random.Random(P.sseed("POS", s.task_id, s.trial, s.id))   # 与 pool_tau2.cmd_design 相同
            idx, _ = P.cand_positions(s, tasks[s.task_id], dom, "before_bad", 0.8)
            n_none += not idx
            for t in P.positions(idx, prng):
                rows.append({"pos_id": f"{s.id[:12]}_{t}", "src": src, "sim_id": s.id, "task_id": s.task_id, "domain": dom,
                             "t": t, "fold": folds[f"{dom}:{s.task_id}"]})
    ids = [r["pos_id"] for r in rows]
    dup = [k for k, v in collections.Counter(ids).items() if v > 1]
    if dup:
        print(f"[停下] 位置 id 重复：{dup[:5]}")
        return 1
    rows.sort(key=lambda r: h("POS800", r["pos_id"]))
    print(f"可得位置 {len(rows)} 个（失败 {n_fail} 条；没有可选位置的失败 {n_none} 条）；取前 {min(a.n, len(rows))} 个")
    if len(rows) < a.n:
        print(f"注意：可得位置少于 {a.n}，按登记第 3.3 节全部用上、不补收")
    keep = rows[: a.n]
    wjl(a.out, keep)
    c = collections.Counter((r["fold"], r["domain"]) for r in keep)
    tk = collections.Counter(r["fold"] for r in {(r["fold"], r["domain"], r["task_id"]): r for r in keep}.values())
    print("位置：" + "；".join(f"{f} {d} {n}" for (f, d), n in sorted(c.items())) + f"；有位置的任务 A {tk['A']}、B {tk['B']}")
    print("== 完成")
    return 0


# ---------------------------------------------------------------- branches
def is_repair(c):
    return "free" not in c["kinds"] and c["kinds"] != ["orig"]


def cmd_branches(a):
    V = {(r["pos_id"], r["cand_id"]): bool(r["ok"]) for r in jl(a.adm_v2)}
    pos = {r["pos_id"]: r for r in jl(a.positions)}
    design = [d for f in a.design for d in jl(f)]
    out, n_drop, n_miss = [], collections.Counter(), []
    for d in design:
        if d["pos_id"] not in pos:
            continue
        keep = []
        for c in d["cands"]:
            if c["cand_id"] == "free" or c["kinds"] == ["orig"]:
                keep.append(c)
                continue
            k = (d["pos_id"], c["cand_id"])
            if c.get("restore_ok") and k not in V:
                n_miss.append(k)
                continue
            if V.get(k, False):
                keep.append(c)
            else:
                n_drop["+".join(c["kinds"])] += 1
        out.append(dict(d, cands=keep, fold=pos[d["pos_id"]]["fold"]))
    missing_pos = sorted(set(pos) - {d["pos_id"] for d in out})
    if n_miss or missing_pos:
        print(f"[停下] v2 判定缺 {len(n_miss)} 个候选（例 {n_miss[:3]}）；设计文件缺 {len(missing_pos)} 个位置（例 {missing_pos[:3]}）")
        return 1
    out.sort(key=lambda d: d["pos_id"])
    wjl(a.out, out)
    for fold in "AB":
        wjl(str(a.out).replace(".jsonl", f"_{fold}.jsonl"), [d for d in out if d["fold"] == fold])
    nb = sum(sum(1 for c in d["cands"] if c.get("restore_ok")) for d in out)
    print(f"位置 {len(out)} 个，续跑分支 {nb} 个（每位置 {nb / max(1, len(out)):.2f}，含自由选择与原动作）；按 v2 去掉的候选 {dict(n_drop)}")
    print("== 完成")
    return 0


# ---------------------------------------------------------------- w0
def load_rates(paths, model):
    """{pos: {cand: {k: reward}}}，只取 executor == model、score == orig_task、reward 不为空的行。"""
    best = {}
    for p in paths:
        for r in jl(p):
            if r["executor"] != model or r.get("score") != "orig_task" or r["temp"] == 0:
                continue
            key = (r["pos_id"], r["cand_id"], r["k"])
            if r.get("reward") is not None or key not in best:
                best[key] = r
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (pos, cand, k), r in best.items():
        if r.get("reward") is not None:
            out[pos][cand][k] = float(r["reward"])
    return out


H = [(0, 1, 2), (3, 4, 5)]


def p_on(rr, c, ks):
    v = [rr.get(c, {}).get(k) for k in ks]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def sseed(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def argbest(rr, cands, ks, pos):
    sc = [(p_on(rr, c, ks), sseed("TIE", pos, c), c) for c in cands]   # 并列打破与 probe_analyze / train_tau2 相同
    sc = [x for x in sc if x[0] is not None]
    return max(sc)[2] if sc else None


def g_cross(rr, reps, pos):
    g = []
    for s, e in ((H[0], H[1]), (H[1], H[0])):
        b = argbest(rr, reps, s, pos)
        pb, pf = p_on(rr, b, e), p_on(rr, "free", e)
        if b is not None and pb is not None and pf is not None:
            g.append(pb - pf)
    return sum(g) / len(g) if g else None


def g_half(rr, reps, hh):
    vals = [p_on(rr, c, hh) for c in reps]
    vals = [v for v in vals if v is not None]
    pf = p_on(rr, "free", hh)
    return None if not vals or pf is None else max(vals) - pf


def repair_sets(d, A2, A3):
    reps = [c for c in d["cands"] if c.get("restore_ok") and is_repair(c)]
    return ([c["cand_id"] for c in reps if A2.get((d["pos_id"], c["cand_id"]))],
            [c["cand_id"] for c in reps if A3.get((d["pos_id"], c["cand_id"]))])


def cmd_w0(a):
    design = jl(a.design)
    A2 = {(r["pos_id"], r["cand_id"]): bool(r["ok"]) for r in jl(a.adm_v2)}
    A3 = {(r["pos_id"], r["cand_id"]): bool(r["ok"]) for r in jl(a.adm_v3)}
    R = load_rates(a.rollouts, a.pi0)
    incomplete, info = [], {}
    term = collections.defaultdict(lambda: [0, 0])
    for p in a.rollouts:
        for r in jl(p):
            if r["executor"] == a.pi0 and r.get("score") == "orig_task":
                bad = r.get("reward") is None or "TOO_MANY" in r["termination"] or "MAX_STEPS" in r["termination"]
                term[r["pos_id"]][0] += bad
                term[r["pos_id"]][1] += 1
    for d in design:
        pos, rr = d["pos_id"], R.get(d["pos_id"], {})
        runs = {c["cand_id"]: len(rr.get(c["cand_id"], {})) for c in d["cands"] if c.get("restore_ok")}
        short = {c: n for c, n in runs.items() if n < 6}
        if short:
            incomplete.append((pos, short))
        r2, r3 = repair_sets(d, A2, A3)
        rec = {"fold": d["fold"], "domain": d["domain"], "task_id": d["task_id"], "reps_v2": r2, "reps_v3": r3}
        for name, reps in (("v2", r2), ("v3", r3)):
            rec[f"G_{name}"] = g_cross(rr, reps, pos) if reps else None
            rec[f"Gh_{name}"] = [g_half(rr, reps, hh) if reps else None for hh in H]
            rec[f"best_{name}"] = argbest(rr, reps, tuple(range(6)), pos) if reps else None
        info[pos] = rec
    if incomplete and not a.allow_incomplete:
        print(f"[停下] π0 读数不全的位置 {len(incomplete)} 个（例 {incomplete[:3]}）；先用 --retry-errors 补跑。"
              f"仍补不齐时加 --allow-incomplete（按登记第 7 节：按可用次数算，并报告）")
        return 1
    worth = lambda g: g is not None and g >= THR - 1e-9
    w0 = {v: sorted(p for p, r in info.items() if worth(r[f"G_{v}"])) for v in ("v2", "v3")}
    half = {v: sorted(p for p, r in info.items() if any(worth(g) for g in r[f"Gh_{v}"])) for v in ("v2", "v3")}
    full_set = set(w0["v2"]) | set(w0["v3"]) | set(half["v2"]) | set(half["v3"])
    rest = sorted((p for p in info if p not in full_set), key=lambda p: h("CHECK", p))
    check = sorted(rest[: int(round(a.check_frac * len(rest)))])
    loops = sorted(p for p, (b, n) in term.items() if p in info and n and b / n >= a.loop_frac)
    res = {"pi0": a.pi0, "w0": w0, "half_worth": half, "pi1_full": sorted(full_set), "check": check, "loop_positions": loops,
           "best": {p: {"v2": r["best_v2"], "v3": r["best_v3"]} for p, r in info.items()},
           "incomplete": incomplete, "n_positions": len(info)}
    Path(a.out_dir, "w0.json").write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding="utf-8")
    Path(a.out_dir, "pi0_records.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    full_or_check = full_set | set(check)
    by = {d["pos_id"]: d for d in design}
    for fold in "AB":
        ds = [d for d in design if d["fold"] == fold]
        wjl(Path(a.out_dir, f"design_pi1_free_{fold}.jsonl"), [dict(d, cands=[c for c in d["cands"] if c["cand_id"] == "free"]) for d in ds])
        full_rows = [dict(d, cands=[c for c in d["cands"] if c["cand_id"] == "free" or (is_repair(c) and A2.get((d["pos_id"], c["cand_id"])))])
                     for d in ds if d["pos_id"] in full_or_check]
        wjl(Path(a.out_dir, f"design_pi1_full_{fold}.jsonl"), full_rows)
        # 11-07：行为检查用的位置 = 该折 π1 全部修复候选设计里，每个领域按位置 id 排序的前 6 个（两个领域分开判，补充三）
        chk = [r for dm in ("retail", "airline") for r in sorted((r for r in full_rows if r["domain"] == dm), key=lambda r: r["pos_id"])[:6]]
        wjl(Path(a.out_dir, f"design_pi1check_{fold}.jsonl"), chk)
    s2 = []
    for p in w0["v3"]:
        d = by[p]
        if d["fold"] != a.seed2_fold or info[p]["best_v3"] is None:
            continue
        s2.append(dict(d, cands=[c for c in d["cands"] if c["cand_id"] in ("free", info[p]["best_v3"])]))
    wjl(Path(a.out_dir, f"design_pi1s2_{a.seed2_fold}.jsonl"), s2)
    n = len(info)
    for fold in "AB":
        ps = [p for p in info if info[p]["fold"] == fold]
        print(f"{fold} 折：位置 {len(ps)}；π0 值得修 v3 {sum(p in w0['v3'] for p in ps)}、v2 {sum(p in w0['v2'] for p in ps)}；"
              f"一半内部值得修（v2 ∪ v3）{sum(p in set(half['v2']) | set(half['v3']) for p in ps)}；"
              f"π1 跑全部修复候选 {sum(p in full_or_check for p in ps)}（其中核对样本 {sum(p in check for p in ps)}）")
    print(f"共 {n} 个位置；π0 值得修 v3 {len(w0['v3'])}（{len(w0['v3']) / n:.3f}）、v2 {len(w0['v2'])}；反复报错的位置 {len(loops)}；"
          f"种子 2（{a.seed2_fold} 折）位置 {len(s2)}；读数不全的位置 {len(incomplete)}")
    print("== 完成")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("folds")
    f.add_argument("--ids-retail", default="ids_retail_det.txt")
    f.add_argument("--ids-airline", default="ids_airline.txt")
    f.add_argument("--out-dir", default="pool")
    s = sub.add_parser("select")
    s.add_argument("--src-retail", nargs="+", required=True)
    s.add_argument("--src-airline", nargs="+", required=True)
    s.add_argument("--folds", default="pool/folds.json")
    s.add_argument("--per-task", type=int, default=4)
    s.add_argument("--n", type=int, default=800)
    s.add_argument("--out", default="pool/positions.jsonl")
    b = sub.add_parser("branches")
    b.add_argument("--design", nargs="+", required=True)
    b.add_argument("--positions", default="pool/positions.jsonl")
    b.add_argument("--adm-v2", default="pool/adm_v2.jsonl")
    b.add_argument("--out", default="pool/design_run.jsonl")
    w = sub.add_parser("w0")
    w.add_argument("--design", default="pool/design_run.jsonl")
    w.add_argument("--rollouts", nargs="+", default=["pool/ro_pi0.jsonl"])
    w.add_argument("--pi0", default="4B")
    w.add_argument("--adm-v2", default="pool/adm_v2.jsonl")
    w.add_argument("--adm-v3", default="pool/adm_v3.jsonl")
    w.add_argument("--check-frac", type=float, default=0.15)
    w.add_argument("--loop-frac", type=float, default=0.25, help="π0 在该位置的续跑中问题终止（工具错误过多 / 跑满步数 / 出错）的比例 ≥ 此值")
    w.add_argument("--seed2-fold", default="A")
    w.add_argument("--allow-incomplete", action="store_true")
    w.add_argument("--out-dir", default="pool")
    a = ap.parse_args()
    return {"folds": cmd_folds, "select": cmd_select, "branches": cmd_branches, "w0": cmd_w0}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
