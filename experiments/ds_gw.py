"""Gridworld Datasets：One-step repair value conditional on the implementer（For registration see analysis plan.md 6.5）。

Every one“Source pool”= Some executors own failure trajectory（unscreened，Take front in torrent order N_FAIL records）。per trajectory 4 Organisation（Same rule as the confirmation experiment）
× Replace All Actions，Original action branch separates。All branches of each pool are handed over to two executors for renewal：
    S1   Every Random Branch N1 runs（The first half S1a，The second half S1b）
    G    greedy 1 runs
    S2   per trajectory：by S1a Highest net gain TOPK An alternative branch + Original action branch by location，Again N2 runs（Independent Random）
Unprejudiced estimation of labels：Pick one and look S1a，Label only uses samples that are not relevant to selection——Enter S2 for the replacement branch of the S2 + S1b，For the remaining alternative branch S1b
（If you want to use S1a When Label，Failure to elect will result from“excluded precisely because the estimate was low”and therefore biased downward，High number of candidates）。The original action branch is not subject to selection，Merge All Samples。
deterministic branch（Closed or not within budget）Direct Values。

    python ds_gw.py --stage all --executor q7          # collect + Design the pool of the executor，evaluate its own pool；The other pool has been designed and evaluated together
    python ds_gw.py --stage all --executor q14
    python ds_gw.py --stage all --executor q7          # Third time on single card：Finish q7 Yeah q14 Assessment of the pool（Completed Part Auto Skip）
    python ds_gw.py --stage all --executor q14 --wait  # Double cards running at the same time：evaluate the other pool after its design is ready
    python ds_gw.py --stage export                     # Export data sets（out_ds_gw/dataset/）
    python ds_gw.py --stage analyze                    # Reliance analysis by the implementer（out_ds_gw/analysis.md）
    python ds_gw.py --stage selftest                   # offline self-test（Noise oracle Replace Model）
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import config
import design as D
import oracle
from env import GridWorld
from policies import derive_seed

OUT = config.ROOT / "out_ds_gw"
EXECUTORS = {"q7": os.environ.get("DS_Q7", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"),
             "q14": os.environ.get("DS_Q14", "/root/autodl-tmp/models/Qwen2.5-14B-Instruct")}
SEED_START = {"q7": 5000, "q14": 7000}     # New map torrent（Confirming experimental use 1000–1199，pilot use 0–119）
MOCK_EPS = {"q7": 0.5, "q14": 0.3}         # for self-tests only：Noise oracle Noise
N_FAIL = 100
MAX_COLLECT = 1000
COLLECT_BATCH = 100
N_POSITIONS = 4
POS_SEED = 4242
N1, TOPK, N2 = 10, 3, 24        # S1 split into two halves：The first half（S1a）For the selection of second-stage candidates only，The second half（S1b）Use only as a label
TEMPERATURE = config.TEMPERATURE
STATE = {"mock": False, "out": OUT}


def log(msg):
    print(msg, flush=True)
    try:
        STATE["out"].mkdir(parents=True, exist_ok=True)
        with open(STATE["out"] / "ds.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("[%m-%d %H:%M:%S] ") + msg + "\n")
    except OSError:
        pass


def fp(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


def P(name, pool=None, ex=None):
    o = STATE["out"]
    return {"episodes": o / f"episodes_{pool}.jsonl", "design": o / f"design_{pool}.json",
            "evals": o / f"evals_{pool}_by_{ex}.jsonl"}[name]


def settings():
    return {"N_FAIL": N_FAIL, "N_POSITIONS": N_POSITIONS, "POS_SEED": POS_SEED, "N1": N1, "TOPK": TOPK, "N2": N2,
            "SEED_START": SEED_START, "T": TEMPERATURE, "TOP_P": config.TOP_P, "MAX_NEW_TOKENS": config.MAX_NEW_TOKENS,
            "GRID": config.GRID, "MAX_STEPS": config.MAX_STEPS, "N_TRAPS": config.N_TRAPS, "N_WALLS": config.N_WALLS,
            "code": {n: config.file_sha256(config.ROOT / n)[:16] for n in ("env.py", "model.py", "oracle.py")},
            "mock": STATE["mock"]}


# ---------------------------------------------------------------- Policy
class Executor:
    def __init__(self, name):
        self.name = name
        if STATE["mock"]:
            from policies import NoisyOraclePolicy
            self.policy = NoisyOraclePolicy(MOCK_EPS[name])
            self.policy_id = f"mock-{name}"
        else:
            from policies import make_seeded_llm_class
            cls = make_seeded_llm_class()
            self.policy = cls(temperature=TEMPERATURE, model_dir=EXECUTORS[name])
            self.policy_id = self.policy.cache_id

    def rollouts(self, snaps, seeds, temperature):
        if not STATE["mock"]:
            self.policy.temperature = temperature
            self.policy.escape_temperature = None
        return self.policy.run_rollouts(snaps, seeds)

    def actor(self):
        if STATE["mock"]:
            import confirm
            eps = MOCK_EPS[self.name]

            def act(envs, keys, steps):
                from config import CANDIDATE_ACTIONS
                out = []
                for e, k, s in zip(envs, keys, steps):
                    rng = random.Random(derive_seed("mockds", self.name, k, s))
                    snap = e.snapshot()
                    acts = oracle.table_for(snap).optimal_actions(snap)
                    a = dict(rng.choice(CANDIDATE_ACTIONS)) if (not acts or rng.random() < eps) else dict(rng.choice(acts))
                    out.append((a, {"text": "(mock)", "finish_reason": "stop"}))
                return out
            return act
        import confirm
        return confirm._llm_actor(self.policy, TEMPERATURE)


# ---------------------------------------------------------------- collect
def collect_pool(ex):
    pool = ex.name
    STATE["out"].mkdir(parents=True, exist_ok=True)
    path = P("episodes", pool)
    done = D.load_records(path) if path.exists() else []
    have = {r["traj_id"] for r in done}
    start = SEED_START[pool]
    while True:
        recs = D.load_records(path) if path.exists() else []
        fails = [r for r in recs if not r["success"]]
        if len(fails) >= N_FAIL or len(recs) >= MAX_COLLECT:
            break
        lo = start + len(recs)
        seeds = [s for s in range(lo, lo + COLLECT_BATCH) if f"{pool}_{s:05d}" not in have]
        _collect(path, ex.actor(), ex.policy_id, seeds, pool)
        have |= {f"{pool}_{s:05d}" for s in seeds}
    recs = D.load_records(path)
    fails = sorted([r for r in recs if not r["success"]], key=lambda r: r["seed"])
    log(f"[collect {pool}] {len(recs)} episodes，failed {len(fails)}，Prior {min(N_FAIL, len(fails))} records")
    return fails[:N_FAIL]


def _collect(path, actor, policy_id, seeds, prefix):
    envs, recs = {}, {}
    for s in seeds:
        layout = GridWorld.random_layout_checked(n=config.GRID, n_traps=config.N_TRAPS, n_walls=config.N_WALLS, seed=s)
        env = GridWorld(layout, max_steps=config.MAX_STEPS, seed=s)
        assert oracle.feasible(env.snapshot())
        envs[s] = env
        recs[s] = {"traj_id": f"{prefix}_{s:05d}", "states": [env.snapshot()], "actions": [], "generations": [],
                   "layout": ["".join(r) for r in layout], "seed": s, "policy_id": policy_id}
    alive = list(seeds)
    with path.open("a", encoding="utf-8") as f:
        while alive:
            res = actor([envs[s] for s in alive], alive, [envs[s].steps for s in alive])
            nxt = []
            for s, (act, gen) in zip(alive, res):
                envs[s].step(act)
                recs[s]["actions"].append(act)
                recs[s]["generations"].append(gen)
                recs[s]["states"].append(envs[s].snapshot())
                if envs[s].done:
                    r = recs[s]
                    r.update({"success": envs[s].success, "fail_reason": envs[s].fail_reason, "T": len(r["actions"])})
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                else:
                    nxt.append(s)
            alive = nxt


# ---------------------------------------------------------------- design
def design_pool(pool, fails):
    import confirm
    path = P("design", pool)
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["settings"] == settings():
            return d
        sys.exit(f"[design {pool}] Designed different from current settings；Please change the output directory")
    trajs, branches = [], {}
    for rec in fails:
        D.verify_replay(rec)
        pos = D.select_positions(rec, N_POSITIONS, POS_SEED)
        positions = []
        for i, t in enumerate(pos):
            oid = f"{rec['traj_id']}|t{t}|orig"
            branches[oid] = D._branch(oid, rec["states"][t + 1], rec["traj_id"], t, "orig:" + D.action_key(rec["actions"][t]))
            alts = confirm._alts_at(rec, t)
            for b in alts:
                branches[b["branch_id"]] = b
            positions.append({"t": t, "orig": oid, "alts": [b["branch_id"] for b in alts], "decisive": i == len(pos) - 1})
        trajs.append({"traj_id": rec["traj_id"], "seed": rec["seed"], "fail_reason": rec["fail_reason"], "T": rec["T"],
                      "positions": positions})
    d = {"pool": pool, "trajectories": trajs, "branches": branches, "settings": settings()}
    d["design_id"] = fp({"t": trajs, "s": d["settings"]})
    tmp = path.with_suffix(".tmp")                   # Write ad hoc papers and change names：The process on the other card wont be able to read half the design
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    nst = sum(b["deterministic"] is None for b in branches.values())
    log(f"[design {pool}] {len(trajs)} trajectories、{len(branches)} branches（random {nst}）")
    return d


# ---------------------------------------------------------------- evaluation
def load_evals(path):
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # Half the last line when you interrupt
                out[(r["phase"], r["branch_id"])] = r
    return out


def _run(ex, d, phase, bids, n, temperature, path, sid):
    done = load_evals(path)
    todo = [b for b in bids if (phase, b) not in done]
    if not todo:
        return
    log(f"[eval {d['pool']}←{ex.name} {phase}] Stay {len(todo)} branches × {n} runs")
    t0 = time.time()
    per = max(1, config.ROLLOUT_CHUNK // n)
    with path.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), per):
            chunk = todo[i:i + per]
            snaps, seeds, owner = [], [], []
            for b in chunk:
                for k in range(n):
                    snaps.append(d["branches"][b]["snapshot"])
                    seeds.append(derive_seed("ds", d["pool"], ex.name, phase, b, k))
                    owner.append(b)
            stats = ex.rollouts(snaps, seeds, temperature)
            res = {}
            for b, st in zip(owner, stats):
                res.setdefault(b, []).append((bool(st["success"]), int(st["steps"])))
            for b in chunk:
                rows = res[b]
                f.write(json.dumps({"pool": d["pool"], "executor": ex.name, "phase": phase, "branch_id": b, "n": n,
                                    "results": [r[0] for r in rows], "steps": [r[1] for r in rows], "spec": sid}) + "\n")
            f.flush()
            log(f"  {min(i + per, len(todo))}/{len(todo)} branches，{(time.time() - t0) / 60:.1f} minutes")


def _slice(r, ph):
    res = r["results"]
    h = len(res) // 2
    return res[:h] if ph == "S1a" else res[h:] if ph == "S1b" else res


def _p(d, ev, b, phases):
    """After the branch is merged at the given stage (Success, Number)；deterministic branches return (Value, None)。Phases available S1a / S1b Organisation S1 Two and a half。"""
    det = d["branches"][b]["deterministic"]
    if det is not None:
        return det, None
    k = n = 0
    for ph in phases:
        r = ev.get(("S1" if ph in ("S1a", "S1b") else ph, b))
        if r:
            x = _slice(r, ph)
            k += sum(x)
            n += len(x)
    return k, n


def _rate(kn):
    k, n = kn
    return float(k) if n is None else (k / n if n else float("nan"))


def stage2_choice(d, ev):
    """Press every trajectories S1a Net proceeds TOPK An alternative branch（Breaks the Hashi by branch number，Fixed Recoverable）。"""
    out = {}
    for t in d["trajectories"]:
        sc = []
        for p in t["positions"]:
            po = _rate(_p(d, ev, p["orig"], ("S1a",)))
            for a in p["alts"]:
                pa = _rate(_p(d, ev, a, ("S1a",)))
                sc.append((pa - po, hashlib.sha256(a.encode()).hexdigest(), a))
        sc.sort(key=lambda x: (-x[0], x[1]))
        out[t["traj_id"]] = [a for _, _, a in sc[:TOPK]]
    return out


def eval_pool(ex, d):
    path = P("evals", d["pool"], ex.name)
    sid = fp({"design": d["design_id"], "executor": ex.name, "model": EXECUTORS[ex.name], "policy": ex.policy_id,
              "N1": N1, "N2": N2, "TOPK": TOPK, "T": TEMPERATURE})
    for r in load_evals(path).values():
        if r["spec"] != sid:
            sys.exit(f"[eval {d['pool']}←{ex.name}] Different settings signed for existing results；Please change the output directory")
    nst = sorted(b for b, x in d["branches"].items() if x["deterministic"] is None)
    _run(ex, d, "S1", nst, N1, TEMPERATURE, path, sid)
    _run(ex, d, "G", nst, 1, 0.0, path, sid)
    ev = load_evals(path)
    top = stage2_choice(d, ev)
    s2 = set()
    for t in d["trajectories"]:
        s2 |= {b for b in top[t["traj_id"]] if d["branches"][b]["deterministic"] is None}
        s2 |= {p["orig"] for p in t["positions"] if d["branches"][p["orig"]]["deterministic"] is None}
    _run(ex, d, "S2", sorted(s2), N2, TEMPERATURE, path, sid)
    log(f"[eval {d['pool']}←{ex.name}] complete")


# ---------------------------------------------------------------- Label & Analysis
def labels(d, ev):
    """Estimated unbiased net gain per candidate against the implementer and number of samples used；and S2 Selection。"""
    top = stage2_choice(d, ev)
    lab = {}
    for t in d["trajectories"]:
        tt = set(top[t["traj_id"]])
        for p in t["positions"]:
            ko = _p(d, ev, p["orig"], ("S1", "S2"))
            po = _rate(ko)
            for a in p["alts"]:
                ph = ("S2", "S1b") if a in tt else ("S1b",)
                ka = _p(d, ev, a, ph)
                g = _p(d, ev, a, ("G",))
                go = _p(d, ev, p["orig"], ("G",))
                gain_of = lambda phs: _rate(_p(d, ev, a, phs)) - _rate(_p(d, ev, p["orig"], phs))
                lab[a] = {"traj": t["traj_id"], "t": p["t"], "p_alt": _rate(ka), "n_alt": ka[1], "p_orig": po,
                          "n_orig": ko[1], "gain": _rate(ka) - po, "in_s2": a in tt,
                          "s1a_gain": gain_of(("S1a",)), "s1b_gain": gain_of(("S1b",)), "s1_gain": gain_of(("S1",)),
                          "greedy_gain": _rate(g) - _rate(go)}
    return lab


def analyze():
    """Enforcement dependency：Is the same group of candidates worth different under two implementers?；and“by A Selection、Here B use”Losses。"""
    rng = np.random.default_rng(0)
    L = ["# Gridworld Datasets：Reliance analysis by the implementer", ""]
    res = {}
    for pool in EXECUTORS:
        if not P("design", pool).exists():
            continue
        d = json.loads(P("design", pool).read_text(encoding="utf-8"))
        labs = {}
        for ex in EXECUTORS:
            ev = load_evals(P("evals", pool, ex))
            if not ev:
                continue
            labs[ex] = labels(d, ev)
        if len(labs) < 2:
            L.append(f"- Ji! {pool}：Assessment not completed（{list(labs)}）")
            continue
        exs = list(labs)
        trajs = [t["traj_id"] for t in d["trajectories"]]
        cands = {tid: [a for t in d["trajectories"] if t["traj_id"] == tid for p in t["positions"] for a in p["alts"]]
                 for tid in trajs}
        # per trajectory：Each executor press his own S1 Best candidate selected；Executor E Value（No-Straight Label）
        rows = []
        for tid in trajs:
            cs = cands[tid]
            pick = {e: max(cs, key=lambda a: (labs[e][a]["s1a_gain"], a)) for e in exs}
            row = {"traj": tid}
            for e_exec in exs:
                for e_sel in exs:
                    row[f"{e_sel}->{e_exec}"] = labs[e_exec][pick[e_sel]]["gain"]
                row[f"random->{e_exec}"] = float(np.mean([labs[e_exec][a]["gain"] for a in cs]))
                row[f"greedy->{e_exec}"] = labs[e_exec][max(cs, key=lambda a: (labs[e_exec][a]["greedy_gain"], a))]["gain"]
            # Two executors in S1 Sorting in relation to net proceeds（Half confidence in the executor，See notes to the report）
            v = [[labs[e][a]["s1_gain"] for a in cs] for e in exs]
            row["spearman"] = _spearman(v[0], v[1])
            # Noise cap：Same executor S1 rank correlation between the halves（Half Message；Two and a half N1/2 runs）
            for e in exs:
                row[f"split_{e}"] = _spearman([labs[e][a]["s1a_gain"] for a in cs], [labs[e][a]["s1b_gain"] for a in cs])
            row["spearman_half"] = float(np.nanmean([_spearman([labs[exs[0]][a]["s1a_gain"] for a in cs],
                                                               [labs[exs[1]][a]["s1b_gain"] for a in cs]),
                                                     _spearman([labs[exs[0]][a]["s1b_gain"] for a in cs],
                                                               [labs[exs[1]][a]["s1a_gain"] for a in cs])]))
            row["same_best"] = float(pick[exs[0]] == pick[exs[1]])
            rows.append(row)
        res[pool] = rows

        def boot(key_a, key_b):
            x = np.array([r[key_a] - r[key_b] for r in rows])
            idx = rng.integers(0, len(x), size=(2000, len(x)))
            m = x[idx].mean(axis=1)
            return x.mean(), np.quantile(m, 0.05), np.quantile(m, 0.95)
        L += [f"## Source pool {pool}（{len(rows)} records {pool} Failed trajectories）", "",
              "| selection → Implementation | Net deployment gains | Relative“Match”Losses [90%] |", "|---|---:|---|"]
        for e_exec in exs:
            base = f"{e_exec}->{e_exec}"
            L.append(f"| {base}（Match） | {np.mean([r[base] for r in rows]):+.3f} | — |")
            for k in [f"{e}->{e_exec}" for e in exs if e != e_exec] + [f"greedy->{e_exec}", f"random->{e_exec}"]:
                m, lo, hi = boot(base, k)
                L.append(f"| {k} | {np.mean([r[k] for r in rows]):+.3f} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        mean = lambda k: float(np.nanmean([r[k] for r in rows]))
        gap = np.array([np.nanmean([r["split_" + exs[0]], r["split_" + exs[1]]]) - r["spearman_half"] for r in rows])
        gap = gap[~np.isnan(gap)]
        gm = gap[rng.integers(0, len(gap), size=(2000, len(gap)))].mean(axis=1) if len(gap) else np.array([np.nan])
        res[pool + "_rank_gap"] = [float(gap.mean()) if len(gap) else float("nan"),
                                   float(np.quantile(gm, 0.05)), float(np.quantile(gm, 0.95))]
        L += ["", "Sort Consistency（Net gain from each trajectories Spearman，Average；Same“Half a sample vs Half a sample”are comparable only then）：", "",
              f"- Cross-executors（{exs[0]} Half of it vs {exs[1]} The other half）：{mean('spearman_half'):.3f}",
              f"- Partial half confidence in the same executor：{exs[0]} {mean('split_' + exs[0]):.3f}；{exs[1]} {mean('split_' + exs[1]):.3f}",
              f"- Internal − Cross-implementer coherence：{res[pool + '_rank_gap'][0]:+.3f} "
              f"[{res[pool + '_rank_gap'][1]:+.3f}, {res[pool + '_rank_gap'][2]:+.3f}]"
              "（lower bound > 0：The ranking of the restoration value does change with the executor，rather than sampling noise alone；Intersection 0：cannot distinguish）",
              f"- Two executors press S1a Proportion of selection of the same best candidate：{np.mean([r['same_best'] for r in rows]):.3f}", ""]
    L += ["Annotations：Match selection with the executor S1a Selection、Samples not relevant to selection（S2 + S1b or S1b）evaluation；Cross-check with another executor S1a Selection、"
          "The same applies to the executors own neutral label。"]
    (STATE["out"] / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L))
    return res


def summary(pool, ex=None):
    """Single Pool × Summing up of individual executors（No need for another executor）：Size of collection and design、How common is the one-step repair signal?，
    And there are several options“unscreened failed trajectories”Net gains on deployment（Evaluation with unbiased labels，It can be compared to the confirmed experiment）。"""
    ex = ex or pool
    rng = np.random.default_rng(0)
    d = json.loads(P("design", pool).read_text(encoding="utf-8"))
    ev = load_evals(P("evals", pool, ex))
    lab = labels(d, ev)
    recs = D.load_records(P("episodes", pool))
    br = d["branches"]
    feas = lambda b: 1.0 if _slack(br[b]) > -100 else 0.0
    n_nan = sum(1 for v in lab.values() if not np.isfinite(v["gain"]))
    rows = []
    for t in d["trajectories"]:
        cands = [(a, p["orig"]) for p in t["positions"] for a in p["alts"]]
        g = np.array([lab[a]["gain"] for a, _ in cands])

        def pick(score):
            sc = np.array(score, float)
            m = np.abs(sc - np.nanmax(sc)) <= 1e-9
            return float(np.nanmean(g[m]))
        rows.append({
            "traj": t["traj_id"], "fail": t["fail_reason"],
            "s1a": pick([lab[a]["s1a_gain"] for a, _ in cands]),
            "greedy": pick([lab[a]["greedy_gain"] for a, _ in cands]),
            "oracle": pick([feas(a) - feas(o) for a, o in cands]),
            "oracle_slack": pick([(feas(a) - feas(o)) * 1000 + _slack(br[a]) - _slack(br[o]) for a, o in cands]),
            "random": float(np.nanmean(g)),
            "best_label": float(np.nanmax(g)),
            "signal": float(np.nanmax(g) > 0.10)})

    def ci(key_a, key_b=None, sub=None):
        rr = [r for r in rows if sub is None or r["fail"] == sub]
        x = np.array([r[key_a] - (r[key_b] if key_b else 0.0) for r in rr])
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return "—"
        m = x[rng.integers(0, len(x), size=(2000, len(x)))].mean(axis=1)
        return f"{x.mean():+.3f} [{np.quantile(m, 0.05):+.3f}, {np.quantile(m, 0.95):+.3f}]"
    from collections import Counter
    n_succ = sum(r["success"] for r in recs)
    fr = Counter(t["fail_reason"] for t in d["trajectories"])
    nst = sum(b["deterministic"] is None for b in br.values())
    L = [f"# Gridworld Datasets：Ji! {pool} × executor {ex} Summing up", "",
         f"- collect：{len(recs)} episodes，Success {n_succ}；Failed trajectories in torrent order {len(d['trajectories'])} records（unscreened）；Failed Type {dict(fr)}",
         f"- design：{len(d['trajectories'])} trajectories × At most {N_POSITIONS} Organisation；Alternative Branch {len(lab)} items，Branches {len(br)} items（random {nst}）"
         + (f"；**{n_nan} A candidate has no label（Assessment not completed）**" if n_nan else ""),
         f"- One-step fix the signal：Net gain on labels > 0.10 fraction of trajectories with at least one such candidate {np.mean([r['signal'] for r in rows]):.2f}"
         f"（Confirm that the experiment only uses a filtered signal trail）；Average of candidates with the highest labels per trajectory {np.mean([r['best_label'] for r in rows]):+.3f}"
         "（Maximum value，optimistically biased，For reference only）", "",
         "**Net deployment gains from several options**（Selection of candidates for evaluation with unbiased labels not related to selection；Average when side-by-line；90% Intersection is a trajectories bootstrap）：", "",
         "| Selection | all | " + " | ".join(fr) + " |", "|---|---|" + "---|" * len(fr)]
    for k, lab_ in (("s1a", f"by {ex} Myself {N1 // 2} Keep going（S1a）Selection"), ("greedy", f"by {ex} greedy 1 Subselection"),
                    ("oracle", "oracle Separability（Parallel Average）"), ("oracle_slack", "oracle + slack"), ("random", "random")):
        L.append(f"| {lab_} | {ci(k)} | " + " | ".join(ci(k, sub=f_) for f_ in fr) + " |")
    L += ["", "**Difference**（> 0 positive means selection using the deployed policys own continuations is better）：", "",
          "| Contrast | all | " + " | ".join(fr) + " |", "|---|---|" + "---|" * len(fr)]
    for b, lab_ in (("greedy", "greedy"), ("oracle", "oracle"), ("oracle_slack", "oracle + slack"), ("random", "random")):
        L.append(f"| S1a − {lab_} | {ci('s1a', b)} | " + " | ".join(ci("s1a", b, sub=f_) for f_ in fr) + " |")
    L += ["", f"Annotations：S1a only {N1 // 2} Subsamples（matched selection in the confirmatory experiment is 20 runs），therefore the quantity here“Selected according to deployment strategy”is a conservative variant；"
          "Label is neutral for each candidate，but candidates excluded from stage two have only 5 Subsamples，higher sampling noise。"]
    out = STATE["out"] / f"summary_{pool}_by_{ex}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L))
    return rows


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- Export
def _fmt_action(a):
    if isinstance(a, str):
        tool, _, arg = a.partition(":")
    else:
        tool, arg = a.get("tool"), a.get("arg")
    return f"{tool}({arg})" if arg else f"{tool}()"


def export():
    outd = STATE["out"] / "dataset"
    outd.mkdir(parents=True, exist_ok=True)
    n_items = 0
    with open(outd / "gridworld.jsonl", "w", encoding="utf-8") as f:
        for pool in EXECUTORS:
            if not P("design", pool).exists():
                continue
            d = json.loads(P("design", pool).read_text(encoding="utf-8"))
            recs = {r["traj_id"]: r for r in D.load_records(P("episodes", pool))}
            labs = {ex: labels(d, load_evals(P("evals", pool, ex))) for ex in EXECUTORS
                    if P("evals", pool, ex).exists()}
            for t in d["trajectories"]:
                rec = recs[t["traj_id"]]
                hist = rec["states"][-1]["history"]
                item = {"id": f"gridworld/{pool}/{t['traj_id']}", "env": "gridworld", "source_executor": pool,
                        "seed": rec["seed"], "layout": rec["layout"], "max_steps": config.MAX_STEPS,
                        "fail_reason": t["fail_reason"], "T": t["T"],
                        "trajectory": [{"t": h["step"], "action": _fmt_action(h["action"]), "result": h["result"]}
                                       for h in hist],
                        "positions": [], "labels": {}}
                for p in t["positions"]:
                    item["positions"].append({
                        "t": p["t"], "decisive": p["decisive"],
                        "orig_action": _fmt_action(rec["actions"][p["t"]]),
                        "candidates": [{"cid": a, "action": _fmt_action(d["branches"][a]["action"]),
                                        "oracle_feasible": bool(oracle.feasible(d["branches"][a]["snapshot"])),
                                        "oracle_slack": _slack(d["branches"][a])} for a in p["alts"]]})
                for ex, lab in labs.items():
                    item["labels"][ex] = {a: {k: lab[a][k] for k in ("gain", "p_alt", "n_alt", "p_orig", "n_orig",
                                                                     "in_s2", "greedy_gain")}
                                          for p in t["positions"] for a in p["alts"]}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_items += 1
    card = ["# Gridworld Subset：Data description", "",
            f"- Entry：{n_items}（Every single one = A failure trajectory × 4 Organisation × Replace All Actions）；Source pool：{list(EXECUTORS)}",
            "- Label：`labels[executor][candidate]` = Estimated net deployment gains from the restoration handed over to the implementer for renewal gain = p_alt − p_orig；"
            "n_alt / n_orig As Samples（None Expression of certainty）；in_s2 Expressing the candidature to phase two（Labels only use independent second-stage samples）",
            f"- Sample：S1 Each branch {N1} runs（The first half is only for selection、The second half is used only as a label）；per trajectory S1 The top half of the net gain is the highest {TOPK} A candidate"
            f"Separate from the original action branch {N2} runs；temperature {TEMPERATURE}。Tabs and samples for selection are independent，therefore unbiased for every candidate",
            "- Usage：Method to give one for each entry cid（or give up），use ds_score.py Calculation of average net deployment gains on designated implementers"]
    (outd / "dataset_card.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    log(f"[export] {n_items} records -> {outd}")


def _slack(br):
    s = br["snapshot"]
    if s["done"]:
        return 100.0 if s["success"] else -100.0
    dist, rem = br["oracle_distance"], s["max_steps"] - s["steps"]
    return float(rem - dist) if (dist is not None and dist <= rem) else -100.0


# ---------------------------------------------------------------- The entrance
def stage_all(name, wait):
    ex = Executor(name)
    fails = collect_pool(ex)
    d_own = design_pool(name, fails)
    eval_pool(ex, d_own)
    for other in EXECUTORS:
        if other == name:
            continue
        path = P("design", other)
        while wait and not path.exists():
            log(f"[{name}] Waiting for the pool {other} Design ...")
            time.sleep(60)
        if path.exists():
            eval_pool(ex, json.loads(path.read_text(encoding="utf-8")))
        else:
            log(f"[{name}] Ji! {other} No design yet；Use it later --executor {name} One more run to finish")


def selftest():
    global N_FAIL, COLLECT_BATCH
    tmp = Path(tempfile.mkdtemp(prefix="dsgw_"))
    ok = True
    try:
        STATE.update(mock=True, out=tmp)
        N_FAIL, COLLECT_BATCH = 12, 30
        stage_all("q7", False)
        stage_all("q14", False)
        n_before = sum(1 for _ in open(P("evals", "q14", "q7")) ) if P("evals", "q14", "q7").exists() else 0
        stage_all("q7", False)                              # Finish q7 Yeah q14 Ji!
        n_after = sum(1 for _ in open(P("evals", "q14", "q7")))
        ok &= n_before == 0 and n_after > 0
        print(("  ✓ " if ok else "  ✗ ") + f"Single card order to run：q7 Yeah q14 Ji! {n_before} -> {n_after} Records")
        again = sum(1 for _ in open(P("evals", "q7", "q7")))
        stage_all("q7", False)
        same = sum(1 for _ in open(P("evals", "q7", "q7"))) == again
        print(("  ✓ " if same else "  ✗ ") + "Rerun or not write（Break and break）")
        ok &= same
        d = json.loads(P("design", "q7").read_text())
        ev = load_evals(P("evals", "q7", "q7"))
        lab = labels(d, ev)
        n_s2 = sum(v["in_s2"] for v in lab.values())
        good = n_s2 <= TOPK * len(d["trajectories"]) and all(
            (v["n_alt"] in (None, N2 + N1 - N1 // 2) if v["in_s2"] else v["n_alt"] in (None, N1 - N1 // 2))
            for v in lab.values())
        print(("  ✓ " if good else "  ✗ ") + f"The tag sample number is correct（Candidates for the second phase {n_s2} items）")
        ok &= good
        tmp2 = Path(tempfile.mkdtemp(prefix="dsgw_eval_"))
        shutil.copy(P("design", "q7"), tmp2 / "design_q7.json")
        STATE["out"] = tmp2
        eval_pool(Executor("q14"), json.loads(P("design", "q7").read_text()))
        n_e = sum(1 for _ in open(P("evals", "q7", "q14")))
        good = n_e > 0 and not P("design", "q14").exists()
        print(("  ✓ " if good else "  ✗ ") + f"Only cross-assessments（14B evaluation 7B The pool，Do Not Collect 14B The pool）：{n_e} Records")
        ok &= good
        shutil.rmtree(tmp2, ignore_errors=True)
        STATE["out"] = tmp
        rows = summary("q7", "q7")
        good = len(rows) == 12 and (STATE["out"] / "summary_q7_by_q7.md").exists()
        print(("  ✓ " if good else "  ✗ ") + "One-pool knot")
        ok &= good
        res = analyze()
        good = {"q7", "q14"} <= set(res) and all(len(res[k]) == 12 for k in ("q7", "q14"))
        print(("  ✓ " if good else "  ✗ ") + "Reliance analysis by the implementer")
        ok &= good
        export()
        items = [json.loads(x) for x in (tmp / "dataset" / "gridworld.jsonl").read_text().splitlines()]
        good = len(items) == 24 and all(set(it["labels"]) == {"q7", "q14"} for it in items)
        print(("  ✓ " if good else "  ✗ ") + f"Export {len(items)} records，Each article has a label of two executors")
        ok &= good
        import ds_score
        pred = {it["id"]: it["positions"][-1]["candidates"][0]["cid"] for it in items}
        sc = ds_score.score(items, pred, "q7")
        good = sc["n"] == 24 and -1 <= sc["mean_gain"] <= 1
        print(("  ✓ " if good else "  ✗ ") + f"Offline score {sc}")
        ok &= good
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ds selftest] " + ("all passed" if ok else "failures detected"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["all", "eval", "export", "analyze", "summary", "selftest"])
    ap.add_argument("--pool", choices=list(EXECUTORS), default=None,
                    help="summary：Which pool?（Default with --executor identical）；eval：Jean --executor Just evaluate the pool（does not collect its own pool）")
    ap.add_argument("--executor", choices=list(EXECUTORS))
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if args.stage == "selftest":
        return selftest()
    STATE["mock"] = args.mock
    if args.mock:
        STATE["out"] = config.ROOT / "out_ds_gw_mock"
    if args.stage == "all":
        if not args.executor:
            sys.exit("--stage all Yes --executor")
        stage_all(args.executor, args.wait)
    elif args.stage == "export":
        export()
    elif args.stage == "analyze":
        analyze()
    elif args.stage == "eval":
        # Only cross-assessments：--executor continuation --pool All branches（does not collect its own pool）。Example：14B evaluation 7B The pool
        if not (args.executor and args.pool):
            sys.exit("--stage eval Yes --executor and --pool")
        path = P("design", args.pool)
        if not path.exists():
            sys.exit(f"Nothing {path}：first copy in that pools design file {STATE['out']}/")
        eval_pool(Executor(args.executor), json.loads(path.read_text(encoding="utf-8")))
        log(f"[eval] {args.pool} Ji! ← {args.executor} complete：{P('evals', args.pool, args.executor)}")
    elif args.stage == "summary":
        ex = args.executor or "q7"
        summary(args.pool or ex, ex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
