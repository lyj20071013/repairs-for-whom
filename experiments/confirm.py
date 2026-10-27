"""Gridworld Confirm the experiment：New map、New trajectories，Pre-registration assumptions and judgments rules。

pipeline（There are stop conditions between sections）：
  collect   Use a completely new map feed（SEED_START Get up）Bulk Collection N_COLLECT episodes，Deployment strategy versus pilot exactly identical
  screen    Just look trap_step / timeout_return failed；At an irreversible point（Reconcilable position within the last budget）on，
            Filter Batch Independently（Each branch N_SCREEN runs）To judge whether or not“Effective one-step repair exists”：
            Any alternative action ≥ SCREEN_MIN_SUCC Sub-success。Not enough MIN_INFORMATIVE The rule is to stop，Do not enter the main experiment。
  main      trajectories with signals（At most N_MAIN_MAX records，Fixed Feed Random）× 4 Organisation（Last solventable position + 3 A random earlier location for a segment）
            × Replace All Actions；Deployment strategy per branch N_DEPLOY runs（Cross-drafting），before TRACE_DEPLOY Sub-store step-by-step records；
            greedy for every branch 1 times and save the records；oracle not used GPU。
  analyze   Pre-registration tests（See HYPOTHESES），Holm Correct only for primary tests C1–C3。

Filters a different random stream from the main experiment；Screening results do not participate in any selection or evaluation，Only determine the sample total：
“Deployment strategy has failed trajectories that have been effectively repaired one step at an irreversible point”。This is how the papers are presented。

  python confirm.py --stage prep           # Phase I：collect -> screen（approximately 25 minutes），decide after inspecting screening results
  python confirm.py --stage run            # Phase II：main -> analyze（approximately 3 hours）
  python confirm.py --stage all            # Two-stage run（Output in out_confirm/）
  python confirm.py --stage analyze        # Analysis only
  python confirm.py --stage all --mock     # none GPU：Noise oracle Replace 7B，Check process only
"""
import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import config
import design as D
import oracle
import runner
from env import GridWorld
from policies import derive_seed, PolicyPool, NoisyOraclePolicy, trace_of

# ================================================================ Pre-registration settings
OUT = config.ROOT / "out_confirm"
SEED_START = 1000              # New map torrent starting point（out_120 uses 0–119）
N_COLLECT = 200
FAIL_TYPES = ("trap_step", "timeout_return")
N_SCREEN = 10
SCREEN_MIN_SUCC = 3            # An alternative action on an irreversible point ≥ 3/10 success means“signal detected”
MIN_INFORMATIVE = 20           # Stop with less than this
N_MAIN_MAX = 40                # Revision 2：Use all filtered trajectories（Actual 34 records），originally 30
MAIN_SEED = 2027
N_POSITIONS = 4
N_DEPLOY = 40                  # in each of the two folds 20
TRACE_DEPLOY = 4
N_GREEDY = 1
KS = (3, 5, 10)                # A small number of rerun agents；k=3 Corresponding DoVer per intervention 3 runs
DEPLOY, GREEDY, T03 = "deploy_T07", "llm_greedy", "llm_T03"
N_T03 = 20                     # temperature 0.3 Select Batch（Revision 1 Accession），Same number of samples per discount for deployment conditions
B_SCREEN, B_MAIN = "SCR", "M"
DELTA = 0.10
ALPHA = 0.05                   # 90% interval；primary tests Holm Correct，Family visibility 0.05（One side）
N_BOOT = 2000

HYPOTHESES = {
    "C1": "Unreversible point：Select actions according to deployment policy − Random selection in an action that restores solvency > 0（Restorability is given only“Where?”，Do Not Give“In exchange for what?”）",
    "C2": "Samples based on deployment strategy − select using greedy replay > 0",
    "C3": "Samples based on deployment strategy − By environment oracle Selection（Parallel Average）> 0",
    "C4": "By deployment strategy 20 Subselection − Use only 3 Subselection > 0（DoVer Small run）",
    "C5": "oracle Break the parallels by the remaining balance − Pure oracle > 0（Low-cost improvements）",
    "C6": "Selected according to deployment strategy − By Temperature 0.3 Sample Selection：90% Upper Inter-District Border < δ And here it is δ considered safe within（Candidate security agent；Revision 1 Accession）",
    "M1": "Mechanisms：Greedy keeps running against the wall/Proportion of repeats the same action after crashing the door > Sampling and running",
}
PRIMARY = ("C1", "C2", "C3")
# Two dimensions：Inconsistencies in the subject of the assessment（The estimated success rate of the deployment strategy is no longer an estimate） vs Insufficient assessment budget（Same strategy，too few samples）
DIMENSION = {"C2": "Inconsistencies in the subject of the assessment：greedy is the other one（Determination）Policy",
             "C3": "Inconsistencies in the subject of the assessment：Solverability in lieu of deployment strategy success rate",
             "C6": "Inconsistencies in the subject of the assessment：Close Decoding（temperature 0.3）",
             "C4": "Insufficient assessment budget：The same strategy 3 A sample（Yes DoVer Every intervention 3 Sub-Verification Budget Incentives）"}
AMENDMENTS = [
    "Revision 1（2026-10-22 16:10，Before the main experiment runs,；Filter data not affected）：Add Temperature 0.3 Select Batch with C6；"
    "Also report success rate targets after restoration；Distinction in the report“Inconsistencies in the subject of the assessment”and“Insufficient assessment budget”。Reason：External review notes v1 Medium Temperature 0.3 of"
    "“Security”The conclusions come from samples diluted by floor trajectories，And the net gain may be different from the post-repair success rate when compared across locations。",
    "Revision 2（2026-10-22 16:30，After seeing the number and type of filters,、Before the main experiment runs,）：Main experiment uses all filtered trajectories"
    "（N_MAIN_MAX by 30 Amend to read 40；Actual 34 records），Keep All trap_step trajectory（C1 Main sources）；"
    "Add descriptive reports by failure type。If the main experiment has been pressed 30 Bar start，Only stratification reports are used，Keep the sample 30 records。",
]


def _paths(out):
    return {"episodes": out / "episodes.jsonl", "screen_design": out / "screen_design.json",
            "screen": out / "screen_rollouts.jsonl", "screen_result": out / "screen_result.json",
            "main_design": out / "main_design.json", "main": out / "main_rollouts.jsonl",
            "analysis": out / "confirm_analysis.json", "report": out / "confirm_report.md"}


def settings():
    keys = ["SEED_START", "N_COLLECT", "FAIL_TYPES", "N_SCREEN", "SCREEN_MIN_SUCC", "MIN_INFORMATIVE", "N_MAIN_MAX",
            "MAIN_SEED", "N_POSITIONS", "N_DEPLOY", "TRACE_DEPLOY", "N_GREEDY", "N_T03", "KS", "DELTA", "ALPHA",
            "N_BOOT", "AMENDMENTS"]
    return {k: globals()[k] for k in keys}


# ================================================================ collect
def _llm_actor(policy, temperature):
    from model import _extract_action

    def act(envs, keys, steps):
        prompts = [policy._render(e.observe(), e.history) for e in envs]
        params = [policy._make_params(temperature=temperature, top_p=config.TOP_P, max_tokens=config.MAX_NEW_TOKENS,
                                      seed=derive_seed("collect", k, s)) for k, s in zip(keys, steps)]
        outs = policy.llm.generate(prompts, params, use_tqdm=False)
        return [(_extract_action(o.outputs[0].text),
                 {"text": o.outputs[0].text, "finish_reason": getattr(o.outputs[0], "finish_reason", None)})
                for o in outs]
    return act


def _mock_actor(eps=0.55):
    from config import CANDIDATE_ACTIONS

    def act(envs, keys, steps):
        out = []
        for e, k, s in zip(envs, keys, steps):
            rng = random.Random(derive_seed("mockcollect", k, s))
            snap = e.snapshot()
            acts = oracle.table_for(snap).optimal_actions(snap)
            a = dict(rng.choice(CANDIDATE_ACTIONS)) if (not acts or rng.random() < eps) else dict(rng.choice(acts))
            out.append((a, {"text": "(mock)", "finish_reason": "stop"}))
        return out
    return act


def collect(path, actor, policy_id, n=N_COLLECT, seed_start=SEED_START):
    done = {r["traj_id"] for r in D.load_records(path)} if path.exists() else set()
    seeds = [s for s in range(seed_start, seed_start + n) if f"cf{s:05d}" not in done]
    if not seeds:
        print("[collect] All completed")
        return
    envs, recs = {}, {}
    for s in seeds:
        layout = GridWorld.random_layout_checked(n=config.GRID, n_traps=config.N_TRAPS, n_walls=config.N_WALLS, seed=s)
        env = GridWorld(layout, max_steps=config.MAX_STEPS, seed=s)
        assert oracle.feasible(env.snapshot()), f"seed {s} Unsolved within initial state budget"
        envs[s] = env
        recs[s] = {"traj_id": f"cf{s:05d}", "states": [env.snapshot()], "actions": [], "generations": [],
                   "layout": ["".join(r) for r in layout], "seed": s, "run_id": config.run_id(), "policy_id": policy_id}
    alive = list(seeds)
    print(f"[collect] collect {len(seeds)} episodes（seed {seeds[0]}–{seeds[-1]}）")
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


# ================================================================ screen
def _alts_at(rec, t):
    out = []
    factual = rec["actions"][t]
    for cand in config.CANDIDATE_ACTIONS:
        if D.action_key(cand) == D.action_key(factual):
            continue
        env = GridWorld.restore(rec["states"][t])
        env.step(dict(cand))
        bid = f"{rec['traj_id']}|t{t}|{D.action_key(cand)}"
        out.append(D._branch(bid, env.snapshot(), rec["traj_id"], t, D.action_key(cand)))
    return out


def last_feasible(rec):
    T = len(rec["actions"])
    return max(t for t in range(T) if oracle.feasible(rec["states"][t]))


def build_screen(records):
    cands = sorted([r for r in records if not r["success"] and r["fail_reason"] in FAIL_TYPES],
                   key=lambda r: r["traj_id"])
    branches, trajs = {}, []
    for rec in cands:
        D.verify_replay(rec)
        t = last_feasible(rec)
        assert not oracle.feasible(rec["states"][t + 1]), "Irreversible after point should be unbundled in the budget"
        alts = _alts_at(rec, t)
        for b in alts:
            branches[b["branch_id"]] = b
        trajs.append({"traj_id": rec["traj_id"], "fail_reason": rec["fail_reason"], "t_star": t,
                      "alts": [b["branch_id"] for b in alts]})
    d = {"kind": "screen", "trajectories": trajs, "branches": branches,
         "rules": {"fail_types": FAIL_TYPES, "n_screen": N_SCREEN, "min_succ": SCREEN_MIN_SUCC}}
    d["design_id"] = config.fingerprint({"t": trajs, "r": d["rules"]})
    return d


def screen_result(sd, results):
    informative = []
    for t in sd["trajectories"]:
        best = 0
        for b in t["alts"]:
            br = sd["branches"][b]
            if br["deterministic"] is not None:
                k = br["deterministic"] * N_SCREEN
            else:
                k = results[(DEPLOY, B_SCREEN, b)]["successes"]
            best = max(best, k)
        if best >= SCREEN_MIN_SUCC:
            informative.append(t["traj_id"])
    return informative


# ================================================================ main design
def build_main(records, informative):
    by_id = {r["traj_id"]: r for r in records}
    ids = sorted(informative)
    rng = random.Random(MAIN_SEED)
    if len(ids) > N_MAIN_MAX:
        ids = sorted(rng.sample(ids, N_MAIN_MAX))
    branches, trajs = {}, []
    for tid in ids:
        rec = by_id[tid]
        t_star = last_feasible(rec)
        pos_list = D.select_positions(rec, N_POSITIONS, MAIN_SEED)
        assert pos_list[-1] == t_star
        positions = []
        for t in pos_list:
            oid = f"{tid}|t{t}|orig"
            branches[oid] = D._branch(oid, rec["states"][t + 1], tid, t, "orig:" + D.action_key(rec["actions"][t]))
            alts = _alts_at(rec, t)
            for b in alts:
                branches[b["branch_id"]] = b
            positions.append({"t": t, "orig": oid, "alts": [b["branch_id"] for b in alts], "decisive": t == t_star})
        trajs.append({"traj_id": tid, "fail_reason": rec["fail_reason"], "T": rec["T"], "positions": positions})
    d = {"kind": "confirm_main", "trajectories": trajs, "branches": branches, "settings": settings(),
         "hypotheses": HYPOTHESES,
         "code": {n: config.file_sha256(config.ROOT / n) for n in ("env.py", "model.py", "oracle.py", "confirm.py")}}
    d["design_id"] = config.fingerprint({"t": trajs, "s": d["settings"], "h": HYPOTHESES})
    return d


# ================================================================ analyze
def _slack(br):
    s = br["snapshot"]
    if s["done"]:
        return 100.0 if s["success"] else -100.0
    dist = br["oracle_distance"]
    rem = s["max_steps"] - s["steps"]
    return float(rem - dist) if (dist is not None and dist <= rem) else -100.0


def _best(s):
    return (np.abs(s - s.max(axis=-1, keepdims=True)) <= 1e-12).astype(float)


def _mv(mask, v):
    return (mask * v).sum(axis=-1) / mask.sum(axis=-1)


def _z_sig(ka, na, ko, no):
    """One-sided two-rater z Test（α=0.05）：Substitute branch superior to original action branch。n=None is a given value。"""
    pa = ka if na is None else ka / na
    po = ko if no is None else ko / no
    if pa <= po:
        return False
    var = (0 if na is None else pa * (1 - pa) / na) + (0 if no is None else po * (1 - po) / no)
    return True if var == 0 else (pa - po) / math.sqrt(var) > 1.645


def _selection_values(d, results, objective, n_boot, seed, extra=None, oracle_value=None, slack_value=None):
    """Calculate the value matrix of the selection rules under deployment conditions under given objectives V[name]: (n_boot+1, n_traj)，number 0 Estimates of behavioural points。

    objective = "net"：Scores and values are net gains p(Substitution) − p(original action)；"post"：Both success rates after repair p(Substitution)。
    """
    rng = np.random.default_rng(seed)
    br = d["branches"]
    trajs = d["trajectories"]
    nT = len(trajs)
    net = objective == "net"
    # oracle_value / slack_value For other environments only（ALFWorld）Enter when copying the analysis；Default is gridworld Definitions
    if oracle_value is None:
        O = {b: (1.0 if oracle.feasible(x["snapshot"]) else 0.0) for b, x in br.items()}
    else:
        O = {b: float(oracle_value(x)) for b, x in br.items()}
    G = {}
    for b, x in br.items():
        if x["deterministic"] is not None:
            G[b] = x["deterministic"]
        else:
            G[b] = results[(GREEDY, B_MAIN, b)]["successes"] / results[(GREEDY, B_MAIN, b)]["n"]
    SL = {b: (_slack(x) if slack_value is None else float(slack_value(x))) for b, x in br.items()}
    has_t03 = any(k[0] == T03 and k[1] == B_MAIN for k in results)

    names = ["matched", "greedy", "oracle", "oracle_slack", "random", "dec_deploy", "dec_oracle_random",
             "last_recoverable", "latest_significant", "earliest_sufficient"] + [f"k{k}" for k in KS] + \
            (["t03"] if has_t03 else []) + (list(extra.names) if extra is not None else [])
    V = {k: np.full((n_boot + 1, nT), np.nan) for k in names}

    def score(Pa, Po):
        return Pa - Po if net else Pa

    for ti, traj in enumerate(trajs):
        cands = [(a, p["orig"], p["t"], p["decisive"]) for p in traj["positions"] for a in p["alts"]]
        bids = sorted({b for a, o, _, _ in cands for b in (a, o)})
        col = {b: j for j, b in enumerate(bids)}
        ia = np.array([col[a] for a, *_ in cands])
        io = np.array([col[o] for _, o, *_ in cands])
        tpos = np.array([t for *_, t, _ in cands])
        dec = np.array([dd for *_, dd in cands])
        # Two-fold reduction in conditions of deployment：(B+1, 2, n_branch)；raw[f][j] is the list of the discounted results（For k Sub-samples）
        P = np.empty((n_boot + 1, 2, len(bids)))
        raw = [[None] * len(bids), [None] * len(bids)]
        PT = np.empty((n_boot + 1, len(bids)))          # temperature 0.3 Select Batch
        for j, b in enumerate(bids):
            if br[b]["deterministic"] is not None:
                P[:, :, j] = br[b]["deterministic"]
                PT[:, j] = br[b]["deterministic"]
                continue
            res = results[(DEPLOY, B_MAIN, b)]["results"]
            h = len(res) // 2
            for f, part in enumerate((res[:h], res[h:])):
                raw[f][j] = part
                k, n = sum(part), len(part)
                P[0, f, j] = k / n
                P[1:, f, j] = rng.binomial(n, (k + 0.5) / (n + 1), size=n_boot) / n
            if has_t03:
                rt = results[(T03, B_MAIN, b)]
                k, n = rt["successes"], rt["n"]
                PT[0, j] = k / n
                PT[1:, j] = rng.binomial(n, (k + 0.5) / (n + 1), size=n_boot) / n
        R = score(P[:, :, ia], P[:, :, io])
        Rall = R.mean(axis=1)

        def crossfit(score_fn):
            v = 0
            for f in (0, 1):
                v = v + _mv(score_fn(f), R[:, 1 - f, :])
            return v / 2

        neg = -1e9
        V["matched"][:, ti] = crossfit(lambda f: _best(R[:, f, :]))
        V["last_recoverable"][:, ti] = crossfit(lambda f: _best(np.where(dec, R[:, f, :], neg)))
        rec_alt = np.array([O[a] == 1.0 for a, *_ in cands]) & dec
        if rec_alt.sum() >= 2:
            V["dec_deploy"][:, ti] = crossfit(lambda f: _best(np.where(rec_alt, R[:, f, :], neg)))
            V["dec_oracle_random"][:, ti] = (Rall * rec_alt).sum(axis=-1) / rec_alt.sum()
        arr = lambda M: np.array([score(M[a], M[o]) for a, o, *_ in cands])
        so = arr(O)
        V["oracle"][:, ti] = _mv(_best(so)[None, :], Rall)
        V["oracle_slack"][:, ti] = _mv(_best(so * 1000 + arr(SL))[None, :], Rall)
        V["greedy"][:, ti] = _mv(_best(arr(G))[None, :], Rall)
        V["random"][:, ti] = Rall.mean(axis=-1)
        if has_t03:
            V["t03"][:, ti] = _mv(_best(score(PT[:, ia], PT[:, io])), Rall)

        # k Sub-samples：Select the front of the observed k runs，bootstrap No re-sampling（Conservative）
        for k in KS:
            vals = 0
            for f in (0, 1):
                def pk(b):
                    x = raw[f][col[b]]
                    return br[b]["deterministic"] if x is None else sum(x[:k]) / k
                sc = np.array([score(pk(a), pk(o)) for a, o, *_ in cands])
                vals = vals + _mv(_best(sc)[None, :], R[:, 1 - f, :])
            V[f"k{k}"][:, ti] = vals / 2

        # Positioning rules（Secondary analysis）：in fold f make the decision on，Another discounted evaluation
        def latest_sig(f):
            sig_t = []
            for c, (a, o, t, _) in enumerate(cands):
                xa, xo = raw[f][col[a]], raw[f][col[o]]
                ka, na = (br[a]["deterministic"], None) if xa is None else (sum(xa), len(xa))
                ko, no = (br[o]["deterministic"], None) if xo is None else (sum(xo), len(xo))
                if _z_sig(ka, na, ko, no):
                    sig_t.append(t)
            sc = R[0, f, :]
            m = _best(np.where(tpos == max(sig_t), sc, neg)) if sig_t else _best(sc)
            return np.broadcast_to(m, (n_boot + 1, len(cands)))

        def earliest_suff(f):
            # AgenTracer Inspired：Scan by location，First“Single replay successful”Location；average ties among actions with a single successful replay at that position
            first = {}
            for c, (a, o, t, _) in enumerate(cands):
                xa = raw[f][col[a]]
                ok = br[a]["deterministic"] == 1.0 if xa is None else bool(xa[0])
                if ok:
                    first.setdefault(t, []).append(c)
            m = np.zeros(len(cands))
            if first:
                m[first[min(first)]] = 1.0
            else:
                m[:] = 1.0      # Degradation to Random when not found
            return np.broadcast_to(m, (n_boot + 1, len(cands)))

        V["latest_significant"][:, ti] = crossfit(latest_sig)
        V["earliest_sufficient"][:, ti] = crossfit(earliest_suff)
        # Ex post facto selection rules（methods_gw.py）；Default not to enable，The results of the registered analysis remain unchanged
        if extra is not None:
            for name, val in extra(traj=traj, cands=cands, raw=raw, R=R, Rall=Rall, col=col, br=br).items():
                V[name][:, ti] = val

    idx = np.random.default_rng(seed + 1).integers(0, nT, size=(n_boot, nT))
    ftypes = np.array([t["fail_reason"] for t in trajs])

    def contrast(a, b, fail_type=None):
        """fail_type The given time is calculated only within the failed type of trajectory（Description Layer，Revision 2）。"""
        diff = V[a] - V[b]
        if fail_type is None:
            sub_idx = idx
        else:
            cols = np.flatnonzero(ftypes == fail_type)
            if len(cols) == 0:
                return None
            diff = diff[:, cols]
            sub_idx = np.random.default_rng(seed + 7).integers(0, len(cols), size=(n_boot, len(cols)))
        if np.all(np.isnan(diff[0])):
            return {"point": float("nan"), "ci90": [float("nan"), float("nan")], "p_one_sided": float("nan"),
                    "n_traj": 0}
        point = float(np.nanmean(diff[0]))
        means = np.nanmean(np.take_along_axis(diff[1:], sub_idx, axis=1), axis=1)
        means = means[~np.isnan(means)]
        lo, hi = float(np.quantile(means, ALPHA)), float(np.quantile(means, 1 - ALPHA))
        p_one = float((np.sum(means <= 0) + 1) / (len(means) + 1))
        # Aligning deviations（2026-10-23 After-action，Registered ci90 / p_one_sided No change）：parameters bootstrap use (k+0.5)/(n+1) Shrink，
        # The success rate is close 0 or 1 when bootstrap The worlds true values are systematically below point estimates，Move as a whole down by the percentile（ALFWorld the point estimate can even fall outside the interval）。
        # Cross-programming estimates in bootstrap unbiased within that simulated world，bootstrap Average bootstrap The real value of the world；Move the distribution evenly to the center as a point。
        shift = float(np.mean(means)) - point
        p_bc = float((np.sum(means - shift <= 0) + 1) / (len(means) + 1))
        return {"point": point, "ci90": [lo, hi], "p_one_sided": p_one, "n_traj": int(np.sum(~np.isnan(diff[0]))),
                "ci90_bc": [lo - shift, hi - shift], "p_bc": p_bc, "boot_shift": shift}

    return V, names, contrast


def analyze(d, results, n_boot=N_BOOT, seed=0):
    V, names, contrast = _selection_values(d, results, "net", n_boot, seed)
    out = {
        "objective": "net（Main objective：net gain）",
        "C1": contrast("dec_deploy", "dec_oracle_random"),
        "C2": contrast("matched", "greedy"),
        "C3": contrast("matched", "oracle"),
        "C4": contrast("matched", "k3"),
        "C5": contrast("oracle_slack", "oracle"),
        "positive_control": contrast("matched", "random"),
        "secondary": {
            "matched_minus_oracle_slack": contrast("matched", "oracle_slack"),
            "matched_minus_k5": contrast("matched", "k5"),
            "matched_minus_k10": contrast("matched", "k10"),
            "matched_minus_last_recoverable": contrast("matched", "last_recoverable"),
            "matched_minus_latest_significant": contrast("matched", "latest_significant"),
            "matched_minus_earliest_sufficient": contrast("matched", "earliest_sufficient"),
        },
        "values": {k: float(np.nanmean(V[k][0])) for k in names},
    }
    if "t03" in names:
        out["C6"] = contrast("matched", "t03")
    pc_ok = out["positive_control"]["ci90"][0] > DELTA
    out["positive_control"]["verdict"] = "passed" if pc_ok else "failed：insufficient discriminative signal，All conclusions are not valid"
    # Holm（primary tests）
    order = sorted(PRIMARY, key=lambda h: out[h]["p_one_sided"])
    m, rejected_all = len(order), True
    for i, h in enumerate(order):
        thr = ALPHA / (m - i)
        rej = rejected_all and out[h]["p_one_sided"] <= thr
        out[h]["holm_threshold"] = thr
        out[h]["holm_reject"] = bool(rej)
        rejected_all = rejected_all and rej
    for h in ("C1", "C2", "C3", "C4", "C5", "C6"):
        if h not in out:
            continue
        v = out[h]
        lo, hi = v["ci90"]
        if not pc_ok:
            v["verdict"] = "insufficient discriminative signal"
        elif h in PRIMARY:
            v["verdict"] = ("confirmed（Holm Correct）" if v["holm_reject"] else "not confirmed（Holm Correct）") + \
                           ("；lower bound > 0.10" if lo > DELTA else "")
        elif h == "C6":
            v["verdict"] = ("Wrong choice（lower bound > 0）" if lo > 0 else
                            "at δ Inside can be considered safe（upper bound < 0.10）" if hi < DELTA else "inconclusive（Inter-district crossing δ）")
        else:
            v["verdict"] = ("confirmed（lower bound > 0）" if lo > 0 else "not confirmed") + ("；lower bound > 0.10" if lo > DELTA else "")

    # Secondary objectives：Success rate after restoration（The ranking of net proceeds may differ when moving across location）
    Vp, names_p, contrast_p = _selection_values(d, results, "post", n_boot, seed + 10)
    post = {"positive_control": contrast_p("matched", "random"),
            "matched_minus_greedy": contrast_p("matched", "greedy"),
            "matched_minus_oracle": contrast_p("matched", "oracle"),
            "matched_minus_k3": contrast_p("matched", "k3"),
            "oracle_slack_minus_oracle": contrast_p("oracle_slack", "oracle")}
    if "t03" in names_p:
        post["matched_minus_t03"] = contrast_p("matched", "t03")
    post["values"] = {k: float(np.nanmean(Vp[k][0])) for k in names_p}
    out["post_objective"] = post
    # Revision 2：Layer by Failed Type，Just a description，No tests
    pairs = {"C1": ("dec_deploy", "dec_oracle_random"), "C2": ("matched", "greedy"), "C3": ("matched", "oracle"),
             "C4": ("matched", "k3"), "C5": ("oracle_slack", "oracle"), "positive_control": ("matched", "random")}
    if "t03" in names:
        pairs["C6"] = ("matched", "t03")
    out["by_fail_type"] = {ft: {h: contrast(a, b, fail_type=ft) for h, (a, b) in pairs.items()}
                           for ft in sorted({t["fail_reason"] for t in d["trajectories"]})}
    out["M1"] = mechanism_m1(d, results)
    out["n_trajectories"] = len(d["trajectories"])
    out["design_id"] = d["design_id"]
    return out


def _repeat_after_block(traces):
    blocked = rep = 0
    for tr in traces:
        for i in range(len(tr) - 1):
            if tr[i][2] in ("wall", "boundary", "locked_door"):
                blocked += 1
                rep += int(tr[i + 1][0] == tr[i][0] and tr[i + 1][1] == tr[i][1])
    return rep, blocked


def mechanism_m1(d, results):
    by_traj = {}
    for (c, b, bid), r in results.items():
        if b != B_MAIN or "traces" not in r:
            continue
        tid = bid.split("|")[0]
        rep, blk = _repeat_after_block(r["traces"])
        x = by_traj.setdefault(tid, {"greedy": [0, 0], "deploy": [0, 0]})
        key = "greedy" if c == GREEDY else "deploy"
        x[key][0] += rep
        x[key][1] += blk
    g = [v["greedy"] for v in by_traj.values()]
    s = [v["deploy"] for v in by_traj.values()]
    rate = lambda xs: (sum(a for a, _ in xs) / max(1, sum(b for _, b in xs)))
    rng = np.random.default_rng(1)
    keys = list(by_traj)
    diffs = []
    for _ in range(2000):
        pick = [by_traj[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        diffs.append(rate([p["greedy"] for p in pick]) - rate([p["deploy"] for p in pick]))
    return {"greedy_repeat_after_block": rate(g), "greedy_blocked_events": sum(b for _, b in g),
            "deploy_repeat_after_block": rate(s), "deploy_blocked_events": sum(b for _, b in s),
            "diff_ci90": [float(np.quantile(diffs, ALPHA)), float(np.quantile(diffs, 1 - ALPHA))]}


def report(res, screen_info):
    def row(h):
        v = res[h]
        return (f"| {h}{'（primary）' if h in PRIMARY else ''} | {HYPOTHESES[h]} | {DIMENSION.get(h, '—')} | {v['point']:+.3f} | "
                f"[{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}] | {v['p_one_sided']:.4f} | {v['n_traj']} | {v['verdict']} |")

    pc = res["positive_control"]
    L = ["# Gridworld Confirming test results\n",
         f"- collect {screen_info['n_collected']} episodes（Success {screen_info['n_success']}）；Filtering Candidates {screen_info['n_screen']} records"
         f"（{screen_info['fail_types']}）；Filtered {screen_info['n_informative']} records；Main experiment {res['n_trajectories']} records",
         "- Overall：Adoption of pre-established independent selection rules（Any alternative action on an irreversible point is on 10 successful continuations out of ≥ 3 runs）of "
         "trap_step / timeout_return Failed trajectories。screening itself is noisy，The conclusion is only for this whole group。",
         "- The reference is under a limited sample“Select by deployment strategy”（Every discount 20 runs，Crossing），not a known true optimal repair。",
         f"- positive control（Selected according to deployment strategy − Random selection）：{pc['point']:+.3f} [{pc['ci90'][0]:+.3f}, {pc['ci90'][1]:+.3f}] → {pc['verdict']}",
         "- Revision of registration：" + "；".join(AMENDMENTS) + "\n",
         "## Main objective：net gain（p(Substitution) − p(original action)）\n",
         "| Test | content | Dimensions | A point estimate | 90% interval | One side p | trajectories | judgments |",
         "|---|---|---|---:|---|---:|---:|---|"]
    for h in ("C1", "C2", "C3", "C6", "C4", "C5"):
        if h in res:
            L.append(row(h))
    m = res["M1"]
    L.append(f"\n**M1 Mechanisms**：Hit the wall/Proportion of repeats the same action after crashing the door：greedy {m['greedy_repeat_after_block']:.3f}"
             f"（{m['greedy_blocked_events']} runs）vs Sample {m['deploy_repeat_after_block']:.3f}（{m['deploy_blocked_events']} runs）；"
             f"Difference 90% interval [{m['diff_ci90'][0]:+.3f}, {m['diff_ci90'][1]:+.3f}]")
    L.append("\n**Layer by Failed Type**（Revision 2，Description Only、No tests；Fewer samples per type，Wider area）：\n")
    types = list(res["by_fail_type"])
    L.append("| Test | " + " | ".join(f"{ft}：A point estimate [90% interval]（trajectories）" for ft in types) + " |")
    L.append("|---|" + "---|" * len(types))
    for h in ("C1", "C2", "C3", "C6", "C4", "C5", "positive_control"):
        cells = []
        for ft in types:
            v = res["by_fail_type"][ft].get(h)
            cells.append("—" if (v is None or v["n_traj"] == 0) else
                         f"{v['point']:+.3f} [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}]（{v['n_traj']}）")
        if any(c != "—" for c in cells) or h in res:
            L.append(f"| {h} | " + " | ".join(cells) + " |")
    L.append("\n**Secondary analysis**（No multicomparison correction done，Description Only）：\n")
    L.append("| comparison | A point estimate | 90% interval |")
    L.append("|---|---:|---|")
    for k, v in res["secondary"].items():
        L.append(f"| {k} | {v['point']:+.3f} | [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}] |")
    L.append("\n Average net gains per selection rule：" + "；".join(f"{k} {v:+.3f}" for k, v in res["values"].items()))
    po = res["post_objective"]
    L.append("\n## Secondary objectives：Success rate after restoration（p(Substitution)）\n")
    L.append("When cross-position comparisons，Net gains may differ in the ranking of post-remediation success rates；The following selection is based on the repair success rate、Evaluation by post-repair success rate。\n")
    L.append("| comparison | A point estimate | 90% interval |")
    L.append("|---|---:|---|")
    for k, v in po.items():
        if k != "values":
            L.append(f"| {k} | {v['point']:+.3f} | [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}] |")
    L.append("\n Average post-repair success rate per selection rule：" + "；".join(f"{k} {v:+.3f}" for k, v in po["values"].items()))
    L.append("\n“Selected according to deployment strategy”Cross-drafting with positioning rules（One-for-one choice、Another discounted evaluation）；Inter-trajectories are trajectories + Branch-by-Segment Arguments bootstrap；"
             "k The sub-agents choose to use the pre-observed k runs，bootstrap No re-sampling（Conservative）；primary tests C1–C3 Do it Holm Correct，The rest is not corrected。")
    return "\n".join(L) + "\n"


# ================================================================ The entrance
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["collect", "screen", "prep", "main", "analyze", "run", "all"],
                    help="prep = collect + screen；run = main + analyze")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--n-collect", type=int, default=N_COLLECT, help="For simulation only；Please use the default for the full run")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    P = _paths(out)
    pool = PolicyPool(mock=args.mock)
    stages = {"all": ["collect", "screen", "main", "analyze"], "prep": ["collect", "screen"],
              "run": ["main", "analyze"]}.get(args.stage, [args.stage])

    if "collect" in stages:
        if args.mock:
            actor, pid = _mock_actor(), "mock"
        else:
            pol = pool.get(DEPLOY)
            actor, pid = _llm_actor(pol, config.CONDITIONS[DEPLOY]["temperature"]), pol.cache_id
        collect(P["episodes"], actor, pid, n=args.n_collect)
        recs = D.load_records(P["episodes"])
        print(f"[collect] Total {len(recs)} episodes，Success {sum(r['success'] for r in recs)}；Failed Type",
              dict(Counter(r["fail_reason"] for r in recs if not r["success"])))

    if "screen" in stages:
        recs = D.load_records(P["episodes"])
        sd = build_screen(recs)
        if P["screen_design"].exists() and json.loads(P["screen_design"].read_text(encoding="utf-8"))["design_id"] != sd["design_id"]:
            sys.exit("The filter design is different from the current one；Please change --out-dir")
        P["screen_design"].write_text(json.dumps(sd, ensure_ascii=False), encoding="utf-8")
        print(f"[screen] candidate {len(sd['trajectories'])} records，Random Branch "
              f"{sum(b['deterministic'] is None for b in sd['branches'].values())} items × {N_SCREEN} runs")
        runner.run_condition(sd, DEPLOY, B_SCREEN, pool.get(DEPLOY), n=N_SCREEN, out_path=P["screen"], mock=args.mock)
        inf = screen_result(sd, runner.load_results(P["screen"]))
        ftype = {t["traj_id"]: t["fail_reason"] for t in sd["trajectories"]}
        by_type = {ft: {"candidates": sum(v == ft for v in ftype.values()),
                        "informative": sum(ftype[i] == ft for i in inf)} for ft in FAIL_TYPES}
        info = {"n_collected": len(recs), "n_success": sum(r["success"] for r in recs),
                "fail_reason_counts": dict(Counter(r["fail_reason"] for r in recs if not r["success"])),
                "n_screen": len(sd["trajectories"]), "fail_types": list(FAIL_TYPES),
                "n_informative": len(inf), "by_type": by_type, "informative": inf,
                "will_run_main": min(len(inf), N_MAIN_MAX)}
        P["screen_result"].write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[screen] trajectories with signals {len(inf)} records（Threshold ≥ {SCREEN_MIN_SUCC}/{N_SCREEN}）；By Type：{by_type}")
        print(f"[screen] Collection success rate {info['n_success']}/{len(recs)}；Failed Type {info['fail_reason_counts']}")
        print(f"[screen] If you enter the main experiment,，Will be used {info['will_run_main']} trajectories；Results written {P['screen_result']}")
        if len(inf) < MIN_INFORMATIVE:
            sys.exit(f"[screen] fewer informative trajectories than {MIN_INFORMATIVE} records：Stop by pre-rule，Do not enter the main experiment。")

    if "main" in stages:
        recs = D.load_records(P["episodes"])
        info = json.loads(P["screen_result"].read_text(encoding="utf-8"))
        if info["n_informative"] < MIN_INFORMATIVE:
            sys.exit("Filter failed，Do not run main experiment")
        md = build_main(recs, info["informative"])
        if P["main_design"].exists() and json.loads(P["main_design"].read_text(encoding="utf-8"))["design_id"] != md["design_id"]:
            sys.exit("The main experiment design is different from the current one；Please change --out-dir")
        P["main_design"].write_text(json.dumps(md, ensure_ascii=False), encoding="utf-8")
        nst = sum(b["deterministic"] is None for b in md["branches"].values())
        print(f"[main] {len(md['trajectories'])} trajectories，Random Branch {nst} items；7B continuation {nst * (N_DEPLOY + N_GREEDY + N_T03)} records")
        runner.run_condition(md, DEPLOY, B_MAIN, pool.get(DEPLOY), n=N_DEPLOY, out_path=P["main"], mock=args.mock,
                             keep_traces=TRACE_DEPLOY)
        runner.run_condition(md, GREEDY, B_MAIN, pool.get(GREEDY), n=N_GREEDY, out_path=P["main"], mock=args.mock,
                             keep_traces=N_GREEDY)
        runner.run_condition(md, T03, B_MAIN, pool.get(T03), n=N_T03, out_path=P["main"], mock=args.mock)

    if "analyze" in stages:
        md = json.loads(P["main_design"].read_text(encoding="utf-8"))
        info = json.loads(P["screen_result"].read_text(encoding="utf-8"))
        res = analyze(md, runner.load_results(P["main"]))
        P["analysis"].write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        rep = report(res, info)
        if args.mock:
            rep = "> **Simulate Run**：7B By the noise oracle Replace，Numbers have no scientific meaning。\n\n" + rep
        P["report"].write_text(rep, encoding="utf-8")
        print(rep)


if __name__ == "__main__":
    main()
