"""Experimental design：Select trajectories、Select Intervention Location、enumerate candidate repairs，Generate all repeat branches。

All rules depend on the environment oracle with fixed seeds，Freeze and write before any model is renewed design.json。

Mark with pilot identical：states[t] after executing t Status after an action h_t；actions[t] copy h_t Become h_{t+1}。
“In Position t Intervention”Point at h_t Use alternative action a' Replace actions[t]。
  - Original Action Branch：from h_{t+1}（The next real state of affairs）continuation；
  - Alternative Branch：at h_t Implementation a' continue from the resulting state。
Both branches sampled，net gain R_t(a') = p(Substitution) - p(original action)。

The outcome of the branch is determined in two cases，no sampling required：
  - Branching is over（Step on the trap、No key to open the door、It just happened to be a success）；
  - The branch state budget is not clear（F=0）：The success rate of any strategy is exactly what it is 0。
"""
import json
import random
from pathlib import Path

import config
import oracle
from env import GridWorld


def load_records(path):
    recs = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def prior_inspected_ids(paths):
    """already inspected V Curve trajectory。Error reporting on missing files，Avoid mixing the examined trajectories into the official experiment。"""
    ids = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"not found {p}：cannot exclude previously inspected trajectories。Please confirm pilot_vshape/out_120 Full Synchronization。")
        data = json.loads(p.read_text(encoding="utf-8"))
        ids.update(data.get("curves", {}).keys())
    return ids


def action_key(action):
    tool = action.get("tool")
    arg = action.get("arg")
    return f"{tool}:{arg}" if arg else str(tool)


def verify_replay(rec):
    """Freezing trajectories must be able to be reset gradually by the current environment，If not, refuse to use。"""
    env = GridWorld.restore(rec["states"][0])
    for t, act in enumerate(rec["actions"]):
        env.step(act)
        if env.snapshot() != rec["states"][t + 1]:
            raise ValueError(f"{rec['traj_id']}: number {t + 1} The step reset is not consistent with the saving state")


def select_trajectories(records, n, seed, excluded):
    """Excludes the curves that have been seen in the failed trajectory，Randomly extracted by end-of-event ratio（Maximum residual distribution）。"""
    pool = sorted([r for r in records if not r["success"] and r["traj_id"] not in excluded],
                  key=lambda r: r["traj_id"])
    if len(pool) < n:
        raise ValueError(f"Available failure trajectories only {len(pool)} records，Less {n}")
    groups = {}
    for r in pool:
        groups.setdefault(r["fail_reason"], []).append(r)
    quotas = {k: n * len(v) / len(pool) for k, v in groups.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(seed)
    picked = []
    for k in sorted(groups):
        rows = groups[k][:]
        rng.shuffle(rows)
        picked.extend(rows[: alloc[k]])
    return sorted(picked, key=lambda r: r["traj_id"]), alloc


def select_positions(rec, k, seed):
    """at“Dissolved within pre-intervention state budget”Organisation k items。

    F(h_t)=0 No alternative action can be successful，the intervention is uninformative，So exclude。
    Rules：Always include the last solventable position；The rest divides the earlier solvency positions k-1 Paragraph，Pick one random piece。
    """
    T = len(rec["actions"])
    feasible_t = [t for t in range(T) if oracle.feasible(rec["states"][t])]
    if len(feasible_t) <= k:
        return feasible_t
    last = feasible_t[-1]
    earlier = feasible_t[:-1]
    rng = random.Random(f"{seed}|{rec['traj_id']}")
    chosen = []
    m = k - 1
    for b in range(m):
        lo = b * len(earlier) // m
        hi = (b + 1) * len(earlier) // m
        chosen.append(rng.choice(earlier[lo:hi]))
    return sorted(set(chosen)) + [last]


def _branch(branch_id, snap, traj_id, t, akey):
    if snap["done"]:
        det = 1.0 if snap["success"] else 0.0
    elif not oracle.feasible(snap):
        det = 0.0
    else:
        det = None
    return {"branch_id": branch_id, "traj_id": traj_id, "t": t, "action": akey,
            "snapshot": snap, "deterministic": det,
            "remaining_budget": snap["max_steps"] - snap["steps"],
            "oracle_distance": oracle.table_for(snap).distance(snap)}


def build_design(source=None, n_traj=None, n_positions=None):
    source = Path(source or config.SOURCE_EPISODES)
    n_traj = config.N_TRAJ if n_traj is None else n_traj
    n_positions = config.N_POSITIONS if n_positions is None else n_positions
    records = load_records(source)
    excluded = prior_inspected_ids(config.PRIOR_INSPECTED)
    for rec in records:
        verify_replay(rec)
    picked, alloc = select_trajectories(records, n_traj, config.TRAJ_SEED, excluded)

    branches, trajectories = {}, []
    for rec in picked:
        positions = []
        for t in select_positions(rec, n_positions, config.POSITION_SEED):
            factual = rec["actions"][t]
            orig_id = f"{rec['traj_id']}|t{t}|orig"
            branches[orig_id] = _branch(orig_id, rec["states"][t + 1], rec["traj_id"], t,
                                        "orig:" + action_key(factual))
            alts = []
            for cand in config.CANDIDATE_ACTIONS:
                if action_key(cand) == action_key(factual):
                    continue
                env = GridWorld.restore(rec["states"][t])
                env.step(dict(cand))
                bid = f"{rec['traj_id']}|t{t}|{action_key(cand)}"
                branches[bid] = _branch(bid, env.snapshot(), rec["traj_id"], t, action_key(cand))
                alts.append(bid)
            positions.append({"t": t, "orig": orig_id, "alts": alts,
                              "factual_action": factual,
                              "remaining_budget_before": rec["states"][t]["max_steps"] - rec["states"][t]["steps"]})
        trajectories.append({"traj_id": rec["traj_id"], "fail_reason": rec["fail_reason"],
                             "T": rec["T"], "positions": positions})

    design = {
        # Only file names and Hashi，Design signature remains unchanged when machine regenerates
        "source": {"file": source.name, "sha256": config.file_sha256(source),
                   "n_records": len(records), "excluded_prior_inspected": sorted(excluded)},
        "rules": {"n_traj": n_traj, "traj_seed": config.TRAJ_SEED, "allocation": alloc,
                  "n_positions": n_positions, "position_seed": config.POSITION_SEED,
                  "candidate_actions": config.CANDIDATE_ACTIONS},
        "code": {n: config.file_sha256(config.ROOT / n) for n in ("env.py", "oracle.py", "design.py")},
        "trajectories": trajectories,
        "branches": branches,
    }
    design["design_id"] = config.fingerprint({k: design[k] for k in ("source", "rules", "trajectories")})
    return design


def summarize(design):
    br = design["branches"].values()
    n_det = sum(b["deterministic"] is not None for b in br)
    stoch = [b for b in br if b["deterministic"] is None]
    n_pos = sum(len(t["positions"]) for t in design["trajectories"])
    n_cand = sum(len(p["alts"]) for t in design["trajectories"] for p in t["positions"])
    return {
        "trajectories": len(design["trajectories"]),
        "positions": n_pos,
        "candidates": n_cand,
        "branches": len(design["branches"]),
        "deterministic_branches": n_det,
        "stochastic_branches": len(stoch),
        "mean_remaining_budget_stochastic": (sum(b["remaining_budget"] for b in stoch) / len(stoch)) if stoch else 0,
        "allocation": design["rules"]["allocation"],
    }
