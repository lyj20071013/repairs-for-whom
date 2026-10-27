"""Analysis of relocation losses。

For each trajectory，Candidatures Gathered I = {(position t, Alternative Actions a')}。One“Select Source” s（A batch of certain conditions，
Or before m Subsampling）Gives an estimated score for each candidate，The highest-scoring candidate î(s)（Take the average of the groupings in parallel rows）。
All options are deployed c0 Independent Authentication Batch V Up-to-date evaluation：

    value_V(s)            = R_V(î(s))                          Net proceeds of restoration under conditions of deployment selected
    loss_vs_matched(s)    = R_V(î(c0,S)) - R_V(î(s))          Relative“Use the deployment conditions themselves”Losses
    regret_naive(s)       = max_i R_V(i) - R_V(î(s))          Relative V Max. value（Maximum for the same batch，Large，For reference only）

the primary comparison is loss_vs_matched：Both options are evaluated in the same independent batch，Noise selection has been factored in，
the difference does not require V Max on，So you dont“Winners curse”Impact。

interval：Re-sampling by trajectories，Simultaneous re-sampling of the success of each random branch（Two distributions，
use (k+0.5)/(n+1) Avoid 0/1 Deterioration），Selection and evaluation are recalculated in each re-sampled sample。
"""
import json
import math

import numpy as np

import config
from runner import load_results


# ---------------------------------------------------------------- Data Load
class Source:
    """A source of selection：Condition + Batch + optionally use the first m Subsampling。"""

    def __init__(self, cond, batch, m=None):
        self.cond, self.batch, self.m = cond, batch, m

    @property
    def label(self):
        return f"{self.cond}/{self.batch}" + (f"[:{self.m}]" if self.m else "")


def default_sources():
    srcs = [Source(c, "S") for c in config.BATCHES if "S" in config.BATCHES[c]]
    srcs += [Source(config.DEPLOY, "S", m) for m in config.PRECISION_PROXIES]
    return srcs


def is_deterministic_condition(cond):
    """oracle and greedy decoding have no sampling randomness，Their estimates are processed at precise values，bootstrap No re-sampling。"""
    spec = config.CONDITIONS.get(cond, {})
    return spec.get("kind") == "oracle" or (spec.get("kind") == "llm" and spec.get("temperature") == 0)


def load_counts(design, results, cond, batch, m=None):
    """Back {branch_id: (k, n)}；n=None Indicates the exact value（Determining branch or condition of certainty）。Missing branch returns None。"""
    exact = is_deterministic_condition(cond)
    out = {}
    for bid, br in design["branches"].items():
        if br["deterministic"] is not None:
            out[bid] = (br["deterministic"], None)
            continue
        rec = results.get((cond, batch, bid))
        if rec is None:
            return None
        res = rec["results"][:m] if m else rec["results"]
        out[bid] = (sum(res) / len(res), None) if exact else (sum(res), len(res))
    return out


def _p(kn):
    k, n = kn
    return float(k) if n is None else k / n


# ---------------------------------------------------------------- Selection and evaluation
def candidates(traj):
    return [(alt, pos["orig"], pos["t"]) for pos in traj["positions"] for alt in pos["alts"]]


def scores(cands, P, objective):
    """P: {branch_id: Probability}。Returns the candidate score array。"""
    alt = np.array([P[a] for a, _, _ in cands])
    if objective == "post":
        return alt
    orig = np.array([P[o] for _, o, _ in cands])
    return alt - orig


def tied_best(score_matrix):
    """score_matrix: (B, n_cand)。Returns the top of the row 0/1 Mask。"""
    best = score_matrix.max(axis=1, keepdims=True)
    return (np.abs(score_matrix - best) <= 1e-12).astype(float)


def masked_mean(mask, values):
    """mask: (B, n_cand)，values: (B, n_cand) or (n_cand,)。"""
    return (mask * values).sum(axis=1) / mask.sum(axis=1)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    # Average & Handle Parallel
    for arr, r in ((a, ra), (b, rb)):
        for v in np.unique(arr):
            idx = arr == v
            r[idx] = r[idx].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- Main Analysis
def analyze(design, results, sources=None, objective=None, delta=None, alpha=None,
            n_boot=None, seed=None):
    objective = objective or config.OBJECTIVE
    delta = config.DELTA if delta is None else delta
    alpha = config.ALPHA if alpha is None else alpha
    n_boot = config.N_BOOTSTRAP if n_boot is None else n_boot
    seed = config.BOOTSTRAP_SEED if seed is None else seed
    sources = sources or default_sources()
    matched = Source(config.DEPLOY, "S")

    valid = load_counts(design, results, config.DEPLOY, config.VALID_BATCH)
    if valid is None:
        raise ValueError("Lack of deployment conditions validation batch V The full result，Unable to evaluate")
    counts = {}
    for src in sources:
        c = load_counts(design, results, src.cond, src.batch, src.m)
        if c is not None:
            counts[src.label] = c
    if matched.label not in counts:
        raise ValueError("Batch selection for lack of deployment conditions S，Could not calculate loss relative to match selection")
    sources = [s for s in sources if s.label in counts]

    trajs = design["trajectories"]
    cand_lists = [candidates(t) for t in trajs]
    pos_of = {(t["traj_id"], a): ti for t, cl in zip(trajs, cand_lists) for a, _, ti in cl}
    pos_of = {a: ti for (_, a), ti in pos_of.items()}
    rng = np.random.default_rng(seed)

    # —— A point estimate ——
    PV = {b: _p(kn) for b, kn in valid.items()}
    point = {}
    per_traj = {}
    for src in sources:
        P = {b: _p(kn) for b, kn in counts[src.label].items()}
        rows = []
        for traj, cands in zip(trajs, cand_lists):
            s = scores(cands, P, objective)[None, :]
            mask = tied_best(s)
            rv_net = scores(cands, PV, "net")
            rv_post = scores(cands, PV, "post")
            chosen = [cands[j] for j in np.flatnonzero(mask[0])]
            rows.append({
                "traj_id": traj["traj_id"],
                "value_net": float(masked_mean(mask, rv_net)[0]),
                "value_post": float(masked_mean(mask, rv_post)[0]),
                "best_naive_net": float(rv_net.max()),
                "n_tied": int(mask.sum()),
                "chosen": [c[0] for c in chosen],
                "chosen_positions": sorted({c[2] for c in chosen}),
                "spearman_vs_V": _spearman(s[0], rv_net if objective == "net" else rv_post),
            })
        per_traj[src.label] = rows

    base_rows = {r["traj_id"]: r for r in per_traj[matched.label]}
    key = "value_net" if objective == "net" else "value_post"
    for src in sources:
        rows = per_traj[src.label]
        losses = [base_rows[r["traj_id"]][key] - r[key] for r in rows]

        # When each side chooses one randomly and evenly in a parallel collection，Pick the same candidate / Probability of the same location。
        # （checking only whether tie sets overlap severely inflates agreement when ties are numerous。）
        def p_same(a, b):
            return len(set(a) & set(b)) / (len(a) * len(b))

        def p_same_pos(a_cands, b_cands):
            def dist(cs):
                out = {}
                for c in cs:
                    out[pos_of[c]] = out.get(pos_of[c], 0) + 1 / len(cs)
                return out
            pa, pb = dist(a_cands), dist(b_cands)
            return sum(v * pb.get(t, 0) for t, v in pa.items())

        same_cand = [p_same(r["chosen"], base_rows[r["traj_id"]]["chosen"]) for r in rows]
        same_pos = [p_same_pos(r["chosen"], base_rows[r["traj_id"]]["chosen"]) for r in rows]
        sp = [r["spearman_vs_V"] for r in rows if r["spearman_vs_V"] is not None]
        point[src.label] = {
            "value_net": float(np.mean([r["value_net"] for r in rows])),
            "value_post": float(np.mean([r["value_post"] for r in rows])),
            "loss_vs_matched": float(np.mean(losses)),
            "regret_naive_net": float(np.mean([r["best_naive_net"] - r["value_net"] for r in rows])),
            "agree_candidate": float(np.mean(same_cand)),
            "agree_position": float(np.mean(same_pos)),
            "mean_spearman_vs_V": float(np.mean(sp)) if sp else None,
            "mean_tied": float(np.mean([r["n_tied"] for r in rows])),
        }
    # Randomly Candidate Baseline：Balanced on the candidates
    rand_vals = [float(np.mean(scores(c, PV, objective))) for c in cand_lists]
    point["random_candidate"] = {"value_" + objective: float(np.mean(rand_vals)),
                                 "loss_vs_matched": float(np.mean(
                                     [base_rows[t["traj_id"]][key] - v for t, v in zip(trajs, rand_vals)]))}

    # —— bootstrap ——
    def draw(kn_map, bids):
        """Back (n_boot, len(bids)) Probability matrix of the；A determinative branch is a constant column。"""
        mat = np.empty((n_boot, len(bids)))
        for j, b in enumerate(bids):
            k, n = kn_map[b]
            if n is None:
                mat[:, j] = k
            else:
                p_gen = (k + 0.5) / (n + 1)
                mat[:, j] = rng.binomial(n, p_gen, size=n_boot) / n
        return mat

    traj_vals = {src.label: np.empty((n_boot, len(trajs))) for src in sources}
    rand_traj = np.empty((n_boot, len(trajs)))
    for ti, cands in enumerate(cand_lists):
        bids = sorted({b for a, o, _ in cands for b in (a, o)})
        col = {b: j for j, b in enumerate(bids)}
        ia = np.array([col[a] for a, _, _ in cands])
        io = np.array([col[o] for _, o, _ in cands])
        V = draw(valid, bids)
        v_eval = V[:, ia] - V[:, io] if objective == "net" else V[:, ia]
        rand_traj[:, ti] = v_eval.mean(axis=1)
        for src in sources:
            S = draw(counts[src.label], bids)
            s = S[:, ia] - S[:, io] if objective == "net" else S[:, ia]
            traj_vals[src.label][:, ti] = masked_mean(tied_best(s), v_eval)
    idx = rng.integers(0, len(trajs), size=(n_boot, len(trajs)))  # Re-sampling of trajectories
    lo_q, hi_q = alpha, 1 - alpha
    base_boot = np.take_along_axis(traj_vals[matched.label], idx, axis=1).mean(axis=1)

    # positive control：Losses on random selection of candidates relative to matching selection。only if this is confirmed > δ，Experiments are the only ones who can identify poor choices，
    # “Equivalent”is the conclusion informative；otherwise（For example, the deployment strategy has been almost successful 0）It can only be judged“insufficient discriminative signal”。
    rand_loss = base_boot - np.take_along_axis(rand_traj, idx, axis=1).mean(axis=1)
    rand_ci = [float(np.quantile(rand_loss, lo_q)), float(np.quantile(rand_loss, hi_q))]
    discriminative = rand_ci[0] > delta
    intervals = {}
    for src in sources:
        val = np.take_along_axis(traj_vals[src.label], idx, axis=1).mean(axis=1)
        loss = base_boot - val
        ci = [float(np.quantile(loss, lo_q)), float(np.quantile(loss, hi_q))]
        # Main issues（Prior selection）：Can you measure it? ≥ δ Losses on relocation；Can you rule it out? ≥ δ Losses。No, no, no, no, no“Equivalent”。
        if src.label == matched.label:
            verdict = "References"
        elif ci[0] > delta:
            verdict = "Meaningful loss（lower bound > δ）"
        elif ci[0] > 0:
            verdict = "significantly worse，But no confirmation of excess δ"
        elif ci[1] < delta:
            verdict = ("rules out ≥ δ Losses（upper bound < δ）" if discriminative
                       else "insufficient discriminative signal（The randomly selected loss is not confirmed > δ，It cant be ruled out）")
        else:
            verdict = "inconclusive（Inter-district crossing δ）"
        intervals[src.label] = {
            "value_ci": [float(np.quantile(val, lo_q)), float(np.quantile(val, hi_q))],
            "loss_vs_matched_ci": ci,
            "verdict": verdict,
        }

    # —— Measuring reliability：V Two-and-a-half-point consistency between the batches ——
    split = []
    for cands in cand_lists:
        halves = []
        for part in (0, 1):
            P = {}
            for b, (k, n) in valid.items():
                if n is None:
                    P[b] = float(k)
                else:
                    res = results[(config.DEPLOY, config.VALID_BATCH, b)]["results"]
                    h = res[: len(res) // 2] if part == 0 else res[len(res) // 2:]
                    P[b] = sum(h) / len(h)
            halves.append(scores(cands, P, objective))
        r = _spearman(*halves)
        if r is not None:
            split.append(r)

    # —— Coverage ——
    rv_by_traj = [scores(c, PV, "net") for c in cand_lists]
    coverage = {
        "trajectories": len(trajs),
        "candidates": int(sum(len(c) for c in cand_lists)),
        "deterministic_branches": int(sum(b["deterministic"] is not None for b in design["branches"].values())),
        "traj_with_beneficial_candidate_V(>δ)": int(sum((r > delta).any() for r in rv_by_traj)),
        "traj_all_candidates_nonpositive_V": int(sum((r <= 0).all() for r in rv_by_traj)),
    }

    point["random_candidate"]["loss_vs_matched_ci"] = rand_ci
    return {
        "objective": objective, "delta": delta, "alpha": alpha, "n_boot": n_boot,
        "matched_source": matched.label,
        "positive_control": {"random_loss_ci": rand_ci, "discriminative": bool(discriminative)},
        "point": point, "intervals": intervals, "per_traj": per_traj,
        "V_split_half_mean_spearman": float(np.mean(split)) if split else None,
        "coverage": coverage,
        "sources": [s.label for s in sources],
    }


# ---------------------------------------------------------------- Floor effects check
def inspect(design, results, delta=None):
    """For any existing (Condition, Batch) describe the results，no validation batch required，It can be used after a small test run。

    Focus：Is the success rate of random branches significant? 0（Floor effects）；Estimated net proceeds from the recovery of candidates exceeded δ Percentage。
    If the vast majority of branches are 0，Conditions“It looks like the same price”merely because there is no distinguishable signal，The difficulty of the task or the budget should be addressed first。
    """
    delta = config.DELTA if delta is None else delta
    keys = sorted({(c, b) for c, b, _ in results})
    rows = []
    for cond, batch in keys:
        recs = [results[(cond, batch, bid)] for bid, br in design["branches"].items()
                if br["deterministic"] is None and (cond, batch, bid) in results]
        if not recs:
            continue
        ps = sorted(r["successes"] / r["n"] for r in recs)
        P = {}
        for bid, br in design["branches"].items():
            if br["deterministic"] is not None:
                P[bid] = br["deterministic"]
            elif (cond, batch, bid) in results:
                r = results[(cond, batch, bid)]
                P[bid] = r["successes"] / r["n"]
        Rs = [P[a] - P[o] for t in design["trajectories"] for a, o, _ in candidates(t) if a in P and o in P]
        orig_ps = [P[p["orig"]] for t in design["trajectories"] for p in t["positions"] if p["orig"] in P]
        rows.append({
            "condition": cond, "batch": batch, "branches_done": len(recs), "n_per_branch": recs[0]["n"],
            "mean_p": sum(ps) / len(ps), "frac_p_zero": sum(p == 0 for p in ps) / len(ps),
            "median_p": ps[len(ps) // 2], "mean_p_orig": (sum(orig_ps) / len(orig_ps)) if orig_ps else None,
            "frac_candidates_R_gt_delta": (sum(r > delta for r in Rs) / len(Rs)) if Rs else None,
            "frac_candidates_R_lt_minus_delta": (sum(r < -delta for r in Rs) / len(Rs)) if Rs else None,
            "malformed_per_rollout": sum(r["malformed"] for r in recs) / sum(r["n"] for r in recs),
            "mean_steps": sum(r["mean_steps"] for r in recs) / len(recs),
        })
    return rows


# ---------------------------------------------------------------- report
def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def build_report(res, design, mock=False):
    obj = res["objective"]
    L = []
    if mock:
        L.append("> **Simulate Run**：7B The conditions are noisey oracle Replace，Numbers only for checking process，no scientific interpretation。\n")
    L.append("# Rehabilitate selected conditions migration：Results report\n")
    L.append(f"- Objective：`{obj}`（net = Substitute branch success rate − Success rate of the original action branch；post = Substitute branch success rate）")
    L.append(f"- Conditions of deployment c0 = `{config.DEPLOY}`；Evaluation approvals = `{config.VALID_BATCH}`；Match Selection = `{res['matched_source']}`")
    L.append(f"- δ = {res['delta']}；Zone As {int(100 * (1 - 2 * res['alpha']))}% Cluster + parameters bootstrap（{res['n_boot']} runs）")
    L.append(f"- design：{json.dumps(res['coverage'], ensure_ascii=False)}")
    L.append(f"- Deployment of two-and-a-half candidate scores for validation batch Spearman Mean：{_fmt(res['V_split_half_mean_spearman'])}（Measuring reliability references）")
    pc = res["positive_control"]
    L.append(f"- **positive control**：Random selection of candidate relative to the loss range selected [{_fmt(pc['random_loss_ci'][0])}, "
             f"{_fmt(pc['random_loss_ci'][1])}]；" + (
                 "lower bound > δ，Experiments recognize poor choices，“rules out ≥ δ Losses”The conclusions are valid。" if pc["discriminative"] else
                 "**The bottom line is not exceeded δ：The experiment doesnt have the ability to identify the poor options（Common cause is the general proximity to the success rate of the deployment strategy 0），"
                 "This report cannot be used to state that any assessment conditions are safe。**"))
    L.append("")
    L.append("## Performance of selected sources under deployment conditions\n")
    L.append("| Select Source | Net proceeds of recovery selected | interval | Losses relative to matching selection | interval | judgments | Probability of co-opting | Probability of the same position | Average number of parallels |")
    L.append("|---|---:|---|---:|---|---|---:|---:|---:|")
    for label in res["sources"]:
        p, ci = res["point"][label], res["intervals"][label]
        L.append(f"| `{label}` | {_fmt(p['value_' + obj])} | [{_fmt(ci['value_ci'][0])}, {_fmt(ci['value_ci'][1])}] | "
                 f"{_fmt(p['loss_vs_matched'])} | [{_fmt(ci['loss_vs_matched_ci'][0])}, {_fmt(ci['loss_vs_matched_ci'][1])}] | "
                 f"{ci['verdict']} | {_fmt(p['agree_candidate'], 2)} | {_fmt(p['agree_position'], 2)} | {_fmt(p['mean_tied'], 1)} |")
    rc = res["point"]["random_candidate"]
    rci = rc.get("loss_vs_matched_ci", [None, None])
    L.append(f"| Random Candidate（positive control） | {_fmt(rc['value_' + obj])} | — | {_fmt(rc['loss_vs_matched'])} | "
             f"[{_fmt(rci[0])}, {_fmt(rci[1])}] | — | — | — | — |")
    L.append("\n“Losses relative to matching selection”> 0 Expressed that the rehabilitation selected by the source was less effective than the deployment itself under conditions of deployment（Same number of samples）Selected repairs。")
    L.append("“Same candidate/Probability of the same position”：When each side randomly chooses one of the best pairs，Pick the same candidate/Probability of the same location（Average for trajectories）。")
    n_inf = res["coverage"]["traj_with_beneficial_candidate_V(>δ)"]
    L.append(f"trajectories with signals（Net gain from the deployment of at least one candidate in the certification batch > δ）Total {n_inf} records；Any one-step restoration on the rest of the trajectories is almost ineffective for deployment strategy，"
             "All the sources losses on these trajectories are close 0，The average loss will be directed to 0 Dim。")
    L.append("`regret_naive`（Relative V Max. value）Bigger than the same batch，only in analysis.json reported in，Undecided conclusion。")
    return "\n".join(L) + "\n"
