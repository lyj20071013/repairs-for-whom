"""Rapidly confirm the experiment（approximately 30–40 minutes）：Check on new trajectories v1 Three hypotheses from exploratory analysis。

v1（out/，20 trajectories）now used only as a development set；confirm using previously unused trajectories here。
Assumptions、trajectories Selection Rules、The rules of judgment are all written down below，fixed before model inference：

  H1 Recoverability is too thick：At the unreversible step，It can recover“within budget”Between Alternative Actions，7B The success rate is very different。
     Statistics = On that position“by 7B Select the best action”Relative“Random Selection in Resumeable Actions”Net income difference（Crossing）。
  H2 Irreversible points are not the best fixes：Allow repair in earlier locations，higher gain than repairing only at the irreversible step。
     Statistics = “Press in two positions 7B Pick the best”Relative“Press only in irreversible position 7B Pick the best”Net income difference（Crossing）。
  H3 greedy replay is inaccurate for selection：Estimation of selected restorations by greedy decoding，relative to selection by 7B Sample estimated loss of selected repairs。
  judgments：90% trajectories Cluster bootstrap Lower Inter-District Border > 0 i.e“confirmed”；At the same time, the reporting point estimates are consistent with δ=0.10 Relationship。
  positive control：Random Candidate Relative 7B Loss(s) of selection，Bottom bound > 0.10，otherwise classify the entire result as“insufficient discriminative signal”。

design：
  - trajectory：out_120 Can not open message、Not in v1 Failed trajectories；Only take timeout_return and trap_step
    （v1 middle timeout_before_door 5/5 For the floor）。timeout_return Take All，trap_step Randomly smoked with a fixed feed。
  - position：Irreversible Point t*（Solvable location in the last budget；After the original action F=0）＋ an earlier solvable position（Fixed Feed Random）。
  - candidate：All Legal Alternatives（Same v1）。
  - Sample：Conditions of deployment（7B, T=0.7）Each branch 40 runs，before 20 / after 20 Cross-drafting：
    A discount for selection、Another discount for evaluation，Ill trade it for averages；greedy for every branch 1 runs；oracle not used GPU。

  python quick.py --stage all          # design -> plan -> rollouts -> analyze，Output in out_quick/
  python quick.py --stage analyze      # Analysis only
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

import config
import design as D
import oracle
from env import GridWorld

QUICK_OUT = config.ROOT / "out_quick"
V1_DESIGN = config.ROOT / "out" / "design.json"     # It has to exist，For exclusion v1 Used trajectories
FAIL_TYPES_ALL = ["timeout_return"]                  # Take All
FAIL_TYPES_SAMPLED = {"trap_step": 8}                # Random strip extraction
SEED = 1022
N_DEPLOY = 40                                        # in each of the two folds 20
N_GREEDY = 1
DEPLOY, GREEDY = "deploy_T07", "llm_greedy"
BATCH = "Q"
DELTA = 0.10
ALPHA = 0.05
N_BOOT = 2000


# ---------------------------------------------------------------- design
def build(source=None):
    source = Path(source or config.SOURCE_EPISODES)
    recs = D.load_records(source)
    for r in recs:
        D.verify_replay(r)
    excluded = D.prior_inspected_ids(config.PRIOR_INSPECTED)
    if not V1_DESIGN.exists():
        raise FileNotFoundError(f"not found {V1_DESIGN}：It needs to be ruled out v1 Used trajectories")
    v1_ids = {t["traj_id"] for t in json.loads(V1_DESIGN.read_text(encoding="utf-8"))["trajectories"]}
    pool = sorted([r for r in recs if not r["success"] and r["traj_id"] not in excluded | v1_ids],
                  key=lambda r: r["traj_id"])
    rng = random.Random(SEED)
    picked = [r for r in pool if r["fail_reason"] in FAIL_TYPES_ALL]
    for ftype, k in FAIL_TYPES_SAMPLED.items():
        rows = [r for r in pool if r["fail_reason"] == ftype]
        rng.shuffle(rows)
        picked += rows[:k]
    picked.sort(key=lambda r: r["traj_id"])

    branches, trajs = {}, []
    for rec in picked:
        T = len(rec["actions"])
        feas = [t for t in range(T) if oracle.feasible(rec["states"][t])]
        t_star = feas[-1]
        earlier = [t for t in feas if t < t_star]
        prng = random.Random(f"{SEED}|{rec['traj_id']}")
        positions_t = ([prng.choice(earlier)] if earlier else []) + [t_star]
        positions = []
        for t in positions_t:
            factual = rec["actions"][t]
            oid = f"{rec['traj_id']}|t{t}|orig"
            branches[oid] = D._branch(oid, rec["states"][t + 1], rec["traj_id"], t, "orig:" + D.action_key(factual))
            alts = []
            for cand in config.CANDIDATE_ACTIONS:
                if D.action_key(cand) == D.action_key(factual):
                    continue
                env = GridWorld.restore(rec["states"][t])
                env.step(dict(cand))
                bid = f"{rec['traj_id']}|t{t}|{D.action_key(cand)}"
                branches[bid] = D._branch(bid, env.snapshot(), rec["traj_id"], t, D.action_key(cand))
                alts.append(bid)
            positions.append({"t": t, "orig": oid, "alts": alts, "decisive": t == t_star,
                              "factual_action": factual})
        trajs.append({"traj_id": rec["traj_id"], "fail_reason": rec["fail_reason"], "T": rec["T"],
                      "positions": positions})
    design = {
        "kind": "quick_confirmatory",
        "source": {"file": source.name, "sha256": config.file_sha256(source),
                   "excluded_prior_inspected": sorted(excluded), "excluded_v1": sorted(v1_ids)},
        "rules": {"fail_types_all": FAIL_TYPES_ALL, "fail_types_sampled": FAIL_TYPES_SAMPLED, "seed": SEED,
                  "n_deploy": N_DEPLOY, "n_greedy": N_GREEDY, "delta": DELTA, "alpha": ALPHA,
                  "candidate_actions": config.CANDIDATE_ACTIONS},
        "code": {n: config.file_sha256(config.ROOT / n) for n in ("env.py", "oracle.py", "design.py", "quick.py")},
        "trajectories": trajs, "branches": branches,
    }
    design["design_id"] = config.fingerprint({k: design[k] for k in ("source", "rules", "trajectories")})
    return design


# ---------------------------------------------------------------- Analysis
def _folds(design, results):
    """each branchs two folds (k, n)；deterministic branch n=None。"""
    out = {}
    for bid, br in design["branches"].items():
        if br["deterministic"] is not None:
            out[bid] = ((br["deterministic"], None), (br["deterministic"], None))
            continue
        rec = results.get((DEPLOY, BATCH, bid))
        if rec is None:
            raise ValueError(f"Lack of conditions for deployment：{bid}")
        res = rec["results"]
        h = len(res) // 2
        out[bid] = ((sum(res[:h]), h), (sum(res[h:]), len(res) - h))
    return out


def _greedy(design, results):
    out = {}
    for bid, br in design["branches"].items():
        if br["deterministic"] is not None:
            out[bid] = br["deterministic"]
        else:
            r = results.get((GREEDY, BATCH, bid))
            if r is None:
                raise ValueError(f"Lack of greedy results：{bid}")
            out[bid] = r["successes"] / r["n"]
    return out


def _best_mask(s):
    return (np.abs(s - s.max(axis=-1, keepdims=True)) <= 1e-12).astype(float)


def _mval(mask, v):
    return (mask * v).sum(axis=-1) / mask.sum(axis=-1)


def analyze(design, results, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    folds = _folds(design, results)
    G = _greedy(design, results)
    O = {b: (1.0 if oracle.feasible(br["snapshot"]) else 0.0) for b, br in design["branches"].items()}

    names = ["matched", "decisive_only", "decisive_deploy", "decisive_oracle_random",
             "oracle", "greedy", "random"]
    vals = {k: np.zeros((n_boot + 1, len(design["trajectories"]))) for k in names}   # number 0 Estimates of behavioural points
    informative, earlier_wins, per_traj = [], [], []

    for ti, traj in enumerate(design["trajectories"]):
        cands = [(a, p["orig"], p["t"], p["decisive"]) for p in traj["positions"] for a in p["alts"]]
        bids = sorted({b for a, o, _, _ in cands for b in (a, o)})
        col = {b: j for j, b in enumerate(bids)}
        ia = np.array([col[a] for a, *_ in cands]); io = np.array([col[o] for _, o, *_ in cands])
        dec = np.array([d for *_, d in cands])
        # Two-fold probabilities matrix：(n_boot+1, 2, n_branch)，number 0 Behavioral observations
        F = np.empty((n_boot + 1, 2, len(bids)))
        for j, b in enumerate(bids):
            for f in (0, 1):
                k, n = folds[b][f]
                if n is None:
                    F[:, f, j] = k
                else:
                    F[0, f, j] = k / n
                    F[1:, f, j] = rng.binomial(n, (k + 0.5) / (n + 1), size=n_boot) / n
        R = F[:, :, ia] - F[:, :, io]                         # (B+1, 2, n_cand) net gain
        # Crossing：in fold f Top Selection，evaluate on the other fold，Two-to-one average
        def crossfit(mask_fn):
            v = 0
            for f in (0, 1):
                m = mask_fn(R[:, f, :])
                v = v + _mval(m, R[:, 1 - f, :])
            return v / 2
        neg = -1e9
        vals["matched"][:, ti] = crossfit(_best_mask)
        vals["decisive_only"][:, ti] = crossfit(lambda s: _best_mask(np.where(dec, s, neg)))
        # H1：Compare only in irreversible position“by 7B Select Action”and“Random Selection in Resumeable Actions”
        rec_alt = np.array([O[a] == 1.0 for a, *_ in cands]) & dec
        if rec_alt.sum() >= 2:
            vals["decisive_deploy"][:, ti] = crossfit(lambda s: _best_mask(np.where(rec_alt, s, neg)))
            Rall = R.mean(axis=1)
            vals["decisive_oracle_random"][:, ti] = (Rall * rec_alt).sum(axis=-1) / rec_alt.sum()
        else:
            vals["decisive_deploy"][:, ti] = np.nan
            vals["decisive_oracle_random"][:, ti] = np.nan
        Rall = R.mean(axis=1)                                 # Two-fold merger（40 runs）Rules for evaluating non-dependency on deployment estimates
        so = np.array([O[a] - O[o] for a, o, *_ in cands])
        vals["oracle"][:, ti] = _mval(_best_mask(so)[None, :], Rall)
        sg = np.array([G[a] - G[o] for a, o, *_ in cands])
        vals["greedy"][:, ti] = _mval(_best_mask(sg)[None, :], Rall)
        vals["random"][:, ti] = Rall.mean(axis=-1)
        r0 = Rall[0]
        informative.append(bool(r0.max() > DELTA))
        e_best, d_best = r0[~dec].max() if (~dec).any() else -1, r0[dec].max()
        earlier_wins.append(bool(e_best > d_best + DELTA))
        per_traj.append({"traj_id": traj["traj_id"], "fail_reason": traj["fail_reason"],
                         "max_R": float(r0.max()), "best_earlier": float(e_best), "best_decisive": float(d_best),
                         "n_recoverable_alts_at_decisive": int(rec_alt.sum())})

    idx = rng.integers(0, len(design["trajectories"]), size=(n_boot, len(design["trajectories"])))

    def contrast(a, b, nan_ok=False):
        diff = vals[a] - vals[b]                              # (B+1, n_traj)
        point = float(np.nanmean(diff[0]))
        boot = np.take_along_axis(diff[1:], idx, axis=1)
        means = np.nanmean(boot, axis=1)
        lo, hi = float(np.nanquantile(means, ALPHA)), float(np.nanquantile(means, 1 - ALPHA))
        n_used = int(np.sum(~np.isnan(diff[0])))
        return {"point": point, "ci90": [lo, hi], "n_traj": n_used}

    res = {
        "H1_decisive_action_choice": contrast("decisive_deploy", "decisive_oracle_random"),
        "H2_earlier_position_gain": contrast("matched", "decisive_only"),
        "H3_greedy_loss": contrast("matched", "greedy"),
        "oracle_loss": contrast("matched", "oracle"),
        "positive_control_random_loss": contrast("matched", "random"),
    }
    pc_ok = res["positive_control_random_loss"]["ci90"][0] > DELTA
    for k, v in res.items():
        if k.startswith("positive"):
            v["verdict"] = "passed" if pc_ok else "failed：insufficient discriminative signal"
            continue
        lo, hi = v["ci90"]
        if not pc_ok:
            v["verdict"] = "insufficient discriminative signal"
        elif lo > DELTA:
            v["verdict"] = "confirmed，And down the line > 0.10"
        elif lo > 0:
            v["verdict"] = "confirmed（lower bound > 0）"
        elif hi < 0:
            v["verdict"] = "In the opposite direction"
        else:
            v["verdict"] = "not confirmed"
    res["informative_trajectories"] = f"{sum(informative)}/{len(informative)}"
    res["earlier_beats_decisive_by_delta"] = f"{sum(earlier_wins)}/{len(earlier_wins)}"
    res["per_traj"] = per_traj
    res["design_id"] = design["design_id"]
    return res


def report(res):
    L = ["# Rapidly confirm the results of the experiment（New trajectories，Pre-established assumptions and rules of judgments）\n",
         f"- trajectories with signals（At least one candidate net gain under deployment > 0.10）：{res['informative_trajectories']}",
         f"- The best fixes are earlier locations than the best fixes for irreversible points 0.10 More trajectories：{res['earlier_beats_decisive_by_delta']}\n",
         "| Test | A point estimate | 90% interval | trajectories | judgments |", "|---|---:|---|---:|---|"]
    labels = {"H1_decisive_action_choice": "H1 Unreversible point：by 7B Select Action − Random selection in recovery action",
              "H2_earlier_position_gain": "H2 Two positions selected − Select only irreversible points",
              "H3_greedy_loss": "H3 by 7B Sample Selection − select using greedy replay",
              "oracle_loss": "（References）by 7B Sample Selection − by oracle Selection（Parallel Average）",
              "positive_control_random_loss": "positive control：by 7B Sample Selection − Random selection"}
    for k, lab in labels.items():
        v = res[k]
        L.append(f"| {lab} | {v['point']:+.3f} | [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}] | {v['n_traj']} | {v['verdict']} |")
    L.append("\n all“by 7B Selection”The rules are drawn in two folds：One-for-one choice、Another discounted evaluation。Inter-trajectories are trajectories + Branch-by-Segment Arguments bootstrap。")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- The entrance
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["design", "plan", "rollouts", "analyze", "all"])
    ap.add_argument("--out-dir", type=Path, default=QUICK_OUT)
    ap.add_argument("--mock", action="store_true", help="7B Well change the noise oracle，Check process only")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dpath, rpath = out / "design.json", out / "rollouts.jsonl"
    import runner
    from policies import PolicyPool

    if args.stage in ("design", "all"):
        d = build()
        if dpath.exists() and json.loads(dpath.read_text(encoding="utf-8"))["design_id"] != d["design_id"]:
            sys.exit("Existing design.json Different from current design；Please change --out-dir")
        dpath.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        print(f"Quick Design：{len(d['trajectories'])} trajectories，design_id {d['design_id'][:12]}…")
    d = json.loads(dpath.read_text(encoding="utf-8"))
    if args.stage in ("plan", "all"):
        st = [b for b in d["branches"].values() if b["deterministic"] is None]
        from collections import Counter
        print("Failed Type：", dict(Counter(t["fail_reason"] for t in d["trajectories"])))
        print(f"Random Branch {len(st)} items；7B continuation {len(st) * (N_DEPLOY + N_GREEDY)} records"
              f"（pilot Test-in-the-art 4.4 records/seconds，approximately {len(st) * (N_DEPLOY + N_GREEDY) / 4.4 / 60:.0f} minutes，Faster when the remaining steps are less from the starting point）")
    if args.stage in ("rollouts", "all"):
        pool = PolicyPool(mock=args.mock)
        runner.run_condition(d, DEPLOY, BATCH, pool.get(DEPLOY), n=N_DEPLOY, out_path=rpath, mock=args.mock)
        runner.run_condition(d, GREEDY, BATCH, pool.get(GREEDY), n=N_GREEDY, out_path=rpath, mock=args.mock)
    if args.stage in ("analyze", "all"):
        res = analyze(d, runner.load_results(rpath))
        (out / "quick_analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        md = report(res)
        if args.mock:
            md = "> **Simulate Run**：7B By the noise oracle Replace，Numbers have no scientific meaning。\n\n" + md
        (out / "quick_report.md").write_text(md, encoding="utf-8")
        print(md)


if __name__ == "__main__":
    main()
