"""Mechanism follow-up experiments：greedy picks the wrong fixes，Is it because certainty decoding repeats the same mistake?？

Confirming the experiment（out_confirm/）Same one 34 trajectories、762 On a random branch，Add a follow-up strategy：
  greedy + Run away（llm_greedy_escape）：greedy by default；If the current physical state（position、Keys、Gems、The door、Maps）
  already visited during this continuation，use this temperature for the current step 0.7 Sample。Hit the wall/hitting a door leaves the state unchanged，triggers immediately。
Deployment strategy（40 runs）、greedy（1 runs）Direct re-use of data out_confirm/main_rollouts.jsonl，Dont run again。
“greedy + Run away”Each branch 10 runs。

Prior registration（2026-10-22 21:05，Before running）：
  E1（primary）  by“greedy+Run away”10 Subselection − Choose with greedy > 0（90% Lower Inter-District Border > 0）。greedy is certain（The results of the multi-run are still the same），
            So this is a comparison of two strategies“Estimated objective”，not a sample-size comparison。
  E2        By deployment strategy 10 Subselection − by“greedy+Run away”10 Subselection：the interval is ±0.10 Internal → The post-exfiltration estimated target is in line with the deployment strategy
  E3        Recipe level：Rate of success gap in escape recovery = (p Run away − p greedy) / (p Deployment − p greedy)
  Reported separately（description）：1 runs、5 Similar comparison under sub-samples；escape trigger rate；Ratio of repeating the same action after crashing the wall
  Design correction record：First draft（20:55）use“1 Subsamples, etc. Budget”The master test；Simulation test shows deployment strategy 1 Loss of selection of secondary samples（approximately 0.37）
  More than greedy（approximately 0.23），i.e 1 The noise of secondary random samples hides the difference in the estimated target。Before running, insert 10 runs（no real data are present at this point）。
  Interpretation：E1 Confirm and E2 at ±0.10 Internal → “Duplicate error due to certainty”is the main reason greedy replay selects the wrong repair；
        only E1 confirmed → Part of the reason；E1 not confirmed → The main reason is not to repeat mistakes，instead attributable to the greedy action distribution itself。
  all“Selected according to deployment strategy”The rules are cross-referenced in two folds；All rules that do not depend on deployment data selection 40 Evaluation on sub-deployment sample。

  python mechanism.py --stage all      # approximately 30 minutes；results are in out_confirm/mechanism_report.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config
import confirm
import runner
from policies import PolicyPool

ESC = "llm_greedy_escape"
B_MECH = "MECH"
N_ESC = 10
TRACE_ESC = 2
DELTA = 0.10
ALPHA = 0.05
N_BOOT = 2000
REGISTERED = "2026-10-22 21:05"


def _p(rec, k=None):
    res = rec["results"][:k] if k else rec["results"]
    return sum(res) / len(res)


def analyze(d, main_res, mech_res, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    br = d["branches"]
    trajs = d["trajectories"]
    nT = len(trajs)
    names = ["matched", "greedy", "esc1", "esc5", "esc10", "dep1", "dep5", "dep10", "random"]
    V = {k: np.full((n_boot + 1, nT), np.nan) for k in names}

    for ti, traj in enumerate(trajs):
        cands = [(a, p["orig"]) for p in traj["positions"] for a in p["alts"]]
        bids = sorted({b for c in cands for b in c})
        col = {b: j for j, b in enumerate(bids)}
        ia = np.array([col[a] for a, _ in cands])
        io = np.array([col[o] for _, o in cands])
        P = np.empty((n_boot + 1, 2, len(bids)))       # 20% reduction in deployment（Evaluation and selection by deployment）
        raw = [[None] * len(bids), [None] * len(bids)]
        E1 = np.empty((n_boot + 1, len(bids)))          # escape at index 1 runs（0/1，Select Use）
        E5 = np.empty((n_boot + 1, len(bids)))          # before escape 5 runs
        E10 = np.empty((n_boot + 1, len(bids)))         # Run away 10 runs
        Gm = np.empty(len(bids))
        for j, b in enumerate(bids):
            det = br[b]["deterministic"]
            if det is not None:
                P[:, :, j] = det
                E1[:, j] = det
                E5[:, j] = det
                E10[:, j] = det
                Gm[j] = det
                continue
            res = main_res[(confirm.DEPLOY, confirm.B_MAIN, b)]["results"]
            h = len(res) // 2
            for f, part in enumerate((res[:h], res[h:])):
                raw[f][j] = part
                k, n = sum(part), len(part)
                P[0, f, j] = k / n
                P[1:, f, j] = rng.binomial(n, (k + 0.5) / (n + 1), size=n_boot) / n
            er = mech_res[(ESC, B_MECH, b)]["results"]
            # With deployment 1/5 runs、matches greedy behavior：Select to use observed samples，bootstrap do not resample within（Retrench evaluation and trajectories only）
            E1[:, j] = er[0]
            E5[:, j] = sum(er[:5]) / 5
            E10[:, j] = sum(er[:10]) / 10
            Gm[j] = main_res[(confirm.GREEDY, confirm.B_MAIN, b)]["successes"]
        R = P[:, :, ia] - P[:, :, io]
        Rall = R.mean(axis=1)

        def crossfit(fn):
            return sum(confirm._mv(fn(f), R[:, 1 - f, :]) for f in (0, 1)) / 2

        V["matched"][:, ti] = crossfit(lambda f: confirm._best(R[:, f, :]))
        V["greedy"][:, ti] = confirm._mv(confirm._best(Gm[ia] - Gm[io])[None, :], Rall)
        V["esc1"][:, ti] = confirm._mv(confirm._best(E1[:, ia] - E1[:, io]), Rall)
        V["esc5"][:, ti] = confirm._mv(confirm._best(E5[:, ia] - E5[:, io]), Rall)
        V["esc10"][:, ti] = confirm._mv(confirm._best(E10[:, ia] - E10[:, io]), Rall)
        for k, name in ((1, "dep1"), (5, "dep5"), (10, "dep10")):
            def sel(f, k=k):
                sc = np.array([(br[a]["deterministic"] if raw[f][col[a]] is None else sum(raw[f][col[a]][:k]) / k)
                               - (br[o]["deterministic"] if raw[f][col[o]] is None else sum(raw[f][col[o]][:k]) / k)
                               for a, o in cands])
                return np.broadcast_to(confirm._best(sc), (n_boot + 1, len(cands)))
            V[name][:, ti] = crossfit(sel)
        V["random"][:, ti] = Rall.mean(axis=-1)

    idx = np.random.default_rng(seed + 1).integers(0, nT, size=(n_boot, nT))

    def contrast(a, b):
        diff = V[a] - V[b]
        means = np.take_along_axis(diff[1:], idx, axis=1).mean(axis=1)
        return {"point": float(diff[0].mean()), "ci90": [float(np.quantile(means, ALPHA)),
                                                         float(np.quantile(means, 1 - ALPHA))],
                "p_one_sided": float((np.sum(means <= 0) + 1) / (len(means) + 1))}

    out = {"E1": contrast("esc10", "greedy"), "E2": contrast("dep10", "esc10"),
           "desc": {"esc1_minus_greedy": contrast("esc1", "greedy"), "dep1_minus_esc1": contrast("dep1", "esc1"),
                    "dep1_minus_greedy": contrast("dep1", "greedy"), "esc5_minus_greedy": contrast("esc5", "greedy"),
                    "dep5_minus_esc5": contrast("dep5", "esc5"), "matched_minus_esc10": contrast("matched", "esc10")},
           "matched_minus_greedy": contrast("matched", "greedy"),
           "positive_control": contrast("matched", "random"),
           "values": {k: float(V[k][0].mean()) for k in names}}
    e1, e2 = out["E1"]["ci90"], out["E2"]["ci90"]
    out["E1"]["verdict"] = "confirmed（lower bound > 0）" if e1[0] > 0 else "not confirmed"
    out["E2"]["verdict"] = ("the gap is ±0.10 Internal" if (e2[0] > -DELTA and e2[1] < DELTA)
                            else "Gaps exceeded or crossed ±0.10")
    if e1[0] > 0 and e2[0] > -DELTA and e2[1] < DELTA:
        out["interpretation"] = "Repetitious errors due to certainty are the main cause of the greedy of the wrong fixes"
    elif e1[0] > 0:
        out["interpretation"] = "Duplicate errors due to certainty are part of the reason"
    else:
        out["interpretation"] = "The main reason is not to repeat mistakes，instead attributable to the greedy action distribution itself"
    return out


def rollout_level(d, main_res, mech_res, seed=0):
    """E3：Success-to-recovery ratio at the continuity level（Cluster by trajectories bootstrap）。"""
    by_traj = {}
    for b, x in d["branches"].items():
        if x["deterministic"] is not None:
            continue
        tid = b.split("|")[0]
        g = main_res[(confirm.GREEDY, confirm.B_MAIN, b)]["successes"]
        dp = _p(main_res[(confirm.DEPLOY, confirm.B_MAIN, b)])
        e = _p(mech_res[(ESC, B_MECH, b)])
        by_traj.setdefault(tid, []).append((g, e, dp))
    keys = sorted(by_traj)

    def stats(ks):
        rows = [r for k in ks for r in by_traj[k]]
        g, e, dp = (float(np.mean([r[i] for r in rows])) for i in range(3))
        return g, e, dp, ((e - g) / (dp - g) if dp != g else float("nan"))

    g, e, dp, ratio = stats(keys)
    rng = np.random.default_rng(seed)
    boots = [stats([keys[i] for i in rng.integers(0, len(keys), len(keys))])[3] for _ in range(N_BOOT)]
    boots = [b for b in boots if not np.isnan(b)]
    return {"greedy": g, "escape": e, "deploy": dp, "recovery_ratio": ratio,
            "recovery_ci90": [float(np.quantile(boots, ALPHA)), float(np.quantile(boots, 1 - ALPHA))]}


def trace_stats(mech_res, main_res):
    def agg(recs):
        rep = blk = esc_steps = steps = revisit_end = n = 0
        for r in recs:
            for tr in r.get("traces", []):
                n += 1
                steps += len(tr)
                for i in range(len(tr) - 1):
                    if tr[i][2] in ("wall", "boundary", "locked_door"):
                        blk += 1
                        rep += int(tr[i + 1][0] == tr[i][0] and tr[i + 1][1] == tr[i][1])
        return {"rollouts_with_trace": n, "repeat_after_block": rep / max(blk, 1), "blocked_events": blk}
    esc = [r for k, r in mech_res.items() if k[0] == ESC]
    out = {"escape": agg(esc), "greedy": agg([r for k, r in main_res.items() if k[0] == confirm.GREEDY]),
           "deploy": agg([r for k, r in main_res.items() if k[0] == confirm.DEPLOY])}
    tot_steps = sum(r["mean_steps"] * r["n"] for r in esc)
    tot_esc = sum(r.get("escapes", 0) for r in esc)
    out["escape_trigger_rate_per_step"] = tot_esc / max(tot_steps, 1)
    return out


def report(res, roll, tr):
    v = res["values"]
    L = ["# Mechanism follow-up experiments：greedy + Run away\n",
         f"- Prior registration：{REGISTERED}（Before running）；The sample is the same as the confirmed experiment（34 trajectories、762 Random branch）",
         f"- positive control（Selected according to deployment strategy − Random selection）：{res['positive_control']['point']:+.3f} "
         f"[{res['positive_control']['ci90'][0]:+.3f}, {res['positive_control']['ci90'][1]:+.3f}]",
         f"- **Interpretation（By the pre-rule）：{res['interpretation']}**\n",
         "| Test | content | A point estimate | 90% interval | judgments |", "|---|---|---:|---|---|",
         f"| E1（primary） | With greedy+Run away 10 Subselection − Choose with greedy | {res['E1']['point']:+.3f} | "
         f"[{res['E1']['ci90'][0]:+.3f}, {res['E1']['ci90'][1]:+.3f}] | {res['E1']['verdict']} |",
         f"| E2 | By deployment strategy 10 Subselection − With greedy+Run away 10 Subselection | {res['E2']['point']:+.3f} | "
         f"[{res['E2']['ci90'][0]:+.3f}, {res['E2']['ci90'][1]:+.3f}] | {res['E2']['verdict']} |",
         f"| References | By deployment strategy 20 Subselection − Choose with greedy（Confirming the experiment C2） | {res['matched_minus_greedy']['point']:+.3f} | "
         f"[{res['matched_minus_greedy']['ci90'][0]:+.3f}, {res['matched_minus_greedy']['ci90'][1]:+.3f}] | — |",
         "\n**description：Trade-off between sample volume and certainty**\n", "| comparison | A point estimate | 90% interval |", "|---|---:|---|"] + [
         f"| {k} | {v['point']:+.3f} | [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}] |" for k, v in res["desc"].items()] + [
         "\n Rule selection to restore the average net gain under deployment conditions：" + "；".join(f"{k} {x:+.3f}" for k, x in v.items()),
         f"\n**E3 Recipe level**：Average success rate for random branches：greedy {roll['greedy']:.3f}，greedy+Run away {roll['escape']:.3f}，"
         f"Deployment of sampling {roll['deploy']:.3f}；Percentage gap in escape recovery {roll['recovery_ratio']:.2f} "
         f"[{roll['recovery_ci90'][0]:.2f}, {roll['recovery_ci90'][1]:.2f}]",
         f"\n**trajectories Record**：escape trigger rate {tr['escape_trigger_rate_per_step']:.3f}（Every step）；Hit the wall/Repeat the same action when you hit the door："
         f"greedy {tr['greedy']['repeat_after_block']:.3f}（{tr['greedy']['blocked_events']} runs），"
         f"greedy+Run away {tr['escape']['repeat_after_block']:.3f}（{tr['escape']['blocked_events']} runs），"
         f"Deployment of sampling {tr['deploy']['repeat_after_block']:.3f}（{tr['deploy']['blocked_events']} runs）",
         "\n Inter-trajectories are trajectories + Parameters for the deployment of the evaluation sample bootstrap；greedy、greedy+escape and deployed policies 1/5 The next choice is made using the observed samples，"
         "bootstrap do not resample within（Rules are treated consistently）。"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["rollouts", "analyze", "all"])
    ap.add_argument("--confirm-dir", type=Path, default=confirm.OUT)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    cdir = args.confirm_dir.resolve()
    P = confirm._paths(cdir)
    if not P["main_design"].exists():
        sys.exit(f"not found {P['main_design']}：requires the main confirmatory experiment data")
    d = json.loads(P["main_design"].read_text(encoding="utf-8"))
    mech_path = cdir / "mechanism_rollouts.jsonl"
    if args.stage in ("rollouts", "all"):
        pol = PolicyPool(mock=args.mock).get(ESC)
        runner.run_condition(d, ESC, B_MECH, pol, n=N_ESC, out_path=mech_path, mock=args.mock, keep_traces=TRACE_ESC)
    if args.stage in ("analyze", "all"):
        main_res = runner.load_results(P["main"])
        mech_res = runner.load_results(mech_path)
        res = analyze(d, main_res, mech_res)
        res["rollout_level"] = rollout_level(d, main_res, mech_res)
        res["traces"] = trace_stats(mech_res, main_res)
        (cdir / "mechanism_analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        rep = report(res, res["rollout_level"], res["traces"])
        if args.mock:
            rep = "> **Simulate Run**：7B By the noise oracle Replace，Numbers have no scientific meaning。\n\n" + rep
        (cdir / "mechanism_report.md").write_text(rep, encoding="utf-8")
        print(rep)


if __name__ == "__main__":
    main()
