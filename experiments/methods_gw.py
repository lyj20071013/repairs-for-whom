"""Gridworld：published method“Evaluation segment”Put in a fixed candidate protocol to confirm the experiment，Measuring how much of their chosen repair is lost under deployment conditions。

After-action analysis（2026-10-22 23:30 Accession，After confirming that the results of the experiment had been released,；Marked as exploratory in the paper / descriptive），Dont run again 7B：
- The determination rules are fully calculated using the confirmed deployment samples available from the experiment（Cross-drafting：apply the rule on one fold，Another discounted evaluation）；
- Rec（Only judgment、Not implemented）You have to run first judge_gw.py，Read here out_confirm/judge_*.jsonl。

Rules（When no acceptable fixes are found“no repair”，Net deployment gains are recorded as 0；Coverage rate reported separately）：
  earliest_sufficient      Scan by location，First single place for success（AgenTracer / REFLECT Type）；Random when a registered version is not found
  earliest_suff_norepair   Ibid，Do not fix when not found
  flip1_any                accept if one replay succeeds，Random in accepted fixes（CausalFlow Type“Flip and repair”）
  dover_2of3               Every candidate runs again 3 runs，≥ 2 Second-success is acceptance，Random in accepted fixes（DoVer Authentication rules for）
  latest_significant       At the end of the night（CAR Type；20 Subsamples + One side z Test）；Registered
  judge:<name>             LLM Read log directly to select a fix，Not implemented

    python methods_gw.py [--out-dir out_confirm]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config
import confirm
import runner

RULE_DESC = {
    "earliest_sufficient": "First Full：A single replay is confirmed，Ordered by Location（AgenTracer/REFLECT Type；Random when not found，Registered）",
    "earliest_suff_norepair": "First Full：Ibid，Do not fix when not found",
    "flip1_any": "A single flip will be accepted，Random Among Recipients（CausalFlow Type）",
    "dover_2of3": "Rerun 3 runs ≥ 2 Second-success is acceptance，Random Among Recipients（DoVer Authentication rules）",
    "latest_significant": "At the end of the night（CAR Type，20 Subsamples，Registered）",
}
REFERENCE = ["greedy", "oracle", "oracle_slack", "k3", "k10", "random"]


class Extra:
    def __init__(self, judges):
        self.judges = judges
        self.names = ["earliest_suff_norepair", "flip1_any", "dover_2of3"] + \
            [x for n in judges for x in (f"judge:{n}", f"judge_valid:{n}")]
        self.coverage = {n: [] for n in self.names}
        self.judge_valid = {n: 0 for n in judges}

    def __call__(self, traj, cands, raw, R, Rall, col, br):
        nb = R.shape[0]

        def ksucc(f, b, k):
            x = raw[f][col[b]]
            return br[b]["deterministic"] * k if x is None else sum(x[:k])

        def crossfit(pick):
            v, cov = 0.0, 0.0
            for f in (0, 1):
                cs = pick(f)
                if cs:
                    v = v + R[:, 1 - f, cs].mean(axis=-1)
                    cov += 0.5
                else:
                    v = v + np.zeros(nb)
            return v / 2, cov

        out = {}
        rules = {
            "flip1_any": lambda f: [c for c, (a, *_r) in enumerate(cands) if ksucc(f, a, 1) >= 1],
            "dover_2of3": lambda f: [c for c, (a, *_r) in enumerate(cands) if ksucc(f, a, 3) >= 2],
        }

        def earliest(f):
            acc = [(t, c) for c, (a, o, t, _d) in enumerate(cands) if ksucc(f, a, 1) >= 1]
            if not acc:
                return []
            t0 = min(t for t, _ in acc)
            return [c for t, c in acc if t == t0]
        rules["earliest_suff_norepair"] = earliest
        for name, fn in rules.items():
            v, cov = crossfit(fn)
            out[name] = v
            self.coverage[name].append(cov)

        idx = {a: c for c, (a, *_r) in enumerate(cands)}
        for jn, picks in self.judges.items():
            b = picks.get(traj["traj_id"])
            if b in idx:
                out[f"judge:{jn}"] = Rall[:, idx[b]]
                out[f"judge_valid:{jn}"] = Rall[:, idx[b]]
                self.judge_valid[jn] += 1
                self.coverage[f"judge:{jn}"].append(1.0)
                self.coverage[f"judge_valid:{jn}"].append(1.0)
            else:                                   # Invalid answer：Main row by random selection；“Valid only”exclude this trajectory in that row
                out[f"judge:{jn}"] = Rall.mean(axis=-1)
                out[f"judge_valid:{jn}"] = np.full(nb, np.nan)
                self.coverage[f"judge:{jn}"].append(0.0)
        return out


def load_judges(out_dir):
    judges = {}
    for p in sorted(out_dir.glob("judge_*.jsonl")):
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if rows:
            judges[rows[0]["judge"]] = {r["traj_id"]: r["branch_id"] for r in rows}
    return judges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=confirm.OUT)
    ap.add_argument("--n-boot", type=int, default=confirm.N_BOOT)
    args = ap.parse_args()
    out = args.out_dir
    d = json.loads((out / "main_design.json").read_text(encoding="utf-8"))
    results = runner.load_results(out / "main_rollouts.jsonl")
    judges = load_judges(out)
    ex = Extra(judges)
    V, names, contrast = confirm._selection_values(d, results, "net", args.n_boot, 0, extra=ex)
    nT = len(d["trajectories"])
    rows = []
    order = ["matched"] + list(RULE_DESC) + [x for j in judges for x in (f"judge:{j}", f"judge_valid:{j}")] + REFERENCE
    order = [n for n in order if n in V]
    seen = set()
    for n in order:
        if n in seen:
            continue
        seen.add(n)
        c = None if n == "matched" else contrast("matched", n)
        cov = float(np.mean(ex.coverage[n])) if n in ex.coverage else 1.0
        cov = float(np.mean(ex.coverage[n])) if ex.coverage.get(n) else 1.0
        rows.append({"name": n, "value": float(np.nanmean(V[n][0])), "loss_vs_matched": c, "coverage": cov})
    res = {"design_id": d["design_id"], "n_trajectories": nT, "rows": rows,
           "judge_valid": ex.judge_valid, "judges": list(judges),
           "note": "After-action analysis（After confirmation of the results of the experiment, insert），exploratory；Do Not Fix As 0；The decision is not answered by random selection"}
    (out / "methods_analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# Gridworld：Evaluation of published methodologies（After-action analysis，exploratory）", "",
         f"Confirm the same experiment {nT} trajectories、Same group of candidates（4 Organisation × Replace All Actions），Cross-cutting evaluation of deployment conditions。"
         "“loss” = By deployment strategy 20 Net deployment gain from sub-matching selection − The methodology selects net deployment gains from restoration；90% Inter-trajectories are trajectories + parameters bootstrap。"
         "Do not fix if no acceptable fixes are found（Remember 0）；Coverage = Give the proportion of fixes。", "",
         "| Methodology（Evaluation segment） | Net deployment gains | Relatively matched loss | 90% interval | Coverage |", "|---|---:|---:|---|---:|"]
    desc = dict(RULE_DESC)
    desc.update({"matched": "Match Selection：Deployment strategy 20 runs（References）", "greedy": "greedy replay 1 runs", "oracle": "oracle Restorability",
                 "oracle_slack": "oracle + slack", "k3": "Deployment strategy 3 choose the best across these runs", "k10": "Deployment strategy 10 choose the best across these runs",
                 "random": "random（positive control）"})
    for j in judges:
        desc[f"judge:{j}"] = f"LLM Rec {j}：Read logs directly selected，Not implemented（Effective answer {ex.judge_valid[j]}/{nT}，score invalid responses as random selection）"
        desc[f"judge_valid:{j}"] = f"　Ibid，It only gives a valid answer {ex.judge_valid[j]} Comparison of trajectories"
    for r in rows:
        c = r["loss_vs_matched"]
        if c is None:
            L.append(f"| {desc.get(r['name'], r['name'])} | {r['value']:+.3f} | — | — | {r['coverage']:.2f} |")
        else:
            L.append(f"| {desc.get(r['name'], r['name'])} | {r['value']:+.3f} | {c['point']:+.3f} | "
                     f"[{c['ci90'][0]:+.3f}, {c['ci90'][1]:+.3f}] | {r['coverage']:.2f} |")
    if not judges:
        L += ["", "（judge results are not available yet：Run first judge_gw.py。）"]
    # Matching of the judges（Same trajectories、Same deployment sample）：Dissipation and Reference。Difference = Net gain on deployment of the presiding judge − Net gains on deployment compared
    picks = load_judges(out)
    pair_rows = []
    for j in judges:
        if not j.endswith("_think"):
            continue
        base = j[:-len("_think")]
        comps = [(f"judge:{j}_noexec", "Remove the executor information（Take it easy）", f"{j}_noexec"), (f"judge:{base}", "Close Thinking（Take it easy）", base),
                 ("oracle", "oracle Restorability", None), ("oracle_slack", "oracle + slack", None),
                 ("greedy", "greedy replay 1 runs", None), ("k3", "Deployment strategy 3 choose the best across these runs", None)]
        for other, lab, jn in comps:
            if other not in V:
                continue
            c = contrast(f"judge:{j}", other)
            same = ""
            if jn is not None:
                a, b = picks[j], picks[jn]
                same = f"{sum(a.get(t) is not None and a.get(t) == b.get(t) for t in a)}/{nT}"
            pair_rows.append(f"| {j} − {lab} | {c['point']:+.3f} | [{c['ci90'][0]:+.3f}, {c['ci90'][1]:+.3f}] | "
                             f"[{c['ci90_bc'][0]:+.3f}, {c['ci90_bc'][1]:+.3f}] | {same} |")
    if pair_rows:
        L += ["", "**Matching of the judges**（Difference = Thinking about the net deployment gains of the main judge − control；> 0 That means the Lords judges are better）：", "",
              "| comparison | Difference | 90% interval | deviation correction interval | Select Same |", "|---|---:|---|---|---:|"] + pair_rows
    (out / "methods_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
