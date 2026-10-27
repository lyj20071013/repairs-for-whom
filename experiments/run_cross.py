"""Minimum cross-test main entrance。

  python run_cross.py --stage selftest                 # none GPU：environment/oracle/Intervention/seed/LLM Interface/Full Process Simulation
  python run_cross.py --stage design                   # Freeze design（Environment Only oracle），Write out/design.json
  python run_cross.py --stage plan                     # Print the number of postbars for each condition，Estimated time
  python run_cross.py --stage rollouts                 # by config.BATCHES Run on all terms（7B Load only once）
  python run_cross.py --stage rollouts --condition llm_T03 --batch S   # Just one
  python run_cross.py --stage inspect                  # Success rate distribution and floor effects check for results obtained
  python run_cross.py --stage analyze                  # Analysis of relocation losses，Write analysis.json / report.md
  python run_cross.py --stage all                      # design -> rollouts -> analyze

  --mock：7B The conditions are changed to noise oracle，For use in none GPU The machine runs through the process（The results have no scientific meaning）。
  --out-dir：Output Directory，default out/。Once youve stopped, you can repeat the same order。
"""
import argparse
import json
import sys
from pathlib import Path

import config


def _load_design():
    if not config.DESIGN_JSON.exists():
        sys.exit("Nothing design.json，Run first --stage design")
    return json.loads(config.DESIGN_JSON.read_text(encoding="utf-8"))


def stage_design(source=None, force=False):
    import design as D
    d = D.build_design(source=source)
    if config.DESIGN_JSON.exists() and not force:
        old = json.loads(config.DESIGN_JSON.read_text(encoding="utf-8"))
        if old["design_id"] != d["design_id"]:
            sys.exit("Existing design.json Different from regenerated design；To protect the running results，Please change --out-dir or add --force")
    config.OUT.mkdir(parents=True, exist_ok=True)
    config.DESIGN_JSON.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(D.summarize(d), ensure_ascii=False, indent=2))
    print(f"design_id = {d['design_id'][:16]}… Write {config.DESIGN_JSON}")


def stage_plan():
    import runner
    p = runner.plan(_load_design())
    print(f"Random Branch {p['stochastic_branches']} items；Average surplus budget from branch starting point {p['mean_remaining_budget']:.1f} steps")
    for cond, batch, kind, n, count in p["rows"]:
        print(f"  {cond:18s} {batch}  {kind:13s} Each branch {n:4d} runs  Total {count:7d} Keep running")
    print(f"7B Total follow-up {p['llm_rollouts_total']} records。pilot of out_120 Main Curve approximation 16,992 records，It can be converted in accordance with its time-consuming nature。")


def stage_rollouts(only_cond=None, only_batch=None, mock=False):
    import runner
    from policies import PolicyPool
    d = _load_design()
    pool = PolicyPool(mock=mock)
    jobs = [(c, b) for c, bs in config.BATCHES.items() for b in bs]
    if only_cond:
        jobs = [(c, b) for c, b in jobs if c == only_cond and (only_batch is None or b == only_batch)]
        if not jobs:
            sys.exit(f"config.BATCHES None {only_cond}/{only_batch}")
    # CPU Well run first，7B The conditions are in the back，Load models only once。
    jobs.sort(key=lambda cb: config.CONDITIONS[cb[0]]["kind"] == "llm")
    for cond, batch in jobs:
        runner.run_condition(d, cond, batch, pool.get(cond), mock=mock)


def stage_analyze(mock=False):
    import analysis
    from runner import load_results
    d = _load_design()
    res = analysis.analyze(d, load_results(config.ROLLOUTS_JSONL))
    res["design_id"] = d["design_id"]
    res["mock"] = mock
    config.ANALYSIS_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = analysis.build_report(res, d, mock=mock)
    config.REPORT_MD.write_text(md, encoding="utf-8")
    print(md)
    print(f"Write {config.ANALYSIS_JSON} and {config.REPORT_MD}")


def stage_inspect():
    import analysis
    from runner import load_results
    rows = analysis.inspect(_load_design(), load_results(config.ROLLOUTS_JSONL))
    if not rows:
        sys.exit("It hasnt worked out yet")
    print(f"δ = {config.DELTA}")
    print("Condition/Batch                Completed Branch Each branch  Average p  p=0 Percentage  Original action average p  R>δPercentage  R<-δPercentage  malformed/continuation  Average number of steps")
    for r in rows:
        mo = "—" if r["mean_p_orig"] is None else f"{r['mean_p_orig']:.3f}"
        gt = "—" if r["frac_candidates_R_gt_delta"] is None else f"{r['frac_candidates_R_gt_delta']:.3f}"
        lt = "—" if r["frac_candidates_R_lt_minus_delta"] is None else f"{r['frac_candidates_R_lt_minus_delta']:.3f}"
        print(f"{r['condition'] + '/' + r['batch']:24s} {r['branches_done']:8d} {r['n_per_branch']:6d} "
              f"{r['mean_p']:6.3f} {r['frac_p_zero']:8.3f} {mo:>11s} {gt:>8s} {lt:>9s} "
              f"{r['malformed_per_rollout']:14.3f} {r['mean_steps']:8.1f}")
    print("\n if p=0 the proportion is high and R>δ the proportion is low，Means that deployment strategies have little resilience in these situations，"
          "The differences between the conditions cannot be detected；The floor effect should be dealt with first，Well run the whole experiment again。")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["selftest", "design", "plan", "rollouts", "inspect", "analyze", "all"])
    ap.add_argument("--condition")
    ap.add_argument("--batch")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--source", type=Path, help="Freezing trajectories，default pilot_vshape/out_120/episodes.jsonl")
    ap.add_argument("--force", action="store_true", help="Overwrite Different design.json（invalidates existing continuation caches）")
    ap.add_argument("--n-traj", type=int, help="Overwrite config.N_TRAJ，For example, first use 5 trial trajectories")
    args = ap.parse_args()
    if args.out_dir:
        config.set_output_dir(args.out_dir)
    if args.n_traj:
        config.N_TRAJ = args.n_traj

    if args.stage == "selftest":
        import selftest
        selftest.main()
    elif args.stage == "design":
        stage_design(args.source, args.force)
    elif args.stage == "plan":
        stage_plan()
    elif args.stage == "rollouts":
        stage_rollouts(args.condition, args.batch, args.mock)
    elif args.stage == "inspect":
        stage_inspect()
    elif args.stage == "analyze":
        stage_analyze(args.mock)
    elif args.stage == "all":
        stage_design(args.source, args.force)
        stage_plan()
        stage_rollouts(mock=args.mock)
        stage_analyze(args.mock)


if __name__ == "__main__":
    main()
