import argparse
import json
from pathlib import Path
from .dataset import Dataset
from .io import read_json, write_json, write_jsonl


def main(argv=None):
    ap = argparse.ArgumentParser(prog="repair-eval")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate", help="Verify hashes, joins and complete outcome coverage")
    p.add_argument("bundle")
    p = sub.add_parser("import", help="Create a new portable snapshot from an existing experiment")
    p.add_argument("--source", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("export-tasks", help="Export only public, outcome-free selector inputs")
    p.add_argument("bundle")
    p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate", help="Evaluate a selector or fixed predictions")
    p.add_argument("bundle")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--selector", help="Python module:function")
    g.add_argument("--predictions")
    p.add_argument("--uses-rollouts", action="store_true")
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--splits", type=int, default=50)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metric", choices=["success", "reward"])
    p.add_argument("--objective", choices=["net", "post"], default="net")
    p.add_argument("--out", required=True)
    p = sub.add_parser("baselines", help="Budget curves and substitute-executor losses")
    p.add_argument("bundle")
    p.add_argument("--profile", choices=["standard","legacy69"], default="standard")
    p.add_argument("--metric", choices=["success","reward"])
    p.add_argument("--objective", choices=["net","post"], default="net")
    p.add_argument("--out", required=True)
    p = sub.add_parser("reproduce", help="Recompute primary results and compare with original outputs")
    p.add_argument("--data", default="data")
    p.add_argument("--reference", default="provenance/paper_reference.json")
    p.add_argument("--out", default="outputs/reproduction")
    p = sub.add_parser("plan-replay", help="Create reproducible jobs; does not run an environment/model")
    p.add_argument("bundle")
    p.add_argument("--executor", required=True)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--batch", default="RE1")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run-replay", help="Run a prepared plan with an explicit backend")
    p.add_argument("plan")
    p.add_argument("--backend", required=True, help="module:factory implementing run(jobs)")
    p.add_argument("--config", required=True, help="JSON backend configuration")
    p.add_argument("--out", required=True)
    p = sub.add_parser("attach-replay", help="Create a new bundle including a completed replay")
    p.add_argument("bundle")
    p.add_argument("--plan", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == "validate":
            result = Dataset(args.bundle).validate()
        elif args.command == "import":
            from .importer import import_pool
            result = import_pool(args.source, args.out, args.pool)
        elif args.command == "export-tasks":
            ds = Dataset(args.bundle)
            out = Path(args.out)
            if out.exists():
                raise ValueError("Task export directory already exists")
            write_jsonl(out / "tasks.jsonl", ds.tasks)
            write_json(out / "predictions.template.json", {"bundle_id": ds.manifest["bundle_id"],
                                                          "choices": {t["id"]: None for t in ds.tasks}})
            result = {"tasks": len(ds.tasks), "output": str(out)}
        elif args.command == "evaluate":
            from .evaluate import evaluate, load_selector
            result = evaluate(Dataset(args.bundle), selector=load_selector(args.selector) if args.selector else None,
                              predictions=read_json(args.predictions) if args.predictions else None,
                              uses_rollouts=args.uses_rollouts, budget=args.budget, metric=args.metric,
                              objective=args.objective, splits=args.splits, seed=args.seed, replicates=args.bootstrap)
            write_json(args.out, result)
            result = {k:v for k,v in result.items() if k != "rows"}
        elif args.command == "baselines":
            from .paper import analyze
            result = analyze(Dataset(args.bundle), args.profile, args.metric, args.objective)
            write_json(args.out, result)
            result = {"tests":result["tests"], "saved":args.out}
        elif args.command == "reproduce":
            from .paper import reproduce
            result = reproduce(args.data, args.reference, args.out)
            print(json.dumps({k:result[k] for k in ["passed","pending_tests","comparisons"]}, indent=2))
            return 0 if result["passed"] else 1
        elif args.command == "plan-replay":
            from .replay import plan_replay
            result = plan_replay(Dataset(args.bundle), args.executor, args.model, args.samples, args.batch, args.out)
        elif args.command == "run-replay":
            from .replay import run_replay
            result = run_replay(args.plan, args.backend, read_json(args.config), args.out)
        elif args.command == "attach-replay":
            from .replay import attach_replay
            result = attach_replay(Dataset(args.bundle), args.plan, args.results, args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, ImportError) as exc:
        ap.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
