"""Explicit historical profiles for numerical reproduction, not new-method tuning."""
from pathlib import Path
import numpy as np
from ._paper import current, legacy69
from .stats import bootstrap, holm
from .dataset import Dataset
from .io import read_json, write_json
from .io import seed_for


def analyze(ds, profile="standard", metric=None, objective="net", splits=50):
    metric = metric or ds.manifest["primary_metric"]
    if objective not in ("net", "post") or splits < 1:
        raise ValueError("Invalid objective/splits")
    deployer = ds.manifest["deployer"]
    td = ds.arrays(deployer, metric)
    other = {e: ds.arrays(e, metric) for e in ds.rollouts if e != deployer}
    if profile == "legacy69":
        if ds.manifest["pool_id"] != "alfworld-q7" or "q14" not in other:
            raise ValueError("legacy69 requires the ALFWorld Q7 pool and complete Q14 outcomes")
        rows = legacy69.analyze_core(td, other["q14"], ds.metadata, obj=objective, n_perm=splits)
    elif profile == "standard":
        for executor in other:
            if executor not in current.EXEC_CODE:
                current.EXEC_CODE[executor] = seed_for("executor-stream", executor)
        rows = current.analyze_core(td, other, ds.metadata, obj=objective, n_perm=splits)
    else:
        raise ValueError("Unknown numerical profile")
    # No NaN or unrelated exploratory summaries in the portable report.
    keep = ["matched", "greedy", "random", "planner_earliest", "budget_all"]
    keep += [f"{p}_k{k}" for p in ("self", "7B", "14B", *other) for k in (1,2,3,5,10,20)]
    clean = [{k: float(r[k]) for k in keep if k in r} for r in rows]
    stats = {k: bootstrap([r[k] for r in clean]) for k in clean[0] if k != "budget_all"}
    tests = {"positive_control": bootstrap([r["matched"]-r["random"] for r in clean]),
             "greedy_loss": bootstrap([r["matched"]-r["greedy"] for r in clean])}
    h = td[0]["S"].shape[1] // 2
    for ex in other:
        key = f"14B_k{h}" if profile == "legacy69" and ex == "q14" else f"{ex}_k{h}"
        if key in clean[0]:
            tests[f"{ex}_loss"] = bootstrap([r["matched"]-r[key] for r in clean])
    tests["positive_control"]["pass"] = tests["positive_control"]["ci"][0] > .10
    return {"pool": ds.manifest["pool_id"], "bundle_id": ds.manifest["bundle_id"],
            "profile": profile, "metric": metric, "objective": objective,
            "splits": splits, "seed": 0, "tests": tests, "curves": stats, "rows": clean,
            "pending_executors": ds.manifest.get("pending_executors", [])}


def reproduce(data_root, reference_file, output):
    refs = read_json(reference_file)
    results, tests, comparisons = {}, {}, []
    mapping = {
        "alfworld-q7": {"P1": "greedy_loss", "P2": "q14_loss"},
        "alfworld-l8": {"H2": "q7_loss", "H3": "greedy_loss"},
        "webshop-q7": {"P1w": "greedy_loss", "H1w": "l8_loss", "P2w": "q14_loss"},
        "webshop-l8": {"H2w": "q7_loss", "H3w": "greedy_loss"},
    }
    for pool_id in mapping:
        ds = Dataset(Path(data_root) / pool_id)
        profile = "legacy69" if pool_id == "alfworld-q7" else "standard"
        res = analyze(ds, profile)
        results[pool_id] = res
        expected = refs[pool_id]
        if len(res["rows"]) != len(expected["rows"]):
            raise ValueError(f"Reference row count differs: {pool_id}")
        errors = [abs(row[k]-ref[k]) for row, ref in zip(res["rows"], expected["rows"]) for k in ref]
        comparisons.append({"pool": pool_id, "max_abs_row_error": max(errors), "pass": max(errors) < 1e-12})
        for name, key in mapping[pool_id].items():
            if key in res["tests"]:
                tests[name] = dict(res["tests"][key], pool=pool_id,
                                   positive_control_pass=res["tests"]["positive_control"]["pass"])
                if name in expected["tests"]:
                    ref = expected["tests"][name]
                    err = max(abs(tests[name]["mean"]-ref["mean"]),
                              max(abs(a-b) for a,b in zip(tests[name]["ci"],ref["ci"])),
                              abs(tests[name]["p"]-ref["p"]))
                    comparisons.append({"test": name, "max_abs_stat_error": err, "pass": err < 1e-12})
        print(f"Recomputed {pool_id}; max row error {max(errors):.3g}", flush=True)
    # H1 was registered after the RNG streams were separated. Do not overwrite P1/P2.
    ds = Dataset(Path(data_root) / "alfworld-q7")
    modern = analyze(ds)
    results["alfworld-q7-current"] = modern
    tests["H1"] = dict(modern["tests"]["l8_loss"], pool="alfworld-q7",
                        positive_control_pass=modern["tests"]["positive_control"]["pass"])
    ref = refs["H1"]["reference"]
    err = max(abs(tests["H1"]["mean"]-ref["mean"]),
              max(abs(a-b) for a,b in zip(tests["H1"]["ci"],ref["ci"])))
    comparisons.append({"test": "H1", "max_abs_stat_error": err, "pass": err < 1e-12})
    for family, members in [("ALF-6.9", ["P1","P2"]), ("ALF-6.10", ["H1","H2","H3"]),
                             ("WebShop", ["P1w","P2w","H1w","H2w","H3w"])]:
        group = {k: tests[k] for k in members if k in tests}
        holm(group, len(members))
        for test in group.values():
            test["family"] = family
    historical = refs["H1"].get("historical")
    notices = ([{"test":"H1", "historical_mean":historical["mean"], "current_rerun_mean":tests["H1"]["mean"],
                 "note":"Existing tests_610_rerun.json and out_612 match current code. The older tests_610.json uses a different numerical result; retained, not silently overwritten."}]
               if historical and abs(historical["mean"]-tests["H1"]["mean"])>1e-12 else [])
    report = {"passed": all(c["pass"] for c in comparisons), "comparisons": comparisons, "version_notices":notices,
              "tests": tests, "pending_tests": ["P2w"] if "P2w" not in tests else [],
              "results": results, "scope": "Unscreened primary tests and budget rows; not all appendix analyses"}
    output = Path(output)
    write_json(output / "reproduction.json", report)
    lines = ["# Paper reproduction", "", f"Reference checks passed: **{report['passed']}**", "",
             "| Test | Selection loss | 90% CI | Holm reject |", "|---|---:|---|---|"]
    for name, st in tests.items():
        lines.append(f"| {name} | {st['mean']:.6f} | [{st['ci'][0]:.6f}, {st['ci'][1]:.6f}] | {st['holm_reject']} |")
    lines += ["", "Pending: " + (", ".join(report["pending_tests"]) or "none"), "",
              "Profiles retain historical random streams. The standard profile is for new comparisons.",
              "This is an offline numerical check, not a new GPU replication."]
    for notice in notices:
        lines += ["", f"H1 version note: historical {notice['historical_mean']:.9f}; current archived rerun {notice['current_rerun_mean']:.9f}. Both versions are retained in provenance/paper_reference.json."]
    (output / "reproduction.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report
