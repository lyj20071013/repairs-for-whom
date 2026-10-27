"""Maintainer tool: import complete pools and extract the historical scoring functions.

Only explicitly listed code and data files are read; no experiment is run.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repair_eval.importer import POOLS, import_pool
from repair_eval.io import sha256, write_json


def freeze_cores(source):
    dest = ROOT / "repair_eval" / "_paper"
    dest.mkdir(exist_ok=True)
    (dest / "__init__.py").write_text('"""Frozen, offline-only numerical kernels from the paper."""\n', encoding="utf-8")
    provenance = {}
    for name, relative in [("legacy69", "archive/rt_before_merge/ds_alfw_1240.py"), ("current", "ds_alfw.py")]:
        p = source / relative
        code = p.read_text(encoding="utf-8")
        tree = ast.parse(code)
        keep = {"_full", "_gain", "_val", "_best", "_has", "analyze_core"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep]
        header = '"""Frozen paper kernel; changes require a new protocol version. See docs/PROTOCOL.md."""\n'
        header += 'import numpy as np\nN_PERM = 50\nEXEC_CODE = {"q7": 1, "q14": 2, "l8": 3}\n\n'
        text = header + "\n\n".join(ast.get_source_segment(code, n) for n in nodes) + "\n"
        (dest / f"{name}.py").write_text(text, encoding="utf-8")
        provenance[name] = {"source": relative, "source_sha256": sha256(p),
                            "functions": [n.name for n in nodes], "extracted_sha256": sha256(dest / f"{name}.py")}
    write_json(ROOT / "provenance" / "scoring_sources.json", provenance)


def references(source):
    out = {}
    for pool_id, (folder, env, deployer, _) in POOLS.items():
        path = source / folder / "analysis.json"
        # Source analyses include NaN in unrelated descriptive fields. Only finite
        # primary rows and tests are retained in the reference snapshot.
        obj = json.loads(path.read_text(encoding="utf-8"))
        if env == "webshop":
            obj = obj["reward"]
        keys = ["matched", "greedy", "random"]
        if pool_id == "alfworld-q7":
            keys += [f"7B_k{k}" for k in (1,2,3,5)] + [f"14B_k{k}" for k in (1,2,3,5,10,20)]
        else:
            keys += [f"self_k{k}" for k in (1,2,3,5)]
            for ex in POOLS[pool_id][3]:
                if ex != deployer:
                    keys += [f"{ex}_k{k}" for k in (1,2,3,5,10,20)]
        out[pool_id] = {"source_sha256": sha256(path),
                        "rows": [{k: r[k] for k in keys if k in r} for r in obj["net"]["rows"]],
                        "tests": {k: {field:v[field] for field in ("mean","ci","p","n")} for k,v in obj["tests"].items()}}
    h = source / "out_ds_alfw_l8" / "tests_610_rerun.json"
    old = source / "out_ds_alfw_l8" / "tests_610.json"
    out["H1"] = {"reference": json.loads(h.read_text(encoding="utf-8"))["H1"], "source_sha256": sha256(h),
                 "source": "out_ds_alfw_l8/tests_610_rerun.json",
                 "historical": json.loads(old.read_text(encoding="utf-8"))["H1"],
                 "historical_sha256": sha256(old),
                 "note": "Current code matches the existing rerun and out_612; historical H1 remains recorded separately."}
    write_json(ROOT / "provenance" / "paper_reference.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args()
    source = Path(args.source)
    freeze_cores(source)
    references(source)
    for pool_id in POOLS:
        print(import_pool(source, Path(args.data_dir) / pool_id, pool_id), flush=True)


if __name__ == "__main__":
    main()
