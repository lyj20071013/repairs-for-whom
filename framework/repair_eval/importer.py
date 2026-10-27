"""Read-only migration from the authors' experiment folders to portable bundles."""
import copy
from pathlib import Path
from .io import read_json, read_jsonl, write_json, write_jsonl, sha256, seed_for, fingerprint
from .dataset import Dataset

MODELS = {"q7": "Qwen2.5-7B-Instruct", "q14": "Qwen2.5-14B-Instruct", "l8": "Llama-3.1-8B-Instruct"}
POOLS = {
    "alfworld-q7": ("out_ds_alfw", "alfworld", "q7", {"q7": 10, "q14": 20, "l8": 10}),
    "alfworld-l8": ("out_ds_alfw_l8", "alfworld", "l8", {"l8": 10, "q7": 10}),
    "webshop-q7": ("out_ws_A", "webshop", "q7", {"q7": 10, "l8": 10, "q14": 20}),
    "webshop-l8": ("out_ws_B", "webshop", "l8", {"l8": 10, "q7": 10}),
}


def _relative_game(path):
    path = path.replace("\\", "/")
    marker = "json_2.1.1/"
    if marker not in path:
        raise ValueError("Cannot make ALFWorld game path portable")
    return marker + path.split(marker, 1)[1]


def import_pool(source_root, output, pool_id):
    """Never overwrite a bundle or alter the experiment directory."""
    root, output = Path(source_root), Path(output)
    folder, env, deployer, expected = POOLS[pool_id]
    src = root / folder
    if output.exists():
        raise ValueError(f"Output already exists; create a new snapshot: {output}")
    d = read_json(src / "design.json")
    pos = read_json(src / "positions.json")
    if d["positions_id"] != pos["positions_id"]:
        raise ValueError("Design/position fingerprint mismatch")
    # Verify the source design's own content fingerprint, as used by the original code.
    import hashlib, json
    fields = (("prefix", "oracle_F", "oracle_slack", "proposers", "planner", "random") if env == "alfworld"
              else ("prefix", "proposers", "planner", "random", "exhaustive", "deterministic"))
    d0 = {"t": d["trajectories"], "b": {k: {kk: v.get(kk) for kk in fields}
                                      for k, v in d["branches"].items()}, "s": d["settings"]}
    actual = hashlib.sha256(json.dumps(d0, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    if actual != d["design_id"]:
        raise ValueError("Source design content does not match design_id")
    eps = {e["rid"]: e for e in read_jsonl(src / "episodes.jsonl")}
    positions = {t["traj_id"]: t for t in pos["trajectories"]}
    batch = "D" if env == "alfworld" else "WS1"
    metrics = ["success"] if env == "alfworld" else ["success", "reward"]
    branches, trajectories, tasks = {}, [], []
    max_steps = 50 if env == "alfworld" else 25
    for tr in d["trajectories"]:
        tid = tr["traj_id"]
        ep, pp = eps[tid], positions[tid]
        if len(ep["cmds"]) != len(ep["obs"]):
            raise ValueError(f"Misaligned action/observation log: {tid}")
        candidates, public_cands, public_positions = [], [], []
        for pi, p in enumerate(tr["positions"]):
            step = p["t"] + (1 if env == "alfworld" else 0)
            source_pos = next(x for x in pp["positions"] if x["t"] == p["t"])
            public_positions.append({"step": step, "original_action": ep["cmds"][step-1],
                                     **{k: source_pos[k] for k in ["obs", "admissible", "clicks", "search", "page"] if k in source_pos}})
            for a in p["alts"]:
                cid = f"c{len(candidates):03d}"
                candidates.append({"id": cid, "branch_id": a, "original_branch_id": p["orig"], "position_index": pi})
                public_cands.append({"id": cid, "step": step, "action": d["branches"][a]["prefix"][-1]})
            for bid in [p["orig"]] + p["alts"]:
                b = d["branches"][bid]
                if b["prefix"][:-1] != ep["cmds"][:step-1]:
                    raise ValueError(f"Prefix does not match failed log: {bid}")
                replay = {"game": _relative_game(b["gamefile"])} if env == "alfworld" else {"goal": b["goal"]}
                det = {"success": None if b["deterministic"] is None else float(b["deterministic"])}
                if env == "webshop":
                    det["reward"] = b["det_reward"]
                meta = {k: copy.deepcopy(b.get(k, default)) for k, default in {
                    "planner": False, "random": False, "proposers": [], "flash_rank": None,
                    "oracle_F": 0.0, "oracle_slack": 0.0, "exhaustive": False}.items()}
                branches[bid] = {"traj_id": tid, "t": step, "source_t": b["t"], "prefix": b["prefix"],
                                 "deterministic": det, "metadata": meta, "replay": replay}
        trajectories.append({"traj_id": tid, "candidates": candidates, "max_steps": max_steps})
        # Public tasks intentionally omit candidate provenance, privileged state and outcomes.
        # Shuffle once independently of labels so candidate order cannot encode planner priority.
        import random
        random.Random(seed_for("public-candidates", pool_id, tid)).shuffle(public_cands)
        # IDs are re-assigned after shuffling; update the private join at the same time.
        remap = {c["id"]: f"c{i:03d}" for i, c in enumerate(public_cands)}
        for c in candidates:
            c["id"] = remap[c["id"]]
        for c in public_cands:
            c["id"] = remap[c["id"]]
        tasks.append({"id": tid, "environment": env, "deployer": deployer,
                      "instruction": ep.get("task", ep.get("instruction")),
                      "initial_description": ep.get("desc", "[button] Search [button_]"),
                      "history": [{"step": i+1, "action": cmd, "observation": ob}
                                  for i, (cmd, ob) in enumerate(zip(ep["cmds"], ep["obs"]))],
                      "positions": public_positions, "candidates": public_cands, "max_steps": max_steps})
    stochastic = {b for b in branches if branches[b]["deterministic"]["success"] is None}
    executors, pending, outcomes = {}, [], {}
    source_hashes = {name: sha256(src / name) for name in ["design.json", "positions.json", "episodes.jsonl"]}
    for ex, n in expected.items():
        path = src / f"evals_{ex}.jsonl"
        if not path.exists():
            pending.append({"executor": ex, "status": "missing", "expected_samples": n})
            continue
        source_hashes[path.name] = sha256(path)
        records = {}
        for r in read_jsonl(path):
            key = (r["condition"], r["batch"], r["branch_id"])
            if key in records and records[key] != r:
                raise ValueError(f"Conflicting source records: {ex}, {key}")
            records[key] = r
        keys = {(f"{ex}_{suffix}", batch, b) for suffix in ["T07", "greedy"] for b in stochastic}
        if set(records) - keys:
            raise ValueError(f"Unexpected branches/conditions in {path.name}")
        if keys - set(records):
            if ex == deployer:
                raise ValueError("Deployer rollout file is incomplete")
            pending.append({"executor": ex, "status": "incomplete", "missing_records": len(keys-set(records)), "expected_samples": n})
            continue
        out = []
        for mode, suffix in [("sample", "T07"), ("greedy", "greedy")]:
            spec_ids = set()
            for b in sorted(stochastic):
                r = records[(f"{ex}_{suffix}", batch, b)]
                size = n if mode == "sample" else 1
                if r["n"] != size or len(r["results"]) != size:
                    raise ValueError(f"Wrong rollout count: {ex}/{b}")
                spec_ids.add(r["spec_id"])
                row = {"branch_id": b, "mode": mode, "source_spec_id": r["spec_id"],
                       "seeds": [seed_for(f"{ex}_{suffix}", batch, b, k) for k in range(size)],
                       "success": [float(x) for x in r["results"]]}
                if env == "webshop":
                    row["reward"] = r["rewards"]
                for name in ["steps", "invalid", "retries", "revisit", "trunc", "drop"]:
                    if name in r:
                        row[name] = r[name]
                out.append(row)
            if len(spec_ids) != 1:
                raise ValueError(f"Mixed source execution specifications: {ex}/{mode}")
        executors[ex] = {"model": MODELS[ex], "samples": n, "temperature": 0.7, "top_p": 0.95,
                         "greedy_temperature": 0.0, "max_invalid_retries": 2,
                         "history": "actions and observations; no earlier thoughts",
                         "seed_provenance": "reconstructed from original SHA256(condition|batch|branch|index), not observed RNG state"}
        outcomes[ex] = out
    if deployer not in executors:
        raise ValueError("Cannot import without complete deployed-agent outcomes")
    output.mkdir(parents=True)
    design = {"trajectories": trajectories, "branches": branches}
    write_json(output / "design.json", design)
    write_jsonl(output / "tasks.jsonl", tasks)
    for ex, records in outcomes.items():
        write_jsonl(output / "outcomes" / f"{ex}.jsonl", records)
    manifest = {"schema_version": "repair-eval/1", "pool_id": pool_id, "environment": env,
                "source_design_id": d["design_id"], "source_folder": folder, "source_hashes": source_hashes,
                "deployer": deployer, "primary_metric": "reward" if env == "webshop" else "success",
                "metrics": metrics, "executors": executors, "pending_executors": pending,
                "max_steps": max_steps, "screening": "first failures, not screened by repair value",
                "restrictions": "ALFWorld excludes two-object tasks and requires >=5 planner slack; WebShop requires >=5 remaining steps",
                "source_settings": {k: v for k, v in d["settings"].items() if k != "MODEL_DIR"},
                "files": {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    manifest["bundle_id"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    return Dataset(output).validate()
