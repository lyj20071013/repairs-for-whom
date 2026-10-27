"""Versioned bundle validation and conversion to branch matrices."""
from pathlib import Path
import numpy as np
from .io import read_json, read_jsonl, sha256, safe_member, fingerprint


class Dataset:
    def __init__(self, path, verify=True):
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        if self.manifest.get("schema_version") != "repair-eval/1":
            raise ValueError("Unsupported bundle schema")
        if self.manifest.get("bundle_id") != fingerprint({k:v for k,v in self.manifest.items() if k!="bundle_id"}):
            raise ValueError("Manifest fingerprint mismatch")
        required = {"design.json", "tasks.jsonl"} | {f"outcomes/{ex}.jsonl" for ex in self.manifest["executors"]}
        if not required.issubset(self.manifest["files"]):
            raise ValueError("Manifest is missing required file hashes")
        if verify:
            for name, expected in self.manifest["files"].items():
                if sha256(safe_member(self.path, name)) != expected:
                    raise ValueError(f"Bundle hash mismatch: {name}")
        self.design = read_json(self.path / "design.json")
        self.tasks = read_jsonl(self.path / "tasks.jsonl")
        self.trajs = self.design["trajectories"]
        self.branches = self.design["branches"]
        self.rollouts = {}
        for executor in self.manifest["executors"]:
            rows = read_jsonl(safe_member(self.path, f"outcomes/{executor}.jsonl"))
            indexed = {}
            for r in rows:
                key = (r["mode"], r["branch_id"])
                if key in indexed:
                    raise ValueError(f"Duplicate rollout record: {executor}, {key}")
                indexed[key] = r
            self.rollouts[executor] = indexed
        self.validate()

    def validate(self):
        ids = [t["traj_id"] for t in self.trajs]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty or duplicate trajectory IDs")
        if [t["id"] for t in self.tasks] != ids:
            raise ValueError("Public task order differs from evaluation design")
        if self.manifest["deployer"] not in self.rollouts:
            raise ValueError("Missing deployed executor")
        seen = set()
        for tr, task in zip(self.trajs, self.tasks):
            public = {c["id"] for c in task["candidates"]}
            private = {c["id"] for c in tr["candidates"]}
            if not private or public != private or len(private) != len(tr["candidates"]):
                raise ValueError(f"Invalid candidate mapping: {tr['traj_id']}")
            for c in tr["candidates"]:
                for bid in (c["branch_id"], c["original_branch_id"]):
                    if bid not in self.branches or self.branches[bid]["traj_id"] != tr["traj_id"]:
                        raise ValueError(f"Invalid branch join: {bid}")
                    seen.add(bid)
                a, o = (self.branches[c[k]] for k in ("branch_id", "original_branch_id"))
                if a["t"] != o["t"] or a["prefix"][:-1] != o["prefix"][:-1]:
                    raise ValueError("Alternative and original must intervene at the same history")
        if seen != set(self.branches):
            raise ValueError("Orphan branch in bundle")
        for bid, b in self.branches.items():
            if len(b["prefix"]) != b["t"] or not b["prefix"]:
                raise ValueError(f"Prefix must include intervention, at 1-based step t: {bid}")
            for metric in self.manifest["metrics"]:
                value = b["deterministic"].get(metric)
                if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"Invalid deterministic outcome: {bid}")
        stochastic = {bid for bid, b in self.branches.items() if b["deterministic"]["success"] is None}
        for ex, records in self.rollouts.items():
            n = self.manifest["executors"][ex]["samples"]
            if not isinstance(n, int) or n < 2:
                raise ValueError("At least two stochastic samples per executor are required")
            expected = {(mode, b) for mode in ("sample", "greedy") for b in stochastic}
            if set(records) != expected:
                raise ValueError(f"Incomplete/extra rollout coverage for {ex}: expected {len(expected)}, got {len(records)}")
            for (mode, bid), r in records.items():
                size = n if mode == "sample" else 1
                if len(r["seeds"]) != size or len(set(r["seeds"])) != size:
                    raise ValueError(f"Invalid sample seeds: {ex}, {bid}")
                for metric in self.manifest["metrics"]:
                    v = np.asarray(r[metric], float)
                    if v.shape != (size,) or not np.isfinite(v).all() or ((v < 0) | (v > 1)).any():
                        raise ValueError(f"Invalid {metric} outcomes: {ex}, {mode}, {bid}")
                    if metric == "success" and not np.isin(v, [0, 1]).all():
                        raise ValueError("Success must be binary")
        return {"pool": self.manifest["pool_id"], "trajectories": len(ids),
                "candidates": sum(len(t["candidates"]) for t in self.trajs),
                "branches": len(self.branches), "executors": list(self.rollouts),
                "pending": self.manifest.get("pending_executors", [])}

    def arrays(self, executor, metric=None):
        metric = metric or self.manifest["primary_metric"]
        if metric not in self.manifest["metrics"] or executor not in self.rollouts:
            raise ValueError(f"Unavailable executor/metric: {executor}/{metric}")
        n = self.manifest["executors"][executor]["samples"]
        records = self.rollouts[executor]
        result = []
        for tr in self.trajs:
            cands = [(c["branch_id"], c["original_branch_id"], c["position_index"]) for c in tr["candidates"]]
            bids = sorted({b for c in cands for b in c[:2]})
            col = {b: j for j, b in enumerate(bids)}
            det = np.array([np.nan if self.branches[b]["deterministic"][metric] is None
                            else float(self.branches[b]["deterministic"][metric]) for b in bids])
            rand = np.flatnonzero(np.isnan(det))
            samples = np.asarray([records[("sample", bids[j])][metric] for j in rand], float).reshape(len(rand), n)
            greedy = det.copy()
            for j in rand:
                greedy[j] = records[("greedy", bids[j])][metric][0]
            result.append({"cands": cands, "bids": bids, "col": col, "det": det, "rand": rand,
                           "S": samples, "G": greedy, "ia": np.array([col[c[0]] for c in cands]),
                           "io": np.array([col[c[1]] for c in cands])})
        return result

    def metadata(self, _ti, bid):
        return self.branches[bid]["metadata"]
