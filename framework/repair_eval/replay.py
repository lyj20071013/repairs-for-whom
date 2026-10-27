"""Backend-neutral replay plans, checked resumable results, and immutable attachment."""
import importlib
import inspect
import json
import os
import re
import shutil
from pathlib import Path
import numpy as np
from .dataset import Dataset
from .io import fingerprint, seed_for, read_json, read_jsonl, write_json, write_jsonl, sha256


def plan_replay(ds, executor, model, samples, batch, output):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", executor) or not batch or samples < 2:
        raise ValueError("Use a simple executor ID, a nonempty batch and >=2 samples")
    if executor in ds.rollouts:
        raise ValueError("Executor ID already exists; use a distinct ID for a new protocol")
    output = Path(output)
    if output.exists():
        raise ValueError("Replay plan exists; do not overwrite frozen plans")
    jobs = []
    tasks = {t["id"]: t for t in ds.tasks}
    for bid, b in sorted(ds.branches.items()):
        if b["deterministic"]["success"] is not None:
            continue
        t = tasks[b["traj_id"]]
        for mode, n, temperature in [("sample",samples,.7),("greedy",1,0.)]:
            for index in range(n):
                job = {"branch_id":bid, "trajectory":b["traj_id"], "environment":ds.manifest["environment"],
                       "mode":mode, "index":index, "temperature":temperature, "top_p":.95,
                       "max_steps":ds.manifest["max_steps"], "prefix":b["prefix"], "replay":b["replay"],
                       "seed":seed_for(executor, batch, bid, mode, index),
                       "expected_prefix_observations":[h["observation"] for h in t["history"][:b["t"]-1]]}
                job["job_id"] = fingerprint(job)
                jobs.append(job)
    plan = {"schema_version":"repair-replay/1", "bundle_id":ds.manifest["bundle_id"],
            "pool_id":ds.manifest["pool_id"], "environment":ds.manifest["environment"],
            "executor":executor, "model":model, "samples":samples, "batch":batch,
            "metrics":ds.manifest["metrics"], "jobs":jobs,
            "protocol":"paper runtime prompts/history/retries; new rollout seeds"}
    plan["plan_id"] = fingerprint(plan)
    write_json(output, plan)
    return {"plan_id":plan["plan_id"], "jobs":len(jobs), "output":str(output), "gpu_started":False}


def load_plan(path):
    plan = read_json(path)
    if plan.get("schema_version") != "repair-replay/1" or plan["plan_id"] != fingerprint({k:v for k,v in plan.items() if k!="plan_id"}):
        raise ValueError("Invalid replay plan fingerprint")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", plan["executor"]):
        raise ValueError("Invalid executor ID")
    ids = [j["job_id"] for j in plan["jobs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate replay jobs")
    for j in plan["jobs"]:
        if j["job_id"] != fingerprint({k:v for k,v in j.items() if k!="job_id"}):
            raise ValueError("Invalid job fingerprint")
    return plan


def check_result(r, jobs, metrics):
    if r.get("job_id") not in jobs:
        raise ValueError("Backend returned an unknown job ID")
    for metric in metrics:
        x = r.get(metric)
        if not isinstance(x, (int,float,bool)) or not np.isfinite(x) or not 0<=x<=1:
            raise ValueError(f"Invalid replay outcome: {metric}")
    if r["success"] not in (0,1,False,True):
        raise ValueError("Success must be binary")


def read_results(path, plan):
    jobs = {j["job_id"]:j for j in plan["jobs"]}
    result = {}
    if Path(path).exists():
        for r in read_jsonl(path):
            check_result(r, jobs, plan["metrics"])
            if r.get("plan_id") != plan["plan_id"]:
                raise ValueError("Replay results belong to another plan")
            if r["job_id"] in result:
                raise ValueError("Duplicate replay result")
            result[r["job_id"]] = r
    return result


def run_replay(plan_path, backend_spec, config, output):
    plan = load_plan(plan_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    try:
        fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("Output is locked; inspect the running process before removing a stale run.lock") from exc
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        module, name = backend_spec.split(":",1)
        factory = getattr(importlib.import_module(module),name)
        backend_file = inspect.getsourcefile(factory)
        identity = {"plan_id":plan["plan_id"], "backend":backend_spec, "backend_sha256":sha256(backend_file),
                    "config_sha256":fingerprint(config), "model":plan["model"]}
        runtime = Path(config["runtime_root"]) if "runtime_root" in config else None
        if runtime:
            identity["runtime_sha256"] = fingerprint({p.relative_to(runtime).as_posix():sha256(p)
                                                       for p in sorted(runtime.rglob("*.py"))})
        meta = output / "run.json"
        if meta.exists() and read_json(meta) != identity:
            raise ValueError("Cannot resume after changing plan, backend, model or configuration")
        done = read_results(output / "results.jsonl",plan)
        if done and not meta.exists():
            raise ValueError("Results exist without run provenance")
        write_json(meta,identity)
        pending = [j for j in plan["jobs"] if j["job_id"] not in done]
        if not pending:
            return {"completed":len(done), "remaining":0, "resumed_without_backend":True}
        backend = factory(dict(config, expected_model=plan["model"]))
        if getattr(backend,"synthetic",False) and plan["environment"] != "synthetic":
            raise ValueError("Synthetic backend cannot produce outcomes for a real environment")
        byid = {j["job_id"]:j for j in pending}
        try:
            with (output / "results.jsonl").open("a",encoding="utf-8",newline="\n") as f:
                for r in backend.run(pending):
                    check_result(r,byid,plan["metrics"])
                    if r["job_id"] in done:
                        raise ValueError("Backend returned duplicate result")
                    r = dict(r,plan_id=plan["plan_id"])
                    f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
                    f.flush()
                    done[r["job_id"]]=r
                    if len(done)%100==0:
                        print(f"Completed {len(done)}/{len(plan['jobs'])}",flush=True)
        finally:
            if hasattr(backend,"close"):
                backend.close()
        remaining = len(plan["jobs"])-len(done)
        if remaining:
            raise ValueError(f"Backend stopped with {remaining} pending jobs; completed rows retained")
        return {"completed":len(done),"remaining":0}
    finally:
        lock.unlink(missing_ok=True)


def attach_replay(ds, plan_path, results_dir, output):
    plan = load_plan(plan_path)
    if plan["bundle_id"] != ds.manifest["bundle_id"]:
        raise ValueError("Plan and bundle do not match")
    ex = plan["executor"]
    if ex in ds.rollouts:
        raise ValueError("Cannot overwrite existing executor outcomes")
    results_dir, output = Path(results_dir), Path(output)
    run_meta = read_json(results_dir/"run.json")
    if run_meta["plan_id"] != plan["plan_id"]:
        raise ValueError("Replay provenance does not match plan")
    rows = read_results(results_dir/"results.jsonl",plan)
    if len(rows)!=len(plan["jobs"]):
        raise ValueError("Cannot attach incomplete executor results")
    if output.exists():
        raise ValueError("Destination bundle exists; use a new version directory")
    grouped = {}
    for j in plan["jobs"]:
        grouped.setdefault((j["mode"],j["branch_id"]),[]).append((j,rows[j["job_id"]]))
    records=[]
    for (mode,bid), pairs in sorted(grouped.items()):
        pairs.sort(key=lambda x:x[0]["index"])
        records.append({"mode":mode,"branch_id":bid,"source_spec_id":plan["plan_id"],
                        "seeds":[j["seed"] for j,r in pairs],
                        **{metric:[float(r[metric]) for j,r in pairs] for metric in plan["metrics"]}})
    shutil.copytree(ds.path,output)
    write_jsonl(output/"outcomes"/f"{ex}.jsonl",records)
    manifest = read_json(output/"manifest.json")
    manifest["parent_bundle_id"]=manifest.pop("bundle_id")
    manifest["executors"][ex]={"model":plan["model"],"samples":plan["samples"],"temperature":.7,"top_p":.95,
                               "replay_provenance":run_meta,"plan_id":plan["plan_id"]}
    manifest["pending_executors"]=[x for x in manifest.get("pending_executors",[]) if x["executor"]!=ex]
    manifest["files"][f"outcomes/{ex}.jsonl"]=sha256(output/"outcomes"/f"{ex}.jsonl")
    manifest["bundle_id"]=fingerprint(manifest)
    write_json(output/"manifest.json",manifest)
    return Dataset(output).validate()
