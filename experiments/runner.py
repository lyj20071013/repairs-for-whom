"""by (Condition, Batch) Resume all random branches，Branch-by-Face Disk，supports resuming interrupted runs。

Cache Keys：(condition, batch, branch_id)。Every tape spec_id（design + Condition + Batch + Samples +
Policy Signature），Inconsistentness is a refusal to share。Departure of the deterministic branch from writing，Direct extraction from the analysis phase。
"""
import json
import time
from pathlib import Path

import config
from policies import derive_seed

try:
    from tqdm import tqdm
except ImportError:  # unavailable in the offline environment tqdm Degradation to no progress
    tqdm = None


def spec_id(design_id, cond_name, batch, n, mock):
    spec = config.CONDITIONS[cond_name]
    policy_sig = config.run_id() if spec["kind"] == "llm" and not mock else "cpu"
    return config.fingerprint({"design": design_id, "condition": cond_name, "spec": spec,
                               "batch": batch, "n": n, "policy": policy_sig, "mock": bool(mock),
                               "top_p": config.TOP_P, "max_new_tokens": config.MAX_NEW_TOKENS})


def load_results(path):
    """Read Continue Cache。Allow one line at the end of the document to write half the record（_Other Organiser，or was interrupted when writing）：
    Ignore it；The damage to any remaining location is reported to be incorrect。"""
    done = {}
    path = Path(path)
    if not path.exists():
        return done
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    complete, tail = lines[:-1], lines[-1]   # If the last part isnt empty,，The note doesnt end with a line break
    for line in complete:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Damaged Relay Cache {path}；Change, please --out-dir") from exc
        done[(rec["condition"], rec["batch"], rec["branch_id"])] = rec
    if tail.strip():
        try:
            rec = json.loads(tail.decode("utf-8"))
            done[(rec["condition"], rec["batch"], rec["branch_id"])] = rec
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # Last line not finished：The branch is considered incomplete
    return done


def _repair_tail(path):
    """remove the unfinished trailing fragment before resuming continuations，Guaranteed to start with the new line。"""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        return
    cut = raw.rfind(b"\n") + 1
    with path.open("r+b") as f:
        f.truncate(cut)
    print(f"  Intercepted {path.name} Unfinished at the end {len(raw) - cut} Bytes（The corresponding branch will run again）")


def run_condition(design, cond_name, batch, policy, n=None, out_path=None, chunk=None, mock=False,
                  keep_traces=0):
    """keep_traces > 0 when，Before each branch is saved keep_traces step-by-step records of continuations（Without prejudice to result and cache signature）。"""
    if cond_name not in config.CONDITIONS:
        raise ValueError(f"Unknown {cond_name}")
    n = config.BATCHES[cond_name][batch] if n is None else n
    out_path = Path(out_path or config.ROLLOUTS_JSONL)
    chunk = chunk or config.ROLLOUT_CHUNK
    sid = spec_id(design["design_id"], cond_name, batch, n, mock)
    _repair_tail(out_path)
    done = load_results(out_path)
    for (c, b, _), rec in done.items():
        if c == cond_name and b == batch and rec["spec_id"] != sid:
            raise ValueError(f"{cond_name}/{batch} Caches are not consistent with the current design or configuration；Change, please --out-dir")

    pending = [b for b in design["branches"].values()
               if b["deterministic"] is None and (cond_name, batch, b["branch_id"]) not in done]
    if not pending:
        print(f"[{cond_name}/{batch}] All completed")
        return
    units_per_chunk = max(1, chunk // n)
    chunks = [pending[i:i + units_per_chunk] for i in range(0, len(pending), units_per_chunk)]
    total = len(pending) * n
    print(f"[{cond_name}/{batch}] Stand-by branch {len(pending)} items × {n} runs = {total} Keep running，min {len(chunks)} Batch")
    bar = tqdm(total=total, desc=f"{cond_name}/{batch}") if tqdm else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fout:
        for units in chunks:
            snaps, seeds = [], []
            for br in units:
                for k in range(n):
                    snaps.append(br["snapshot"])
                    seeds.append(derive_seed(cond_name, batch, br["branch_id"], k))
            t0 = time.time()
            stats = policy.run_rollouts(snaps, seeds)
            dt = max(time.time() - t0, 1e-6)
            for j, br in enumerate(units):
                block = stats[j * n:(j + 1) * n]
                rec = {
                    "condition": cond_name, "batch": batch, "branch_id": br["branch_id"],
                    "n": n, "successes": sum(bool(s["success"]) for s in block),
                    "results": [int(bool(s["success"])) for s in block],
                    "mean_steps": sum(s["steps"] for s in block) / n,
                    "malformed": sum(s["malformed"] for s in block),
                    "escapes": sum(s.get("escapes", 0) for s in block),
                    **({"traces": [s.get("trace", []) for s in block[:keep_traces]]} if keep_traces else {}),
                    "spec_id": sid, "design_id": design["design_id"],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if bar:
                bar.update(len(snaps))
                bar.set_postfix_str(f"{len(snaps) / dt:.1f} rollout/s")
            else:
                print(f"  complete {len(snaps)} records，{len(snaps) / dt:.1f} rollout/s", flush=True)
    if bar:
        bar.close()


def plan(design):
    """Print Every (Condition, Batch) Number of repeat strips，Use to estimate time on the server。"""
    stoch = [b for b in design["branches"].values() if b["deterministic"] is None]
    mean_budget = sum(b["remaining_budget"] for b in stoch) / max(len(stoch), 1)
    rows, llm_total = [], 0
    for cond, batches in config.BATCHES.items():
        kind = config.CONDITIONS[cond]["kind"]
        for batch, n in batches.items():
            count = len(stoch) * n
            rows.append((cond, batch, kind, n, count))
            if kind == "llm":
                llm_total += count
    return {"stochastic_branches": len(stoch), "mean_remaining_budget": mean_budget,
            "rows": rows, "llm_rollouts_total": llm_total}
