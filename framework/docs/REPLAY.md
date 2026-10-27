# Continuations with a new executor

Offline evaluation does not require this workflow. It adds new-executor outcomes for fixed candidates; it **does not collect that executor's own failure pool**.

## 1. Freeze a plan on CPU

```console
python -m repair_eval plan-replay data/alfworld-q7 --executor my14b --model Qwen2.5-14B-Instruct --samples 10 --batch RE1 --out outputs/my14b_plan.json
```

The plan includes every stochastic branch, the sample count, one separate greedy run, original-action baselines, fixed seeds and original prefix observations. Deterministically terminal branches need no new stochastic runs. An existing executor name cannot overwrite outcomes; changes to prompts, weights or decoding require a new ID.

`--model` identifies the model explicitly; its actual path is configured separately in the backend. Creating a plan does not load a model or start inference.

## 2. Prepare the server

Copy the project to a Linux GPU server. The offline core needs only NumPy. Live continuations additionally require:

- vLLM, model weights and tokenizers compatible with the original experiments; context limits are 8192 for ALFWorld and 16384 for WebShop.
- The ALFWorld TextWorld/PDDL backend and `json_2.1.1` data, obtained separately.
- The original WebShop environment, 100k products/attributes and human goals, a Pyserini BM25 index, Java and spaCy. Environment and inference processes may use separate Python installations.

`runtime/` includes original experiment components, setup scripts and a minimal official WebShop archive. `provenance/runtime_sources.json` retains original-source hashes; localized release hashes are recorded separately. Read setup scripts and configure paths for your server. The framework does not automatically install, download or start them.

Copy `examples/replay_backend.json` and fill in model, data and Python paths. Set `runtime_root` to this project's `runtime`. For Llama, set `llama_stop_tokens=true` to retain the original EOS/end-of-turn rules. A new model family's chat template, stop tokens and parsing compatibility require a smoke check.

## 3. Run explicitly on GPU

```bash
PYTHONHASHSEED=0 python -m repair_eval run-replay outputs/my14b_plan.json --backend repair_eval.backends.paper_runtime:create --config my_backend.json --out outputs/my14b_run
```

The backend uses the original experiment's prompts, action parser, retries and batched continuation logic. The scheduler first verifies original prefix observations at each position and stops on a mismatch. Environment errors, model errors and missing observations are not recorded as task failures.

Completed records are written and flushed individually. Rerun the same command after interruption. Existing job IDs are deduplicated, and changes to configuration, backend code, runtime components or plans prevent resuming into the same output. If a killed process leaves `run.lock`, first confirm the old process is gone, then remove the lock. A damaged trailing JSONL line causes an error rather than silent deletion; back up the file and confirm that the final record is incomplete before repairing it.

Replays use a new random batch. Individual outcomes need not equal released samples, and a new replay must not be represented as the original registered P2w batch.

## 4. Attach complete results

```console
python -m repair_eval attach-replay data/alfworld-q7 --plan outputs/my14b_plan.json --results outputs/my14b_run --out data_updates/alfworld-q7-my14b
python -m repair_eval baselines data_updates/alfworld-q7-my14b --out outputs/my14b_comparison.json
```

Attachment requires all planned jobs and creates a new data version, leaving the original unchanged. Successful attachment establishes format and coverage, not behavioral equivalence to the original executor.

## Custom backends

Provide `module:create(config)` returning an object with `run(jobs)`, which yields records:

```python
{"job_id": job["job_id"], "success": 0.0, "reward": 0.6}
```

ALFWorld requires success; WebShop also requires reward. Do not duplicate a sample to meet the sample count. An optional `close()` method can release environments. Job prefixes include the substituted action, so continuations start after it with `max_steps - len(prefix)` steps left. Model history must use actual feedback produced by that prefix, not feedback following the replaced action in the old failed trajectory.

## Validation status

Tests cover resume, fingerprint rejection, complete attachment and ALFWorld/WebShop adapters using simulated environments and models. This packaging step did not run live GPU, ALFWorld or WebShop integration. Validate dependencies and paths with a small live smoke plan before launching a full replay.
