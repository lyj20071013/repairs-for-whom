> Repository snapshot: 2026-10-27. Original project code is Apache-2.0; see [repository notices](../THIRD_PARTY_NOTICES.md). Current release scope and pending P2w are authoritative in [release status](../docs/RELEASE_STATUS.md).

# RepairEval

Executor-conditioned evaluation of single-step repairs for failed LLM agents.

This research snapshot packages four unscreened failure pools from **Repairs for Whom? Evaluator–Executor Mismatch in Selecting Repairs for Failed LLM Agents**: 240 failed trajectories, 6,071 candidate repairs, original-action baselines, and per-rollout outcomes from the available executors.

## CPU quickstart

Python 3.10+ and NumPy are sufficient for all offline commands. From the project directory:

```sh
python -m pip install "numpy>=1.24,<3"
python -m repair_eval validate data/alfworld-q7
python -m repair_eval reproduce
python -m unittest discover -s tests -v
```

The reproduction command recomputes the unscreened primary results and budget values from raw outcomes, compares trajectory-level values with stored source references, and writes a report under `outputs/reproduction`. It does not reproduce every appendix experiment.

## Add a selector

```python
def select(task, context):
    return task["candidates"][0]["id"]
```

Selectors return a candidate ID, a probability distribution over candidates (for exact tie averaging), or `None` to abstain under the net-gain objective. Tasks contain the failed log and available candidate actions, with candidate sources hidden and candidate order independently shuffled.

```sh
python -m repair_eval evaluate data/alfworld-q7 --selector examples.selectors:earliest --out outputs/earliest.json
python -m repair_eval evaluate data/alfworld-q7 --selector examples.selectors:sampled_gain --uses-rollouts --budget 5 --out outputs/matched.json
```

A rollout-based selector receives only the selection fold, never held-out outcomes. It must be stateless across folds. The Python API is not a security sandbox. Public outcomes make this a reproducible research resource, not a hidden test server; tuning on outcomes requires a separate development/test split or new trajectories.

## Estimand

For each executor, repair value is the expected outcome after replacing an action **minus the expected outcome after keeping the original action at the same position**. Outcomes are success for ALFWorld and reward (primary) or success (secondary) for WebShop.

Selection loss compares the held-out value of finite-sample matched selection with the method's held-out value. It may be negative. Matched selection is not an oracle. Cross-fitting prevents the samples used for selecting a candidate from also evaluating that choice. Confidence intervals resample trajectories, not individual rollouts.

## Versioned protocols

`standard` uses separate random streams per trajectory/executor; `legacy69` preserves the early ALFWorld Q7 P1/P2 numerical stream. H1 is compared with the source project's existing rerun (0.071217305); the earlier recorded value (0.074470916) remains in provenance. The current WebShop Q7 snapshot has no P2w/14B outcomes yet; missing data are never interpreted as failures.

## Replay and extension

`plan-replay` creates frozen replay jobs, `run-replay` executes an explicitly configured backend with resumption checks, and `attach-replay` creates a new complete data snapshot. The optional `paper_runtime` backend bridges the packaged original ALFWorld/WebShop runtime and vLLM. It requires separate model/environment installation and has been tested here with simulated backends, not real GPU environments. See `docs/REPLAY.md`.

Gridworld's two-stage sampling, screened-pool bootstrap, preference-reversal follow-ups, and human annotation analyses have not yet been migrated. Original environment datasets and model weights are not bundled. Original code is Apache-2.0. Formal paper citation metadata will follow arXiv v1; third-party materials retain their original terms.
