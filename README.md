# Repairs for Whom?

**Evaluator–Executor Mismatch in Selecting Repairs for Failed LLM Agents**

[Data index](docs/DATA_INDEX.md) · [Evaluation protocol](framework/docs/PROTOCOL.md) · [Analysis plan](analysis_plan/README.md)

A repair can be valuable for one executor and ineffective for another. This repository contains the experiment code, failed trajectories, fixed candidate repairs, individual continuation outcomes, analysis records, and a reusable evaluation framework for studying that mismatch.

**Snapshot: 2026-10-28.** Includes the completed WebShop 14B evaluation (P2w), its individual continuation outcomes, and updated analyses. At five rollouts per candidate, selecting with 14B instead of the deployed 7B agent loses **0.073** in net reward (90% CI **[0.046, 0.101]**). The arXiv identifier will be added after the first preprint release.

## Start here — reproduce without a GPU

Use Python 3.10+ from the repository root:

```bash
python -m pip install -r requirements-offline.txt
python tools/verify_release.py
cd framework
python -m repair_eval validate data/alfworld-q7
python -m repair_eval reproduce
```

The last command recalculates the primary results from individual outcomes and checks them against the archived reference records. Reports are written to `framework/outputs/reproduction/`. It requires neither model weights nor installation of ALFWorld or WebShop. It does not rerun GPU experiments.

## Included data

The standardized framework contains four unscreened failure pools:

| Environment | Deployed agent | Failed trajectories | Candidate repairs | Available continuation executors |
|---|---|---:|---:|---|
| ALFWorld | Qwen2.5-7B-Instruct | 60 | 1,106 | Qwen 7B, Qwen 14B, Llama 8B |
| ALFWorld | Llama-3.1-8B-Instruct | 60 | 1,075 | Llama 8B, Qwen 7B |
| WebShop | Qwen2.5-7B-Instruct | 60 | 2,071 | Qwen 7B, Llama 8B, Qwen 14B |
| WebShop | Llama-3.1-8B-Instruct | 60 | 1,819 | Llama 8B, Qwen 7B |

Total: **240 failures, 6,071 candidate repairs and 6,791 branches including original actions**. These standardized records duplicate the corresponding raw records for ease of use; they are not additional independent samples.

`experiments/out_*/` also contains the gridworld experiments, screened ALFWorld pool, executor ladders, independent preference checks, the low-cost flash proposal pipeline, and feasibility/smoke records. See the [data index](docs/DATA_INDEX.md) for their roles. The unified framework does not replace the original specialized analyses for those additional studies.

## Evaluate a repair selector

From `framework/`:

```bash
python -m repair_eval evaluate data/alfworld-q7 --selector examples.selectors:earliest --out outputs/earliest.json
python -m repair_eval evaluate data/alfworld-q7 --selector examples.selectors:sampled_gain --uses-rollouts --budget 5 --out outputs/matched.json
python -m repair_eval export-tasks data/alfworld-q7 --out outputs/blind_tasks
```

Implement `select(task, context)` following [the selector examples](framework/examples/selectors.py). The rollout-based interface supplies a selection fold and scores the choice on an independent evaluation fold. No-rollout tasks hide outcomes and privileged candidate-source labels. The full repository nevertheless contains the outcomes: it is a reproducibility resource, not a hidden leaderboard. Avoid selecting or tuning on the evaluation outcomes.

## Repository layout

```text
experiments/       Original experimental code and raw out_* records
framework/         Reusable CPU evaluation and optional replay adapters
analysis_plan/     Original cumulative analysis plan, with dated amendments
paper_figs/        Scripts for regenerating the paper figures
manuscript/        Current LaTeX source, bibliography, styles and figure PDFs
docs/              Data index, reproduction guide and release status
provenance/        SHA256 inventory, source mapping and verification records
tools/             Release integrity checker
```

## Protocol and provenance

- The primary value is repair gain relative to continuing with the original action at the same position, evaluated under the deployed executor. WebShop uses reward as its primary outcome; full success is secondary.
- Cross-fitting separates selection from evaluation. Trajectories are the bootstrap unit. See [PROTOCOL.md](framework/docs/PROTOCOL.md) for random streams, tie handling, budgets and Holm families.
- Historical H1 is approximately 0.07447; the recovered/rerun record used by the framework is approximately 0.07122. Both versions are retained and identified rather than silently merged.
- The [analysis plan](analysis_plan/论文方案.md) is a cumulative working record containing pre-specified designs, subsequent amendments and exploratory results. Its recorded dates are not a claim of an independently timestamped public preregistration.
- `provenance/source_map.json` records the original source and SHA256 of copied files. `provenance/release_manifest.json` verifies the packaged bytes. English localization and source-hash changes are recorded in `provenance/english_localization.json`.

## Run new continuations or regenerate figures

See [REPRODUCING.md](docs/REPRODUCING.md) and [the replay guide](framework/docs/REPLAY.md). Live continuations require the external environment data, model weights and a suitable Linux/GPU environment. No inference is launched by the CPU quickstart.

Model weights, the WebShop product corpus/index, ALFWorld installation data, API keys, SSH transfer settings and process logs are not included. The environment setup code and data fingerprints are retained. The previous snapshot remains in Git history; `provenance/p2w_update.json` records the new source hashes and version change.

## Tests

```bash
cd framework
python -m unittest discover -s tests -v
```

See [release status](docs/RELEASE_STATUS.md) for the checks actually run on this package. GPU adapters have mock coverage; a fresh live GPU/environment integration run has not been performed for this packaging step.

## Language and source identity

Documentation, source comments and runtime messages are in English. Original research plans, archived reports and raw experimental records retain their original language and bytes. Localization changes source hashes; start a fresh output directory for new live runs rather than bypassing historical resume checks. See [release status](docs/RELEASE_STATUS.md).

## License and citation

Original project code is licensed under [Apache-2.0](LICENSE). Third-party code, model weights and environment-derived text retain their respective rights; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Paper: *Repairs for Whom? Evaluator–Executor Mismatch in Selecting Repairs for Failed LLM Agents*. Formal citation metadata will be added when the arXiv v1 identifier is available.
