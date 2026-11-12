# Follow-up studies (added 2026-11-13)

This directory holds the code and records of two companion projects whose results were merged into the final manuscript:

| Directory | Original server project | What it contains | Manuscript |
|---|---|---|---|
| `repair_training/` | `/root/paper/repair_training` | One round of LoRA training of the ALFWorld Qwen2.5-7B agent on verified repairs (successors `S_s11`, `S_s12`, controls `U`, `R`, `A`, `DPO-S`, `S-plan`), and their end-to-end evaluation | Section "The Same Model, After Training"; Appendix "Training Version and Per-Position Labels" |
| `repair_budget/alfw_stale/` | `/root/paper/repair_budget/alfw_stale` | Re-evaluation of the fixed ALFWorld pool under the trained successors (point and action staleness), Experiment B (successors trained without planner labels, `out_np/`), self-evaluation and post hoc analyses | Section "The Same Model, After Training"; Appendix "Training Version" |
| `repair_budget/tau2_pilot/` | `/root/paper/repair_budget/tau2_pilot` | τ²-bench (airline, retail) pilots, probe, main shared pool with Qwen3.5-4B, label training of 4B successors (π1, folds A/B), drift/first-action sampling | Appendix "τ²-bench"; Appendix "Training Version" |
| `repair_budget/tools/` | `/root/paper/repair_budget/tools` | Label-reliability analysis | Appendix "Training Version" |

The τ²-bench shared-pool analysis for the 2B/4B evaluator–executor test (P1τ, P2τ, H1/H2) lives with the main experiments under `experiments/tau2r/` and `experiments/out_tau_pool/`; it reads the pool built here.

## Registrations and results

Dated registration (登记_*), definition (定义_*), amendment and result (*结果*) notes are kept next to the code in their original Chinese, byte-for-byte except for one redaction (see below):

- `repair_budget/登记_ALFWorld过期_20261030.md` → `A层结果_20261101.md` (ALFWorld staleness, layer A).
- `repair_budget/登记_τ²探针_20261031.md`, `敏感性分析_可执行候选_20261105.md`, `登记_τ²主实验_20261105.md` → `τ²主实验结果_20261110.md`.
- `repair_budget/登记_ALFWorld无规划器后继_20261110.md`, `定义_训练前的自己当评估者_20261110.md` → `实验B结果_20261112.md` (Experiment B).
- `repair_budget/事后分析_A与B_定义_20261110.md` → `事后分析_A与B_结果_20261110.md` (post hoc, defined before computing).
- `repair_training/论文方案.md` is the cumulative plan of the training project (pre-specified successor training and evaluation).
- `repair_budget/tau2_pilot/*结果_tau2.md`, `步骤_tau2先导.md`: pilot notes.

Read each note's own status line: some analyses are pre-specified, some post hoc, some exploratory. The manuscript's status table (Appendix "Status of Every Analysis") is the authoritative classification.

## Code identity and paths

- Code is released **byte-identical to what was run**, including its original Chinese comments and the historical server paths (`/root/paper/...`, `/root/autodl-tmp/...`). Unlike the 2026-10-28 English edition of `experiments/`, this code and the new `experiments/tau2r/` and `experiments/posthoc/` scripts were **not** localized, so they still match the file hashes cited in the registration notes (for example, `experiments/tau2r/h1h2_analyze.py` has the SHA256 prefix `fc0c52b7` recorded in `analysis_plan/amendments_202611/登记_策略差异与损失_20261111.md`).
- `repair_training/alfw/` and `repair_training/judge_backends.py` are the original (pre-localization) copies used by the training project; the English edition is in `experiments/alfw/`.
- `repair_training/data_records/trainsets.json` and `games.json` were written to `logs/` on the server; they are moved here because `logs/` is excluded from the repository. Scripts that read `logs/trainsets.json` need that path adjusted.
- `repair_training/MANIFEST.json` lists `alfw_setup.sh` and `tools/sync.py`; the former is `experiments/alfw_setup.sh`, the latter (a private transfer helper) is not released.
- `tau2_pilot/install_tau.sh` expects a trimmed τ²-bench snapshot (`tau2-bench-122e243-trim2.tar.gz`, upstream commit prefix `122e243`) that is not redistributed; obtain τ²-bench from its upstream repository at that commit.

## Not included

- Model weights and LoRA adapters (Qwen3.5-2B/4B/27B, Qwen2.5-7B successors).
- Full τ²-bench conversation transcripts stored beside rollout files as `*.msgs.jsonl.gz` (about 120 MB in total), and the τ²-bench `data/simulations/*/results.json` source runs (about 270 MB). The released `design_*.jsonl` files contain the prefixes and candidates the analyses use, and the `ro_*.jsonl` files contain every rollout outcome. The transcripts can be provided on request or as a GitHub Release asset.
- Process logs (`logs/`, `*.log`) and server connection settings. One line in `tau2_pilot/步骤_tau2先导.md` that contained a server SSH address was replaced with a note; this is recorded in `provenance/update_20261113.json`.
