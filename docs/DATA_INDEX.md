# Data index

Paths below are relative to `experiments/`. Raw code and records retain their existing file layout. Standardized datasets are under `framework/data/`; they duplicate four raw pools rather than adding samples.

| Study | Raw directory | Key files / role |
|---|---|---|
| Early gridworld development | `out/`, `out_quick/` | design, rollouts, exploratory reports; not the main confirmatory pool |
| Gridworld confirmation | `out_confirm/` | `episodes.jsonl`, `main_design.json`, `main_rollouts.jsonl`; screen records, judge outputs and mechanism rollouts |
| Unscreened gridworld | `out_ds_gw/` | `episodes_q7.jsonl`, `design_q7.json`, `evals_q7_by_*.jsonl`; 14B records are not asserted complete |
| ALFWorld feasibility | `out_alfw/` | replay checks, baseline episodes and reports |
| Screened ALFWorld | `out_alfw_b/` | `episodes.jsonl`, `main_design.json`, `main_rollouts.jsonl`, candidate screen and judges |
| Executor ladder | `out_ladder_gw/`, `out_ladder_alfw/` | `q14.jsonl`, `qflash.jsonl`, ladder reports |
| Unscreened ALFWorld, Qwen7B deployer | `out_ds_alfw/` | `episodes.jsonl`, `positions.json`, `design.json`, `evals_q7.jsonl`, `evals_q14.jsonl`, `evals_l8.jsonl` |
| Unscreened ALFWorld, Llama8B deployer | `out_ds_alfw_l8/` | corresponding episodes/design/outcomes; `tests_610.json` and `tests_610_rerun.json` preserve historical versions |
| Follow-up/budget/saturation analyses | `out_612/` | `analysis_612.json`, `report_612.md` |
| Independent preference check | `out_rev612/q14/`, `out_rev612/l8/` | frozen cases/design and fresh executor outcomes; aggregate at `out_rev612/` |
| Flash location/proposal + Qwen7B verification (G) | `out_g612/` | `locate.jsonl`, `proposals.jsonl`, `design.json`, `evals_q7.jsonl`; GV/GE/GG identify independent phases |
| Low-budget method | `out_method/` | method reports and JSON summaries |
| Label/ranking consequences | `out_consequence/` | gridworld/ALFWorld summaries |
| Shared-repair analysis | `out_share/` | exploratory results; follow the documented limitations |
| WebShop, Qwen7B deployer (A) | `out_ws_A/` | failed episodes, fixed design, proposals, Qwen7B/Llama8B/14B outcomes; completed P2w, timing, and pre-Q14 analysis snapshots |
| WebShop, Llama8B deployer (B) | `out_ws_B/` | failed episodes, design and Llama8B/Qwen7B outcomes |
| WebShop joint primary analysis | `out_ws_summary/` | `tests_ws.json`, `report_ws.md`; all five registered WebShop tests, including P2w, with the complete Holm family |
| Successor-as-evaluator, 14B executor check (exploratory) | `out_rev14/` | `report.md`, `rev14.json`, produced by `posthoc/rev14.py` |
| τ²-bench shared pool, 2B ← 4B (P1τ, P2τ, H1/H2) | `out_tau_pool/` | `design_p4sub.jsonl` (fixed branches), `ro_2b.jsonl`, `ro_2b_greedy.jsonl`, `ro_2b_fill.jsonl` (2B outcomes; 4B outcomes are in `followup/repair_budget/tau2_pilot/pool/ro_pi0.jsonl`), `report_main_v3.md`/`analysis_main_v3.json` (primary), `report_sens_*` (sensitivity), `report_h1h2.*`; code in `tau2r/` |
| Post hoc loss share and headroom | `posthoc/` | `ratio_loss.{py,md,json}`, `headroom.{py,md,json}`; definitions in `analysis_plan/amendments_202611/` |
| Feasibility/smoke records | names containing `smoke`, `out_ws_S/`, `out_harness/` | engineering checks; do not pool with formal experiments |

Individual raw records may include original machine-local paths or model cache names. They identify the historical configuration, not required installation locations. Some rollout files store compact success/reward/step outcomes rather than a full textual trace for every continuation. Full failed source trajectories and recorded proposal logs are included; this release does not invent missing continuation transcripts.

## Follow-up studies (`followup/`, paths relative to the repository root)

| Study | Directory | Key files / role |
|---|---|---|
| Successor training (ALFWorld, Qwen2.5-7B) | `followup/repair_training/` | `out_rt/eval_rollouts.jsonl` (end-to-end evaluation of all trained variants), `out_rt/report.md`, `analysis.json`, `data_records/trainsets.json` (training labels), `prelim/` (pre-specified point selection and power) |
| Fixed-pool staleness under successors | `followup/repair_budget/alfw_stale/out_stale/` | `full.jsonl` (all candidates re-evaluated under successors), `two.jsonl`, `drift.jsonl`, `reverse.jsonl`, `free_b.jsonl`; `report_stale.md`, `planner_split.*`, `self_eval.*`, `posthoc_noplanner.*`, `stale_predict.*`, `repair_windows.*` |
| Experiment B: successors trained without planner labels | `followup/repair_budget/alfw_stale/out_np/` | `full.jsonl`, `drift.jsonl`, `report_np.*`, `self_eval_np.*`, `train_log_s11/s12.json`, `trainsets_np.json` |
| Label reliability | `followup/repair_budget/label_reliability.*` | split-half reliability of position labels; code in `followup/repair_budget/tools/` |
| τ²-bench pilots and probe | `followup/repair_budget/tau2_pilot/probe/` and pilot notes | probe designs, rollouts, LoRA parity checks, power analysis; engineering and gating records |
| τ²-bench main pool and label training (π0 = Qwen3.5-4B, π1 = 4B-A/4B-B) | `followup/repair_budget/tau2_pilot/pool/` | `positions.jsonl`, `design_{ret,air,run}.jsonl`, `folds.json`, `adm_v3.jsonl` (executable candidates), `ro_pi0.jsonl`, `ro_pi1_{A,B}.jsonl`, `samples_{A,B}.jsonl` (training sets), `drift*.jsonl`, `first_p4*.jsonl`/`first_cls.jsonl` (first-action samples for H1/H2), `report_main.*`, `report_terms.*`, `report_drift_cls.*` |

The archived analysis implementations in `experiments/archive/` are retained for provenance. Use the framework for the tested primary-result reproduction path. Do not silently replace old analysis files with freshly recomputed results.
