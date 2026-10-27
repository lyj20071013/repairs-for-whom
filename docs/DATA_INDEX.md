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
| WebShop, Qwen7B deployer (A) | `out_ws_A/` | failed episodes, fixed design, proposals, Qwen7B/Llama8B outcomes; P2w absent |
| WebShop, Llama8B deployer (B) | `out_ws_B/` | failed episodes, design and Llama8B/Qwen7B outcomes |
| WebShop joint primary analysis | `out_ws_summary/` | `tests_ws.json`, `report_ws.md`; unfinished P2w remains pending in its Holm family |
| Feasibility/smoke records | names containing `smoke`, `out_ws_S/`, `out_harness/` | engineering checks; do not pool with formal experiments |

Individual raw records may include original machine-local paths or model cache names. They identify the historical configuration, not required installation locations. Some rollout files store compact success/reward/step outcomes rather than a full textual trace for every continuation. Full failed source trajectories and recorded proposal logs are included; this release does not invent missing continuation transcripts.

The archived analysis implementations in `experiments/archive/` are retained for provenance. Use the framework for the tested primary-result reproduction path. Do not silently replace old analysis files with freshly recomputed results.
