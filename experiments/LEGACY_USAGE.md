# Original experiment workflow

Run these commands from `experiments/` in a compatible Linux environment. Configure model and data paths first. CPU reproduction is documented in [REPRODUCING.md](../docs/REPRODUCING.md); the commands below can launch GPU inference or paid API calls when explicitly invoked.

The cumulative analysis plan remains an unchanged research record under `analysis_plan/`. Existing raw results must not be overwritten. English localization changes source-file hashes, so new live runs should use a fresh output directory and retain their own fingerprints; do not bypass a resume mismatch.

## ALFWorld unscreened pools: sections 6.9 and 6.10

`ds_alfw.py` supports collection, position selection, candidate proposals, design freezing, executor continuations and analysis. The original Qwen7B pool used separate Qwen7B and Qwen14B servers:

```bash
python ds_alfw.py --stage smoke
python ds_alfw.py --stage prep7
# Transfer the generated bundle to the Qwen14B server:
python ds_alfw.py --stage prep14 --bundle out_ds_alfw/bundle_for_14b.tar.gz
# Return the frozen design to the Qwen7B server before run7:
python ds_alfw.py --stage run7
# Merge complete executor records before analysis:
python ds_alfw.py --stage analyze
```

Qwen7B has 10 stochastic continuations plus one greedy continuation per stochastic branch; Qwen14B has 20 plus one. Obtain model weights separately and configure `DS_Q14` and `DS_L8` as needed. The second deployed model is Llama-3.1-8B-Instruct:

```bash
python ds_alfw.py --profile l8 --stage smoke
python ds_alfw.py --stage p610
python ds_alfw.py --stage selftest
```

`p610` runs the Llama failure-pool workflow and Llama continuations on the existing Qwen7B design. Its inputs therefore include the complete section 6.9 design and Qwen7B records. The Llama and Qwen7B continuation batches each use 10 stochastic runs plus one greedy run. Historical Phi-4-mini smoke results are feasibility records, not part of the final comparison.

## Follow-ups and independent preference checks: section 6.12

`posthoc/ds612.py` reads `out_ds_alfw/` and `out_ds_alfw_l8/`. Missing or incomplete executor batches are excluded from comparisons that require them.

```bash
python posthoc/ds612.py analyze
python posthoc/ds612.py rev-freeze --pair q14
python posthoc/ds612.py rev-freeze --pair l8
python posthoc/ds612.py rev-eval --pair q14 --executor q7
python posthoc/ds612.py rev-eval --pair q14 --executor q14
python posthoc/ds612.py rev-analyze
python posthoc/ds612.py selftest
```

The CPU analysis covers model-only candidates, both pools and transfer directions, budget curves, and the F1–F4 diagnostics. Each preference check uses frozen branches under `out_rev612/<pair>/`. The Qwen14B pair is frozen unconditionally, with the trigger statistic still reported and 15 controls added; the Llama pair follows its registered trigger. Each executor receives 30 fresh continuations per selected stochastic branch in batch R612. Keep the same ALFWorld data on both machines and return the complete evaluation and timing files before analysis. Rerunning an interrupted evaluation resumes compatible saved work. Timing files describe completed evaluation runs.

## Non-privileged repair pipeline G

Qwen3.8-flash with thinking reads each failed log, selects one step and proposes two alternative actions. Qwen7B uses three acceptance continuations (GV), 10 independent evaluation continuations (GE), and one greedy continuation (GG) per stochastic branch. The input is the section 6.9 pool; outputs go to `out_g612/`.

```bash
export QWEN_API_KEY=YOUR_API_KEY
python posthoc/g612.py check
python posthoc/g612.py all
python posthoc/g612.py analyze
python posthoc/g612.py selftest
```

`all` runs proposal preparation followed by continuations. API responses and completed branches are saved incrementally. Supply your own API key through the environment; never commit it.

## WebShop setup and feasibility checks: section 6.13

The paper uses the 100k-product configuration, not the 1,000-product preview. Obtain the upstream product, attribute and human-instruction files separately. `webshop/make_100k.py` prepares the product subset and fingerprints; `webshop/setup_webshop.sh` creates the environment and index. Inspect and configure their paths before running them.

The original WebShop environment runs in a separate process and Python environment from vLLM. The original setup used Python 3.12, Java 21, Pyserini 2.3.0, CPU PyTorch and spaCy for WebShop. The default search is Pyserini BM25. The optional pure-Python search shim is a different condition and must not silently replace it. `test_stubs/` is for offline workflow tests only and changes reward details.

```bash
bash webshop/setup_webshop.sh
python webshop/ws_smoke.py check --models q7,l8
python webshop/ws_smoke.py all --models q7,l8
python webshop/ws_smoke.py selftest
```

Use `--models q14` on the Qwen14B server. The revised smoke protocol uses 60 goals and a 25-step cap; `out_ws_smoke25/` records this version. The older 15-step run remains in `out_ws_smoke/` as a historical record. Cross-server reports check data, goal ordering, prices and search fingerprints before combining results. Configure `WS_WORKERS` for available RAM; the original worker footprint was approximately 3 GB per process.

## WebShop formal pools

| Profile | Deployed model | Output | Continuation executors |
|---|---|---|---|
| S | Qwen7B | `out_ws_S/` | Qwen7B: 2 stochastic + 1 greedy; workflow smoke only |
| A | Qwen7B | `out_ws_A/` | Qwen7B/Llama: 10 + 1 each; Qwen14B: 20 + 1 when available |
| B | Llama8B | `out_ws_B/` | Llama/Qwen7B: 10 + 1 each |

Formal proposers are Qwen7B, Llama and flash. Profile S uses Qwen7B and flash. Run profiles sequentially on the configured server:

```bash
export QWEN_API_KEY=YOUR_API_KEY
python webshop/ws_formal.py check --profile S
python webshop/ws_formal.py all --profile S
# After the smoke checks pass:
python webshop/ws_formal.py all --profile A
python webshop/ws_formal.py all --profile B
```

Use the CLI stage and model options to run individual proposal/evaluation batches on the corresponding server. Each output directory allows one active process. P2w is pending in this release; do not treat its missing records as failures. Consult the frozen plan before adding or merging its registered batch.

## Earlier gridworld and screened ALFWorld experiments

These workflows retain their specialized analysis code and raw records. They are not interchangeable with the unscreened-pool scorer. Run `python <script>.py --help` for their stage options; `run_bg.sh` dispatches named workflows and supports status/log inspection. It launches detached processes, so use it only when background execution is intended.

The earliest `run_cross.py` workflow expects a sibling `pilot_vshape` directory and its frozen `out_120` trajectories, which are not fully bundled here. Its original self-test compares selected source files with that pilot. Localization changes comments and display text, so a byte-identity check against an unlocalized external pilot will differ; use the unchanged original source for historical live-run identity checks rather than bypassing them.

Raw outputs typically include a frozen `design.json`, individual records in `rollouts.jsonl` or `evals_<executor>.jsonl`, and derived `analysis.json`/Markdown reports. Historical reports and cumulative research records remain unchanged, including their original language. New reports generated by the localized source use English display text.
