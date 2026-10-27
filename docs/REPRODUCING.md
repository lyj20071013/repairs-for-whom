# Reproducing the experiments

## CPU result reproduction

From the repository root, install `requirements-offline.txt`, run `python tools/verify_release.py`, then change to `framework/` and run `python -m repair_eval reproduce`. This recalculates the four primary unscreened pools using the archived protocol/reference profiles. See `framework/docs/PROTOCOL.md` for statistical details.

The original analysis scripts remain under `experiments/`. Some commands write reports into `out_*`; run them in a disposable copy when inspecting the released raw snapshot. The tested framework path writes to `framework/outputs/` instead. Original scripts contain historical path defaults and runtime-specific dependencies; their mere presence is not a claim that every script has been ported to Windows.

## Regenerate figures

From the repository root:

```bash
python -m pip install -r requirements-figures.txt
python paper_figs/fig1_example.py --rt experiments --out outputs/figures/fig1_example.pdf
python paper_figs/fig2_forest.py --rt experiments --out outputs/figures/fig2_forest.pdf
python paper_figs/fig3_curves.py --rt experiments --out outputs/figures/fig3_curves.pdf
python paper_figs/fig3_budget.py --rt experiments --out outputs/figures/fig3_budget.pdf
```

The figure scripts preserve the paper's historical source choices, including the original H1 record in the forest plot. The framework's H1 verification uses the documented recovered/rerun record. Do not relabel one as the other.

## New model continuations

Read `framework/docs/REPLAY.md`. It describes job plans, seed handling, prefix verification, interruption recovery and importing completed results as a new dataset version. First use a tiny live integration test in your own GPU environment. This packaging run did not execute new GPU continuations.

For the original experiment workflow, see `experiments/LEGACY_USAGE.md`. Linux environment helpers are `experiments/alfw_setup.sh` and `experiments/webshop/setup_webshop.sh`; inspect their paths before use. The former expects an existing compatible vLLM environment. The WebShop setup uses a separate environment from the inference process. The repository does not supply an exact universal GPU dependency lock.

## External environment data and models

- ALFWorld: [official repository](https://github.com/alfworld/alfworld). Use the text environment and configure `ALFWORLD_DATA`; defaults are documented in `experiments/alfw/acfg.py`.
- WebShop: [official repository](https://github.com/princeton-nlp/WebShop). The paper uses 100,000 products, not the 1,000-product preview. `experiments/webshop/make_100k.py` prepares the subset and a data fingerprint from the original product/attribute files. The human instruction file is also required. Compare fingerprints to `provenance/webshop_data_100k_manifest.json` when available and the raw run metadata. Upstream download availability is external to this release.
- Default WebShop search is Pyserini/BM25. Do not substitute the optional pure-Python shim and claim identical paper conditions without validating its behavior.
- Models: Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct and Llama-3.1-8B-Instruct are downloaded separately under their providers' terms. Configure the paths in the original config modules / documented environment variables. API calls use `QWEN_API_KEY`, provided by the user; no key is included.

## Completed P2w snapshot

The canonical `framework/data/webshop-q7/` bundle includes the original registered Qwen2.5-14B batch: 2,188 stochastic branches, each with 20 sampled continuations and one greedy continuation (45,948 continuations total). The 63 terminal branches use their known outcomes. No new inference was run during packaging.

Raw outcomes and timings are in `experiments/out_ws_A/evals_q14.jsonl` and `timing_q14.json`. Joint tests are in `experiments/out_ws_summary/tests_ws.json`. Run the CPU reproduction command above to check P2w and its budget curve against the updated reference. The old bundle and references remain in Git history; pre-Q14 raw analyses are retained alongside the new results.

For future additions, import into a fresh output directory, validate complete branch coverage, then update references, documentation and release hashes together. Do not represent a new replay batch as the original registered experiment.

## Manuscript

`manuscript/example_paper.tex` and `.bib` are the current source snapshot. The manuscript includes completed P2w results and the named-author arXiv source. Use a compatible TeX installation to build it; this packaging run does not claim a successful LaTeX build. Existing figure PDFs are included. The arXiv identifier will be added after publication.
