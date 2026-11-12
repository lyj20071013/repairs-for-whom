# Release status

Data snapshot: **2026-11-13** (final manuscript). Previous snapshot: 2026-10-28, including completed P2w. The 2026-11-13 update is described in the next section; the remaining sections describe the 2026-10-28 packaging and still apply to the files it covers.

## 2026-11-13 update

Added: the successor-training and staleness studies and the τ²-bench pilots, main pool and label-training study (`followup/`); the τ²-bench 2B ← 4B shared-pool analysis (`experiments/tau2r/`, `experiments/out_tau_pool/`); post hoc analyses (`experiments/posthoc/ratio_loss.*`, `headroom.*`, `rev14.py`, `experiments/out_rev14/`); the November registration notes (`analysis_plan/amendments_202611/`). Replaced: `analysis_plan/论文方案.md` (cumulative plan, now including 6.14). Removed: `manuscript/` (LaTeX source; the paper is on arXiv as 2610.38610). Unchanged: all other 2026-10-28 files, including `framework/` and its reference records.

Checks run during this packaging (CPU only, no new inference):

- `python tools/verify_release.py` against the regenerated SHA256 inventory.
- Every added file is a byte copy of the file on the authors' machine, with one exception: a line holding a server SSH address in `followup/repair_budget/tau2_pilot/步骤_tau2先导.md` was replaced by a note. Source paths and hashes are in `provenance/update_20261113.json`.
- Four archived reports were regenerated from the released records and compared byte-for-byte; all were identical: the τ²-bench shared-pool primary analysis (`experiments/out_tau_pool/report_main_v3.md`, via `tau2r/pool_analyze.py`), the τ²-bench H1/H2 report (`report_h1h2.md`, via `tau2r/h1h2_analyze.py report`), the post hoc loss-share and headroom tables (`posthoc/ratio_loss.*`, `posthoc/headroom.*`), and the Experiment B report (`followup/repair_budget/alfw_stale/out_np/report_np.*`). Commands are in `provenance/update_20261113.json`.
- A text scan for credentials, SSH settings and e-mail addresses over all added files.

Not checked: the other follow-up reports were not regenerated (several need τ²-bench source runs or model servers that are not part of the release; for example, the τ²-bench training study's end-to-end line reads `results.json` files that are not included). No GPU or environment run was repeated. The added code was not localized and was not covered by the framework unit tests.

## Verified (2026-10-28 packaging)

- 19 framework unit tests passed, including cross-fitting, outcome isolation, integrity checks and mock replay/resume.
- All four standardized bundles passed schema, identity, candidate/branch joins, outcome coverage and hash validation.
- CPU reproduction passed all 14 comparisons: four pool-level row comparisons and ten statistical checks. Maximum absolute difference from their archived reference records was zero. P2w is included.
- The original packaging record contains four figure-generation checks. The P2w update reruns the changed forest plot and checks all plotted values against the original analysis sources; this is not a new visual review of every figure.
- Original source-copy hashes and the legacy experimental MANIFEST were checked. The English edition additionally localizes source comments, docstrings and runtime text. Computation and originally English prompt/parser literals are unchanged; research-result bytes are preserved. The original code MANIFEST is retained in provenance and the installed MANIFEST reflects localized source hashes.
- The release has a SHA256 file inventory and a standalone integrity checker. Validation details are in `provenance/release_validation.json`.

## Scope

This snapshot includes the four primary unscreened pools plus the original supplementary experimental records. Smoke runs and historical screened pools are labeled in the data index and must not be counted as additional confirmatory samples. Stored continuation records contain individual outcomes and metadata; complete generated text for every continuation is not available.

P2w (WebShop Qwen 14B evaluation) is complete and included in both raw and standardized records. Historical H1 and its recovered/rerun version are both retained; the framework reproduces the latter, while the original forest-plot script uses the historical record. The analysis plan is a cumulative local record, not an independently timestamped public preregistration.

No fresh GPU/environment integration run or LaTeX compilation was performed during packaging. Model weights, environment installations and the WebShop product corpus/index must be obtained separately. Human annotation submissions are not included.

Original code is Apache-2.0; third-party and environment-derived material retain their respective rights. The source repository is https://github.com/lyj20071013/repairs-for-whom. No DOI, arXiv identifier or conference acceptance is claimed by this snapshot.

## English edition

The public guides and code-facing explanations were localized on 2026-10-28. The cumulative analysis plan, historical Markdown reports and raw JSON/JSONL records retain their original language. `english_localization.json` records source hashes, AST checks and record-integrity checks. Source-file hashing intentionally detects this edition as different from the historical source; do not bypass resume or external-pilot byte-identity checks. Use fresh output directories for new live runs.
