# Release status

Data snapshot: **2026-10-28**, including completed P2w. Packaging verified on **2026-10-28**.

## Verified

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
