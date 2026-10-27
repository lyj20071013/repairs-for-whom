# Scoring protocol and versions

## Estimand

A repair replaces one action at one position. Each candidate links an alternative branch to the original-action branch at **the same position**. For the designated deployed executor:

`gain = mean(alternative outcome) - mean(original-action outcome)`

ALFWorld uses binary success. WebShop uses reward in [0, 1] as its primary outcome and success as a secondary outcome. The original-action baseline varies by position: the failed source trajectory does not imply a zero continuation baseline. The `post` objective uses only the post-repair outcome and is distinct from net gain.

`selection_loss = held-out value of matched selection - held-out value of the method`

Matched selection uses finite samples; it is not a true oracle, and selection loss may be negative. Maximizing held-out labels does not give an unbiased upper bound. Abstention has net gain zero. This version rejects `post` plus abstention because there is no unique position-specific baseline for that combination.

## Sample separation

The available deployed-executor records have 10 samples per stochastic branch. Independently shuffle each branch's samples into two folds of five, swap selection and evaluation roles, and repeat 50 times. Deployed-executor selection sees only its selection fold, using k <= 5 samples. Both alternative and original-action branches follow this split.

Fixed predictions, greedy replay and other-executor selection do not use the deployed executor's stochastic selection samples. They are scored on all deployed-executor samples, equivalent to averaging over the symmetric evaluation folds. Greedy replay comes from a separate batch, not a chosen stochastic sample.

Deterministically terminal branches use their known outcomes; no artificial stochastic observations or rollout costs are assigned. Ties within 1e-12 are averaged uniformly, without favoring planner actions through candidate order.

A sample is identified by `(executor, mode, branch, sample index)`. Historical seeds are reconstructed according to their original definitions. This establishes the seed-generation convention, not verification of every random state in the inference stack. Original records do not fully specify weight revisions; token-for-token reproduction across hardware is not claimed.

## Statistics

Average cross-fitting repetitions within each trajectory, then bootstrap trajectories 2,000 times for percentile 90% confidence intervals. Primary comparisons use paired differences. The one-sided tail proportion `(#bootstrap means <= 0 + 1)/(B+1)` follows the paper's convention; it is not an exact randomization test.

The positive-control gate requires the lower 90% confidence bound of matched minus random to exceed 0.10. Results remain reportable if this fails, but a small substitution loss then does not establish that the substitute is adequate.

New methods evaluated with `evaluate` receive unadjusted descriptive comparisons. They do not become confirmatory tests automatically. `reproduce` uses the registered families: P1/P2 (m=2), H1/H2/H3 (m=3), and P1w/P2w/H1w/H2w/H3w (m=5). Missing tests enter their family with p=1.

## Historical numerical versions

- `legacy69` extracts numerical functions from `archive/rt_before_merge/ds_alfw_1240.py`. Its single random stream is consumed sequentially across trajectories and 14B budget samples, so the frozen comparison profile must be retained. It is used for the original ALFWorld Qwen7B P1/P2 analyses.
- `standard` extracts numerical functions from the current `ds_alfw.py`. Deployed-executor streams are separated by trajectory; other-executor streams by trajectory and executor. It is used for later method comparisons and WebShop/Llama analyses.
- Historical H1 in `tests_610.json` is 0.074470916; `tests_610_rerun.json` and `out_612/analysis_612.json` contain 0.071217305. Current code reproduces the latter exactly. Both references and their source hashes are retained. Seeds are not adjusted to match the older number. The full numerical state of that historical run has not been located, so the cause cannot be inferred from the size of the discrepancy alone.

`provenance/scoring_sources.json` records original function sources and hashes. `paper_reference.json` is used only for verification; reference means are not inputs to scoring. English localization changes comments and display text, not the numerical implementation; see the repository localization audit for packaged hashes.

## Intended use

This is a public research resource, not a hidden test server. Tuning against published outcomes requires a separate development/test trajectory split or fresh failures. Hiding evaluation folds from the selector API prevents ordinary programming mistakes, but cannot stop a plugin from reading other local files. Report actual information access and budgets.

The unified scorer currently implements the unscreened-pool protocol. Gridworld two-stage sampling, screened-pool hierarchical bootstrap, and preference-reversal checks following disputed-case selection require their own estimators and must not simply be pooled into this scorer.
