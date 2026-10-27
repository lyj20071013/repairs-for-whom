# Bundle schema: repair-eval/1

Each pool is an immutable directory:

- `manifest.json`: schema, pool_id, bundle_id, deployed executor, metrics, complete and pending executors, source SHA256 hashes and per-file hashes.
- `tasks.jsonl`: public inputs per trajectory: task, failure log, positions, executable-action information and candidate IDs/actions. Planner identity, candidate sources, oracle signals and rollout outcomes are hidden. Candidate order is shuffled with a fixed outcome-independent random stream; IDs are assigned afterward.
- `design.json`: joins for scoring and replay. Each candidate links alternative and original-action branches. Branches store prefixes, replay resource identifiers, deterministic outcomes and metadata.
- `outcomes/<executor>.jsonl`: one record per `(branch_id, mode)`. `sample` stores all individual observations; `greedy` stores one separate observation. Success and reward are distinct metrics; averages do not replace individual observations.

## Indices and prefixes

Public `step` and standardized branch `t` are **1-based**. The prefix includes the substituted action, so `len(prefix) == t`. Use `prefix[:-1]` to restore the state before intervention and `prefix` for the state afterward. Original ALFWorld indices are zero-based and WebShop indices are one-based; `source_t` and original branch IDs are retained for tracing records.

Both branches of a candidate pair must belong to the same trajectory and step, with identical prefixes except for the final action. The original action is a net-gain baseline, not an implicit candidate. Any no-intervention choice must be specified explicitly in the design.

ALFWorld paths are normalized to `json_2.1.1/.../game.tw-pddl` and resolved against the runtime data root. WebShop retains the goal index: use the same 100k-product configuration, goal ordering, price seed and BM25 search environment. Check code/data fingerprints together with observed replay prefixes; a model name alone does not establish identical environments or executors.

## Selectors

```python
def select(task: dict, context: dict):
    return task['candidates'][0]['id']
```

Return a candidate ID, a `{candidate_id: probability}` distribution summing to one, or `None` to abstain under the net objective. Each trajectory receives one position/action choice. By default, `context` contains only a seed. With `--uses-rollouts`, it also contains split, fold, metric, objective, budget, and alternative/original means on the selection fold for each candidate. The selector is called once per fold and must not accumulate samples across calls: the two folds exchange roles.

A fixed-prediction file has this form:

```json
{"bundle_id": "the exported manifest bundle_id", "choices": {"trajectory-id": "c003"}}
```

It must cover the entire pool; missing choices are not implicit abstentions. Binding predictions to bundle_id prevents accidental reuse after candidate ordering changes.

## New executors and completeness

Each executor must have the same number of stochastic samples plus one independent greedy observation for every stochastic branch. Partial results are excluded from formal scoring. Deterministic branches need no stochastic continuations. Import rejects duplicate keys, conflicting records, non-finite values, out-of-range rewards and incomplete branch coverage.

Changes to model, prompt, decoding or invalid-action rules require a new executor ID. Different protocols are not automatically merged under one model name.

`bundle_id` is the SHA256 of canonical manifest JSON with the bundle_id field omitted. Hashes detect accidental changes; they are not signatures from a trusted third party.
