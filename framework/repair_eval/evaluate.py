"""Read-only selectors see public task fields and, optionally, a selection fold."""
import importlib
import copy
import numpy as np
from .stats import bootstrap


def full(a, means):
    v = np.nan_to_num(a["det"]).copy()
    v[a["rand"]] = means
    return v


def gains(a, values, objective="net"):
    return values[a["ia"]] - (values[a["io"]] if objective == "net" else 0.0)


def weights(choice, ids):
    if choice is None:
        return np.zeros(len(ids))
    if isinstance(choice, str):
        choice = {choice: 1.0}
    if not isinstance(choice, dict) or not choice or set(choice) - set(ids):
        raise ValueError("Selection must be a candidate ID or a nonempty candidate probability mapping")
    w = np.asarray([choice.get(cid, 0.0) for cid in ids], float)
    if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Selection probabilities must be finite, nonnegative, and sum to one")
    return w


def best_weights(g):
    mask = g >= g.max() - 1e-12
    return mask.astype(float) / mask.sum()


def load_selector(spec):
    module, name = spec.split(":", 1)
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise ValueError("Selector is not callable")
    return fn


def evaluate(ds, selector=None, predictions=None, uses_rollouts=False, budget=5,
             metric=None, objective="net", splits=50, seed=0, replicates=2000):
    if objective not in ("net", "post") or splits < 1:
        raise ValueError("Invalid objective/splits")
    if (selector is None) == (predictions is None):
        raise ValueError("Provide either a selector or a predictions file")
    if predictions is not None:
        if predictions.get("bundle_id") != ds.manifest["bundle_id"]:
            raise ValueError("Predictions belong to a different bundle")
        predictions = predictions["choices"]
        if set(predictions) != {t["id"] for t in ds.tasks}:
            raise ValueError("Predictions must contain every task exactly once; missing is not abstention")
        if uses_rollouts:
            raise ValueError("Fixed predictions cannot declare access to cross-fitting folds")
    metric = metric or ds.manifest["primary_metric"]
    arrays = ds.arrays(ds.manifest["deployer"], metric)
    rows = []
    for ti, (task, tr, a) in enumerate(zip(ds.tasks, ds.trajs, arrays)):
        ids = [c["id"] for c in tr["candidates"]]
        n = a["S"].shape[1]
        h = n // 2
        if n % 2 or not 1 <= budget <= h:
            raise ValueError(f"Requires even deployed sample count and 1 <= budget <= {h}")
        rng = np.random.default_rng([seed, ti])
        selected, matched = [], []
        gd = gains(a, full(a, a["S"].mean(axis=1)), objective)
        # No-rollout selection is called exactly once and never sees any labels.
        if not uses_rollouts:
            c = predictions[task["id"]] if predictions is not None else selector(copy.deepcopy(task), {"seed": seed+ti})
            if c is None and objective == "post":
                raise ValueError("Abstention has no unique post-repair baseline across positions")
            w = weights(c, ids)
            selected.append(float(w @ gd))
        for split in range(splits):
            pm = np.argsort(rng.random(a["S"].shape), axis=1)
            z = np.take_along_axis(a["S"], pm, axis=1)
            for fold, (aa, bb) in enumerate(((slice(0, h), slice(h, n)), (slice(h, n), slice(0, h)))):
                sel = full(a, z[:, aa][:, :budget].mean(axis=1))
                ev = gains(a, full(a, z[:, bb].mean(axis=1)), objective)
                matched.append(float(best_weights(gains(a, sel, objective)) @ ev))
                if uses_rollouts:
                    # Only the selection outcomes reach the plugin. No held-out values or metadata.
                    context = {"seed": seed+ti, "split": split, "fold": fold, "objective": objective,
                               "metric": metric, "budget": budget, "selection": {}}
                    # Use the public randomized candidate order, never the source
                    # design order (which often puts the privileged candidate first).
                    index = {cid: ci for ci, cid in enumerate(ids)}
                    for public_candidate in task["candidates"]:
                        cid = public_candidate["id"]
                        ci = index[cid]
                        context["selection"][cid] = {"alternative": float(sel[a["ia"][ci]]),
                                                      "original": float(sel[a["io"][ci]])}
                    c = selector(copy.deepcopy(task), context)
                    if c is None and objective == "post":
                        raise ValueError("Post-repair objective does not define abstention")
                    selected.append(float(weights(c, ids) @ ev))
        value, ref = float(np.mean(selected)), float(np.mean(matched))
        rows.append({"trajectory": task["id"], "value": value, "matched": ref,
                     "loss": ref-value, "random": float(gd.mean())})
    result = {"bundle_id": ds.manifest["bundle_id"], "pool": ds.manifest["pool_id"],
              "executor": ds.manifest["deployer"], "metric": metric, "objective": objective,
              "protocol": "standard-v1; per-trajectory independent split streams",
              "settings": {"budget": budget, "splits": splits, "seed": seed, "bootstrap": replicates},
              "selector_access": "selection fold only" if uses_rollouts else "public task only",
              "values": bootstrap([r["value"] for r in rows], seed, replicates),
              "selection_loss": bootstrap([r["loss"] for r in rows], seed, replicates),
              "positive_control": bootstrap([r["matched"]-r["random"] for r in rows], seed, replicates),
              "rows": rows}
    result["positive_control"]["pass"] = result["positive_control"]["ci"][0] > 0.10
    return result
