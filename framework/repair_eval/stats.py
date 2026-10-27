"""Paired trajectory bootstrap; the resampling unit is never a rollout."""
import numpy as np


def bootstrap(values, seed=0, replicates=2000, confidence=0.90):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Bootstrap needs a nonempty, finite trajectory vector")
    if replicates < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(replicates, len(x)))].mean(axis=1)
    tail = (1 - confidence) / 2
    # Preserve the paper's exact quantiles for its default confidence.
    lo, hi = (0.05, 0.95) if confidence == 0.90 else (tail, 1 - tail)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(means, lo)), float(np.quantile(means, hi))],
            "p": float((np.sum(means <= 0) + 1) / (replicates + 1)), "n": len(x),
            "confidence": confidence, "replicates": replicates, "seed": seed,
            "p_method": "paper one-sided bootstrap tail; alternative mean > 0"}


def holm(tests, family_size, alpha=0.05):
    if family_size < len(tests) or family_size < 1:
        raise ValueError("Family size must include all available and pending tests")
    ranked = sorted((float(v["p"]), k) for k, v in tests.items())
    ranked += [(1.0, None)] * (family_size - len(ranked))
    running_max = 0.0
    for j, (p, name) in enumerate(ranked):
        running_max = max(running_max, min(1.0, (family_size - j) * p))
        if name is not None:
            tests[name].update(p_holm=running_max, holm_reject=running_max <= alpha,
                               family_size=family_size)
    return tests
