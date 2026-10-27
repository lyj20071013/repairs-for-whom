"""Minimal selectors. No model calls or private outcome access."""


def uniform(task, context):
    ids = [c["id"] for c in task["candidates"]]
    return {cid: 1 / len(ids) for cid in ids}


def earliest(task, context):
    step = min(c["step"] for c in task["candidates"])
    ids = [c["id"] for c in task["candidates"] if c["step"] == step]
    return {cid: 1 / len(ids) for cid in ids}


def sampled_gain(task, context):
    """Matched selector example: uses only the supplied selection fold."""
    scores = {cid: v["alternative"] - (v["original"] if context["objective"] == "net" else 0)
              for cid, v in context["selection"].items()}
    best = max(scores.values())
    ids = [cid for cid, v in scores.items() if v >= best - 1e-12]
    return {cid: 1 / len(ids) for cid in ids}
