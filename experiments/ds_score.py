"""Disconnect Dataset：Method gives a candidate number for each entry cid（or None Expression of waiver），Net deployment gain based on the label of the designated implementer。

    python ds_score.py --data out_ds_gw/dataset/gridworld.jsonl --pred my_predictions.json --executor q7

predictions.json：{Entry id: cid or null}。Missing entries by relinquishment（Proceeds 0）。
report：Average net deployment gains（Cluster with trajectories bootstrap 90% interval）、Drop ratio，And two references：
Random selection（Average of all candidates for this entry）and an approximate upper bound（Candidates with highest labels per entry；Label Noise，slightly optimistically biased）。
"""
import argparse
import json
import sys

import numpy as np


def score(items, pred, executor, n_boot=2000, seed=0):
    gains, rand, best, abst = [], [], [], 0
    for it in items:
        lab = it["labels"].get(executor)
        if lab is None:
            continue
        c = pred.get(it["id"])
        if c is None:
            gains.append(0.0)
            abst += 1
        else:
            if c not in lab:
                raise ValueError(f"{it['id']}: candidate {c} Not in the entry")
            gains.append(lab[c]["gain"])
        vals = [v["gain"] for v in lab.values()]
        rand.append(float(np.mean(vals)))
        best.append(float(np.max(vals)))
    g = np.array(gains)
    rng = np.random.default_rng(seed)
    if len(g) == 0:
        return {"n": 0}
    m = g[rng.integers(0, len(g), size=(n_boot, len(g)))].mean(axis=1)
    return {"n": len(g), "mean_gain": float(g.mean()), "ci90": [float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))],
            "abstain_frac": abst / len(g), "random_ref": float(np.mean(rand)), "upper_ref_approx": float(np.mean(best))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--executor", required=True)
    args = ap.parse_args()
    items = [json.loads(x) for x in open(args.data, encoding="utf-8") if x.strip()]
    pred = json.load(open(args.pred, encoding="utf-8"))
    print(json.dumps(score(items, pred, args.executor), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
