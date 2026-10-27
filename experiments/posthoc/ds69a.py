"""6.9a EXPLORATION ANALYSIS：model-proposed candidates only（analysis plan.md 6.9a）。Usage：python posthoc/ds69a.py out_ds_alfw（CPU，approximately 1 minutes）"""
import json, sys, copy
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import ds_alfw as D, runner
D.use_out(sys.argv[1])
d = D.jload(D.P("design"))
ev = runner.load_results(D.P("evals", "q7"))
br = d["branches"]
def restrict(d, keep):
    d2 = copy.deepcopy(d); empty = []
    for t in d2["trajectories"]:
        for p in t["positions"]:
            p["alts"] = [a for a in p["alts"] if keep(br[a])]
        t["positions"] = [p for p in t["positions"] if p["alts"]]
        if not t["positions"]:
            empty.append(t["traj_id"])
    d2["trajectories"] = [t for t in d2["trajectories"] if t["positions"]]
    return d2, empty
meta = lambda ti, b: br[b]
out = {}
for tag, keep in (("all", lambda b: True), ("model", lambda b: bool(b["proposers"])),
                  ("model_no_planner_overlap", lambda b: bool(b["proposers"]) and not b["planner"])):
    d2, empty = restrict(d, keep)
    T7 = [D._traj_arrays(d2, t, ev, "q7", 10) for t in d2["trajectories"]]
    res = {"n_traj": len(T7), "n_empty": len(empty)}
    for obj in ("net", "post"):
        R = D.analyze_core(T7, None, meta, obj=obj)
        # No trajectories of candidate：no repair（net gain 0）；only in net Add them to the mean value，post explain separately under
        pad = [0.0] * len(empty) if obj == "net" else []
        f = lambda k: [r[k] for r in R]
        dif = lambda a, b: [r[a] - r[b] for r in R] + pad
        r = {"matched": D.boot(f("matched") + pad), "greedy": D.boot(f("greedy") + pad),
             "matched-greedy": D.boot(dif("matched", "greedy")), "matched-random": D.boot(dif("matched", "random")),
             "frac_best_gt_0.10": float(np.mean([x > 0.10 for x in f("best_in_sample")] + [False] * len(empty)))}
        for k in (1, 2, 3, 5):
            r[f"7B_k{k}"] = D.boot(f(f"7B_k{k}") + pad)
            if all(f"SLfl_k{k}" in x for x in R):
                r[f"loss_SLfl_k{k}"] = D.boot(dif("matched", f"SLfl_k{k}"))
        r["recall_SLfl"] = float(np.mean(f("recall_SLfl")))
        res[obj] = r
    out[tag] = res
fmt = lambda b: f"{b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]"
for tag, res in out.items():
    print(f"== {tag}: trajectory {res['n_traj']}（No candidate {res['n_empty']}）")
    for obj in ("net", "post"):
        r = res[obj]
        print(f"  [{obj}] Match {fmt(r['matched'])}  greedy {fmt(r['greedy'])}  Match−greedy {fmt(r['matched-greedy'])}  "
              f"Match−random {fmt(r['matched-random'])}  best>0.10 {r['frac_best_gt_0.10']:.2f}")
        print("        7B k:", "  ".join(f"k{k} {fmt(r[f'7B_k{k}'])}" for k in (1, 2, 3, 5)))
        if "loss_SLfl_k5" in r:
            print("        SL-flash loss:", "  ".join(f"k{k} {fmt(r[f'loss_SLfl_k{k}'])}" for k in (1, 2, 3, 5)),
                  f" Recall {r['recall_SLfl']:.2f}")
json.dump(out, open(sys.argv[1] + "/analysis_69a.json", "w"), ensure_ascii=False, indent=1)
