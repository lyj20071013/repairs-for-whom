"""6.12：Additional analysis of integration of results（A、B、C）Reverse Check with Preferred（D）。For registration see repair_transfer/analysis plan.md 6.12（Including Revisions 1、2）。

Put it on repair_transfer_llama（6.10 %1 of the code and data in the folder）of posthoc/ under；Hes not here MANIFEST in，Adding in does not affect a running mission。
Read-only ds_alfw.py output of the，Dont change anything about it。

    python posthoc/ds612.py analyze                              # A、B、C -> out_612/report_612.md、analysis_612.json（CPU，A few minutes）
    python posthoc/ds612.py rev-freeze --pair q14                # D：7B vs 14B，Unconditional freezing（Revision 1）+ 15 Article Contrast -> out_rev612/q14/
    python posthoc/ds612.py rev-freeze --pair l8                 # D：7B vs Llama，On trigger；Freezing only when satisfied -> out_rev612/l8/
    python posthoc/ds612.py rev-eval --pair q14 --executor q7    # Append each branch of the frozen branch 30 runs（New Batch R612）；--executor q14 Same thing
    python posthoc/ds612.py rev-analyze                          # D：-> out_rev612/report.md
    python posthoc/ds612.py selftest                             # Simulate Data Passage + known-answer test

data：out_ds_alfw/（6.9 Design；evals_q7 / timing_q7、evals_q14 / timing_q14（Run from here 14B copy from the machine running）、
evals_l8 / timing_l8（p610 of ①））and out_ds_alfw_l8/（6.10 ②：evals_l8、evals_q7）。Which executor is missing?，skip comparisons requiring it。
"""
import argparse
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ds_alfw as DS          # noqa: E402
import runner                 # noqa: E402
import alfw_stage_b as SB     # noqa: E402
from alfw import acfg         # noqa: E402

OUT = {"a": ROOT / "out_612", "rev": ROOT / "out_rev612"}
POOLS = {"7B": {"dir": ROOT / "out_ds_alfw", "deployer": "q7", "proposers": ("q7", "q14", "qflash")},
         "Llama": {"dir": ROOT / "out_ds_alfw_l8", "deployer": "l8", "proposers": ("l8", "q7", "qflash")}}
EXECS = ("q7", "q14", "l8")
LAB = DS.LABEL
BATCH = "D"
K_EQ = 5
KS_SELF = (1, 2, 3, 4, 5)       # additionally compute only k = 4（Simulate for）；Random numeric usage versus ds_alfw exactly identical，recomputed P1–H3 matches it element by element
REV = {"batch": "R612", "n": 30, "k_self": 5, "k_other": {"q14": 20, "l8": 10},
       "unconditional": ("q14",),                  # 6.12 Revision 1：14B A pair is no longer subject to trigger conditions
       "control": {"q14": {"n": 15, "seed": 2612}}}  # 6.12 Revision 1：Controls
N_BOOT_CASE = 500


def log(msg):
    print(msg, flush=True)


def f3(b):
    return f"{b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]" if b and b.get("n") else "—"


# ================================================================ Read Data
def load_pool(name, pdir=None):
    spec = POOLS[name]
    pdir = Path(pdir or spec["dir"])
    if not (pdir / "design.json").exists():
        return None
    d = DS.jload(pdir / "design.json")
    stoch = [b for b, x in d["branches"].items() if x["deterministic"] is None]
    ex, incomplete = {}, {}
    for n in EXECS:
        f = pdir / f"evals_{n}.jsonl"
        if not f.exists():
            continue
        ev = runner.load_results(f)
        miss = sum((f"{n}_T07", BATCH, b) not in ev or (f"{n}_greedy", BATCH, b) not in ev for b in stoch)
        if miss:
            incomplete[n] = miss
            continue
        ex[n] = {"ev": ev, "n": min(len(ev[(f"{n}_T07", BATCH, b)]["results"]) for b in stoch)}
    timing = {}
    for n in ex:
        f = pdir / f"timing_{n}.json"
        if f.exists():
            runs = DS.jload(f)["runs"]
            for cond in ("T07", "greedy"):
                rr = [r for r in runs if r["cond"] == f"{n}_{cond}/{BATCH}" and r.get("rollouts")]
                if rr:
                    timing[(n, cond)] = {"sec": sum(r["wall"] for r in rr) / sum(r["rollouts"] for r in rr),
                                         "tp": rr[-1].get("tp", 1), "gpu": (rr[-1].get("gpu") or {}).get("names")}
    return {"name": name, "dir": pdir, "d": d, "ex": ex, "incomplete": incomplete, "timing": timing,
            "deployer": spec["deployer"], "proposers": spec["proposers"],
            "analysis": DS.jload(pdir / "analysis.json") if (pdir / "analysis.json").exists() else None}


def arrays(pool, d, n):
    e = pool["ex"][n]
    return [DS._traj_arrays(d, t, e["ev"], n, e["n"]) for t in d["trajectories"]]


def run_core(pool, d, T, others, obj):
    TD = arrays(pool, d, T)
    O = {o: arrays(pool, d, o) for o in others}
    meta = lambda ti, b: pool["d"]["branches"][b]
    rows = DS.analyze_core(TD, O, meta, obj=obj, ks_self=KS_SELF, proposers=pool["proposers"])
    for r, t in zip(rows, d["trajectories"]):
        r["traj_id"] = t["traj_id"]
        r.pop("best_is", None)
    return rows


def k_eq(pool, T, o):
    return min(K_EQ, pool["ex"][T]["n"] // 2, pool["ex"][o]["n"])


# ================================================================ A、B、C
def per_traj(rows, fn, all_ids, pad):
    """A value per trajectory；trajectory absent from the design（Model Subset is empty）Time pad（None = Not included）。"""
    by = {r["traj_id"]: fn(r) for r in rows}
    out = []
    for tid in all_ids:
        if tid in by:
            out.append(by[tid])
        elif pad is not None:
            out.append(pad)
    return out


def boot_indep_diff(x, y, seed=0, n=DS.N_BOOT):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    mx = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    my = y[rng.integers(0, len(y), size=(n, len(y)))].mean(axis=1)
    m = mx - my
    return {"mean": float(x.mean() - y.mean()), "ci": [float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))],
            "p": float((np.sum(m <= 0) + 1) / (n + 1)), "n": int(min(len(x), len(y)))}


def analyze(pool_dirs=None, out_dir=None):
    pool_dirs = pool_dirs or {}
    out_dir = Path(out_dir or OUT["a"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pools = {k: load_pool(k, pool_dirs.get(k)) for k in POOLS}
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "status": {}, "cells": {}, "tests": {}, "A": {}, "B": {},
           "C": {}, "check": {}}
    for name, pool in pools.items():
        res["status"][name] = None if pool is None else {
            "dir": str(pool["dir"]), "complete": {n: pool["ex"][n]["n"] for n in pool["ex"]},
            "incomplete_branches": pool["incomplete"], "n_traj": len(pool["d"]["trajectories"])}
    # ---- Cell-by-calculation：Ji! × Candidatures Gathered × Target implementer × Objective
    rows = {}
    variants = {}
    for name, pool in pools.items():
        if pool is None or pool["deployer"] not in pool["ex"]:
            continue
        d_all = pool["d"]
        d_mod, empty = DS.restrict_design(d_all, lambda b: bool(b.get("proposers")))
        variants[name] = {"all": (d_all, []), "model": (d_mod, empty)}
        for var, (dv, _) in variants[name].items():
            for T in pool["ex"]:
                others = [o for o in pool["ex"] if o != T]
                for obj in ("net", "post"):
                    rows[(name, var, T, obj)] = run_core(pool, dv, T, others, obj)
    ids = {name: [t["traj_id"] for t in pools[name]["d"]["trajectories"]] for name in variants}

    def series(name, var, T, obj, fn, level=False):
        key = (name, var, T, obj)
        if key not in rows:
            return None
        try:
            fn(rows[key][0])
        except (KeyError, IndexError):
            return None
        # Subset as an empty trajectory：margin mark 0（both abstain from repair）；Net gain on horizontal value 0、Post-repair success rate not included（Same 6.9a）
        pad = None if var == "all" else (0.0 if (not level or obj == "net") else None)
        return per_traj(rows[key], fn, ids[name], pad)

    def eq(name, var, T, o, obj):
        if (name, var, T, obj) not in rows or o not in pools[name]["ex"]:
            return None
        k = k_eq(pools[name], T, o)
        return series(name, var, T, obj, lambda r: r[f"self_k{k}"] - r[f"{o}_k{k}"])

    def b_or_none(x):
        return DS.boot(x) if x is not None and len(x) else None

    # ---- Main test of registration（Recalculation）and A：Model Subset
    spec = {"P1": ("7B", "q7", None, "Match − 7B greedy"), "P2": ("7B", "q7", "q14", "Same thing 5 runs：7B select using the deployed executor − by 14B Selection"),
            "H1": ("7B", "q7", "l8", "Same thing 5 runs：7B select using the deployed executor − by Llama Selection"),
            "H2": ("Llama", "l8", "q7", "Same thing 5 runs：Llama select using the deployed executor − by 7B Selection"), "H3": ("Llama", "l8", None, "Match − Llama greedy")}
    for tid, (name, T, o, lab) in spec.items():
        if name not in variants or T not in pools[name]["ex"]:
            continue
        entry = {"label": lab, "pool": name}
        for var in ("all", "model"):
            for obj in ("net", "post"):
                if o is None:
                    x = series(name, var, T, obj, lambda r: r["matched"] - r["greedy"])
                else:
                    x = eq(name, var, T, o, obj)
                entry[f"{var}_{obj}"] = b_or_none(x)
        res["tests"][tid] = entry
    for name in variants:
        T = pools[name]["deployer"]
        res["tests"][f"PC_{name}"] = {"label": f"positive control（{name} Ji!）：Match − random", "pool": name,
                                      **{f"{v}_{o}": b_or_none(series(name, v, T, o, lambda r: r["matched"] - r["random"]))
                                         for v in ("all", "model") for o in ("net", "post")}}
    # A interpretation（Registration）
    for tid in ("P2", "H1", "H2"):
        t = res["tests"].get(tid)
        if not t or not t.get("all_net") or not t.get("model_net"):
            continue
        full_ok, sub = t["all_net"]["ci"][0] > 0, t["model_net"]
        if sub["ci"][0] > 0:
            v = "Model Subset Up and Down > 0：The executor relies on not being a planner candidate"
        elif full_ok:
            v = "All candidates up and down > 0，Model Sub-area Containment 0：Candidate construction is the key limit"
        else:
            v = "All the candidates and subsets are uncertain"
        res["A"][tid] = v
    # ---- B：Two pools × Two directions（q7 and l8）
    dirs = {"l8→q7": ("q7", "l8"), "q7→l8": ("l8", "q7")}       # Orientation X→Y：by X Selection、Here Y use = Objective Y、other X
    for name in variants:
        for dn, (T, o) in dirs.items():
            for var in ("all", "model"):
                for obj in ("net", "post"):
                    if T in pools[name]["ex"] and o in pools[name]["ex"]:
                        x = eq(name, var, T, o, obj)
                        res["B"][f"{name}|{dn}|{var}|{obj}"] = b_or_none(x)
                        res["B"].setdefault("_series", {})[f"{name}|{dn}|{var}|{obj}"] = x
    S = res["B"].get("_series", {})
    for dn in dirs:                       # Orientation（executor）Effects：Is the same direction in the two pools?
        for var in ("all", "model"):
            a, b = S.get(f"7B|{dn}|{var}|net"), S.get(f"Llama|{dn}|{var}|net")
            if a is not None and b is not None:
                res["B"][f"pooldiff|{dn}|{var}"] = boot_indep_diff(a, b)
    for name in variants:                 # The difference between the two directions in the pool（Same trajectories，Match）
        for var in ("all", "model"):
            a, b = S.get(f"{name}|l8→q7|{var}|net"), S.get(f"{name}|q7→l8|{var}|net")
            if a is not None and b is not None:
                res["B"][f"dirdiff|{name}|{var}"] = DS.boot([x - y for x, y in zip(a, b)])
    if "7B" in variants and "q14" in pools["7B"]["ex"]:   # description（Hes not here 6.12 List）：7B In the pool 7B Select to 14B use
        for var in ("all", "model"):
            res["B"][f"7B|q7→q14|{var}|net"] = b_or_none(eq("7B", var, "q14", "q7", "net"))
    res["B"].pop("_series", None)
    # ---- C：Budget curve（Number of times and CMS seconds）and SL
    for name in variants:
        pool, T = pools[name], pools[name]["deployer"]
        # Number of random branches per trajectory（Model Subset by Subset；No trajectories for candidates 0）
        nbs = {var: sum(r["budget_all"] for r in rows[(name, var, T, "net")]) / len(ids[name]) for var in ("all", "model")}
        cur = {"deployer": T, "branches_per_traj": nbs["all"], "branches_per_traj_model": nbs["model"], "rows": []}

        def cost(sel, k, cond="T07"):
            tm = pool["timing"].get((sel, cond))
            return (k * nb, k * nb * tm["sec"] * tm["tp"] if tm else None)
        for var in ("all", "model"):
            nb = nbs[var]
            for sel in [T] + [o for o in pool["ex"] if o != T]:
                ks = KS_SELF if sel == T else (1, 2, 3, 5, 10, 20)
                for k in ks:
                    key = f"self_k{k}" if sel == T else f"{sel}_k{k}"
                    vals = {obj: b_or_none(series(name, var, T, obj, lambda r, key=key: r[key], level=True))
                            for obj in ("net", "post")}
                    if vals["net"] is None:
                        continue
                    ro, gs = cost(sel, k)
                    cur["rows"].append({"var": var, "selector": sel, "k": k, "rollouts": ro, "gpu_sec": gs, **vals})
                gkey = "greedy" if sel == T else f"greedy_{sel}"
                vals = {obj: b_or_none(series(name, var, T, obj, lambda r, gkey=gkey: r[gkey], level=True))
                        for obj in ("net", "post")}
                ro, gs = cost(sel, 1, "greedy")
                cur["rows"].append({"var": var, "selector": sel, "k": "greedy", "rollouts": ro, "gpu_sec": gs, **vals})
            for o, kk in REV["k_other"].items():       # D %1 trigger statistics（And as a description report）
                if o in pool["ex"] and T == "q7":
                    k2 = min(kk, pool["ex"][o]["n"])
                    ks_ = min(REV["k_self"], pool["ex"][T]["n"] // 2)
                    cur[f"trigger_{o}_{var}"] = b_or_none(series(name, var, T, "net",
                                                                 lambda r, o=o, k2=k2, ks_=ks_: r[f"self_k{ks_}"] - r[f"{o}_k{k2}"]))
        sl = {}
        R = rows[(name, "all", T, "net")]
        for tag in ("SLpl", "SLfl"):
            for k in (1, 2, 3, 5):
                if all(f"{tag}_k{k}" in r for r in R):
                    sl[f"{tag}_k{k}"] = {"loss": DS.boot([r["matched"] - r[f"{tag}_k{k}"] for r in R]),
                                         "budget": float(np.mean([k * r[f"budget_{tag}"] for r in R]))}
            sl[f"recall_{tag}"] = float(np.mean([r[f"recall_{tag}"] for r in R]))
        cur["SL"] = sl
        res["C"][name] = cur
    old = ROOT / "out_method" / "method_alfw.json"
    if old.exists():
        m = DS.jload(old)
        res["C"]["SL_old_screened"] = {k: {"budget": v["budget"], "loss": v["loss"]} for k, v in m["rows"].items()
                                       if k.startswith("screening m=1 × ") or k in ("greedy 1 runs",)}
    # ---- Consistency reconciliation：recomputed P1–H3 and ds_alfw of analysis.json
    for tid, (name, T, o, _) in spec.items():
        if tid in res["tests"] and pools[name] and pools[name]["analysis"]:
            got = pools[name]["analysis"]["tests"].get(tid)
            mine = res["tests"][tid].get("all_net")
            if got and mine:
                res["check"][tid] = {"ds_alfw": got["mean"], "ds612": mine["mean"], "diff": mine["mean"] - got["mean"]}
    DS.jdump(out_dir / "analysis_612.json", res)
    (out_dir / "report_612.md").write_text(report(res), encoding="utf-8")
    log(f"[612] -> {out_dir / 'report_612.md'}")
    return res


def report(res):
    L = ["# 6.12 Additional analysis of integration of results（A、B、C）", "",
         "For registration see analysis plan.md 6.12（2026-10-24 21:15，P2 and 6.10 Before the official results come out,）。A、B Additional analysis of registration prior to outcome，exploratory、excluded from Holm；"
         "P1–H3 registered tests recomputed with the same implementation（The formal determination shall be as follows: ds_alfw Reporting status，See consistency check）。Value = 90% trajectory bootstrap interval。", "",
         f"Generate Time {res['generated']}。Data Status：{json.dumps(res['status'], ensure_ascii=False)}", ""]
    if res["check"]:
        def verdict(dv):
            return ("Same one by one" if abs(dv) < 1e-9 else
                    "The Monte Carlo error of rehearsing（analysis.json Mostly from repair_transfer Old code：A single random stream，See 6.10）" if abs(dv) < 0.005
                    else "different，needs inspection")
        L += ["**Consistency reconciliation**（all candidates、net gain；and within the same pool ds_alfw Generated analysis.json comparison）：" +
              "；".join(f"{k} ds_alfw {v['ds_alfw']:+.4f} / This is the script {v['ds612']:+.4f}（{verdict(v['diff'])}）"
                       for k, v in res["check"].items()), ""]
    L += ["## primary tests（Recalculation）and A：model-proposed candidates only", "",
          "| Read | Contrast | all candidates·net gain | all candidates·Success rate after restoration | model-proposed candidates·net gain | model-proposed candidates·Success rate after restoration |",
          "|---|---|---|---|---|---|"]
    for tid in ("PC_7B", "P1", "P2", "H1", "PC_Llama", "H2", "H3"):
        t = res["tests"].get(tid)
        if t:
            L.append(f"| {tid} | {t['label']} | {f3(t['all_net'])} | {f3(t['all_post'])} | {f3(t['model_net'])} | "
                     f"{f3(t['model_post'])} |")
    if res["A"]:
        L += ["", "A interpretation（Registration）：" + "；".join(f"{k}：{v}" for k, v in res["A"].items())]
    L += ["", "trajectories with no model-proposed candidates：Difference by 0（both abstain from repair）、Net gain on horizontal value by 0、Post-repair success rate not included（Same 6.9a）。", "",
          "## B：Two failed pools × Two directions（by X Selection、Here Y use；Same thing 5 runs）", "",
          "| Orientation | 7B Ji!·all | 7B Ji!·model-proposed candidates | Llama Ji!·all | Llama Ji!·model-proposed candidates |", "|---|---|---|---|---|"]
    B = res["B"]
    for dn in ("l8→q7", "q7→l8"):
        L.append(f"| {dn.replace('l8', 'Llama').replace('q7', '7B')} | " +
                 " | ".join(f3(B.get(f"{p}|{dn}|{v}|net")) for p in ("7B", "Llama") for v in ("all", "model")) + " |")
    L += ["", "The difference between the two pools in the same direction（7B Ji! − Llama Ji!，Independence bootstrap）："]
    for dn in ("l8→q7", "q7→l8"):
        L.append(f"- {dn.replace('l8', 'Llama').replace('q7', '7B')}：all {f3(B.get(f'pooldiff|{dn}|all'))}；"
                 f"model-proposed candidates {f3(B.get(f'pooldiff|{dn}|model'))}")
    L += ["", "The difference between the two directions in the same pool（Llama→7B − 7B→Llama，Match）："]
    for p in ("7B", "Llama"):
        L.append(f"- {p} Ji!：all {f3(B.get(f'dirdiff|{p}|all'))}；model-proposed candidates {f3(B.get(f'dirdiff|{p}|model'))}")
    if "7B|q7→q14|all|net" in B:
        L += ["", f"description（Hes not here 6.12 List）：7B In the pool 7B Selection、Here 14B use：all {f3(B['7B|q7→q14|all|net'])}；"
                  f"model-proposed candidates {f3(B.get('7B|q7→q14|model|net'))}"]
    L += ["", "Under the goal of the post-repair success rate B See analysis_612.json。", "", "## C：Budget curve（Select the value of restoration on the deployed person）and SL", ""]
    for name in ("7B", "Llama"):
        c = res["C"].get(name)
        if not c:
            continue
        L += [f"**{name} Ji!，deployed executor {LAB[c['deployer']]}**（Average per trajectory {c['branches_per_traj']:.1f} Random branch；"
              f"Model Subset {c['branches_per_traj_model']:.1f} items）", "",
              "| candidate | Chooser | k | Number of repeats per trajectories | Cmcard sec | net gain | Success rate after restoration |", "|---|---|---|---:|---:|---|---|"]
        for r in c["rows"]:
            gs = f"{r['gpu_sec']:.0f}" if r["gpu_sec"] is not None else "—"
            L.append(f"| {'all' if r['var'] == 'all' else 'model'} | {LAB[r['selector']]} | {r['k']} | {r['rollouts']:.0f} | {gs} | "
                     f"{f3(r['net'])} | {f3(r['post'])} |")
        for o in REV["k_other"]:
            for var in ("all", "model"):
                if c.get(f"trigger_{o}_{var}"):
                    L.append(f"\nD %1 trigger statistics（{'all' if var == 'all' else 'model'}candidate）：7B 5 runs − {LAB[o]} "
                             f"{REV['k_other'][o]} runs = {f3(c[f'trigger_{o}_{var}'])}"
                             + ("（The trigger conditions for registration are for all candidates only）" if var == "model" else
                                f" → {'satisfied（lower bound > 0）' if c[f'trigger_{o}_{var}']['ci'][0] > 0 else 'not satisfied'}"))
        L += ["", "| SL（New pool，Method of freezing） | Number of repeats per trajectories | Relatively matched loss |", "|---|---:|---|"]
        for k, v in c["SL"].items():
            if isinstance(v, dict):
                L.append(f"| {k} | {v['budget']:.0f} | {f3(v['loss'])} |")
        L.append(f"\n Screening recall rate：planner {c['SL'].get('recall_SLpl', float('nan')):.2f}；flash {c['SL'].get('recall_SLfl', float('nan')):.2f}")
        L.append("")
    old = res["C"].get("SL_old_screened")
    if old:
        L += ["**SL On the old sifting sample**（stage B，30 records，out_method/method_alfw.json；Report with the new pool）", "",
              "| Modalities | budget | loss [90%] |", "|---|---:|---|"]
        for k, v in old.items():
            L.append(f"| {k} | {v['budget']:.0f} | {v['loss'][0]:+.3f} [{v['loss'][1]:+.3f}, {v['loss'][2]:+.3f}] |")
    return "\n".join(L) + "\n"


# ================================================================ D：preference-reversal check
# 6.12 D + Revision 1：Every pair（7B, X）Free Freeze In out_rev612/<X>/，It doesnt affect each other（Llama A couple can wait p610 freeze after the run completes）。
#   X = 14B：No longer subject to trigger conditions（Revision 1），Trigger statistics to report as usual；Add 15 Bar Retract trajectories。
#   X = Llama：Still in the original trigger condition（7B 5 runs − Llama 10 runs Bottom bounds > 0）；No control group。
def _best_bid(g, cands, exclude=()):
    idx = [i for i in range(len(g)) if cands[i][0] not in exclude]
    if not idx:
        return None
    m = max(g[i] for i in idx)
    return min(cands[i][0] for i in idx if g[i] >= m - 1e-12)


def _gain_full(A):
    full = DS._full(A["det"], A["rand"], A["S"].mean(axis=1) if len(A["rand"]) else [])
    return DS._gain(full, A["ia"], A["io"], "net")


def _rev_dir(pair, out_root=None):
    return Path(out_root or OUT["rev"]) / pair


def rev_freeze(pair, force=False, pool_dir=None, out_root=None):
    """Freezing of cases of a pair of executors（In comparison with）。do not modify an already frozen design。"""
    out_dir = _rev_dir(pair, out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "cases.json").exists() and DS.jload(out_dir / "cases.json").get("frozen"):
        log(f"[rev] {LAB[pair]} - Theyre frozen（{out_dir / 'cases.json'}），No change")
        return 0
    pool = load_pool("7B", pool_dir)
    if pool is None or "q7" not in pool["ex"]:
        log("[rev] 7B Pool or 7B continuations are incomplete")
        return 1
    if pair not in pool["ex"]:
        log(f"[rev] {LAB[pair]} at 7B The running on the pool is incomplete（{pool['incomplete'].get(pair, 'No file')}），Cant freeze；run after completion")
        return 1
    d = pool["d"]
    rows = run_core(pool, d, "q7", [pair], "net")
    ks = min(REV["k_self"], pool["ex"]["q7"]["n"] // 2)
    k2 = min(REV["k_other"][pair], pool["ex"][pair]["n"])
    b = DS.boot([r[f"self_k{ks}"] - r[f"{pair}_k{k2}"] for r in rows])
    trig = dict(b, stat=f"7B {ks} runs − {LAB[pair]} {k2} runs（net gain）", met=bool(b["ci"][0] > 0))
    log(f"[rev] Trigger Statistics {trig['stat']}：{f3(b)} → {'satisfied' if trig['met'] else 'not satisfied'}")
    unconditional = pair in REV["unconditional"]
    go_ahead = unconditional or trig["met"] or force
    rec = {"pair": pair, "frozen": go_ahead, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "trigger": trig,
           "rule": "unconditional（6.12 Revision 1）" if unconditional else ("Force" if (force and not trig["met"]) else "Trigger Condition"),
           "forced": bool(force and not unconditional and not trig["met"]),
           "source_design_id": d["design_id"], "n": REV["n"], "batch": REV["batch"],
           "n_source": {n: pool["ex"][n]["n"] for n in pool["ex"]}, "cases": [], "control": []}
    if not go_ahead:
        DS.jdump(out_dir / "cases.json", rec)
        log(f"[rev] {LAB[pair]}：Trigger condition not met，do not freeze、Dont run。write results to {out_dir / 'cases.json'}")
        return 0
    A7, AO = arrays(pool, d, "q7"), arrays(pool, d, pair)
    need, agree = set(), []
    for t, a7, ao in zip(d["trajectories"], A7, AO):
        g7, go = _gain_full(a7), _gain_full(ao)
        a, b = _best_bid(g7, a7["cands"]), _best_bid(go, ao["cands"])
        if a == b:
            agree.append((t["traj_id"], a7, g7, go))
            continue
        ca = next(c for c in a7["cands"] if c[0] == a)
        cb = next(c for c in a7["cands"] if c[0] == b)
        i_a, i_b = a7["cands"].index(ca), a7["cands"].index(cb)
        rec["cases"].append({"traj_id": t["traj_id"], "a": a, "b": b, "oa": ca[1], "ob": cb[1],
                             "old": {"q7": [float(g7[i_a]), float(g7[i_b])], pair: [float(go[i_a]), float(go[i_b])]}})
        need |= {a, b, ca[1], cb[1]}
    log(f"[rev] {LAB[pair]}：{len(rec['cases'])} / {len(d['trajectories'])} The best candidates for the trajectories are different，Freezing as Case")
    if pair in REV["control"]:
        spec = REV["control"][pair]
        elig = [x for x in agree if len(x[1]["cands"]) >= 2]          # trajectories with only one candidate have no“Jia!”
        rng = np.random.default_rng(spec["seed"])
        pick = sorted(rng.choice(len(elig), size=min(spec["n"], len(elig)), replace=False).tolist()) if elig else []
        for j in pick:
            tid, a7, g7, go = elig[j]
            a = _best_bid(g7, a7["cands"])
            b = _best_bid(g7, a7["cands"], exclude=(a,))            # 7B The second best in all the samples（Column by Branch Number）
            ca = next(c for c in a7["cands"] if c[0] == a)
            cb = next(c for c in a7["cands"] if c[0] == b)
            i_a, i_b = a7["cands"].index(ca), a7["cands"].index(cb)
            rec["control"].append({"traj_id": tid, "a": a, "b": b, "oa": ca[1], "ob": cb[1],
                                   "old": {"q7": [float(g7[i_a]), float(g7[i_b])], pair: [float(go[i_a]), float(go[i_b])]}})
            need |= {a, b, ca[1], cb[1]}
        rec["control_rule"] = (f"The best candidate is the same、and at least 2 A candidate {len(elig)} trajectories（By Design Order）middle，"
                               f"np.random.default_rng({spec['seed']}).choice sample without replacement {len(pick)} records；"
                               "a = Best common candidate，b = 7B Second best candidate for all samples（Column by Branch Number First）")
        log(f"[rev] control：{len(rec['control'])} records（from {len(elig)} Draw in the same trajectories as the best candidate，seed {spec['seed']}）")
    br = {b: d["branches"][b] for b in sorted(need)}
    sub = {"kind": "rev612", "pair": pair, "branches": br, "trajectories": [], "source_design_id": d["design_id"]}
    sub["design_id"] = DS.fp({"src": d["design_id"], "pair": pair, "branches": sorted(br), "batch": REV["batch"]})
    rec["design_id"] = sub["design_id"]
    rec["executors"] = ["q7", pair]
    nst = sum(1 for x in br.values() if x["deterministic"] is None)
    rec["n_rollouts"] = {e: nst * REV["n"] for e in rec["executors"]}
    DS.jdump(out_dir / "design.json", sub)
    DS.jdump(out_dir / "cases.json", rec)
    log(f"[rev] freeze {LAB[pair]} - Yeah（{rec['rule']}）：{len(br)} branches（random {nst} items），Each executor {nst * REV['n']} Keep running")
    return 0


def rev_eval(pair, executor, out_root=None, n=None):
    out_dir = _rev_dir(pair, out_root)
    if not (out_dir / "cases.json").exists():
        sys.exit(f"[rev] {out_dir} Not yet cases.json（Run first rev-freeze --pair {pair}）")
    c = DS.jload(out_dir / "cases.json")
    if not c.get("frozen"):
        sys.exit(f"[rev] {LAB[pair]} Theyre not frozen（Do not run when trigger conditions are not met）")
    if executor not in c["executors"]:
        sys.exit(f"[rev] the executors in this pair are {c['executors']}，Does not include {executor}")
    DS.use_out(out_dir)
    d = DS.jload(out_dir / "design.json")
    n = n or c["n"]
    if not DS.STATE["mock"]:
        try:
            import selftest as grid_selftest
            grid_selftest.t0_manifest()
        except AssertionError as e:
            log(f"Code version check failed：{e}")
            return 1
        if not DS.wait_gpu("rev-eval"):
            return 1
    holder = {}

    def backend():
        if "b" not in holder:
            holder["b"] = DS.make_exec_backend(executor)
        return holder["b"]
    ctx = {"backend": backend, "stop_after": DS.STATE.get("stop_after")}
    path = out_dir / f"evals_{executor}.jsonl"
    cond = f"{executor}_T07"
    have = runner.load_results(path) if path.exists() else {}
    todo = sum(1 for b, x in d["branches"].items() if x["deterministic"] is None and (cond, c["batch"], b) not in have)
    log(f"[rev] {LAB[pair]} - Yeah · executor {LAB[executor]}：Stay {todo} branches × {n} runs")
    try:
        with SB.make_pool() as pl:
            ctx["pool"] = pl
            tm = SB.run_branches(ctx, d, cond, c["batch"], n, path, acfg.TEMPERATURE)
            if tm:
                f = out_dir / f"timing_{executor}.json"
                old = DS.jload(f) if f.exists() else {"runs": []}
                old["runs"].append({"cond": f"{cond}/{c['batch']}", "wall": tm["wall"], "policy_steps": tm["policy_steps"],
                                    "rollouts": todo * n, "gpu": DS.gpu_info(), "tp": int(os.environ.get("LADDER_TP", "1")),
                                    "time": time.strftime("%Y-%m-%d %H:%M")})
                DS.jdump(f, old)
    finally:
        DS.release_vllm(holder.get("b"))
    have = runner.load_results(path)
    left = [b for b, x in d["branches"].items() if x["deterministic"] is None and (cond, c["batch"], b) not in have]
    log(f"[rev] {LAB[pair]} - Yeah · {LAB[executor]}：" + ("All completed" if not left else f"Not much {len(left)} branches，The same order will be replaced by the same order"))
    return 0 if not left else 1


def _vals(d, ev, executor, batch, b):
    x = d["branches"][b]
    if x["deterministic"] is not None:
        return np.array([float(x["deterministic"])])
    return np.asarray(ev[(f"{executor}_T07", batch, b)]["results"], float)


def _rev_rows(items, V, o, rng):
    rows = []
    for cs in items:
        def delta(e, obj, draw=None, cs=cs):
            m = (lambda b: V[e][b].mean()) if draw is None else draw
            if obj == "net":
                return (m(cs["a"]) - m(cs["oa"])) - (m(cs["b"]) - m(cs["ob"]))
            return m(cs["a"]) - m(cs["b"])
        r = {"traj_id": cs["traj_id"], "a": cs["a"], "b": cs["b"]}
        for obj in ("net", "post"):
            r[f"d7_{obj}"], r[f"do_{obj}"] = float(delta("q7", obj)), float(delta(o, obj))
            r[f"I_{obj}"] = r[f"d7_{obj}"] - r[f"do_{obj}"]
        # per trajectory：Intra-branch samples bootstrap，The difference between the two executors（net gain）One 90% interval
        for e, key in (("q7", "d7"), (o, "do")):
            bs = []
            for _ in range(N_BOOT_CASE):
                cache = {}

                def draw(b, e=e, cache=cache):
                    if b not in cache:
                        v = V[e][b]
                        cache[b] = v[rng.integers(0, len(v), len(v))].mean()
                    return cache[b]
                bs.append(delta(e, "net", draw))
            r[f"{key}_ci"] = [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))]
        rows.append(r)
    return rows


def _rev_summary(rows):
    out = {"n": len(rows), "rows": rows}
    for obj in ("net", "post"):
        for k in ("I", "d7", "do"):
            out[f"{k}_{obj}"] = DS.boot([r[f"{k}_{obj}"] for r in rows])
    sig_pos = lambda ci: ci[0] > 0
    sig_neg = lambda ci: ci[1] < 0
    out["n_point_opposite"] = int(sum(r["d7_net"] > 0 and r["do_net"] < 0 for r in rows))     # 7B Preferences a、X Preferences b
    out["n_sig_opposite"] = int(sum(sig_pos(r["d7_ci"]) and sig_neg(r["do_ci"]) for r in rows))
    out["n_point_both_a"] = int(sum(r["d7_net"] > 0 and r["do_net"] > 0 for r in rows))       # Both executors prefer it a
    out["n_sig_both_a"] = int(sum(sig_pos(r["d7_ci"]) and sig_pos(r["do_ci"]) for r in rows))
    out["n_d7_pos"] = int(sum(r["d7_net"] > 0 for r in rows))                                  # 7B Still prefer a（Organisation）
    out["n_do_neg"] = int(sum(r["do_net"] < 0 for r in rows))                                  # X Still prefer b（Organisation）
    return out


def rev_analyze(out_root=None, seed=0):
    root = Path(out_root or OUT["rev"])
    res, L = {"pairs": {}}, ["# 6.12 D：preference-reversal check（independent continuations）", ""]
    for pair in REV["k_other"]:
        out_dir = root / pair
        if not (out_dir / "cases.json").exists():
            continue
        c = DS.jload(out_dir / "cases.json")
        trig_line = f"Trigger Statistics {c['trigger']['stat']}：{f3(c['trigger'])} → {'satisfied' if c['trigger']['met'] else 'not satisfied'}"
        if not c.get("frozen"):
            L += [f"## 7B vs {LAB[pair]}", "", trig_line + "；Not frozen by registration、Dont run。", ""]
            continue
        d = DS.jload(out_dir / "design.json")
        evs = {e: runner.load_results(out_dir / f"evals_{e}.jsonl") for e in c["executors"]
               if (out_dir / f"evals_{e}.jsonl").exists()}
        if set(evs) != set(c["executors"]):
            log(f"[rev] {LAB[pair]} - Yeah：Missing {sorted(set(c['executors']) - set(evs))} A new rerun，skip")
            L += [f"## 7B vs {LAB[pair]}", "", trig_line, "", "new continuations remain incomplete，Not analysed。", ""]
            continue
        try:
            V = {e: {b: _vals(d, evs[e], e, c["batch"], b) for b in d["branches"]} for e in c["executors"]}
        except KeyError:
            log(f"[rev] {LAB[pair]} - Yeah：The new run is incomplete，skip（rev-eval complete the missing records and rerun）")
            L += [f"## 7B vs {LAB[pair]}", "", trig_line, "", "new continuations remain incomplete，Not analysed。", ""]
            continue
        rng = np.random.default_rng(seed)
        cases = _rev_summary(_rev_rows(c["cases"], V, pair, rng))
        ctrl = _rev_summary(_rev_rows(c.get("control", []), V, pair, rng)) if c.get("control") else None
        cases["verdict"] = ("Average Interaction Bottom > 0：In these cases, the two executors do prefer opposite repairs（not estimation noise）；establishes existence only，cannot estimate prevalence"
                            if cases["n"] and cases["I_net"]["ci"][0] > 0 else "Average interactive interval content 0（Or no case）：New sample not supported“preferences are genuinely opposite”")
        res["pairs"][pair] = {"cases": cases, "control": ctrl, "rule": c["rule"], "trigger": c["trigger"]}
        pct = lambda k, v: f"{v[k]} / {v['n']}（{v[k] / v['n']:.0%}）" if v and v["n"] else "—"
        L += [f"## 7B vs {LAB[pair]}", "",
              f"Freeze on {c['frozen_at']}（{c['rule']}）；New Batch {c['batch']}，Each branch {c['n']} runs；New samples only。{trig_line}", "",
              "Δ_e = [R_e(a) − R_e(oa)] − [R_e(b) − R_e(ob)]（net gain；o Original action for the position of the candidate）；Interactive I = Δ_7B − Δ_X。"
              "“Visible”= within-branch samples for this trajectory bootstrap（500 runs）of 90% the interval excludes 0。", "",
              "| Group | trajectories | Δ_7B | Δ_X | Interactive I（net gain） | Interactive I（Success rate after restoration） |", "|---|---:|---|---|---|---|",
              f"| preference-reversal cases（a = 7B best，b = {LAB[pair]} best） | {cases['n']} | {f3(cases['d7_net'])} | {f3(cases['do_net'])} | "
              f"{f3(cases['I_net'])} | {f3(cases['I_post'])} |"]
        if ctrl:
            L.append(f"| control（a = shared optimum，b = 7B Jia!） | {ctrl['n']} | {f3(ctrl['d7_net'])} | {f3(ctrl['do_net'])} | "
                     f"{f3(ctrl['I_net'])} | {f3(ctrl['I_post'])} |")
        L += ["", "| Read | preference-reversal cases | control |", "|---|---|---|",
              f"| 7B Still prefer a（A point estimate） | {pct('n_d7_pos', cases)} | {pct('n_d7_pos', ctrl)} |",
              f"| {LAB[pair]} Preferences b（A point estimate） | {pct('n_do_neg', cases)} | — |",
              f"| the two point estimates have opposite signs（7B Preferences a、{LAB[pair]} Preferences b） | {pct('n_point_opposite', cases)} | {pct('n_point_opposite', ctrl)} |",
              f"| **both preferences are significant and opposite** | {pct('n_sig_opposite', cases)} | {pct('n_sig_opposite', ctrl)} |",
              f"| both executors prefer a（A point estimate） | {pct('n_point_both_a', cases)} | {pct('n_point_both_a', ctrl)} |",
              f"| **Both sides are visible and in the same direction（Preferences a）** | {pct('n_sig_both_a', cases)} | {pct('n_sig_both_a', ctrl)} |",
              "", f"Reading：{cases['verdict']}。",
              "Checkout“How much can be recovered from the new sample?”Baseline：In reverse cases“both preferences are significant and opposite”The ratio is in contrast“Both sides are visible and in the same direction”Scales；"
              "Cases selected on the basis of available data，cannot estimate reversal prevalence among all failures。", ""]
        if c.get("control_rule"):
            L += [f"The rule of extraction for comparison：{c['control_rule']}。", ""]
    DS.jdump(root / "analysis_rev.json", res)
    (root / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"[rev] -> {root / 'report.md'}")
    return 0


# ================================================================ self-test
def _synth_pool(dirp, ntraj, probs_fn, nexec, seed, cand_meta):
    """Synthetic pool：per trajectory 1 Organisation；cand_meta = [(name, Metadata)]；probs_fn(i) -> {executor: {Name or name 'orig': success rate}}。"""
    rng = np.random.default_rng(seed)
    dirp = Path(dirp)
    dirp.mkdir(parents=True, exist_ok=True)
    trajs, br = [], {}
    recs = {e: [] for e in nexec}
    for i in range(ntraj):
        tid = f"s{i:03d}"
        oid = f"{tid}|t5|orig"
        alts = [f"{tid}|t5|{nm}" for nm, _ in cand_meta]
        trajs.append({"traj_id": tid, "gamefile": "mock://0", "task_type": "x", "T": 50,
                      "positions": [{"t": 5, "orig": oid, "alts": alts, "L": 3, "slack": 10}]})
        base = {"traj_id": tid, "t": 5, "gamefile": "mock://0", "prefix": [], "deterministic": None, "oracle_L": 3,
                "oracle_F": 1.0, "oracle_slack": 10.0, "oracle_timeout": False}
        br[oid] = dict(base, branch_id=oid, action="orig:x", expert=False)
        for (nm, meta), bid in zip(cand_meta, alts):
            br[bid] = dict(base, branch_id=bid, action=nm, expert=meta.get("planner", False),
                           proposers=meta.get("proposers", []), planner=meta.get("planner", False),
                           random=meta.get("random", False), flash_rank=meta.get("flash_rank"))
        pr = probs_fn(i)
        for e, n in nexec.items():
            for bid in [oid] + alts:
                p = pr[e]["orig" if bid == oid else bid.split("|")[-1]]
                res = (rng.random(n) < p).tolist()
                recs[e].append({"condition": f"{e}_T07", "batch": BATCH, "branch_id": bid, "n": n, "results": res,
                                "successes": int(sum(res)), "steps": [10] * n, "invalid": [0] * n})
                recs[e].append({"condition": f"{e}_greedy", "batch": BATCH, "branch_id": bid, "n": 1,
                                "results": [bool(p > 0.5)], "successes": int(p > 0.5), "steps": [10], "invalid": [0]})
    DS.jdump(dirp / "design.json", {"kind": "synth", "trajectories": trajs, "branches": br, "design_id": f"synth{seed}"})
    for e, rr in recs.items():
        with open(dirp / f"evals_{e}.jsonl", "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r) + "\n")
    return dirp


def _synth_rev_evals(out_dir, probs_fn, execs, n, seed):
    """Generate samples of new batches for frozen branches at real success rates（Replace GPU continuation，For known answers only）。"""
    rng = np.random.default_rng(seed)
    c = DS.jload(out_dir / "cases.json")
    d = DS.jload(out_dir / "design.json")
    for e in execs:
        with open(out_dir / f"evals_{e}.jsonl", "w", encoding="utf-8") as f:
            for bid, x in d["branches"].items():
                i = int(x["traj_id"][1:])
                p = probs_fn(i)[e]["orig" if bid.endswith("|orig") else bid.split("|")[-1]]
                res = (rng.random(n) < p).tolist()
                f.write(json.dumps({"condition": f"{e}_T07", "batch": c["batch"], "branch_id": bid, "n": n,
                                    "results": res}) + "\n")


def selftest():
    import shutil
    ok = [True]

    def check(cnd, msg):
        print(("  ✓ " if cnd else "  ✗ ") + msg, flush=True)
        ok[0] = ok[0] and bool(cnd)
    tmp = Path(tempfile.mkdtemp(prefix="ds612_"))
    saved = json.loads(json.dumps(DS.CFG))
    saved_extra = json.loads(json.dumps(DS.EXTRA_EXEC))
    saved_out = dict(DS.PROFILE_OUT)
    try:
        # ---------------- 1. Simulation environment access：Two pools、Three executors，ds_alfw its own analysis + This is the script
        print("[ds612 selftest] Simulation Data：7B Ji!（7B / 14B / Llama）and Llama Ji!（Llama / 7B）")
        DS.STATE["mock"] = True
        DS.CFG.update(N_COLLECT=40, N_TRAJ=6, N_EXEC={"q7": 10, "q14": 20})
        DS.EXTRA_EXEC["q7"] = {"l8": 10}
        acfg.B_MAX_ACTIVE = 60
        A, C = tmp / "A", tmp / "C"
        DS.PROFILE_OUT.update(q7=A, l8=C)
        DS.use_out(A)
        assert DS.stage_collect() == 0 and DS.stage_positions() == 0
        assert all(DS.stage_propose(n) == 0 for n in DS.CFG["PROPOSERS"]) and DS.stage_design() == 0
        assert all(DS.stage_eval(n) == 0 for n in ("q7", "q14", "l8")) and DS.stage_analyze() == 0
        DS.CFG.clear()
        DS.CFG.update(saved)
        DS.apply_profile("l8")
        DS.CFG.update(N_COLLECT_MAX=200, COLLECT_CHUNK=15, N_FAIL_TARGET=8, N_TRAJ=5, N_EXEC={"l8": 10, "q7": 10})
        DS.use_out(C)
        assert DS.stage_collect() == 0 and DS.stage_positions() == 0
        assert all(DS.stage_propose(n) == 0 for n in DS.CFG["PROPOSERS"]) and DS.stage_design() == 0
        assert DS.stage_eval("l8") == 0 and DS.stage_eval("q7") == 0 and DS.stage_analyze() == 0
        res = analyze({"7B": A, "Llama": C}, tmp / "out612")
        check(all(k in res["tests"] for k in ("P1", "P2", "H1", "H2", "H3", "PC_7B", "PC_Llama")),
              "Recounted registration is complete（P1、P2、H1、H2、H3 Compared to two positives）")
        check(res["check"] and all(abs(v["diff"]) < 1e-12 for v in res["check"].values()),
              f"recomputed P1–H3 and ds_alfw of analysis.json Same one by one（{ {k: round(v['diff'], 15) for k, v in res['check'].items()} }）")
        check(all(res["tests"][t].get("model_net") for t in ("P2", "H1", "H2")), "A：Model Sub-Aggregation P2、H1、H2")
        need_b = [f"{p}|{dn}|{v}|net" for p in ("7B", "Llama") for dn in ("l8→q7", "q7→l8") for v in ("all", "model")]
        check(all(res["B"].get(k) for k in need_b) and "pooldiff|l8→q7|all" in res["B"] and "dirdiff|7B|all" in res["B"],
              "B：Two pools × Two directions × Two candidate groups，The difference between the pool and the direction")
        check(abs(res["B"]["7B|l8→q7|all|net"]["mean"] - res["tests"]["H1"]["all_net"]["mean"]) < 1e-12 and
              abs(res["B"]["Llama|q7→l8|all|net"]["mean"] - res["tests"]["H2"]["all_net"]["mean"]) < 1e-12,
              "B Diagonal with H1、H2 identical（The orientation label is not reversed）")
        c7 = res["C"]["7B"]
        check(any(r["selector"] == "q14" and r["k"] == 20 for r in c7["rows"]) and c7.get("trigger_q14_all")
              and any(r["gpu_sec"] for r in c7["rows"]), "C：Budget curve 14B k = 20、Cmcard Second Conversion and D %1 trigger statistics")
        check((tmp / "out612" / "report_612.md").exists(), "report")
        # ---------------- 2. D complete the workflow in a mock environment（14B A pair of unconditionals + control；Llama A pair of triggers）
        print("[ds612 selftest] D：freeze → continuation（New Batch，Including Interruption）→ Analysis（mock environment）")
        R = tmp / "rev"
        rc = rev_freeze("q14", pool_dir=A, out_root=R)
        cs = DS.jload(R / "q14" / "cases.json")
        check(rc == 0 and cs["frozen"] and cs["rule"].startswith("unconditional") and not cs["forced"],
              f"14B A pair of unconditional freezes（Trigger Condition {'satisfied' if cs['trigger']['met'] else 'not satisfied'}，As usual）")
        check(rev_freeze("q14", pool_dir=A, out_root=R) == 0 and
              DS.jload(R / "q14" / "cases.json")["frozen_at"] == cs["frozen_at"], "The frozen ones wont be rewritten")
        pool = load_pool("7B", A)
        A7, A14 = arrays(pool, pool["d"], "q7"), arrays(pool, pool["d"], "q14")
        by = {t["traj_id"]: (x, y) for t, x, y in zip(pool["d"]["trajectories"], A7, A14)}
        ok_ctrl = True
        for it in cs["control"]:
            x, y = by[it["traj_id"]]
            g7, g14 = _gain_full(x), _gain_full(y)
            ok_ctrl &= (_best_bid(g7, x["cands"]) == it["a"] == _best_bid(g14, y["cands"])
                        and _best_bid(g7, x["cands"], exclude=(it["a"],)) == it["b"] != it["a"])
        case_ids = {x["traj_id"] for x in cs["cases"]}
        check(ok_ctrl and not case_ids & {x["traj_id"] for x in cs["control"]} and
              len(cs["control"]) == min(15, sum(1 for t in pool["d"]["trajectories"] if t["traj_id"] not in case_ids
                                                and len(by[t["traj_id"]][0]["cands"]) >= 2)),
              f"control：{len(cs['control'])} records，a = Both sides are the best、b = 7B Jia!，No overlap with case")
        rc = rev_freeze("l8", pool_dir=A, out_root=R)
        cl = DS.jload(R / "l8" / "cases.json")
        check(rc == 0 and cl["frozen"] == cl["trigger"]["met"] and not cl["control"],
              f"Llama A pair of triggers（{'satisfied → freeze' if cl['trigger']['met'] else 'not satisfied → do not freeze'}），no controls")
        DS.STATE["stop_after"] = 3
        try:
            rev_eval("q14", "q7", out_root=R, n=4)
        except KeyboardInterrupt:
            pass
        DS.STATE["stop_after"] = None
        check(rev_eval("q14", "q7", out_root=R, n=4) == 0 and rev_eval("q14", "q14", out_root=R, n=4) == 0,
              "14B - Yeah：7B and 14B continuation（7B Completion after midway break）")
        if cl["frozen"]:
            check(rev_eval("l8", "q7", out_root=R, n=4) == 0 and rev_eval("l8", "l8", out_root=R, n=4) == 0, "Llama A pair of reruns")
        ev = runner.load_results(R / "q14" / "evals_q7.jsonl")
        check(ev and all(k[1] == REV["batch"] for k in ev), "New Rerun with New batch（and 6.9 of D Batch Feeds Different）")
        check(rev_analyze(R) == 0 and (R / "report.md").exists() and "q14" in DS.jload(R / "analysis_rev.json")["pairs"],
              "D Analysis and reporting")
        # ---------------- 3. known-answer test
        print("[ds612 selftest] known-answer test（Synthetic Data）")
        good = known_answers(tmp)
        check(good, "A、B、D The answer is known")
    finally:
        DS.CFG.clear()
        DS.CFG.update(saved)
        DS.EXTRA_EXEC.clear()
        DS.EXTRA_EXEC.update(saved_extra)
        DS.PROFILE_OUT.clear()
        DS.PROFILE_OUT.update(saved_out)
        DS.PROFILE["name"] = "q7"
        DS.STATE["mock"] = False
        if os.environ.get("DS_KEEP"):
            print(f"[ds612 selftest] Keep temporary directory {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ds612 selftest] {'all passed' if ok[0] else 'failures detected'}")
    return 0 if ok[0] else 1


def known_answers(tmp):
    good = True
    P = ("P", {"planner": True})
    M1, M2 = ("M1", {"proposers": ["q7"]}), ("M2", {"proposers": ["q14"]})
    # A：The executor disagrees only on model candidates（Planners are the best on both sides）-> all candidates ≈ 0，Models are obvious subsets > 0
    pr = lambda i: {"q7": {"orig": 0.05, "P": 0.98, "M1": 0.5, "M2": 0.1},
                    "q14": {"orig": 0.05, "P": 0.98, "M1": 0.1, "M2": 0.5}}
    _synth_pool(tmp / "kA", 200, pr, {"q7": 10, "q14": 20}, 1, [P, M1, M2])
    r = analyze({"7B": tmp / "kA", "Llama": tmp / "none"}, tmp / "kA_out")
    full, sub = r["tests"]["P2"]["all_net"], r["tests"]["P2"]["model_net"]
    ok1 = full["mean"] < 0.05 and sub["ci"][0] > 0.2
    print(f"    A：all candidates P2 {f3(full)}；model-proposed candidates P2 {f3(sub)} -> {r['A'].get('P2')}")
    good &= ok1
    # B：Orientation Label。7B Ji-li! Llama almost indifferent between the two candidates（Shade B A little），7B Visible bias A：
    #    by Llama Select to 7B using this incurs large loss；by 7B Select to Llama using this incurs little loss
    prB = lambda i: {"q7": {"orig": 0.0, "M1": 0.7, "M2": 0.1}, "l8": {"orig": 0.0, "M1": 0.5, "M2": 0.55}}
    _synth_pool(tmp / "kB", 200, prB, {"q7": 10, "l8": 10}, 2, [M1, M2])
    r = analyze({"7B": tmp / "kB", "Llama": tmp / "none"}, tmp / "kB_out")
    big, small = r["B"]["7B|l8→q7|all|net"], r["B"]["7B|q7→l8|all|net"]
    ok2 = big["ci"][0] > 0.15 and small["mean"] < 0.10
    print(f"    B：by Llama Select to 7B {f3(big)}；by 7B Select to Llama {f3(small)}")
    good &= ok2
    # D：genuine reversal（7B: a .7 b .2；14B: a .2 b .7）-> Interactive ≈ 1；No difference（Both sides .5）-> Interactive ≈ 0（The selection bias was eliminated by the new sample）
    for tag, prD, want in (("rev", lambda i: {"q7": {"orig": 0.0, "M1": 0.7, "M2": 0.2},
                                              "q14": {"orig": 0.0, "M1": 0.2, "M2": 0.7}}, "pos"),
                           ("null", lambda i: {"q7": {"orig": 0.0, "M1": 0.5, "M2": 0.5},
                                               "q14": {"orig": 0.0, "M1": 0.5, "M2": 0.5}}, "zero")):
        pdir = _synth_pool(tmp / f"kD_{tag}", 120, prD, {"q7": 10, "q14": 20}, 3, [M1, M2])
        R = tmp / f"kD_{tag}_rev"
        rev_freeze("q14", pool_dir=pdir, out_root=R)
        _synth_rev_evals(R / "q14", prD, ["q7", "q14"], REV["n"], 4)
        rev_analyze(R)
        v = DS.jload(R / "analysis_rev.json")["pairs"]["q14"]["cases"]
        okD = (v["I_net"]["ci"][0] > 0.8) if want == "pos" else (v["I_net"]["ci"][0] < 0 < v["I_net"]["ci"][1]
                                                                 and abs(v["I_net"]["mean"]) < 0.1)
        print(f"    D（{tag}）：{v['n']} One case，Interactive {f3(v['I_net'])}，both preferences are significant and opposite {v['n_sig_opposite']}")
        good &= okD
    # D control：Both enforcers clearly preferred it M1、Jia! M2 -> control 15 records，b = M2，Most of them have a strong preference for both sides a；the same seed selects the same trajectories
    M3 = ("M3", {"proposers": ["qflash"]})
    prC = lambda i: {"q7": {"orig": 0.0, "M1": 0.8, "M2": 0.4, "M3": 0.05},
                     "q14": {"orig": 0.0, "M1": 0.85, "M2": 0.4, "M3": 0.05}}
    pdir = _synth_pool(tmp / "kC", 80, prC, {"q7": 10, "q14": 20}, 5, [M1, M2, M3])
    picks = []
    for k in (1, 2):
        R = tmp / f"kC_rev{k}"
        rev_freeze("q14", pool_dir=pdir, out_root=R)
        picks.append([x["traj_id"] for x in DS.jload(R / "q14" / "cases.json")["control"]])
    R = tmp / "kC_rev1"
    c = DS.jload(R / "q14" / "cases.json")
    _synth_rev_evals(R / "q14", prC, ["q7", "q14"], REV["n"], 6)
    rev_analyze(R)
    v = DS.jload(R / "analysis_rev.json")["pairs"]["q14"]["control"]
    n_m2 = sum(x["b"].endswith("|M2") for x in c["control"])
    okC = (len(c["control"]) == 15 and picks[0] == picks[1] and n_m2 >= 13 and v["n_sig_both_a"] >= 13
           and v["n_sig_opposite"] == 0)
    print(f"    D control：{len(c['control'])} records（b = M2 of {n_m2} records），Both sides have a strong preference a {v['n_sig_both_a']}，"
          f"both preferences are significant and opposite {v['n_sig_opposite']}；Two freezes on the same batch：{picks[0] == picks[1]}")
    good &= okC
    return bool(good)


# ================================================================ The entrance
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyze", "rev-freeze", "rev-eval", "rev-analyze", "selftest"])
    ap.add_argument("--pair", choices=list(REV["k_other"]), help="rev-freeze / rev-eval：Which one?（7B and q14 or l8）")
    ap.add_argument("--executor", choices=list(EXECS), help="rev-eval：Which executor is this run?")
    ap.add_argument("--force", action="store_true", help="rev-freeze --pair l8：If the trigger is not met, the trigger is frozen（Use only when user decides）")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    DS.STATE["mock"] = a.mock
    if a.cmd == "analyze":
        analyze()
        return 0
    if a.cmd in ("rev-freeze", "rev-eval") and not a.pair:
        sys.exit(f"{a.cmd} Yes --pair q14 or --pair l8")
    if a.cmd == "rev-freeze":
        return rev_freeze(a.pair, force=a.force)
    if a.cmd == "rev-eval":
        if not a.executor:
            sys.exit("rev-eval Yes --executor（q7，Or the other executor of this pair）")
        return rev_eval(a.pair, a.executor)
    if a.cmd == "rev-analyze":
        return rev_analyze()
    return 0


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != acfg.HASH_SEED:
        os.environ["PYTHONHASHSEED"] = acfg.HASH_SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.exit(main())
