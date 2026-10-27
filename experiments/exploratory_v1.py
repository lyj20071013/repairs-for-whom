"""EXPLORATION ANALYSIS（After，Non-pre-registration）：v1 results（design 42ea91e5…）per-trajectory breakdown。

    python exploratory_v1.py [Results Directory，default out]

content：Net gain on deployment conditions for recovery from selected sources on a trajectory-by-trajectories basis；Consistency probability when randomly breaking parallels；
oracle Loss by“position / action”Split；slack Break the parallels oracle；The final solventable location rule；
CAR inspired by“At the end of the night”Rules。Only a cluster re-sampling of trajectories，Not included V Batch sample noise。
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis
from runner import load_results
D = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "out")
d = json.load(open(f"{D}/design.json")); R = load_results(f"{D}/rollouts.jsonl")
def P(cond, batch, m=None):
    c = analysis.load_counts(d, R, cond, batch, m); return {b: analysis._p(kn) for b, kn in c.items()}
PV = P("deploy_T07", "V")
srcs = {"matched": P("deploy_T07","S"), "T03": P("llm_T03","S"), "greedy": P("llm_greedy","S"),
        "oracle": P("oracle","S"), "noisy": P("noisy_oracle_e30","S"), "S5": P("deploy_T07","S",5)}
# greedy determinism
dis = sum(1 for (c,b,bid),r in R.items() if c=="llm_greedy" and len(set(r["results"]))>1)
print("greedy: branches where 2 samples disagree:", dis, "/ 440")
rows = []
for t in d["trajectories"]:
    cands = analysis.candidates(t)
    rv = analysis.scores(cands, PV, "net")
    out = {"traj": t["traj_id"], "fail": t["fail_reason"], "T": t["T"], "n": len(cands), "maxV": rv.max()}
    sel = {}
    for k, Pk in srcs.items():
        s = analysis.scores(cands, Pk, "net")
        mask = np.abs(s - s.max()) <= 1e-12
        sel[k] = set(np.flatnonzero(mask))
        out[k] = rv[mask].mean()
    for k in srcs:
        A, B = sel[k], sel["matched"]
        out["pagree_"+k] = len(A & B) / (len(A) * len(B))   # Probability to select the same candidate when each side randomly breaks the parallels
        out["nt_"+k] = len(A)
    rows.append(out)
print(f"\n{'traj':8s} {'fail':20s} T  maxV   matched  T03    greedy oracle noisy  S5    | ties m/g/o")
for o in rows:
    print(f"{o['traj']:8s} {o['fail']:20s} {o['T']:2d} {o['maxV']:+.2f}  {o['matched']:+.3f} {o['T03']:+.3f} {o['greedy']:+.3f} {o['oracle']:+.3f} {o['noisy']:+.3f} {o['S5']:+.3f} | {o['nt_matched']}/{o['nt_greedy']}/{o['nt_oracle']}")
for k in srcs:
    loss = np.array([o["matched"] - o[k] for o in rows])
    print(f"{k:8s} mean loss {loss.mean():+.3f}; trajs with loss>0.1: {int((loss>0.1).sum())}; top-3 share of total loss {np.sort(loss)[::-1][:3].sum()/max(loss.sum(),1e-9):.2f}; P(same cand | random tie-break) {np.mean([o['pagree_'+k] for o in rows]):.2f}")

print("\n" + "=" * 100)
PV, PS, PO, PG = P("deploy_T07","V"), P("deploy_T07","S"), P("oracle","S"), P("llm_greedy","S")
br = d["branches"]
br = d["branches"]
for t in d["trajectories"]:
    cands = analysis.candidates(t)
    rv = analysis.scores(cands, PV, "net")
    if rv.max() <= 0.1: continue
    last_t = max(p["t"] for p in t["positions"])
    print(f"\n== {t['traj_id']} {t['fail_reason']} T={t['T']} positions={[p['t'] for p in t['positions']]} (Last solventable position {last_t})")
    for p in t["positions"]:
        o = p["orig"]
        fa = p["factual_action"]; fa = fa["tool"] + (":" + fa["arg"] if fa.get("arg") else "")
        alts = sorted(p["alts"], key=lambda a: -(PV[a] - PV[o]))
        s = "  ".join(f"{a.split('|')[-1]}:{PV[a]-PV[o]:+.2f}{'*' if PO[a]-PO[o]>0 else ''}" for a in alts)
        print(f"  t={p['t']:2d} Original={fa:10s} p_orig_V={PV[o]:.2f} F(orig)={int(PO[o])} | {s}")
    for name, Px in (("matched", PS), ("oracle", PO), ("greedy", PG)):
        sc = analysis.scores(cands, Px, "net"); m = np.abs(sc - sc.max()) <= 1e-12
        chosen = [cands[j] for j in np.flatnonzero(m)]
        pos = sorted({c[2] for c in chosen})
        print(f"  {name:8s} Selected {len(chosen)} items，position {pos}，Average net gain under deployment {rv[m].mean():+.3f}")

print("\n" + "=" * 100)
PV, PS, PO, PG, P3 = P("deploy_T07","V"), P("deploy_T07","S"), P("oracle","S"), P("llm_greedy","S"), P("llm_T03","S")
br = d["branches"]
def slack(b):
    s = br[b]["snapshot"]; dist = br[b]["oracle_distance"]
    if s["done"]: return 100 if s["success"] else -100
    if dist is None or dist > s["max_steps"] - s["steps"]: return -100
    return s["max_steps"] - s["steps"] - dist
def pick(scores):
    s = np.asarray(scores, float); return np.abs(s - s.max()) <= 1e-12
sel = {}
rows = []
for t in d["trajectories"]:
    c = analysis.candidates(t); rv = analysis.scores(c, PV, "net")
    last = max(p["t"] for p in t["positions"])
    S = analysis.scores(c, PS, "net"); O = analysis.scores(c, PO, "net")
    out = {"matched": rv[pick(S)].mean(), "oracle": rv[pick(O)].mean()}
    # oracle + slack Break Parallels（Dictionary order：first ΔF，then Δslack）
    ds = np.array([slack(a) - slack(o) for a, o, _ in c], float)
    lex = O * 1000 + ds
    out["oracle_slack_tiebreak"] = rv[pick(lex)].mean()
    # Pure slack Progress（How much closer to success?）
    out["slack_only"] = rv[pick(ds)].mean()
    # Staple position to final solver position，Movement impact deployment conditions S Selection
    at_last = np.array([ti == last for _, _, ti in c])
    S2 = np.where(at_last, S, -9); out["last_pos+deployS_action"] = rv[pick(S2)].mean()
    # Position to be deployed S Resolves（Take S Best position），Actions in this position by oracle Selection（Parallel Average）
    pos_m = {c[j][2] for j in np.flatnonzero(pick(S))}
    O2 = np.where([ti in pos_m for _, _, ti in c], O, -9); out["deployS_pos+oracle_action"] = rv[pick(O2)].mean()
    out["informative"] = rv.max() > 0.1
    rows.append(out)
keys = [k for k in rows[0] if k not in ("informative",)]
rng = np.random.default_rng(0)
print(f"{'Selection Rule':32s} Average net gain  Relative matching loss  [trajectories Cluster 90% interval]  only 8 Loss on a signal trail")
for k in keys:
    loss = np.array([r["matched"] - r[k] for r in rows])
    boot = [loss[rng.integers(0, len(loss), len(loss))].mean() for _ in range(4000)]
    inf = np.array([r["matched"] - r[k] for r in rows if r["informative"]])
    print(f"{k:32s} {np.mean([r[k] for r in rows]):+.3f}    {loss.mean():+.3f}      [{np.quantile(boot,.05):+.3f}, {np.quantile(boot,.95):+.3f}]     {inf.mean():+.3f}")

# —— CAR inspired by“At the end of the night”Rules（Use of deployment conditions S Batch，One-sided two-rater z Test α=0.05）——
import math
def z_pos(a, o):
    ka, na = R[("deploy_T07","S",a)]["successes"] if br[a]["deterministic"] is None else None, 50
    return None
def p_and_n(b):
    if br[b]["deterministic"] is not None: return br[b]["deterministic"], None
    r = R[("deploy_T07","S",b)]; return r["successes"]/r["n"], r["n"]
def significant(a, o):
    pa, na = p_and_n(a); po, no = p_and_n(o)
    if pa <= po: return False
    var = (pa*(1-pa)/na if na else 0) + (po*(1-po)/no if no else 0)
    if var == 0: return pa > po
    return (pa - po) / math.sqrt(var) > 1.645
poc_vals, poc_loss, inf_loss = [], [], []
for t, r in zip(d["trajectories"], rows):
    c = analysis.candidates(t); rv = analysis.scores(c, PV, "net"); S = analysis.scores(c, PS, "net")
    sig_pos = sorted({ti for (a, o, ti) in c if significant(a, o)})
    if sig_pos:
        latest = sig_pos[-1]
        S2 = np.where([ti == latest for _, _, ti in c], S, -9); v = rv[pick(S2)].mean()
    else:
        v = rv[pick(S)].mean()   # Backup matching options when no significant location is available
    poc_vals.append(v); poc_loss.append(r["matched"] - v)
    if r["informative"]: inf_loss.append(r["matched"] - v)
poc_loss = np.array(poc_loss)
boot = [poc_loss[rng.integers(0, 20, 20)].mean() for _ in range(4000)]
print(f"{'CAR Inspired：At the end of the night+Service S Best of all':32s} {np.mean(poc_vals):+.3f}    {poc_loss.mean():+.3f}      [{np.quantile(boot,.05):+.3f}, {np.quantile(boot,.95):+.3f}]     {np.mean(inf_loss):+.3f}")
