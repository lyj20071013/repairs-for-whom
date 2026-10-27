"""Frozen paper kernel; changes require a new protocol version. See docs/PROTOCOL.md."""
import numpy as np
N_PERM = 50
EXEC_CODE = {"q7": 1, "q14": 2, "l8": 3}

def _full(det, rand, vals):
    v = np.nan_to_num(det).copy()
    v[rand] = vals
    return v

def _gain(v, ia, io, obj):
    return v[..., ia] - (v[..., io] if obj == "net" else 0.0)

def _val(mask, g):
    return float((mask * g).sum() / mask.sum())

def _best(score):
    score = np.asarray(score, float)
    return score >= np.nanmax(score) - 1e-12

def _has(m, p):
    return bool(m.get(p)) if p in ("planner", "random") else p in m.get("proposers", [])

def analyze_core(TD, others, branch_meta, obj="net", seed=0, n_perm=N_PERM, ks_self=(1, 2, 3, 5),
                 ks_other=(1, 2, 3, 5, 10, 20), proposers=("q7", "q14", "qflash")):
    """One-line statistics per trajectories（all are values of selected repairs under the deployed executor）。TD：Personnel deployed _traj_arrays；others：{executor: On the same list
    _traj_arrays}；branch_meta(trajectories Serial Numbers, branch) -> Metadata。Random Number Press（trajectory，executor）one independent stream each。"""
    others = others or {}
    srcs = list(proposers) + ["planner", "random"]
    rows = []
    for ti, A in enumerate(TD):
        rng = np.random.default_rng([seed, ti])
        ia, io, det, rand = A["ia"], A["io"], A["det"], A["rand"]
        nD = A["S"].shape[1]
        h = nD // 2
        fullD = _full(det, rand, A["S"].mean(axis=1) if len(rand) else [])
        gD = _gain(fullD, ia, io, obj)
        meta = [branch_meta(ti, a) for a, _, _ in A["cands"]]
        r = {}
        # ---- Deployers themselves k Subselection（Crossing）：The first half of each rehearsing（before k runs）、The second half，Reconcile
        perms = [np.argsort(rng.random(A["S"].shape), axis=1) for _ in range(n_perm)]

        def crossfit(k, sub=None):
            vals = []
            for pm in perms:
                Z = np.take_along_axis(A["S"], pm, axis=1)
                for a_, b_ in ((slice(0, h), slice(h, nD)), (slice(h, nD), slice(0, h))):
                    sel = _full(det, rand, Z[:, a_][:, :k].mean(axis=1))
                    evv = _full(det, rand, Z[:, b_].mean(axis=1))
                    sc = _gain(sel, ia, io, obj)
                    if sub is not None:
                        sc = np.where(sub, sc, -np.inf)
                    vals.append(_val(_best(sc), _gain(evv, ia, io, obj)))
            return float(np.mean(vals))
        for k in ks_self:
            if k <= h:
                r[f"self_k{k}"] = crossfit(k)
        r["matched"] = r[f"self_k{h}"] if f"self_k{h}" in r else crossfit(h)
        # ---- You dont have to choose a sample of people to deploy：Evaluation of all samples of deployed personnel
        r["greedy"] = _val(_best(_gain(A["G"], ia, io, obj)), gD)
        planner = np.array([m["planner"] for m in meta])
        pos_i = np.array([c[2] for c in A["cands"]])
        if planner.any():
            first = pos_i[planner].min()
            r["planner_earliest"] = _val(planner & (pos_i == first), gD)
        else:
            r["planner_earliest"] = float(gD.mean())
        F = np.array([m["oracle_F"] for m in meta])
        SL = np.array([m["oracle_slack"] for m in meta])
        Fo = np.array([branch_meta(ti, o)["oracle_F"] for _, o, _ in A["cands"]])
        SLo = np.array([branch_meta(ti, o)["oracle_slack"] for _, o, _ in A["cands"]])
        if obj == "net":
            r["oracle_slack"] = _val(_best((F - Fo) * 1000 + (SL - SLo)), gD)
        else:
            r["oracle_slack"] = _val(_best(F * 1000 + SL), gD)
        r["slack_abs"] = _val(_best(F * 1000 + SL), gD)
        r["random"] = float(gD.mean())
        r["best_in_sample"] = float(gD.max())
        # ---- Choice of other implementers（k runs，Rehearsal），Evaluation of all samples of deployed personnel
        g_other = {}
        for o, TO in others.items():
            B = TO[ti]
            ro = np.random.default_rng([seed, ti, EXEC_CODE.get(o, 9)])
            n_o = B["S"].shape[1]
            g_other[o] = _gain(_full(B["det"], B["rand"], B["S"].mean(axis=1) if len(B["rand"]) else []), ia, io, obj)
            for k in ks_other:
                if k <= n_o:
                    vals = []
                    for _ in range(n_perm if k < n_o else 1):
                        pm = np.argsort(ro.random(B["S"].shape), axis=1)
                        Z = np.take_along_axis(B["S"], pm, axis=1)[:, :k].mean(axis=1)
                        vals.append(_val(_best(_gain(_full(B["det"], B["rand"], Z), ia, io, obj)), gD))
                    r[f"{o}_k{k}"] = float(np.mean(vals))
            if f"{o}_k{h}" in r:
                r[f"eq_minus_{o}"] = r["matched"] - r[f"{o}_k{h}"]
            r[f"greedy_{o}"] = _val(_best(_gain(B["G"], ia, io, obj)), gD)
        # ---- Methodology：Screening for follow-up
        alts_by_pos = {}
        for c in range(len(meta)):
            alts_by_pos.setdefault(pos_i[c], []).append(c)
        surv_pl, surv_fl = np.zeros(len(meta), bool), np.zeros(len(meta), bool)
        for i, cs in alts_by_pos.items():
            # Leave every position oracle Feasibility × 1000 + The highest remaining 1 items；Column by Branch Number（Fixed、independent of outcomes）
            surv_pl[min(cs, key=lambda c: (-(F[c] * 1000 + SL[c]), A["cands"][c][0]))] = True
            fl = [c for c in cs if meta[c].get("flash_rank") is not None]
            if fl:
                surv_fl[min(fl, key=lambda c: meta[c]["flash_rank"])] = True
        nb_rand = lambda cset: len({A["col"][x] for c in cset for x in A["cands"][c][:2]} & set(A["rand"].tolist()))
        r["budget_all"] = nb_rand(range(len(meta)))
        for tag, sub in (("SLpl", surv_pl), ("SLfl", surv_fl)):
            r[f"budget_{tag}"] = nb_rand(np.flatnonzero(sub))
            for k in ks_self:
                if k <= h:
                    r[f"{tag}_k{k}"] = crossfit(k, sub) if sub.any() else (0.0 if obj == "net" else float(fullD[io].mean()))
            best = _best(gD)
            r[f"recall_{tag}"] = float((best & sub).sum() / best.sum())
        # ---- proposer × executor
        for p in srcs:
            m_ = np.array([_has(m, p) for m in meta])
            r[f"prop_self_{p}"] = float(gD[m_].mean()) if m_.any() else np.nan
            for o, g_o in g_other.items():
                r[f"prop_{o}_{p}"] = float(g_o[m_].mean()) if m_.any() else np.nan
        bD = np.flatnonzero(_best(gD))
        r["best_is"] = {p: float(sum(_has(meta[c], p) for c in bD) / len(bD)) for p in srcs}
        rows.append(r)
    return rows
