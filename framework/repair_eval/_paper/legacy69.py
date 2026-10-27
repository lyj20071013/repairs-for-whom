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

def analyze_core(trajs7, trajs14, branch_meta, obj="net", seed=0, n_perm=N_PERM, ks7=(1, 2, 3, 5),
                 ks14=(1, 2, 3, 5, 10, 20)):
    """One-line statistics per trajectories（7B Value of deployment on board）。trajs7 / trajs14：_traj_arrays Outcome（Same group of candidates）；branch_meta(trajectories Serial Numbers, branch) -> Metadata。"""
    rng = np.random.default_rng(seed)
    rows = []
    for ti, A in enumerate(trajs7):
        B = trajs14[ti] if trajs14 is not None else None
        ia, io, det, rand = A["ia"], A["io"], A["det"], A["rand"]
        n7 = A["S"].shape[1]
        h = n7 // 2
        full7 = _full(det, rand, A["S"].mean(axis=1) if len(rand) else [])
        g7 = _gain(full7, ia, io, obj)
        meta = [branch_meta(ti, a) for a, _, _ in A["cands"]]
        r = {}
        # ---- 7B Myself k Subselection（Crossing）：The first half of each rehearsing（before k runs）、The second half，Reconcile
        perms = [np.argsort(rng.random(A["S"].shape), axis=1) for _ in range(n_perm)]

        def crossfit(k, sub=None):
            vals = []
            for pm in perms:
                Z = np.take_along_axis(A["S"], pm, axis=1)
                for a_, b_ in ((slice(0, h), slice(h, n7)), (slice(h, n7), slice(0, h))):
                    sel = _full(det, rand, Z[:, a_][:, :k].mean(axis=1))
                    evv = _full(det, rand, Z[:, b_].mean(axis=1))
                    sc = _gain(sel, ia, io, obj)
                    if sub is not None:
                        sc = np.where(sub, sc, -np.inf)
                    vals.append(_val(_best(sc), _gain(evv, ia, io, obj)))
            return float(np.mean(vals))
        for k in ks7:
            if k <= h:
                r[f"7B_k{k}"] = crossfit(k)
        r["matched"] = r[f"7B_k{h}"] if f"7B_k{h}" in r else crossfit(h)
        # ---- not used 7B Selection of Samples：at 7B Evaluation on all samples
        G7 = _gain(A["G"], ia, io, obj)
        r["greedy"] = _val(_best(G7), g7)
        planner = np.array([m["planner"] for m in meta])
        pos_i = np.array([c[2] for c in A["cands"]])
        if planner.any():
            first = pos_i[planner].min()
            r["planner_earliest"] = _val(planner & (pos_i == first), g7)
        else:
            r["planner_earliest"] = float(g7.mean())
        F = np.array([m["oracle_F"] for m in meta])
        SL = np.array([m["oracle_slack"] for m in meta])
        Fo = np.array([branch_meta(ti, o)["oracle_F"] for _, o, _ in A["cands"]])
        SLo = np.array([branch_meta(ti, o)["oracle_slack"] for _, o, _ in A["cands"]])
        if obj == "net":
            r["oracle_slack"] = _val(_best((F - Fo) * 1000 + (SL - SLo)), g7)
        else:
            r["oracle_slack"] = _val(_best(F * 1000 + SL), g7)
        r["slack_abs"] = _val(_best(F * 1000 + SL), g7)
        r["random"] = float(g7.mean())
        r["best_in_sample"] = float(g7.max())
        # ---- 14B Selection（k runs，Rehearsal），at 7B Evaluation on all samples
        if B is not None:
            n14 = B["S"].shape[1]
            full14 = _full(B["det"], B["rand"], B["S"].mean(axis=1) if len(B["rand"]) else [])
            g14 = _gain(full14, ia, io, obj)
            for k in ks14:
                if k <= n14:
                    vals = []
                    for _ in range(n_perm if k < n14 else 1):
                        pm = np.argsort(rng.random(B["S"].shape), axis=1)
                        Z = np.take_along_axis(B["S"], pm, axis=1)[:, :k].mean(axis=1)
                        vals.append(_val(_best(_gain(_full(B["det"], B["rand"], Z), ia, io, obj)), g7))
                    r[f"14B_k{k}"] = float(np.mean(vals))
            # Same thing h runs：7B Crossing vs 14B h runs（all in 7B Up-to-date evaluation）
            r["eq_7B_minus_14B"] = r["matched"] - r[f"14B_k{h}"]
            r["greedy14"] = _val(_best(_gain(B["G"], ia, io, obj)), g7)
            r["_g14"] = g14
        # ---- Methodology：Screening for follow-up
        alts_by_pos = {}
        for c, m in zip(range(len(meta)), meta):
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
            for k in ks7:
                if k <= h:
                    r[f"{tag}_k{k}"] = crossfit(k, sub) if sub.any() else (0.0 if obj == "net" else float(full7[io].mean()))
            best = _best(g7)
            r[f"recall_{tag}"] = float((best & sub).sum() / best.sum())
        # ---- proposer × executor
        for p in ("q7", "q14", "qflash", "planner", "random"):
            m_ = np.array([(p in m["proposers"]) if p.startswith("q") else bool(m[p]) for m in meta])
            r[f"prop7_{p}"] = float(g7[m_].mean()) if m_.any() else np.nan
            if B is not None:
                r[f"prop14_{p}"] = float(r["_g14"][m_].mean()) if m_.any() else np.nan
        r["best_is"] = {p: float(sum((p in meta[c]["proposers"]) if p.startswith("q") else bool(meta[c][p])
                                     for c in np.flatnonzero(_best(g7))) / _best(g7).sum())
                        for p in ("q7", "q14", "qflash", "planner", "random")}
        r.pop("_g14", None)
        rows.append(r)
    return rows
