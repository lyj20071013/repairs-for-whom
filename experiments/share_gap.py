"""Sharing the cost of repair（After-action analysis，exploratory；For registration see analysis plan.md 6.8）：Cross-executor selection loss，From“Cant tell the difference between good and bad”or“genuinely different preferences”。

To a trajectories、Executor pair (A, B)：
    pick_e  = argmax R̂_e（e Choose the net gain on the sample for yourself），pick_AB = argmax (R̂_A + R̂_B) / 2，Averages the grouping of the columns when side-by-side
    L_{A→B} = V_B(pick_B) − V_B(pick_A)                      Cross-selection loss
    Δ_B     = V_B(pick_B) − V_B(pick_AB)                     By B The cost of sharing options when targeting
    G_AB    = (Δ_A + Δ_B) / 2                                Sharing the cost of repair（true value ≥ 0）
    reversal     = L_{A→B} > 0 and L_{B→A} > 0
    margin of choice = V_e(pick_e) − V_e(random)                        use e additional value from selection using its own labels（Resolution Description）
V always evaluate on samples disjoint from selection。

data：
    alfw  ALFWorld stage B（out_alfw_b）+ Stairs（out_ladder_alfw）；7B 30 runs、14B / flash Each 10 runs
    gw    gridworld Confirm the experiment（out_confirm）+ Stairs（out_ladder_gw）；7B 40 runs、14B / flash Each 10 runs
    q7    gridworld Datasets q7 Ji!（out_ds_gw，7B and 14B Two-stage labels for two executors；14B You cant count until the evaluation is complete）
alfw / gw：Each executor uses the same 10 runs（7B Cut 10 Next piece，Average individual blocks），Each piece 5 Selection 5 Commentary、Crossing，Random reordering of intra-branch samples N_PERM Lower Average。
q7：Designed by data set S1a Selection、Evaluation with unbiased labels not related to selection。
References：Parameter zero hypothesis simulated by the same sample structure——Consolidation of zero assumptions（Registration：The real success rate for both enforcers is equal to a combined estimate）Aligning zero with sorting
（6.8 Computation added：Level of respective retention success rate，Net proceeds reordered in each other  s order，G true value = 0；**Withdrawn**——Cut [0, 1] It changes the extraction value，
See 6.8 Revision 2）；alfw / gw Other empirical references（7B The two that dont work together 10 Sub-bults）。
Known limits：when one executor is nearly flat but noisy Ĝ inflated、both null models under-correct（selftest [4]）。
interval：A statistic per trajectory，trajectory bootstrap，90% Percent Zone。

    python share_gap.py --env all          # alfw + gw（+ q7，if 14B Assessment completed）
    python share_gap.py --env alfw|gw|q7
    python share_gap.py --selftest         # Test for known answers（Simulation Data，Do not read any outcome document）
output out_share/share_gap.md / .json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config

BLOCK = 10          # Samples to be compared per executor（And the steps 10 Unanimously）
N_PERM = 50         # Random reordering of intra-branch samples（Just lower the difference between the cut and the cut）
N_PERM_NULL = 8
N_BOOT = 2000
N_NULL = 200
ALPHA = 0.05        # 90% interval
PAIRS = (("7B", "14B"), ("7B", "flash"))
PRACTICAL = 0.05    # Reading 2 upper threshold value for（and 6.5 identical）
EPS = 1e-12
STAT_KEYS = ("L_AB", "L_BA", "D_A", "D_B", "G", "rev", "head_A", "head_B", "same")
STAT_LABEL = {"L_AB": "Cross-cutting losses L_{A→B}", "L_BA": "Cross-cutting losses L_{B→A}", "D_A": "Δ_A（A The cost of sharing options）",
              "D_B": "Δ_B（B The cost of sharing options）", "G": "**Sharing the cost of repair G**", "rev": "reversal rate（In both directions > 0）",
              "head_A": "margin of choice A", "head_B": "margin of choice B", "same": "Probability of choosing the same candidate"}


# ================================================================ Statistical core
def _value(mask, g):
    """The averages on the collegiate arrays：mask, g A shape like that (..., C)。"""
    return (mask * g).sum(axis=-1) / mask.sum(axis=-1)


def _argmax_mask(g):
    return g >= g.max(axis=-1, keepdims=True) - EPS


def pair_stats(gsA, geA, gsB, geB):
    """gs*：Net proceeds of selection，ge*：Net proceeds from evaluation（Unintersected with Selection），A shape like that (..., C)。Returns individual statistics，A shape like that (...)。"""
    mA, mB = _argmax_mask(gsA), _argmax_mask(gsB)
    mS = _argmax_mask((gsA + gsB) / 2)
    vAA, vAB, vAS = _value(mA, geA), _value(mB, geA), _value(mS, geA)      # at A Up-to-date evaluation
    vBB, vBA, vBS = _value(mB, geB), _value(mA, geB), _value(mS, geB)      # at B Up-to-date evaluation
    L_AB, L_BA = vBB - vBA, vAA - vAB
    D_A, D_B = vAA - vAS, vBB - vBS
    same = (mA & mB).sum(axis=-1) / (mA.sum(axis=-1) * mB.sum(axis=-1))   # Select one randomly when side-by-side，probability of selecting the same candidate
    return {"L_AB": L_AB, "L_BA": L_BA, "D_A": D_A, "D_B": D_B, "G": (D_A + D_B) / 2,
            "rev": ((L_AB > EPS) & (L_BA > EPS)).astype(float),
            "head_A": vAA - geA.mean(axis=-1), "head_B": vBB - geB.mean(axis=-1), "same": same}


def multi_G(gs, ge):
    """Shared repair costs for three or more practitioners：gs, ge As List，Each (..., C)。"""
    mS = _argmax_mask(sum(gs) / len(gs))
    return np.mean([_value(_argmax_mask(s), e) - _value(mS, e) for s, e in zip(gs, ge)], axis=0)


# ================================================================ alfw / gw：Wait for the sample、Crossing
class Traj:
    """A trajectories：candidate (alt, orig) Columns and subscripts、deterministic branch value、Sample matrix on random branches of each executor (n_rand, n)。"""

    def __init__(self, tid, ia, io, det, samples):
        self.id, self.ia, self.io, self.det, self.S = tid, ia, io, det, samples
        self.rand = np.flatnonzero(np.isnan(det))

    def gains(self, Y, rng, n_perm):
        """Y：A sample from the random branch of the executor (n_rand, BLOCK)。Back (sel, ev)，A shape like that (n_perm, 2, C)：
        Two cuts（First Half / Second half，and with the roles swapped），Average 2D when cross-mixing。"""
        nb = len(self.det)
        h = BLOCK // 2
        r = np.empty((n_perm, 2, 2, nb))                    # perm, Cut, (Selection, Commentary), branch
        r[..., :] = np.nan_to_num(self.det)                 # deterministic branch：Selection、use the same value for evaluation
        if len(self.rand):
            order = np.argsort(rng.random((n_perm, len(self.rand), BLOCK)), axis=-1)
            Z = np.take_along_axis(np.broadcast_to(Y, (n_perm,) + Y.shape), order, axis=-1)
            a, b = Z[..., :h].mean(axis=-1), Z[..., h:].mean(axis=-1)
            r[:, 0, 0, self.rand], r[:, 0, 1, self.rand] = a, b
            r[:, 1, 0, self.rand], r[:, 1, 1, self.rand] = b, a
        g = r[..., self.ia] - r[..., self.io]
        return g[:, :, 0, :], g[:, :, 1, :]

    def blocks(self, ex):
        X = self.S[ex]
        return [X[:, i * BLOCK:(i + 1) * BLOCK] for i in range(X.shape[1] // BLOCK)]


def load_ladder(env, root):
    import consequence
    trajs, br, _ = consequence.load(env, root)
    out = []
    for t in trajs:
        cands = [(a, p["orig"]) for p in t["positions"] for a in p["alts"]]
        bids = sorted({b for c in cands for b in c})
        col = {b: j for j, b in enumerate(bids)}
        det = np.array([np.nan if br[b]["det"] is None else float(br[b]["det"]) for b in bids])
        S = {}
        for ex in ("7B", "14B", "flash"):
            rows = [np.concatenate(br[b][ex]).astype(float) for b in bids if br[b]["det"] is None]
            S[ex] = np.array(rows) if rows else np.zeros((0, 0))
        out.append(Traj(t["id"], np.array([col[a] for a, _ in cands]), np.array([col[o] for _, o in cands]), det, S))
    return out


def traj_pair(tr, A, B, rng, n_perm=N_PERM):
    """On a trajectories (A, B) Statistics of the United Nations Population Fund：A、B Every one of them 10 Next piece；How many executors?（7B）Average individual blocks。"""
    acc = []
    for YA in tr.blocks(A):
        for YB in tr.blocks(B):
            sA, eA = tr.gains(YA, rng, n_perm)
            sB, eB = tr.gains(YB, rng, n_perm)
            acc.append({k: float(np.mean(v)) for k, v in pair_stats(sA, eA, sB, eB).items()})
    return {k: float(np.mean([a[k] for a in acc])) for k in STAT_KEYS}


def traj_pseudo(tr, ex, rng, n_perm=N_PERM):
    """Empirical references：The same executors two separate 10 The second block is two“executor”（true value G = L = 0）。"""
    bl = tr.blocks(ex)
    acc = []
    for i in range(len(bl)):
        for j in range(i + 1, len(bl)):
            sA, eA = tr.gains(bl[i], rng, n_perm)
            sB, eB = tr.gains(bl[j], rng, n_perm)
            acc.append({k: float(np.mean(v)) for k, v in pair_stats(sA, eA, sB, eB).items()})
    return {k: float(np.mean([a[k] for a in acc])) for k in STAT_KEYS} if acc else None


def traj_triple(tr, exs, rng, n_perm=N_PERM):
    acc = []
    for Y7 in tr.blocks(exs[0]):
        gs, ge = [], []
        for ex, Y in zip(exs, [Y7] + [tr.blocks(e)[0] for e in exs[1:]]):
            s, e = tr.gains(Y, rng, n_perm)
            gs.append(s)
            ge.append(e)
        acc.append(float(np.mean(multi_G(gs, ge))))
    return float(np.mean(acc))


def _shrink(X):
    return (X.sum(axis=1) + 0.5) / (X.shape[1] + 1)


def align(full_lead, full_follow, ia, io, movable):
    """Sort Alignment：Reservations follow Executor s own net gain margin（Entire flatness），But press lead Redistribution of the net proceeds of the implementer to the candidate，
    align the two executors optimal candidates（true value G = 0）。only in movable（Random alternative branch）Rehearsal between；The original action branch remains unchanged。Back to New follow Success vector。"""
    out = full_follow.copy()
    c = np.flatnonzero(movable)
    if len(c) < 2:
        return out
    R_lead = full_lead[ia[c]] - full_lead[io[c]]
    R_fol = np.sort(full_follow[ia[c]] - full_follow[io[c]])[::-1]
    order = np.argsort(-R_lead, kind="stable")
    new = np.empty(len(c))
    new[order] = R_fol
    out[ia[c]] = np.clip(full_follow[io[c]] + new, 0.0, 1.0)
    return out


def null_ladder(trajs, A, B, rng, n_sim=N_NULL, mode="pooled"):
    """Parameter zero assumption，By Same Sample Structure（Each 10 runs）mock；Back (n_sim, Statistics) orbital averages。
    pooled（Registration）：Every Random Branch A、B The real success rate is equal to the combined estimate。
    aligned（6.8 Computation added）：Implementers keep their own estimates of success rates，but B Net proceeds by A Reordering（simulate the opposite selection direction for half B Reordering A），
    The two are the best candidates、true value G = 0，At the same time, keep their respective success rates and flat levels。"""
    worlds = []
    for tr in trajs:
        if mode == "pooled":
            k = tr.S[A].sum(axis=1) + tr.S[B].sum(axis=1)
            n = tr.S[A].shape[1] + tr.S[B].shape[1]
            p = (k + 0.5) / (n + 1)
            worlds.append([(p, p)])
        else:
            fA, fB = np.nan_to_num(tr.det), np.nan_to_num(tr.det)
            fA[tr.rand], fB[tr.rand] = _shrink(tr.S[A]), _shrink(tr.S[B])
            movable = np.isnan(tr.det[tr.ia])
            worlds.append([(fA[tr.rand], align(fA, fB, tr.ia, tr.io, movable)[tr.rand]),
                           (align(fB, fA, tr.ia, tr.io, movable)[tr.rand], fB[tr.rand])])
    sims = np.empty((n_sim, len(STAT_KEYS)))
    for s in range(n_sim):
        per = []
        for tr, w in zip(trajs, worlds):
            pA, pB = w[s % len(w)]
            YA = (rng.random((len(pA), BLOCK)) < pA[:, None]).astype(float)
            YB = (rng.random((len(pB), BLOCK)) < pB[:, None]).astype(float)
            sA, eA = tr.gains(YA, rng, N_PERM_NULL)
            sB, eB = tr.gains(YB, rng, N_PERM_NULL)
            st = pair_stats(sA, eA, sB, eB)
            per.append([float(np.mean(st[k])) for k in STAT_KEYS])
        sims[s] = np.mean(per, axis=0)
    return sims


# ================================================================ q7：A two-stage label for the data set
Q7_NAME = {"q7": "7B", "q14": "14B"}


def load_q7(root):
    import ds_gw
    ds_gw.STATE["out"] = Path(root) / "out_ds_gw"
    if not ds_gw.P("design", "q7").exists():
        return None, "Missing design_q7.json"
    d = json.loads(ds_gw.P("design", "q7").read_text(encoding="utf-8"))
    evs = {}
    rand = {b for b, x in d["branches"].items() if x["deterministic"] is None}
    for ex in ("q7", "q14"):
        ev = ds_gw.load_evals(ds_gw.P("evals", "q7", ex))
        if not ev:
            return None, f"Missing evals_q7_by_{ex}.jsonl"
        # The assessment must be complete（S1、greedy、S2 all complete）counts only then：S1 stage-two candidates are not fixed while incomplete，some trajectories appear labeled but have only S1b of 5 runs
        done = {ph: {b for (p, b) in ev if p == ph} for ph in ("S1", "G", "S2")}
        if done["S1"] >= rand and done["G"] >= rand:
            top = ds_gw.stage2_choice(d, ev)
            s2 = set()
            for t in d["trajectories"]:
                s2 |= {b for b in top[t["traj_id"]] if b in rand}
                s2 |= {p["orig"] for p in t["positions"] if p["orig"] in rand}
            missing_s2 = len(s2 - done["S2"])
        else:
            missing_s2 = None
        if missing_s2 != 0:
            return None, (f"evals_q7_by_{ex} incomplete：S1 {len(done['S1'] & rand)}/{len(rand)}，greedy {len(done['G'] & rand)}/{len(rand)}，"
                          f"S2 " + ("Not started" if missing_s2 is None else f"Not available {missing_s2} branches"))
        evs[ex] = ev
    return (d, evs), None


def q7_stats(d, evs):
    """A statistical dictionary per trajectory；Candidates without labels（Assessment not completed）skip these trajectories。Back (rows, n_skipped)。"""
    import ds_gw
    labs = {ex: ds_gw.labels(d, ev) for ex, ev in evs.items()}
    # The original action success rate used in the evaluation is used only for S1b + S2：ds_gw.labels Original Action Label Contains S1a，whereas S1a Participated in the selection（6.8 Revision，23:35，Look 14B Before the results；
    # at 7B Self-appraisal of this lead S1a High net proceeds of selection 0.006）
    po = {ex: {p["orig"]: ds_gw._rate(ds_gw._p(d, ev, p["orig"], ("S1b", "S2")))
               for t in d["trajectories"] for p in t["positions"]} for ex, ev in evs.items()}
    rows, skipped = [], 0
    for t in d["trajectories"]:
        cs = [a for p in t["positions"] for a in p["alts"]]
        orig = {a: p["orig"] for p in t["positions"] for a in p["alts"]}
        g = {ex: (np.array([labs[ex][a]["s1a_gain"] for a in cs]),
                  np.array([labs[ex][a]["p_alt"] - po[ex][orig[a]] for a in cs]))
             for ex in ("q7", "q14")}
        if any(not np.all(np.isfinite(x)) for v in g.values() for x in v):
            skipped += 1
            continue
        st = pair_stats(g["q7"][0], g["q7"][1], g["q14"][0], g["q14"][1])
        rows.append({k: float(v) for k, v in st.items()})
    return rows, skipped


def _sim_evals(d, p, rng):
    """Simulation of the results of an executors assessment by two-stage design of the data set（success rate p[b]）。"""
    import ds_gw
    ev = {}
    rand = [b for b, x in d["branches"].items() if x["deterministic"] is None]
    for b in rand:
        ev[("S1", b)] = {"results": list(rng.random(ds_gw.N1) < p[b])}
    top = ds_gw.stage2_choice(d, ev)
    s2 = set()
    for t in d["trajectories"]:
        s2 |= {b for b in top[t["traj_id"]] if d["branches"][b]["deterministic"] is None}
        s2 |= {q["orig"] for q in t["positions"] if d["branches"][q["orig"]]["deterministic"] is None}
    for b in s2:
        ev[("S2", b)] = {"results": list(rng.random(ds_gw.N2) < p[b])}
    return ev


def _q7_aligned(d, p_lead, p_fol):
    """and align() Same sorting alignment，By trajectories q7 Designed branch dictionary：Back to New follow Success Dictionary。"""
    out = dict(p_fol)
    val = lambda pp, b: float(d["branches"][b]["deterministic"]) if d["branches"][b]["deterministic"] is not None else pp[b]
    for t in d["trajectories"]:
        c = [(a, q["orig"]) for q in t["positions"] for a in q["alts"] if d["branches"][a]["deterministic"] is None]
        if len(c) < 2:
            continue
        R_lead = np.array([val(p_lead, a) - val(p_lead, o) for a, o in c])
        R_fol = np.sort([val(p_fol, a) - val(p_fol, o) for a, o in c])[::-1]
        for rank, j in enumerate(np.argsort(-R_lead, kind="stable")):
            a, o = c[j]
            out[a] = float(np.clip(val(p_fol, o) + R_fol[rank], 0.0, 1.0))
    return out


def null_q7(d, evs, rng, n_sim=N_NULL, mode="pooled"):
    """and null_ladder The same two zeros，Design simulations by two phases of the data set。"""
    import ds_gw
    rand = [b for b, x in d["branches"].items() if x["deterministic"] is None]
    own = {}
    for ex, ev in evs.items():
        own[ex] = {}
        for b in rand:
            k, n = ds_gw._p(d, ev, b, ("S1", "S2"))
            own[ex][b] = (k + 0.5) / (n + 1)
    if mode == "pooled":
        kn = {b: [sum(x) for x in zip(*(ds_gw._p(d, ev, b, ("S1", "S2")) for ev in evs.values()))] for b in rand}
        p = {b: (k + 0.5) / (n + 1) for b, (k, n) in kn.items()}
        worlds = [(p, p)]
    else:
        worlds = [(own["q7"], _q7_aligned(d, own["q7"], own["q14"])), (_q7_aligned(d, own["q14"], own["q7"]), own["q14"])]
    sims = np.empty((n_sim, len(STAT_KEYS)))
    for s in range(n_sim):
        pA, pB = worlds[s % len(worlds)]
        rows, _ = q7_stats(d, {"q7": _sim_evals(d, pA, rng), "q14": _sim_evals(d, pB, rng)})
        sims[s] = [np.mean([r[k] for r in rows]) for k in STAT_KEYS]
    return sims


# ================================================================ Summary
def boot(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, ALPHA)), float(np.quantile(m, 1 - ALPHA))


def summarize(rows, nulls, pseudo_rows, rng):
    """rows：Statistics for each trajectory；nulls：{name: Zero Hypothesis Simulation (n_sim, K)}（The first is the registered zero-sum scenario）；pseudo_rows：Empirical references（Yes None）。"""
    out = {}
    for j, k in enumerate(STAT_KEYS):
        x = np.array([r[k] for r in rows])
        m, lo, hi = boot(x, rng)
        o = {"obs": [m, lo, hi], "null": {}}
        for name, sims in nulls.items():
            nm = float(sims[:, j].mean())
            o["null"][name] = {"mean": nm, "q95": float(np.quantile(sims[:, j], 0.95)), "minus": [m - nm, lo - nm, hi - nm],
                               "p": float((np.sum(sims[:, j] >= m) + 1) / (len(sims) + 1))}
        if pseudo_rows is not None:
            o["pseudo"] = boot(np.array([r[k] for r in pseudo_rows]), rng)
            o["minus_pseudo"] = boot(x - np.array([r[k] for r in pseudo_rows]), rng)
        out[k] = o
    lab = np.array([r["L_BA"] for r in rows])       # B → A Cross-cutting losses（A = 7B corresponds to“by B Selection、Here 7B use”）
    dA = np.array([r["D_A"] for r in rows])
    out["ratio_DA_over_L_BA"] = float(dA.mean() / lab.mean()) if abs(lab.mean()) > 1e-9 else float("nan")
    out["n_traj"] = len(rows)
    return out


def fmt(v):
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"


NULL_LABEL = {"pooled": "Consolidation of zero assumptions（Registration）", "aligned": "Sorting Zero Hypothesis（Withdrawn，See 6.8 Revision 2）"}


def table(A, B, s):
    names = list(s["G"]["null"])
    ps = "pseudo" in s["G"]
    head = f"| Statistics（A = {A}，B = {B}） | Observation [90%] |"
    for n in names:
        head += f" {NULL_LABEL[n]}：Mean | Observation − Average [90%] |"
    if ps:
        head += " Empirical references 7B–7B′ | Observation − Empirical references [90%] |"
    L = [head, "|---|---|" + "---|---|" * len(names) + ("---|---|" if ps else "")]
    for k in STAT_KEYS:
        o = s[k]
        row = f"| {STAT_LABEL[k]} | {fmt(o['obs'])} |"
        if k.startswith("head"):          # margin of choice：Zero scenario versus 7B even the reference is not comparable，Only observational values
            row += " — | — |" * len(names) + (" — | — |" if ps else "")
        else:
            for n in names:
                row += f" {o['null'][n]['mean']:+.3f} | {fmt(o['null'][n]['minus'])} |"
            if ps:
                row += f" {o['pseudo'][0]:+.3f} | {fmt(o['minus_pseudo'])} |"
        L.append(row)
    L.append(f"\nΔ_A / L_{{B→A}} = {s['ratio_DA_over_L_BA']:.2f}（A After changing to share options，What percentage of cross-lost losses remain）")
    return L


def run_ladder(env, root, rng):
    trajs = load_ladder(env, root)
    res, L = {}, []
    title = {"alfw": "ALFWorld stage B + Stairs", "gw": "gridworld Confirm the experiment + Stairs"}[env]
    L += [f"## {title}（{len(trajs)} trajectories，signal detected after screening；Each executor 10 runs，5 Selection 5 Commentary）", ""]
    pseudo = [traj_pseudo(tr, "7B", rng) for tr in trajs]
    for A, B in PAIRS:
        rows = [traj_pair(tr, A, B, rng) for tr in trajs]
        nulls = {m: null_ladder(trajs, A, B, rng, mode=m) for m in ("pooled", "aligned")}
        s = summarize(rows, nulls, pseudo, rng)
        res[f"{A}-{B}"] = s
        L += [f"### {A} and {B}", ""] + table(A, B, s) + [""]
    g3 = [traj_triple(tr, ("7B", "14B", "flash"), rng) for tr in trajs]
    res["G3"] = boot(g3, rng)
    L += [f"- The third implementer shares the cost of repair G3（7B、14B、flash，Just a description）：{fmt(res['G3'])}", ""]
    return res, L


def run_q7(root, rng):
    data, why = load_q7(root)
    if data is None:
        return None, [f"## gridworld q7 Ji!：skip（{why}）", ""]
    d, evs = data
    rows, skipped = q7_stats(d, evs)
    if not rows:
        return None, ["## gridworld q7 Ji!：skip（No trajectories for complete tabs）", ""]
    nulls = {m: null_q7(d, evs, rng, mode=m) for m in ("pooled", "aligned")}
    s = summarize(rows, nulls, None, rng)
    L = [f"## gridworld q7 Ji!（{len(rows)} The strips are not filtered 7B Failed trajectories" + (f"，{skipped} Bars skipped because the labels were incomplete" if skipped else "")
         + "；S1a Selection、No bias in label evaluation）", ""] + table("7B", "14B", s) + [""]
    return s, L


def verdict(res):
    """Reader  s methods of registration（6.8），Only the Lord is right (7B, 14B)。"""
    if "alfw" not in res or "gw" not in res:
        return "（Yes alfw and gw The results of both environments can only be judged by registration reading）"
    ga = res["alfw"]["7B-14B"]["G"]["null"]["pooled"]["minus"]
    gg = res["gw"]["7B-14B"]["G"]["null"]["pooled"]["minus"]
    big_bias = max(abs(res[e]["7B-14B"]["G"]["null"]["pooled"]["mean"]) for e in ("alfw", "gw")) > PRACTICAL
    if big_bias:
        return "Under the zero scenario Ĝ The deviation > 0.05：30 The level of data is not enough to answer → Reading 3（etc q7 Pool with ALFWorld Datasets）"
    if ga[1] > 0:
        return "Reading 1：ALFWorld middle 7B and 14B genuine preference conflict exists（Ĝ − Zero hypothetical average Bottom bounds > 0）→ Keep the data set labeled as core by the executor"
    if ga[1] <= 0 <= ga[2] and gg[1] <= 0 <= gg[2] and ga[2] < PRACTICAL and gg[2] < PRACTICAL:
        return "Reading 2：A conflict between two environments that has no real meaning → Narrowing resolution and selection uncertainty for proxy assessments，Executor condition label reduced to additional information"
    return "Reading 3：cannot distinguish → etc q7 Ji!（100 Bar does not filter）and ALFWorld A two-stage label for the data set，Do not modify the data set design accordingly"


def run(env, root, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    res, L = {}, ["# Sharing the cost of repair：Resolution loss vs Real Preference Conflict（After-action analysis，exploratory；For registration see analysis plan.md 6.8）", "",
                  "all V Theyre all evaluated on samples that dont match the choice。G true value ≥ 0，But under a limited sample, Ĝ biased，therefore the reported quantity is“Observation − References”。Three references：",
                  "",
                  "- **Consolidation of zero assumptions（Registration）**：The two executors have the same real success rate in each branch（Combined estimates），Simulate by the same sample structure "
                  + str(N_NULL) + " runs；",
                  "- **Sorting Zero Hypothesis（Computation added）**：Implementers keep their own estimates of success rates（Horizontal and flat），But the net proceeds are reordered according to each others order，"
                  "The two are the best candidates、true value G = 0。Thats the part that goes beyond it“Sorting is really different”；its own L Yes“The other s extraction shape"
                  "（for example, cannot distinguish among good candidates）”Losses caused；",
                  "- **Empirical references**：7B The two that dont work together 10 The second block as two executors。",
                  "",
                  "**Sorting zero assumption withdrawn（6.8 Revision 2）**：If you add the net gain from the re-allocation to the success rate of the original action, youll have to cut it off [0, 1]，Cut changed the extraction value"
                  "（About half the trail），The best fixes of the two are still inconsistent after partial trajectories are aligned；This column is reserved for retroactive purposes only，No explanation。", "",
                  "Known limits（selftest [4]）：if one executor is nearly flat but noisy（Similar flash Up to the top），Ĝ And the reverse rate will be pushed up by noise，both null models under-correct"
                  "（Complete flat-time residue +0.03 to +0.06）；in this case only“L_{A→B} > 0”（For the flat side, by the other side）Not affected。"
                  "Select margin approach 0 indicates this case。", ""]
    envs = ("alfw", "gw", "q7") if env == "all" else (env,)
    for e in envs:
        if e == "q7":
            s, LL = run_q7(root, rng)
            if s is not None:
                res["q7"] = s
        else:
            res[e], LL = run_ladder(e, root, rng)
        L += LL
    if env == "all":
        res["verdict"] = verdict(res)
        L += ["## Read by registration method（Yeah, thats right 7B and 14B）", "", res["verdict"], ""]
    out = root / "out_share"
    out.mkdir(exist_ok=True)
    (out / "share_gap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "share_gap.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return res


# ================================================================ Test for known answers
def _synthetic(n_traj, pA, pB, poA, poB, rng, nA=30, nB=10):
    """One position per trajectory：original action + len(pA) An alternative action；pA / pB the two executors true success probabilities on alternative branches，poA / poB As Original Action。"""
    trajs = []
    for i in range(n_traj):
        C = len(pA)
        ia, io = np.arange(1, C + 1), np.zeros(C, int)
        det = np.full(C + 1, np.nan)
        SA = (rng.random((C + 1, nA)) < np.r_[poA, pA][:, None]).astype(float)
        SB = (rng.random((C + 1, nB)) < np.r_[poB, pB][:, None]).astype(float)
        trajs.append(Traj(f"s{i}", ia, io, det, {"A": SA, "B": SB}))
    return trajs


def _synthetic_q7(n_traj, pA, pB, poA, poB):
    """and ds_gw Designing small designs of co-constructs（One position per trajectory），And the real success rate dictionary for the two executors。"""
    d = {"pool": "syn", "trajectories": [], "branches": {}}
    tA, tB = {}, {}
    for i in range(n_traj):
        tid = f"s{i}"
        o = f"{tid}|orig"
        alts = [f"{tid}|a{j}" for j in range(len(pA))]
        d["trajectories"].append({"traj_id": tid, "fail_reason": "syn", "positions": [{"t": 0, "orig": o, "alts": alts}]})
        for b, a_, b_ in [(o, poA, poB)] + list(zip(alts, pA, pB)):
            d["branches"][b] = {"deterministic": None}
            tA[b], tB[b] = a_, b_
    return d, tA, tB


def selftest():
    def check(cond, msg):
        if not cond:
            raise AssertionError(msg)
        print("  OK ", msg)
    rng = np.random.default_rng(123)

    def summary_of(pA, pB, poA=0.0, poB=None, n=80):
        poB = poA if poB is None else poB
        trajs = _synthetic(n, np.array(pA), np.array(pB), poA, poB, rng)
        rows = [traj_pair(tr, "A", "B", rng, n_perm=20) for tr in trajs]
        nulls = {m: null_ladder(trajs, "A", "B", rng, n_sim=40, mode=m) for m in ("pooled", "aligned")}
        return summarize(rows, nulls, None, rng)

    mp = lambda s, k="G", m="pooled": s[k]["null"][m]["minus"]

    print("[1] The two executors are exactly the same：G = L = 0")
    s = summary_of([0.6, 0.3, 0.1], [0.6, 0.3, 0.1])
    check(abs(mp(s)[0]) < 0.03 and abs(mp(s, m="aligned")[0]) < 0.03,
          f"Ĝ − Zero hypothetical average ≈ 0（Merge {mp(s)[0]:+.3f}，Alignment {mp(s, m='aligned')[0]:+.3f}）")
    check(abs(s["L_AB"]["obs"][0]) < 0.04 and abs(s["L_BA"]["obs"][0]) < 0.04,
          f"in both directions L ≈ 0（{s['L_AB']['obs'][0]:+.3f}, {s['L_BA']['obs'][0]:+.3f}）")
    check(s["G"]["null"]["pooled"]["mean"] <= 0.005, f"Under the zero scenario Ĝ not positive（Personally chooses to be louder）：{s['G']['null']['pooled']['mean']:+.3f}")

    print("[2] Real Preference Conflict：A Preferences r1、B Preferences r2，G true value = 0.8 − 0.5 = 0.3")
    s = summary_of([0.8, 0.2], [0.2, 0.8])
    check(mp(s)[1] > 0.1 and mp(s, m="aligned")[1] > 0.1, f"Ĝ − Zero hypothetical average Bottom bounds > 0.1（{fmt(mp(s))}）")
    check(s["L_AB"]["obs"][1] > 0.1 and s["L_BA"]["obs"][1] > 0.1, "Cross-cutting losses in both directions are significant")

    print("[3] Success rates vary、The best candidate is the same（Similar 14B）：G true value = 0")
    s = summary_of([0.6, 0.3, 0.1, 0.05, 0.0, 0.0], [0.9, 0.7, 0.5, 0.4, 0.3, 0.3], poA=0.0, poB=0.2)
    check(abs(mp(s)[0]) < 0.02 and abs(mp(s, m="aligned")[0]) < 0.02,
          f"Both zeros are calibrated（Merge {mp(s)[0]:+.3f}，Alignment {mp(s, m='aligned')[0]:+.3f}）")

    print("[4] Known limits：B flat but noisy（Similar flash Up to the top），G true value = 0")
    s = summary_of([0.6, 0.3, 0.1, 0.05, 0.0, 0.0], [0.85] * 6, poA=0.0, poB=0.75)
    check(mp(s)[0] > 0.02, f"Ĝ The noise of the flat executor pushed up（Merge {mp(s)[0]:+.3f}，Alignment {mp(s, m='aligned')[0]:+.3f}）；"
          "in this case do not interpret G Read Preferred Conflict")
    check(abs(s["L_AB"]["obs"][0]) < 0.04, f"But press A select for the flat executor B No loss：L_{{A→B}} ≈ 0（{s['L_AB']['obs'][0]:+.3f}）"
          "——“Cross-cutting losses in both directions > 0”unaffected by this noise")
    check(s["head_B"]["obs"][0] < 0.05 < s["head_A"]["obs"][0], "B The margin of choice ≈ 0，A Thats right（It can be used to identify this situation）")

    print("[5] Cross-mixing：all true values are 0 _Other Organiser ≈ 0")
    s = summary_of([0.4, 0.4, 0.4], [0.4, 0.4, 0.4], poA=0.4)
    check(abs(s["head_A"]["obs"][0]) < 0.03 and abs(s["head_B"]["obs"][0]) < 0.03,
          f"margin of choice ≈ 0（{s['head_A']['obs'][0]:+.3f}, {s['head_B']['obs'][0]:+.3f}）")

    print("[6] One by one of the statistical cores")
    gsA, geA = np.array([0.4, 0.2, 0.2]), np.array([0.5, 0.1, 0.0])
    gsB, geB = np.array([0.0, 0.6, 0.6]), np.array([0.0, 0.3, 0.7])
    st = pair_stats(gsA, geA, gsB, geB)
    # pick_A = {0}；pick_B = {1, 2}；Share = (0.2, 0.4, 0.4) → {1, 2}
    check(np.isclose(st["L_AB"], 0.5 - 0.0) and np.isclose(st["L_BA"], 0.5 - 0.05), "Cross-cutting losses")
    check(np.isclose(st["D_A"], 0.45) and np.isclose(st["D_B"], 0.0) and np.isclose(st["G"], 0.225), "Δ and G")
    check(st["rev"] == 1.0 and st["same"] == 0.0, "reversal and same-selection probabilities")
    check(np.isclose(multi_G([gsA, gsB], [geA, geB]), 0.225), "multi_G Consistency with two versions")
    fA, fB = np.array([0.1, 0.9, 0.5, 0.2]), np.array([0.3, 0.4, 0.8, 0.6])
    al = align(fA, fB, np.array([1, 2, 3]), np.array([0, 0, 0]), np.array([True, True, True]))
    check(np.allclose(al, [0.3, 0.8, 0.6, 0.4]), "Sort Alignment：B Net proceeds {0.5, 0.3, 0.1} by A reassign in order")

    print("[7] q7 path（and ds_gw Two-stage design of the same structure，Simulation assessment）")
    for name, pA, pB, want in (("identical", [0.6, 0.3, 0.1, 0.1], [0.6, 0.3, 0.1, 0.1], "zero"),
                               ("Conflict", [0.8, 0.2, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1], "pos")):
        d, tA, tB = _synthetic_q7(60, pA, pB, 0.0, 0.0)
        evs = {"q7": _sim_evals(d, tA, rng), "q14": _sim_evals(d, tB, rng)}
        rows, skipped = q7_stats(d, evs)
        nulls = {m: null_q7(d, evs, rng, n_sim=20, mode=m) for m in ("pooled", "aligned")}
        s = summarize(rows, nulls, None, rng)
        if want == "zero":
            check(skipped == 0 and abs(mp(s)[0]) < 0.04, f"{name}：Ĝ − Merge zero assumptions average ≈ 0（{mp(s)[0]:+.3f}）")
        else:
            check(mp(s)[1] > 0.1 and mp(s, m="aligned")[1] > 0.1, f"{name}：Ĝ − Zero hypothetical average Bottom bounds > 0.1（{fmt(mp(s))}）")
    print("\n=== share_gap selftest all passed ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="all", choices=["all", "alfw", "gw", "q7"])
    ap.add_argument("--root", default=str(config.ROOT), help="Ham out_alfw_b / out_ladder_alfw / out_confirm / out_ladder_gw / out_ds_gw Table of contents")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.env, args.root, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
