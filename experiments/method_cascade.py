"""Methodology（Offline assessment）：Well use cheap environmental signals to sift out a few candidates，Well concentrate our deployment strategys own renewals on these candidates。

Motivation（From results already available）：“Whats the move?”Most of them can be sent to the environment（planner / oracle slack），
whereas“At what point?”Depends on the executor himself，Well have to do it by deploying the strategy。therefore：
    1. screening：Only the best environmental signal is left at each location m An alternative action（by oracle Separability × 1000 + slack，Absolute value [u]）；
    2. evaluation：A continuation of the deployment strategy for the sifting candidates and their original action branches only——Every one k runs（SL-k），or distributed on a rotational phase-out basis（SL-SH）；
    3. selection：Highest estimated net gain candidate。

Assessment of agreements（Same line of thought as the confirmation of the experiment，No bias）：
- gridworld Confirm the experiment（34 records）and ALFWorld stage B（30 records）：Deployment of the sample in half；The method is randomly sequenced from a drop（When the budget is not adequate
  Only the first few），Net proceeds of elected candidates with another discount；Average two-to-one call，Yeah R Average of secondary random sampling order。trajectories Cluster bootstrap report an interval。
- gridworld q7 Ji!（100 records，unscreened）：Use only methods S1 The first half（Each branch 5 runs），Evaluation with an independent and unbiased label（ds_gw.labels）。
- budget = Number of extensions of deployments actually used per trajectory（deterministic branches consume no rollout budget）。

**Development separate from testing**：Design of methods（m、Budget programme）only in gridworld Confirm the experimental comparison、Set it down；
No change after，reuse unchanged ALFWorld and q7 Ikegami!（analysis plan.md 6.6 Finalization of records）。

    python method_cascade.py --env gw          # Development：gridworld Confirm the experiment（out_confirm）
    python method_cascade.py --env alfw        # Leave the test：ALFWorld（out_alfw_b）
    python method_cascade.py --env pool        # Leave the test：gridworld q7 Ji!（out_ds_gw）
output out_method/method_<env>.md / .json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import config

OUT = config.ROOT / "out_method"
R_REP = 30                 # Number of repetitions of random sampling sequences
DELTA = 0.10


# ---------------------------------------------------------------- Data suitability
class Branch:
    __slots__ = ("det", "folds", "score", "greedy")

    def __init__(self, det, folds, score, greedy):
        self.det, self.folds, self.score, self.greedy = det, folds, score, greedy


def load_gw(out_dir):
    import confirm
    import oracle
    import runner
    d = json.loads((out_dir / "main_design.json").read_text(encoding="utf-8"))
    res = runner.load_results(out_dir / "main_rollouts.jsonl")
    B = {}
    for b, x in d["branches"].items():
        feas = 1.0 if oracle.feasible(x["snapshot"]) else 0.0
        score = feas * 1000 + confirm._slack(x)
        if x["deterministic"] is not None:
            B[b] = Branch(float(x["deterministic"]), None, score, float(x["deterministic"]))
        else:
            r = res[(confirm.DEPLOY, confirm.B_MAIN, b)]["results"]
            h = len(r) // 2
            g = res[(confirm.GREEDY, confirm.B_MAIN, b)]
            B[b] = Branch(None, [np.array(r[:h], float), np.array(r[h:], float)], score, g["successes"] / g["n"])
    trajs = [{"id": t["traj_id"], "positions": [{"t": p["t"], "orig": p["orig"], "alts": list(p["alts"])}
                                                 for p in t["positions"]]} for t in d["trajectories"]]
    return trajs, B, "crossfit"


def load_alfw(out_dir):
    import runner
    d = json.loads((out_dir / "main_design.json").read_text(encoding="utf-8"))
    res = runner.load_results(out_dir / "main_rollouts.jsonl")
    B = {}
    for b, x in d["branches"].items():
        score = float(x["oracle_F"]) * 1000 + float(x["oracle_slack"])
        if x["deterministic"] is not None:
            B[b] = Branch(float(x["deterministic"]), None, score, float(x["deterministic"]))
        else:
            r = res[("deploy_T07", "M", b)]["results"]
            h = len(r) // 2
            g = res[("llm_greedy", "M", b)]
            B[b] = Branch(None, [np.array(r[:h], float), np.array(r[h:], float)], score, g["successes"] / g["n"])
    trajs = [{"id": t["traj_id"], "positions": [{"t": p["t"], "orig": p["orig"], "alts": list(p["alts"])}
                                                 for p in t["positions"]]} for t in d["trajectories"]]
    return trajs, B, "crossfit"


def load_pool(out_dir, pool="q7", ex="q7"):
    import ds_gw
    ds_gw.STATE["out"] = out_dir
    d = json.loads(ds_gw.P("design", pool).read_text(encoding="utf-8"))
    ev = ds_gw.load_evals(ds_gw.P("evals", pool, ex))
    lab = ds_gw.labels(d, ev)
    B, LAB = {}, {}
    for b, x in d["branches"].items():
        sl = ds_gw._slack(x)
        score = (1.0 if sl > -100 else 0.0) * 1000 + sl
        if x["deterministic"] is not None:
            B[b] = Branch(float(x["deterministic"]), None, score, float(x["deterministic"]))
        else:
            s1 = ev[("S1", b)]["results"]
            g = ev[("G", b)]["results"]
            # The choice can only be used S1 The first half（S1a）；Evaluation with an unbiased label independent of it
            B[b] = Branch(None, [np.array(s1[:len(s1) // 2], float)], score, float(np.mean(g)))
    for a, v in lab.items():
        LAB[a] = v["gain"]
    trajs = [{"id": t["traj_id"], "positions": [{"t": p["t"], "orig": p["orig"], "alts": list(p["alts"])}
                                                 for p in t["positions"]]} for t in d["trajectories"]]
    return trajs, B, LAB


# ---------------------------------------------------------------- Methodology
def _mean(x, default=np.nan):
    return float(np.mean(x)) if len(x) else default


class Sampler:
    """Sampling in random order from a drop；Record the number of repeats that have been used。"""

    def __init__(self, B, fold, rng):
        self.B, self.fold, self.rng = B, fold, rng
        self.perm, self.used, self.n = {}, 0, {}

    def avail(self, b):
        br = self.B[b]
        return 0 if br.det is not None else len(br.folds[self.fold])

    def draw(self, b, k):
        """Put the branch on b Number of samples to be filled in k（No more than the total of this discount）。"""
        br = self.B[b]
        if br.det is not None:
            return
        if b not in self.perm:
            self.perm[b] = self.rng.permutation(len(br.folds[self.fold]))
            self.n[b] = 0
        k = min(k, len(br.folds[self.fold]))
        if k > self.n[b]:
            self.used += k - self.n[b]
            self.n[b] = k

    def est(self, b):
        br = self.B[b]
        if br.det is not None:
            return br.det
        n = self.n.get(b, 0)
        if n == 0:
            return np.nan
        return float(br.folds[self.fold][self.perm[b][:n]].mean())


def _cands(t):
    return [(a, p["orig"], i) for i, p in enumerate(t["positions"]) for a in p["alts"]]


def shortlist(t, B, m, rng):
    """Every position to the environment signal（Separability × 1000 + slack）Prior m An alternative action；Random break when sidelined。m=None _Other Organiser。"""
    out = []
    for i, p in enumerate(t["positions"]):
        alts = list(p["alts"])
        if m is None or m >= len(alts):
            out += [(a, p["orig"], i) for a in alts]
            continue
        sc = np.array([B[a].score for a in alts]) + rng.random(len(alts)) * 1e-6
        idx = np.argsort(-sc)[:m]
        out += [(alts[j], p["orig"], i) for j in idx]
    return out


def pick_best(cands, S):
    """Highest pool of candidates for estimated net gains（Column All，Average for evaluation）。"""
    g = np.array([S.est(a) - S.est(o) for a, o, _ in cands])
    g = np.where(np.isnan(g), -np.inf, g)
    best = g.max()
    return [c for c, x in zip(cands, g) if x >= best - 1e-12]


def m_uniform(t, B, S, rng, m=None, k=1):
    """（After filtering）Each candidate and his or her original branch of motion will be separate k Keep going。m=None Thats not filtered“Each candidate k runs”（k = All of it That is, the matching selection）。"""
    C = shortlist(t, B, m, rng)
    for a, o, _ in C:
        S.draw(a, k)
        S.draw(o, k)
    return pick_best(C, S)


def m_sh(t, B, S, rng, m=1, budget=60):
    """Filtering and phasing out（successive halving）：Each round distributes the budget on an average basis to the remaining candidates（Substitute branch with original action branch），Half of the estimated poor phase-out。"""
    C = shortlist(t, B, m, rng)
    rounds = max(1, math.ceil(math.log2(max(2, len(C)))))
    per_round = budget / rounds
    alive = list(C)
    n_cur = 0
    for r in range(rounds):
        pos = {o for _, o, _ in alive}
        n_br = len(alive) + len(pos)
        n_cur += max(1, int(per_round // max(1, n_br)))
        for a, o, _ in alive:
            S.draw(a, n_cur)
            S.draw(o, n_cur)
        if r < rounds - 1 and len(alive) > 1:
            g = np.array([S.est(a) - S.est(o) for a, o, _ in alive])
            keep = np.argsort(-np.where(np.isnan(g), -np.inf, g))[:max(1, math.ceil(len(alive) / 2))]
            alive = [alive[j] for j in sorted(keep)]
    return pick_best(alive, S)


def m_prior(t, B, S, rng):
    """Zero budget：Only by environmental signal（Gains across position relative to the original action：Poor solver × 1000 + Balance difference，as registered oracle + slack）。
    The filter inside the position is set in absolute terms（Same position as the original action，The two orders are identical）。"""
    C = _cands(t)
    sc = np.array([B[a].score - B[o].score for a, o, _ in C])
    return [c for c, x in zip(C, sc) if x >= sc.max() - 1e-9]


def m_greedy(t, B, S, rng):
    """Every branch continues with greedy 1 runs（budget = Random Branches）。"""
    C = _cands(t)
    g = np.array([B[a].greedy - B[o].greedy for a, o, _ in C])
    S.used += sum(1 for b in {x for a, o, _ in C for x in (a, o)} if B[b].det is None)
    return [c for c, x in zip(C, g) if x >= g.max() - 1e-12]


# ---------------------------------------------------------------- evaluation
def evaluate(trajs, B, mode, method, reps=R_REP, seed=0):
    """Back to every trajectories (Value, budget)。mode="crossfit"：One foreplay、Another discount；otherwise mode A signbox with no bias。"""
    vals, buds = [], []
    for ti, t in enumerate(trajs):
        v_acc, b_acc, n = 0.0, 0.0, 0
        folds = (0, 1) if mode == "crossfit" else (0,)
        for f in folds:
            for r in range(reps):
                rng = np.random.default_rng([seed, ti, f, r])
                S = Sampler(B, f, rng)
                chosen = method(t, B, S, rng)
                if mode == "crossfit":
                    ev = []
                    for a, o, _ in chosen:
                        pa = B[a].det if B[a].det is not None else B[a].folds[1 - f].mean()
                        po = B[o].det if B[o].det is not None else B[o].folds[1 - f].mean()
                        ev.append(pa - po)
                    v_acc += float(np.mean(ev))
                else:
                    v_acc += float(np.mean([mode[a] for a, _, _ in chosen]))
                b_acc += S.used
                n += 1
        vals.append(v_acc / n)
        buds.append(b_acc / n)
    return np.array(vals), np.array(buds)


def boot_ci(x, n_boot=2000, seed=1):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def run(env, out_dir=None):
    if env == "gw":
        trajs, B, mode = load_gw(Path(out_dir or config.ROOT / "out_confirm"))
        K = 20
    elif env == "alfw":
        trajs, B, mode = load_alfw(Path(out_dir or config.ROOT / "out_alfw_b"))
        K = 15
    else:
        trajs, B, mode = load_pool(Path(out_dir or config.ROOT / "out_ds_gw"))
        K = 5
    rows = {}

    def add(name, method, reps=R_REP):
        v, b = evaluate(trajs, B, mode, method, reps=reps)
        rows[name] = (v, b)

    ks = [k for k in (1, 2, 3, 5, 10, 15, 20) if k <= K]
    for k in ks:
        add(f"all candidates × {k} runs", lambda t, B_, S, rng, k=k: m_uniform(t, B_, S, rng, None, k), reps=1 if k == K else R_REP)
    for m in (1, 2):
        for k in ks:
            add(f"screening m={m} × {k} runs", lambda t, B_, S, rng, k=k, m=m: m_uniform(t, B_, S, rng, m, k))
    n_br = np.mean([len({x for a, o, _ in _cands(t) for x in (a, o) if B[x].det is None}) for t in trajs])
    for mult in (1, 2, 3, 5):
        bud = mult * n_br
        for m in (1, 2):
            add(f"screening m={m} + Phase-out，budget {mult}×", lambda t, B_, S, rng, m=m, bud=bud: m_sh(t, B_, S, rng, m, bud))
    add("use environment signals only：oracle + slack（Zero budget）", m_prior, reps=R_REP)
    add("greedy 1 runs", m_greedy, reps=1)
    ref = f"all candidates × {K} runs"
    v_ref = rows[ref][0]
    res = {"env": env, "n_traj": len(trajs), "branches_per_traj": float(n_br), "ref": ref, "rows": {}}
    L = [f"# Methodology（Offline assessment）：{ {'gw': 'gridworld Confirm the experiment（Development Set）', 'alfw': 'ALFWorld stage B（Leave the test）', 'pool': 'gridworld q7 Ji!（Leave the test）'}[env] }", "",
         f"{len(trajs)} trajectories；Average per trajectory {n_br:.1f} A branch that needs to be renewed（Substitution + original action）。References = {ref}（That is, the matching selection）。"
         f"Value = Net deployment gain from selected restorations；loss = References − Thats the method（90% Intersection is a trajectories bootstrap，Random Sampling Order Repeated {R_REP} Lower Average）。", "",
         "| Methodology | Budget for the renewal of each trajectory | budget / References | Net deployment gains | Relative reference losses [90%] |", "|---|---:|---:|---:|---|"]
    b_ref = rows[ref][1].mean()
    for name, (v, b) in rows.items():
        loss = boot_ci(v_ref - v)
        res["rows"][name] = {"budget": float(b.mean()), "value": float(v.mean()), "loss": loss}
        L.append(f"| {name} | {b.mean():.0f} | {b.mean() / max(b_ref, 1e-9):.2f} | {v.mean():+.3f} | "
                 + ("—" if name == ref else f"{loss[0]:+.3f} [{loss[1]:+.3f}, {loss[2]:+.3f}]") + " |")
    # Matching（Same trajectories）：Difference = The former − The latter Net deployment gains，> 0 That means the former is better
    k_lo = 3 if 3 <= K else K
    pairs = [(f"screening m=1 × {k_lo} runs", "greedy 1 runs", "Budget Close：After filtering 3 runs vs greedy"),
             (f"screening m=1 × {k_lo} runs", "all candidates × 1 runs", "Budget Close：After filtering 3 runs vs All candidates 1 runs"),
             (f"screening m=1 × {min(10, K)} runs", "use environment signals only：oracle + slack（Zero budget）", "Screening for follow-up vs use environment signals only")]
    L += ["", "**Matching**（Same trajectories；Difference = The former − The latter，> 0 That means the former is better）：", "",
          "| comparison | The former / Budget of the latter | Difference [90%] |", "|---|---|---|"]
    res["pairs"] = {}
    for a_, b_, lab in pairs:
        if a_ in rows and b_ in rows:
            c = boot_ci(rows[a_][0] - rows[b_][0])
            res["pairs"][lab] = c
            L.append(f"| {lab} | {rows[a_][1].mean():.0f} / {rows[b_][1].mean():.0f} | {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}] |")
    # Reached reference 95% Minimum budget for value
    L += ["", "**Reaching reference value 95% Minimum budget required**（Only look in top table configuration）：", ""]
    for fam, pref in (("unscreened（all candidates × k）", "all candidates"), ("screening m=1 × k", "screening m=1 ×"), ("screening m=2 × k", "screening m=2 ×"),
                      ("screening m=1 + Phase-out", "screening m=1 +"), ("screening m=2 + Phase-out", "screening m=2 +")):
        ok = [(r["budget"], n) for n, r in res["rows"].items() if n.startswith(pref) and r["value"] >= 0.95 * v_ref.mean()]
        L.append(f"- {fam}：" + (f"{min(ok)[0]:.0f} runs（{min(ok)[1]}；References {min(ok)[0] / b_ref:.0%}）" if ok else "Not achieved in the above table"))
    OUT.mkdir(exist_ok=True)
    (OUT / f"method_{env}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"method_{env}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["gw", "alfw", "pool"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    run(args.env, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
