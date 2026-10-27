"""Analysis of consequences（Offline，exploratory；For registration see analysis plan.md 6.7）：Define it with another executor“Correct repair”，Hows the conclusion going to change?。

A. “which step to repair”Label of %1：use 7B / 14B / flash Success is defined separately（Best candidate for net gain per trajectory），
   and oracle of“Decisive steps”、Choice of judges、Planner Illustrative、7B greedy。Read：and 7B Probability of label consistency（Select one randomly in parallel rows）。
   - Compared with the same number of samples for the serial tags（Each side 5 runs），Noise cap = 7B two disjoint sets 5 Consistency rate between sub-samples；
   - The label that doesnt have to be renewed with 7B A discounted label comparison，Noise cap = 7B Consistency rate between two discounts。
B. Method ranking：Same selection，Use separately 7B、14B、flash Net proceeds of recovery from follow-up evaluation selected（Corresponds to“Which executor is the benchmark for counterfactual re-scoring?”）。
   Half for each executor（7B：Two discounts；14B / flash：10 Second split 5 + 5），use one half for method selection、The other half is for evaluation，swap roles and average，
   Ensure that evaluation does not intersect with choice。Read：Rankings under the three evaluations、Kendall τ（trajectory bootstrap interval）、the top-ranked method。

    python consequence.py --env gw      # gridworld Confirm the experiment + out_ladder_gw
    python consequence.py --env alfw    # ALFWorld stage B + out_ladder_alfw
output out_consequence/consequence_<env>.md / .json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import config

OUT = config.ROOT / "out_consequence"
EVALS = ("7B", "14B", "flash")


# ---------------------------------------------------------------- data
def load(env, root):
    import methods_gw
    import runner
    root = Path(root)
    if env == "gw":
        import confirm
        import ladder_gw
        import oracle
        out = root / "out_confirm"
        d = json.loads((out / "main_design.json").read_text(encoding="utf-8"))
        res = runner.load_results(out / "main_rollouts.jsonl")
        ladder_gw.STATE["ladder"] = root / "out_ladder_gw"
        lad = {"14B": ladder_gw.load("q14"), "flash": ladder_gw.load("qflash")}
        get = lambda name, b: lad[name][("S", b)]["results"]
        dep_key, gr_key = (confirm.DEPLOY, confirm.B_MAIN), (confirm.GREEDY, confirm.B_MAIN)
        feas = lambda x: 1.0 if oracle.feasible(x["snapshot"]) else 0.0
        slack = confirm._slack
        judges = methods_gw.load_judges(out)
    else:
        out = root / "out_alfw_b"
        d = json.loads((out / "main_design.json").read_text(encoding="utf-8"))
        res = runner.load_results(out / "main_rollouts.jsonl")
        lad = {n: runner.load_results(root / "out_ladder_alfw" / f"{q}.jsonl") for n, q in (("14B", "q14"), ("flash", "qflash"))}
        q = {"14B": "q14", "flash": "qflash"}
        get = lambda name, b: lad[name][(f"{q[name]}_T07", "L", b)]["results"]
        dep_key, gr_key = ("deploy_T07", "M"), ("llm_greedy", "M")
        feas = lambda x: float(x["oracle_F"])
        slack = lambda x: float(x["oracle_slack"])
        judges = methods_gw.load_judges(out)
    br = {}
    for b, x in d["branches"].items():
        info = {"det": x["deterministic"], "feas": feas(x), "slack": slack(x), "expert": bool(x.get("expert"))}
        if x["deterministic"] is None:
            r = np.array(res[dep_key + (b,)]["results"], float)
            h = len(r) // 2
            info["7B"] = [r[:h], r[h:]]
            g = res[gr_key + (b,)]
            info["greedy"] = g["successes"] / g["n"]
            for n in ("14B", "flash"):
                e = np.array(get(n, b), float)
                info[n] = [e[:len(e) // 2], e[len(e) // 2:]]
        else:
            info["greedy"] = float(x["deterministic"])
        br[b] = info
    trajs = [{"id": t["traj_id"], "positions": [{"t": p["t"], "orig": p["orig"], "alts": list(p["alts"]),
                                                 "decisive": bool(p.get("decisive"))} for p in t["positions"]]}
             for t in d["trajectories"]]
    return trajs, br, judges


# ---------------------------------------------------------------- Tools
def P(br, b, ex, split, k=None):
    """executor ex In the first place, split The success rate on the sample（k：Use only the front k items）。use the known value for deterministic branches。"""
    x = br[b]
    if x["det"] is not None:
        return float(x["det"])
    s = x[ex][split]
    s = s if k is None else s[:k]
    return float(s.mean())


def cands(t):
    return [(a, p["orig"], i) for i, p in enumerate(t["positions"]) for a in p["alts"]]


def argmax_set(items, scores):
    sc = np.asarray(scores, float)
    m = sc.max()
    return [it for it, s in zip(items, sc) if s >= m - 1e-12]


def zsig(ka, na, ko, no):
    import confirm
    return confirm._z_sig(ka, na, ko, no)


# ---------------------------------------------------------------- Selection（Use the first s Data selection）
def methods(env, judges):
    M = {}

    def deploy_k(k):
        return lambda t, br, s: argmax_set(cands(t), [P(br, a, "7B", s, k) - P(br, o, "7B", s, k) for a, o, _ in cands(t)])
    M["7B Deployment 10 runs"] = deploy_k(10)
    M["7B Deployment 3 runs"] = deploy_k(3)
    M["7B greedy 1 runs"] = lambda t, br, s: argmax_set(cands(t), [br[a]["greedy"] - br[o]["greedy"] for a, o, _ in cands(t)])

    def first_k_succ(br, b, s, k):
        x = br[b]
        return x["det"] * k if x["det"] is not None else float(x["7B"][s][:k].sum())

    def dover(t, br, s):
        acc = [c for c in cands(t) if first_k_succ(br, c[0], s, 3) >= 2]
        return acc or None                    # None：no repair（net gain 0）

    def causalflow(t, br, s):
        acc = [c for c in cands(t) if first_k_succ(br, c[0], s, 1) >= 1]
        return acc or None

    def agentracer(t, br, s):
        acc = [c for c in cands(t) if first_k_succ(br, c[0], s, 1) >= 1]
        if not acc:
            return None
        i0 = min(c[2] for c in acc)
        return [c for c in acc if c[2] == i0]

    def car(t, br, s):
        C = cands(t)
        sig = []
        for a, o, i in C:
            xa, xo = br[a], br[o]
            ka, na = (xa["det"], None) if xa["det"] is not None else (xa["7B"][s].sum(), len(xa["7B"][s]))
            ko, no = (xo["det"], None) if xo["det"] is not None else (xo["7B"][s].sum(), len(xo["7B"][s]))
            if zsig(ka, na, ko, no):
                sig.append(i)
        sub = [c for c in C if c[2] == max(sig)] if sig else C
        return argmax_set(sub, [P(br, a, "7B", s) - P(br, o, "7B", s) for a, o, _ in sub])
    M["DoVer：3 Secondary ≥ 2 Second-success is acceptance"] = dover
    M["CausalFlow：A single flip will be accepted"] = causalflow
    M["AgenTracer：First single successful position"] = agentracer
    M["CAR：At the end of the night"] = car

    def score(br, b):
        return br[b]["feas"] * 1000 + br[b]["slack"]
    M["oracle Separability"] = lambda t, br, s: argmax_set(cands(t), [br[a]["feas"] - br[o]["feas"] for a, o, _ in cands(t)])
    M["oracle + slack"] = lambda t, br, s: argmax_set(cands(t), [score(br, a) - score(br, o) for a, o, _ in cands(t)])
    if env == "alfw":
        def planner_earliest(t, br, s):
            ex = [c for c in cands(t) if br[c[0]]["expert"]]
            i0 = min(c[2] for c in ex)
            return [c for c in ex if c[2] == i0]
        M["Planner actions are in the earliest position"] = planner_earliest

    def shortlist_k(k):
        def f(t, br, s):
            C = []
            for i, p in enumerate(t["positions"]):
                sc = [score(br, a) for a in p["alts"]]
                best = max(sc)
                tied = [a for a, x in zip(p["alts"], sc) if x >= best - 1e-9]
                C.append((sorted(tied)[0], p["orig"], i))      # Coupled with the smallest number（Fixed、independent of outcomes）
            return argmax_set(C, [P(br, a, "7B", s, k) - P(br, o, "7B", s, k) for a, o, _ in C])
        return f
    M["Document Method SL(1, 10)"] = shortlist_k(10)
    for jn, lab in (("qwen7b_self", "7B Self-assessment decision"), ("qwen3.8-flash_think", "flash Rec（Read-only Log）"),
                    ("qwen3.8-flash_think_informed", "flash informed judge")):
        if jn in judges:
            def jf(t, br, s, jn=jn):
                b = judges[jn].get(t["id"])
                C = cands(t)
                hit = [c for c in C if c[0] == b]
                return hit or C                   # Invalid answer：By Random
            M[lab] = jf
    for ex in ("14B", "flash"):
        M[f"{ex} run using the deployed executor 5 runs"] = (lambda ex_: lambda t, br, s: argmax_set(
            cands(t), [P(br, a, ex_, s) - P(br, o, ex_, s) for a, o, _ in cands(t)]))(ex)
    M["random"] = lambda t, br, s: cands(t)
    return M


def value(br, chosen, ex, split):
    if chosen is None:
        return 0.0
    return float(np.mean([P(br, a, ex, split) - P(br, o, ex, split) for a, o, _ in chosen]))


# ---------------------------------------------------------------- A：Location Label
def pos_label(t, br, ex, split, k=None):
    best = {}
    for a, o, i in cands(t):
        g = P(br, a, ex, split, k) - P(br, o, ex, split, k)
        best[i] = max(best.get(i, -9), g)
    m = max(best.values())
    return {i for i, g in best.items() if g >= m - 1e-12}


def agree(A, B):
    return len(A & B) / (len(A) * len(B))


def part_a(env, trajs, br, judges, rng):
    rows = {}
    # Retract：Each side 5 runs
    def sub5(t, k_idx):
        """7B of 5 Sub-samples label：take disjoint samples from each of the two folds 5 runs（k_idx=0/1 It corresponds to two）。"""
        best = {}
        for a, o, i in cands(t):
            def p5(b):
                x = br[b]
                if x["det"] is not None:
                    return float(x["det"])
                allr = np.concatenate(x["7B"])
                return float(allr[k_idx * 5:(k_idx + 1) * 5].mean())
            best[i] = max(best.get(i, -9), p5(a) - p5(o))
        m = max(best.values())
        return {i for i, g in best.items() if g >= m - 1e-12}
    r7, r14, rf = [], [], []
    for t in trajs:
        L0, L1 = sub5(t, 0), sub5(t, 1)
        r7.append(agree(L0, L1))
        r14.append(np.mean([agree(sub5(t, kk), pos_label(t, br, "14B", h)) for kk in (0, 1) for h in (0, 1)]))
        rf.append(np.mean([agree(sub5(t, kk), pos_label(t, br, "flash", h)) for kk in (0, 1) for h in (0, 1)]))
    rows["continuation（Each side 5 runs）"] = {"7B additionally 5 runs（Noise cap）": r7, "14B 5 runs": r14, "flash 5 runs": rf}
    # A label that doesnt have to be renewed vs 7B A discount
    r_ceiling, other = [], {}
    for t in trajs:
        L = [pos_label(t, br, "7B", s) for s in (0, 1)]
        r_ceiling.append(agree(L[0], L[1]))
        labs = {}
        dec = {i for i, p in enumerate(t["positions"]) if p["decisive"]}
        if dec:
            labs["oracle Decisive steps" if env == "gw" else "Organisation（ALFWorld of“Decisive”Tag）"] = dec
        labs["7B greedy"] = {c[2] for c in argmax_set(cands(t), [br[a]["greedy"] - br[o]["greedy"] for a, o, _ in cands(t)])}
        for jn, lab in (("qwen3.8-flash_think", "flash Rec"), ("qwen3.8-flash_think_informed", "flash informed judge"),
                        ("qwen7b_self", "7B Self-assessment decision")):
            if jn in judges:
                b = judges[jn].get(t["id"])
                hit = {c[2] for c in cands(t) if c[0] == b}
                labs[lab] = hit or set(range(len(t["positions"])))
        if env == "alfw":
            labs["Planner Illustrative（Initial Location）"] = {0}
        for k, v in labs.items():
            other.setdefault(k, []).append(np.mean([agree(v, L[s]) for s in (0, 1)]))
    rows["requires no continuations（Yeah 7B A discounted label）"] = dict({"7B Another discount（Noise cap）": r_ceiling}, **other)
    return rows


# ---------------------------------------------------------------- B：Method ranking
def part_b(env, trajs, br, judges, n_boot=2000, seed=0):
    M = methods(env, judges)
    names = list(M)
    V = {ex: np.zeros((len(trajs), len(names))) for ex in EVALS}
    for ti, t in enumerate(trajs):
        for j, n in enumerate(names):
            for ex in EVALS:
                V[ex][ti, j] = np.mean([value(br, M[n](t, br, s), ex, 1 - s) for s in (0, 1)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(trajs), size=(n_boot, len(trajs)))

    def tau(x, y):
        c = d_ = 0
        for i, j in itertools.combinations(range(len(x)), 2):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            c += s > 0
            d_ += s < 0
        return (c - d_) / max(1, len(x) * (len(x) - 1) / 2)
    out = {"names": names, "mean": {ex: V[ex].mean(axis=0).tolist() for ex in EVALS}, "tau": {}}
    for ex in ("14B", "flash"):
        point = tau(V["7B"].mean(axis=0), V[ex].mean(axis=0))
        bt = [tau(V["7B"][ii].mean(axis=0), V[ex][ii].mean(axis=0)) for ii in idx[:500]]
        out["tau"][ex] = [point, float(np.quantile(bt, 0.05)), float(np.quantile(bt, 0.95))]
    return out


def run(env, root):
    trajs, br, judges = load(env, root)
    rng = np.random.default_rng(0)
    A = part_a(env, trajs, br, judges, rng)
    B = part_b(env, trajs, br, judges)
    title = {"gw": "gridworld Confirm the experiment", "alfw": "ALFWorld stage B"}[env]
    L = [f"# Analysis of consequences：{title}（{len(trajs)} trajectories；exploratory，For registration see analysis plan.md 6.7）", "",
         "## A. “which step to repair”Label with 7B Deployment of label consistency rate", "",
         "Label for each trajectory = The best place for the net gain；Consistency rate = Two labels（Select one randomly when side-by-side）Probability for pointing to the same location。"
         "trajectories in parentheses bootstrap 90% interval。", ""]
    rng2 = np.random.default_rng(1)

    def ci(x):
        x = np.asarray(x, float)
        m = x[rng2.integers(0, len(x), size=(2000, len(x)))].mean(axis=1)
        return f"{x.mean():.2f} [{np.quantile(m, 0.05):.2f}, {np.quantile(m, 0.95):.2f}]"
    for blk, rows in A.items():
        L += [f"**{blk}**", "", "| Label Source | and 7B Probability of label consistency |", "|---|---|"]
        for k, v in rows.items():
            L.append(f"| {k} | {ci(v)} |")
        L.append("")
    L += ["## B. Whos going to score it with?，How will the rankings change?", "",
          "Same selection；“7B Score”= Real benefits at deployment，“14B / flash Score”= If the benchmark uses these models for counter-fact replaying to judge repair,。"
          "All points are not interwoven with the data used in the method selection。Table shows the average net gain on the recovery selected（Rank in parentheses）。", "",
          "| Selection | 7B Score | 14B Score | flash Score |", "|---|---|---|---|"]
    ranks = {ex: (-np.array(B["mean"][ex])).argsort().argsort() + 1 for ex in EVALS}
    order = np.argsort(-np.array(B["mean"]["7B"]))
    for j in order:
        n = B["names"][j]
        L.append(f"| {n} | " + " | ".join(f"{B['mean'][ex][j]:+.3f}（{ranks[ex][j]}）" for ex in EVALS) + " |")
    L += ["", f"- Method ranking with 7B used for scoring Kendall τ：14B Score {B['tau']['14B'][0]:+.2f} "
              f"[{B['tau']['14B'][1]:+.2f}, {B['tau']['14B'][2]:+.2f}]；flash Score {B['tau']['flash'][0]:+.2f} "
              f"[{B['tau']['flash'][1]:+.2f}, {B['tau']['flash'][2]:+.2f}]",
          "- First in line：" + "；".join(f"{ex} Score → {B['names'][int(np.argmin(ranks[ex]))]}" for ex in EVALS), ""]
    OUT.mkdir(exist_ok=True)
    (OUT / f"consequence_{env}.json").write_text(json.dumps({"A": {k: {kk: float(np.mean(vv)) for kk, vv in v.items()}
                                                                      for k, v in A.items()}, "B": B},
                                                           ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"consequence_{env}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["gw", "alfw"])
    ap.add_argument("--root", default=str(config.ROOT), help="Ham out_confirm / out_ladder_gw / out_alfw_b / out_ladder_alfw Table of contents")
    args = ap.parse_args()
    run(args.env, args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
