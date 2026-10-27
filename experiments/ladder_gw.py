"""Gridworld：A new model will be used to confirm the same branch of the experiment（Followers ladder）+ Reliance pre-screening by the executor。

at out_confirm on the main experimental design（34 trajectories、All Random Branches），Let another model（e.g Qwen2.5-14B）as a continuation executor：
    S  Every Random Branch N_EVAL runs（temperature 0.7，The same prompts and decoding as deployment conditions）
    G  greedy 1 runs
Analysis（Same selection and evaluation code as the validation experiment confirm._selection_values）：
  1. Followers ladder：Choose restoration based on the models repeat results，evaluate under the deployed executor 7B use，Relative Match Selection（7B 20 runs）Losses；
  2. Reliance pre-screening by the executor：Sorting of net gains from each trajectories，7B Consistency with the model，Contrast 7B Consistency between two separate samples of their own
     （Both sides 10 Subsamples，Comparable）。Internal coherence − Cross-model coherence of 90% Lower Inter-District Border > 0，The sorting of restoration value changes with the implementer。

    python ladder_gw.py --stage run --evaluator q14      # approximately 1–1.5 hours（14B）
    python ladder_gw.py --stage smoke --evaluator qflash # Qwen3.8-flash（API）smoke test：30 branches × 2 runs，Estimated full cost and time
    python ladder_gw.py --stage run --evaluator qflash   # Qwen3.8-flash Full（does not occupy GPU；QWEN_API_KEY）
    python ladder_gw.py --stage analyze                  # out_ladder_gw/ladder_report.md
input（Confirm the design of the experiment、7B Samples deployed、Outcome of the decision）Read From out_confirm/；The output of the successors staircase is left on its own out_ladder_gw/：
    out_ladder_gw/<model>.jsonl（e.g q14.jsonl、qflash.jsonl）、<model>_smoke.json、ladder_report.md
Different servers running out <model>.jsonl Cuff the same one out_ladder_gw/ then analyze Merge。
    python ladder_gw.py --stage selftest                 # Noise oracle Replace Model，Check process only
Model Path：DS_Q14 / DS_Q32 Environmental variables（default /root/autodl-tmp/models/Qwen2.5-14B-Instruct etc）。
When you cant put a single card down,（14B Requirements 32GB This is visible、32B Requirements 70GB Above）：LADDER_TP=2 Use two cards in parallel。
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import config
import confirm
import runner
from policies import derive_seed

EVALUATORS = {"q14": os.environ.get("DS_Q14", "/root/autodl-tmp/models/Qwen2.5-14B-Instruct"),
              "q32": os.environ.get("DS_Q32", "/root/autodl-tmp/models/Qwen2.5-32B-Instruct"),
              "qflash": "api:" + os.environ.get("LADDER_QFLASH_MODEL", "qwen3.8-flash"),
              # hosted on the Qwen platform Qwen2.5（Nothing GPU Server Time；Same weight as local，Different end of service）：
              # q7api For calibration“API Trusteeship vs Local vLLM”Consistency（a smoke test is sufficient），q14api / q32api For the staircases of the same family
              "q7api": "api:" + os.environ.get("LADDER_Q7API_MODEL", "qwen2.5-7b-instruct"),
              "q14api": "api:" + os.environ.get("LADDER_Q14API_MODEL", "qwen2.5-14b-instruct"),
              "q32api": "api:" + os.environ.get("LADDER_Q32API_MODEL", "qwen2.5-32b-instruct"),
              "q27": "api:" + os.environ.get("LADDER_Q27_MODEL", "qwen3.8-27b")}
API_CHUNK = 320     # API How many more of each batch? rollout（Ends the downboard and prints the accumulated costs in each batch）
N_EVAL = 10
SUB = 10            # When comparing dependency 7B Size of each sample（and N_EVAL identical，Comparable）；40 Sub-deployment sample split 4 disjoint sets
TP = int(os.environ.get("LADDER_TP", "1"))
STATE = {"mock": False, "out": confirm.OUT, "ladder": config.ROOT / "out_ladder_gw"}


def log(msg):
    print(msg, flush=True)


def path_of(name):
    return STATE["ladder"] / f"{name}.jsonl"


def migrate_legacy():
    """The old version wrote the results of the steps out_confirm/ladder_<model>.jsonl：If you find it, move it out_ladder_gw/<model>.jsonl（only if absent from the new location）。"""
    import shutil
    STATE["ladder"].mkdir(parents=True, exist_ok=True)
    for n in EVALUATORS:
        for old, new in ((STATE["out"] / f"ladder_{n}.jsonl", path_of(n)),
                         (STATE["out"] / f"ladder_{n}_smoke.json", STATE["ladder"] / f"{n}_smoke.json")):
            if old.exists() and not new.exists():
                shutil.move(str(old), str(new))
                log(f"[Collate] {old} -> {new}")
    old = STATE["out"] / "ladder_report.md"
    if old.exists():
        shutil.move(str(old), str(STATE["ladder"] / "ladder_report_old.md"))
        log(f"[Collate] {old} -> {STATE['ladder'] / 'ladder_report_old.md'}（Old report，Restart analyze Itll generate a new one）")


def load(name):
    out = {}
    p = path_of(name)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(r["phase"], r["branch_id"])] = r
    return out


def make_policy(name):
    if STATE["mock"]:
        from policies import NoisyOraclePolicy
        return NoisyOraclePolicy(0.3), "mock"
    if EVALUATORS[name].startswith("api:"):
        from api_policy import APIAgentPolicy
        pol = APIAgentPolicy(EVALUATORS[name][4:], temperature=config.TEMPERATURE)
        return pol, pol.cache_id
    from policies import make_seeded_llm_class
    if TP > 1:
        # model.AgentPolicy No length parallel parameters accepted；No, no, no, no model.py（Its Hash is part of the signature that has been saved），Replace only in this process vllm.LLM
        import functools
        import vllm
        vllm.LLM = functools.partial(vllm.LLM, tensor_parallel_size=TP)
    pol = make_seeded_llm_class()(temperature=config.TEMPERATURE, model_dir=EVALUATORS[name])
    return pol, pol.cache_id


def run(name):
    migrate_legacy()
    d = json.loads((STATE["out"] / "main_design.json").read_text(encoding="utf-8"))
    done = load(name)
    bad = {r["design_id"] for r in done.values()} - {d["design_id"]}
    if bad:
        sys.exit(f"{path_of(name)} contains records from a different design；Please change --out-dir or delete the document")
    nst = sorted(b for b, x in d["branches"].items() if x["deterministic"] is None)
    if all((ph, b) in done for ph in ("S", "G") for b in nst):
        log(f"[{name}] All completed")
        return
    pol, pid = make_policy(name)
    if hasattr(pol, "preflight"):
        outs = pol.preflight()
        log(f"[{name}] Interface check through（temperature {pol.temperature} and 0 Theyre all accepted）：{[o[:40] for o in outs]}；Together {pol.workers}")
    total = sum(nn for ph, nn in (("S", N_EVAL), ("G", 1)) for b in nst if (ph, b) not in done)   # By rollout Number of articles
    finished, t_all = 0, time.time()
    runner._repair_tail(path_of(name))
    with path_of(name).open("a", encoding="utf-8") as f:
        for phase, n, temp in (("S", N_EVAL, config.TEMPERATURE), ("G", 1, 0.0)):
            todo = [b for b in nst if (phase, b) not in done]
            log(f"[{name} {phase}] Stay {len(todo)} branches × {n} runs")
            per = max(1, (API_CHUNK if hasattr(pol, "usage") else config.ROLLOUT_CHUNK) // n)
            t0, skipped = time.time(), 0
            for i in range(0, len(todo), per):
                chunk = todo[i:i + per]
                snaps, seeds, owner = [], [], []
                for b in chunk:
                    for k in range(n):
                        snaps.append(d["branches"][b]["snapshot"])
                        seeds.append(derive_seed("ladder", name, phase, b, k))
                        owner.append(b)
                if not STATE["mock"]:
                    pol.temperature, pol.escape_temperature = temp, None
                stats = pol.run_rollouts(snaps, seeds)
                res, err = {}, set()
                for b, st in zip(owner, stats):
                    res.setdefault(b, []).append(bool(st["success"]))
                    if st.get("error"):
                        err.add(b)
                for b in chunk:
                    if b in err:                 # API Call error branch not to write（It cant be considered a failure），complete on rerun
                        skipped += 1
                        continue
                    f.write(json.dumps({"evaluator": name, "phase": phase, "branch_id": b, "n": n,
                                        "results": res[b], "policy": pid, "design_id": d["design_id"]}) + "\n")
                f.flush()
                finished += len(chunk) * n
                el = (time.time() - t_all) / 60
                msg = (f"  {min(i + per, len(todo))}/{len(todo)}，{(time.time() - t0) / 60:.1f} minutes"
                       f"（all {finished}/{total} records rollout，estimated time remaining {el / finished * (total - finished):.0f} minutes）")
                if hasattr(pol, "usage"):
                    from api_policy import cost_yuan
                    u = pol.usage
                    msg += (f"；Cumulative call {u['calls']}，Cache hit {u['cached_tokens'] / max(1, u['prompt_tokens']):.0%}，"
                            f"Cost estimates ¥{cost_yuan(u):.1f}，Try again {u['retries']}，Error branch {skipped}")
                log(msg)
            if skipped:
                log(f"[{name} {phase}] {skipped} branches have API Call Error，not written；The same order will be replaced by the same order")
    left = sum(1 for ph in ("S", "G") for b in nst if (ph, b) not in load(name))
    log(f"[{name}] This operation is over，Time {(time.time() - t_all) / 60:.0f} minutes；" +
        ("All branches completed。Next：python ladder_gw.py --stage analyze" if left == 0
         else f"also {left} items（stage, branch）incomplete，The same order will be replaced by the same order"))


def smoke(name, n_branch=30, n_roll=2):
    """API continuation-executor smoke test：Fixed smoke n_branch Random branch × n_roll runs（temperature 0.7），Save the result、Do not enter any analysis。
    report：Every single one rollout Number of calls、token、Cache hit、Costs、Time delay，and 7B Comparison of success rates in the same batch of branches，And the full cost and time estimate。"""
    import random
    migrate_legacy()
    d = json.loads((STATE["out"] / "main_design.json").read_text(encoding="utf-8"))
    res7 = runner.load_results(STATE["out"] / "main_rollouts.jsonl")
    nst = sorted(b for b, x in d["branches"].items() if x["deterministic"] is None)
    pick = random.Random(0).sample(nst, min(n_branch, len(nst)))
    pol, pid = make_policy(name)
    snaps = [d["branches"][b]["snapshot"] for b in pick for _ in range(n_roll)]
    seeds = [derive_seed("ladder_smoke", name, b, k) for b in pick for k in range(n_roll)]
    t0 = time.time()
    stats = pol.run_rollouts(snaps, seeds)
    wall = time.time() - t0
    ok = [s for s in stats if not s.get("error")]
    p_eval = np.mean([s["success"] for s in ok]) if ok else float("nan")
    p7 = np.mean([np.mean(res7[(confirm.DEPLOY, confirm.B_MAIN, b)]["results"]) for b in pick])
    n_full = len(nst) * (N_EVAL + 1)
    L = [f"[smoke {name}] {len(pick)} branches × {n_roll} runs = {len(stats)} records rollout，Time {wall / 60:.1f} minutes，Error {len(stats) - len(ok)} records",
         f"  success rate：{name} {p_eval:.3f}；7B On the same branch {p7:.3f}",
         f"  Average number of steps {np.mean([s['steps'] for s in ok]):.1f}，Unsolved Actions {sum(s['malformed'] for s in ok)} runs"]
    summary = {"name": name, "policy": pid, "n_rollouts": len(stats), "errors": len(stats) - len(ok), "wall_s": wall,
               "success_eval": p_eval, "success_7b_same_branches": p7}
    if hasattr(pol, "usage"):
        from api_policy import cost_yuan
        u = pol.usage
        c = cost_yuan(u)
        per_roll = c / max(1, len(stats))
        L += [f"  call {u['calls']} runs（Every single one rollout {u['calls'] / max(1, len(stats)):.1f}），input {u['prompt_tokens'] / max(1, u['calls']):.0f} "
              f"token/runs，including cache hits {u['cached_tokens'] / max(1, u['prompt_tokens']):.0%}；output {u['completion_tokens'] / max(1, u['calls']):.0f} token/runs，"
              f"Interrupted by length {u['length_cut']} runs；Average time delay {u['seconds_in_calls'] / max(1, u['calls']):.1f} seconds/runs；Try again {u['retries']} runs",
              f"  estimated smoke-test cost ¥{c:.2f}（Well see about the unit price api_policy.cost_yuan，Based on platform billing）",
              f"  Full forecast（{len(nst)} branches × {N_EVAL} runs + greedy 1 runs = {n_full} records rollout，Together {pol.workers}）："
              f"approximately ¥{per_roll * n_full:.0f}，approximately {wall / max(1, len(stats)) * n_full / 3600:.1f} hours"]
        L += ["  Original Output Example："] + [f"    {x[:200]!r}" for x in pol.samples[:2]]
        summary.update(usage=u, cost_yuan=c, projected_cost_yuan=per_roll * n_full,
                       projected_hours=wall / max(1, len(stats)) * n_full / 3600)
    (STATE["ladder"] / f"{name}_smoke.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                                          encoding="utf-8")
    log("\n".join(L))
    return summary


class LadderExtra:
    """copy“Choosing results by a model”The selection rules are being analysed to confirm the experiment（handled identically to the greedy rule：Select Fixed，Re-sampling of evaluations only）。"""

    def __init__(self, evals):
        self.evals = evals
        self.names = [x for n in evals for x in (f"eval:{n}", f"eval3:{n}", f"evalG:{n}")]

    def __call__(self, traj, cands, raw, R, Rall, col, br):
        out = {}
        for n, ev in self.evals.items():
            def p(b, phase, k=None):
                if br[b]["deterministic"] is not None:
                    return br[b]["deterministic"]
                r = ev[(phase, b)]["results"]
                r = r if k is None else r[:k]
                return sum(r) / len(r)
            for nm, phase, k in ((f"eval:{n}", "S", None), (f"eval3:{n}", "S", 3), (f"evalG:{n}", "G", None)):
                sc = np.array([p(a, phase, k) - p(o, phase, k) for a, o, *_ in cands])
                best = (np.abs(sc - sc.max()) <= 1e-12).astype(float)
                out[nm] = (Rall * best).sum(axis=-1) / best.sum()
        return out


def _rank(x):
    """Average（Parallel Average）；success rate is 0.1 Multipliers，ties are common。"""
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def dependence(d, res, ev, n_deploy=None):
    """per trajectory：Sorting of net proceeds from candidatures。7B Samples of deployments（gridworld 40 runs，ALFWorld 30 runs）split into disjoint sets 10 The next one：
    Internal coherence = 6 Right between them Spearman Average；Cross-model coherence = 4 Separated from the model 10 The second one Spearman Average。
    On both sides 10 runs，The same noise level；If the executor does not affect the sorting，The two expectations are close。"""
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)   # All candidates are on the same trajectories：No correlation factor defined，As nan
    rows = []
    br = d["branches"]
    n_deploy = n_deploy or confirm.N_DEPLOY
    nsub = n_deploy // SUB
    for t in d["trajectories"]:
        cands = [(a, p["orig"]) for p in t["positions"] for a in p["alts"]]

        def p7(b, lo, hi):
            if br[b]["deterministic"] is not None:
                return br[b]["deterministic"]
            r = res[(confirm.DEPLOY, confirm.B_MAIN, b)]["results"][lo:hi]
            return sum(r) / len(r)

        def pe(b):
            if br[b]["deterministic"] is not None:
                return br[b]["deterministic"]
            r = ev[("S", b)]["results"]
            return sum(r) / len(r)
        g7 = [[p7(a, i * SUB, (i + 1) * SUB) - p7(o, i * SUB, (i + 1) * SUB) for a, o in cands] for i in range(nsub)]
        ge = [pe(a) - pe(o) for a, o in cands]
        within = np.nanmean([_spearman(g7[i], g7[j]) for i in range(nsub) for j in range(i + 1, nsub)])
        cross = np.nanmean([_spearman(g7[i], ge) for i in range(nsub)])
        all7 = np.array([p7(a, 0, n_deploy) - p7(o, 0, n_deploy) for a, o in cands])
        rows.append({"traj": t["traj_id"], "within7": float(within), "cross": float(cross),
                     "same_best": float(np.argmax(all7) == np.argmax(ge))})
    return rows


def analyze():
    migrate_legacy()
    d = json.loads((STATE["out"] / "main_design.json").read_text(encoding="utf-8"))
    res = runner.load_results(STATE["out"] / "main_rollouts.jsonl")
    evals = {}
    for n in EVALUATORS:
        ev = load(n)
        nst = [b for b, x in d["branches"].items() if x["deterministic"] is None]
        if ev and all(("S", b) in ev and ("G", b) in ev for b in nst):
            evals[n] = ev
        elif ev:
            log(f"[analyze] {n} incomplete（{len(ev)} Records），skip")
    if not evals:
        log("[analyze] complete continuation results are unavailable")
        return None
    import methods_gw
    judges = methods_gw.load_judges(STATE["out"])
    ej, el = methods_gw.Extra(judges), LadderExtra(evals)

    class _Both:
        names = ej.names + el.names

        def __call__(self, **kw):
            o = ej(**kw)
            o.update(el(**kw))
            return o
    V, names, contrast = confirm._selection_values(d, res, "net", confirm.N_BOOT, 0, extra=_Both())
    f = lambda c: (f"{c['point']:+.3f} [{c['ci90'][0]:+.3f}, {c['ci90'][1]:+.3f}]"
                   f"（Correct [{c['ci90_bc'][0]:+.3f}, {c['ci90_bc'][1]:+.3f}]）")
    L = ["# Gridworld：Change the pattern（Followers ladder）Reliance pre-screening with executors", "",
         f"Confirm the same experiment {len(d['trajectories'])} trajectories、Same group of candidates；Rehabilitate by the results of the repeat of the models，evaluate under the deployed executor 7B use。"
         "loss = Match Selection（7B 20 runs）− This way；90% Inter-trajectories are trajectories + parameters bootstrap，Brackets for deviation correction（analysis plan.md 6.3c）。", "",
         "| Selection | Net deployment gains | Relatively matched loss [90%] |", "|---|---:|---|",
         f"| Match：7B 20 runs | {np.nanmean(V['matched'][0]):+.3f} | — |",
         f"| 7B Deployment 10 runs | {np.nanmean(V['k10'][0]):+.3f} | {f(contrast('matched', 'k10'))} |",
         f"| 7B Deployment 3 runs | {np.nanmean(V['k3'][0]):+.3f} | {f(contrast('matched', 'k3'))} |",
         f"| 7B greedy | {np.nanmean(V['greedy'][0]):+.3f} | {f(contrast('matched', 'greedy'))} |"]
    for n in evals:
        for nm, lab in ((f"eval:{n}", f"{n} Sample {N_EVAL} runs"), (f"eval3:{n}", f"{n} Sample 3 runs"), (f"evalG:{n}", f"{n} greedy")):
            L.append(f"| {lab} | {np.nanmean(V[nm][0]):+.3f} | {f(contrast('matched', nm))} |")
    for j, lab in (("qwen3.8-flash_think", "Qwen3.8-flash Rec（Read-only Log）"), ("qwen7b_self", "7B Self-assessment decision")):
        if f"judge:{j}" in names:
            L.append(f"| {lab} | {np.nanmean(V[f'judge:{j}'][0]):+.3f} | {f(contrast('matched', f'judge:{j}'))} |")
    L += [f"| oracle | {np.nanmean(V['oracle'][0]):+.3f} | {f(contrast('matched', 'oracle'))} |",
          f"| oracle + slack | {np.nanmean(V['oracle_slack'][0]):+.3f} | {f(contrast('matched', 'oracle_slack'))} |",
          f"| random（positive control） | {np.nanmean(V['random'][0]):+.3f} | {f(contrast('matched', 'random'))} |", "",
          "Same thing 10 Keep going，use 7B select using the deployed executor vs Use another model（all assigned to 7B use）。Difference > 0 Expressing the estimated loss of deployment gains due to a change of implementer；"
          "7B its own 10 Next is cross-compartment（Another discounted evaluation），Another model is chosen in all 40 Sub-evaluation，both are unbiased。", "",
          "| Contrast | Difference [90%] |", "|---|---|"]
    for n in evals:
        L.append(f"| 7B 10 runs − {n} 10 runs | {f(contrast('k10', f'eval:{n}'))} |")
        L.append(f"| 7B 3 runs − {n} 3 runs | {f(contrast('k3', f'eval3:{n}'))} |")
    L += ["", "**Read Log vs run using the deployed executor**（Same model；Difference = The former − The latter）In comparison with：", "", "| Contrast | Difference [90%] |", "|---|---|"]
    if "qflash" in evals and "judge:qwen3.8-flash_think" in names:
        L.append(f"| flash Read Log − flash run using the deployed executor 10 runs | {f(contrast('judge:qwen3.8-flash_think', 'eval:qflash'))} |")
    if "judge:qwen7b_self" in names:
        L.append(f"| 7B run using the deployed executor 10 runs − 7B Read Log | {f(contrast('k10', 'judge:qwen7b_self'))} |")
    for n in evals:
        L.append(f"| {n} run using the deployed executor 10 runs − oracle | {f(contrast(f'eval:{n}', 'oracle'))} |")
    L.append("")
    for n in evals:
        c = contrast("k10", f"eval:{n}")
        lo, hi = c["ci90"]
        if lo > 0:
            msg = (f"**{n}：Visible executor dependency。** use {n} The results of the run-off were chosen for repair，7B Ratio of deployment gains to 7B lower than matched selection {c['point']:+.3f}"
                   f"（Lower Inter-District Border {lo:+.3f} > 0）。executor-specific repair-value labels are necessary。")
        elif hi < 0:
            msg = (f"**{n}：Reverse direction。** use {n} selection is unexpectedly better than 7B Myself 10 selection with this many samples（upper bound {hi:+.3f} < 0）："
                   "More powerful models may have a lower noise，This in itself doesnt mean the executor has nothing to do with it，It needs to be read in conjunction with the sorting below。")
        else:
            msg = (f"**{n}：Difference not detected**（{c['point']:+.3f}，interval [{lo:+.3f}, {hi:+.3f}] Ham 0）。"
                   + ("upper bound < 0.05：switching executor has little cost in this setting，The executor condition label can only be used as additional information to the data set。"
                      if hi < 0.05 else "Wider area，No meaningful difference can be ruled out；See Sort Consistency，Add samples, if necessary。"))
        L += [msg, ""]
    L += ["The main reading is the selection loss comparison above（No bias，Directly corresponding deployment gains）。The next order of consistency is a description：The two models have different success rates，"
          "The two noise levels are different，therefore“Internal vs Cross-model”only approximately comparable。", ""]
    rng = np.random.default_rng(0)
    for n, ev in evals.items():
        rows = dependence(d, res, ev)
        gap = np.array([r["within7"] - r["cross"] for r in rows])
        gap = gap[~np.isnan(gap)]
        m = gap[rng.integers(0, len(gap), size=(2000, len(gap)))].mean(axis=1)
        cap7 = np.mean([np.mean(res[(confirm.DEPLOY, confirm.B_MAIN, b)]["results"]) for b in d["branches"]
                        if d["branches"][b]["deterministic"] is None])
        cape = np.mean([sum(v["results"]) / v["n"] for (ph, b), v in ev.items() if ph == "S"])
        L += [f"## Reliance pre-screening by the executor：7B vs {n}", "",
              f"- Average success rate for random branches（Crude indicators of capacity）：7B {cap7:.3f}；{n} {cape:.3f}",
              f"- Consistency in ranking of candidate net proceeds（Each {SUB} Subsamples，per trajectory Spearman Averaged after averaged trajectories；"
              f"7B Internal {confirm.N_DEPLOY // SUB} pairwise across disjoint sample sets）：7B Internal "
              f"{np.nanmean([r['within7'] for r in rows]):.3f}；7B vs {n} {np.nanmean([r['cross'] for r in rows]):.3f}",
              f"- Internal − Cross-model：{gap.mean():+.3f} [{np.quantile(m, 0.05):+.3f}, {np.quantile(m, 0.95):+.3f}]"
              "（lower bound > 0：Sort does change with the executor；Intersection 0：cannot distinguish from sampling noise）",
              f"- 7B（40 runs）And the best candidate {n} Best proportion in the eye：{np.mean([r['same_best'] for r in rows]):.3f}", ""]
    out = STATE["ladder"] / "ladder_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "smoke", "analyze", "selftest"])
    ap.add_argument("--smoke-n", type=int, default=30)
    ap.add_argument("--evaluator", choices=list(EVALUATORS))
    ap.add_argument("--out-dir", type=Path, default=None, help="Confirm the output of the experiment（input），default out_confirm")
    ap.add_argument("--ladder-dir", type=Path, default=None, help="Step Output，default out_ladder_gw")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if args.out_dir:
        STATE["out"] = args.out_dir.resolve()
    if args.ladder_dir:
        STATE["ladder"] = args.ladder_dir.resolve()
    STATE["mock"] = args.mock or args.stage == "selftest"
    if args.stage == "selftest":
        import shutil
        import tempfile
        src = Path(args.out_dir or confirm.OUT)
        if not (src / "main_design.json").exists():
            sys.exit(f"{src} confirmatory experiment data are missing under main_design.json；required by self-tests（Or use it --out-dir Point to a simulation of confirmed experimental output）")
        tmp = Path(tempfile.mkdtemp(prefix="ladder_"))
        try:
            for n in ("main_design.json", "main_rollouts.jsonl"):
                shutil.copy(src / n, tmp / n)
            STATE["out"], STATE["ladder"] = tmp, tmp / "ladder"
            run("q14")
            lines = path_of("q14").read_text(encoding="utf-8").splitlines(keepends=True)
            full = {(json.loads(x)["phase"], json.loads(x)["branch_id"]): json.loads(x)["results"] for x in lines}
            # Simulation aborted：Get rid of the last one 30 records，Ill leave you half a line
            path_of("q14").write_text("".join(lines[:-30]) + lines[-30][:25], encoding="utf-8")
            run("q14")
            again = path_of("q14").read_text(encoding="utf-8").splitlines()
            rec = {(json.loads(x)["phase"], json.loads(x)["branch_id"]): json.loads(x)["results"] for x in again}
            ok = len(again) == len(lines) and rec == full
            print(("  ✓ " if ok else "  ✗ ") + f"Breaking up the follow-up：{len(again)}/{len(lines)} records，results match an uninterrupted run" )
            run("q14")
            same = len(path_of("q14").read_text(encoding="utf-8").splitlines()) == len(lines)
            print(("  ✓ " if same else "  ✗ ") + "Do not rerun and write again")
            ok = ok and same
            L = analyze()
            ok = ok and L is not None and any("q14 Sample" in x for x in L)
            print("[ladder selftest] " + ("all passed" if ok else "failures detected"))
            return 0 if ok else 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if args.stage in ("run", "smoke"):
        if not args.evaluator:
            sys.exit(f"--stage {args.stage} Yes --evaluator")
        if args.stage == "smoke":
            smoke(args.evaluator, n_branch=args.smoke_n)
        else:
            run(args.evaluator)
    else:
        analyze()
    return 0


if __name__ == "__main__":
    sys.exit(main())
