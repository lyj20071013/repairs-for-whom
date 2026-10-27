"""ALFWorld：A new model for the running phase B The same branch of the main experiment（Followers ladder），connect to the registered analysis。

at out_alfw_b/main_design.json（30 trajectories × 3 Organisation × 7 branches = 630 Random branch）on，Make another model a repeater：
    S  Each branch 10 runs（temperature 0.7）        G  greedy 1 runs
prompt、Status Line、Unexecutable Reaction、Script context、Gradual Feeds、Action parsing with 7B exactly identical（alfw.rollout / alfw.agent）。
    q14     Local Qwen2.5-14B-Instruct（vLLM，bf16；DS_Q14 Specify Path；Two cards LADDER_TP=2）
    qflash  Qwen3.8-flash（API，thinking disabled；QWEN_API_KEY）
    q7api / q14api  hosted on the Qwen platform Qwen2.5-7B / 14B（only if absent GPU Time reserve；q7api For calibration API Consistency with Local）

    python ladder_alfw.py --stage smoke --evaluator qflash   # 20 branches × 2 runs：success rate、Cost and full estimate（Save the result）
    python ladder_alfw.py --stage run --evaluator q14        # Full（supports resuming interrupted runs）
    python ladder_alfw.py --stage analyze                    # out_alfw_b/ladder_report.md（All the models that have run out are listed）
    python ladder_alfw.py --stage pack                       # package the resources used by the main experiment 30 games，For use by another server
    python ladder_alfw.py --stage selftest                   # mock environment + Simulation model takes a full course
input（stage B Main Experiment Design、7B Samples deployed、Outcome of the decision）Read From out_alfw_b/；The output of the successors staircase is left on its own out_ladder_alfw/：
    <model>.jsonl、<model>_smoke.jsonl / .json、ladder_report.md、ladder.log、ladder_games.tar.gz
Different servers running out <model>.jsonl Cuff the same one out_ladder_alfw/ then analyze Merge。
"""
import argparse
import json
import os
import random
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

import alfw_stage_b as SB
from alfw import acfg

EVALUATORS = {"q14": os.environ.get("DS_Q14", "/root/autodl-tmp/models/Qwen2.5-14B-Instruct"),
              "qflash": "api:" + os.environ.get("LADDER_QFLASH_MODEL", "qwen3.8-flash"),
              "q7api": "api:" + os.environ.get("LADDER_Q7API_MODEL", "qwen2.5-7b-instruct"),
              "q14api": "api:" + os.environ.get("LADDER_Q14API_MODEL", "qwen2.5-14b-instruct")}
N_EVAL = 10
LB = "L"                      # Batch name of the successor ladder
TP = int(os.environ.get("LADDER_TP", "1"))
LADDER = {"dir": Path(__file__).resolve().parent / "out_ladder_alfw"}


def use_ladder_dir(d=None):
    LADDER["dir"] = Path(d).resolve() if d else LADDER["dir"]
    LADDER["dir"].mkdir(parents=True, exist_ok=True)
    SB.STATE["log_file"] = LADDER["dir"] / "ladder.log"


def log(msg):
    SB.log(msg)


def path_of(name, smoke=False):
    return LADDER["dir"] / (f"{name}_smoke.jsonl" if smoke else f"{name}.jsonl")


def conds(name):
    return f"{name}_T07", f"{name}_greedy"


def make_backend(name):
    from alfw import agent
    if SB.STATE["mock"] or SB.STATE["mock_llm"]:
        return agent.MockBackend()
    spec = EVALUATORS[name]
    if spec.startswith("api:"):
        from alfw.apibackend import APIBackend
        b = APIBackend(spec[4:])
        outs = b.preflight()
        log(f"[{name}] Interface check through（temperature {acfg.TEMPERATURE} and 0 Theyre all accepted）：{[o[:30] for o in outs]}；Together {b.workers}；"
            f"token Count{'Use local wordboxes' if b.tok is not None else 'Crude estimate by character'}")
        return b
    if TP > 1:
        # agent.VLLMBackend No length parallel parameters accepted；No, no, no, no agent.py（its hash is from stage B Part of the signature of the result），Replace only in this process vllm.LLM
        import functools
        import vllm
        vllm.LLM = functools.partial(vllm.LLM, tensor_parallel_size=TP)
    t0 = time.time()
    b = agent.VLLMBackend(model_dir=spec)
    log(f"[{name}] Load Model {time.time() - t0:.0f}s（{spec}）")
    return b


def _stochastic(md):
    return sorted(b for b, x in md["branches"].items() if x["deterministic"] is None)


def run(name, smoke_n=None, smoke_k=2):
    """Full（smoke_n=None）Or smoke（smoke_n branches × smoke_k runs，temperature 0.7，Save the result、rerun each time）。"""
    md = SB.jload(SB.P("main_design"))
    holder = {}
    ctx = {"backend": lambda: holder["b"] if "b" in holder else holder.setdefault("b", make_backend(name))}
    if smoke_n:
        pick = random.Random(0).sample(_stochastic(md), min(smoke_n, len(_stochastic(md))))
        md = dict(md, branches={b: md["branches"][b] for b in pick})
        path = path_of(name, smoke=True)
        if path.exists():
            path.unlink()
        phases = [(f"{name}_smoke", "SMK", smoke_k, acfg.TEMPERATURE)]
    else:
        path = path_of(name)
        cs, cg = conds(name)
        phases = [(cs, LB, N_EVAL, acfg.TEMPERATURE), (cg, LB, 1, 0.0)]
    t_all = time.time()
    tms = []
    try:
        with SB.make_pool() as pl:
            ctx["pool"] = pl
            for cond, batch, n, temp in phases:
                tm = SB.run_branches(ctx, md, cond, batch, n, path, temp)
                if tm:
                    tms.append(tm)
    finally:
        b = holder.get("b")
        if b is not None and hasattr(b, "close"):
            b.close()
    if smoke_n:
        return smoke_summary(name, md, path, holder.get("b"), tms, smoke_k)
    left = _missing(name)
    log(f"[{name}] This operation is over，Time {(time.time() - t_all) / 60:.0f} minutes；" +
        ("All branches completed。Next：python ladder_alfw.py --stage analyze" if not left
         else f"also {len(left)} items（Condition, branch）incomplete，The same order will be replaced by the same order"))
    return 0


def _missing(name):
    import runner
    md = SB.jload(SB.P("main_design"))
    have = runner.load_results(path_of(name))
    cs, cg = conds(name)
    return [(c, b) for c in (cs, cg) for b in _stochastic(md) if (c, LB, b) not in have]


def smoke_summary(name, md, path, backend, tms, k):
    import runner
    full_md = SB.jload(SB.P("main_design"))
    res7 = runner.load_results(SB.P("main"))
    rec = runner.load_results(path)
    bs = sorted(md["branches"])
    p_eval = np.mean([np.mean(rec[(f"{name}_smoke", "SMK", b)]["results"]) for b in bs])
    p7 = np.mean([np.mean(res7[("deploy_T07", SB.B_MAIN, b)]["results"]) for b in bs])
    steps = np.mean([np.mean(rec[(f"{name}_smoke", "SMK", b)]["steps"]) for b in bs])
    inv = sum(sum(rec[(f"{name}_smoke", "SMK", b)]["invalid"]) for b in bs)
    n_roll = len(bs) * k
    wall = sum(t["wall"] for t in tms) if tms else float("nan")
    n_full = len(_stochastic(full_md)) * (N_EVAL + 1)
    L = [f"[smoke {name}] {len(bs)} branches × {k} runs = {n_roll} records rollout，When running again {wall / 60:.1f} minutes（Load without Model）",
         f"  success rate：{name} {p_eval:.3f}；7B On the same branch {p7:.3f}",
         f"  Average number of steps {steps:.1f}；Unexecutable after answering again {inv} runs"]
    summary = {"name": name, "n_rollouts": n_roll, "wall_s": wall, "success_eval": float(p_eval),
               "success_7b_same_branches": float(p7), "mean_steps": float(steps),
               "projected_hours": wall / n_roll * n_full / 3600}
    if backend is not None and hasattr(backend, "usage"):
        from api_client import cost_yuan
        u = backend.usage
        c = cost_yuan(u)
        L += [f"  call {u['calls']} runs（Every single one rollout {u['calls'] / n_roll:.1f}），input {u['prompt_tokens'] / max(1, u['calls']):.0f} "
              f"token/runs，including cache hits {u['cached_tokens'] / max(1, u['prompt_tokens']):.0%}；output "
              f"{u['completion_tokens'] / max(1, u['calls']):.0f} token/runs，Interrupted by length {u['length_cut']} runs；Average time delay "
              f"{u['seconds_in_calls'] / max(1, u['calls']):.1f} seconds/runs；Try again {u['retries']} runs",
              f"  estimated smoke-test cost ¥{c:.2f}（Based on platform billing）"]
        # Time：the smoke test has only n_roll records、Move in by wave，Wall clocks are mostly the longest rollout Resolves，Linear extrapolation is grossly overestimated。
        # Full amount at the same time B_MAX_ACTIVE records、Maximum simultaneous distribution per wave workers A request：Time ≈ Total calls × Average time delay / Together × 1.5（The costs of the wave and the repeat）
        lat = u["seconds_in_calls"] / max(1, u["calls"])
        calls_full = u["calls"] / n_roll * n_full
        conc = min(backend.workers, acfg.B_MAX_ACTIVE)
        h = calls_full * lat / conc * 1.5 / 3600
        summary["projected_hours_api"] = h
        L += [f"  Full forecast（{len(_stochastic(full_md))} branches × {N_EVAL} runs + greedy 1 runs = {n_full} records rollout，approximately {calls_full:,.0f} Sub-calls）："
              f"approximately ¥{c / n_roll * n_full:.0f}；Timing {h:.1f} hours（Together {conc}，QWEN_EXEC_WORKERS Modifiable；"
              f"linearly extrapolated from smoke-test wall time {summary['projected_hours']:.1f} The hours overrated），Subject to full log"]
        L += ["  Original Output Example："] + [f"    {x[:200]!r}" for x in backend.client.samples[:2]]
        summary.update(usage=u, cost_yuan=c, projected_cost_yuan=c / n_roll * n_full)
    else:
        # Local GPU：the smoke test has only n_roll The strips are running，GPU far below full utilization；Full amount at the same time B_MAX_ACTIVE records，much higher throughput。
        # by“Full vomiting ≈ 7B stage B measured 26 steps/seconds × 7/Number of parameters”Rough estimate（14B approximately 13 steps/seconds），use the complete-run logs for the actual estimate“steps/seconds”Subject to。
        import re as _re
        m = _re.search(r"(\d+(?:\.\d+)?)B", EVALUATORS.get(name, ""))
        size = float(m.group(1)) if m else 7.0
        sps = 26.0 * 7.0 / size
        h = steps * n_full / sps / 3600
        summary["projected_hours_full_concurrency"] = h
        L += [f"  Full forecast：the smoke test has only {n_roll} The strips are running at the same time，At this speed of linear extrapolation {summary['projected_hours']:.1f} hours，but this substantially overestimates——"
              f"All at once {acfg.B_MAX_ACTIVE} records，GPU much higher throughput。By contract {sps:.0f} steps/Seconds are rough：approximately {h:.1f} hours"
              f"（{n_full} records rollout × Average {steps:.0f} steps）；use the complete-run logs“steps/seconds”Subject to"]
    (LADDER["dir"] / f"{name}_smoke.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    log("\n".join(L))
    return 0


# ---------------------------------------------------------------- Analysis
class _Combined:
    def __init__(self, parts):
        self.parts = parts
        self.names = [n for p in parts for n in p.names]

    def __call__(self, **kw):
        out = {}
        for p in self.parts:
            out.update(p(**kw))
        return out


def load_evals(md):
    import runner
    evals = {}
    nst = _stochastic(md)
    for n in EVALUATORS:
        p = path_of(n)
        if not p.exists():
            continue
        rec = runner.load_results(p)
        cs, cg = conds(n)
        ev = {}
        for b in nst:
            if (cs, LB, b) in rec:
                ev[("S", b)] = rec[(cs, LB, b)]
            if (cg, LB, b) in rec:
                ev[("G", b)] = rec[(cg, LB, b)]
        if all(("S", b) in ev and ("G", b) in ev for b in nst):
            evals[n] = ev
        else:
            log(f"[analyze] {n} incomplete（{len(ev)}/{2 * len(nst)}），skip")
    return evals


def analyze():
    import confirm
    import runner
    import ladder_gw as LG
    md = SB.jload(SB.P("main_design"))
    res = runner.load_results(SB.P("main"))
    evals = load_evals(md)
    if not evals:
        log("[analyze] complete continuation results are unavailable")
        return None
    judges = SB.load_judges()
    ex = _Combined([SB.make_extra(judges), LG.LadderExtra(evals)])
    V, names, contrast = confirm._selection_values(md, res, "net", confirm.N_BOOT, 0, extra=ex,
                                                   oracle_value=lambda x: x["oracle_F"],
                                                   slack_value=lambda x: x["oracle_slack"])

    def f(c):
        return (f"{c['point']:+.3f} | [{c['ci90'][0]:+.3f}, {c['ci90'][1]:+.3f}] | "
                f"[{c['ci90_bc'][0]:+.3f}, {c['ci90_bc'][1]:+.3f}]")
    val = lambda k: float(np.nanmean(V[k][0]))
    n_dep = acfg.B_N_DEPLOY
    L = ["# ALFWorld：Change the pattern（Followers ladder）", "",
         f"stage B The main experiment is the same {len(md['trajectories'])} trajectories、630 branches；Rehabilitate by the results of the repeat of the models，evaluate under the deployed executor 7B use。"
         f"loss = Match Selection（7B Every discount {n_dep // 2} runs，Crossing）− This way。90% Inter-trajectories are trajectories + parameters bootstrap；"
         "Right as deviation correction（See analysis plan.md 6.3c）。ALFWorld All branches can be solved，oracle Select equals random selection。", "",
         "| Selection | Net deployment gains | Relatively matched loss | 90% interval | deviation correction interval |", "|---|---:|---:|---|---|",
         f"| Match：7B {n_dep // 2} runs | {val('matched'):+.3f} | — | — | — |"]
    rows = [("k10", "7B Deployment 10 runs"), ("k3", "7B Deployment 3 runs"), ("greedy", "7B greedy")]
    for n in evals:
        rows += [(f"eval:{n}", f"{n} continuation {N_EVAL} runs"), (f"eval3:{n}", f"{n} continuation 3 runs"), (f"evalG:{n}", f"{n} greedy")]
    rows += [("expert_earliest", "Planner actions are in the earliest position"), ("slack_abs", "Select in absolute remainder"), ("oracle_slack", "oracle + slack（Definition of registration）")]
    for j, lab in (("qwen3.8-flash_think", "Qwen3.8-flash Rec（Read-only Log）"),
                   ("qwen3.8-flash_think_informed", "Qwen3.8-flash informed judge"), ("qwen7b_self", "7B Self-assessment decision")):
        if f"judge:{j}" in names:
            rows.append((f"judge:{j}", lab))
    rows += [("random", "random（= oracle；positive control）")]
    for k, lab in rows:
        if k in names:
            L.append(f"| {lab} | {val(k):+.3f} | {f(contrast('matched', k))} |")
    L += ["", "**Same number of repeats：7B select using the deployed executor vs Select by another model**（all assigned to 7B use；Difference > 0 Expressed loss of deployment gains due to change of executioner）：", "",
          "| Contrast | Difference | 90% interval | deviation correction interval |", "|---|---:|---|---|"]
    for n in evals:
        L.append(f"| 7B 10 runs − {n} 10 runs | {f(contrast('k10', f'eval:{n}'))} |")
        L.append(f"| 7B 3 runs − {n} 3 runs | {f(contrast('k3', f'eval3:{n}'))} |")
    L += ["", "**Read Log vs run using the deployed executor**（Same model；Difference = The former − The latter）In comparison with：", "",
          "| Contrast | Difference | 90% interval | deviation correction interval |", "|---|---:|---|---|"]
    pairs = []
    if "qflash" in evals:
        for j, lab in (("qwen3.8-flash_think", "flash Read Log"), ("qwen3.8-flash_think_informed", "flash Informed reading of logs")):
            if f"judge:{j}" in names:
                pairs.append((f"judge:{j}", "eval:qflash", f"{lab} − flash run using the deployed executor 10 runs"))
    if "judge:qwen7b_self" in names:
        pairs.append(("k10", "judge:qwen7b_self", "7B run using the deployed executor 10 runs − 7B Read Log"))
    for n in evals:
        pairs += [(f"eval:{n}", "random", f"{n} run using the deployed executor − random（= oracle）"),
                  (f"eval:{n}", "expert_earliest", f"{n} run using the deployed executor − Planner actions are in the earliest position")]
    for a, b, lab in pairs:
        L.append(f"| {lab} | {f(contrast(a, b))} |")
    L.append("")
    for n, ev in evals.items():
        c = contrast("k10", f"eval:{n}")
        lo, hi = c["ci90"]
        L.append(f"**{n}：" + ("Visible executor dependency" if lo > 0 else "Reverse direction" if hi < 0 else "Difference not detected") +
                 f"**（7B 10 runs − {n} 10 runs = {c['point']:+.3f}，interval [{lo:+.3f}, {hi:+.3f}]）")
    L.append("")
    for n, ev in evals.items():
        rows_d = LG.dependence(md, res, ev, n_deploy=n_dep)
        gap = np.array([r["within7"] - r["cross"] for r in rows_d])
        gap = gap[~np.isnan(gap)]
        rng = np.random.default_rng(0)
        m = gap[rng.integers(0, len(gap), size=(2000, len(gap)))].mean(axis=1) if len(gap) else np.array([np.nan])
        cap7 = np.mean([np.mean(res[("deploy_T07", SB.B_MAIN, b)]["results"]) for b in _stochastic(md)])
        cape = np.mean([np.mean(v["results"]) for (ph, b), v in ev.items() if ph == "S"])
        L += [f"## Enforcement dependency：7B vs {n}（description）", "",
              f"- Average success rate for random branches：7B {cap7:.3f}；{n} {cape:.3f}",
              f"- Consistency in ranking of candidate net proceeds（Each 10 runs；7B Internal {n_dep // 10} pairwise across disjoint sample sets）：7B Internal "
              f"{np.nanmean([r['within7'] for r in rows_d]):.3f}；7B vs {n} {np.nanmean([r['cross'] for r in rows_d]):.3f}；"
              f"Internal − Cross-model {gap.mean():+.3f} [{np.quantile(m, 0.05):+.3f}, {np.quantile(m, 0.95):+.3f}]",
              f"- 7B（{n_dep} runs）And the best candidate {n} Best proportion in the eye：{np.mean([r['same_best'] for r in rows_d]):.3f}", ""]
    out = LADDER["dir"] / "ladder_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L))
    return L


# ---------------------------------------------------------------- Packing game data
def pack():
    md = SB.jload(SB.P("main_design"))
    games = sorted({t["gamefile"] for t in md["trajectories"]})
    root = acfg.DATA
    out = LADDER["dir"] / "ladder_games.tar.gz"
    n = 0
    with tarfile.open(out, "w:gz") as tf:
        for g in games:
            if not Path(g).is_file():           # mock environment（mock://）No file
                continue
            d = Path(g).parent
            for p in sorted(d.iterdir()):
                tf.add(p, arcname=str(p.relative_to(root)))
                n += 1
        logic = root / "logic"
        if logic.exists():
            tf.add(logic, arcname="logic")
    log(f"[pack] {len(games)} Game（{n} File）-> {out}（{out.stat().st_size / 1e6:.1f} MB）。"
        f"On another server：mkdir -p {root} && tar xzf ladder_games.tar.gz -C {root}")
    return 0


# ---------------------------------------------------------------- self-test
def selftest():
    import shutil
    import tempfile
    import alfw_selftest_b as ST
    tmp = Path(tempfile.mkdtemp(prefix="ladder_alfw_"))
    ok = True
    try:
        ST.small()
        A = tmp / "A"
        ST.run(A, ["collect", "design", "screen", "select"])      # select after run Will return，main Run alone again
        ST.run(A, ["main"])
        SB.STATE.update(mock=True, mock_llm=False, out=A)
        use_ladder_dir(tmp / "ladder")
        run("q14")
        n1 = len(path_of("q14").read_text().splitlines())
        md = SB.jload(SB.P("main_design"))
        ok &= n1 == 2 * len(_stochastic(md)) and not _missing("q14")
        print(("  ✓ " if ok else "  ✗ ") + f"Full：{n1} Records（{len(_stochastic(md))} branches × Sample / greedy）")
        lines = path_of("q14").read_text().splitlines(keepends=True)
        path_of("q14").write_text("".join(lines[:-5]) + lines[-5][:30])
        run("q14")
        again = path_of("q14").read_text().splitlines()
        same = sorted(again) == sorted(x.rstrip("\n") for x in lines)
        print(("  ✓ " if same else "  ✗ ") + "Breaking up the follow-up：Fill it up and the results are the same as a run")
        ok &= same
        run("qflash", smoke_n=4)
        sm = json.loads((LADDER["dir"] / "qflash_smoke.json").read_text())
        good = sm["n_rollouts"] == 8 and not path_of("qflash").exists()
        print(("  ✓ " if good else "  ✗ ") + "smoke test：Save the result，Do not enter analysis")
        ok &= good
        L = analyze()
        good = L is not None and any("q14 continuation 10 runs" in x for x in L) and any("7B 10 runs − q14 10 runs" in x for x in L)
        print(("  ✓ " if good else "  ✗ ") + "Analysis and reporting")
        ok &= good
        rc = pack()
        good = rc == 0 and (LADDER["dir"] / "ladder_games.tar.gz").exists() and (LADDER["dir"] / "ladder.log").exists()
        print(("  ✓ " if good else "  ✗ ") + "Packing game data（Empty packages in simulated environments）")
        ok &= good
    finally:
        SB.STATE.pop("log_file", None)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ladder-alfw selftest] {'all passed' if ok else 'failures detected'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "smoke", "analyze", "pack", "selftest"])
    ap.add_argument("--evaluator", choices=list(EVALUATORS))
    ap.add_argument("--smoke-n", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=None, help="stage B Output of the main experiment（input），default out_alfw_b")
    ap.add_argument("--ladder-dir", type=Path, default=None, help="Step Output，default out_ladder_alfw")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if args.out_dir:
        SB.STATE["out"] = args.out_dir.resolve()
    SB.STATE["mock"] = args.mock
    if args.stage == "selftest":
        return selftest()
    use_ladder_dir(args.ladder_dir)
    if not args.mock:
        try:
            import selftest as grid_selftest
            grid_selftest.t0_manifest()
        except AssertionError as e:
            log(f"Code version check failed：{e}")
            return 1
    if args.stage in ("run", "smoke"):
        if not args.evaluator:
            sys.exit(f"--stage {args.stage} Yes --evaluator")
        return run(args.evaluator, smoke_n=args.smoke_n if args.stage == "smoke" else None)
    if args.stage == "analyze":
        return 0 if analyze() is not None else 1
    return pack()


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != acfg.HASH_SEED:
        os.environ["PYTHONHASHSEED"] = acfg.HASH_SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.exit(main())
