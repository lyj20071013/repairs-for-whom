"""ALFWorld stage B offline self-test（mock environment + mock model，run a small end-to-end test）。by alfw_stage_b.py --stage selftest call。

check：streaming and batched continuations match record by record；design invariants（remaining-step slack、candidates are distinct and executable、correct prefixes、planner-suggested action）；
resumed results exactly match an uninterrupted run；analysis and report generation succeed；judge results can be included in the analysis。
"""
import json
import shutil
import tempfile
from pathlib import Path

import alfw_stage_b as SB
from alfw import acfg, agent, mockenv, pool, rollout

OK = [True]


def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    OK[0] = OK[0] and bool(cond)


def small():
    acfg.B_N_COLLECT, acfg.B_MAX_SCREEN, acfg.B_MIN_INFORMATIVE, acfg.B_SCREEN_MIN_DIFF = 60, 14, 1, 1
    acfg.B_N_MAIN_MAX, acfg.B_N_DEPLOY, acfg.B_MAX_ACTIVE = 6, 6, 40


def run(out, stages, stop_after=None):
    SB.STATE.update(mock=True, mock_llm=False, out=out)
    ctx = {"pool": None, "_backend": None}
    ctx["backend"] = lambda: ctx.setdefault("b", agent.MockBackend())
    if stop_after is not None:
        ctx["stop_after"] = stop_after
    with pool.ProcPool("mock", 2) as pl:
        ctx["pool"] = pl
        d = None
        for st in stages:
            if st == "collect":
                SB.stage_collect(ctx)
            elif st == "design":
                d = SB.stage_design(ctx)
            elif st == "screen":
                SB.run_branches(ctx, d or SB.jload(SB.P("design")), "deploy_T07", SB.B_SCR, acfg.B_N_SCREEN,
                                SB.P("screen"), acfg.TEMPERATURE)
            elif st == "select":
                return SB.stage_select(d or SB.jload(SB.P("design")))
            elif st == "main":
                md = SB.jload(SB.P("main_design"))
                SB.run_branches(ctx, md, "deploy_T07", SB.B_MAIN, acfg.B_N_DEPLOY, SB.P("main"), acfg.TEMPERATURE)
                SB.run_branches(ctx, md, "llm_greedy", SB.B_MAIN, acfg.B_N_GREEDY, SB.P("main"), 0.0)


def main():
    small()
    tmp = Path(tempfile.mkdtemp(prefix="alfw_b_selftest_"))
    try:
        print("[selftest-B] streaming and batched continuations match record by record")
        be = agent.MockBackend()
        specs = [{"rid": f"e{i}", "gamefile": f"mock://{300 + i}", "seed": agent.derive_seed("sb", i),
                  "prefix": ["look"] * (i % 4)} for i in range(30)]
        with pool.InlinePool("mock") as p:
            eps, _ = rollout.run_episodes(be, p, specs, temperature=0.7, chunk=7)
            got = {}
            rollout.run_stream(be, p, iter(specs), 0.7, lambda e: got.__setitem__(e["rid"], e), max_active=5)
        check(all([s["cmd"] for s in e["steps"]] == [s["cmd"] for s in got[e["rid"]]["steps"]] and
                  e["won"] == got[e["rid"]]["won"] for e in eps), "30 records rollout have identical action sequences and outcomes")

        print("[selftest-B] collect → design → screening → selection")
        A = tmp / "A"
        md = run(A, ["collect", "design", "screen", "select"])
        d = json.loads((A / "design.json").read_text())
        eps = {e["rid"]: e for e in (json.loads(x) for x in (A / "episodes.jsonl").read_text().splitlines())}
        bad = []
        for t in d["trajectories"]:
            ep = eps[t["traj_id"]]
            for p in t["positions"]:
                env = mockenv.MockEnv()
                env.load(ep["gamefile"])
                env.reset()
                for c in ep["cmds"][:p["t"]]:
                    env.step(c)
                plan = env.plan()
                adm = env._admissible(env.s)
                slack = (acfg.MAX_STEPS - p["t"]) - len(plan)
                if slack < acfg.B_MIN_SLACK:
                    bad.append(("slack", t["traj_id"], p["t"]))
                acts = [d["branches"][b]["action"] for b in p["alts"]]
                if len(set(acts)) != len(acts) or ep["cmds"][p["t"]] in acts or any(a not in adm for a in acts):
                    bad.append(("cands", t["traj_id"], p["t"]))
                ex = [d["branches"][b]["action"] for b in p["alts"] if d["branches"][b]["expert"]]
                if len(ex) > 1 or (ex and ex[0] != plan[0]):
                    bad.append(("expert", t["traj_id"], p["t"]))
                if d["branches"][p["orig"]]["prefix"] != ep["cmds"][:p["t"] + 1]:
                    bad.append(("orig_prefix", t["traj_id"], p["t"]))
                for b in p["alts"]:
                    br = d["branches"][b]
                    if br["prefix"] != ep["cmds"][:p["t"]] + [br["action"]]:
                        bad.append(("alt_prefix", b))
                    e2 = mockenv.MockEnv()
                    e2.load(ep["gamefile"])
                    e2.reset()
                    st = None
                    for c in br["prefix"]:
                        st = e2.step(c)
                    if not st["won"]:
                        Lb = len(e2.plan())
                        rem = acfg.MAX_STEPS - len(br["prefix"])
                        if br["oracle_F"] != (1.0 if Lb <= rem else 0.0) or br["oracle_slack"] != float(rem - Lb):
                            bad.append(("oracle", b))
                    elif br["deterministic"] != 1.0:
                        bad.append(("det", b))
            if [p["t"] for p in t["positions"]] != sorted(p["t"] for p in t["positions"]) or not t["positions"][-1]["decisive"]:
                bad.append(("order", t["traj_id"]))
        check(not bad, f"design invariants（{len(d['trajectories'])} trajectories）" + ("" if not bad else f"：{bad[:5]}"))
        check(md is not None and len(md["trajectories"]) <= acfg.B_N_MAIN_MAX, "included in the main experiment after screening")
        info = json.loads((A / "screen_result.json").read_text())
        check(info["n_screened"] == len(d["trajectories"]), f"screening results {info['n_informative']}/{info['n_screened']} signal detected")

        print("[selftest-B] resume after interruption = uninterrupted run")
        B = tmp / "B"
        shutil.copytree(A, B)
        run(A, ["main"])
        try:
            run(B, ["main"], stop_after=3)
            interrupted = False
        except KeyboardInterrupt:
            interrupted = True
        n_partial = len((B / "main_rollouts.jsonl").read_text().splitlines())
        run(B, ["main"])
        ra = {(r["condition"], r["branch_id"]): r["results"] for r in map(json.loads, (A / "main_rollouts.jsonl").read_text().splitlines())}
        rb = {(r["condition"], r["branch_id"]): r["results"] for r in map(json.loads, (B / "main_rollouts.jsonl").read_text().splitlines())}
        check(interrupted and n_partial >= 3 and ra == rb, f"saved before interruption {n_partial} branches，after resuming {len(rb)} branch results match exactly")

        print("[selftest-B] analysis and judging")
        SB.STATE.update(mock=True, out=A)
        out = SB.stage_analyze(n_boot=200)
        check(out is not None and all(h in out for h in ("B2", "B3", "B4", "B5", "B6", "B7", "positive_control")),
              "primary and secondary tests have results")
        check(set(out["rules"]) == set(SB.RULES), "decision rules from all four published methods have results")

        class Args:
            judge, model, no_thinking, no_executor_info, retry_invalid, max_tokens, dry_run = \
                "mock", None, False, False, False, None, False
        SB.stage_judge(Args)
        out2 = SB.stage_analyze(n_boot=200)
        check("mock" in out2["judges"], "judge results included in analysis")
        rep = (A / "report.md").read_text()
        check("positive control" in rep and "B2" in rep, "report generation")

        print("[selftest-B] informed judge")
        recs = pool.parse_receptacles("You are in the middle of a room. Looking quickly around you, you see a cabinet 6, "
                                      "a cabinet 5, a coffeemachine 1, a countertop 2, and a toaster 1.\n\nYour task is to: x.")
        check(recs == ["cabinet 6", "cabinet 5", "coffeemachine 1", "countertop 2", "toaster 1"], f"parse the initial container list：{recs}")
        with pool.ProcPool("mock", 2) as pl:
            rc = SB.stage_survey({"pool": pl})
        sv = json.loads((A / "survey.json").read_text())
        md = json.loads((A / "main_design.json").read_text())
        check(rc == 0 and set(sv) == {t["traj_id"] for t in md["trajectories"]} and all(v["receptacles"] for v in sv.values())
              and all("you see" not in v["text"] for v in sv.values()), "survey：container contents recorded for every trajectorys game")
        check(all(x["ok"] for v in sv.values() for x in v["receptacles"]), "every container was visited")
        closed = [x for v in sv.values() for x in v["receptacles"] if x["closed"]]
        check(all(x["open_obs"] for x in closed), f"open closed containers before recording（{len(closed)} items）")
        eps = {e["rid"]: e for e in SB.load_episodes()}
        t0 = md["trajectories"][0]
        m0 = SB.judge_messages(eps[t0["traj_id"]], [], hidden=None)[1]["content"]
        m1 = SB.judge_messages(eps[t0["traj_id"]], [], hidden=sv[t0["traj_id"]]["text"])[1]["content"]
        check(SB.HIDDEN_HEADER not in m0 and SB.HIDDEN_HEADER in m1 and m1.replace(
            f"{SB.HIDDEN_HEADER}\n{sv[t0['traj_id']]['text']}\n\n", "") == m0, "informed prompt = original prompt + hidden-information block（everything else is identical）")

        class Args2(Args):
            informed, effort = True, None
        SB.stage_judge(Args2)
        out3 = SB.stage_analyze(n_boot=200)
        check("mock_informed" in out3["judges"] and "mock" in out3["judge_describe"], "informed-judge results and metadata included in analysis")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[selftest-B] {'all passed' if OK[0] else 'failures detected'}")
    return 0 if OK[0] else 1
