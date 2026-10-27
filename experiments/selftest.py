"""none GPU self-test。Any failure should be repaired first，Get on the server and run 7B。

1. Reuse Source with pilot_vshape Bytes Bytes Same
2. Quick oracle and diagnose.shortest_solution Same as always
3. Intervention Branch：The original action branch equals the next state saved；The substitute branch is real step Generate；F=0 the branch truly cannot succeed
4. seed：Sure、Batch-to-Batch Different；Noise with seeds oracle The results can be repeated
5. 7B Packaging（Fake reasoning，Unloaded weights）：Parameter by Request and Torrent、temperature 0 No Torrent、The prompts is consistent with the collection protocol、End not advancing
6. Test of the known answers analysed
7. Full Process Simulation（--mock Equivalent）：design -> rollouts -> analyze
"""
import json
import random
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import config


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  OK ", msg)


MANIFEST = config.ROOT / "MANIFEST.json"


def manifest_files():
    alfw = [f"alfw/{p.name}" for p in (config.ROOT / "alfw").glob("*.py")]
    return sorted([p.name for p in config.ROOT.glob("*.py")] + ["run_bg.sh"] + alfw)


def write_manifest():
    data = {n: config.file_sha256(config.ROOT / n) for n in manifest_files()}
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written {MANIFEST.name}（{len(data)} File）")


def t0_manifest():
    """All code files are consistent with the list issued with the release，Prevent only some files from synchronizing。"""
    print("[0] Code Version Consistency（MANIFEST.json）")
    if not MANIFEST.exists():
        raise AssertionError("Missing MANIFEST.json：Please take the whole thing repair_transfer Synchronise folder to server")
    want = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stale = [n for n, h in want.items() if not (config.ROOT / n).exists() or config.file_sha256(config.ROOT / n) != h]
    if stale:
        raise AssertionError("The following documents are not consistent with the List:（likely missing the latest synchronized version）：" + "、".join(stale)
                             + "。Please re-sync the entire folder；if the modification was intentional，Run python selftest.py --write-manifest Update List。")
    check(True, f"{len(want)} The document is consistent with the list")


def t1_provenance():
    """env.py / model.py DEFINITION OF ENVIRONMENTAL TRANSFER AND 7B prompts protocol for %1，It must be consistent with the collection of frozen trajectories；
    diagnose.py an offline diagnostic tool only（only its following component is used here BFS _Other Organiser，number 2 Step-by-state check），Warnings only for inconsistencies。"""
    print("[1] Reuse Source Consistency")
    if not config.PILOT_DIR.exists():
        print("  skip：Not found pilot_vshape directory")
        return
    for name in ("env.py", "model.py"):
        check(config.file_sha256(config.ROOT / name) == config.file_sha256(config.PILOT_DIR / name),
              f"{name} and pilot_vshape identical")
    name = "diagnose.py"
    if (config.PILOT_DIR / name).exists() and \
            config.file_sha256(config.ROOT / name) != config.file_sha256(config.PILOT_DIR / name):
        print(f"  This function is EXPERIMENTAL {name} and pilot_vshape Versions in different versions（It doesnt affect the experiment；Available diff View Differences）")
    else:
        print(f"  OK  {name} and pilot_vshape identical")


def t2_oracle(records, n=60):
    print("[2] oracle and BFS identical")
    import oracle
    from diagnose import shortest_solution
    rng = random.Random(0)
    states = [s for r in records for s in r["states"]]
    for s in rng.sample(states, min(n, len(states))):
        sol = shortest_solution(s)
        tab = oracle.table_for(s)
        assert (sol is not None) == tab.feasible(s), "Discrepancies in solvency"
        if sol is not None and not s["done"]:
            assert len(sol) == tab.distance(s), "The shortest distance is inconsistent"
    check(True, f"{min(n, len(states))} The resolutionability of a random state corresponds to the shortest distance")


def t3_design(d, records):
    print("[3] Intervention Branch")
    import oracle
    from env import GridWorld
    from policies import OraclePolicy, NoisyOraclePolicy
    by_id = {r["traj_id"]: r for r in records}
    for traj in d["trajectories"]:
        rec = by_id[traj["traj_id"]]
        for pos in traj["positions"]:
            t = pos["t"]
            check_orig = d["branches"][pos["orig"]]["snapshot"] == rec["states"][t + 1]
            assert check_orig, "The original action branch is not the next state to save"
            assert oracle.feasible(rec["states"][t]), "The location of the intervention must be within the budget"
            for bid in pos["alts"]:
                br = d["branches"][bid]
                env = GridWorld.restore(rec["states"][t])
                cand = next(a for a in config.CANDIDATE_ACTIONS if bid.endswith("|" + (
                    f"{a['tool']}:{a['arg']}" if a.get("arg") else a["tool"])))
                env.step(dict(cand))
                assert env.snapshot() == br["snapshot"], "Substitute Branch and Real step different"
                assert br["snapshot"]["steps"] == rec["states"][t]["steps"] + 1, "The replacement action does not consume a step"
    check(True, "All original and alternative branches can be recreated by the real environment")
    zero = [b for b in d["branches"].values() if b["deterministic"] == 0.0 and not b["snapshot"]["done"]]
    for br in zero[:15]:
        assert not OraclePolicy().run_rollouts([br["snapshot"]])[0]["success"]
        assert not any(r["success"] for r in NoisyOraclePolicy(0.3).run_rollouts([br["snapshot"]] * 5, list(range(5))))
    check(True, f"F=0 branch（A spot check {min(15, len(zero))} items）at oracle cannot succeed under either policy or random behavior")
    stoch = [b for b in d["branches"].values() if b["deterministic"] is None]
    res = OraclePolicy().run_rollouts([b["snapshot"] for b in stoch[:30]])
    check(all(r["success"] for r in res), "stochastic branches in oracle all continuations succeed（and F=1 identical）")


def t4_seeds(d):
    print("[4] seed")
    from policies import derive_seed, NoisyOraclePolicy
    a = derive_seed("deploy_T07", "S", "ep0001|t3|pick", 0)
    check(a == derive_seed("deploy_T07", "S", "ep0001|t3|pick", 0), "Same field gets the same torrent")
    check(a != derive_seed("deploy_T07", "V", "ep0001|t3|pick", 0), "Select Batch Unlike Verify Batch Feeds")
    br = next(b for b in d["branches"].values() if b["deterministic"] is None)
    seeds = [derive_seed("x", "S", br["branch_id"], k) for k in range(20)]
    p = NoisyOraclePolicy(0.5)
    r1 = [r["success"] for r in p.run_rollouts([br["snapshot"]] * 20, seeds)]
    r2 = [r["success"] for r in p.run_rollouts([br["snapshot"]] * 20, seeds)]
    check(r1 == r2, "Noise oracle Resume results under the same torrent")


def t5_llm_contract(d):
    print("[5] 7B Packaging Interface（Fake reasoning）")
    import model
    from env import GridWorld
    from policies import make_seeded_llm_class
    cls = make_seeded_llm_class()
    calls = []

    def generate(prompts, params, **kw):
        calls.append((list(prompts), list(params)))
        return [SimpleNamespace(outputs=[SimpleNamespace(text='Thought: wait.\nAction: {"tool": "look"}')])
                for _ in prompts]

    pol = cls.__new__(cls)
    pol.temperature = 0.7
    pol._make_params = lambda **kw: SimpleNamespace(**kw)
    pol.llm = SimpleNamespace(generate=generate)
    pol.tokenizer = SimpleNamespace(apply_chat_template=lambda msgs, **kw: msgs)

    br = next(b for b in d["branches"].values() if b["deterministic"] is None)
    done_snap = next(b for b in d["branches"].values() if b["snapshot"]["done"])["snapshot"]
    snap = br["snapshot"]
    out = pol.run_rollouts([snap, snap, done_snap], [11, 12, 13])
    remaining = snap["max_steps"] - snap["steps"]
    check(out[0]["steps"] == remaining and not out[0]["success"], "only look Failed after running out of time of the remaining budget")
    check(out[2]["steps"] == 0 and out[2]["success"] == done_snap["success"], "The final branch does not push forward and inherits the result")
    check(all(len(p) == len(ps) for p, ps in calls), "Each request has its own sampling parameters")
    first_seeds = [ps.seed for ps in calls[0][1]]
    check(first_seeds[0] != first_seeds[1], "Different from the same branch rollout Seeds are different")
    check(calls[0][1][0].seed != calls[1][1][0].seed, "Same rollout There are different steps to the seeds")
    env = GridWorld.restore(snap)
    expected = model.AgentPolicy.build_messages(env.observe(), env.history)
    check(calls[0][0][0] == expected, "Follow-up alerts are the same as the message from the collection（History contains only actions and results）")
    calls.clear()
    pol.temperature = 0.0
    pol.run_rollouts([snap], [5])
    check(not hasattr(calls[0][1][0], "seed"), "temperature 0（greedy）Do Not Send Seeds")
    # Run away：The first step is greedy；look Do not change physical state，And every step of it goes back to the status quo ante → Switch to flight temperature and bring seeds
    calls.clear()
    pol.temperature, pol.escape_temperature = 0.0, 0.7
    out = pol.run_rollouts([snap], [5])
    first, second = calls[0][1][0], calls[1][1][0]
    check(first.temperature == 0.0 and not hasattr(first, "seed"), "The first step of escape strategy is still greedy")
    check(second.temperature == 0.7 and hasattr(second, "seed"), "switch temperature upon revisiting a state 0.7 And bring the seeds")
    check(out[0]["escapes"] == out[0]["steps"] - 1, "Number of escapes = step count − 1（only look when）")
    pol.escape_temperature = None


def t6_analysis_known_answer():
    print("[6] Analysis：known-answer test")
    import analysis
    snap = {"steps": 1, "max_steps": 30, "done": False, "success": False}
    design = {"design_id": "toy", "trajectories": [{"traj_id": "toy", "positions": [
        {"t": 0, "orig": "o", "alts": ["a", "b"]}]}],
        "branches": {k: {"branch_id": k, "deterministic": None, "snapshot": snap} for k in ("o", "a", "b")}}
    R = {}

    def put(cond, batch, bid, k, n):
        R[(cond, batch, bid)] = {"results": [1] * k + [0] * (n - k)}

    # Conditions of deployment：a clearly better；Proxy conditions：b It looks better
    for b, kS, kV in (("o", 10, 20), ("a", 60, 120), ("b", 12, 24)):
        put(config.DEPLOY, "S", b, kS, 100)
        put(config.DEPLOY, "V", b, kV, 200)
    for b, k in (("o", 10), ("a", 10), ("b", 90)):
        put("proxy", "S", b, k, 100)
    old = dict(config.BATCHES), dict(config.CONDITIONS)
    config.BATCHES["proxy"] = {"S": 100}
    config.CONDITIONS["proxy"] = {"kind": "noisy_oracle", "eps": 0.0}
    try:
        res = analysis.analyze(design, R, sources=[analysis.Source(config.DEPLOY, "S"), analysis.Source("proxy", "S")],
                               objective="net", n_boot=300)
    finally:
        config.BATCHES.clear(); config.BATCHES.update(old[0])
        config.CONDITIONS.clear(); config.CONDITIONS.update(old[1])
    loss = res["point"]["proxy/S"]["loss_vs_matched"]
    check(abs(loss - (0.60 - 0.12 - 0.0)) < 1e-9, f"Loss on proxy wrong candidate = 0.48（Got it {loss:.3f}）")
    check(res["intervals"]["proxy/S"]["verdict"].startswith("practically meaningful"), "The loss was found to be meaningful（> δ）")
    check(res["point"][f"{config.DEPLOY}/S"]["loss_vs_matched"] == 0.0, "Match selects relative to self-lost as 0")
    check(res["positive_control"]["discriminative"], "the positive control passes when preferences differ（Random selection loss > δ）")

    # Floor effects：All branches are close to success 0，Proxy and deployment“It looks the same”，but equivalence cannot be established。
    R.clear()
    for b in ("o", "a", "b"):
        put(config.DEPLOY, "S", b, 1, 100)
        put(config.DEPLOY, "V", b, 1, 200)
        put("proxy", "S", b, 0, 100)
    res = analysis.analyze(design, R, sources=[analysis.Source(config.DEPLOY, "S"), analysis.Source("proxy", "S")],
                           objective="net", n_boot=300)
    check(not res["positive_control"]["discriminative"], "The positives dont pass when the floor is working")
    check(not res["intervals"]["proxy/S"]["verdict"].startswith("Exclude"), "The floor effect will not be given“No big loss”Conclusions")


def t6b_partial_line():
    print("[6b] Continue Cache Halfline")
    import runner
    tmp = Path(tempfile.mkdtemp(prefix="rt_tail_")) / "r.jsonl"
    good = json.dumps({"condition": "c", "batch": "S", "branch_id": "x", "results": [1]})
    tmp.write_text(good + "\n" + good.replace('"x"', '"y"')[:25], encoding="utf-8")
    done = runner.load_results(tmp)
    check(set(done) == {("c", "S", "x")}, "Ignore unwritten half-lines while reading")
    runner._repair_tail(tmp)
    check(tmp.read_text(encoding="utf-8") == good + "\n", "Cut off half a line before you go back，Completed records retention")
    tmp.write_text("{broken\n" + good + "\n", encoding="utf-8")
    try:
        runner.load_results(tmp)
        raise AssertionError("Midline damage should be compensated for")
    except ValueError:
        check(True, "The middle line damage is still misreported")
    shutil.rmtree(tmp.parent, ignore_errors=True)


def t7_pipeline(source):
    print("[7] Full Process Simulation（7B The conditions are noisey oracle Replace）")
    import run_cross
    tmp = Path(tempfile.mkdtemp(prefix="rt_selftest_"))
    saved = (config.N_TRAJ, config.N_BOOTSTRAP, json.dumps(config.BATCHES), config.OUT)
    try:
        config.set_output_dir(tmp)
        config.N_TRAJ, config.N_BOOTSTRAP = 4, 200
        for c in config.BATCHES:
            config.BATCHES[c] = {b: min(n, 20) for b, n in config.BATCHES[c].items()}
        run_cross.stage_design(source)
        run_cross.stage_rollouts(mock=True)
        run_cross.stage_rollouts(mock=True)        # The second time should all be hit by a cache
        run_cross.stage_analyze(mock=True)
        res = json.loads(config.ANALYSIS_JSON.read_text(encoding="utf-8"))
        check(set(res["intervals"]) == set(res["sources"]), "Each selection source has a range and a judgment")
    finally:
        config.N_TRAJ, config.N_BOOTSTRAP = saved[0], saved[1]
        config.BATCHES.clear(); config.BATCHES.update(json.loads(saved[2]))
        config.set_output_dir(saved[3])
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    import design as D
    source = config.SOURCE_EPISODES
    if not Path(source).exists():
        raise SystemExit(f"frozen trajectories not found {source}；Please confirm pilot_vshape/out_120 Top level of this table of contents")
    records = D.load_records(source)
    t0_manifest()
    t1_provenance()
    t2_oracle(records)
    saved = config.N_TRAJ
    config.N_TRAJ = 6
    d = D.build_design(source)
    config.N_TRAJ = saved
    t3_design(d, records)
    t4_seeds(d)
    t5_llm_contract(d)
    t6_analysis_known_answer()
    t6b_partial_line()
    t7_pipeline(source)
    print("\n=== selftest all passed ===")


if __name__ == "__main__":
    import sys
    if "--write-manifest" in sys.argv:
        write_manifest()
    else:
        main()
