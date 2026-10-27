"""ALFWorld stage A：Feasibility tests（Before running the pilot experiment,）。

    python alfw_stage_a.py --part selftest   offline self-test（mock environment + mock model，not required alfworld / GPU）
    python alfw_stage_a.py --part a1         Install and Data：version、Three separate games、Single Load/Reset/Planning
    python alfw_stage_a.py --part a2         Replay certainty and time-consuming：20 episodes × 40 Walk randomly，compare five replay methods
    python alfw_stage_a.py --part a3         planner：Whether the plan is implemented and the target is reached in an arbitrary state、Time-consuming、Cost coherence
    python alfw_stage_a.py --part a4         7B Baseline（GPU）：train random 100 episodes + valid_unseen 134 episodes，temperature 0.7
    python alfw_stage_a.py --part report     Summary to out_alfw/stageA_report.md（apply the criteria fixed in advance）
    python alfw_stage_a.py --part all        selftest -> a1 -> a2 -> a3 -> a4 -> report
Add --mock：Run flow using all simulation environment and simulation models（Output in out_alfw_mock/，results have no scientific interpretation）。
Add --mock-llm：real ALFWorld + mock model（none GPU Time Check A4 pipeline，Output in out_alfw_mockllm/）。

criteria are specified in alfw/acfg.py of GATES，and recorded in analysis plan.md 6.3a（Before running）。
"""
import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from alfw import acfg

SCRIPT = Path(__file__).resolve()
STATE = {"mock": False, "mock_llm": False, "out": acfg.OUT}


# ---------------------------------------------------------------- Tools
def log(msg):
    print(msg, flush=True)
    try:
        STATE["out"].mkdir(parents=True, exist_ok=True)
        with open(STATE["out"] / "stageA.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("[%m-%d %H:%M:%S] ") + msg + "\n")
    except OSError:
        pass


def save(name, obj):
    STATE["out"].mkdir(parents=True, exist_ok=True)
    (STATE["out"] / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def load(name):
    p = STATE["out"] / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def kind():
    return "mock" if STATE["mock"] else "tw"


def split_games(split):
    from alfw import games
    if STATE["mock"]:
        n, off = {"train": (200, 0), "valid_seen": (20, 500), "valid_unseen": (30, 1000)}[split]
        return [f"{games.MOCK_PREFIX}{off + i}" for i in range(n)]
    return games.cached_collect(split)


def qstats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    q = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return {"n": len(xs), "mean": statistics.fmean(xs), "p50": q(0.5), "p95": q(0.95), "max": xs[-1]}


def wilson(k, n, z=1.645):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rss_mb():
    from alfw.pool import _rss_mb
    return _rss_mb()


def mem_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        return None


def walk_cmd(rng, st):
    """next command in the random walk：approximately 10% Unlawful orders（Test “Nothing happens.” Treatment）。"""
    if rng.random() < 0.1 or not st["admissible"]:
        return "jump over the moon"
    return rng.choice(st["admissible"])


# ---------------------------------------------------------------- selftest
def part_selftest():
    from alfw import agent, games, mockenv, pool, rollout, twenv
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(("  ✓ " if cond else "  ✗ ") + msg)
        ok = ok and bool(cond)

    log("[selftest] Action Parsing")
    adm = ["go to cabinet 1", "move apple 1 to countertop 1", "take apple 1 from countertop 1", "look"]
    cases = [
        ("Thought: x\nAction: go to cabinet 1", "go to cabinet 1", "valid"),
        ("**Action:** Go to Cabinet 1.", "go to cabinet 1", "valid"),
        ("Thought: x\nAction: put apple 1 in/on countertop 1", "move apple 1 to countertop 1", "alias"),
        ("Thought: x\nAction: put apple 1 on countertop 1", "move apple 1 to countertop 1", "alias"),
        ("Thought: only thinking", "", "no_action"),
        ("Thought: x\nAction:\ngo to cabinet 1", "go to cabinet 1", "valid"),
        ("Thought: I will go there. Action: go to cabinet 1", "go to cabinet 1", "valid"),
        ("Action: `look`", "look", "valid"),
        ("Action: fly to the moon", "fly to the moon", "invalid"),
        ("Thought: a\nAction: look\nThought: b\nAction: go to cabinet 1", "look", "valid"),
    ]
    for text, cmd, k in cases:
        r = agent.parse_action(text, adm)
        check(r == {"cmd": cmd, "kind": k}, f"{text[:40]!r} -> {r}")

    log("[selftest] Initial observation parsing and prefix stability")
    intro = ("-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you, "
             "you see a cabinet 1, a countertop 1.\n\nYour task is to: put a clean apple in fridge.")
    task, desc = agent.parse_intro(intro)
    check(task == "put a clean apple in fridge." and desc.startswith("You are in the middle"), f"Tasks={task!r}")
    hist = [{"cmd": f"go to cabinet {i}", "obs": f"You arrive at cabinet {i}."} for i in range(1, 6)]
    m4 = agent.build_messages(task, desc, hist[:4], ["look"])[1]["content"]
    m5 = agent.build_messages(task, desc, hist[:5], ["look", "inventory"])[1]["content"]
    cut = m4.index("\n\nYou are now at")
    check(m5.startswith(m4[:cut]), "A two-step reminder sharing history prefix（Enable prefix cache）")
    check("Thought" not in m5.split("Steps so far:")[1].split("You have used")[0], "History does not include Thought")

    hs = [{"obs": "You arrive at desk 1. On the desk 1, you see a alarmclock 1."},
          {"obs": "You pick up the alarmclock 1 from the desk 1."}, {"obs": "Nothing happens."},
          {"obs": "You arrive at sidetable 2. On the sidetable 2, you see a desklamp 1."}]
    check(agent.track_state(hs) == ("sidetable 2", "alarmclock 1"), f"position/Handheld Tracking {agent.track_state(hs)}")
    hs.append({"obs": "You move the alarmclock 1 to the sidetable 2."})
    check(agent.track_state(hs) == ("sidetable 2", None), "Put it down and hold it empty")
    check(agent.track_state([]) == (None, None), "Initial Status")
    m_state = agent.build_messages(task, desc, [dict(h, cmd="x") for h in hs], ["look"])[1]["content"]
    check("You are now at: sidetable 2. You are holding: nothing." in m_state, "The prompts contains the current position and the handheld")

    log("[selftest] mock environment：Planner executable、Best of all、Status Keys")
    n_ok = 0
    for i in range(30):
        env = mockenv.MockEnv()
        env.load(f"mock://{i}")
        st = env.reset()
        plan = env.plan()
        good = bool(plan)
        for c in plan:
            good &= c in st["admissible"]
            st = env.step(c)
        n_ok += int(good and st["won"])
    check(n_ok == 30, f"30 Plan to execute and reach the target in the Bureau simulation game：{n_ok}")
    env = mockenv.MockEnv()
    env.load("mock://3")
    env.reset()
    k0 = env.state_key()
    env.step("jump over the moon")
    check(env.state_key() == k0, "Invalid order does not change state")
    st = env.step(env.reset()["admissible"][0])
    check(env.state_key() != k0, "Legal move to change state")

    log("[selftest] Game Collection Filter")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / acfg.SPLIT_DIRS["valid_unseen"]

        def mk(rel, task_type="pick_and_place_simple", game=True, solvable=True, key=True):
            d = base / rel
            d.mkdir(parents=True)
            (d / "traj_data.json").write_text(json.dumps({"task_type": task_type}))
            if game:
                gd = {"solvable": solvable} if key else {}
                (d / "game.tw-pddl").write_text(json.dumps(gd))
        mk("pick_and_place_simple-Apple-None-Fridge-1/trial_1")
        mk("pick_heat_then_place_in_recep-Egg-None-Fridge-2/trial_2")
        mk("pick_and_place_simple-Apple-None-Fridge-3/trial_3", solvable=False)
        mk("pick_and_place_with_movable_recep-X/trial_4")
        mk("pick_and_place_simple-AppleSliced-None-Fridge-5/trial_5")
        mk("pick_and_place_simple-Apple-None-Fridge-6/trial_6", game=False)
        mk("pick_and_place_simple-Apple-None-Fridge-7/trial_7", key=False)
        mk("unknown_type-X/trial_8", task_type="unknown")
        got, stats = games.collect("valid_unseen", data_root=tmp)
        check(len(got) == 2 and stats["unsolvable"] == 1 and stats["movable_or_sliced"] == 2
              and stats["no_game"] == 1 and stats["no_solvable_key"] == 1 and stats["task_type"] == 1,
              f"Filter Statistics {stats}")
        check(got == sorted(got), "Results sorted")

    log("[selftest] Consistency of process with multi-process environment pool results")
    items = [{"rid": f"r{i}", "gamefile": f"mock://{i % 5}", "prefix": ["look", "jump"] * (i % 3),
              "want_key": True} for i in range(12)]
    with pool.InlinePool("mock") as p1, pool.ProcPool("mock", 2) as p2:
        o1, o2 = p1.open(items), p2.open(items)
        strip = lambda d: {k: {kk: vv for kk, vv in v.items() if kk not in ("t", "dt")} for k, v in d.items()}
        check(strip(o1) == strip(o2), "open identical")
        steps = [{"rid": it["rid"], "cmd": o1[it["rid"]]["state"]["admissible"][0], "want_key": True} for it in items]
        check(strip(p1.step(steps)) == strip(p2.step(steps)), "step identical")
        pl1, pl2 = p1.plan([it["rid"] for it in items]), p2.plan([it["rid"] for it in items])
        check({k: v["plan"] for k, v in pl1.items()} == {k: v["plan"] for k, v in pl2.items()}, "plan identical")
        p1.close([it["rid"] for it in items])
        p2.close([it["rid"] for it in items])
        probes = [{"gamefile": f"mock://{i % 4}", "prefix": ["look"] * i} for i in range(9)]
        pr1, pr2 = p1.probe(probes), p2.probe(probes)
        check([r["plan"] for r in pr1] == [r["plan"] for r in pr2], "probe Consistency and order of input")
        n_before = sum(s["n_envs"] for s in p2.stats())
        again = [dict(it, rid=it["rid"] + "b") for it in items]
        p2.open(again)
        p2.close([it["rid"] for it in again])
        sts = p2.stats()
        check(all(s["hashseed"] == acfg.HASH_SEED for s in sts), f"Subprocess PYTHONHASHSEED={[s['hashseed'] for s in sts]}")
        check(sum(s["n_envs"] for s in sts) == n_before, f"Turn on the reuse environment example after closing（Total {sum(s['n_envs'] for s in sts)} items）")

    log("[selftest] Batch rollout（mock model）")
    be = agent.MockBackend()
    specs = [{"rid": f"e{i}", "gamefile": f"mock://{100 + i}", "seed": agent.derive_seed("st", i)} for i in range(24)]
    with pool.ProcPool("mock", 2) as p:
        eps, tm = rollout.run_episodes(be, p, specs, temperature=0.7, keep_text=True)
        eps2, _ = rollout.run_episodes(be, p, specs, temperature=0.7)
    check(all(len(e["steps"]) <= acfg.MAX_STEPS for e in eps), "Step not to exceed maximum")
    check([[s["cmd"] for s in e["steps"]] for e in eps] == [[s["cmd"] for s in e["steps"]] for e in eps2],
          "Same sequence as the torrents rerun action")
    n_won = sum(e["won"] for e in eps)
    good = True
    for e in eps:
        env = mockenv.MockEnv()
        env.load(e["gamefile"])
        st = env.reset()
        won_at = None
        for j, s in enumerate(e["steps"]):
            st = env.step(s["cmd"])
            good &= st["obs"] == s["obs"]
            if st["won"] and won_at is None:
                won_at = j + 1
        good &= (won_at == e["won_step"]) and (bool(won_at) == e["won"])
    check(good, f"The moment of gradual observation and success can be repeated independently（Success {n_won}/24）")
    check(0 < n_won < 24, "Simulation models have both been successful and unsuccessful（Overwrite Two Paths）")
    kinds = {}
    for e in eps:
        for s in e["steps"]:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    first_bad = sum(1 for e in eps for s in e["steps"] if s.get("first_kind") in ("invalid", "no_action"))
    final_bad = sum(1 for e in eps for s in e["steps"] if s["kind"] in ("invalid", "no_action"))
    check(all(kinds.get(k, 0) > 0 for k in ("valid", "alias")) and first_bad > 0, f"all parser branches are covered {kinds}")
    check(final_bad < first_bad and tm["retries"] > 0,
          f"Unexecutable Reaction：Initial answer cannot be implemented {first_bad} steps，Unexecutable after answering again {final_bad} steps，retry {tm['retries']} runs")
    check(tm["policy_steps"] == sum(len(e["steps"]) for e in eps), "step counts agree")

    log("[selftest] Run from the middle of the trajectories（Prefix）")
    src = max(eps, key=lambda e: len(e["steps"]))
    pre = [s["cmd"] for s in src["steps"][:10]]
    with pool.InlinePool("mock") as p:
        cont, _ = rollout.run_episodes(be, p, [{"rid": "c", "gamefile": src["gamefile"], "seed": 7, "prefix": pre}],
                                       temperature=0.7)
    c = cont[0]
    check([s["obs"] for s in c["steps"][:10]] == [s["obs"] for s in src["steps"][:10]], "Prefix observations are consistent with the original trajectory")
    check(c["prefix_len"] == 10 and all(s["kind"] == "prefix" for s in c["steps"][:10]), "Prefix mark is correct")
    check(len(c["steps"]) <= acfg.MAX_STEPS, "Prefix to step budget")

    log("[selftest] omit the earliest history when the context is inadequate")
    small = agent.MockBackend(context=700)
    long_hist = [{"cmd": f"go to cabinet {i}", "obs": "You arrive at cabinet. On the cabinet, you see nothing." * 2}
                 for i in range(40)]
    pr, n, drop = rollout.fit_prompt(small, "t", "d", long_hist, ["look"] * 5, 64, 50)
    check(drop > 0 and n is not None and n <= 700 - 64 - 8, f"omitted {drop} Step back {n} token")

    log("[selftest] report generation（Simulation Data）")
    a4 = summarize_a4(eps, {"wall": 10.0, "gen": 6.0, "env": 3.0, "policy_steps": 500, "waves": 50,
                            "gen_tokens": 1000, "max_prompt_tokens": 100, "open": 0.1}, {}, {}, ["e0"])
    check("groups" in a4 and a4["groups"]["all"]["n"] == 24, "Summary statistics")
    log(f"[selftest] {'all passed' if ok else 'failures detected'}")
    return ok


# ---------------------------------------------------------------- A1
def part_a1():
    from alfw import games, twenv
    res = {"python": sys.version.split()[0], "platform": platform.platform(), "hashseed": os.environ.get("PYTHONHASHSEED"),
           "cpu": os.cpu_count(), "mem_gb": mem_gb(), "workers": acfg.n_workers(), "data": str(acfg.DATA),
           "mock": STATE["mock"], "errors": [], "warnings": []}
    log(f"[A1] Python {res['python']}，CPU {res['cpu']}，Memory {res['mem_gb']} GB，PYTHONHASHSEED={res['hashseed']}")
    vers = {}
    from importlib import metadata
    for pkg in ("alfworld", "textworld", "fast-downward-textworld", "vllm", "torch", "transformers", "numpy",
                "tatsu", "jericho"):
        try:
            vers[pkg] = metadata.version(pkg)
        except Exception:
            vers[pkg] = None
    res["versions"] = vers
    log(f"[A1] version {vers}")
    if not STATE["mock"]:
        for mod in ("textworld", "fast_downward", "alfworld.agents.environment.alfred_tw_env",
                    "textworld.envs.pddl"):
            try:
                importlib.import_module(mod)
            except Exception as e:
                res["errors"].append(f"import {mod}: {e!r}")
        logic = acfg.DATA / "logic"
        for f in ("alfred.pddl", "alfred.twl2"):
            if not (logic / f).exists():
                res["warnings"].append(f"Missing {logic / f}（alfworld-download will copy；Our game files bring their own logic，Does not affect running）")
        try:
            du = shutil.disk_usage(acfg.DATA)
            res["disk_free_gb"] = round(du.free / 1e9, 1)
        except OSError:
            res["errors"].append(f"Data Directory does not exist：{acfg.DATA}")
        from alfw import twenv
        tmpdir = twenv.use_data_disk_tmp()
        lib_mb = twenv.lib_size_mb()
        try:
            tfree = shutil.disk_usage(tmpdir).free / 1e9
        except OSError:
            tfree = None
        need = (lib_mb or 0) * 400 / 1e3          # approximately 400 Temporary occupation when an environmental example exists
        res["tmp"] = {"dir": tmpdir, "free_gb": None if tfree is None else round(tfree, 1),
                      "libdownward_mb": None if lib_mb is None else round(lib_mb, 1), "need_gb_for_400_envs": round(need, 1)}
        log(f"[A1] Temporary Directory {tmpdir}：Remaining {res['tmp']['free_gb']} GB；libdownward.so {res['tmp']['libdownward_mb']} MB/Example，"
            f"400 EXPERIENCE REQUIREMENTS {need:.1f} GB")
        if tfree is not None and tfree < need:
            res["errors"].append(f"Temporary Directory {tmpdir} insufficient free space（{tfree:.1f} GB < {need:.1f} GB）；use ALFW_TMP point to a disk with more space")
    if res["errors"]:
        for e in res["errors"]:
            log(f"[A1] ✗ {e}")
        res["ok"] = False
        save("a1.json", res)
        return False

    counts, ttypes = {}, {}
    for split in acfg.SPLIT_DIRS:
        try:
            gs = split_games(split)
        except Exception as e:
            res["errors"].append(f"{split}: {e!r}")
            continue
        counts[split] = len(gs)
        tt = {}
        for g in gs:
            t = games.task_type_of(g)
            tt[t] = tt.get(t, 0) + 1
        ttypes[split] = tt
        exp = acfg.EXPECTED_COUNTS[split]
        flag = "" if STATE["mock"] or abs(len(gs) - exp) <= 0.02 * exp else f"（Expected {exp}）"
        if flag:
            res["warnings"].append(f"{split} Number of games {len(gs)}{flag}")
        log(f"[A1] {split}: {len(gs)} episodes{flag}  Task Type {tt}")
    res["counts"], res["task_types"] = counts, ttypes

    try:
        g = split_games("valid_unseen")[0]
        env = twenv.make_env(kind())
        t0 = time.perf_counter()
        env.load(g)
        t1 = time.perf_counter()
        st = env.reset()
        t2 = time.perf_counter()
        res["smoke"] = {"game": games.short_name(g), "t_load": t1 - t0, "t_reset": t2 - t1,
                        "intro": st["obs"][:1200], "n_admissible": len(st["admissible"]),
                        "admissible_head": st["admissible"][:15]}
        for c in ("look", "inventory", "jump over the moon"):
            ts = time.perf_counter()
            s2 = env.step(c)
            res["smoke"][f"step:{c}"] = {"obs": s2["obs"][:300], "dt": time.perf_counter() - ts}
        ts = time.perf_counter()
        plan = env.plan()
        res["smoke"]["plan"] = plan
        res["smoke"]["t_plan"] = time.perf_counter() - ts
        log(f"[A1] Single game：Load {t1 - t0:.2f}s，Reset {t2 - t1:.2f}s，Executable command {len(st['admissible'])} records，"
            f"Planning {len(plan)} steps（{res['smoke']['t_plan']:.2f}s）")
        log("[A1] Initial observations：\n" + st["obs"][:600])
        log(f"[A1] Planning：{plan}")
        if not plan:
            res["errors"].append("Initial status planner returns empty plan")
    except Exception:
        res["errors"].append("single-episode smoke test failed：\n" + traceback.format_exc())
    res["ok"] = not res["errors"] and all(counts.get(s, 0) > 0 for s in acfg.SPLIT_DIRS)
    for e in res["errors"]:
        log(f"[A1] ✗ {e}")
    for w in res["warnings"]:
        log(f"[A1] ! {w}")
    save("a1.json", res)
    log(f"[A1] {'passed' if res['ok'] else 'failed'}")
    return res["ok"]


# ---------------------------------------------------------------- A2
def _replay_all(env, walks, plan_at=()):
    from alfw import twenv
    return [twenv.replay(env, w["gamefile"], w["cmds"], plan_at=plan_at) for w in walks]


def _same(walks, reps):
    """compare fingerprints episode by episode，Back (All the same, Inconsistent Bureau and First Inconsistencies)。"""
    bad = []
    for w, r in zip(walks, reps):
        if r["hashes"] != w["hashes"]:
            j = next((i for i, (a, b) in enumerate(zip(w["hashes"], r["hashes"])) if a != b), None)
            bad.append({"game": w["name"], "first_diff_step": j})
    return not bad, bad


def _subprocess_replay(walks, patches, hashseed, keep):
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.json", Path(tmp) / "out.json"
        fin.write_text(json.dumps({"patches": patches, "walks": [{"gamefile": w["gamefile"], "cmds": w["cmds"]}
                                                                  for w in walks]}))
        cmd = [sys.executable, str(SCRIPT), "--part", "replay-worker", "--in", str(fin), "--out", str(fout)]
        if STATE["mock"]:
            cmd.append("--mock")
        if keep:
            cmd.append("--keep-hashseed")
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        if p.returncode != 0:
            raise RuntimeError(f"Subprocess Replay Failed：\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
        return json.loads(fout.read_text())


def part_replay_worker(fin, fout):
    from alfw import twenv
    d = json.loads(Path(fin).read_text())
    if d.get("patches") and not STATE["mock"]:
        twenv.install_patches(translate=True, grammar=True)
    env = twenv.make_env(kind())
    reps = [twenv.replay(env, w["gamefile"], w["cmds"]) for w in d["walks"]]
    Path(fout).write_text(json.dumps({"hashseed": os.environ.get("PYTHONHASHSEED"),
                                      "hashes": [r["hashes"] for r in reps]}))


def part_a2():
    from alfw import agent, games, twenv
    gs = games.sample(split_games("train"), acfg.A2_N_GAMES, acfg.A2_SEED)
    log(f"[A2] {len(gs)} episodes × {acfg.A2_WALK} Walk randomly（References：No acceleration patches、Load the same example sequentially）")
    if not STATE["mock"]:
        twenv.install_patches(translate=False, grammar=False)
    env = twenv.make_env(kind())
    walks, t_load, t_reset, t_step = [], [], [], []
    for gi, g in enumerate(gs):
        rng = random.Random(agent.derive_seed("A2", gi))
        t0 = time.perf_counter()
        env.load(g)
        t1 = time.perf_counter()
        st = env.reset()
        t2 = time.perf_counter()
        t_load.append(t1 - t0)
        t_reset.append(t2 - t1)
        cmds, hashes, keys = [], [twenv.transcript_hash(st)], [env.state_key()]
        n_adm = [len(st["admissible"])]
        for _ in range(acfg.A2_WALK):
            c = walk_cmd(rng, st)
            ts = time.perf_counter()
            st = env.step(c)
            t_step.append(time.perf_counter() - ts)
            cmds.append(c)
            hashes.append(twenv.transcript_hash(st))
            keys.append(env.state_key())
            n_adm.append(len(st["admissible"]))
        walks.append({"gamefile": g, "name": games.short_name(g), "cmds": cmds, "hashes": hashes, "keys": keys,
                      "n_admissible": n_adm})
    res = {"games": [w["name"] for w in walks], "timing_unpatched": {
        "load": qstats(t_load), "reset": qstats(t_reset), "step": qstats(t_step)},
        "n_admissible": qstats([n for w in walks for n in w["n_admissible"]])}
    log(f"[A2] Not accelerated：Load {res['timing_unpatched']['load']}，Reset {res['timing_unpatched']['reset']}，"
        f"One step {res['timing_unpatched']['step']}")
    checks = {}

    # c1 New examples per game
    r0 = rss_mb()
    fresh = [twenv.make_env(kind()) for _ in walks]
    r1 = rss_mb()
    reps = [twenv.replay(e, w["gamefile"], w["cmds"]) for e, w in zip(fresh, walks)]
    r2 = rss_mb()
    checks["fresh_instance"] = _same(walks, reps)
    res["rss_per_env_mb"] = {"construct": (r1 - r0) / len(walks), "after_load": (r2 - r0) / len(walks)}
    # c2 Same example reset Play it again
    reps = [twenv.replay(e, w["gamefile"], w["cmds"]) for e, w in zip(fresh, walks)]
    checks["reset_reuse"] = _same(walks, reps)
    del fresh
    # Leak：The same example is loaded over and over again between different games
    e = twenv.make_env(kind())
    rs = []
    for k in range(2):
        for w in walks:
            e.load(w["gamefile"])
            e.reset()
            rs.append(rss_mb())
    res["rss_growth_per_load_mb"] = (rs[-1] - rs[len(walks) - 1]) / len(walks)

    # c3 Accelerating patches
    if STATE["mock"]:
        checks["patched"] = (True, [])
        res["timing_patched"] = res["timing_unpatched"]
    else:
        twenv.install_patches(translate=True, grammar=True)
        pe = twenv.make_env(kind())
        first = _replay_all(pe, walks)
        second = _replay_all(pe, walks)
        okf, badf = _same(walks, first)
        oks, bads = _same(walks, second)
        checks["patched"] = (okf and oks, badf + bads)
        res["timing_patched"] = {"load": qstats([r["t_load"] for r in first]),
                                 "reset": qstats([r["t_reset"] for r in second]),
                                 "step": qstats([t for r in second for t in r["step_times"]])}
        log(f"[A2] After the acceleration,：Reset {res['timing_patched']['reset']}，One step {res['timing_patched']['step']}")

    # c4 / c5 Subprocess
    sub = _subprocess_replay(walks, True, acfg.HASH_SEED, keep=False)
    checks["subprocess_same_seed"] = _same(walks, [{"hashes": h} for h in sub["hashes"]])
    try:
        sub2 = _subprocess_replay(walks, True, "12345", keep=True)
        res["subprocess_other_seed_identical"] = _same(walks, [{"hashes": h} for h in sub2["hashes"]])[0]
        res["subprocess_other_seed_hashseed"] = sub2["hashseed"]
    except Exception as ex:
        res["subprocess_other_seed_identical"] = None
        res["subprocess_other_seed_error"] = str(ex)[-500:]

    res["checks"] = {k: {"identical": v[0], "mismatches": v[1]} for k, v in checks.items()}
    res["ok"] = all(v[0] for v in checks.values())
    for k, v in res["checks"].items():
        log(f"[A2] {'✓' if v['identical'] else '✗'} {k}" + ("" if v["identical"] else f" different {v['mismatches'][:5]}"))
    log(f"[A2] different PYTHONHASHSEED And if the sub-process is consistent or not?（Just an explanation，fixed seeds for formal runs）：{res.get('subprocess_other_seed_identical')}")
    log(f"[A2] Memory of each environmental example {res['rss_per_env_mb']['after_load']:.1f} MB；Reload Every Growth "
        f"{res['rss_growth_per_load_mb']:.2f} MB")
    save("a2.json", res)
    save("a2_walks.json", walks)
    log(f"[A2] {'passed' if res['ok'] else 'failed'}")
    return res["ok"]


# ---------------------------------------------------------------- A3
PATCHES_ON = {"translate": True, "grammar": True}


def part_a3():
    """Planner check。It takes a long time to plan（Dozens of seconds for individual status），All in parallel with the environmental sub-process。"""
    from alfw import pool, twenv
    walks = load("a2_walks.json")
    if walks is None:
        log("[A3] You need to run first a2（reuse its random walks）")
        return False
    n = acfg.n_workers()
    t_start = time.perf_counter()
    res = {"probe_steps": list(acfg.A3_PROBE_STEPS), "workers": n}
    probes = [{"gamefile": w["gamefile"], "prefix": w["cmds"][:s], "verify": True, "game": w["name"], "step": s}
              for w in walks for s in acfg.A3_PROBE_STEPS]
    out = {}
    for name, patches in (("on", PATCHES_ON), ("off", None)):
        if STATE["mock"] and name == "off":
            out["off"] = out["on"]
            continue
        with pool.make_pool(kind(), n, None if STATE["mock"] else patches) as p:
            t0 = time.perf_counter()
            out[name] = p.probe(probes)
            log(f"[A3] Accelerating patches{'on' if name == 'on' else 'off'}：{len(probes)} Time for a detection point {time.perf_counter() - t0:.0f}s")
            if name == "on":
                # Call the planner without changing the subsequent transfer：_Other Organiser plan() Replaying is progressively consistent with reference
                reps = p.replay([{"gamefile": w["gamefile"], "cmds": w["cmds"], "plan_at": list(acfg.A3_PROBE_STEPS)}
                                 for w in walks])
                # Cost coherence（The best circumstantial evidence）：Every step of the way
                bw = walks[:acfg.A3_BELLMAN_GAMES]
                bitems = [{"gamefile": w["gamefile"], "prefix": w["cmds"][:t]} for w in bw for t in range(len(w["cmds"]) + 1)]
                t0 = time.perf_counter()
                bres = p.probe(bitems)
                log(f"[A3] Cost coherence：{len(bitems)} Secondary planning hours {time.perf_counter() - t0:.0f}s")
    rows = []
    for pr, on, off in zip(probes, out["on"], out["off"]):
        if on["won"]:
            continue
        rows.append({"game": pr["game"], "step": pr["step"], "len": len(on["plan"]), "cost": twenv.plan_cost(on["plan"]),
                     "dt": on["dt_plan"], "dt_off": off["dt_plan"], "admissible_ok": bool(on["plan"]) and on["admissible_ok"],
                     "reaches_goal": on["reaches_goal"], "same_as_unpatched": on["plan"] == off["plan"]})
    valid = [r["admissible_ok"] and r["reaches_goal"] for r in rows]
    res["n_probes"] = len(rows)
    res["valid_frac"] = sum(valid) / max(1, len(valid))
    res["latency"] = qstats([r["dt"] for r in rows])
    res["latency_off"] = qstats([r["dt_off"] for r in rows])
    res["plan_len"] = qstats([r["len"] for r in rows])
    res["patched_plan_identical_frac"] = sum(r["same_as_unpatched"] for r in rows) / max(1, len(rows))
    res["slowest"] = sorted(({k: r[k] for k in ("game", "step", "len", "dt")} for r in rows), key=lambda r: -r["dt"])[:5]
    res["failures"] = [{k: r[k] for k in ("game", "step", "len", "admissible_ok", "reaches_goal")}
                       for r in rows if not (r["admissible_ok"] and r["reaches_goal"])][:20]
    log(f"[A3] {len(rows)} A detection site：The plan can be implemented and the target reached {res['valid_frac']:.3f}；Time-consuming {res['latency']}；"
        f"Length of the plan {res['plan_len']}；The plan is the same before and after the accelerated patches {res['patched_plan_identical_frac']:.3f}")

    ok_side, bad = _same(walks, reps)
    res["plan_no_side_effect"] = ok_side
    log(f"[A3] {'✓' if ok_side else '✗'} Call the planner without changing the subsequent transfer" + ("" if ok_side else f" {bad[:3]}"))

    viol, n_pairs, len_viol = 0, 0, 0
    k = 0
    for w in bw:
        ps = []
        for t in range(len(w["cmds"]) + 1):
            r = bres[k]
            k += 1
            ps.append(None if r["won"] else r["plan"])
        for t in range(len(w["cmds"])):
            if not ps[t] or ps[t + 1] is None:
                continue
            ct, cn = twenv.plan_cost(ps[t]), twenv.plan_cost(ps[t + 1])
            same_state = w["keys"][t] == w["keys"][t + 1]
            step_cost = 0 if same_state else twenv.plan_cost([w["cmds"][t]])
            n_pairs += 1
            viol += int(ct > step_cost + cn)
            len_viol += int(len(ps[t]) > (0 if same_state else 1) + len(ps[t + 1]))
    res["bellman"] = {"pairs": n_pairs, "cost_violations": viol, "length_violations": len_viol}
    log(f"[A3] Cost coherence：{n_pairs} Infractions in neighbouring states {viol} runs（Step-gauge violation {len_viol} runs；"
        "Breach of the Note Planner is not the best，Right now“within budget”is a conservative assessment）")
    g = acfg.GATES
    res["ok"] = (res["valid_frac"] >= g["a3_valid_frac"] and res["latency"].get("p50", 1e9) <= g["a3_median_latency_s"]
                 and ok_side)
    res["wall"] = time.perf_counter() - t_start
    save("a3.json", res)
    log(f"[A3] {'passed' if res['ok'] else 'failed'}（Time {res['wall']:.0f}s）")
    return res["ok"]


# ---------------------------------------------------------------- A4
def summarize_a4(eps, timing, l0, slack, train_rids):
    from alfw import games
    from alfw.rollout import episode_stats

    def group(es):
        n = len(es)
        k = sum(e["won"] for e in es)
        st = [episode_stats(e) for e in es]
        steps = sum(s["n"] for s in st) or 1
        return {"n": n, "success": k, "rate": k / max(1, n), "ci90": wilson(k, n),
                "steps_success": qstats([e["won_step"] for e in es if e["won"]]),
                "invalid": sum(s["invalid"] for s in st) / steps, "no_action": sum(s["no_action"] for s in st) / steps,
                "alias": sum(s["alias"] for s in st) / steps, "trunc": sum(s["trunc"] for s in st) / steps,
                "first_bad": sum(s["first_bad"] for s in st) / steps, "retries_per_step": sum(s["retries"] for s in st) / steps,
                "hist_drop": sum(s["drop"] for s in st) / steps, "revisit": sum(s["revisit"] for s in st) / steps,
                "repeat": sum(s["repeat"] for s in st) / steps, "max_prompt_tokens": max([s["max_ntok"] for s in st] or [0])}

    tr = set(train_rids)
    out = {"groups": {"all": group(eps), "train": group([e for e in eps if e["rid"] in tr]),
                      "valid_unseen": group([e for e in eps if e["rid"] not in tr])}, "by_task": {}}
    for split, es in (("train", [e for e in eps if e["rid"] in tr]), ("valid_unseen", [e for e in eps if e["rid"] not in tr])):
        bt = {}
        for e in es:
            bt.setdefault(games.task_type_of(e["gamefile"]), []).append(e)
        out["by_task"][split] = {t: {"n": len(v), "rate": sum(x["won"] for x in v) / len(v)} for t, v in sorted(bt.items())}
    # The relationship between the initial optimal length and success
    if l0:
        succ = [l0[e["rid"]] for e in eps if e["won"] and e["rid"] in l0]
        fail = [l0[e["rid"]] for e in eps if not e["won"] and e["rid"] in l0]
        out["l0"] = {"success": qstats(succ), "failure": qstats(fail)}
    if slack:
        curve = {}
        for rid, pts in slack.items():
            for t, s in pts.items():
                curve.setdefault(int(t), []).append(s)
        out["slack_curve"] = {t: {"n": len(v), "feasible_frac": sum(1 for x in v if x >= 0) / len(v),
                                  "median_slack": statistics.median(v)} for t, v in sorted(curve.items())}
        last = []
        for rid, pts in slack.items():
            fe = [int(t) for t, s in pts.items() if s >= 0]
            last.append(max(fe) if fe else None)
        out["last_feasible_probe"] = qstats([x for x in last if x is not None])
        out["never_feasible_frac"] = sum(1 for x in last if x is None) / max(1, len(last))
    out["timing"] = timing
    ps = max(1, timing.get("policy_steps", 0))
    out["sec_per_policy_step"] = timing["wall"] / ps
    out["hours_per_100k_steps"] = timing["wall"] / ps * 1e5 / 3600
    out["nominal_stage_b_hours"] = timing["wall"] / ps * acfg.NOMINAL_B_STEPS / 3600
    return out


def part_a4(n_train=None, n_valid=None):
    from alfw import agent, games, pool, rollout
    a2 = load("a2.json")
    patched_ok = bool(a2 and a2.get("checks", {}).get("patched", {}).get("identical"))
    patches = PATCHES_ON if (patched_ok and not STATE["mock"]) else None
    n_train = n_train or acfg.A4_N_TRAIN
    train = games.sample(split_games("train"), n_train, acfg.A4_SEED)
    vu = split_games("valid_unseen")
    if n_valid is not None:
        vu = vu[:n_valid]
    specs = [{"rid": f"tr{i:03d}", "gamefile": g, "seed": agent.derive_seed("A4", g)} for i, g in enumerate(train)]
    specs += [{"rid": f"vu{i:03d}", "gamefile": g, "seed": agent.derive_seed("A4", g)} for i, g in enumerate(vu)]
    train_rids = [s["rid"] for s in specs if s["rid"].startswith("tr")]
    log(f"[A4] {len(train)} episodes train + {len(vu)} episodes valid_unseen，temperature {acfg.TEMPERATURE}，At most {acfg.MAX_STEPS} steps；"
        f"Environment subprocess {acfg.n_workers()} items，Accelerating patches {'on' if patches else 'off'}")
    # The same setup already exists rollout Reuse directly when the result is（Recalculating planning detection and aggregation only，Not anymore GPU）
    key = {"specs": [[s["rid"], s["gamefile"], s["seed"]] for s in specs], "T": acfg.TEMPERATURE,
           "max_steps": acfg.MAX_STEPS, "max_new": acfg.MAX_NEW_TOKENS, "model": str(acfg.MODEL_DIR),
           "agent_sha": hashlib.sha256((Path(__file__).parent / "alfw" / "agent.py").read_bytes()).hexdigest(),
           "rollout_sha": hashlib.sha256((Path(__file__).parent / "alfw" / "rollout.py").read_bytes()).hexdigest(),
           "mock": STATE["mock"] or STATE["mock_llm"]}
    ep_path, tm_path = STATE["out"] / "a4_episodes.jsonl", STATE["out"] / "a4_rollout_meta.json"
    eps = None
    t_model = 0.0
    if ep_path.exists() and tm_path.exists():
        meta = json.loads(tm_path.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            eps = [json.loads(x) for x in ep_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            timing, t_model = meta["timing"], meta.get("t_model_load", 0.0)
            log(f"[A4] Reuse already available {len(eps)} episodes rollout（Set the same），Recalculating planning detection and aggregation only")
    with pool.make_pool(kind(), acfg.n_workers(), patches) as p:
        if eps is None:
            t0 = time.perf_counter()
            backend = agent.make_backend("mock" if (STATE["mock"] or STATE["mock_llm"]) else "vllm")
            t_model = time.perf_counter() - t0
            log(f"[A4] Load Model {t_model:.0f}s")
            eps, timing = rollout.run_episodes(backend, p, specs, temperature=acfg.TEMPERATURE, keep_text=True, log=log)
            log(f"[A4] rollout complete：{timing['wall']:.0f}s（Generate {timing['gen']:.0f}s，environment {timing['env']:.0f}s，"
                f"{timing['policy_steps']} steps）")
            with open(ep_path, "w", encoding="utf-8") as f:         # save to disk first，no need to rerun if later planner probes fail GPU
                for e in eps:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            tm_path.write_text(json.dumps({"key": key, "timing": timing, "t_model_load": t_model}), encoding="utf-8")
            del backend
        # Planning length of initial status；Losing train On the board 5 Legacy of steps。some planner calls timed out，Timeout as unknown
        items = [{"gamefile": s["gamefile"], "prefix": [], "timeout": acfg.PLAN_TIMEOUT} for s in specs]
        keys = [("l0", s["rid"], 0) for s in specs]
        trs = set(train_rids)
        for e in eps:
            if e["won"] or e["rid"] not in trs:
                continue
            cmds = [s["cmd"] for s in e["steps"]]
            for t in range(0, acfg.MAX_STEPS, acfg.A4_SLACK_EVERY):
                items.append({"gamefile": e["gamefile"], "prefix": cmds[:t], "timeout": acfg.PLAN_TIMEOUT})
                keys.append(("slack", e["rid"], t))
        log(f"[A4] Planning for detection {len(items)} runs（{p.n} worker processes，Single timeout {acfg.PLAN_TIMEOUT}s）")
        t1 = time.perf_counter()
        pr = []
        B = 128
        for j in range(0, len(items), B):
            pr += p.probe(items[j:j + B])
            log(f"  Planning for detection {min(j + B, len(items))}/{len(items)}，Time {time.perf_counter() - t1:.0f}s，"
                f"timeout {sum(r['timeout'] for r in pr)}")
        t_probe = time.perf_counter() - t1
        l0, slack, n_timeout = {}, {}, 0
        for (kind_, rid, t), r in zip(keys, pr):
            if r["timeout"]:
                n_timeout += 1
                continue
            if kind_ == "l0":
                l0[rid] = len(r["plan"])
            else:
                slack.setdefault(rid, {})[t] = (acfg.MAX_STEPS - t) - len(r["plan"]) if not r["won"] else 0
        wstats = p.stats()
    summ = summarize_a4(eps, timing, l0, slack, train_rids)
    summ["t_model_load"] = t_model
    summ["t_probe"] = t_probe
    summ["n_probe"] = len(items)
    summ["n_probe_timeout"] = n_timeout
    summ["probe_latency"] = qstats([r["dt_plan"] for r in pr if not r["timeout"]])
    summ["workers"] = wstats
    summ["patches"] = bool(patches)
    summ["train_games"] = [games.short_name(g) for g in train]
    summ["prompt_example"] = None
    g = acfg.GATES
    tr = summ["groups"]["train"]
    summ["gates"] = {
        "success_in_range": g["a4_success_min"] <= tr["rate"] <= g["a4_success_max"],
        "bad_action_ok": tr["invalid"] + tr["no_action"] <= g["a4_bad_action_max"],
        "trunc_ok": tr["trunc"] <= g["a4_trunc_max"],
        "hist_drop_ok": tr["hist_drop"] <= g["a4_hist_drop_max"],
    }
    summ["ok"] = all(summ["gates"].values())
    # Record a real prompts，Enable manual check formatting
    e0 = eps[0]
    from alfw.agent import build_messages
    summ["prompt_example"] = build_messages(e0["task"], e0["desc"], e0["steps"][:3], ["(admissible list)"])
    save("a4.json", summ)
    for name in ("train", "valid_unseen"):
        gg = summ["groups"][name]
        log(f"[A4] {name}: Success {gg['success']}/{gg['n']} = {gg['rate']:.3f} 90%CI [{gg['ci90'][0]:.3f}, {gg['ci90'][1]:.3f}]；"
            f"Invalid {gg['invalid']:.3f}，No Action {gg['no_action']:.3f}，truncation {gg['trunc']:.3f}，"
            f"Go back to the status quo ante {gg['revisit']:.3f}，Maximum prompts {gg['max_prompt_tokens']} token")
    log(f"[A4] each 10 A thousand steps {summ['hours_per_100k_steps']:.2f} hours；Nominal phase B approximately {summ['nominal_stage_b_hours']:.1f} hours")
    log(f"[A4] judgments {summ['gates']} -> {'passed' if summ['ok'] else 'failed'}")
    return summ["ok"]


# ---------------------------------------------------------------- report
def part_report():
    a1, a2, a3, a4 = (load(n) for n in ("a1.json", "a2.json", "a3.json", "a4.json"))
    L = ["# ALFWorld stage A：Feasibility inspection report", ""]
    if STATE["mock"]:
        L += ["> **Simulate Run**：The environment and the model are simulated，Check process only，The numbers dont mean anything。", ""]
    L += ["The determination criteria are fixed before running（alfw/acfg.py of GATES；analysis plan.md 6.3a）。", "",
          "| Part | content | results |", "|---|---|---|"]
    st = lambda d: "—（not run）" if d is None else ("✓ passed" if d.get("ok") else "✗ failed")
    L.append(f"| A1 | installation、data、single-episode smoke test | {st(a1)} |")
    L.append(f"| A2 | Replay certainty（There are four ways of coming together）And time-consuming | {st(a2)} |")
    L.append(f"| A3 | planner：Rate of implementation ≥ {acfg.GATES['a3_valid_frac']}、Median number of hours spent ≤ {acfg.GATES['a3_median_latency_s']}s、No side effects | {st(a3)} |")
    L.append(f"| A4 | 7B Baseline：train success rate ∈ [{acfg.GATES['a4_success_min']}, {acfg.GATES['a4_success_max']}]、"
             f"Invalid+No Action ≤ {acfg.GATES['a4_bad_action_max']}、truncation ≤ {acfg.GATES['a4_trunc_max']} | {st(a4)} |")
    L.append("")
    if a1:
        L += ["## A1 Install and Data", "",
              f"- Python {a1.get('python')}；CPU {a1.get('cpu')}；Memory {a1.get('mem_gb')} GB；Disk Remains {a1.get('disk_free_gb')} GB；PYTHONHASHSEED={a1.get('hashseed')}",
              f"- version：{a1.get('versions')}",
              f"- Number of games：{a1.get('counts')}"]
        for split, tt in (a1.get("task_types") or {}).items():
            L.append(f"  - {split}：{tt}")
        sm = a1.get("smoke") or {}
        if sm:
            L += [f"- Single game（{sm.get('game')}）：Load {sm.get('t_load', 0):.2f}s，Reset {sm.get('t_reset', 0):.2f}s，"
                  f"Executable command {sm.get('n_admissible')} records，Planning {len(sm.get('plan') or [])} steps（{sm.get('t_plan', 0):.2f}s）",
                  "", "```", sm.get("intro", ""), "```", "", f"Planning：`{sm.get('plan')}`"]
        for e in a1.get("errors", []):
            L.append(f"- ✗ {e}")
        for w in a1.get("warnings", []):
            L.append(f"- ! {w}")
        L.append("")
    if a2:
        L += ["## A2 Replay certainty and time-consuming", "", "| check | identical |", "|---|---|"]
        names = {"fresh_instance": "Replay examples of new environments per board", "reset_reuse": "Same example reset Play it again",
                 "patched": "Accelerating patches（Translation Cache + Syntax Cache）", "subprocess_same_seed": "New process（Same PYTHONHASHSEED）"}
        for k, v in a2["checks"].items():
            L.append(f"| {names.get(k, k)} | {'✓' if v['identical'] else '✗ ' + str(v['mismatches'][:3])} |")
        L += ["", f"- different PYTHONHASHSEED Consistency of the new process（Description；Regularly Run As 0）：{a2.get('subprocess_other_seed_identical')}",
              f"- Time-consuming（seconds，Not accelerated）：Load {_fmt(a2['timing_unpatched']['load'])}；Reset {_fmt(a2['timing_unpatched']['reset'])}；One step {_fmt(a2['timing_unpatched']['step'])}",
              f"- Time-consuming（seconds，After the acceleration,）：Reset {_fmt(a2['timing_patched']['reset'])}；One step {_fmt(a2['timing_patched']['step'])}",
              f"- Number of enforceable orders：{_fmt(a2['n_admissible'])}",
              f"- Memory of each environmental example {a2['rss_per_env_mb']['after_load']:.1f} MB；Reload every increase in the same case {a2['rss_growth_per_load_mb']:.2f} MB", ""]
    if a3:
        L += ["## A3 planner", "",
              f"- {a3['n_probes']} A detection site（20 the specified step of a random-walk episode {a3['probe_steps']} steps）：The plan can be implemented and the target reached **{a3['valid_frac']:.3f}**",
              f"- Individual planning takes time（seconds）：{_fmt(a3['latency'])}（Not accelerated {_fmt(a3['latency_off'])}）",
              f"- Length of the plan：{_fmt(a3['plan_len'])}；The same proportion of the plan before and after accelerated patches {a3['patched_plan_identical_frac']:.3f}",
              f"- Call the planner without changing the subsequent transfer：{'✓' if a3['plan_no_side_effect'] else '✗'}",
              f"- Cost coherence（Best side testimony）：{a3['bellman']}",
              ]
        if a3.get("slowest"):
            L.append(f"- Slowest planning：{a3['slowest']}")
        if a3.get("failures"):
            L.append(f"- Examples of failure：{a3['failures'][:5]}")
        L.append("")
    if a4:
        L += ["## A4 7B Baseline（temperature 0.7，At most 50 steps）", "",
              "“Initial answer cannot be implemented”= Before answering again,；“Invalid / No Action”= Unexecutable after answering again、Percentage of actual delivery to environment and consumption of step。", "",
              "| Division | Number of bureaux | success rate | 90% interval | Initial answer cannot be implemented | Number of responses per step | Invalid | No Action | Alias | truncation | Go back to the old state | Repeated consecutively | Maximum prompts |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in ("train", "valid_unseen", "all"):
            g = a4["groups"][name]
            L.append(f"| {name} | {g['n']} | {g['rate']:.3f} | [{g['ci90'][0]:.3f}, {g['ci90'][1]:.3f}] | {g.get('first_bad', 0):.3f} | "
                     f"{g.get('retries_per_step', 0):.3f} | {g['invalid']:.3f} | "
                     f"{g['no_action']:.3f} | {g['alias']:.3f} | {g['trunc']:.3f} | {g['revisit']:.3f} | {g['repeat']:.3f} | {g['max_prompt_tokens']} |")
        L += ["", "Success rates by mission type："]
        for split, bt in a4["by_task"].items():
            L.append(f"- {split}：" + "；".join(f"{t} {v['rate']:.2f}（{v['n']}）" for t, v in bt.items()))
        if a4.get("l0"):
            L.append(f"- Optimal Plan Length for Initial Status：Success {_fmt(a4['l0']['success'])}；Failed game {_fmt(a4['l0']['failure'])}")
        if a4.get("slack_curve"):
            L += ["", "Losing train Commissioner，number t Step“Remaining Step(s) − Length of planning”（slack；≥ 0 i.e., within budget）：", "",
                  "| t | Number of bureaux | Completable Ratio | Median number of leftovers |", "|---:|---:|---:|---:|"]
            for t, v in a4["slack_curve"].items():
                L.append(f"| {t} | {v['n']} | {v['feasible_frac']:.2f} | {v['median_slack']} |")
            L.append(f"\n- The last step to complete the detection（each 5 Step by step）：{_fmt(a4.get('last_feasible_probe'))}；Percentage that cannot be completed from the beginning {a4.get('never_feasible_frac', 0):.2f}")
        tm = a4["timing"]
        L += ["", f"- Time-consuming：Load Model {a4.get('t_model_load', 0):.0f}s；rollout {tm['wall']:.0f}s（Generate {tm['gen']:.0f}s，environment {tm['env']:.0f}s，"
                  f"Open {tm['open']:.0f}s），Total {tm['policy_steps']} steps、{tm['waves']} Wheel；Planning for detection {a4.get('n_probe')} runs {a4.get('t_probe', 0):.0f}s"
                  f"（timeout {a4.get('n_probe_timeout', 0)} runs；Single time-consuming {_fmt(a4.get('probe_latency'))}）",
              f"- **Arithmetic extrapolation**：each 10 A thousand steps {a4['hours_per_100k_steps']:.2f} hours；Nominal phase B（{acfg.NOMINAL_B_STEPS:,} steps）approximately {a4['nominal_stage_b_hours']:.1f} hours"
              f"（This is a parallel push {a4['groups']['all']['n']} records；stage B advance more in parallel、Share more prefixes，should be faster in practice）",
              f"- judgments：{a4['gates']}", ""]
    L += ["## Conclusions under the pre-rule", ""] + verdict(a1, a2, a3, a4)
    path = STATE["out"] / "stageA_report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"[report] Written {path}")
    return True


def _fmt(q):
    if not q or not q.get("n"):
        return "—"
    f = lambda x: f"{x:.3g}" if isinstance(x, float) else str(x)
    return f"Mean {f(q['mean'])}，Median {f(q['p50'])}，p95 {f(q['p95'])}，Max {f(q['max'])}（n={q['n']}）"


def verdict(a1, a2, a3, a4):
    out = []
    if a1 is None or not a1.get("ok"):
        return ["- Environment not ready（A1）：Install or data as per bug in the report first，Well run again。"]
    if a2 and not a2.get("ok"):
        out.append("- **A2 failed：Replay is uncertain**。stage B Dependence“Replay prefix recovery state”，investigate first（inspect mismatched episodes and their first mismatched step）；"
                   "If you cant fix it in a day,，Changed in accordance with the rules TextCraft。")
    if a3 and not a3.get("ok"):
        out.append("- **A3 failed：Planner is unreliable or too slow**。oracle Success and feasibility F use an approximation instead（If the solution is known），Phase to be modified B Well register after design。")
    if a4:
        tr = a4["groups"]["train"]
        gs = a4["gates"]
        if tr["rate"] < acfg.GATES["a4_success_min"]:
            out.append(f"- **Floor effects risk**：train success rate {tr['rate']:.3f} < {acfg.GATES['a4_success_min']}。By the pre-rule：Just one prompts revision"
                       "（Add a model trajectory），Rerun A4；Reuse if still below the threshold TextCraft or stronger implementation models。")
        elif tr["rate"] > acfg.GATES["a4_success_max"]:
            out.append(f"- success rate {tr['rate']:.3f} too high：There are too few failures，stage B more episodes are needed to obtain enough failed trajectories。")
        if not gs["bad_action_ok"] or not gs["trunc_ok"] or not gs["hist_drop_ok"]:
            out.append("- **Agreement issues**：Invalid/No Action、Intercept or History-Extracted Ratio。Check it out first a4_episodes.jsonl Original Output in，Rerun after correcting parsing or prompts A4，"
                       "This amendment is not taken into account above“A reminder revision”。")
    if not out and a2 and a3 and a4:
        out.append("- **all passed**：You can go to the stage B Design and registration（Collecting Failed trajectories → screening → Main experiment）。")
    elif not out:
        out.append("- All thats run through；Some of them are still not running。")
    return out


# ---------------------------------------------------------------- The entrance
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    help="selftest | a1 | a2 | a3 | a4 | report | all，Or connect to more than one comma，e.g a3,a4,report")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-llm", action="store_true", help="Real environment + mock model（none GPU Time Check A4 pipeline；Output in out_alfw_mockllm/）")
    ap.add_argument("--keep-hashseed", action="store_true")
    ap.add_argument("--in", dest="fin")
    ap.add_argument("--out", dest="fout")
    ap.add_argument("--a4-n-train", type=int, default=None)
    ap.add_argument("--a4-n-valid", type=int, default=None, help="For local process testing only；All for Full Run 134 episodes")
    args = ap.parse_args()
    STATE["mock"] = args.mock
    STATE["mock_llm"] = args.mock_llm
    if args.mock or args.mock_llm:
        STATE["out"] = acfg.ROOT / ("out_alfw_mock" if args.mock else "out_alfw_mockllm")
        acfg.OUT = STATE["out"]
    if args.part == "replay-worker":
        part_replay_worker(args.fin, args.fout)
        return 0
    log(f"===== alfw_stage_a --part {args.part}{' --mock' if args.mock else ''}  PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} =====")
    if args.part in ("selftest", "all"):
        if not part_selftest():
            log("self-test failed，stop。")
            return 1
        if args.part == "selftest":
            return 0
    if args.part == "all":
        try:
            import selftest as grid_selftest
            grid_selftest.t0_manifest()
        except AssertionError as e:
            log(f"Code version check failed：{e}")
            return 1
        ok1 = part_a1()
        if not ok1:
            part_report()
            return 1
        for fn in (part_a2, part_a3):
            try:
                fn()
            except Exception:
                log(traceback.format_exc())
        try:
            part_a4(args.a4_n_train, args.a4_n_valid)
        except Exception:
            log(traceback.format_exc())
        part_report()
        return 0
    if not STATE["mock"]:
        try:
            import selftest as grid_selftest
            grid_selftest.t0_manifest()
        except AssertionError as e:
            log(f"Code version check failed：{e}")
            return 1
    fns = {"a1": part_a1, "a2": part_a2, "a3": part_a3, "report": part_report,
           "a4": lambda: part_a4(args.a4_n_train, args.a4_n_valid)}
    parts = [x.strip() for x in args.part.split(",") if x.strip()]
    unknown = [x for x in parts if x not in fns]
    if unknown:
        log(f"Unknown：{unknown}")
        return 2
    rc = 0
    for x in parts:
        try:
            rc |= 0 if fns[x]() else 1
        except Exception:
            log(traceback.format_exc())
            rc = 1
    if len(parts) > 1 and "report" not in parts:
        part_report()
    return rc


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != acfg.HASH_SEED and "--keep-hashseed" not in sys.argv:
        os.environ["PYTHONHASHSEED"] = acfg.HASH_SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.exit(main())
