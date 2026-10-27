"""ALFWorld stage B：Main experiment（See you in advance for registration analysis plan.md 6.3b；Set All in alfw/acfg.py of B_*）。

    python alfw_stage_b.py --stage prep      First paragraph：collect -> design -> screen -> select（Well see the results and decide whether to continue）
    python alfw_stage_b.py --stage run       Second paragraph：main -> analyze
    python alfw_stage_b.py --stage analyze   Analysis only（judge results are in out_alfw_b/judge_*.jsonl，Some of them are integrated）
    python alfw_stage_b.py --stage judge --judge api|vllm [--no-executor-info] [--retry-invalid --max-tokens N]
    python alfw_stage_b.py --stage selftest  offline self-test（mock environment + mock model，run a small end-to-end test）
Add --mock：mock environment + mock model；--mock-llm：real ALFWorld + mock model（Check Process Use）。

collect   train Newly smoked B_N_COLLECT episodes（Exclude A2/A4 Used Game with“Take two things”Tasks），Different conditions of deployment 1 episodes
design    Unable to get a fixed torrent in a failed trajectories B_MAX_SCREEN records；Seeking planning lengths at each step，slack ≥ B_MIN_SLACK split steps into early/middle/Take one position in each of the three segments at night；
          candidate = Next steps proposed by the planners + Random executable command；Once more for each branch（oracle Feasibility and residual）
screen    Independent Filter Batch：The original action branch is separate from the substitute branch B_N_SCREEN runs
select    signal detected = An alternative branch in a position is more successful than the original action branch ≥ B_SCREEN_MIN_DIFF runs；Not enough B_MIN_INFORMATIVE The rule is to stop
main      Deployment conditions per branch B_N_DEPLOY runs（Cross-drafting）、greedy 1 runs；Turns out to be branch-by-branch，supports resuming interrupted runs
analyze   and gridworld Confirm that the experiment has exactly the same selection and evaluation code（confirm._selection_values），oracle replace with ALFWorld planner
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from alfw import acfg

SCRIPT = Path(__file__).resolve()
STATE = {"mock": False, "mock_llm": False, "out": acfg.OUT_B, "small": False}
PATCHES_ON = {"translate": True, "grammar": True}
B_SCR, B_MAIN = "SCR", "M"

HYPOTHESES = {
    "B2": "Selected according to deployment strategy − select using greedy replay > 0（primary tests）",
    "B3": "Selected according to deployment strategy − by oracle Feasibility options（Average when side-by-side）> 0（primary tests）",
    "B4": "By deployment strategy 15 Subselection − Use only 3 Subselection > 0（primary tests；DoVer Small run）",
    "B5": "oracle Break the parallels as much as possible − Pure oracle > 0",
    "B6": "Selected according to deployment strategy − by oracle + Residue Selection：lower bound > 0 Wrong choice；upper bound < δ is within δ Internal security；otherwise inconclusive",
    "B7": "Selected according to deployment strategy − The step the planner suggested（Average position per position）> 0",
    "B8": "Selected according to deployment strategy − Qwen3.8-flash Thinkin about the judges（Only if you dont do it）：Three classifications with one B6",
    "B9": "Selected according to deployment strategy − 7B Self-assessment decision > 0",
}
PRIMARY = ("B2", "B3", "B4")
RULES = {"dover_2of3": "DoVer：Rerun 3 runs ≥ 2 Second-success is acceptance", "flip1_any": "CausalFlow：A single flip will be accepted",
         "earliest_suff_norepair": "AgenTracer / REFLECT：First single reset successful position",
         "latest_significant": "CAR：At the end of the night（15 Subsamples）"}


# ---------------------------------------------------------------- Foundation
def log(msg):
    print(msg, flush=True)
    try:
        lf = Path(STATE.get("log_file") or STATE["out"] / "stageB.log")     # Step Script writes the log to out_ladder_alfw/
        lf.parent.mkdir(parents=True, exist_ok=True)
        with open(lf, "a", encoding="utf-8") as f:
            f.write(time.strftime("[%m-%d %H:%M:%S] ") + msg + "\n")
    except OSError:
        pass


def P(name):
    return STATE["out"] / {"episodes": "episodes.jsonl", "episodes_meta": "episodes_meta.json",
                           "design": "design.json", "screen": "screen_rollouts.jsonl",
                           "screen_result": "screen_result.json", "main_design": "main_design.json",
                           "main": "main_rollouts.jsonl", "analysis": "analysis.json", "report": "report.md",
                           "survey": "survey.json"}[name]


def jdump(path, obj):
    STATE["out"].mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def kind():
    return "mock" if STATE["mock"] else "tw"


def fingerprint(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def code_sha():
    root = SCRIPT.parent
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()[:16]
            for n in ("alfw/agent.py", "alfw/rollout.py", "alfw/twenv.py", "alfw_stage_b.py", "confirm.py")}


def settings():
    keys = [k for k in dir(acfg) if k.startswith("B_") and k != "B_MAX_ACTIVE"]
    s = {k: getattr(acfg, k) for k in keys}
    s.update({"MAX_STEPS": acfg.MAX_STEPS, "TEMPERATURE": acfg.TEMPERATURE, "TOP_P": acfg.TOP_P,
              "MAX_NEW_TOKENS": acfg.MAX_NEW_TOKENS, "MODEL_DIR": str(acfg.MODEL_DIR), "mock": STATE["mock"],
              "mock_llm": STATE["mock_llm"]})
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}


def split_games(split):
    from alfw import games
    if STATE["mock"]:
        n, off = {"train": (200, 0), "valid_unseen": (30, 1000)}[split]
        return [f"{games.MOCK_PREFIX}{off + i}" for i in range(n)]
    return games.cached_collect(split)


def make_pool():
    from alfw import pool
    return pool.make_pool(kind(), acfg.n_workers(), None if STATE["mock"] else PATCHES_ON)


def make_backend():
    from alfw import agent
    return agent.make_backend("mock" if (STATE["mock"] or STATE["mock_llm"]) else "vllm")


PROBE_TIME = {"n": 0, "t": 0.0}


def probe_all(pl, items, label):
    out, B, t0 = [], 256, time.perf_counter()
    for j in range(0, len(items), B):
        out += pl.probe(items[j:j + B])
        log(f"  {label} Planning {min(j + B, len(items))}/{len(items)}，Time {time.perf_counter() - t0:.0f}s，"
            f"timeout {sum(r['timeout'] for r in out)}")
    PROBE_TIME["n"] += len(items)
    PROBE_TIME["t"] += time.perf_counter() - t0
    return out


# ---------------------------------------------------------------- collect
def stage_collect(ctx):
    from alfw import agent, games, rollout
    train = split_games("train")
    excl = set(games.sample(train, acfg.A2_N_GAMES, acfg.A2_SEED)) | set(games.sample(train, acfg.A4_N_TRAIN, acfg.A4_SEED))
    cand = [g for g in train if g not in excl and games.task_type_of(g) not in acfg.B_EXCLUDE_TYPES]
    gs = games.sample(cand, acfg.B_N_COLLECT, acfg.B_SEED)
    specs = [{"rid": f"c{i:03d}", "gamefile": g, "seed": agent.derive_seed("B1", g)} for i, g in enumerate(gs)]
    key = fingerprint({"specs": [[s["rid"], s["gamefile"], s["seed"]] for s in specs], "settings": settings(),
                       "code": code_sha()["alfw/agent.py"] + code_sha()["alfw/rollout.py"]})
    if P("episodes").exists() and P("episodes_meta").exists() and jload(P("episodes_meta")).get("key") == key:
        log(f"[collect] Reuse already available {len(specs)} episodes")
        return
    log(f"[collect] Optional Game {len(cand)} episodes（Exclude A2/A4 previously used {len(excl)} Office of the Secretary-General of the United Nations {acfg.B_EXCLUDE_TYPES}），sample {len(gs)} episodes")
    eps, timing = rollout.run_episodes(ctx["backend"](), ctx["pool"], specs, temperature=acfg.TEMPERATURE,
                                       keep_text=False, want_key=False, log=log)
    with open(P("episodes"), "w", encoding="utf-8") as f:
        for e in eps:
            f.write(json.dumps({"rid": e["rid"], "gamefile": e["gamefile"], "seed": e["seed"], "task": e["task"],
                                "desc": e["desc"], "won": e["won"], "won_step": e["won_step"],
                                "task_type": games.task_type_of(e["gamefile"]),
                                "cmds": [s["cmd"] for s in e["steps"]], "obs": [s["obs"] for s in e["steps"]],
                                "kinds": [s["kind"] for s in e["steps"]]}, ensure_ascii=False) + "\n")
    jdump(P("episodes_meta"), {"key": key, "timing": timing})
    n_s = sum(e["won"] for e in eps)
    log(f"[collect] complete：Success {n_s}/{len(eps)}，failed {len(eps) - n_s}；Time {timing['wall'] / 60:.1f} minutes")


def load_episodes():
    return [json.loads(x) for x in P("episodes").read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- design
def _strata_positions(eligible, k, rng):
    if len(eligible) <= k:
        return sorted(eligible)
    parts = np.array_split(np.array(sorted(eligible)), k)
    return sorted(int(rng.choice(list(p))) for p in parts if len(p))


def stage_design(ctx):
    from alfw import agent, games
    if P("design").exists():
        d = jload(P("design"))
        if d.get("settings") == settings():
            log(f"[design] Reuse already designed {d['design_id'][:12]}")
            return d
        sys.exit("[design] Designed different from current settings；Please change the output directory")
    eps = load_episodes()
    fails = sorted([e for e in eps if not e["won"]], key=lambda e: e["rid"])
    rng = random.Random(acfg.B_SEED)
    chosen = sorted(rng.sample(fails, min(acfg.B_MAX_SCREEN, len(fails))), key=lambda e: e["rid"])
    log(f"[design] failed {len(fails)} records，Take {len(chosen)} records；Step-by-step planning length")
    pl = ctx["pool"]
    items, keys = [], []
    for e in chosen:
        for t in range(len(e["cmds"])):
            items.append({"gamefile": e["gamefile"], "prefix": e["cmds"][:t], "timeout": acfg.B_PLAN_TIMEOUT,
                          "want_adm": True})
            keys.append((e["rid"], t))
    pr = probe_all(pl, items, "All trajectories")
    step = {k: r for k, r in zip(keys, pr)}

    trajs, branches, skipped = [], {}, []
    for e in chosen:
        T = len(e["cmds"])
        L = {t: (None if step[(e["rid"], t)]["timeout"] else len(step[(e["rid"], t)]["plan"])) for t in range(T)}
        elig = [t for t in range(T) if L[t] is not None and (acfg.MAX_STEPS - t) - L[t] >= acfg.B_MIN_SLACK]
        if not elig:
            skipped.append(e["rid"])
            continue
        prng = random.Random(agent.derive_seed("B2", e["rid"]))
        pos = _strata_positions(elig, acfg.B_N_POSITIONS, prng)
        positions = []
        for t in pos:
            r = step[(e["rid"], t)]
            adm, fact = r["admissible"], e["cmds"][t]
            expert = r["plan"][0] if r["plan"] else None
            alts = []
            use_exp = expert is not None and expert != fact and expert in adm
            if use_exp:
                alts.append((expert, True))
            pool_c = sorted(a for a in adm if a != fact and a != expert)
            k = acfg.B_N_RANDOM + (0 if use_exp else 1)
            alts += [(a, False) for a in prng.sample(pool_c, min(k, len(pool_c)))]
            oid = f"{e['rid']}|t{t}|orig"
            branches[oid] = {"branch_id": oid, "traj_id": e["rid"], "t": t, "action": "orig:" + fact,
                             "gamefile": e["gamefile"], "prefix": e["cmds"][:t + 1], "expert": False}
            ids = []
            for a, is_exp in alts:
                bid = f"{e['rid']}|t{t}|{a}"
                branches[bid] = {"branch_id": bid, "traj_id": e["rid"], "t": t, "action": a,
                                 "gamefile": e["gamefile"], "prefix": e["cmds"][:t] + [a], "expert": is_exp}
                ids.append(bid)
            positions.append({"t": t, "orig": oid, "alts": ids, "decisive": False, "L": L[t],
                              "slack": (acfg.MAX_STEPS - t) - L[t], "expert_used": use_exp})
        positions[-1]["decisive"] = True             # ALFWorld No irreversible dots：“decisive” = Last position of this trajectory
        trajs.append({"traj_id": e["rid"], "gamefile": e["gamefile"], "task_type": e["task_type"],
                      "fail_reason": "timeout", "T": T, "positions": positions, "n_eligible": len(elig)})
    log(f"[design] {len(trajs)} trajectories have an optional location（{len(skipped)} records without）；branch {len(branches)} items，One by one")
    bids = sorted(branches)
    pr = probe_all(pl, [{"gamefile": branches[b]["gamefile"], "prefix": branches[b]["prefix"],
                         "timeout": acfg.B_PLAN_TIMEOUT} for b in bids], "branch")
    n_to = 0
    for b, r in zip(bids, pr):
        br = branches[b]
        rem = acfg.MAX_STEPS - len(br["prefix"])
        if r["won"]:
            br.update(deterministic=1.0, oracle_L=0, oracle_F=1.0, oracle_slack=100.0, oracle_timeout=False)
        elif r["timeout"]:
            n_to += 1
            br.update(deterministic=None, oracle_L=None, oracle_F=1.0, oracle_slack=-50.0, oracle_timeout=True)
        else:
            Lb = len(r["plan"])
            br.update(deterministic=None, oracle_L=Lb, oracle_F=1.0 if Lb <= rem else 0.0,
                      oracle_slack=float(rem - Lb), oracle_timeout=False)
    d = {"kind": "alfw_screen", "trajectories": trajs, "branches": branches, "skipped": skipped,
         "settings": settings(), "code": code_sha(), "n_branch_timeout": n_to}
    d["design_id"] = fingerprint({"t": trajs, "b": {k: {kk: v[kk] for kk in ("prefix", "oracle_F", "oracle_slack")}
                                                    for k, v in branches.items()}, "s": d["settings"]})
    jdump(P("design"), d)
    nst = sum(1 for b in branches.values() if b["deterministic"] is None)
    log(f"[design] complete {d['design_id'][:12]}：{len(trajs)} trajectories、{len(branches)} branches（random {nst}，"
        f"Planning timeout {n_to}）")
    return d


# ---------------------------------------------------------------- The branchs running again（supports resuming interrupted runs）
def run_branches(ctx, d, cond, batch, n, path, temperature):
    import runner
    from alfw import agent, rollout
    path = Path(path)
    runner._repair_tail(path)
    done = runner.load_results(path)
    sid = fingerprint({"design": d["design_id"], "cond": cond, "batch": batch, "n": n, "T": temperature,
                       "code": {k: v for k, v in code_sha().items() if k.startswith("alfw/")},
                       "mock": STATE["mock"] or STATE["mock_llm"]})
    for (c, b, bid), rec in done.items():
        if c == cond and b == batch and rec.get("spec_id") != sid:
            sys.exit(f"[{cond}/{batch}] Different settings signed for existing results（Code or design changed）；Please change the output directory")
    todo = [b for b in sorted(d["branches"]) if d["branches"][b]["deterministic"] is None
            and (cond, batch, b) not in done]
    total = sum(1 for b in d["branches"].values() if b["deterministic"] is None)
    log(f"[{cond}/{batch}] Random Branch {total} items，Completed {total - len(todo)}，Stay {len(todo)} × {n} runs")
    if not todo:
        return None
    backend = ctx["backend"]()
    buf = {}
    f = open(path, "a", encoding="utf-8")

    def specs():
        for b in todo:
            br = d["branches"][b]
            for k in range(n):
                yield {"rid": f"{b}#{k}", "gamefile": br["gamefile"], "prefix": br["prefix"],
                       "seed": agent.derive_seed(cond, batch, b, k)}

    def on_done(e):
        b, k = e["rid"].rsplit("#", 1)
        st = rollout.episode_stats(e)
        buf.setdefault(b, {})[int(k)] = (bool(e["won"]), st["n"], st["invalid"] + st["no_action"], st["revisit"],
                                         st["retries"])
        if len(buf[b]) == n:
            rows = [buf[b][i] for i in range(n)]
            rec = {"condition": cond, "batch": batch, "branch_id": b, "spec_id": sid, "n": n,
                   "results": [r[0] for r in rows], "successes": sum(r[0] for r in rows),
                   "steps": [r[1] for r in rows], "invalid": [r[2] for r in rows], "revisit": [r[3] for r in rows],
                   "retries": [r[4] for r in rows]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            del buf[b]
            if ctx.get("stop_after") is not None:          # for self-tests only：Simulation stop
                ctx["stop_after"] -= 1
                if ctx["stop_after"] <= 0:
                    raise KeyboardInterrupt("selftest interrupt")
    try:
        tm = rollout.run_stream(backend, ctx["pool"], specs(), temperature, on_done,
                                max_active=acfg.B_MAX_ACTIVE, log=log, want_key=True)
        log(f"[{cond}/{batch}] complete：{tm['wall'] / 60:.1f} minutes，{tm['policy_steps']} steps，"
            f"{tm['policy_steps'] / max(tm['wall'], 1e-9):.1f} steps/seconds")
        ctx.setdefault("timings", {})[f"{cond}/{batch}"] = tm
        return tm
    finally:
        f.close()


# ---------------------------------------------------------------- screen / select
def screen_result(d, results):
    from alfw import acfg as A
    rows, inf = [], []
    br = d["branches"]

    def k_of(b):
        x = br[b]
        if x["deterministic"] is not None:
            return int(round(x["deterministic"] * A.B_N_SCREEN))
        return results[("deploy_T07", B_SCR, b)]["successes"]
    for t in d["trajectories"]:
        best, best_alt = -99, 0
        for p in t["positions"]:
            ko = k_of(p["orig"])
            for a in p["alts"]:
                ka = k_of(a)
                best = max(best, ka - ko)
                best_alt = max(best_alt, ka)
        rows.append({"traj_id": t["traj_id"], "task_type": t["task_type"], "best_diff": best, "best_alt": best_alt})
        if best >= A.B_SCREEN_MIN_DIFF:
            inf.append(t["traj_id"])
    return inf, rows


def stage_select(d):
    import runner
    res = runner.load_results(P("screen"))
    inf, rows = screen_result(d, res)
    by_type = {}
    for r in rows:
        bt = by_type.setdefault(r["task_type"], {"screened": 0, "informative": 0})
        bt["screened"] += 1
        bt["informative"] += int(r["traj_id"] in inf)
    hist = {}
    for r in rows:
        hist[r["best_diff"]] = hist.get(r["best_diff"], 0) + 1
    info = {"n_screened": len(rows), "n_informative": len(inf), "informative": inf, "by_type": by_type,
            "best_diff_hist": dict(sorted(hist.items())), "any_alt_ge2": sum(r["best_alt"] >= 2 for r in rows),
            "rule": f"An alternative branch in a position is more successful than the original action branch ≥ {acfg.B_SCREEN_MIN_DIFF}/{acfg.B_N_SCREEN} runs",
            "will_run_main": min(len(inf), acfg.B_N_MAIN_MAX), "design_id": d["design_id"]}
    jdump(P("screen_result"), info)
    log(f"[select] screening {len(rows)} records，signal detected {len(inf)} records（{info['rule']}）；By Task Type {by_type}；"
        f"Maximum margin distribution {info['best_diff_hist']}")
    if len(inf) < acfg.B_MIN_INFORMATIVE:
        log(f"[select] fewer informative trajectories than {acfg.B_MIN_INFORMATIVE} records：Stop by pre-rule，Do not enter the main experiment。")
        return None
    ids = sorted(random.Random(acfg.B_SEED + 1).sample(inf, min(len(inf), acfg.B_N_MAIN_MAX)))
    keep = set(ids)
    trajs = [t for t in d["trajectories"] if t["traj_id"] in keep]
    bset = {b for t in trajs for p in t["positions"] for b in [p["orig"]] + p["alts"]}
    md = {"kind": "alfw_main", "trajectories": trajs, "branches": {b: d["branches"][b] for b in sorted(bset)},
          "settings": d["settings"], "code": code_sha(), "screen_design_id": d["design_id"],
          "hypotheses": HYPOTHESES}
    md["design_id"] = fingerprint({"screen": d["design_id"], "ids": ids, "h": HYPOTHESES})
    if P("main_design").exists() and jload(P("main_design"))["design_id"] != md["design_id"]:
        sys.exit("[select] The main experiment design is different from the current one；Please change the output directory")
    jdump(P("main_design"), md)
    nst = sum(1 for b in md["branches"].values() if b["deterministic"] is None)
    log(f"[select] Main experiment {len(trajs)} trajectories、Random Branch {nst} items；Deployment {acfg.B_N_DEPLOY} runs + greedy {acfg.B_N_GREEDY} runs"
        f" = {nst * (acfg.B_N_DEPLOY + acfg.B_N_GREEDY)} Keep running")
    return md


# ---------------------------------------------------------------- analyze
def load_judges():
    judges = {}
    for p in sorted(STATE["out"].glob("judge_*.jsonl")):
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if rows:
            judges[rows[0]["judge"]] = {r["traj_id"]: r["branch_id"] for r in rows}
    return judges


def make_extra(judges):
    import methods_gw

    class ExtraB(methods_gw.Extra):
        def __init__(self, j):
            super().__init__(j)
            # expert_earliest / slack_abs：Ex post facto exploratory baseline（2026-10-23，After reading the main test results,）——The strongest planner inspiration：
            # Planner actions are in the earliest position；by“Absolute residual”（rather than slack gain relative to the original action）Selection。
            for n in ("expert_random", "expert_latest", "expert_earliest", "slack_abs"):
                self.names.append(n)
                self.coverage[n] = []

        def __call__(self, traj, cands, raw, R, Rall, col, br):
            out = super().__call__(traj, cands, raw, R, Rall, col, br)
            ex = [c for c, (a, o, t, _d) in enumerate(cands) if br[a].get("expert")]
            nb = R.shape[0]
            if ex:
                out["expert_random"] = Rall[:, ex].mean(axis=-1)
                out["expert_latest"] = Rall[:, max(ex, key=lambda c: cands[c][2])]
                out["expert_earliest"] = Rall[:, min(ex, key=lambda c: cands[c][2])]
            else:
                for n in ("expert_random", "expert_latest", "expert_earliest"):
                    out[n] = np.full(nb, np.nan)
            sc = np.array([br[a]["oracle_F"] * 1000 + br[a]["oracle_slack"] for a, *_ in cands])
            m = (np.abs(sc - sc.max()) <= 1e-9).astype(float)
            out["slack_abs"] = (Rall * m).sum(axis=-1) / m.sum()
            for n in ("expert_random", "expert_latest", "expert_earliest"):
                self.coverage[n].append(float(bool(ex)))
            self.coverage["slack_abs"].append(1.0)
            return out
    return ExtraB(judges)


def stage_analyze(n_boot=None):
    import confirm
    import runner
    md = jload(P("main_design"))
    res = runner.load_results(P("main"))
    nst = [b for b, x in md["branches"].items() if x["deterministic"] is None]
    missing = [b for b in nst if ("deploy_T07", B_MAIN, b) not in res or ("llm_greedy", B_MAIN, b) not in res]
    if missing:
        log(f"[analyze] also {len(missing)} The branchs not finished，complete first main")
        return None
    judges = load_judges()
    ex = make_extra(judges)
    n_boot = n_boot or confirm.N_BOOT
    V, names, contrast = confirm._selection_values(md, res, "net", n_boot, 0, extra=ex,
                                                   oracle_value=lambda x: x["oracle_F"],
                                                   slack_value=lambda x: x["oracle_slack"])
    J = "judge:qwen3.8-flash_think"      # Revision（2026-10-23 12:55，ALFWorld Before any judges runs）：B8 by DeepSeek Changed to Questions
    pairs = {"B2": ("matched", "greedy"), "B3": ("matched", "oracle"), "B4": ("matched", "k3"),
             "B5": ("oracle_slack", "oracle"), "B6": ("matched", "oracle_slack"), "B7": ("matched", "expert_random"),
             "positive_control": ("matched", "random")}
    if J in names:
        pairs["B8"] = ("matched", J)
    if "judge:qwen7b_self" in names:
        pairs["B9"] = ("matched", "judge:qwen7b_self")
    out = {h: contrast(a, b) for h, (a, b) in pairs.items()}
    pc_ok = out["positive_control"]["ci90"][0] > acfg_delta()
    out["positive_control"]["verdict"] = "passed" if pc_ok else "failed：insufficient discriminative signal，All conclusions are not valid"
    order = sorted(PRIMARY, key=lambda h: out[h]["p_one_sided"])
    m, ok_all = len(order), True
    for i, h in enumerate(order):
        thr = confirm.ALPHA / (m - i)
        rej = ok_all and out[h]["p_one_sided"] <= thr
        out[h]["holm_threshold"], out[h]["holm_reject"] = thr, bool(rej)
        ok_all = ok_all and rej
    for h in pairs:
        if h == "positive_control":
            continue
        v = out[h]
        lo, hi = v["ci90"]
        if not pc_ok:
            v["verdict"] = "insufficient discriminative signal"
        elif h in PRIMARY:
            v["verdict"] = ("confirmed（Holm Correct）" if v["holm_reject"] else "not confirmed（Holm Correct）") + \
                           ("；lower bound > 0.10" if lo > acfg_delta() else "")
        elif h in ("B6", "B8"):
            v["verdict"] = ("Wrong choice（lower bound > 0）" if lo > 0 else "at δ Internal security（upper bound < 0.10）" if hi < acfg_delta()
                            else "inconclusive（Inter-district crossing 0 and δ）")
        else:
            v["verdict"] = ("confirmed（lower bound > 0）" if lo > 0 else "not confirmed") + ("；lower bound > 0.10" if lo > acfg_delta() else "")
    out["rules"] = {r: contrast("matched", r) for r in RULES if r in names}
    out["exploratory"] = {r: contrast("matched", r) for r in ("expert_earliest", "slack_abs", "expert_latest", "k5", "k10")
                          if r in names}
    # oracle When the feasibility is equal to all the candidates（ALFWorld There is no irreversible action，Common），Pure oracle Selection is random selection，B3 Collapse with positive
    out["oracle_degenerate"] = bool(len({float(x["oracle_F"]) for x in md["branches"].values()}) == 1)
    out["judges"] = {j: contrast("matched", f"judge:{j}") for j in judges}
    # Matching of the judges（After-action，exploratory；Difference = Net deployment gains from the former − The latter）
    J = "judge:qwen3.8-flash_think"
    jp = [("noexec", J, J + "_noexec", "Think about it − Remove the executor information"),
          ("nothink", J, "judge:qwen3.8-flash", "Think about it − Close Thinking"),
          ("informed_gain", J + "_informed", J, "informed judge − Original version of the decision（Fill in the improvements resulting from hidden information）"),
          ("informed_vs_expert_earliest", J + "_informed", "expert_earliest", "informed judge − Planner actions are in the earliest position"),
          ("informed_vs_slack_abs", J + "_informed", "slack_abs", "informed judge − Select in absolute remainder"),
          ("informed_vs_oracle_slack", J + "_informed", "oracle_slack", "informed judge − oracle + slack（Definition of registration）"),
          ("think_vs_expert_earliest", J, "expert_earliest", "Think about it − Planner actions are in the earliest position")]
    out["judge_pairs"] = {}
    for key, a, b, lab in jp:
        if a in names and b in names:
            c = contrast(a, b)
            c["label"] = lab
            out["judge_pairs"][key] = c
    out["judge_describe"] = judge_describe(md, res, judges)
    out["values"] = {k: float(np.nanmean(V[k][0])) for k in names}
    out["coverage"] = {k: float(np.mean(v)) for k, v in ex.coverage.items() if v}
    Vp, names_p, contrast_p = confirm._selection_values(md, res, "post", n_boot, 10,
                                                        oracle_value=lambda x: x["oracle_F"],
                                                        slack_value=lambda x: x["oracle_slack"])
    out["post_objective"] = {"values": {k: float(np.nanmean(Vp[k][0])) for k in names_p},
                             "matched_minus_greedy": contrast_p("matched", "greedy"),
                             "matched_minus_oracle": contrast_p("matched", "oracle"),
                             "matched_minus_k3": contrast_p("matched", "k3")}
    out["descriptive"] = describe(md, res)
    out["n_trajectories"] = len(md["trajectories"])
    out["design_id"] = md["design_id"]
    jdump(P("analysis"), out)
    P("report").write_text(report(out, md), encoding="utf-8")
    log(f"[analyze] Report written {P('report')}")
    return out


def judge_describe(md, res, judges):
    """Description in the sample（Use all deployment samples）：What did the judges choose?——Proportion of Planner Actions、Location Distribution、whether the best position was selected；
    and“Switch to planner action in the position selected by the judges”and“Best in the sample”Net proceeds，used to decompose loss into“In exchange for what?”and“Which step?”。"""
    br = md["branches"]

    def p(b):
        if br[b]["deterministic"] is not None:
            return br[b]["deterministic"]
        r = res[("deploy_T07", B_MAIN, b)]
        return r["successes"] / r["n"]
    out = {}
    for j, picks in judges.items():
        n = nv = ne = nbest = 0
        rank = [0, 0, 0]
        vj, vexp, vbest = [], [], []
        for t in md["trajectories"]:
            n += 1
            b = picks.get(t["traj_id"])
            pos = sorted(t["positions"], key=lambda q: q["t"])
            g = lambda a, q: p(a) - p(q["orig"])
            best_overall = max(g(a, q) for q in pos for a in q["alts"])
            hit = [i for i, q in enumerate(pos) if b in q["alts"]]
            if not hit:
                continue
            nv += 1
            i = hit[0]
            q = pos[i]
            rank[min(2, round(2 * i / max(1, len(pos) - 1)))] += 1
            ne += bool(br[b].get("expert"))
            best_i = max(range(len(pos)), key=lambda k: max(g(a, pos[k]) for a in pos[k]["alts"]))
            nbest += best_i == i
            ex = [a for a in q["alts"] if br[a].get("expert")]
            vj.append(g(b, q))
            vexp.append(g(ex[0], q) if ex else g(b, q))
            vbest.append(best_overall)
        out[j] = {"n": n, "valid": nv, "expert_frac": ne / max(1, nv), "position_early_mid_late": rank,
                  "best_position_frac": nbest / max(1, nv), "value_in_sample": float(np.mean(vj)) if vj else None,
                  "value_planner_action_same_position": float(np.mean(vexp)) if vexp else None,
                  "value_best_in_sample": float(np.mean(vbest)) if vbest else None}
    return out


def acfg_delta():
    import confirm
    return confirm.DELTA


def describe(md, res):
    """descriptive：Continue Length、not executable、A circle（Go back to the status quo ante）——Deployment vs greedy；Percentage of planner recommended steps to be optimal。"""
    def agg(cond):
        st, inv, rev, n = 0, 0, 0, 0
        for (c, b, bid), r in res.items():
            if c == cond and b == B_MAIN:
                st += sum(r["steps"])
                inv += sum(r["invalid"])
                rev += sum(r["revisit"])
                n += r["n"]
        return {"rollouts": n, "mean_steps": st / max(1, n), "invalid_per_step": inv / max(1, st),
                "revisit_per_step": rev / max(1, st)}
    succ = {}
    for cond in ("deploy_T07", "llm_greedy"):
        k = n = 0
        for (c, b, bid), r in res.items():
            if c == cond and b == B_MAIN:
                k += r["successes"]
                n += r["n"]
        succ[cond] = k / max(1, n)
    best_is_expert, n_traj = 0, 0
    for t in md["trajectories"]:
        best, best_b = -9, None
        for p in t["positions"]:
            ro = res.get(("deploy_T07", B_MAIN, p["orig"]))
            po = md["branches"][p["orig"]]["deterministic"] if ro is None else ro["successes"] / ro["n"]
            for a in p["alts"]:
                ra = res.get(("deploy_T07", B_MAIN, a))
                pa = md["branches"][a]["deterministic"] if ra is None else ra["successes"] / ra["n"]
                if pa - po > best:
                    best, best_b = pa - po, a
        n_traj += 1
        best_is_expert += int(bool(best_b and md["branches"][best_b].get("expert")))
    return {"deploy": agg("deploy_T07"), "greedy": agg("llm_greedy"), "branch_success": succ,
            "best_candidate_is_expert_frac": best_is_expert / max(1, n_traj)}


def report(out, md):
    def f(c):
        s = f"{c['point']:+.3f} | [{c['ci90'][0]:+.3f}, {c['ci90'][1]:+.3f}]"
        return s + (f" | [{c['ci90_bc'][0]:+.3f}, {c['ci90_bc'][1]:+.3f}]" if "ci90_bc" in c else " | —")
    L = ["# ALFWorld stage B：Main Experiment Report", ""]
    if STATE["mock"] or STATE["mock_llm"]:
        L += ["> **Simulate Run**：The numbers dont mean anything。", ""]
    L += [f"design {out['design_id'][:12]}，{out['n_trajectories']} trajectories；δ = {acfg_delta()}，90% interval；"
          "primary tests B2–B4 Holm Correct。loss = Net gains on deployment selected by matching deployment strategy − Net deployment gains from this modality。", "",
          f"**positive control**（Match − random）：{f(out['positive_control'])} → {out['positive_control']['verdict']}", "",
          "| Test | content | A point estimate | 90% interval（Registration） | deviation correction interval | judgments |", "|---|---|---:|---|---|---|"]
    for h in ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"):
        if h in out:
            L.append(f"| {h} | {HYPOTHESES[h]} | {f(out[h])} | {out[h]['verdict']} |")
    L += ["", "**Determination rules for published methods**（Same deployment sample；Do not fix if no acceptable fixes are found）：", "",
          "| Rules | Relatively matched loss | 90% interval（Registration） | deviation correction interval | Coverage |", "|---|---:|---|---|---:|"]
    for r, c in out["rules"].items():
        L.append(f"| {RULES[r]} | {f(c)} | {out['coverage'].get(r, 1.0):.2f} |")
    if out["judges"]:
        L += ["", "**LLM Rec**：", "", "| Rec | Relatively matched loss | 90% interval（Registration） | deviation correction interval |", "|---|---:|---|---|"]
        for j, c in out["judges"].items():
            L.append(f"| {j} | {f(c)} |")
    if out.get("judge_describe"):
        L += ["", "**What did the judges choose?**（Description in the sample，Use all deployment samples；“Change position to planner action”and“Best in the sample”used to decompose loss into“In exchange for what?”and“Which step?”）：", "",
              "| Rec | Validity | Select Planner Actions | early / middle / Organisation | Select Best Location | Net proceeds from selection of judges | Change position to planner action | Best in the sample |",
              "|---|---:|---:|---|---:|---:|---:|---:|"]
        for j, v in out["judge_describe"].items():
            if not v["valid"]:
                continue
            L.append(f"| {j} | {v['valid']}/{v['n']} | {v['expert_frac']:.2f} | {' / '.join(map(str, v['position_early_mid_late']))} | "
                     f"{v['best_position_frac']:.2f} | {v['value_in_sample']:+.3f} | {v['value_planner_action_same_position']:+.3f} | "
                     f"{v['value_best_in_sample']:+.3f} |")
    if out.get("judge_pairs"):
        L += ["", "**Matching of the judges**（After-action，exploratory；Difference = Net deployment gains from the former − The latter，> 0 That means the former is better）：", "",
              "| comparison | Difference | 90% interval（Method of registration） | deviation correction interval |", "|---|---:|---|---|"]
        for c in out["judge_pairs"].values():
            L.append(f"| {c['label']} | {f(c)} |")
    if out.get("exploratory"):
        lab = {"expert_earliest": "Planner actions are in the earliest position", "slack_abs": "Select in absolute remainder（oracle Feasible + Maximum residual）",
               "expert_latest": "Planner action at the latest", "k5": "Deployment strategy 5 runs", "k10": "Deployment strategy 10 runs"}
        L += ["", "**An exploratory comparison**（After-action，Do not test correction）：", "",
              "| Modalities | Relatively matched loss | 90% interval（Method of registration） | deviation correction interval |", "|---|---:|---|---|"]
        for r, c in out["exploratory"].items():
            L.append(f"| {lab.get(r, r)} | {f(c)} |")
    notes = ["deviation correction interval：parameters bootstrap of (k+0.5)/(n+1) The contractions bootstrap Distribution overall below point estimate（The success rate is close 0/1 especially apparent when）；"
             "Reconcile the distribution to a point-estimated centre。The registration judgments is still based on the registration area。"]
    if out.get("oracle_degenerate"):
        notes.append("**oracle The feasibility is the same for all candidates**（all are solvable）：Pure oracle Select an equal random selection，B3 Collapse with positive，Not as independent evidence；"
                     "and oracle the relevant informative comparisons are B5/B6（slack）Compared to the exploratory nature of the top。")
    L += [""] + [f"- {x}" for x in notes]
    L += ["", "Net deployment gains from restoration selected by option：" + "；".join(f"{k} {v:+.3f}" for k, v in out["values"].items()), "",
          "**description**：" + json.dumps(out["descriptive"], ensure_ascii=False), "",
          "Secondary objectives（Success rate after restoration）：" + "；".join(f"{k} {v:+.3f}" for k, v in out["post_objective"]["values"].items())]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- judge
JUDGE_SYSTEM = ("You are an expert debugger of AI agents. You will read the log of a failed run of an agent in a "
                "text-based household environment and decide which single one-step fix is most likely to make the run succeed.")


HIDDEN_HEADER = ("Hidden information that the agent itself does NOT have (it only sees what it observes along the way): "
                 "the contents of every receptacle at the start of the run. Objects can later be moved by the agent's own "
                 "actions, as shown in the log.")


def judge_messages(ep, cands, executor_info=True, hidden=None):
    """hidden：informed judge（2026-10-23 Joiner s Decomposition）The contents of each container when extra openings are obtained（survey.json of text）；None is the original registered prompt。"""
    lines = [f"Step {i}: > {c or '(no action)'}\n{o}" for i, (c, o) in enumerate(zip(ep["cmds"], ep["obs"]))]
    cl = [f'C{i + 1}: at step {c["t"]}, replace "{c["orig"]}" with "{c["alt"]}"' for i, c in enumerate(cands)]
    user = ("The agent acts in a text-based household environment. At every step it sees the task, the room description, "
            "its history of actions with the environment's feedback, and the list of actions possible right now, and it "
            "picks one of them. An action that is not possible does nothing (\"Nothing happens.\") but uses up a step. "
            f"The step budget is {acfg.MAX_STEPS} steps.\n\n"
            + ("The agent is a 7B language model that samples its actions at temperature 0.7.\n\n" if executor_info else "")
            + f"Task: {ep['task']}\n\n{ep['desc']}\n\n"
            + (f"{HIDDEN_HEADER}\n{hidden}\n\n" if hidden is not None else "")
            + f"Log of the failed run ({len(lines)} steps; the agent ran out of steps):\n" + "\n".join(lines)
            + "\n\nCandidate fixes. Each replaces exactly one action of the log; the environment executes the new action, "
              "and then the SAME agent continues on its own from there with the remaining steps:\n" + "\n".join(cl)
            + "\n\nWhich single fix is most likely to make this agent's run succeed? Think step by step briefly, "
              "then end with one line: Choice: C<number>")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def _contents(r):
    """survey One line -> “a creditcard 1, and a laptop 1” / “nothing”。"""
    o = r["open_obs"] or r["obs"]
    ms = re.findall(r"you see (.+?)\.(?:\s|$)", o)
    if ms:
        return ms[-1].strip()
    return re.sub(r"^You arrive at [^.]*\.\s*", "", o).strip() or "(no description)"


def stage_survey(ctx):
    """Hide information for informed judges：Game for each trajectory of the main experiment，the contents of each container at the start（One by one go to / open Read environmental feedback）。"""
    md = jload(P("main_design"))
    trajs = md["trajectories"]
    res = ctx["pool"].survey([{"gamefile": t["gamefile"]} for t in trajs])
    out, n_empty, n_fail, n_detour = {}, 0, 0, 0
    for t, r in zip(trajs, res):
        rows = r["receptacles"]
        n_empty += not rows
        n_fail += sum(not x.get("ok", True) for x in rows)
        n_detour += sum(bool(x.get("detour")) for x in rows)
        text = "\n".join(f"- {x['name']}{' (closed at start)' if x['closed'] else ''}: {_contents(x)}" for x in rows)
        out[t["traj_id"]] = {"gamefile": t["gamefile"], "receptacles": rows, "text": text}
    jdump(P("survey"), out)
    nr = [len(v["receptacles"]) for v in out.values()]
    log(f"[survey] {len(out)} episodes；Each container {min(nr) if nr else 0}–{max(nr) if nr else 0} items；"
        f"The container was not deconstructed {n_empty} episodes；Same position as the previous container、Well go around then {n_detour} items；not yet visited {n_fail} items -> {P('survey')}")
    if out:
        k = next(iter(out))
        log(f"[survey] Example: {k}：\n" + "\n".join(out[k]["text"].splitlines()[:8]))
    return 0 if (n_empty == 0 and n_fail == 0) or STATE["mock"] else 1


def stage_judge(args):
    import judge_backends as JB
    from alfw.agent import derive_seed
    md = jload(P("main_design"))
    eps = {e["rid"]: e for e in load_episodes()}
    informed = bool(getattr(args, "informed", False))
    hidden = {}
    if informed:
        if not P("survey").exists():
            log("[judge] An informed judge needs to run first --stage survey（Record the contents of each container at the start of the game）")
            return 1
        hidden = {k: v["text"] for k, v in jload(P("survey")).items()}
    items = []
    for t in md["trajectories"]:
        ep = eps[t["traj_id"]]
        cands = []
        for p in t["positions"]:
            for b in p["alts"]:
                cands.append({"branch_id": b, "t": p["t"], "orig": ep["cmds"][p["t"]], "alt": md["branches"][b]["action"]})
        random.Random(derive_seed("judge", t["traj_id"])).shuffle(cands)
        items.append({"traj_id": t["traj_id"], "cands": cands,
                      "messages": judge_messages(ep, cands, executor_info=not args.no_executor_info,
                                                 hidden=hidden[t["traj_id"]] if informed else None)})
    if args.dry_run:
        m = items[0]["messages"]
        print(m[0]["content"] + "\n\n" + m[1]["content"])
        return 0
    if STATE["mock"] or STATE["mock_llm"] or args.judge == "mock":
        judge = JB.MockJudge()
    else:
        judge = JB.make_judge(args.judge, model=args.model, model_dir=acfg.MODEL_DIR, thinking=not args.no_thinking,
                             effort=args.effort)
    if args.no_executor_info:
        judge.name += "_noexec"
    if informed:
        judge.name += "_informed"
        judge.settings["informed"] = True
    if args.max_tokens and hasattr(judge, "max_tokens"):
        judge.max_tokens = args.max_tokens
        judge.settings["max_tokens"] = args.max_tokens
    path = STATE["out"] / f"judge_{judge.name}.jsonl"
    old = {}
    if args.retry_invalid and path.exists():
        old = {r["traj_id"]: r for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}
        items = [it for it in items if old.get(it["traj_id"], {}).get("branch_id") is None]
        log(f"[judge] Again {len(items)} The rule is not answered")
        if getattr(judge, "seed", None) is not None:      # The same prompts will be given when the torrent is fixed：Ask again for a new seed
            judge.seed += 1
            judge.settings["seed"] = judge.seed
    outs = JB.ask_resumable(judge, [it["traj_id"] for it in items], [it["messages"] for it in items],
                            STATE["out"] / "judge_cache" / f"{judge.name}.jsonl")
    rows = []
    for it, o in zip(items, outs):
        k = JB.parse_choice(o["text"], len(it["cands"]))
        rows.append({"traj_id": it["traj_id"], "judge": judge.name, "settings": judge.settings,
                     "design_id": md["design_id"], "choice": None if k is None else k + 1,
                     "branch_id": None if k is None else it["cands"][k]["branch_id"], "text": o["text"],
                     "finish_reason": o.get("finish_reason"), "error": o.get("error"), "usage": o.get("usage"),
                     "retried": bool(old)})
    if old:
        for r in rows:
            old[r["traj_id"]] = r
        rows = [old[k] for k in sorted(old)]
    JB.write_jsonl(path, rows)
    log(f"[judge] {judge.name}：{sum(r['branch_id'] is not None for r in rows)}/{len(rows)} Validity -> {path}")
    return 0


# ---------------------------------------------------------------- Smoke test（real 7B、Very small，approximately 15–20 minutes）
SMOKE = {"B_N_COLLECT": 16, "B_MAX_SCREEN": 4, "B_N_SCREEN": 2, "B_MIN_INFORMATIVE": 1, "B_SCREEN_MIN_DIFF": -99,
         "B_N_MAIN_MAX": 2, "B_N_DEPLOY": 4}
FULL = {k: getattr(acfg, k) for k in SMOKE}          # Formal setting（Time-consuming for extrapolation）


def smoke_summary(ctx):
    """Inspection of all segments of the current water line produced results、Unenforceable rate of follow-up is normal，And the time it takes to get out of operation with actual speed。"""
    import runner
    ok, notes = True, []
    for name in ("episodes", "design", "screen", "screen_result", "main_design", "main", "analysis", "report"):
        if not P(name).exists():
            ok = False
            notes.append(f"Missing {P(name).name}")
    eps = load_episodes() if P("episodes").exists() else []
    d = jload(P("design")) if P("design").exists() else {"trajectories": [], "branches": {}}
    res = runner.load_results(P("main")) if P("main").exists() else {}
    scr = runner.load_results(P("screen")) if P("screen").exists() else {}
    steps = sum(sum(r["steps"]) for r in list(res.values()) + list(scr.values()))
    inval = sum(sum(r["invalid"]) for r in list(res.values()) + list(scr.values()))
    inv_rate = inval / max(1, steps)
    if inv_rate > 0.20:
        ok = False
        notes.append(f"Percentage of unexecutable actions in continuation {inv_rate:.3f} > 0.20")
    tms = ctx.get("timings", {})
    tot_steps = sum(t["policy_steps"] for t in tms.values())
    tot_wall = sum(t["wall"] for t in tms.values())
    sps = tot_steps / max(tot_wall, 1e-9)
    mean_ep = float(np.mean([len(e["cmds"]) for e in eps])) if eps else 35.0
    rl = [st / r["n"] for r in list(res.values()) + list(scr.values()) for st in [sum(r["steps"])]]
    mean_roll = float(np.mean(rl)) if rl else 25.0
    br_per_traj = len(d["branches"]) / max(1, len(d["trajectories"]))
    full_steps = {"collect": FULL["B_N_COLLECT"] * mean_ep,
                  "screen": FULL["B_MAX_SCREEN"] * br_per_traj * FULL["B_N_SCREEN"] * mean_roll,
                  "main": FULL["B_N_MAIN_MAX"] * br_per_traj * (FULL["B_N_DEPLOY"] + acfg.B_N_GREEDY) * mean_roll}
    probe_per = PROBE_TIME["t"] / max(1, PROBE_TIME["n"])
    full_probes = FULL["B_MAX_SCREEN"] * (mean_ep + br_per_traj)
    hours = {k: v / max(sps, 1e-9) / 3600 for k, v in full_steps.items()}
    hours["planner"] = probe_per * full_probes / 3600
    summ = {"ok": ok, "notes": notes, "invalid_rate": inv_rate, "steps_per_sec": sps,
            "mean_episode_steps": mean_ep, "mean_rollout_steps": mean_roll, "branches_per_traj": br_per_traj,
            "projected_hours": hours, "projected_total_hours": sum(hours.values()),
            "note": "The speed is small and measured，Officially running and higher，should be faster in practice"}
    jdump(STATE["out"] / "smoke_summary.json", summ)
    log(f"[smoke] {'passed' if ok else 'failed'} {notes if notes else ''}；continuation {sps:.1f} steps/seconds，not executable {inv_rate:.3f}；"
        f"Extra-publish formal-operational contract {summ['projected_total_hours']:.1f} hours "
        + "（" + "，".join(f"{k} {v:.1f}h" for k, v in hours.items()) + "）")
    return 0 if ok else 1


# ---------------------------------------------------------------- The entrance
def run_stages(stages, args):
    ctx = {"pool": None, "_backend": None}
    STATE["ctx"] = ctx

    def backend():
        if ctx["_backend"] is None:
            t0 = time.perf_counter()
            ctx["_backend"] = make_backend()
            log(f"Load Model {time.perf_counter() - t0:.0f}s")
        return ctx["_backend"]
    ctx["backend"] = backend
    if stages == ["analyze"]:            # Analysis only：No environmental process pool required（not installed ALFWorld You can run on a machine）
        log("===== analyze =====")
        return 0 if stage_analyze() is not None else 1
    with make_pool() as pl:
        ctx["pool"] = pl
        d = None
        for st in stages:
            log(f"===== {st} =====")
            if st == "collect":
                stage_collect(ctx)
            elif st == "design":
                d = stage_design(ctx)
            elif st == "screen":
                d = d or jload(P("design"))
                run_branches(ctx, d, "deploy_T07", B_SCR, acfg.B_N_SCREEN, P("screen"), acfg.TEMPERATURE)
            elif st == "select":
                d = d or jload(P("design"))
                if stage_select(d) is None:
                    return 1
            elif st == "main":
                md = jload(P("main_design"))
                run_branches(ctx, md, "deploy_T07", B_MAIN, acfg.B_N_DEPLOY, P("main"), acfg.TEMPERATURE)
                run_branches(ctx, md, "llm_greedy", B_MAIN, acfg.B_N_GREEDY, P("main"), 0.0)
            elif st == "analyze":
                stage_analyze()
            elif st == "survey":
                if stage_survey(ctx):
                    return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["prep", "run", "all", "smoke", "collect", "design", "screen", "select", "main", "analyze",
                             "judge", "survey", "selftest"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--judge", default="api", choices=["api", "vllm", "mock"])
    ap.add_argument("--model", default=None, help="API A judges model：qwen3.8-flash（default）or deepseek-flash etc")
    ap.add_argument("--effort", default=None, help="The intensity of thinking（By default of the service provider）")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--no-executor-info", action="store_true")
    ap.add_argument("--retry-invalid", action="store_true")
    ap.add_argument("--informed", action="store_true", help="informed judge：Add the contents of each container at the start of the prompts（Run first --stage survey）")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    STATE["mock"], STATE["mock_llm"] = args.mock, args.mock_llm
    if args.out_dir:
        STATE["out"] = args.out_dir.resolve()
    elif args.mock or args.mock_llm:
        STATE["out"] = acfg.ROOT / ("out_alfw_b_mock" if args.mock else "out_alfw_b_mockllm")
    if args.stage == "selftest":
        import alfw_selftest_b
        return alfw_selftest_b.main()
    if args.stage == "smoke":            # real 7B A very small step towards a complete process；Save Feeds and Output Directory，The results are not part of any analysis
        for k, v in SMOKE.items():
            setattr(acfg, k, v)
        acfg.B_SEED = acfg.B_SEED + 100
        if not args.out_dir:
            STATE["out"] = acfg.ROOT / ("out_alfw_b_smoke" + ("_mock" if args.mock else "_mockllm" if args.mock_llm else ""))
    if not (args.mock or args.mock_llm):
        try:
            import selftest as grid_selftest
            grid_selftest.t0_manifest()
        except AssertionError as e:
            log(f"Code version check failed：{e}")
            return 1
    log(f"===== alfw_stage_b --stage {args.stage}  PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} "
        f"output {STATE['out']} =====")
    if args.stage == "judge":
        return stage_judge(args)
    stages = {"prep": ["collect", "design", "screen", "select"], "run": ["main", "analyze"],
              "all": ["collect", "design", "screen", "select", "main", "analyze"],
              "smoke": ["collect", "design", "screen", "select", "main", "analyze"]}.get(args.stage, [args.stage])
    try:
        rc = run_stages(stages, args)
        if args.stage == "smoke":
            rc = rc or smoke_summary(STATE["ctx"])
        return rc
    except SystemExit:
        raise
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != acfg.HASH_SEED:
        os.environ["PYTHONHASHSEED"] = acfg.HASH_SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.exit(main())
