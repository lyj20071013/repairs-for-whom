"""Exploratory non-privileged repair pipeline G (analysis plan section 6.12 G).

Flash with thinking selects one step and proposes two replacements from the
60 unscreened Qwen7B ALFWorld failures. Qwen7B supplies three independent
acceptance runs (GV), ten evaluation runs (GE), and one greedy run (GG) per
stochastic branch. The pipeline has no planner-based privileged screening.
Commands: check, prep, eval, all, analyze, selftest. API keys are read only
from QWEN_API_KEY. Records are saved incrementally for compatible resumption.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds_alfw as DS          # noqa: E402
import runner                 # noqa: E402
import alfw_stage_b as SB     # noqa: E402
from alfw import acfg         # noqa: E402
import ds612 as R             # noqa: E402

GCFG = {
    "N_PROPOSE": 2,          # Positioned steps up 2 An alternative action
    "N_VERIFY": 3,           # GV：Every Random Branch 3 runs
    "ACCEPT": 2,             # Two out of three
    "N_EVAL": 10,            # GE：Every Random Branch 10 runs
    "B_VERIFY": "GV", "B_EVAL": "GE", "B_GREEDY": "GG",
    "PLAN_TIMEOUT": 60,
    "MODEL": "qwen3.8-flash", "THINKING": True, "EFFORT": "medium",
    "EXECUTOR": "q7",
}
PATHS = {"pool": ROOT / "out_ds_alfw", "out": ROOT / "out_g612"}
STATE = {"calls": Counter()}           # Self-check：mock API Number of calls
VARIANTS = ("G", "G_nv", "G_greedy", "G_rel", "G_opt")
VLAB = {"G": "G（Three two acceptances）", "G_nv": "G_nv（Non-acceptance，First proposal）", "G_greedy": "G_greedy（greedy acceptance check）",
        "G_rel": "G_rel（Receiving and inspection > original action，description）", "G_opt": "G_opt（The two proposals are better，optimistic upper bound）",
        "6.9 all matched": "6.9 Match（all candidates，5 Selection 5 Commentary）", "6.9 model matched": "6.9 Match（model-proposed candidates）",
        "6.9 all greedy": "6.9 greedy replay（all candidates）", "6.9 model greedy": "6.9 greedy replay（model-proposed candidates）",
        "6.9 all SLfl_k1": "6.9 SL-flash k = 1", "6.9 all SLfl_k3": "6.9 SL-flash k = 3",
        "6.9 all SLfl_k5": "6.9 SL-flash k = 5"}


def log(msg):
    SB.log(msg)


def P(name):
    o = PATHS["out"]
    return {"locate": o / "locate.jsonl", "locate_cache": o / "api_cache" / "locate.jsonl",
            "steps": o / "steps.json", "proposals": o / "proposals.jsonl",
            "propose_cache": o / "api_cache" / "propose.jsonl", "design": o / "design.json",
            "evals": o / f"evals_{GCFG['EXECUTOR']}.jsonl", "timing": o / f"timing_{GCFG['EXECUTOR']}.json",
            "analysis": o / "analysis_g.json", "report": o / "report_g.md"}[name]


def settings():
    return {"G": GCFG, "MAX_STEPS": acfg.MAX_STEPS, "TEMPERATURE": acfg.TEMPERATURE, "TOP_P": acfg.TOP_P,
            "MODEL_DIR": str(acfg.MODEL_DIR), "deployer": DS.deployer(), "mock": DS.STATE["mock"]}


def use_out(out=None, pool=None):
    if pool is not None:
        PATHS["pool"] = Path(pool).resolve()
    if out is not None:
        PATHS["out"] = Path(out).resolve()
    DS.use_out(PATHS["out"])                 # Log to out_g612/ds.log；SB Simulate Switch Follow DS.STATE["mock"]


def load_pool_inputs():
    pd = PATHS["pool"]
    need = [pd / f for f in ("episodes.jsonl", "positions.json")]
    miss = [f.name for f in need if not f.exists()]
    if miss:
        sys.exit(f"[G] {pd} Missing 6.9 of {miss}")
    pos = DS.jload(pd / "positions.json")
    eps = {e["rid"]: e for e in DS.jl_read(pd / "episodes.jsonl")}
    return pos, eps


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ================================================================ localization
LOCATE_SYSTEM = ("You are an expert debugger of AI agents. You will read the log of a failed run of an agent in a "
                 "text-based household environment and find the step at which the run should be repaired.")
_LOC_RE = re.compile(r"step\s*\**\s*[:：]\s*\**[ \t]*(-?\d+)", re.I)


def preamble(ep):
    """and 6.9 Suggested prompts（ds_alfw.propose_messages）The same beginning：Environmental description、deployed executor、Tasks、Start Description（Self-checking is word-by-word）。"""
    return ("The agent acts in a text-based household environment. At every step it sees the task, the room description, "
            "its history of actions with the environment's feedback, and the list of actions possible right now, and it "
            "picks one of them. An action that is not possible does nothing (\"Nothing happens.\") but uses up a step. "
            f"The step budget is {acfg.MAX_STEPS} steps.\n\n"
            f"The agent is {DS.MODEL_DESC[DS.deployer()]} that samples its actions at temperature 0.7.\n\n"
            f"Task: {ep['task']}\n\n{ep['desc']}\n\n")


def locate_messages(ep):
    lines = [f"Step {i}: > {c or '(no action)'}\n{o}" for i, (c, o) in enumerate(zip(ep["cmds"], ep["obs"]))]
    T = len(lines)
    user = (preamble(ep)
            + f"Log of the failed run ({T} steps; the agent ran out of steps):\n" + "\n".join(lines)
            + f"\n\nWe can replace the agent's action at ONE step (any step from Step 0 to Step {T - 1}) by another "
              "action that is possible at that step, and then let the SAME agent continue on its own from there with "
              "the remaining steps. At which step should the action be replaced so that the agent is most likely to "
              "succeed? Think briefly, then end with exactly this line:\nStep: <number>")
    return [{"role": "system", "content": LOCATE_SYSTEM}, {"role": "user", "content": user}]


def parse_locate(text, T):
    """Take the last one `Step: <n>`；It has to be 0 ≤ n < T Integer。Back (t or None, Reason for invalidity or None)。"""
    ms = list(_LOC_RE.finditer(text or ""))
    if not ms:
        return None, "missing"
    t = int(ms[-1].group(1))
    if not 0 <= t < T:
        return None, "out_of_range"
    return t, None


class MockLocator:
    """Offline test：Take a step on the Hashi with the prompts；A small number of output blanks or cross-borders，To overwrite the parse branch。"""

    def __init__(self):
        self.name = "mock_locator"
        self.settings = {"backend": "mock", "name": self.name}

    def ask(self, msgs_list, on_done=None):
        import hashlib
        import random
        out = []
        for m in msgs_list:
            STATE["calls"]["locate"] += 1
            text = m[-1]["content"]
            T = int(re.search(r"Log of the failed run \((\d+) steps", text).group(1))
            rng = random.Random(hashlib.sha1(text.encode()).hexdigest())
            u, t = rng.random(), rng.randrange(T)
            if u < 0.12:
                body = "I cannot tell."
            elif u < 0.22:
                body = f"Step: {T + 3}"
            elif u < 0.35:
                body = f"Looking at Step {max(0, t - 1)}: it went wrong.\n**Step:** {t}"
            else:
                body = f"Reasoning (mock). Step: 0 looks fine.\nStep: {t}"
            out.append({"text": body, "finish_reason": "stop", "error": None,
                        "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 20}, "reasoning_chars": 100})
            if on_done:
                on_done(len(out) - 1, out[-1])
        return out


class _CountingProposer(DS.MockProposer):
    def ask(self, msgs_list, on_done=None):
        STATE["calls"]["propose"] += len(msgs_list)
        return super().ask(msgs_list, on_done)


def make_flash(role):
    import judge_backends as JB
    if DS.STATE["mock"]:
        return MockLocator() if role == "locator" else _CountingProposer("qflash_think")
    j = JB.make_judge("api", model=os.environ.get("DS_QFLASH_MODEL", GCFG["MODEL"]), thinking=GCFG["THINKING"],
                      effort=GCFG["EFFORT"])
    j.name = "qflash_think"
    j.settings["role"] = role
    return j


def _api_ok():
    if DS.STATE["mock"]:
        return True
    if not (os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")):
        log("[G] No environmental variables found QWEN_API_KEY（API Keys read from environment variables only）：first export QWEN_API_KEY=YOUR_API_KEY")
        return False
    return True


def stage_locate():
    import judge_backends as JB
    pos, eps = load_pool_inputs()
    trajs = pos["trajectories"]
    keys = [t["traj_id"] for t in trajs]
    msgs = [locate_messages(eps[k]) for k in keys]
    j = make_flash("locator")
    log(f"[G locate] {len(keys)} trajectories（flash thinking {GCFG['EFFORT']}）")
    outs = JB.ask_resumable(j, keys, msgs, P("locate_cache"))
    rows, why = [], Counter()
    for k, o in zip(keys, outs):
        o = o or {}
        T = len(eps[k]["cmds"])
        if o.get("error"):
            t, w = None, "api_error"
        else:
            t, w = parse_locate(o.get("text"), T)
        why[w or "valid"] += 1
        rows.append({"traj_id": k, "t": t, "why": w, "T": T, "text": o.get("text"), "error": o.get("error"),
                     "finish_reason": o.get("finish_reason"), "usage": o.get("usage"),
                     "reasoning_chars": o.get("reasoning_chars"), "settings": j.settings,
                     "positions_id": pos["positions_id"]})
    write_jsonl(P("locate"), rows)
    log(f"[G locate] Validity {why['valid']}/{len(rows)}；{dict(why)}")
    if why["api_error"]:
        log(f"[G locate] {why['api_error']} Error calling bar；rerun the same command to retry only these requests")
        return 1
    return 0


def load_locate():
    return {r["traj_id"]: r for r in DS.jl_read(P("locate"))}


# ================================================================ Positioned Steps：Executable command（Planner only describes）
def stage_steps():
    pos, eps = load_pool_inputs()
    loc = load_locate()
    if len(loc) != len(pos["trajectories"]) or any(r["why"] == "api_error" for r in loc.values()):
        sys.exit("[G steps] localization is incomplete（Run first prep）")
    key = DS.fp(sorted((k, r["t"]) for k, r in loc.items()))
    if P("steps").exists():
        s = DS.jload(P("steps"))
        if s.get("locate_fp") == key and s.get("settings") == settings():
            log(f"[G steps] Reuse already available {len(s['rows'])} A locator")
            return 0
        sys.exit("[G steps] Existing steps.json Different from current location results；Please change the output directory")
    items, ks = [], []
    for t in pos["trajectories"]:
        r = loc[t["traj_id"]]
        if r["t"] is None:
            continue
        e = eps[t["traj_id"]]
        items.append({"gamefile": e["gamefile"], "prefix": e["cmds"][:r["t"]], "timeout": GCFG["PLAN_TIMEOUT"],
                      "want_adm": True})
        ks.append(t["traj_id"])
    with SB.make_pool() as pl:
        pr = SB.probe_all(pl, items, "Positioning Step") if items else []
    rows = {}
    for k, r in zip(ks, pr):
        t = loc[k]["t"]
        L = None if r["timeout"] else len(r["plan"])
        rows[k] = {"t": t, "orig_action": eps[k]["cmds"][t], "admissible": list(r["admissible"]),
                   "planner": r["plan"][0] if r["plan"] else None, "L": L,
                   "slack": None if L is None else (acfg.MAX_STEPS - t) - L, "timeout": bool(r["timeout"])}
    DS.jdump(P("steps"), {"rows": rows, "locate_fp": key, "settings": settings()})
    log(f"[G steps] {len(rows)} A locator（Planning timeout {sum(v['timeout'] for v in rows.values())}）")
    return 0


# ================================================================ proposal（6.9 preserve prompts and parsing rules）
def stage_propose():
    import judge_backends as JB
    pos, eps = load_pool_inputs()
    steps = DS.jload(P("steps"))["rows"]
    items = []
    for t in pos["trajectories"]:
        k = t["traj_id"]
        if k not in steps:
            continue
        s = steps[k]
        p = {"t": s["t"], "orig_action": s["orig_action"], "admissible": s["admissible"]}
        items.append({"traj_id": k, "t": s["t"], "pos": p,
                      "messages": DS.propose_messages(eps[k], p, GCFG["N_PROPOSE"])})
    keys = [f"{it['traj_id']}|t{it['t']}" for it in items]
    j = make_flash("proposer")
    log(f"[G propose] {len(items)} A locator（flash thinking {GCFG['EFFORT']}）")
    outs = JB.ask_resumable(j, keys, [it["messages"] for it in items], P("propose_cache")) if items else []
    rows, n_valid, why, n_err = [], 0, Counter(), 0
    for it, o in zip(items, outs):
        o = o or {}
        if o.get("error"):
            acts, bad = [], [{"i": 0, "why": "api_error"}]
            n_err += 1
        else:
            acts, bad = DS.parse_proposals(o.get("text"), it["pos"]["admissible"], it["pos"]["orig_action"],
                                           GCFG["N_PROPOSE"])
        n_valid += len(acts)
        for b in bad:
            why[b["why"]] += 1
        rows.append({"traj_id": it["traj_id"], "t": it["t"], "actions": acts, "bad": bad, "text": o.get("text"),
                     "error": o.get("error"), "finish_reason": o.get("finish_reason"), "usage": o.get("usage"),
                     "reasoning_chars": o.get("reasoning_chars"), "settings": j.settings})
    write_jsonl(P("proposals"), rows)
    log(f"[G propose] Valid proposal {n_valid}/{len(items) * GCFG['N_PROPOSE']}；reason for invalidity {dict(why)}；Error calling {n_err}")
    if n_err:
        log("[G propose] There was an error in calling；rerun the same command to retry only these requests")
        return 1
    return 0


def load_proposals():
    return {r["traj_id"]: r for r in DS.jl_read(P("proposals"))}


# ================================================================ design（Branch-by-branch planning；No change after freezing）
def stage_design():
    pos, eps = load_pool_inputs()
    steps = DS.jload(P("steps"))["rows"]
    props = load_proposals()
    if set(props) != set(steps) or any(r["error"] for r in props.values()):
        sys.exit("[G design] The proposal isnt all done yet（Run first prep）")
    inputs = DS.fp({"steps": {k: [v["t"], v["orig_action"]] for k, v in steps.items()},
                    "props": {k: r["actions"] for k, r in props.items()}, "s": settings(),
                    "pool": pos["positions_id"]})
    if P("design").exists():
        d = DS.jload(P("design"))
        if d.get("inputs_fp") == inputs:
            log(f"[G design] Reuse already designed {d['design_id'][:12]}")
            return 0
        sys.exit("[G design] The design is different from the current input；Please change the output directory")
    trajs, branches = [], {}
    for tr in pos["trajectories"]:
        k = tr["traj_id"]
        if k not in steps:
            continue
        e, s = eps[k], steps[k]
        t, orig = s["t"], s["orig_action"]
        oid = f"{k}|t{t}|orig"
        branches[oid] = {"branch_id": oid, "traj_id": k, "t": t, "action": "orig:" + orig, "gamefile": e["gamefile"],
                         "prefix": e["cmds"][:t + 1], "rank": None}
        alts = []
        for i, a in enumerate(props[k]["actions"]):
            bid = f"{k}|t{t}|{a}"
            branches[bid] = {"branch_id": bid, "traj_id": k, "t": t, "action": a, "gamefile": e["gamefile"],
                             "prefix": e["cmds"][:t] + [a], "rank": i, "planner": a == s["planner"]}
            alts.append(bid)
        trajs.append({"traj_id": k, "gamefile": e["gamefile"], "task_type": tr["task_type"], "T": len(e["cmds"]),
                      "positions": [{"t": t, "orig": oid, "alts": alts, "L": s["L"], "slack": s["slack"]}]})
    log(f"[G design] {len(trajs)} trajectories、{len(branches)} branches；One by one（Determination of the branch of certainty；Feasibility and residuals are described only）")
    bids = sorted(branches)
    with SB.make_pool() as pl:
        pr = SB.probe_all(pl, [{"gamefile": branches[b]["gamefile"], "prefix": branches[b]["prefix"],
                                "timeout": GCFG["PLAN_TIMEOUT"]} for b in bids], "branch") if bids else []
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
    d = {"kind": "g612", "trajectories": trajs, "branches": branches, "inputs_fp": inputs, "settings": settings(),
         "source_positions_id": pos["positions_id"], "n_branch_timeout": n_to}
    d["design_id"] = DS.fp({"t": trajs, "b": {k: {kk: v.get(kk) for kk in ("prefix", "deterministic")}
                                           for k, v in branches.items()}, "s": d["settings"]})
    DS.jdump(P("design"), d)
    nst = sum(1 for b in branches.values() if b["deterministic"] is None)
    log(f"[G design] complete {d['design_id'][:12]}：Random Branch {nst} items（Planning timeout {n_to}），deterministic branch {len(branches) - nst} items")
    return 0


def stage_prep():
    if not _api_ok():
        return 1
    for name, fn in (("locate", stage_locate), ("steps", stage_steps), ("propose", stage_propose),
                     ("design", stage_design)):
        rc = fn()
        if rc != 0:
            log(f"[G prep] {name} Not completed（rc={rc}）；after fixing the issue, rerun the same command to resume here")
            return rc
    return 0


# ================================================================ continuation（7B：GV 3 runs、GE 10 runs、GG greedy 1 runs）
def passes():
    ex = GCFG["EXECUTOR"]
    return [(f"{ex}_T07", GCFG["B_VERIFY"], GCFG["N_VERIFY"], acfg.TEMPERATURE),
            (f"{ex}_T07", GCFG["B_EVAL"], GCFG["N_EVAL"], acfg.TEMPERATURE),
            (f"{ex}_greedy", GCFG["B_GREEDY"], 1, 0.0)]


def missing(d):
    have = runner.load_results(P("evals")) if P("evals").exists() else {}
    return [(c, bt, b) for c, bt, n, _ in passes() for b, x in d["branches"].items()
            if x["deterministic"] is None and (c, bt, b) not in have]


def stage_eval():
    if not P("design").exists():
        sys.exit("[G eval] Not yet design.json（Run first prep）")
    d = DS.jload(P("design"))
    ex = GCFG["EXECUTOR"]
    if not DS.STATE["mock"]:
        try:
            import selftest as grid_selftest
            grid_selftest.t0_manifest()
        except AssertionError as e:
            log(f"Code version check failed：{e}")
            return 1
        if not DS.wait_gpu("g612-eval"):
            return 1
    holder = {}

    def backend():                       # One in three batches vLLM Engine
        if "b" not in holder:
            holder["b"] = DS.make_exec_backend(ex)
        return holder["b"]
    ctx = {"backend": backend, "stop_after": DS.STATE.get("stop_after")}
    try:
        with SB.make_pool() as pl:
            ctx["pool"] = pl
            for cond, batch, n, temp in passes():
                have = runner.load_results(P("evals")) if P("evals").exists() else {}
                todo = sum(1 for b, x in d["branches"].items() if x["deterministic"] is None and (cond, batch, b) not in have)
                tm = SB.run_branches(ctx, d, cond, batch, n, P("evals"), temp)
                if tm:
                    f = P("timing")
                    old = DS.jload(f) if f.exists() else {"runs": []}
                    old["runs"].append({"cond": f"{cond}/{batch}", "wall": tm["wall"], "policy_steps": tm["policy_steps"],
                                        "rollouts": todo * n, "gpu": DS.gpu_info(),
                                        "tp": int(os.environ.get("LADDER_TP", "1")), "time": time.strftime("%Y-%m-%d %H:%M")})
                    DS.jdump(f, old)
    finally:
        DS.release_vllm(holder.get("b"))
    left = missing(d)
    log("[G eval] " + ("All completed" if not left else f"Not much {len(left)} items（Condition, Batch, branch），The same order will be replaced by the same order"))
    return 0 if not left else 1


# ================================================================ Analysis
def _vals(d, ev, b, cond, batch, n):
    x = d["branches"][b]
    if x["deterministic"] is not None:
        return [float(x["deterministic"])] * n
    return [float(v) for v in ev[(cond, batch, b)]["results"][:n]]


def traj_record(d, ev, t):
    """a trajectory in G The full amount you need in there：proposal（In the order of the proposal）of GV Receiving and inspection s、GG Greedy results、GE Mean；The same thing as before。"""
    ex = GCFG["EXECUTOR"]
    p = t["positions"][0]

    def rec(b):
        gv = _vals(d, ev, b, f"{ex}_T07", GCFG["B_VERIFY"], GCFG["N_VERIFY"])
        ge = _vals(d, ev, b, f"{ex}_T07", GCFG["B_EVAL"], GCFG["N_EVAL"])
        gg = _vals(d, ev, b, f"{ex}_greedy", GCFG["B_GREEDY"], 1)
        return {"bid": b, "s": int(round(sum(gv))), "ge": float(np.mean(ge)), "gg": int(gg[0])}
    return {"traj_id": t["traj_id"], "t": p["t"], "orig": rec(p["orig"]), "alts": [rec(a) for a in p["alts"]]}


def choose(r, variant):
    """Returns the selected proposal（r['alts'] One of them）or None（no repair）。r = traj_record Outcome；Proposal in the order in which it is proposed。"""
    alts = r["alts"] if r else []
    if not alts:
        return None
    if variant == "G":
        acc = [a for a in alts if a["s"] >= GCFG["ACCEPT"]]
    elif variant == "G_rel":
        acc = [a for a in alts if a["s"] > r["orig"]["s"]]
    elif variant == "G_nv":
        return alts[0]
    elif variant == "G_greedy":
        return next((a for a in alts if a["gg"] == 1), None)
    elif variant == "G_opt":
        return max(alts, key=lambda a: (a["ge"] - r["orig"]["ge"], -alts.index(a)))
    else:
        raise ValueError(variant)
    if not acc:
        return None
    return max(acc, key=lambda a: (a["s"], -alts.index(a)))          # Maximum number of receiving and inspection；Columns in the order of the proposal


def value(r, variant, obj):
    """net gain：Selected − Same move（GE），no repair 0。Success rate after restoration：Selected GE，Do not fix the original action GE；Do not set to step as None。"""
    c = choose(r, variant)
    if obj == "net":
        return 0.0 if c is None else c["ge"] - r["orig"]["ge"]
    if r is None:
        return None
    return r["orig"]["ge"] if c is None else c["ge"]


def cost(r, d, variant):
    """Number of repeaters deployed（exclude deterministic branches）。"""
    if r is None:
        return 0
    stoch = lambda b: d["branches"][b]["deterministic"] is None
    na = sum(stoch(a["bid"]) for a in r["alts"])
    if variant == "G":
        return GCFG["N_VERIFY"] * na
    if variant == "G_rel":
        return GCFG["N_VERIFY"] * (na + stoch(r["orig"]["bid"]))
    if variant == "G_greedy":
        return na
    return 0


def _paired(x, y):
    """Both sides have a defined match between the trajectories。"""
    z = [a - b for a, b in zip(x, y) if a is not None and b is not None]
    return DS.boot(z)


def _level(x):
    return DS.boot([v for v in x if v is not None])


def analyze(pool_dir=None, out=None, rows69=None):
    if pool_dir is not None or out is not None:
        use_out(out, pool_dir)
    pos, eps = load_pool_inputs()
    d = DS.jload(P("design"))
    ev = runner.load_results(P("evals"))
    left = missing(d)
    if left:
        log(f"[G analyze] continuations still missing {len(left)} items（Condition, Batch, branch），Do not calculate any readings")
        return None
    loc, props = load_locate(), load_proposals()
    if {r["positions_id"] for r in loc.values()} | {d["source_positions_id"]} != {pos["positions_id"]}:
        sys.exit(f"[G analyze] out_g612 Positioning / Thats not the design 6.9 of positions.json（{pos['positions_id'][:12]}）performed on")
    steps = DS.jload(P("steps"))["rows"]
    ids = [t["traj_id"] for t in pos["trajectories"]]
    tr_by = {t["traj_id"]: t for t in d["trajectories"]}
    recs = {k: (traj_record(d, ev, tr_by[k]) if k in tr_by else None) for k in ids}
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "design_id": d["design_id"], "n_traj": len(ids),
           "settings": d["settings"], "levels": {}, "compare": {}, "verdict": {}, "describe": {}, "budget": {}}
    series = {obj: {v: [value(recs[k], v, obj) for k in ids] for v in VARIANTS} for obj in ("net", "post")}
    for obj in ("net", "post"):
        res["levels"][obj] = {v: _level(series[obj][v]) for v in VARIANTS}
    # ---- 6.9 Selection（Same 60 trajectories；With registered analytical code on 7B Recalculate All Samples）
    if rows69 is None:
        pool = R.load_pool("7B", PATHS["pool"])
        if pool is None or "q7" not in pool["ex"]:
            sys.exit(f"[G analyze] {PATHS['pool']} in 6.9 of 7B incomplete continuations")
        d_all = pool["d"]
        d_mod, empty = DS.restrict_design(d_all, lambda b: bool(b.get("proposers")))
        rows69 = {}
        for var, dv in (("all", d_all), ("model", d_mod)):
            for obj in ("net", "post"):
                rr = {r["traj_id"]: r for r in R.run_core(pool, dv, "q7", [], obj)}
                rows69[(var, obj)] = rr
    s69 = {}
    for obj in ("net", "post"):
        for var in ("all", "model"):
            rr = rows69[(var, obj)]
            pad = 0.0 if obj == "net" else None          # Model Subset is an empty trajectory（Same 6.9a）
            get = lambda key: [rr[k][key] if k in rr else pad for k in ids]
            s69[(var, obj, "matched")] = get("matched")
            s69[(var, obj, "greedy")] = get("greedy")
            for kk in (1, 3, 5):
                s69[(var, obj, f"SLfl_k{kk}")] = get(f"SLfl_k{kk}")
        res["levels"][obj].update({f"6.9 {var} {m}": _level(s69[(var, obj, m)]) for var in ("all", "model")
                                   for m in ("matched", "greedy", "SLfl_k1", "SLfl_k3", "SLfl_k5")})
        G = series[obj]["G"]
        cmp_ = {
            "Match(model) − G": _paired(s69[("model", obj, "matched")], G),
            "Match(all) − G": _paired(s69[("all", obj, "matched")], G),
            "G − greedy(model)": _paired(G, s69[("model", obj, "greedy")]),
            "G − greedy(all)": _paired(G, s69[("all", obj, "greedy")]),
            "G − SL-flash k=1": _paired(G, s69[("all", obj, "SLfl_k1")]),
            "G − SL-flash k=3": _paired(G, s69[("all", obj, "SLfl_k3")]),
            "G − SL-flash k=5": _paired(G, s69[("all", obj, "SLfl_k5")]),
            "G − G_nv": _paired(G, series[obj]["G_nv"]),
            "G − G_greedy": _paired(G, series[obj]["G_greedy"]),
            "G_rel − G": _paired(series[obj]["G_rel"], G),
            "G_opt − G": _paired(series[obj]["G_opt"], G),
        }
        res["compare"][obj] = cmp_
    # ---- budget（Every trajectories 7B Number of repeat strips）
    r_all = rows69[("all", "net")]
    r_mod = rows69[("model", "net")]
    res["budget"] = {
        "G": float(np.mean([cost(recs[k], d, "G") for k in ids])),
        "G_rel": float(np.mean([cost(recs[k], d, "G_rel") for k in ids])),
        "G_greedy（greedy）": float(np.mean([cost(recs[k], d, "G_greedy") for k in ids])),
        "G_nv": 0.0,
        "6.9 Match(all)": float(np.mean([5 * r_all[k]["budget_all"] for k in ids])),
        "6.9 Match(model)": float(np.mean([5 * r_mod[k]["budget_all"] if k in r_mod else 0 for k in ids])),
        "6.9 greedy(all)（greedy）": float(np.mean([r_all[k]["budget_all"] for k in ids])),
        "6.9 greedy(model)（greedy）": float(np.mean([r_mod[k]["budget_all"] if k in r_mod else 0 for k in ids])),
        **{f"6.9 SL-flash k={kk}": float(np.mean([kk * r_all[k]["budget_SLfl"] for k in ids])) for kk in (1, 3, 5)},
        "flash call（G）": 2.0, "flash call（6.9 SL-flash，thinking disabled）": 3.0,
    }
    # ---- Reading（Registration，2026-10-26 01:40）
    g = res["levels"]["net"]["G"]
    nv = res["compare"]["net"]["G − G_nv"]
    res["verdict"]["G"] = ("G lower bound > 0：You dont need a privileged signal“LLM localization + LLM proposal + Minor acceptance and inspection by deployed personnel”Itll bring a positive restoration value"
                           if g["ci"][0] > 0 else "G Intersection 0（Or negative）：number 7 Retain section first sentence，G As another unsuccessful attempt,")
    res["verdict"]["G − G_nv"] = ("lower bound > 0：deployed-executor acceptance is useful in this pipeline" if nv["ci"][0] > 0 else
                                  "Intersection 0：One position、The value of acceptances at the time of the two candidates is uncertain")
    # ---- description
    loc_rows = [loc[k] for k in ids]
    located = [k for k in ids if loc[k]["t"] is not None]
    pos69 = {t["traj_id"]: {p["t"] for p in t["positions"]} for t in pos["trajectories"]}
    third = Counter()
    for k in located:
        tt, T = loc[k]["t"], loc[k]["T"]
        third["early" if tt < T / 3 else ("middle" if tt < 2 * T / 3 else "late")] += 1
    sl = [steps[k]["slack"] for k in located if k in steps]
    n_bad = Counter(b["why"] for k in props for b in props[k]["bad"])
    n_prop = sum(len(props[k]["actions"]) for k in props)
    plan_ok = [k for k in located if steps[k]["planner"] is not None and steps[k]["planner"] != steps[k]["orig_action"]]
    plan_in = sum(steps[k]["planner"] in props[k]["actions"] for k in plan_ok)
    acc_any = sum(1 for k in ids if recs[k] and any(a["s"] >= GCFG["ACCEPT"] for a in recs[k]["alts"]))
    s_dist = Counter(a["s"] for k in ids if recs[k] for a in recs[k]["alts"])
    chosen = [choose(recs[k], "G") for k in ids if recs[k]]
    chosen = [c for c in chosen if c is not None]

    def usage(rows):
        u = [r.get("usage") or {} for r in rows]
        return {"n": len(rows), "prompt_tokens": int(sum(x.get("prompt_tokens") or 0 for x in u)),
                "completion_tokens": int(sum(x.get("completion_tokens") or 0 for x in u)),
                "reasoning_chars_median": float(np.median([r.get("reasoning_chars") or 0 for r in rows])) if rows else 0,
                "finish": dict(Counter(str(r.get("finish_reason")) for r in rows))}
    res["describe"] = {
        "located": len(located), "locate_invalid": dict(Counter(r["why"] for r in loc_rows if r["why"])),
        "t_mean": float(np.mean([loc[k]["t"] for k in located])) if located else None,
        "t_quartiles": [float(x) for x in np.quantile([loc[k]["t"] for k in located], [0.25, 0.5, 0.75])] if located else None,
        "thirds": dict(third),
        "slack_ge5": (sum(1 for x in sl if x is not None and x >= 5), len(sl), sum(1 for x in sl if x is None)),
        "same_as_69_position": sum(loc[k]["t"] in pos69[k] for k in located),
        "n_admissible_median": float(np.median([len(steps[k]["admissible"]) for k in located])) if located else None,
        "proposals_valid": (n_prop, GCFG["N_PROPOSE"] * len(props)), "proposals_bad": dict(n_bad),
        "traj_with_valid_proposal": sum(1 for k in props if props[k]["actions"]),
        "planner_in_proposals": (int(plan_in), len(plan_ok)),
        "accepted_any": acc_any, "s_dist": {int(a): int(b) for a, b in sorted(s_dist.items())},
        "chosen_is_planner": sum(bool(d["branches"][c["bid"]].get("planner")) for c in chosen), "n_chosen": len(chosen),
        "chosen_net_mean": float(np.mean([c["ge"] - recs[c["bid"].split("|")[0]]["orig"]["ge"] for c in chosen]))
        if chosen else None,
        "branch_success_alts": float(np.mean([a["ge"] for k in ids if recs[k] for a in recs[k]["alts"]]))
        if any(recs[k] and recs[k]["alts"] for k in ids) else None,
        "branch_success_orig": float(np.mean([recs[k]["orig"]["ge"] for k in ids if recs[k]])) if located else None,
        "deterministic_alts": sum(1 for b in d["branches"].values() if b["rank"] is not None and b["deterministic"] is not None),
        "api_locate": usage(loc_rows), "api_propose": usage(list(props.values())),
    }
    tf = P("timing")
    if tf.exists():
        runs = DS.jload(tf)["runs"]
        res["timing"] = {"wall_min": sum(r["wall"] for r in runs) / 60, "rollouts": sum(r["rollouts"] for r in runs),
                         "gpu": (runs[-1].get("gpu") or {}).get("names")}
    res["per_traj"] = [{"traj_id": k, "t": loc[k]["t"], "why": loc[k]["why"],
                        "proposals": props[k]["actions"] if k in props else [],
                        **({"s": [a["s"] for a in recs[k]["alts"]], "ge_alts": [a["ge"] for a in recs[k]["alts"]],
                            "gg": [a["gg"] for a in recs[k]["alts"]], "ge_orig": recs[k]["orig"]["ge"],
                            "s_orig": recs[k]["orig"]["s"]} if recs[k] else {}),
                        **{f"{v}_net": series["net"][v][i] for v in VARIANTS},
                        "matched_model_net": s69[("model", "net", "matched")][i],
                        "matched_all_net": s69[("all", "net", "matched")][i]}
                       for i, k in enumerate(ids)]
    DS.jdump(P("analysis"), res)
    P("report").write_text(report(res), encoding="utf-8")
    log(f"[G analyze] -> {P('report')}")
    return res


def report(res):
    f3 = R.f3
    L = [f"# 6.12 G：LLM localization + LLM proposal + Receiving and inspection of deployed personnel（{res['generated']}）", "",
         "For registration see analysis plan.md 6.12 G（2026-10-26 01:40）。all exploratory，excluded from Holm。interval：trajectory bootstrap 2,000 runs，90% Percentage。",
         f"design_id `{res['design_id'][:12]}`；60 records 6.9 of 7B Failed trajectories。", ""]
    for v, txt in res["verdict"].items():
        L.append(f"- **Reading（{v}）**：{txt}")
    L += ["", "## Horizontal value（_Other Organiser 7B Value；GE 10 Sub-evaluations）", "", "| Selection | net gain | Success rate after restoration | per trajectory 7B continuation |",
          "|---|---|---|---|"]
    bud = res["budget"]
    bmap = {"G": "G", "G_rel": "G_rel", "G_greedy": "G_greedy（greedy）", "G_nv": "G_nv",
            "6.9 all matched": "6.9 Match(all)", "6.9 model matched": "6.9 Match(model)",
            "6.9 all greedy": "6.9 greedy(all)（greedy）", "6.9 model greedy": "6.9 greedy(model)（greedy）",
            "6.9 all SLfl_k1": "6.9 SL-flash k=1", "6.9 all SLfl_k3": "6.9 SL-flash k=3", "6.9 all SLfl_k5": "6.9 SL-flash k=5"}
    for key in list(VARIANTS) + ["6.9 all matched", "6.9 model matched", "6.9 all greedy", "6.9 model greedy",
                                 "6.9 all SLfl_k1", "6.9 all SLfl_k3", "6.9 all SLfl_k5"]:
        lab = VLAB.get(key, key)
        b = bud.get(bmap.get(key), None)
        L.append(f"| {lab} | {f3(res['levels']['net'][key])} | {f3(res['levels']['post'][key])}"
                 f"（n = {res['levels']['post'][key]['n']}） | {'—' if b is None else f'{b:.1f}'} |")
    L += ["", "（6.9 of“model”= At least one model put forward a candidate（6.9a）；SL-flash = with thinking disabled flash at 6.9 First proposal on three positions。"
          "greedys follow-up is a cure for greedy；G_opt Selection by evaluation sample，optimistically biased。）", "",
          "## Matching（Same 60 trajectories）", "", "| comparison | net gain | Success rate after restoration |", "|---|---|---|"]
    for k in res["compare"]["net"]:
        L.append(f"| {k} | {f3(res['compare']['net'][k])} | {f3(res['compare']['post'][k])}（n = {res['compare']['post'][k]['n']}） |")
    D = res["describe"]
    sg = D["slack_ge5"]
    pp = D["planner_in_proposals"]
    L += ["", "## description", "",
          f"- Position confirmed {D['located']} / {res['n_traj']}；Invalid {D['locate_invalid'] or 'none'}。"
          f"Positioned Steps：Average {D['t_mean']:.1f}，Quarter {D['t_quartiles']}；early / middle / late {D['thirds']}；"
          f"and 6.9 One of the three positions overlaps {D['same_as_69_position']}；This step executes the median command {D['n_admissible_median']}。"
          if D["located"] else f"- Position confirmed 0 / {res['n_traj']}；Invalid {D['locate_invalid']}",
          f"- slack ≥ 5（Planner Count，Just a description）：{sg[0]} / {sg[1]}（Planning timeout {sg[2]}）。",
          f"- proposal：Validity {D['proposals_valid'][0]} / {D['proposals_valid'][1]}，reason for invalidity {D['proposals_bad'] or 'none'}；"
          f"At least one valid proposed trajectory {D['traj_with_valid_proposal']}；Suggest Planner Actions {pp[0]} / {pp[1]}"
          f"（Positioning steps that exist and differ from the original action of the planner action）；proposals that terminate immediately after replacement {D['deterministic_alts']} items。",
          f"- acceptance check：At least one of the proposed accepted trajectories {D['accepted_any']} / {res['n_traj']}；Distribution of proposed acceptances and inspections {D['s_dist']}；"
          f"G repaired {D['n_chosen']} records，The action of the planner {D['chosen_is_planner']} records，Select the average net gain from restoration "
          + (f"{D['chosen_net_mean']:+.3f}" if D["chosen_net_mean"] is not None else "—") + "。",
          f"- 7B success rate（GE）：Proposed Branch " + (f"{D['branch_success_alts']:.3f}" if D["branch_success_alts"] is not None else "—")
          + "，Original Action Branch " + (f"{D['branch_success_orig']:.3f}" if D["branch_success_orig"] is not None else "—") + "。"]
    for tag, key in (("localization", "api_locate"), ("proposal", "api_propose")):
        u = D[key]
        L.append(f"- flash（{tag}）：{u['n']} runs，input {u['prompt_tokens']:,} token、output {u['completion_tokens']:,} token，"
                 f"In the middle of the chain of thought {u['reasoning_chars_median']:.0f} Character，Quantity {u['finish']}。")
    if res.get("timing"):
        tm = res["timing"]
        L.append(f"- continuation：{tm['rollouts']} records，{tm['wall_min']:.1f} minutes（{tm['gpu']}）。")
    L += ["", "In the budget line G_rel With original action 3 Receivables/receivables；flash call：G per trajectory 2 runs（thinking），6.9 of SL-flash 3 runs（thinking disabled）。", ""]
    return "\n".join(L) + "\n"


# ================================================================ check（Check only，Dont change anything）
def stage_check():
    ok = True
    pd = PATHS["pool"]
    for f in ("episodes.jsonl", "positions.json", "design.json", "evals_q7.jsonl"):
        e = (pd / f).exists()
        ok &= e
        print(f"  {'✓' if e else '✗'} 6.9 input {pd / f}")
    if (pd / "positions.json").exists():
        pos = DS.jload(pd / "positions.json")
        n = len(pos["trajectories"])
        ok &= n == 60
        print(f"  {'✓' if n == 60 else '✗'} 6.9 trajectory {n} records（positions_id {pos['positions_id'][:12]}）")
    key = bool(os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))
    ok &= key
    print(f"  {'✓' if key else '✗'} Environmental variables QWEN_API_KEY " + ("set" if key else "No Settings（export QWEN_API_KEY=YOUR_API_KEY）"))
    md = Path(acfg.MODEL_DIR)
    ok &= md.exists()
    print(f"  {'✓' if md.exists() else '✗'} 7B model {md}")
    try:
        import selftest as grid_selftest
        grid_selftest.t0_manifest()
    except AssertionError as e:
        ok = False
        print(f"  ✗ Code List：{e}")
    try:
        fr = DS.gpu_free_frac()
        print(f"  · Portfolio free memory ratio {fr}")
    except Exception as e:                       # only display a notice on machines without a GPU
        print(f"  · A visual card：{e!r}")
    out = PATHS["out"]
    print(f"  · Output Directory {out}（{'Existing：Well keep running' if out.exists() else 'It doesnt exist yet'}）")
    print("[G check] " + ("ready" if ok else "issues found，see above ✗"))
    return 0 if ok else 1


# ================================================================ self-test
def selftest():
    import shutil
    ok = [True]

    def check(c, msg):
        print(("  ✓ " if c else "  ✗ ") + msg, flush=True)
        ok[0] = ok[0] and bool(c)
    tmp = Path(tempfile.mkdtemp(prefix="g612_"))
    saved_cfg = json.loads(json.dumps(DS.CFG))
    saved_paths = dict(PATHS)
    try:
        # ---------------- 1. Parsing and selection rules（known-answer test）
        print("[g612 selftest] Parsing and selection rules")
        check(parse_locate("think...\nStep: 12", 50) == (12, None), "Step: 12 -> 12")
        check(parse_locate("Step: 3 is bad. Step 4: > go\nStep: 7", 50) == (7, None), "Take the last one Step:（in the log Step 4: Not really）")
        check(parse_locate("**Step:** 5", 50) == (5, None) and parse_locate("**Step: 6**", 50) == (6, None), "Bold format")
        check(parse_locate("Step: 50", 50) == (None, "out_of_range") and parse_locate("Step: -1", 50)[0] is None,
              "Cross-border invalid")
        check(parse_locate("I think step 12.", 50) == (None, "missing") and parse_locate(None, 50) == (None, "missing"),
              "Nothing Step: Line Invalid")

        def mk(s, ge, gg, so=0, geo=0.0):
            return {"orig": {"bid": "o", "s": so, "ge": geo, "gg": 0},
                    "alts": [{"bid": f"a{i}", "s": a, "ge": b, "gg": c} for i, (a, b, c) in enumerate(zip(s, ge, gg))]}
        r = mk([3, 2], [0.9, 0.5], [0, 1])
        check(choose(r, "G")["bid"] == "a0" and choose(mk([1, 2], [.1, .6], [0, 0]), "G")["bid"] == "a1",
              "G：Receiving and inspection ≥ 2 Top of the list")
        check(choose(mk([2, 2], [.2, .9], [0, 0]), "G")["bid"] == "a0", "G：Columns in the order of the proposal")
        check(choose(mk([1, 1], [.9, .9], [1, 1]), "G") is None and value(mk([1, 1], [.9, .9], [1, 1]), "G", "net") == 0.0,
              "G：no proposal accepted -> no repair，net gain 0")
        check(value(mk([1, 0], [.3, .1], [0, 0], geo=0.2), "G", "post") == 0.2, "Rehabilitate the original action with a repair success rate if not restored")
        check(choose(r, "G_nv")["bid"] == "a0" and choose(r, "G_greedy")["bid"] == "a1"
              and choose(mk([3, 3], [.5, .5], [0, 0]), "G_greedy") is None, "G_nv Take the first one；G_greedy The first one to succeed in greedy")
        check(choose(mk([1, 2], [.1, .6], [0, 0], so=1), "G_rel")["bid"] == "a1"
              and choose(mk([1, 1], [.1, .6], [0, 0], so=1), "G_rel") is None, "G_rel：Receiving and inspection > original action")
        check(choose(mk([0, 0], [.2, .7], [0, 0], geo=0.1), "G_opt")["bid"] == "a1"
              and value(mk([0, 0], [.2, .7], [0, 0], geo=0.1), "G_opt", "net") == 0.6, "G_opt：Large net gains")
        check(value(None, "G", "net") == 0.0 and value(None, "G", "post") is None and choose({"alts": []}, "G") is None,
              "No moves：net gain 0、Post-repair success rate not included")
        # Big samples know the answer：Original Actions and Proposals 1 success rate 0；Half-trajectories proposal 2 success rate 0.9，The other half 0
        # -> G = 0.9 × P(3 Secondary ≥ 2 Sub-success | 0.9) × Good trajectories Ratio；G_nv = 0；G_greedy = 0.9 × Good trajectories Ratio
        rng = np.random.default_rng(0)
        recs, good = [], []
        for _ in range(4000):
            p1, p2 = 0.0, float(rng.choice([0.0, 0.9]))
            s1, s2 = rng.binomial(3, p1), rng.binomial(3, p2)
            recs.append(mk([s1, s2], [rng.binomial(10, p1) / 10, rng.binomial(10, p2) / 10], [0, int(p2 > 0.5)]))
            good.append(p2 > 0)
        g = np.mean([value(x, "G", "net") for x in recs])
        nv = np.mean([value(x, "G_nv", "net") for x in recs])
        gg = np.mean([value(x, "G_greedy", "net") for x in recs])
        exp = 0.9 * (3 * 0.9 ** 2 * 0.1 + 0.9 ** 3) * np.mean(good)
        check(abs(g - exp) < 0.015 and nv == 0.0 and abs(gg - 0.9 * np.mean(good)) < 0.015,
              f"known-answer test：G ≈ {exp:.3f}（Actual {g:.3f}），G_nv = 0（Actual {nv:.3f}），G_greedy ≈ {0.9 * np.mean(good):.3f}（Actual {gg:.3f}）")

        # ---------------- 2. mock environment + Simulation model runs through the process
        print("[g612 selftest] mock 6.9 The pool（ds_alfw its own mock pipeline）")
        DS.STATE["mock"] = True
        DS.CFG.update(N_COLLECT=40, N_TRAJ=8, N_EXEC={"q7": 10, "q14": 4})
        acfg.B_MAX_ACTIVE = 60
        A = tmp / "A"
        DS.use_out(A)
        assert DS.stage_collect() == 0 and DS.stage_positions() == 0
        assert all(DS.stage_propose(n) == 0 for n in DS.CFG["PROPOSERS"]) and DS.stage_design() == 0
        assert DS.stage_eval("q7") == 0
        print("[g612 selftest] G：localization -> Executable command -> proposal -> design -> continuation（Including Interruption）-> Analysis")
        use_out(tmp / "G", A)
        pos, eps = load_pool_inputs()
        ep0 = eps[pos["trajectories"][0]["traj_id"]]
        p0 = pos["trajectories"][0]["positions"][0]
        a1 = DS.propose_messages(ep0, p0, 2)[1]["content"].split("Log of the failed run")[0]
        a2 = locate_messages(ep0)[1]["content"].split("Log of the failed run")[0]
        check(a1 == a2 and a1 == preamble(ep0), "The beginning of the positiontip with 6.9 The proposal is the same word for word")
        check("Step 0:" in locate_messages(ep0)[1]["content"] and "admissible" not in locate_messages(ep0)[1]["content"].lower()
              and "Possible actions" not in locate_messages(ep0)[1]["content"], "Positioning prompts：The whole log，Do not give executable lists or planners")
        STATE["calls"].clear()
        check(stage_prep() == 0, "prep")
        n_calls = dict(STATE["calls"])
        loc = load_locate()
        n_loc = sum(r["t"] is not None for r in loc.values())
        check(len(loc) == len(pos["trajectories"]) and 0 < n_loc < len(loc),
              f"localization：{len(loc)} records，Validity {n_loc}（the mock deliberately includes invalid outputs）")
        d = DS.jload(P("design"))
        props, steps = load_proposals(), DS.jload(P("steps"))["rows"]
        from alfw import mockenv
        bad = []
        for t in d["trajectories"]:
            k = t["traj_id"]
            e, p = eps[k], t["positions"][0]
            if p["t"] != loc[k]["t"]:
                bad.append(("t", k))
            env = mockenv.MockEnv()
            env.load(e["gamefile"])
            st = env.reset()
            for c in e["cmds"][:p["t"]]:
                st = env.step(c)
            if sorted(st["admissible"]) != sorted(steps[k]["admissible"]):
                bad.append(("adm", k))
            acts = [d["branches"][a]["action"] for a in p["alts"]]
            if acts != props[k]["actions"] or any(a not in st["admissible"] or a == e["cmds"][p["t"]] for a in acts):
                bad.append(("cand", k))
            if d["branches"][p["orig"]]["prefix"] != e["cmds"][:p["t"] + 1]:
                bad.append(("orig_prefix", k))
            for a in p["alts"]:
                if d["branches"][a]["prefix"] != e["cmds"][:p["t"]] + [d["branches"][a]["action"]]:
                    bad.append(("prefix", a))
        check(not bad and {t["traj_id"] for t in d["trajectories"]} == {k for k, r in loc.items() if r["t"] is not None},
              f"Every move youve been located：Original Action Branch + Valid proposal（In the order of the proposal），correct prefixes，Proposals implementable（Problem {bad[:3]}）")
        did = d["design_id"]
        STATE["calls"].clear()
        check(stage_prep() == 0 and not STATE["calls"] and DS.jload(P("design"))["design_id"] == did,
              f"Rerun prep：Do not call again API（First time {n_calls}），Design unchanged")
        P("design").unlink()
        stage_design()
        check(DS.jload(P("design"))["design_id"] == did, "Design recoverable（Redo gets the same design_id）")
        DS.STATE["stop_after"] = 4
        try:
            stage_eval()
        except KeyboardInterrupt:
            pass
        DS.STATE["stop_after"] = None
        check(stage_eval() == 0 and not missing(d), "resume to complete interrupted continuations")
        lines1 = sorted(P("evals").read_text().splitlines())
        P("evals").unlink()
        DS.STATE["n_backend"] = 0
        stage_eval()
        check(DS.STATE["n_backend"] == 1, f"One more run（Three batches）Construction only 1 One executor（Actual {DS.STATE['n_backend']}）")
        check(sorted(P("evals").read_text().splitlines()) == lines1, "Breaking up the continuations has the same result as a run-off")
        ev = runner.load_results(P("evals"))
        st = [b for b, x in d["branches"].items() if x["deterministic"] is None]
        ns = {bt: {len(ev[(c, bt, b)]["results"]) for b in st} for c, bt, n, _ in passes()}
        check(ns == {"GV": {3}, "GE": {10}, "GG": {1}}, f"Every Random Branch GV 3 runs、GE 10 runs、GG 1 runs（{ns}）")
        sig = lambda r, n: [tuple(r[f][i] for f in ("results", "steps", "invalid", "revisit")) for i in range(n)]
        diff = sum(sig(ev[("q7_T07", "GV", b)], 3) != sig(ev[("q7_T07", "GE", b)], 3) for b in st)
        check(diff > 0, f"GV and GE Seeds are independent（{diff} / {len(st)} Front of a branch 3 Second Different）")
        res = analyze()
        check(res is not None and P("report").exists(), "Analysis and reporting")
        ids = [t["traj_id"] for t in pos["trajectories"]]
        tr_by = {t["traj_id"]: t for t in d["trajectories"]}
        man = []
        for k in ids:
            if k not in tr_by:
                man.append(0.0)
                continue
            p = tr_by[k]["positions"][0]
            V = lambda b, bt, n: (ev[("q7_T07", bt, b)]["results"][:n] if d["branches"][b]["deterministic"] is None
                                  else [d["branches"][b]["deterministic"]] * n)
            sc = [(sum(V(a, "GV", 3)), -i, a) for i, a in enumerate(p["alts"]) if sum(V(a, "GV", 3)) >= 2]
            if not sc:
                man.append(0.0)
                continue
            a = max(sc)[2]
            man.append(float(np.mean(V(a, "GE", 10)) - np.mean(V(p["orig"], "GE", 10))))
        check(abs(res["levels"]["net"]["G"]["mean"] - float(np.mean(man))) < 1e-12,
              "G Net proceeds are consistent with the registration rules")
        pool = R.load_pool("7B", A)
        rr = {r["traj_id"]: r for r in R.run_core(pool, pool["d"], "q7", [], "net")}
        check(abs(res["levels"]["net"]["6.9 all matched"]["mean"] - np.mean([rr[k]["matched"] for k in ids])) < 1e-12,
              "6.9 Matches（all candidates）by ds612.run_core Recalculation")
        c = res["compare"]["net"]["Match(all) − G"]
        check(c["n"] == len(ids) and abs(c["mean"] - (np.mean([rr[k]["matched"] for k in ids]) - np.mean(man))) < 1e-12,
              "Matching：Match(all) − G Match all trajectories by trajectories")
        check(all(k in res["compare"]["net"] for k in ("Match(model) − G", "G − SL-flash k=1", "G − G_nv", "G − G_greedy")),
              "Full registration")
        check(res["levels"]["post"]["G"]["n"] == n_loc, f"The recovery success rate is only on the trajectories where the steps are（n = {n_loc}）")
        check(abs(_paired([1.0, None, 0.5], [0.5, 0.2, None])["mean"] - 0.5) < 1e-12 and
              _paired([1.0, None, 0.5], [0.5, 0.2, None])["n"] == 1, "The match is only calculated on a trajectories that is defined on both sides")
        rep = P("report").read_text(encoding="utf-8")
        check("Reading（G）" in rep and "Match(model) − G" in rep and "flash（localization）" in rep, "Reader  s report、Comparison with API Usage")
    finally:
        DS.CFG.clear()
        DS.CFG.update(saved_cfg)
        DS.STATE["mock"] = False
        PATHS.update(saved_paths)
        if os.environ.get("G_KEEP"):
            print(f"[g612 selftest] Keep temporary directory {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"[g612 selftest] {'all passed' if ok[0] else 'failures detected'}")
    return 0 if ok[0] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "prep", "eval", "all", "analyze", "selftest"])
    ap.add_argument("--out-dir", type=Path, default=None, help="default out_g612/")
    ap.add_argument("--pool-dir", type=Path, default=None, help="6.9 Outcome，default out_ds_alfw/")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    DS.STATE["mock"] = a.mock
    if a.cmd == "check":                     # Check only：Do not create output directory、No Log
        PATHS.update({k: Path(v).resolve() for k, v in (("out", a.out_dir), ("pool", a.pool_dir)) if v is not None})
        return stage_check()
    use_out(a.out_dir, a.pool_dir)
    log(f"===== g612 {a.cmd}  PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}  input {PATHS['pool']}  output {PATHS['out']} =====")
    if a.cmd == "prep":
        return stage_prep()
    if a.cmd == "eval":
        return stage_eval()
    if a.cmd == "all":
        rc = stage_prep()
        return rc if rc != 0 else stage_eval()
    if a.cmd == "analyze":
        return 0 if analyze() is not None else 1
    return 0


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != acfg.HASH_SEED:
        os.environ["PYTHONHASHSEED"] = acfg.HASH_SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.exit(main())
