"""ALFWorld：unscreened failed trajectories × Candidates for the model × 7B / 14B executor（For registration see analysis plan.md 6.9）。

Division of tasks between the two servers（14B proposals and continuations run on 14B On the server，7B Yes 7B On the server）：

  7B Servers   python ds_alfw.py --stage smoke              smoke test：run a minimal end-to-end workflow 7B This side（Save out_ds_alfw_smoke/，Do not enter analysis）
              python ds_alfw.py --stage prep7              collect → position → 7B proposal → flash proposal（API）→ Pack it up 14B Servers
  14B Servers  python ds_alfw.py --stage prep14 --bundle out_ds_alfw/bundle_for_14b.tar.gz
                                                          Unpack → 14B proposal → design（Freezing of candidatures、Branch-by-branch planning）→ 14B continuation（20 runs + greedy）
  7B Servers   （copy 14B Server out_ds_alfw/design.json copy back）
              python ds_alfw.py --stage run7               7B continuation（10 runs + greedy）
              （copy 14B Server evals_q14.jsonl、timing_q14.json、proposals_q14.jsonl copy back）
              python ds_alfw.py --stage analyze            -> out_ds_alfw/report.md / analysis.json
  Any machine    python ds_alfw.py --stage selftest           mock environment + Simulation model takes a full course，And test the known answers to the analysis

Separate phase：collect / positions / propose --proposer q7|q14|qflash / design / eval --executor q7|q14 / pack / unpack。
The stages that require different models to run in one sub-process（release GPU memory completely between stages）。All stages can be interrupted。
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np

import alfw_stage_b as SB
from alfw import acfg

SCRIPT = Path(__file__).resolve()
Q14_DIR = os.environ.get("DS_Q14", "/root/autodl-tmp/models/Qwen2.5-14B-Instruct")
CFG = {
    "SEED": 2028,              # sampling seed for fresh games（stage B use 2027）
    "N_COLLECT": 220,          # 7B Run! 1 episodes
    "N_TRAJ": 60,              # by rid Number of failed trajectories in sequence（unscreened；Only exclude those without an optional location）
    "N_POS": 3,                # early / middle / One position at a time of night
    "MIN_SLACK": 5,
    "PLAN_TIMEOUT": 60,
    "N_PROPOSE": 2,            # each proposer contributes at each position 2 items
    "N_RANDOM": 1,             # Add every position 1 A random executable command
    "PROPOSERS": ["q7", "q14", "qflash"],
    "N_EXEC": {"q7": 10, "q14": 20},
    "BATCH": "D",
}
STATE = {"mock": False, "out": acfg.ROOT / "out_ds_alfw", "smoke": False}
N_PERM = 50
N_BOOT = 2000


# ================================================================ Foundation
def log(msg):
    SB.log(msg)


def P(name, x=None):
    o = STATE["out"]
    return {"episodes": o / "episodes.jsonl", "episodes_meta": o / "episodes_meta.json", "positions": o / "positions.json",
            "proposals": o / f"proposals_{x}.jsonl", "propose_cache": o / "propose_cache" / f"{x}.jsonl",
            "design": o / "design.json", "evals": o / f"evals_{x}.jsonl", "timing": o / f"timing_{x}.json",
            "analysis": o / "analysis.json", "report": o / "report.md", "bundle": o / "bundle_for_14b.tar.gz",
            "smoke": o / "smoke_summary.json"}[name]


def use_out(out):
    STATE["out"] = Path(out).resolve()
    STATE["out"].mkdir(parents=True, exist_ok=True)
    SB.STATE["out"] = STATE["out"]
    SB.STATE["log_file"] = STATE["out"] / "ds.log"
    SB.STATE["mock"] = STATE["mock"]
    SB.STATE["mock_llm"] = False


def jdump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jl_read(path):
    out = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                          # Half the last line at the break
    return out


def fp(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def settings():
    s = dict(CFG)
    s.update(MAX_STEPS=acfg.MAX_STEPS, TEMPERATURE=acfg.TEMPERATURE, TOP_P=acfg.TOP_P, MODEL_DIR=str(acfg.MODEL_DIR),
             mock=STATE["mock"], smoke=STATE["smoke"])
    return s


def derive_seed(*parts):
    from alfw.agent import derive_seed as ds
    return ds(*parts)


# ---------------------------------------------------------------- Visible：The stage ends with a visible switchoff vLLM；The process of clearing residuals between phases
def release_vllm(obj):
    """Visible Close vLLM Engine（v1 of EngineCore is a separate process，Recovery only when the process exits is sometimes left behind and continues to take over）。"""
    llm = getattr(obj, "llm", None)
    if llm is None:
        return
    for fn in (lambda: llm.llm_engine.engine_core.shutdown(), lambda: llm.shutdown()):
        try:
            fn()
            break
        except Exception:
            pass
    try:
        del obj.llm
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _proc(pid):
    try:
        st = Path(f"/proc/{pid}/stat").read_text()
        rest = st[st.rfind(")") + 2:].split()
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore").strip()
        return int(rest[1]), int(rest[2]), cmd
    except Exception:
        return None


def leftover_procs():
    """In this process group、already detached from the parent process（The father process is not the process or the ancestors）of vLLM / Python Sons and grandchildren process：The residue left after the subprocess exited。"""
    if not Path("/proc").is_dir():
        return []
    me, pg = os.getpid(), os.getpgid(0)
    anc, p = set(), me
    while p > 1 and p not in anc:
        anc.add(p)
        info = _proc(p)
        if not info:
            break
        p = info[0]
    out = []
    for d_ in os.listdir("/proc"):
        if not d_.isdigit() or int(d_) in anc:
            continue
        info = _proc(int(d_))
        if not info:
            continue
        ppid, pgrp, cmd = info
        if pgrp == pg and ppid not in anc and re.search(r"VLLM|vllm|EngineCore|multiprocessing|python", cmd):
            out.append((int(d_), cmd[:100]))
    return out


def cleanup_leftovers():
    import signal
    left = leftover_procs()
    if not left:
        return
    log(f"[gpu] Stay the last step {len(left)} Process，End them：{left}")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, _ in left:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        for _ in range(20):
            if not leftover_procs():
                return
            time.sleep(0.5)


def gpu_free_frac():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        rows = [[float(x) for x in ln.split(",")] for ln in r.stdout.strip().splitlines() if ln.strip()]
    except Exception:
        return None
    if not rows:
        return None
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis and all(v.strip().isdigit() for v in vis.split(",")):
        keep = {int(v) for v in vis.split(",")}
        rows = [x for x in rows if int(x[0]) in keep] or rows
    return min((t - u) / t for _, u, t in rows)


def wait_gpu(stage, need=0.92, wait_s=90):
    """GPU Before the start of the phase：When the memory comes out,（vLLM requires idle resources ≥ Total 0.9）；if waiting fails, first clean up residual processes from this task，if still unavailable, report the process occupying the resource。"""
    for attempt in range(2):
        t0 = time.time()
        while True:
            f = gpu_free_frac()
            if f is None or f >= need:
                return True
            if time.time() - t0 > wait_s:
                break
            time.sleep(3)
        if attempt == 0:
            cleanup_leftovers()
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        smi = ""
    ps = []
    for d_ in os.listdir("/proc"):
        if d_.isdigit():
            info = _proc(int(d_))
            if info and re.search(r"VLLM|EngineCore|vllm", info[2]):
                ps.append((int(d_), info[0], info[2][:100]))
    log(f"[gpu] {stage} Clear only before starting {gpu_free_frac():.0%}（Yes ≥ {need:.0%}），This remains the case after the process of clearing the remaining parts of the mission。\n"
        f"occupying GPU memory vLLM Relevant processes（pid, Parent Process, command）：{ps}\n{smi}\n"
        f"after confirming no other task uses this GPU：pkill -f EngineCore（or kill Up there pid），Well repeat the same order（Hell run from here）")
    return False


# ================================================================ collect
def excluded_games(train):
    """stage A（A2 / A4）and phase B Collect used games（and alfw_stage_b.stage_collect The sample is exactly the same）。"""
    from alfw import games
    a = set(games.sample(train, acfg.A2_N_GAMES, acfg.A2_SEED)) | set(games.sample(train, acfg.A4_N_TRAIN, acfg.A4_SEED))
    cand_b = [g for g in train if g not in a and games.task_type_of(g) not in acfg.B_EXCLUDE_TYPES]
    b = set(games.sample(cand_b, acfg.B_N_COLLECT, acfg.B_SEED))
    return a | b


def stage_collect():
    from alfw import agent, games, rollout
    train = SB.split_games("train") if not STATE["mock"] else [f"{games.MOCK_PREFIX}{i}" for i in range(800)]
    excl = excluded_games(train)
    cand = [g for g in train if g not in excl and games.task_type_of(g) not in acfg.B_EXCLUDE_TYPES]
    if STATE["smoke"]:                     # A game that smokes without proper running
        formal = set(games.sample(cand, 220, 2028))
        cand = [g for g in cand if g not in formal]
    gs = games.sample(cand, CFG["N_COLLECT"], CFG["SEED"])
    specs = [{"rid": f"n{i:03d}", "gamefile": g, "seed": derive_seed("DS1", g)} for i, g in enumerate(gs)]
    key = fp({"specs": [[s["rid"], s["gamefile"], s["seed"]] for s in specs], "s": settings()})
    if P("episodes").exists() and P("episodes_meta").exists() and jload(P("episodes_meta")).get("key") == key:
        log(f"[collect] Reuse already available {len(specs)} episodes")
        return 0
    log(f"[collect] Optional Game {len(cand)} episodes（Exclusion phase A / B previously used {len(excl)} Office of the Secretary-General of the United Nations {acfg.B_EXCLUDE_TYPES}），sample {len(gs)} episodes")
    with SB.make_pool() as pl:
        be = agent.MockBackend() if STATE["mock"] else agent.VLLMBackend(model_dir=acfg.MODEL_DIR)
        eps, timing = rollout.run_episodes(be, pl, specs, temperature=acfg.TEMPERATURE, keep_text=False,
                                           want_key=False, log=log)
        release_vllm(be)
    with open(P("episodes"), "w", encoding="utf-8") as f:
        for e in sorted(eps, key=lambda x: x["rid"]):
            f.write(json.dumps({"rid": e["rid"], "gamefile": e["gamefile"], "seed": e["seed"], "task": e["task"],
                                "desc": e["desc"], "won": e["won"], "won_step": e["won_step"],
                                "task_type": games.task_type_of(e["gamefile"]),
                                "cmds": [s["cmd"] for s in e["steps"]], "obs": [s["obs"] for s in e["steps"]],
                                "kinds": [s["kind"] for s in e["steps"]]}, ensure_ascii=False) + "\n")
    jdump(P("episodes_meta"), {"key": key, "timing": timing})
    n_s = sum(e["won"] for e in eps)
    log(f"[collect] complete：Success {n_s}/{len(eps)}，failed {len(eps) - n_s}；Time {timing['wall'] / 60:.1f} minutes")
    return 0


def load_episodes():
    return {e["rid"]: e for e in jl_read(P("episodes"))}


# ================================================================ positions（unscreened：by rid Sequencing Failed Trails）
def stage_positions():
    if P("positions").exists():
        d = jload(P("positions"))
        if d.get("settings") == settings():
            log(f"[positions] Reuse already available {len(d['trajectories'])} trajectories")
            return 0
        sys.exit("[positions] The result is different from the current settings；Please change the output directory")
    eps = load_episodes()
    fails = [eps[k] for k in sorted(eps) if not eps[k]["won"]]
    log(f"[positions] failed {len(fails)} records；by rid Order before {CFG['N_TRAJ']} The bar has an optional location（Dont look at any repair signals）")
    trajs, excluded = [], []
    with SB.make_pool() as pl:
        i = 0
        while len(trajs) < CFG["N_TRAJ"] and i < len(fails):
            batch = fails[i:i + max(4, CFG["N_TRAJ"] - len(trajs))]     # Planning only the number of items required at a time（collect replacements for excluded cases）
            i += len(batch)
            items, keys = [], []
            for e in batch:
                for t in range(len(e["cmds"])):
                    items.append({"gamefile": e["gamefile"], "prefix": e["cmds"][:t], "timeout": CFG["PLAN_TIMEOUT"],
                                  "want_adm": True})
                    keys.append((e["rid"], t))
            pr = SB.probe_all(pl, items, "All trajectories")
            step = dict(zip(keys, pr))
            for e in batch:
                if len(trajs) >= CFG["N_TRAJ"]:
                    break
                T = len(e["cmds"])
                L = {t: (None if step[(e["rid"], t)]["timeout"] else len(step[(e["rid"], t)]["plan"])) for t in range(T)}
                elig = [t for t in range(T) if L[t] is not None and (acfg.MAX_STEPS - t) - L[t] >= CFG["MIN_SLACK"]]
                if not elig:
                    excluded.append({"traj_id": e["rid"], "reason": "no remaining slack ≥ 5 Location"})
                    continue
                prng = random.Random(derive_seed("DS2", e["rid"]))
                pos = SB._strata_positions(elig, CFG["N_POS"], prng)
                plist = []
                for t in pos:
                    r = step[(e["rid"], t)]
                    plist.append({"t": t, "orig_action": e["cmds"][t], "admissible": list(r["admissible"]),
                                  "planner": r["plan"][0] if r["plan"] else None, "L": L[t],
                                  "slack": (acfg.MAX_STEPS - t) - L[t]})
                trajs.append({"traj_id": e["rid"], "gamefile": e["gamefile"], "task_type": e["task_type"], "T": T,
                              "n_eligible": len(elig), "positions": plist})
    d = {"trajectories": trajs, "excluded": excluded, "n_fail": len(fails), "n_episodes": len(eps),
         "settings": settings()}
    d["positions_id"] = fp({"t": trajs, "s": d["settings"]})
    jdump(P("positions"), d)
    log(f"[positions] complete {d['positions_id'][:12]}：{len(trajs)} trajectories（Exclude {len(excluded)} There is no optional location for the bar），"
        f"position {sum(len(t['positions']) for t in trajs)} items")
    return 0


# ================================================================ propose
PROPOSE_SYSTEM = ("You are an expert debugger of AI agents. You will read the log of a failed run of an agent in a "
                  "text-based household environment and propose alternative actions for one step of the run.")
_PROP_RE = re.compile(r"^\s*\**\s*action\s*([12])\s*\**\s*[:：]\s*\**[ \t]*(.*)$", re.I | re.M)


def propose_messages(ep, pos, k):
    lines = [f"Step {i}: > {c or '(no action)'}\n{o}" for i, (c, o) in enumerate(zip(ep["cmds"], ep["obs"]))]
    t, orig = pos["t"], pos["orig_action"]
    adm = "\n".join(f"- {a}" for a in pos["admissible"])
    user = ("The agent acts in a text-based household environment. At every step it sees the task, the room description, "
            "its history of actions with the environment's feedback, and the list of actions possible right now, and it "
            "picks one of them. An action that is not possible does nothing (\"Nothing happens.\") but uses up a step. "
            f"The step budget is {acfg.MAX_STEPS} steps.\n\n"
            "The agent is a 7B language model that samples its actions at temperature 0.7.\n\n"
            f"Task: {ep['task']}\n\n{ep['desc']}\n\n"
            f"Log of the failed run ({len(lines)} steps; the agent ran out of steps):\n" + "\n".join(lines)
            + f"\n\nWe consider replacing the agent's action at step {t}, which was: \"{orig}\".\n"
              f"Possible actions at step {t}:\n{adm}\n\n"
              f"Propose the {k} actions from this list (other than \"{orig}\") that, if taken at step {t} instead, are most "
              "likely to let the SAME agent succeed when it continues on its own from there with the remaining steps. "
              "Copy each action exactly as written in the list. Think briefly, then end with exactly these lines:\n"
            + "\n".join(f"Action {i + 1}: <action>" for i in range(k)))
    return [{"role": "system", "content": PROPOSE_SYSTEM}, {"role": "user", "content": user}]


def parse_proposals(text, admissible, orig, k):
    """Take each number for the last time Action rows；It has to be enforceable（Old Syntax: put … in/on … Press executable move Processing）、Different from the original action、Dont repeat it。"""
    from alfw.agent import parse_action
    last = {}
    for m in _PROP_RE.finditer(text or ""):
        last[int(m.group(1))] = m.group(2)
    out, bad = [], []
    for i in range(1, k + 1):
        raw = last.get(i)
        if raw is None:
            bad.append({"i": i, "why": "missing"})
            continue
        r = parse_action("Action: " + raw, admissible)
        if r["kind"] not in ("valid", "alias"):
            bad.append({"i": i, "why": "not_admissible", "raw": raw[:80]})
        elif r["cmd"] == orig:
            bad.append({"i": i, "why": "same_as_original"})
        elif r["cmd"] in out:
            bad.append({"i": i, "why": "duplicate"})
        else:
            out.append(r["cmd"])
    return out, bad


class MockProposer:
    """Offline test：Pick action by torrent from the executable list in the prompts；Small Output Shortcut、Non-executable or repeated，To overwrite the parse branch。"""

    def __init__(self, name):
        self.name = name
        self.settings = {"backend": "mock", "name": name}

    def ask(self, msgs_list, on_done=None):
        out = []
        for m in msgs_list:
            text = m[-1]["content"]
            adm = re.findall(r"^- (.+)$", text.split("Possible actions at step", 1)[1], re.M)
            orig = re.search(r'which was: "(.*?)"', text).group(1)
            rng = random.Random(hashlib.sha1((self.name + text).encode()).hexdigest())
            pool_ = [a for a in adm if a != orig] or adm
            picks = [rng.choice(pool_) for _ in range(2)]
            u = rng.random()
            if u < 0.08:
                body = f"Action 1: {picks[0]}"
            elif u < 0.14:
                body = f"Action 1: {picks[0]}\nAction 2: fly to the moon"
            else:
                body = f"Action 1: {picks[0]}\nAction 2: {picks[1]}"
            out.append({"text": "Reasoning (mock).\n" + body, "finish_reason": "stop", "error": None})
            if on_done:
                on_done(len(out) - 1, out[-1])
        return out


def make_proposer(name):
    import judge_backends as JB
    if STATE["mock"]:
        return MockProposer(name)
    if name in ("q7", "q14"):
        tp = int(os.environ.get("LADDER_TP", "1"))
        if tp > 1:
            import functools
            import vllm
            vllm.LLM = functools.partial(vllm.LLM, tensor_parallel_size=tp)
        j = JB.VLLMJudge(acfg.MODEL_DIR if name == "q7" else Q14_DIR, context=16384, temperature=0.0, max_tokens=1024,
                         name=name)
    elif name == "qflash":
        j = JB.make_judge("api", model=os.environ.get("DS_QFLASH_MODEL", "qwen3.8-flash"), thinking=False)
        j.name = "qflash"
    else:
        raise ValueError(name)
    j.settings["role"] = "proposer"
    return j


def stage_propose(name):
    import judge_backends as JB
    pos = jload(P("positions"))
    eps = load_episodes()
    items = []
    for t in pos["trajectories"]:
        for p in t["positions"]:
            items.append({"traj_id": t["traj_id"], "t": p["t"], "pos": p,
                          "messages": propose_messages(eps[t["traj_id"]], p, CFG["N_PROPOSE"])})
    keys = [f"{it['traj_id']}|t{it['t']}" for it in items]
    old = jl_read(P("proposals", name))
    if [f"{r['traj_id']}|t{r['t']}" for r in old] == keys and all(
            r["positions_id"] == pos["positions_id"] and not r["error"] for r in old):
        log(f"[propose {name}] Reuse already available {len(old)} Organisation")
        return 0
    log(f"[propose {name}] {len(items)} Organisation")
    prop = make_proposer(name)
    outs = JB.ask_resumable(prop, [f"{it['traj_id']}|t{it['t']}" for it in items], [it["messages"] for it in items],
                            P("propose_cache", name))
    rows, n_valid, why = [], 0, {}
    for it, o in zip(items, outs):
        acts, bad = parse_proposals((o or {}).get("text"), it["pos"]["admissible"], it["pos"]["orig_action"],
                                    CFG["N_PROPOSE"])
        n_valid += len(acts)
        for b in bad:
            why[b["why"]] = why.get(b["why"], 0) + 1
        rows.append({"traj_id": it["traj_id"], "t": it["t"], "proposer": name, "actions": acts, "bad": bad,
                     "text": (o or {}).get("text"), "error": (o or {}).get("error"), "settings": prop.settings,
                     "positions_id": pos["positions_id"]})
    with open(P("proposals", name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_err = sum(1 for r in rows if r["error"])
    log(f"[propose {name}] Valid proposal {n_valid}/{len(items) * CFG['N_PROPOSE']}（{n_valid / max(1, len(items) * CFG['N_PROPOSE']):.0%}）；"
        f"reason for invalidity {why}；Error calling {n_err}")
    release_vllm(prop)
    return 1 if n_err else 0


def load_proposals(name):
    return {(r["traj_id"], r["t"]): r for r in jl_read(P("proposals", name))}


# ================================================================ design（Freezing of candidatures，Branch-by-branch planning）
def stage_design(allow_missing=False):
    pos = jload(P("positions"))
    props = {}
    for n in CFG["PROPOSERS"]:
        if P("proposals", n).exists():
            props[n] = load_proposals(n)
        elif not allow_missing:
            sys.exit(f"[design] Missing proposals_{n}.jsonl")
    for n, pr in props.items():
        bad = {r["positions_id"] for r in pr.values()} - {pos["positions_id"]}
        if bad:
            sys.exit(f"[design] proposals_{n}.jsonl From another location file；Please make a new proposal")
    if P("design").exists():
        d = jload(P("design"))
        if d.get("positions_id") == pos["positions_id"] and d.get("proposers") == sorted(props) \
                and d.get("settings") == settings():
            log(f"[design] Reuse already designed {d['design_id'][:12]}")
            return 0
        sys.exit("[design] The design is different from the current input；Please change the output directory")
    eps = load_episodes()
    trajs, branches = [], {}
    for tr in pos["trajectories"]:
        e = eps[tr["traj_id"]]
        plist = []
        for p in tr["positions"]:
            t, orig, adm = p["t"], p["orig_action"], p["admissible"]
            cands = {}                                   # action -> Metadata（Insert Order Fixed：planner、each proposer、random）

            def add(a, src, rank=None):
                c = cands.setdefault(a, {"proposers": [], "planner": False, "random": False, "flash_rank": None})
                if src == "planner":
                    c["planner"] = True
                elif src == "random":
                    c["random"] = True
                else:
                    if src not in c["proposers"]:
                        c["proposers"].append(src)
                    if src == "qflash":
                        c["flash_rank"] = rank
            if p["planner"] is not None and p["planner"] != orig and p["planner"] in adm:
                add(p["planner"], "planner")
            for n in CFG["PROPOSERS"]:
                r = props.get(n, {}).get((tr["traj_id"], t))
                for i, a in enumerate(r["actions"] if r else []):
                    add(a, n, i)
            rng = random.Random(derive_seed("DS3", tr["traj_id"], t))
            pool_ = sorted(a for a in adm if a != orig and a not in cands)
            for a in rng.sample(pool_, min(CFG["N_RANDOM"], len(pool_))):
                add(a, "random")
            oid = f"{tr['traj_id']}|t{t}|orig"
            branches[oid] = {"branch_id": oid, "traj_id": tr["traj_id"], "t": t, "action": "orig:" + orig,
                             "gamefile": tr["gamefile"], "prefix": e["cmds"][:t + 1], "expert": False}
            ids = []
            for a, meta in cands.items():
                bid = f"{tr['traj_id']}|t{t}|{a}"
                branches[bid] = {"branch_id": bid, "traj_id": tr["traj_id"], "t": t, "action": a,
                                 "gamefile": tr["gamefile"], "prefix": e["cmds"][:t] + [a], "expert": meta["planner"],
                                 **meta}
                ids.append(bid)
            plist.append({"t": t, "orig": oid, "alts": ids, "L": p["L"], "slack": p["slack"]})
        trajs.append({"traj_id": tr["traj_id"], "gamefile": tr["gamefile"], "task_type": tr["task_type"],
                      "T": tr["T"], "positions": plist})
    log(f"[design] {len(trajs)} trajectories、{len(branches)} branches；One by one（oracle Feasibility and residual）")
    bids = sorted(branches)
    with SB.make_pool() as pl:
        pr = SB.probe_all(pl, [{"gamefile": branches[b]["gamefile"], "prefix": branches[b]["prefix"],
                                "timeout": CFG["PLAN_TIMEOUT"]} for b in bids], "branch")
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
            br.update(deterministic=None, oracle_L=Lb, oracle_F=1.0 if Lb <= rem else 0.0, oracle_slack=float(rem - Lb),
                      oracle_timeout=False)
    d = {"kind": "ds_alfw", "trajectories": trajs, "branches": branches, "positions_id": pos["positions_id"],
         "proposers": sorted(props), "settings": settings(), "n_branch_timeout": n_to}
    d["design_id"] = fp({"t": trajs, "b": {k: {kk: v.get(kk) for kk in ("prefix", "oracle_F", "oracle_slack", "proposers", "planner", "random")}
                                           for k, v in branches.items()}, "s": d["settings"]})
    jdump(P("design"), d)
    nst = sum(1 for b in branches.values() if b["deterministic"] is None)
    na = [len(p["alts"]) for t in trajs for p in t["positions"]]
    log(f"[design] complete {d['design_id'][:12]}：Random Branch {nst} items（Planning timeout {n_to}）；Candidature for each position {min(na)}–{max(na)} items，"
        f"Average {np.mean(na):.1f}")
    return 0


# ================================================================ eval（The executor continues）
def make_exec_backend(name):
    from alfw import agent
    STATE["n_backend"] = STATE.get("n_backend", 0) + 1          # Self-check：create only one instance per continuation process
    if STATE["mock"]:
        return agent.MockBackend()
    tp = int(os.environ.get("LADDER_TP", "1"))
    if tp > 1:
        import functools
        import vllm
        vllm.LLM = functools.partial(vllm.LLM, tensor_parallel_size=tp)
    return agent.VLLMBackend(model_dir={"q7": acfg.MODEL_DIR, "q14": Q14_DIR}[name])


def _add_timing(name, rec):
    old = jload(P("timing", name)) if P("timing", name).exists() else {"runs": []}
    old["runs"].append(rec)
    jdump(P("timing", name), old)


def recover_timing(name):
    """When no sampling conditions are recorded in the time files，from ds.log in“Stay N × n runs …… complete：X minutes，S steps”restore（For cost conversion only）。
    2026-10-24 of 7B the formal continuation run crashed when starting the greedy condition，The timing of the sampling conditions didnt come in time to write it。"""
    cond = f"{name}_T07/{CFG['BATCH']}"
    if P("timing", name).exists() and any(r["cond"] == cond for r in jload(P("timing", name))["runs"]):
        return
    lf = STATE["out"] / "ds.log"
    if not lf.exists():
        return
    pending, n_rec = None, 0
    for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"\[" + re.escape(cond) + r"\] Random Branch \d+ items，Completed \d+，Stay (\d+) × (\d+) runs", line)
        if m:
            pending = int(m.group(1)) * int(m.group(2))
            continue
        m = re.search(r"\[" + re.escape(cond) + r"\] complete：([\d.]+) minutes，(\d+) steps", line)
        if m and pending:
            _add_timing(name, {"cond": cond, "wall": float(m.group(1)) * 60, "policy_steps": int(m.group(2)),
                               "rollouts": pending, "gpu": gpu_info(), "tp": int(os.environ.get("LADDER_TP", "1")),
                               "time": line[1:15], "recovered_from_log": True})
            pending, n_rec = None, n_rec + 1
    if n_rec:
        log(f"[eval {name}] from ds.log restored {n_rec} Timing records of sampling conditions")


def stage_eval(name, n=None):
    d = jload(P("design"))
    pos = jload(P("positions")) if P("positions").exists() else None
    if pos is not None and pos["positions_id"] != d["positions_id"]:
        sys.exit("[eval] design.json and the local machines positions.json different（Not the same trajectories）；Please check the document that was copied")
    n = n or CFG["N_EXEC"][name]
    holder = {}

    def backend():                                  # create lazily on first use，Reuse Later（One for both vLLM Engine）
        if "b" not in holder:
            holder["b"] = make_exec_backend(name)
        return holder["b"]
    ctx = {"backend": backend, "stop_after": STATE.get("stop_after")}
    recover_timing(name)
    import runner
    try:
        with SB.make_pool() as pl:
            ctx["pool"] = pl
            for cond, nn, temp in ((f"{name}_T07", n, acfg.TEMPERATURE), (f"{name}_greedy", 1, 0.0)):
                have = runner.load_results(P("evals", name)) if P("evals", name).exists() else {}
                todo = sum(1 for b in stochastic(d) if (cond, CFG["BATCH"], b) not in have)
                tm = SB.run_branches(ctx, d, cond, CFG["BATCH"], nn, P("evals", name), temp)
                if tm:                              # record immediately after each condition finishes，retain results even if a later stage fails
                    _add_timing(name, {"cond": f"{cond}/{CFG['BATCH']}", "wall": tm["wall"],
                                       "policy_steps": tm["policy_steps"], "rollouts": todo * nn, "gpu": gpu_info(),
                                       "tp": int(os.environ.get("LADDER_TP", "1")),
                                       "time": time.strftime("%Y-%m-%d %H:%M")})
    finally:
        release_vllm(holder.get("b"))
    left = missing(d, name, n)
    log(f"[eval {name}] " + ("All branches completed" if not left else f"also {len(left)} items（Condition, branch）incomplete，The same order will be replaced by the same order"))
    return 0 if not left else 1


def gpu_info():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=10)
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        names = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        return {"names": names, "visible": vis}
    except Exception:
        return None


def stochastic(d):
    return sorted(b for b, x in d["branches"].items() if x["deterministic"] is None)


def missing(d, name, n=None):
    import runner
    have = runner.load_results(P("evals", name)) if P("evals", name).exists() else {}
    out = []
    for c in (f"{name}_T07", f"{name}_greedy"):
        for b in stochastic(d):
            if (c, CFG["BATCH"], b) not in have:
                out.append((c, b))
    return out


# ================================================================ Pack / Unpack（7B Servers -> 14B Servers）
BUNDLE_FILES = ["episodes.jsonl", "episodes_meta.json", "positions.json", "proposals_q7.jsonl", "proposals_qflash.jsonl"]
# Game files recorded in absolute path：Two servers ALFWORLD_DATA It has to be the same（Both by default /root/autodl-tmp/alfworld_data）


def stage_pack():
    pos = jload(P("positions"))
    games = sorted({t["gamefile"] for t in pos["trajectories"]})
    root = acfg.DATA
    n = 0
    with tarfile.open(P("bundle"), "w:gz") as tf:
        for g in games:
            if not Path(g).is_file():               # mock environment（mock://）No file
                continue
            for p in sorted(Path(g).parent.iterdir()):
                tf.add(p, arcname="alfworld_data/" + str(p.relative_to(root)))
                n += 1
        if (root / "logic").exists():
            tf.add(root / "logic", arcname="alfworld_data/logic")
        info = STATE["out"] / "bundle_info.json"
        jdump(info, {"data_root": str(root), "positions_id": pos["positions_id"]})
        tf.add(info, arcname="out_ds_alfw/bundle_info.json")
        for f in BUNDLE_FILES:
            if (STATE["out"] / f).exists():
                tf.add(STATE["out"] / f, arcname="out_ds_alfw/" + f)
    log(f"[pack] {len(games)} Game（{n} File）and {len(BUNDLE_FILES)} Outcome document -> {P('bundle')}"
        f"（{P('bundle').stat().st_size / 1e6:.1f} MB）。copy to 14B Post-Service Run：python ds_alfw.py --stage prep14 --bundle <path>")
    return 0


def stage_unpack(bundle):
    bundle = Path(bundle)
    n_g = n_f = 0
    with tarfile.open(bundle, "r:gz") as tf:
        for m in tf.getmembers():
            parts = Path(m.name).parts
            if not parts or ".." in parts or Path(m.name).is_absolute():
                sys.exit(f"[unpack] An unsafe path：{m.name}")
            if parts[0] == "alfworld_data":
                dest = acfg.DATA.joinpath(*parts[1:]) if len(parts) > 1 else acfg.DATA
                n_g += m.isfile()
            elif parts[0] == "out_ds_alfw":
                dest = STATE["out"].joinpath(*parts[1:])
                n_f += m.isfile()
            else:
                continue
            if m.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src:
                    dest.write_bytes(src.read())
    info = STATE["out"] / "bundle_info.json"
    if info.exists() and jload(info)["data_root"] != str(acfg.DATA):
        sys.exit(f"[unpack] Packing machines ALFWORLD_DATA yes {jload(info)['data_root']}，this machine is {acfg.DATA}；"
                 f"Game recorded on absolute path，Please be on board export ALFWORLD_DATA={jload(info)['data_root']} Unpack it")
    pos = jload(P("positions"))
    miss = [t["gamefile"] for t in pos["trajectories"] if not t["gamefile"].startswith("mock://")
            and not Path(t["gamefile"]).is_file()]
    if miss:
        sys.exit(f"[unpack] present {len(miss)} A game file does not exist，for example {miss[0]}")
    log(f"[unpack] Game Data {n_g} File -> {acfg.DATA}；Outcome document {n_f} items -> {STATE['out']}")
    return 0


# ================================================================ Analysis
def _traj_arrays(d, t, ev, name, n):
    """A trajectories：candidate (alt, orig, Location Serial Number) Columns and subscripts；A determination value for each branch；executor name The sample matrix on random branches and the result of greedy。"""
    cands = [(a, p["orig"], i) for i, p in enumerate(t["positions"]) for a in p["alts"]]
    bids = sorted({b for c in cands for b in c[:2]})
    col = {b: j for j, b in enumerate(bids)}
    br = d["branches"]
    det = np.array([np.nan if br[b]["deterministic"] is None else float(br[b]["deterministic"]) for b in bids])
    rand = np.flatnonzero(np.isnan(det))
    S = np.array([ev[(f"{name}_T07", CFG["BATCH"], bids[j])]["results"][:n] for j in rand], float).reshape(len(rand), n)
    G = det.copy()
    for j in rand:
        G[j] = float(ev[(f"{name}_greedy", CFG["BATCH"], bids[j])]["results"][0])
    return {"cands": cands, "bids": bids, "col": col, "det": det, "rand": rand, "S": S, "G": G,
            "ia": np.array([col[a] for a, _, _ in cands]), "io": np.array([col[o] for _, o, _ in cands])}


def _full(det, rand, vals):
    v = np.nan_to_num(det).copy()
    v[rand] = vals
    return v


def _gain(v, ia, io, obj):
    return v[..., ia] - (v[..., io] if obj == "net" else 0.0)


def _val(mask, g):
    return float((mask * g).sum() / mask.sum())


def _best(score):
    score = np.asarray(score, float)
    return score >= np.nanmax(score) - 1e-12


def analyze_core(trajs7, trajs14, branch_meta, obj="net", seed=0, n_perm=N_PERM, ks7=(1, 2, 3, 5),
                 ks14=(1, 2, 3, 5, 10, 20)):
    """One-line statistics per trajectories（7B Value of deployment on board）。trajs7 / trajs14：_traj_arrays Outcome（Same group of candidates）；branch_meta(trajectories Serial Numbers, branch) -> Metadata。"""
    rng = np.random.default_rng(seed)
    rows = []
    for ti, A in enumerate(trajs7):
        B = trajs14[ti] if trajs14 is not None else None
        ia, io, det, rand = A["ia"], A["io"], A["det"], A["rand"]
        n7 = A["S"].shape[1]
        h = n7 // 2
        full7 = _full(det, rand, A["S"].mean(axis=1) if len(rand) else [])
        g7 = _gain(full7, ia, io, obj)
        meta = [branch_meta(ti, a) for a, _, _ in A["cands"]]
        r = {}
        # ---- 7B Myself k Subselection（Crossing）：The first half of each rehearsing（before k runs）、The second half，Reconcile
        perms = [np.argsort(rng.random(A["S"].shape), axis=1) for _ in range(n_perm)]

        def crossfit(k, sub=None):
            vals = []
            for pm in perms:
                Z = np.take_along_axis(A["S"], pm, axis=1)
                for a_, b_ in ((slice(0, h), slice(h, n7)), (slice(h, n7), slice(0, h))):
                    sel = _full(det, rand, Z[:, a_][:, :k].mean(axis=1))
                    evv = _full(det, rand, Z[:, b_].mean(axis=1))
                    sc = _gain(sel, ia, io, obj)
                    if sub is not None:
                        sc = np.where(sub, sc, -np.inf)
                    vals.append(_val(_best(sc), _gain(evv, ia, io, obj)))
            return float(np.mean(vals))
        for k in ks7:
            if k <= h:
                r[f"7B_k{k}"] = crossfit(k)
        r["matched"] = r[f"7B_k{h}"] if f"7B_k{h}" in r else crossfit(h)
        # ---- not used 7B Selection of Samples：at 7B Evaluation on all samples
        G7 = _gain(A["G"], ia, io, obj)
        r["greedy"] = _val(_best(G7), g7)
        planner = np.array([m["planner"] for m in meta])
        pos_i = np.array([c[2] for c in A["cands"]])
        if planner.any():
            first = pos_i[planner].min()
            r["planner_earliest"] = _val(planner & (pos_i == first), g7)
        else:
            r["planner_earliest"] = float(g7.mean())
        F = np.array([m["oracle_F"] for m in meta])
        SL = np.array([m["oracle_slack"] for m in meta])
        Fo = np.array([branch_meta(ti, o)["oracle_F"] for _, o, _ in A["cands"]])
        SLo = np.array([branch_meta(ti, o)["oracle_slack"] for _, o, _ in A["cands"]])
        if obj == "net":
            r["oracle_slack"] = _val(_best((F - Fo) * 1000 + (SL - SLo)), g7)
        else:
            r["oracle_slack"] = _val(_best(F * 1000 + SL), g7)
        r["slack_abs"] = _val(_best(F * 1000 + SL), g7)
        r["random"] = float(g7.mean())
        r["best_in_sample"] = float(g7.max())
        # ---- 14B Selection（k runs，Rehearsal），at 7B Evaluation on all samples
        if B is not None:
            n14 = B["S"].shape[1]
            full14 = _full(B["det"], B["rand"], B["S"].mean(axis=1) if len(B["rand"]) else [])
            g14 = _gain(full14, ia, io, obj)
            for k in ks14:
                if k <= n14:
                    vals = []
                    for _ in range(n_perm if k < n14 else 1):
                        pm = np.argsort(rng.random(B["S"].shape), axis=1)
                        Z = np.take_along_axis(B["S"], pm, axis=1)[:, :k].mean(axis=1)
                        vals.append(_val(_best(_gain(_full(B["det"], B["rand"], Z), ia, io, obj)), g7))
                    r[f"14B_k{k}"] = float(np.mean(vals))
            # Same thing h runs：7B Crossing vs 14B h runs（all in 7B Up-to-date evaluation）
            r["eq_7B_minus_14B"] = r["matched"] - r[f"14B_k{h}"]
            r["greedy14"] = _val(_best(_gain(B["G"], ia, io, obj)), g7)
            r["_g14"] = g14
        # ---- Methodology：Screening for follow-up
        alts_by_pos = {}
        for c, m in zip(range(len(meta)), meta):
            alts_by_pos.setdefault(pos_i[c], []).append(c)
        surv_pl, surv_fl = np.zeros(len(meta), bool), np.zeros(len(meta), bool)
        for i, cs in alts_by_pos.items():
            # Leave every position oracle Feasibility × 1000 + The highest remaining 1 items；Column by Branch Number（Fixed、independent of outcomes）
            surv_pl[min(cs, key=lambda c: (-(F[c] * 1000 + SL[c]), A["cands"][c][0]))] = True
            fl = [c for c in cs if meta[c].get("flash_rank") is not None]
            if fl:
                surv_fl[min(fl, key=lambda c: meta[c]["flash_rank"])] = True
        nb_rand = lambda cset: len({A["col"][x] for c in cset for x in A["cands"][c][:2]} & set(A["rand"].tolist()))
        r["budget_all"] = nb_rand(range(len(meta)))
        for tag, sub in (("SLpl", surv_pl), ("SLfl", surv_fl)):
            r[f"budget_{tag}"] = nb_rand(np.flatnonzero(sub))
            for k in ks7:
                if k <= h:
                    r[f"{tag}_k{k}"] = crossfit(k, sub) if sub.any() else (0.0 if obj == "net" else float(full7[io].mean()))
            best = _best(g7)
            r[f"recall_{tag}"] = float((best & sub).sum() / best.sum())
        # ---- proposer × executor
        for p in ("q7", "q14", "qflash", "planner", "random"):
            m_ = np.array([(p in m["proposers"]) if p.startswith("q") else bool(m[p]) for m in meta])
            r[f"prop7_{p}"] = float(g7[m_].mean()) if m_.any() else np.nan
            if B is not None:
                r[f"prop14_{p}"] = float(r["_g14"][m_].mean()) if m_.any() else np.nan
        r["best_is"] = {p: float(sum((p in meta[c]["proposers"]) if p.startswith("q") else bool(meta[c][p])
                                     for c in np.flatnonzero(_best(g7))) / _best(g7).sum())
                        for p in ("q7", "q14", "qflash", "planner", "random")}
        r.pop("_g14", None)
        rows.append(r)
    return rows


def boot(x, seed=0, n=N_BOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "ci": [float("nan")] * 2, "p": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))],
            "p": float((np.sum(m <= 0) + 1) / (n + 1)), "n": int(len(x))}


def stage_analyze():
    import runner
    d = jload(P("design"))
    have = {n: runner.load_results(P("evals", n)) for n in CFG["N_EXEC"] if P("evals", n).exists()}
    if "q7" not in have or missing(d, "q7"):
        log("[analyze] 7B continuations are incomplete（evals_q7.jsonl）")
        return 1
    use14 = "q14" in have and not missing(d, "q14")
    if not use14:
        log("[analyze] 14B or follow-up is missing or incomplete：Reporting only 7B Part（P2 Skip with Budget Curve）")
    n7 = min(len(r["results"]) for (c, _, _), r in have["q7"].items() if c == "q7_T07")
    T7 = [_traj_arrays(d, t, have["q7"], "q7", n7) for t in d["trajectories"]]
    T14 = None
    if use14:
        n14 = min(len(r["results"]) for (c, _, _), r in have["q14"].items() if c == "q14_T07")
        T14 = [_traj_arrays(d, t, have["q14"], "q14", n14) for t in d["trajectories"]]
    meta = lambda ti, b: d["branches"][b]
    res = {"design_id": d["design_id"], "n_traj": len(T7)}
    for obj in ("net", "post"):
        rows = analyze_core(T7, T14, lambda ti, b: meta(ti, b), obj=obj)
        res[obj] = {"rows": rows}
    # ---- Main Test & Read（net gain）
    R = res["net"]["rows"]
    col = lambda k: [r.get(k, np.nan) for r in R]
    diff = lambda a, b: [r[a] - r[b] for r in R]
    out = {"PC": boot(diff("matched", "random")), "P1": boot(diff("matched", "greedy")),
           "S1": boot(diff("matched", "planner_earliest")), "S2": boot(diff("matched", "oracle_slack")),
           "slack_abs": boot(diff("matched", "slack_abs"))}
    if use14:
        out["P2"] = boot(col("eq_7B_minus_14B"))
    prim = [h for h in ("P1", "P2") if h in out]
    order = sorted(prim, key=lambda h: out[h]["p"])
    ok = True
    for i, h in enumerate(order):
        thr = 0.05 / (len(order) - i)
        out[h]["holm_threshold"] = thr
        out[h]["holm_reject"] = bool(ok and out[h]["p"] <= thr)
        ok = ok and out[h]["holm_reject"]
    out["PC"]["pass"] = out["PC"]["ci"][0] > 0.10
    res["tests"] = out
    # ---- Budget curve（Two targets）
    curves = {}
    for obj in ("net", "post"):
        RR = res[obj]["rows"]
        c = {f"7B_k{k}": boot([r[f"7B_k{k}"] for r in RR if f"7B_k{k}" in r]) for k in (1, 2, 3, 5)}
        if use14:
            c.update({f"14B_k{k}": boot([r[f"14B_k{k}"] for r in RR if f"14B_k{k}" in r]) for k in (1, 2, 3, 5, 10, 20)})
            c["eq_7B_minus_14B"] = boot([r["eq_7B_minus_14B"] for r in RR])
        c["greedy"] = boot([r["greedy"] for r in RR])
        c["matched"] = boot([r["matched"] for r in RR])
        curves[obj] = c
    res["curves"] = curves
    # ---- Implementation time-consuming（Number of seconds per line running）
    sec = {}
    for n in CFG["N_EXEC"]:
        if P("timing", n).exists():
            runs = [r for r in jload(P("timing", n))["runs"] if r["cond"] == f"{n}_T07/{CFG['BATCH']}" and r["rollouts"]]
            if runs:
                sec[n] = sum(r["wall"] for r in runs) / sum(r["rollouts"] for r in runs)
                sec[n + "_gpu"] = runs[-1]["gpu"]
                sec[n + "_tp"] = runs[-1]["tp"]
    res["sec_per_rollout"] = sec
    # ---- Methodology
    meth = {}
    for tag in ("SLpl", "SLfl"):
        for k in (1, 2, 3, 5):
            if all(f"{tag}_k{k}" in r for r in R):
                meth[f"{tag}_k{k}"] = {"loss": boot(diff("matched", f"{tag}_k{k}")),
                                       "budget": float(np.mean([k * r[f"budget_{tag}"] for r in R]))}
        meth[f"recall_{tag}"] = float(np.mean(col(f"recall_{tag}")))
    for k in (1, 2, 3, 5):
        if all(f"7B_k{k}" in r for r in R):
            meth[f"all_k{k}"] = {"loss": boot(diff("matched", f"7B_k{k}")),
                                 "budget": float(np.mean([k * r["budget_all"] for r in R]))}
    meth["greedy"] = {"loss": out["P1"], "budget": float(np.mean([r["budget_all"] for r in R]))}
    res["method"] = meth
    # ---- proposer × executor
    prop = {}
    for p in ("q7", "q14", "qflash", "planner", "random"):
        prop[p] = {"7B": boot(col(f"prop7_{p}"))}
        if use14:
            prop[p]["14B"] = boot(col(f"prop14_{p}"))
    if use14:
        inter = [(r["prop7_q7"] - r["prop7_q14"]) - (r["prop14_q7"] - r["prop14_q14"]) for r in R]
        prop["interaction_q7_q14"] = boot(inter)
    prop["best_is"] = {p: float(np.mean([r["best_is"][p] for r in R])) for p in ("q7", "q14", "qflash", "planner", "random")}
    res["proposers"] = prop
    res["describe"] = describe(d, have)
    res["describe"]["frac_traj_best_gain_gt_0.10"] = float(np.mean([r["best_in_sample"] > 0.10 for r in R]))
    for obj in ("net", "post"):
        for r in res[obj]["rows"]:
            r.pop("best_is", None)
    jdump(P("analysis"), res)
    P("report").write_text(report(res, use14), encoding="utf-8")
    log(f"[analyze] -> {P('report')}")
    return 0


def describe(d, have):
    pos = jload(P("positions")) if P("positions").exists() else {}
    out = {"n_episodes": pos.get("n_episodes"), "n_fail": pos.get("n_fail"), "excluded": len(pos.get("excluded", []))}
    eps = load_episodes() if P("episodes").exists() else {}
    if eps:
        out["success_rate"] = float(np.mean([e["won"] for e in eps.values()]))
    na = [len(p["alts"]) for t in d["trajectories"] for p in t["positions"]]
    out["cands_per_position"] = [int(min(na)), float(np.mean(na)), int(max(na))]
    val = {}
    for n in CFG["PROPOSERS"]:
        if P("proposals", n).exists():
            rows = jl_read(P("proposals", n))
            val[n] = float(np.mean([len(r["actions"]) / CFG["N_PROPOSE"] for r in rows]))
    out["proposal_valid_rate"] = val
    for n, ev in have.items():
        s = [np.mean(v["results"]) for (c, _, _), v in ev.items() if c == f"{n}_T07"]
        out[f"branch_success_{n}"] = float(np.mean(s)) if s else None
    return out


def report(res, use14):
    f = lambda b: f"{b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]"
    t = res["tests"]
    L = ["# ALFWorld Do Not Filter General × Candidates for the model × 7B / 14B（For registration see analysis plan.md 6.9）", ""]
    if STATE["mock"] or STATE["smoke"]:
        L += ["> **Simulate or smoke run**：The numbers dont mean anything。", ""]
    ds = res["describe"]
    L += [f"design {res['design_id'][:12]}；{res['n_traj']} trajectories。description：{json.dumps(ds, ensure_ascii=False)}", "",
          "Value = Fixing selected in deployment 7B Difference between net gains above（90% trajectory bootstrap interval）；Match = 7B Myself 5 Subselection、Other 5 Commentary（Crossing）。", "",
          "| Read | Contrast | estimate [90%] | decision |", "|---|---|---|---|",
          f"| positive control | Match − random | {f(t['PC'])} | {'passed' if t['PC']['pass'] else 'failed：insufficient discriminative signal'} |",
          f"| P1（Holm） | Match − 7B greedy | {f(t['P1'])} | {'confirmed' if t['P1'].get('holm_reject') else 'not confirmed'}（p = {t['P1']['p']:.4f}） |"]
    if "P2" in t:
        L.append(f"| P2（Holm） | Same thing 5 runs：7B select using the deployed executor − by 14B Selection | {f(t['P2'])} | "
                 f"{'confirmed' if t['P2'].get('holm_reject') else 'not confirmed'}（p = {t['P2']['p']:.4f}） |")
    L += [f"| S1 | Match − Planner actions are in the earliest position | {f(t['S1'])} | {'confirmed' if t['S1']['ci'][0] > 0 else 'not confirmed'} |",
          f"| S2 | Match − oracle + slack（Definition of registration） | {f(t['S2'])} | {'confirmed' if t['S2']['ci'][0] > 0 else 'not confirmed'} |",
          f"| description | Match − Select in absolute remainder | {f(t['slack_abs'])} | — |", ""]
    sec = res.get("sec_per_rollout", {})
    # Cost converted to CCS：Every second of a wall clock running × Number of parallel cards（The two servers have different visual cards that are just similar，model identifiers are listed below）
    ratio = (sec["q14"] * sec.get("q14_tp", 1)) / (sec["q7"] * sec.get("q7_tp", 1)) if "q7" in sec and "q14" in sec else None
    if ratio:
        L += [f"Surviving costs：7B {sec['q7']:.2f} seconds/records × {sec.get('q7_tp', 1)} Cut（{sec.get('q7_gpu')}）；"
              f"14B {sec['q14']:.2f} seconds/records × {sec.get('q14_tp', 1)} Cut（{sec.get('q14_gpu')}）；margin {ratio:.2f}", ""]
    for obj, lab in (("net", "net gain"), ("post", "Success rate after restoration")):
        c = res["curves"][obj]
        L += [f"**Budget curve（{lab}；_Other Organiser 7B Value）**" +
              (f"；14B It takes about a minute to keep running 7B of {ratio:.1f} Multiply" if ratio else ""), "",
              "| Selection | Value [90%] |", "|---|---|"]
        for k in (1, 2, 3, 5):
            if f"7B_k{k}" in c and c[f"7B_k{k}"]["n"]:
                L.append(f"| 7B {k} runs | {f(c[f'7B_k{k}'])} |")
        for k in (1, 2, 3, 5, 10, 20):
            if f"14B_k{k}" in c and c[f"14B_k{k}"]["n"]:
                L.append(f"| 14B {k} runs" + (f"（≈ 7B {k * ratio:.0f} Second time）" if ratio else "") + f" | {f(c[f'14B_k{k}'])} |")
        L.append(f"| 7B greedy（1 runs / branch） | {f(c['greedy'])} |")
        if "eq_7B_minus_14B" in c:
            L.append(f"| Poor：7B 5 runs − 14B 5 runs | {f(c['eq_7B_minus_14B'])} |")
        L.append("")
    m = res["method"]
    L += ["**Methodology（Freezing SL）**：Relatively matched loss and running budget for each trajectory", "", "| Modalities | budget | loss [90%] |", "|---|---:|---|"]
    for key in sorted(k for k in m if isinstance(m[k], dict)):
        L.append(f"| {key} | {m[key]['budget']:.0f} | {f(m[key]['loss'])} |")
    L += [f"", f"Screening recall rate（7B The best candidate in all samples is screened）：planner {m.get('recall_SLpl', float('nan')):.2f}；"
              f"flash {m.get('recall_SLfl', float('nan')):.2f}", ""]
    pr = res["proposers"]
    L += ["**proposer × executor**（Average net proceeds per candidate；exploratory）", "", "| proposer | at 7B on | at 14B on | yes 7B Proportion of best candidates |",
          "|---|---|---|---:|"]
    for p in ("q7", "q14", "qflash", "planner", "random"):
        L.append(f"| {p} | {f(pr[p]['7B'])} | {f(pr[p]['14B']) if '14B' in pr[p] else '—'} | {pr['best_is'][p]:.2f} |")
    if "interaction_q7_q14" in pr:
        L += ["", f"Interactive [R_7B(7B Tweeting) − R_7B(14B Tweeting)] − [R_14B(7B Tweeting) − R_14B(14B Tweeting)]：{f(pr['interaction_q7_q14'])}"]
    return "\n".join(L) + "\n"


# ================================================================ smoke test（7B Servers，Very small）
SMOKE = {"N_COLLECT": 12, "N_TRAJ": 3, "N_EXEC": {"q7": 2, "q14": 2}, "PROPOSERS": ["q7", "qflash"]}


def smoke_summary():
    import runner
    d = jload(P("design"))
    res = runner.load_results(P("evals", "q7")) if P("evals", "q7").exists() else {}
    steps = sum(sum(r["steps"]) for r in res.values())
    inval = sum(sum(r["invalid"]) for r in res.values())
    eps = load_episodes()
    val = {n: float(np.mean([len(r["actions"]) / CFG["N_PROPOSE"] for r in jl_read(P("proposals", n))]))
           for n in SMOKE["PROPOSERS"] if P("proposals", n).exists()}
    errs = {n: sum(1 for r in jl_read(P("proposals", n)) if r["error"]) for n in SMOKE["PROPOSERS"]
            if P("proposals", n).exists()}
    nb = len(stochastic(d))
    per_traj = nb / max(1, len(d["trajectories"]))
    # Full estimate：estimate stochastic branches per trajectory from the smoke test（One less than the author，Low）；Pining by stage B Main Experiment Actual（7B approximately 3,900 records/hours，
    # 14B About half of it）。Smoke and low，My own seconds/The strips cant be used to estimate the full amount
    full7 = 60 * per_traj * (10 + 1)
    full14 = 60 * per_traj * (20 + 1)
    summ = {"episodes": len(eps), "success": float(np.mean([e["won"] for e in eps.values()])),
            "trajectories": len(d["trajectories"]), "stochastic_branches_per_traj": per_traj,
            "proposal_valid_rate": val, "proposal_errors": errs, "invalid_rate_per_step": inval / max(1, steps),
            "projected_rollouts": {"q7": full7, "q14": full14},
            "projected_hours": {"q7": full7 / 3900, "q14": full14 / 1950}}
    jdump(P("smoke"), summ)
    ok = summ["invalid_rate_per_step"] <= 0.20 and all(v >= 0.5 for v in val.values()) and not any(errs.values())
    log(f"[smoke] {'passed' if ok else 'needs inspection'}：{json.dumps(summ, ensure_ascii=False)}")
    for n in SMOKE["PROPOSERS"]:
        rows = jl_read(P("proposals", n))
        if rows:
            log(f"[smoke] {n} Original proposal by the Secretary-General（Check Format）：\n{(rows[0]['text'] or '')[-600:]}")
    return 0 if ok else 1


# ================================================================ Sub-programming
def sub(stage, extra=()):
    cmd = [sys.executable, str(SCRIPT), "--stage", stage, "--out-dir", str(STATE["out"])] + list(extra)
    if STATE["mock"]:
        cmd.append("--mock")
    if STATE["smoke"]:
        cmd.append("--smoke-mode")
    gpu_stage = stage in ("collect", "eval") or (stage == "propose" and ("q7" in extra or "q14" in extra))
    if gpu_stage and not STATE["mock"] and not wait_gpu(stage):
        return 1
    log(f"----- Subprocess：{' '.join(cmd[2:])}")
    rc = subprocess.call(cmd, env=dict(os.environ, PYTHONHASHSEED=acfg.HASH_SEED))
    cleanup_leftovers()
    if rc != 0:
        log(f"----- Subprocess Failed（rc={rc}）：{stage}；after fixing the issue, rerun the same command to resume here")
    return rc


def pipeline(steps):
    for st, extra in steps:
        rc = sub(st, extra)
        if rc != 0:
            return rc
    return 0


# ================================================================ self-test（mock environment + mock model）
def selftest():
    import shutil
    import tempfile
    from alfw import mockenv
    ok = [True]

    def check(c, msg):
        print(("  ✓ " if c else "  ✗ ") + msg, flush=True)
        ok[0] = ok[0] and bool(c)
    tmp = Path(tempfile.mkdtemp(prefix="ds_alfw_"))
    saved = json.loads(json.dumps(CFG))
    try:
        STATE["mock"] = True
        CFG.update(N_COLLECT=40, N_TRAJ=6, N_EXEC={"q7": 4, "q14": 4})
        acfg.B_MAX_ACTIVE = 40
        A = tmp / "A"
        use_out(A)
        print("[ds_alfw selftest] collect → position → proposal → design → continuation → Analysis（mock）")
        check(stage_collect() == 0, "collect")
        check(stage_positions() == 0, "position")
        pos = jload(P("positions"))
        eps = load_episodes()
        fails = [k for k in sorted(eps) if not eps[k]["won"]]
        ids = [t["traj_id"] for t in pos["trajectories"]]
        check(ids == [k for k in fails if k not in {x["traj_id"] for x in pos["excluded"]}][:len(ids)],
              f"by rid Sequencing Failed Trails、unscreened（{len(ids)} records，Exclude {len(pos['excluded'])} records）")
        check(all(p["slack"] >= CFG["MIN_SLACK"] for t in pos["trajectories"] for p in t["positions"]), "remaining-step slack ≥ 5")
        for n in CFG["PROPOSERS"]:
            check(stage_propose(n) == 0, f"proposal {n}")
        check(stage_design() == 0, "design")
        d = jload(P("design"))
        bad = []
        for t in d["trajectories"]:
            ep = eps[t["traj_id"]]
            pp = {p["t"]: p for p in next(x for x in pos["trajectories"] if x["traj_id"] == t["traj_id"])["positions"]}
            for p in t["positions"]:
                acts = [d["branches"][a]["action"] for a in p["alts"]]
                adm, orig = pp[p["t"]]["admissible"], ep["cmds"][p["t"]]
                if len(set(acts)) != len(acts) or any(a not in adm or a == orig for a in acts):
                    bad.append(("cand", t["traj_id"], p["t"]))
                for a in p["alts"]:
                    b = d["branches"][a]
                    if b["prefix"] != ep["cmds"][:p["t"]] + [b["action"]]:
                        bad.append(("prefix", a))
                    if not (b["proposers"] or b["planner"] or b["random"]):
                        bad.append(("source", a))
                    if b["planner"] and b["action"] != pp[p["t"]]["planner"]:
                        bad.append(("planner", a))
                env = mockenv.MockEnv()
                env.load(ep["gamefile"])
                env.reset()
                for c in ep["cmds"][:p["t"]]:
                    st = env.step(c)
                if p["t"] > 0 and sorted(st["admissible"]) != sorted(pp[p["t"]]["admissible"]):
                    bad.append(("adm", t["traj_id"], p["t"]))
        check(not bad, f"The candidates are different、executable、Different from the original action；correct prefixes；Source recorded；Executable list is consistent with reset（Problem {bad[:3]}）")
        did = d["design_id"]
        P("design").unlink()
        stage_design()
        check(jload(P("design"))["design_id"] == did, "Design recoverable（Redo gets the same design_id）")
        print("[ds_alfw selftest] Pack → Unpack（No game files in the simulation environment，Inspection of outcome documents only）")
        stage_pack()
        B = tmp / "B"
        use_out(B)
        stage_unpack(A / "bundle_for_14b.tar.gz")
        same = all((A / f).read_bytes() == (B / f).read_bytes() for f in BUNDLE_FILES if (A / f).exists())
        check(same, "Unpackaged outcome document is identical to original bytes")
        for f in ("proposals_q14.jsonl",):
            (B / f).write_bytes((A / f).read_bytes())
        stage_design()
        check(jload(P("design"))["design_id"] == did, "The same design with the same input on the other machine")
        print("[ds_alfw selftest] continuation（including resume after interruption）")
        use_out(A)
        STATE["stop_after"] = 5
        try:
            stage_eval("q7")
        except KeyboardInterrupt:
            pass
        STATE["stop_after"] = None
        check(stage_eval("q7") == 0, "7B resume to complete interrupted continuations")
        lines1 = sorted(P("evals", "q7").read_text().splitlines())
        P("evals", "q7").unlink()
        STATE["n_backend"] = 0
        stage_eval("q7")
        check(STATE["n_backend"] == 1, f"A full run（Sample + Two conditions of greedy）Construction only 1 One executor（Actual {STATE['n_backend']}）")
        check(sorted(P("evals", "q7").read_text().splitlines()) == lines1, "Breaking up the continuations has the same result as a run-off")
        check(stage_eval("q14") == 0, "14B continuation（mock）")
        check(stage_analyze() == 0 and P("report").exists(), "Analysis and reporting")
        res = jload(P("analysis"))
        check("P1" in res["tests"] and "P2" in res["tests"] and res["curves"]["net"].get("14B_k1"), "Read it all（P1、P2、Budget curve）")
        print("[ds_alfw selftest] The known answers analysed")
        check(known_answer(), "known-answer test：When the executor prefers the same P2 ≈ 0；Not at the same time P2 Not bad at all；Save best time loss for filtering ≈ 0")
    finally:
        CFG.clear()
        CFG.update(saved)
        STATE["mock"] = False
        if os.environ.get("DS_KEEP"):
            print(f"[ds_alfw selftest] Keep temporary directory {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ds_alfw selftest] {'all passed' if ok[0] else 'failures detected'}")
    return 0 if ok[0] else 1


def known_answer():
    """Synthetic Data：One position per trajectory、Success rate of the original action 0、Two candidates。"""
    rng = np.random.default_rng(1)

    def mk(p7, p14, n=200, n7=10, n14=20):
        T7, T14, metas = [], [], []
        for i in range(n):
            bids = [f"s{i}|o", f"s{i}|a", f"s{i}|b"]
            cands = [(bids[1], bids[0], 0), (bids[2], bids[0], 0)]
            col = {b: j for j, b in enumerate(bids)}
            det = np.array([0.0, np.nan, np.nan])
            rand = np.array([1, 2])

            def arr(p, n_):
                S = (rng.random((2, n_)) < np.array(p)[:, None]).astype(float)
                G = det.copy()
                G[1:] = (np.array(p) > 0.5).astype(float)
                return {"cands": cands, "bids": bids, "col": col, "det": det, "rand": rand, "S": S, "G": G,
                        "ia": np.array([1, 2]), "io": np.array([0, 0])}
            T7.append(arr(p7, n7))
            T14.append(arr(p14, n14))
            metas.append({bids[0]: {"oracle_F": 1.0, "oracle_slack": 0.0, "planner": False, "random": False,
                                    "proposers": []},
                          bids[1]: {"oracle_F": 1.0, "oracle_slack": 5.0, "planner": True, "random": False,
                                    "proposers": ["q7"], "flash_rank": 0},
                          bids[2]: {"oracle_F": 1.0, "oracle_slack": 1.0, "planner": False, "random": True,
                                    "proposers": ["q14"], "flash_rank": None}})
        return T7, T14, metas
    good = True
    T7, T14, M = mk([0.7, 0.1], [0.7, 0.1])
    rows = analyze_core(T7, T14, lambda ti, b: M[ti][b], n_perm=10)
    same = boot([r["eq_7B_minus_14B"] for r in rows])
    good &= abs(same["mean"]) < 0.05
    T7, T14, M = mk([0.7, 0.1], [0.1, 0.7])
    rows = analyze_core(T7, T14, lambda ti, b: M[ti][b], n_perm=10)
    conf = boot([r["eq_7B_minus_14B"] for r in rows])
    good &= conf["ci"][0] > 0.3
    sl = boot([r["matched"] - r["SLpl_k5"] for r in rows])          # The planner filters are just the right ones 7B Best candidate in the world
    good &= abs(sl["mean"]) < 0.05
    print(f"    Same preference P2 {same['mean']:+.3f}；Different preferences P2 {conf['mean']:+.3f} [{conf['ci'][0]:+.3f}, {conf['ci'][1]:+.3f}]；"
          f"SL-planner loss {sl['mean']:+.3f}")
    return good


# ================================================================ The entrance
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "prep7", "prep14", "run7", "collect", "positions", "propose", "design", "eval",
                             "pack", "unpack", "analyze", "selftest"])
    ap.add_argument("--proposer", choices=["q7", "q14", "qflash"])
    ap.add_argument("--executor", choices=["q7", "q14"])
    ap.add_argument("--bundle", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--smoke-mode", action="store_true", help="Internal：smoke-test settings")
    ap.add_argument("--allow-missing-proposers", action="store_true")
    args = ap.parse_args()
    STATE["mock"] = args.mock
    if args.stage == "selftest":
        return selftest()
    STATE["smoke"] = args.smoke_mode or args.stage == "smoke"
    if STATE["smoke"]:
        CFG.update({k: v for k, v in SMOKE.items()})
        CFG["SEED"] = CFG["SEED"] + 100
    use_out(args.out_dir or (acfg.ROOT / ("out_ds_alfw_smoke" if STATE["smoke"] else "out_ds_alfw")))
    if not args.mock:
        try:
            import selftest as grid_selftest
            grid_selftest.t0_manifest()
        except AssertionError as e:
            log(f"Code version check failed：{e}")
            return 1
    log(f"===== ds_alfw --stage {args.stage}  PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} output {STATE['out']} =====")
    try:
        if args.stage == "smoke":
            rc = pipeline([("collect", []), ("positions", []), ("propose", ["--proposer", "q7"]),
                           ("propose", ["--proposer", "qflash"]), ("design", ["--allow-missing-proposers"]),
                           ("eval", ["--executor", "q7"])])
            return rc or smoke_summary()
        if args.stage == "prep7":
            return pipeline([("collect", []), ("positions", []), ("propose", ["--proposer", "q7"]),
                             ("propose", ["--proposer", "qflash"]), ("pack", [])])
        if args.stage == "prep14":
            steps = [("unpack", ["--bundle", str(args.bundle)])] if args.bundle else []
            steps += [("propose", ["--proposer", "q14"]), ("design", []), ("eval", ["--executor", "q14"])]
            return pipeline(steps)
        if args.stage == "run7":
            return pipeline([("eval", ["--executor", "q7"]), ("analyze", [])])
        if args.stage == "collect":
            return stage_collect()
        if args.stage == "positions":
            return stage_positions()
        if args.stage == "propose":
            if not args.proposer:
                sys.exit("--stage propose Yes --proposer")
            return stage_propose(args.proposer)
        if args.stage == "design":
            return stage_design(allow_missing=args.allow_missing_proposers)
        if args.stage == "eval":
            if not args.executor:
                sys.exit("--stage eval Yes --executor")
            return stage_eval(args.executor)
        if args.stage == "pack":
            return stage_pack()
        if args.stage == "unpack":
            if not args.bundle:
                sys.exit("--stage unpack Yes --bundle")
            return stage_unpack(args.bundle)
        if args.stage == "analyze":
            return stage_analyze()
    except SystemExit:
        raise
    except Exception:
        log(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != acfg.HASH_SEED:
        os.environ["PYTHONHASHSEED"] = acfg.HASH_SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.exit(main())
