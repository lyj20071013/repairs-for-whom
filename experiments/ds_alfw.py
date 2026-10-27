"""ALFWorld unscreened failure pools and executor comparisons (plan sections 6.9/6.10).

Qwen7B workflow, using separate Qwen7B and Qwen14B servers:
    python ds_alfw.py --stage smoke
    python ds_alfw.py --stage prep7
    python ds_alfw.py --stage prep14 --bundle out_ds_alfw/bundle_for_14b.tar.gz
    python ds_alfw.py --stage run7
    python ds_alfw.py --stage analyze
Transfer the frozen design back before run7, then merge complete executor records.
Qwen7B uses 10 stochastic continuations plus one greedy run; Qwen14B uses 20 plus one.

Llama deployment and bidirectional cross-family evaluation:
    python ds_alfw.py --profile l8 --stage smoke
    python ds_alfw.py --stage p610

Individual stages: collect, positions, propose, design, eval, pack, unpack.
Use --proposer q7|q14|qflash|l8 or --executor q7|q14|l8 where appropriate.
The p610 workflow also evaluates Llama on the existing section 6.9 design.
Model stages run in separate processes to release GPU memory between stages.
Compatible interrupted stages can resume. --stage selftest uses mock data only.
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
L8_DIR = os.environ.get("DS_L8", "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
MODEL_DESC = {"q7": "a 7B language model", "q14": "a 14B language model", "l8": "an 8B language model"}
LABEL = {"q7": "7B", "q14": "14B", "l8": "Llama-8B", "qflash": "flash", "planner": "planner", "random": "random"}
# 6.9 configuration（Down there CFG）preserve exactly：design、collection-reuse checks also compare this。6.10 The changes are in PROFILES / EXTRA_EXEC in。
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
CFG_69_FP = "2115236fc855711d9e1724f854a5f00dd77b48713bd348590160f9c74be8cb8e"   # Self-check check 6.9 The configuration has not been changed
PROFILES = {   # Second deployment model（6.10）：at 6.9 to overwrite these items on configuration
    "l8": {"DEPLOYER": "l8", "SEED": 2029, "N_COLLECT_MAX": 600, "COLLECT_CHUNK": 100, "N_FAIL_TARGET": 70,
             "PROPOSERS": ["l8", "q7", "qflash"], "N_EXEC": {"l8": 10, "q7": 10}},
}
EXTRA_EXEC = {"q7": {"l8": 10}, "l8": {}}   # 6.10 at 6.9 Designed additional executor（excluded from CFG，6.9 Settings signature remains unchanged）
SMOKES = {"q7": {"N_COLLECT": 12, "N_TRAJ": 3, "N_EXEC": {"q7": 2, "q14": 2}, "PROPOSERS": ["q7", "qflash"]},
          "l8": {"N_COLLECT_MAX": 60, "COLLECT_CHUNK": 20, "N_FAIL_TARGET": 3, "N_TRAJ": 3, "N_EXEC": {"l8": 2},
                   "PROPOSERS": ["l8", "qflash"]}}
OUT_NAMES = {"q7": "out_ds_alfw", "l8": "out_ds_alfw_l8"}
PROFILE = {"name": "q7"}
PROFILE_OUT = {k: acfg.ROOT / v for k, v in OUT_NAMES.items()}     # 6.10 Read the results of the two configurations on aggregation
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


def apply_profile(name, smoke=False):
    PROFILE["name"] = name
    if name != "q7":
        CFG.pop("N_COLLECT", None)
        CFG.update(json.loads(json.dumps(PROFILES[name])))
    if smoke:
        CFG.update(json.loads(json.dumps(SMOKES[name])))
        CFG["SEED"] = CFG["SEED"] + 100


def deployer():
    return CFG.get("DEPLOYER", "q7")


def executors():
    return list(CFG["N_EXEC"]) + [x for x in EXTRA_EXEC.get(PROFILE["name"], {}) if x not in CFG["N_EXEC"]]


def n_exec(name):
    return CFG["N_EXEC"].get(name) or EXTRA_EXEC.get(PROFILE["name"], {})[name]


def model_dir(name):
    return {"q7": acfg.MODEL_DIR, "q14": Q14_DIR, "l8": L8_DIR}[name]


def _stop_ids(tok):
    """Dialogue Ender（Llama-3.1：<|eot_id|>；Phi：<|end|> / <|im_end|>）；generation_config usually already contains，And heres a more visible addition。"""
    try:
        vocab = tok.get_vocab()
    except Exception:
        return []
    return [vocab[t] for t in ("<|eot_id|>", "<|end|>", "<|im_end|>", "<|endoftext|>") if t in vocab]


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


def _ep_row(e):
    from alfw import games
    return {"rid": e["rid"], "gamefile": e["gamefile"], "seed": e["seed"], "task": e["task"], "desc": e["desc"],
            "won": e["won"], "won_step": e["won_step"], "task_type": games.task_type_of(e["gamefile"]),
            "cmds": [s_["cmd"] for s_ in e["steps"]], "obs": [s_["obs"] for s_ in e["steps"]],
            "kinds": [s_["kind"] for s_ in e["steps"]]}


def _write_eps(rows):
    with open(P("episodes"), "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["rid"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stage_collect():
    from alfw import games, rollout
    train = SB.split_games("train") if not STATE["mock"] else [f"{games.MOCK_PREFIX}{i}" for i in range(2000)]
    excl = excluded_games(train)
    cand = [g for g in train if g not in excl and games.task_type_of(g) not in acfg.B_EXCLUDE_TYPES]
    q7_formal = set(games.sample(cand, 220, 2028))          # 6.9 for the formal run 220 episodes
    prof = PROFILE["name"]
    if prof != "q7":                                       # 6.10：not used 6.9 Game
        cand = [g for g in cand if g not in q7_formal]
    if STATE["smoke"]:                                     # A game that smokes without proper running
        formal = q7_formal if prof == "q7" else set(games.sample(cand, PROFILES[prof]["N_COLLECT_MAX"],
                                                                 PROFILES[prof]["SEED"]))
        cand = [g for g in cand if g not in formal]
    adaptive = "N_COLLECT_MAX" in CFG
    gs = games.sample(cand, CFG["N_COLLECT_MAX"] if adaptive else CFG["N_COLLECT"], CFG["SEED"])
    pre = "n" if prof == "q7" else "p"
    specs = [{"rid": f"{pre}{i:03d}", "gamefile": g, "seed": derive_seed("DS1", g)} for i, g in enumerate(gs)]
    key = fp({"specs": [[s_["rid"], s_["gamefile"], s_["seed"]] for s_ in specs], "s": settings()})
    meta = jload(P("episodes_meta")) if P("episodes_meta").exists() else {}
    if meta.get("key") == key and (not adaptive or meta.get("complete")):
        log(f"[collect] Reuse already available {len(load_episodes())} episodes")
        return 0
    D = deployer()
    if not adaptive:
        log(f"[collect] Optional Game {len(cand)} episodes（Exclusion phase A / B previously used {len(excl)} Office of the Secretary-General of the United Nations {acfg.B_EXCLUDE_TYPES}），sample {len(gs)} episodes")
        with SB.make_pool() as pl:
            be = make_exec_backend(D)
            eps, timing = rollout.run_episodes(be, pl, specs, temperature=acfg.TEMPERATURE, keep_text=False,
                                               want_key=False, log=log)
            release_vllm(be)
        _write_eps([_ep_row(e) for e in eps])
        jdump(P("episodes_meta"), {"key": key, "timing": timing})
        n_s = sum(e["won"] for e in eps)
        log(f"[collect] complete：Success {n_s}/{len(eps)}，failed {len(eps) - n_s}；Time {timing['wall'] / 60:.1f} minutes")
        return 0
    # Well take it together in a sample order，Number of failures reached N_FAIL_TARGET Just stop（The results of each game are determined only by games and seeds，independent of chunking）
    rows = list(load_episodes().values()) if meta.get("key") == key else []
    timings = meta.get("timings", []) if meta.get("key") == key else []
    done = len(rows)
    if [r["rid"] for r in sorted(rows, key=lambda x: x["rid"])] != [s_["rid"] for s_ in specs[:done]]:
        sys.exit("[collect] Existing stations are not prefixes to the sampling order；Please change the output directory")
    log(f"[collect] Optional Game {len(cand)} episodes（Exclusion phase A / B previously used {len(excl)} episodes、6.9 of 220 Office of the Secretary-General of the United Nations "
        f"{acfg.B_EXCLUDE_TYPES}）；Up in order {len(gs)} episodes，Each piece {CFG['COLLECT_CHUNK']} episodes，Failed to {CFG['N_FAIL_TARGET']} No more rules"
        f"（Existing {done} episodes）")
    be = None
    with SB.make_pool() as pl:
        while sum(not r["won"] for r in rows) < CFG["N_FAIL_TARGET"] and done < len(specs):
            batch = specs[done:done + CFG["COLLECT_CHUNK"]]
            if be is None:
                be = make_exec_backend(D)
            eps, tm = rollout.run_episodes(be, pl, batch, temperature=acfg.TEMPERATURE, keep_text=False,
                                           want_key=False, log=log)
            rows += [_ep_row(e) for e in eps]
            done += len(batch)
            timings.append(tm)
            _write_eps(rows)
            jdump(P("episodes_meta"), {"key": key, "complete": False, "timings": timings})
            n_f = sum(not r["won"] for r in rows)
            log(f"[collect] Picked {done} episodes：Success {done - n_f}，failed {n_f}（{n_f / done:.0%}）")
        release_vllm(be)
    jdump(P("episodes_meta"), {"key": key, "complete": True, "timings": timings, "n_collected": done})
    n_f = sum(not r["won"] for r in rows)
    log(f"[collect] complete：{done} episodes，Success {done - n_f}，failed {n_f}" +
        ("" if n_f >= CFG["N_FAIL_TARGET"] else f"（Not yet {CFG['N_FAIL_TARGET']} records：no eligible games remain）"))
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
            f"The agent is {MODEL_DESC[deployer()]} that samples its actions at temperature 0.7.\n\n"
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
    if name in ("q7", "q14", "l8"):
        tp = int(os.environ.get("LADDER_TP", "1"))
        if tp > 1:
            import functools
            import vllm
            vllm.LLM = functools.partial(vllm.LLM, tensor_parallel_size=tp)
        j = JB.VLLMJudge(model_dir(name), context=16384, temperature=0.0, max_tokens=1024, name=name)
        if name == "l8":
            from vllm import SamplingParams
            j.params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop_token_ids=_stop_ids(j.tok))
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
    b = agent.VLLMBackend(model_dir=model_dir(name))
    if name == "l8":
        import functools
        b._SP = functools.partial(b._SP, stop_token_ids=_stop_ids(b.tok))
    return b


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
    n = n or n_exec(name)
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
EXEC_CODE = {"q7": 1, "q14": 2, "l8": 3}      # Each executors own random stream（Add or minus implementer does not change other readings）
# Registration check for each configuration：Group -> [(Numbering, Statistics, Annotations)]。6.9 The family is in this configuration Holm；6.10 Group between two configurations，perform in the summary Holm
TESTS = {
    "q7": {"6.9": [("P1", "matched-greedy", "Match − 7B greedy"),
                   ("P2", "eq_minus_q14", "Same thing 5 runs：7B select using the deployed executor − by 14B Selection")],
           "6.10": [("H1", "eq_minus_l8", "Same thing 5 runs：7B select using the deployed executor − by Llama-8B Selection")]},
    "l8": {"6.10": [("H2", "eq_minus_q7", "Same thing 5 runs：Llama-8B select using the deployed executor − by 7B Selection"),
                      ("H3", "matched-greedy", "Match − Llama-8B greedy")]},
}
FAMILY_SIZE = {"6.9": 2, "6.10": 3}


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


def _has(m, p):
    return bool(m.get(p)) if p in ("planner", "random") else p in m.get("proposers", [])


def analyze_core(TD, others, branch_meta, obj="net", seed=0, n_perm=N_PERM, ks_self=(1, 2, 3, 5),
                 ks_other=(1, 2, 3, 5, 10, 20), proposers=("q7", "q14", "qflash")):
    """One-line statistics per trajectories（all are values of selected repairs under the deployed executor）。TD：Personnel deployed _traj_arrays；others：{executor: On the same list
    _traj_arrays}；branch_meta(trajectories Serial Numbers, branch) -> Metadata。Random Number Press（trajectory，executor）one independent stream each。"""
    others = others or {}
    srcs = list(proposers) + ["planner", "random"]
    rows = []
    for ti, A in enumerate(TD):
        rng = np.random.default_rng([seed, ti])
        ia, io, det, rand = A["ia"], A["io"], A["det"], A["rand"]
        nD = A["S"].shape[1]
        h = nD // 2
        fullD = _full(det, rand, A["S"].mean(axis=1) if len(rand) else [])
        gD = _gain(fullD, ia, io, obj)
        meta = [branch_meta(ti, a) for a, _, _ in A["cands"]]
        r = {}
        # ---- Deployers themselves k Subselection（Crossing）：The first half of each rehearsing（before k runs）、The second half，Reconcile
        perms = [np.argsort(rng.random(A["S"].shape), axis=1) for _ in range(n_perm)]

        def crossfit(k, sub=None):
            vals = []
            for pm in perms:
                Z = np.take_along_axis(A["S"], pm, axis=1)
                for a_, b_ in ((slice(0, h), slice(h, nD)), (slice(h, nD), slice(0, h))):
                    sel = _full(det, rand, Z[:, a_][:, :k].mean(axis=1))
                    evv = _full(det, rand, Z[:, b_].mean(axis=1))
                    sc = _gain(sel, ia, io, obj)
                    if sub is not None:
                        sc = np.where(sub, sc, -np.inf)
                    vals.append(_val(_best(sc), _gain(evv, ia, io, obj)))
            return float(np.mean(vals))
        for k in ks_self:
            if k <= h:
                r[f"self_k{k}"] = crossfit(k)
        r["matched"] = r[f"self_k{h}"] if f"self_k{h}" in r else crossfit(h)
        # ---- You dont have to choose a sample of people to deploy：Evaluation of all samples of deployed personnel
        r["greedy"] = _val(_best(_gain(A["G"], ia, io, obj)), gD)
        planner = np.array([m["planner"] for m in meta])
        pos_i = np.array([c[2] for c in A["cands"]])
        if planner.any():
            first = pos_i[planner].min()
            r["planner_earliest"] = _val(planner & (pos_i == first), gD)
        else:
            r["planner_earliest"] = float(gD.mean())
        F = np.array([m["oracle_F"] for m in meta])
        SL = np.array([m["oracle_slack"] for m in meta])
        Fo = np.array([branch_meta(ti, o)["oracle_F"] for _, o, _ in A["cands"]])
        SLo = np.array([branch_meta(ti, o)["oracle_slack"] for _, o, _ in A["cands"]])
        if obj == "net":
            r["oracle_slack"] = _val(_best((F - Fo) * 1000 + (SL - SLo)), gD)
        else:
            r["oracle_slack"] = _val(_best(F * 1000 + SL), gD)
        r["slack_abs"] = _val(_best(F * 1000 + SL), gD)
        r["random"] = float(gD.mean())
        r["best_in_sample"] = float(gD.max())
        # ---- Choice of other implementers（k runs，Rehearsal），Evaluation of all samples of deployed personnel
        g_other = {}
        for o, TO in others.items():
            B = TO[ti]
            ro = np.random.default_rng([seed, ti, EXEC_CODE.get(o, 9)])
            n_o = B["S"].shape[1]
            g_other[o] = _gain(_full(B["det"], B["rand"], B["S"].mean(axis=1) if len(B["rand"]) else []), ia, io, obj)
            for k in ks_other:
                if k <= n_o:
                    vals = []
                    for _ in range(n_perm if k < n_o else 1):
                        pm = np.argsort(ro.random(B["S"].shape), axis=1)
                        Z = np.take_along_axis(B["S"], pm, axis=1)[:, :k].mean(axis=1)
                        vals.append(_val(_best(_gain(_full(B["det"], B["rand"], Z), ia, io, obj)), gD))
                    r[f"{o}_k{k}"] = float(np.mean(vals))
            if f"{o}_k{h}" in r:
                r[f"eq_minus_{o}"] = r["matched"] - r[f"{o}_k{h}"]
            r[f"greedy_{o}"] = _val(_best(_gain(B["G"], ia, io, obj)), gD)
        # ---- Methodology：Screening for follow-up
        alts_by_pos = {}
        for c in range(len(meta)):
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
            for k in ks_self:
                if k <= h:
                    r[f"{tag}_k{k}"] = crossfit(k, sub) if sub.any() else (0.0 if obj == "net" else float(fullD[io].mean()))
            best = _best(gD)
            r[f"recall_{tag}"] = float((best & sub).sum() / best.sum())
        # ---- proposer × executor
        for p in srcs:
            m_ = np.array([_has(m, p) for m in meta])
            r[f"prop_self_{p}"] = float(gD[m_].mean()) if m_.any() else np.nan
            for o, g_o in g_other.items():
                r[f"prop_{o}_{p}"] = float(g_o[m_].mean()) if m_.any() else np.nan
        bD = np.flatnonzero(_best(gD))
        r["best_is"] = {p: float(sum(_has(meta[c], p) for c in bD) / len(bD)) for p in srcs}
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


def holm(tests, ids, m, alpha=0.05):
    """Holm（One side α）：Registered family size m；tests without results use p = 1 Account（can only be more conservative）。"""
    ps = sorted([(tests[i]["p"], i) for i in ids] + [(1.0, None)] * (m - len(ids)))
    ok = True
    for j, (pv, i) in enumerate(ps):
        thr = alpha / (m - j)
        rej = bool(ok and pv <= thr)
        if i is not None:
            tests[i]["holm_threshold"], tests[i]["holm_reject"] = thr, rej
        ok = ok and rej


def stat_values(R, stat):
    if stat == "matched-greedy":
        return [r["matched"] - r["greedy"] for r in R]
    if all(stat in r for r in R):
        return [r[stat] for r in R]
    return None


def restrict_design(d, keep):
    """Keep Only keep(branch) For real candidates；No candidate position、Get rid of the trajectories。Back (New design, No trajectories number for the candidate)。"""
    import copy
    d2, empty = copy.deepcopy(d), []
    br = d["branches"]
    for t in d2["trajectories"]:
        for p in t["positions"]:
            p["alts"] = [a for a in p["alts"] if keep(br[a])]
        t["positions"] = [p for p in t["positions"] if p["alts"]]
        if not t["positions"]:
            empty.append(t["traj_id"])
    d2["trajectories"] = [t for t in d2["trajectories"] if t["positions"]]
    return d2, empty


def stage_analyze():
    import runner
    d = jload(P("design"))
    D = deployer()
    have = {n: runner.load_results(P("evals", n)) for n in executors() if P("evals", n).exists()}
    if D not in have or missing(d, D):
        log(f"[analyze] deployed executor {LABEL[D]} continuations are incomplete（evals_{D}.jsonl）")
        return 1
    nmin = lambda n: min(len(r["results"]) for (c, _, _), r in have[n].items() if c == f"{n}_T07")
    TD = [_traj_arrays(d, t, have[D], D, nmin(D)) for t in d["trajectories"]]
    others = {}
    for n in executors():
        if n == D:
            continue
        if n in have and not missing(d, n):
            others[n] = [_traj_arrays(d, t, have[n], n, nmin(n)) for t in d["trajectories"]]
        else:
            log(f"[analyze] {LABEL[n]} or follow-up is missing or incomplete：skip comparisons requiring that executor")
    meta = lambda ti, b: d["branches"][b]
    props = tuple(CFG["PROPOSERS"])
    res = {"design_id": d["design_id"], "n_traj": len(TD), "profile": PROFILE["name"], "deployer": D,
           "others": sorted(others), "proposers": list(props)}
    for obj in ("net", "post"):
        res[obj] = {"rows": analyze_core(TD, others, meta, obj=obj, proposers=props)}
    R = res["net"]["rows"]
    col = lambda k: [r.get(k, np.nan) for r in R]
    diff = lambda a, b: [r[a] - r[b] for r in R]
    out = {"PC": boot(diff("matched", "random")), "S1": boot(diff("matched", "planner_earliest")),
           "S2": boot(diff("matched", "oracle_slack")), "slack_abs": boot(diff("matched", "slack_abs"))}
    out["PC"]["pass"] = out["PC"]["ci"][0] > 0.10
    for fam, tests in TESTS.get(PROFILE["name"], {}).items():
        for tid, stat, lab in tests:
            v = stat_values(R, stat)
            if v is not None:
                out[tid] = dict(boot(v), label=lab, family=fam, stat=stat)
    if PROFILE["name"] == "q7":
        holm(out, [t for t in ("P1", "P2") if t in out], FAMILY_SIZE["6.9"])
    res["tests"] = out
    # ---- Budget curve（Two targets）
    curves = {}
    for obj in ("net", "post"):
        RR = res[obj]["rows"]
        c = {f"self_k{k}": boot([r[f"self_k{k}"] for r in RR if f"self_k{k}" in r]) for k in (1, 2, 3, 5)}
        for o in others:
            c.update({f"{o}_k{k}": boot([r[f"{o}_k{k}"] for r in RR if f"{o}_k{k}" in r]) for k in (1, 2, 3, 5, 10, 20)})
            c[f"eq_minus_{o}"] = boot([r.get(f"eq_minus_{o}", np.nan) for r in RR])
            c[f"greedy_{o}"] = boot([r[f"greedy_{o}"] for r in RR])
        c["greedy"] = boot([r["greedy"] for r in RR])
        c["matched"] = boot([r["matched"] for r in RR])
        curves[obj] = c
    res["curves"] = curves
    # ---- time per continuation
    sec = {}
    for n in executors():
        if P("timing", n).exists():
            runs = [r for r in jload(P("timing", n))["runs"] if r["cond"] == f"{n}_T07/{CFG['BATCH']}" and r["rollouts"]]
            if runs:
                sec[n] = {"sec": sum(r["wall"] for r in runs) / sum(r["rollouts"] for r in runs),
                          "gpu": runs[-1]["gpu"], "tp": runs[-1]["tp"]}
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
        if all(f"self_k{k}" in r for r in R):
            meth[f"all_k{k}"] = {"loss": boot(diff("matched", f"self_k{k}")),
                                 "budget": float(np.mean([k * r["budget_all"] for r in R]))}
    meth["greedy"] = {"loss": boot(diff("matched", "greedy")), "budget": float(np.mean([r["budget_all"] for r in R]))}
    res["method"] = meth
    # ---- proposer × executor
    srcs = list(props) + ["planner", "random"]
    prop = {p: {"self": boot(col(f"prop_self_{p}")), **{o: boot(col(f"prop_{o}_{p}")) for o in others}} for p in srcs}
    for o in others:
        if o in props:                      # [R_ deployed executor(Redeployed) − R_ deployed executor(o Tweeting)] − [R_o(Redeployed) − R_o(o Tweeting)]
            prop[f"interaction_{o}"] = boot([(r[f"prop_self_{D}"] - r[f"prop_self_{o}"]) -
                                             (r[f"prop_{o}_{D}"] - r[f"prop_{o}_{o}"]) for r in R])
    prop["best_is"] = {p: float(np.mean([r["best_is"][p] for r in R])) for p in srcs}
    res["proposer_x_executor"] = prop
    # ---- model-proposed candidates only（6.9a Practices；6.10 Register as secondary reading）
    d2, empty = restrict_design(d, lambda b: bool(b.get("proposers")))
    TD2 = [_traj_arrays(d2, t, have[D], D, nmin(D)) for t in d2["trajectories"]]
    mo = {"n_traj": len(TD2), "n_empty": len(empty)}
    for obj in ("net", "post"):
        R2 = analyze_core(TD2, None, meta, obj=obj, proposers=props)
        pad = [0.0] * len(empty) if obj == "net" else []
        mo[obj] = {"matched": boot([r["matched"] for r in R2] + pad),
                   "matched-greedy": boot([r["matched"] - r["greedy"] for r in R2] + pad),
                   "frac_best_gt_0.10": float(np.mean([r["best_in_sample"] > 0.10 for r in R2] + [False] * len(empty)))}
    res["model_only"] = mo
    res["describe"] = describe(d, have)
    res["describe"]["frac_traj_best_gain_gt_0.10"] = float(np.mean([r["best_in_sample"] > 0.10 for r in R]))
    for obj in ("net", "post"):
        for r in res[obj]["rows"]:
            r.pop("best_is", None)
    jdump(P("analysis"), res)
    P("report").write_text(report(res), encoding="utf-8")
    log(f"[analyze] -> {P('report')}")
    combine_610()
    return 0


def combine_610():
    """6.10 The three major tests are in two configurations：Read it on both sides analysis.json，Within the family Holm（m = 3），Write Llama-8B Output directory of the configuration。"""
    got = {}
    for prof in ("q7", "l8"):
        f = Path(PROFILE_OUT[prof]) / "analysis.json"
        if f.exists():
            tests = jload(f)["tests"]
            for tid, t in tests.items():
                if t.get("family") == "6.10":
                    got[tid] = dict(t, pc_pass=bool(tests["PC"]["pass"]))
    if not got:
        return
    holm(got, sorted(got), FAMILY_SIZE["6.10"])
    out_dir = Path(PROFILE_OUT["l8"])
    out_dir.mkdir(parents=True, exist_ok=True)
    jdump(out_dir / "tests_610.json", got)
    f = lambda b: f"{b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]"
    L = ["# 6.10 Main Check Summary（Holm，One side α = 0.05，Group Size 3；missing results use p = 1）", "",
         "| Test | Contrast | estimate [90%] | p | decision |", "|---|---|---|---|---|"]
    for tid in ("H1", "H2", "H3"):
        if tid in got:
            t = got[tid]
            verdict = ("insufficient discriminative signal（the deployed executors positive control failed）" if not t["pc_pass"]
                       else "confirmed" if t["holm_reject"] else "not confirmed")
            L.append(f"| {tid} | {t['label']} | {f(t)} | {t['p']:.4f} | {verdict} |")
        else:
            L.append(f"| {tid} | — | results pending | — | — |")
    if STATE["mock"] or STATE["smoke"]:
        L.insert(1, "\n> **Simulate or smoke run**：The numbers dont mean anything。\n")
    (out_dir / "report_610.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"[analyze] 6.10 Summary -> {out_dir / 'report_610.md'}（Existing {sorted(got)}）")


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


def report(res):
    f = lambda b: f"{b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]" if b["n"] else "—"
    t = res["tests"]
    D, others = res["deployer"], res["others"]
    dl = LABEL[D]
    title = {"q7": "ALFWorld Do Not Filter General × Candidates for the model（6.9；Llama-8B See the executor 6.10）",
             "l8": "ALFWorld Second deployment model Llama-8B（6.10）"}[res["profile"]]
    L = [f"# {title}", ""]
    if STATE["mock"] or STATE["smoke"]:
        L += ["> **Simulate or smoke run**：The numbers dont mean anything。", ""]
    L += [f"deployed executor {dl}；Other implementers {[LABEL[o] for o in others] or 'none'}；design {res['design_id'][:12]}；{res['n_traj']} trajectories。",
          f"description：{json.dumps(res['describe'], ensure_ascii=False)}", "",
          f"Value = Select the person to be rehabilitated in the deployment {dl} Difference between net gains above（90% trajectory bootstrap interval）；Match = {dl} Myself 5 Subselection、Other 5 Commentary（Crossing）。", "",
          "| Read | Contrast | estimate [90%] | decision |", "|---|---|---|---|",
          f"| positive control | Match − random | {f(t['PC'])} | {'passed' if t['PC']['pass'] else 'failed：insufficient discriminative signal'} |"]
    for fam, tests in TESTS.get(res["profile"], {}).items():
        for tid, _, lab in tests:
            if tid not in t:
                L.append(f"| {tid}（{fam}） | {lab} | results pending | — |")
            elif fam == "6.9":
                L.append(f"| {tid}（6.9，Holm） | {lab} | {f(t[tid])} | {'confirmed' if t[tid].get('holm_reject') else 'not confirmed'}"
                         f"（p = {t[tid]['p']:.4f}） |")
            else:
                L.append(f"| {tid}（6.10） | {lab} | {f(t[tid])} | p = {t[tid]['p']:.4f}；Decision report_610.md |")
    L += [f"| S1 | Match − Planner actions are in the earliest position | {f(t['S1'])} | {'confirmed' if t['S1']['ci'][0] > 0 else 'not confirmed'} |",
          f"| S2 | Match − oracle + slack（Definition of registration） | {f(t['S2'])} | {'confirmed' if t['S2']['ci'][0] > 0 else 'not confirmed'} |",
          f"| description | Match − Select in absolute remainder | {f(t['slack_abs'])} | — |", ""]
    sec = res.get("sec_per_rollout", {})
    if D in sec:
        L += ["Surviving costs（Every second of a wall clock running × Number of parallel cards；The cards dont match at the same time）：" +
              "；".join(f"{LABEL[n]} {v['sec']:.2f} seconds × {v['tp']} Cut（{(v['gpu'] or {}).get('names')}）"
                       for n, v in sec.items()), ""]
    for obj, lab in (("net", "net gain"), ("post", "Success rate after restoration")):
        c = res["curves"][obj]
        L += [f"**Budget curve（{lab}；_Other Organiser {dl} Value）**", "", "| Selection | Value [90%] |", "|---|---|"]
        for k in (1, 2, 3, 5):
            if c[f"self_k{k}"]["n"]:
                L.append(f"| {dl} {k} runs | {f(c[f'self_k{k}'])} |")
        L.append(f"| {dl} greedy（1 runs / branch） | {f(c['greedy'])} |")
        for o in others:
            ratio = (sec[o]["sec"] * sec[o]["tp"]) / (sec[D]["sec"] * sec[D]["tp"]) if o in sec and D in sec else None
            for k in (1, 2, 3, 5, 10, 20):
                if c.get(f"{o}_k{k}", {}).get("n"):
                    L.append(f"| by {LABEL[o]} {k} runs" + (f"（≈ {dl} {k * ratio:.1f} The next card time）" if ratio else "") +
                             f" | {f(c[f'{o}_k{k}'])} |")
            L.append(f"| by {LABEL[o]} greedy | {f(c[f'greedy_{o}'])} |")
            L.append(f"| Poor：{dl} 5 runs − {LABEL[o]} 5 runs | {f(c[f'eq_minus_{o}'])} |")
        L.append("")
    m = res["method"]
    L += ["**Methodology（Freezing SL）**：Relatively matched loss and running budget for each trajectory", "", "| Modalities | budget | loss [90%] |", "|---|---:|---|"]
    for key in sorted(k for k in m if isinstance(m[k], dict)):
        L.append(f"| {key} | {m[key]['budget']:.0f} | {f(m[key]['loss'])} |")
    L += ["", f"Screening recall rate（{dl} The best candidate in all samples is screened）：planner {m.get('recall_SLpl', float('nan')):.2f}；"
              f"flash {m.get('recall_SLfl', float('nan')):.2f}", ""]
    mo = res["model_only"]
    L += [f"**model-proposed candidates only**（{mo['n_traj']} trajectories have candidates，{mo['n_empty']} records without、Based on non-repair）：Match {f(mo['net']['matched'])}；"
          f"Match − greedy {f(mo['net']['matched-greedy'])}（Success rate after restoration {f(mo['post']['matched-greedy'])}）；"
          f"Net gains available > 0.10 trajectories of Candidates {mo['net']['frac_best_gt_0.10']:.2f}", ""]
    pr = res["proposer_x_executor"]
    hdr = [f"at {dl} on"] + [f"at {LABEL[o]} on" for o in others]
    L += ["**proposer × executor**（Average net gains from candidates from various sources；exploratory）", "",
          "| Source | " + " | ".join(hdr) + f" | yes {dl} Proportion of best candidates |", "|---|" + "---|" * len(hdr) + "---:|"]
    for p in res["proposers"] + ["planner", "random"]:
        L.append(f"| {LABEL.get(p, p)} | " + " | ".join([f(pr[p]["self"])] + [f(pr[p][o]) for o in others]) +
                 f" | {pr['best_is'][p]:.2f} |")
    for o in others:
        if f"interaction_{o}" in pr:
            L += ["", f"Interactive [R_{dl}({dl} Tweeting) − R_{dl}({LABEL[o]} Tweeting)] − [R_{LABEL[o]}({dl} Tweeting) − R_{LABEL[o]}({LABEL[o]} Tweeting)]："
                      f"{f(pr[f'interaction_{o}'])}"]
    return "\n".join(L) + "\n"


# ================================================================ smoke test（7B Servers，Very small）
def smoke_summary():
    import runner
    d = jload(P("design"))
    D = deployer()
    res = runner.load_results(P("evals", D)) if P("evals", D).exists() else {}
    steps = sum(sum(r["steps"]) for r in res.values())
    inval = sum(sum(r["invalid"]) for r in res.values())
    retr = sum(sum(r.get("retries", [0])) for r in res.values())
    eps = load_episodes()
    val = {n: float(np.mean([len(r["actions"]) / CFG["N_PROPOSE"] for r in jl_read(P("proposals", n))]))
           for n in CFG["PROPOSERS"] if P("proposals", n).exists()}
    errs = {n: sum(1 for r in jl_read(P("proposals", n)) if r["error"]) for n in CFG["PROPOSERS"]
            if P("proposals", n).exists()}
    nb = len(stochastic(d))
    per_traj = nb / max(1, len(d["trajectories"]))
    succ = float(np.mean([e["won"] for e in eps.values()]))
    prof = PROFILE["name"]
    formal = PROFILES.get(prof, {"N_EXEC": {"q7": 10, "q14": 20}})
    full = {n: 60 * per_traj * (k + 1) for n, k in formal["N_EXEC"].items()}
    br_succ = float(np.mean([np.mean(r["results"]) for (c, _, _), r in res.items() if c == f"{D}_T07"])) if res else None
    summ = {"deployer": D, "episodes": len(eps), "success": succ, "fails": sum(not e["won"] for e in eps.values()),
            "branch_success": br_succ,
            "trajectories": len(d["trajectories"]), "stochastic_branches_per_traj": per_traj,
            "proposal_valid_rate": val, "proposal_errors": errs, "invalid_rate_per_step": inval / max(1, steps),
            "retries_per_step": retr / max(1, steps), "projected_rollouts": full}
    if prof != "q7":
        summ["projected_games_for_70_failures"] = round(70 / max(1e-9, 1 - succ))
    jdump(P("smoke"), summ)
    ok = summ["invalid_rate_per_step"] <= 0.20 and all(v >= 0.5 for v in val.values()) and not any(errs.values()) \
        and summ["trajectories"] > 0
    log(f"[smoke] {'passed' if ok else 'needs inspection'}：{json.dumps(summ, ensure_ascii=False)}")
    for n in CFG["PROPOSERS"]:
        rows = jl_read(P("proposals", n))
        if rows:
            log(f"[smoke] {LABEL[n]} Original proposal by the Secretary-General（Check Format）：\n{(rows[0]['text'] or '')[-600:]}")
    return 0 if ok else 1


# ================================================================ Sub-programming
def sub(stage, extra=(), profile=None):
    profile = profile or PROFILE["name"]
    out = STATE["out"] if profile == PROFILE["name"] else PROFILE_OUT[profile]
    cmd = [sys.executable, str(SCRIPT), "--stage", stage, "--profile", profile, "--out-dir", str(out)] + list(extra)
    if STATE["mock"]:
        cmd.append("--mock")
    if STATE["smoke"]:
        cmd.append("--smoke-mode")
    gpu_stage = stage in ("collect", "eval") or (stage == "propose" and any(x in extra for x in ("q7", "q14", "l8")))
    if gpu_stage and not STATE["mock"] and not wait_gpu(stage):
        return 1
    log(f"----- Subprocess：{' '.join(cmd[2:])}")
    rc = subprocess.call(cmd, env=dict(os.environ, PYTHONHASHSEED=acfg.HASH_SEED))
    cleanup_leftovers()
    if rc != 0:
        log(f"----- Subprocess Failed（rc={rc}）：{stage}；after fixing the issue, rerun the same command to resume here")
    return rc


def pipeline(steps):
    for step in steps:
        st, extra = step[0], step[1]
        rc = sub(st, extra, step[2] if len(step) > 2 else None)
        if rc != 0:
            return rc
    return 0


def deployer_steps(prof_cfg, smoke=False):
    props = prof_cfg["PROPOSERS"]
    steps = [("collect", []), ("positions", [])] + [("propose", ["--proposer", n]) for n in props]
    steps.append(("design", ["--allow-missing-proposers"] if smoke else []))
    return steps


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
    saved_extra = json.loads(json.dumps(EXTRA_EXEC))
    saved_out = dict(PROFILE_OUT)
    try:
        check(fp(CFG) == CFG_69_FP, "6.9 configuration is unchanged（Set signature unchanged，The collection and design available can be reused）")
        STATE["mock"] = True
        CFG.update(N_COLLECT=40, N_TRAJ=6, N_EXEC={"q7": 4, "q14": 4})
        EXTRA_EXEC["q7"] = {"l8": 4}
        acfg.B_MAX_ACTIVE = 40
        A = tmp / "A"
        PROFILE_OUT.update(q7=A, l8=tmp / "C")
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
        check(stage_analyze() == 0, "analysis also works with deployed-executor outcomes only（Read over by other executors）")
        r0 = jload(P("analysis"))
        check(stage_eval("q14") == 0 and stage_eval("l8") == 0, "14B、Llama-8B at 6.9 The designs running again（mock）")
        check(stage_analyze() == 0 and P("report").exists(), "Analysis and reporting")
        res = jload(P("analysis"))
        check(all(k in res["tests"] for k in ("P1", "P2", "H1")) and res["curves"]["net"]["q14_k1"]["n"]
              and res["curves"]["net"]["l8_k1"]["n"], "Read it all（P1、P2、H1、Budget curves for the two implementers）")
        check(abs(res["tests"]["P1"]["mean"] - r0["tests"]["P1"]["mean"]) < 1e-12,
              "Add other executors without changing their own readings（Random number split by executor）")
        print("[ds_alfw selftest] Second deployment model（Llama-8B Profile，mock）")
        CFG.clear()
        CFG.update(saved)
        apply_profile("l8")
        CFG.update(N_COLLECT_MAX=200, COLLECT_CHUNK=15, N_FAIL_TARGET=8, N_TRAJ=5, N_EXEC={"l8": 4, "q7": 4})
        use_out(tmp / "C")
        check(stage_collect() == 0, "collect（Press the block till it fails）")
        eps2 = load_episodes()
        n_f = sum(not e["won"] for e in eps2.values())
        n_prev = len(eps2) - CFG["COLLECT_CHUNK"]
        prev_f = sum(not eps2[k]["won"] for k in sorted(eps2)[:n_prev])
        check(n_f >= 8 and prev_f < 8 and len(eps2) % 15 == 0 and all(k.startswith("p") for k in eps2),
              f"The piece that was collected at the number of failures that reached the target for the first time stopped（{len(eps2)} episodes，failed {n_f}）")
        check(not ({e["gamefile"] for e in eps2.values()} & {e["gamefile"] for e in eps.values()}),
              "Llama-8B games and 6.9 do not overlap")
        check(stage_collect() == 0 and len(load_episodes()) == len(eps2), "Reuse directly after collection")
        check(stage_positions() == 0 and all(stage_propose(n) == 0 for n in CFG["PROPOSERS"]) and stage_design() == 0,
              "position → Llama-8B / 7B / flash proposal → design")
        msg = propose_messages(eps2[jload(P("positions"))["trajectories"][0]["traj_id"]],
                               jload(P("positions"))["trajectories"][0]["positions"][0], 2)[1]["content"]
        check("The agent is an 8B language model" in msg, "The prompts of the proposal is for a deploymenter Llama-8B（8B）")
        check(stage_eval("l8") == 0 and stage_eval("q7") == 0 and stage_analyze() == 0, "Llama-8B、7B Follow-up and analysis")
        res2 = jload(P("analysis"))
        check("H2" in res2["tests"] and "H3" in res2["tests"] and "interaction_q7" in res2["proposer_x_executor"],
              "Read it all（H2、H3、proposer × Executor Interact）")
        t610 = jload(tmp / "C" / "tests_610.json")
        check(sorted(t610) == ["H1", "H2", "H3"] and all("holm_reject" in v for v in t610.values()),
              "6.10 Summary：H1–H3 Do it across configuration Holm")
        print("[ds_alfw selftest] The known answers analysed")
        check(known_answer(), "known-answer test：When the executor prefers the same P2 ≈ 0；Not at the same time P2 Not bad at all；Save best time loss for filtering ≈ 0；Holm")
    finally:
        CFG.clear()
        CFG.update(saved)
        EXTRA_EXEC.clear()
        EXTRA_EXEC.update(saved_extra)
        PROFILE_OUT.clear()
        PROFILE_OUT.update(saved_out)
        PROFILE["name"] = "q7"
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
    rows = analyze_core(T7, {"q14": T14}, lambda ti, b: M[ti][b], n_perm=10)
    same = boot([r["eq_minus_q14"] for r in rows])
    good &= abs(same["mean"]) < 0.05
    T7, T14, M = mk([0.7, 0.1], [0.1, 0.7])
    rows = analyze_core(T7, {"q14": T14}, lambda ti, b: M[ti][b], n_perm=10)
    conf = boot([r["eq_minus_q14"] for r in rows])
    good &= conf["ci"][0] > 0.3
    sl = boot([r["matched"] - r["SLpl_k5"] for r in rows])          # Planner screening leaves a perfect candidate for deployment
    good &= abs(sl["mean"]) < 0.05
    # Holm：p = 0.01, 0.03 The size of the group 3（Third Missing Press 1）-> 0.01 ≤ 0.05/3 reject；0.03 > 0.05/2 No, no, no, no, no, no
    tt = {"a": {"p": 0.01}, "b": {"p": 0.03}}
    holm(tt, ["a", "b"], 3)
    good &= tt["a"]["holm_reject"] and not tt["b"]["holm_reject"]
    print(f"    Same preference P2 {same['mean']:+.3f}；Different preferences P2 {conf['mean']:+.3f} [{conf['ci'][0]:+.3f}, {conf['ci'][1]:+.3f}]；"
          f"SL-planner loss {sl['mean']:+.3f}；Holm {tt['a']['holm_reject']}/{tt['b']['holm_reject']}")
    return good


# ================================================================ The entrance
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "prep7", "prep14", "run7", "prep", "run", "p610", "collect", "positions", "propose",
                             "design", "eval", "pack", "unpack", "analyze", "selftest"])
    ap.add_argument("--profile", choices=["q7", "l8"], default="q7",
                    help="q7：6.9（Deployment 7B）；l8：6.10 Second deployment model Llama-8B")
    ap.add_argument("--proposer", choices=["q7", "q14", "qflash", "l8"])
    ap.add_argument("--executor", choices=["q7", "q14", "l8"])
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
    apply_profile(args.profile, smoke=STATE["smoke"])
    use_out(args.out_dir or (acfg.ROOT / (OUT_NAMES[args.profile] + ("_smoke" if STATE["smoke"] else ""))))
    if not args.mock:
        try:
            import selftest as grid_selftest
            grid_selftest.t0_manifest()
        except AssertionError as e:
            log(f"Code version check failed：{e}")
            return 1
    log(f"===== ds_alfw --profile {args.profile} --stage {args.stage}  PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} "
        f"output {STATE['out']} =====")
    try:
        if args.stage == "smoke":
            rc = pipeline(deployer_steps(CFG, smoke=True) + [("eval", ["--executor", deployer()])])
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
        if args.stage == "prep":
            return pipeline(deployer_steps(CFG))
        if args.stage == "run":
            return pipeline([("eval", ["--executor", n]) for n in executors()] + [("analyze", [])])
        if args.stage == "p610":
            need = [Path(PROFILE_OUT["q7"]) / f for f in ("design.json", "positions.json", "evals_q7.jsonl")]
            if not all(f.exists() for f in need):        # ① requires 6.9 Design and 7B continuation；Check it out first，Dont worry about it ② discovering this only after a long run
                sys.exit(f"[p610] {PROFILE_OUT['q7']} does not contain 6.9 Outcome（{[f.name for f in need if not f.exists()]}）；"
                         f"first cp -r ../repair_transfer/out_ds_alfw ./")
            # ② Llama-8B Deployment：Ready -> Llama-8B、7B continuation -> Analysis；then ① Llama-8B at 6.9 Designed to be an executor -> Analysis 6.9
            #（Both analyses will be updated 6.10 Summary）
            ph = PROFILES["l8"]
            steps = [(st, ex, "l8") for st, ex in deployer_steps(ph)]
            steps += [("eval", ["--executor", n], "l8") for n in ph["N_EXEC"]] + [("analyze", [], "l8")]
            steps += [("eval", ["--executor", "l8"], "q7"), ("analyze", [], "q7")]
            return pipeline(steps)
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
