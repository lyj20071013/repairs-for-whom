"""WebShop formal experiments (analysis plan section 6.13 and amendments).

Profiles:
  A: Qwen7B deployment, out_ws_A; Qwen7B/Llama 10+1 continuations,
     with Qwen14B 20+1 added when available.
  B: Llama deployment, out_ws_B; Llama/Qwen7B 10+1 continuations.
  S: pipeline smoke test, out_ws_S; 30 goals, 3 failures, Qwen7B/flash
     proposals, and Qwen7B 2+1 continuations per stochastic branch.
Here n+1 means n stochastic runs and one independent greedy run.

Run from the experiment root in the inference environment; WebShop runs in
a subprocess with its own environment:
    python webshop/ws_formal.py check --profile A
    python webshop/ws_formal.py all --profile A
    python webshop/ws_formal.py <stage> --profile A [--model q7|l8|qflash|q14]
Stages: collect, positions, propose, design, eval, analyze, summary.
The selftest stage uses synthetic data and mock models.
Each stage saves progress. Compatible interrupted work resumes on rerun.
Only one process may use an output directory. Flash reads QWEN_API_KEY
from the environment.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from alfw import acfg                                     # noqa: E402
from webshop import wcfg, ws_agent, ws_pool, ws_rollout   # noqa: E402

LABEL = {"q7": "7B", "q14": "14B", "l8": "Llama-8B", "qflash": "flash", "planner": "privileged action", "random": "random",
         "exhaustive": "enumeration"}
MODEL_DESC = {"q7": "a 7B language model", "q14": "a 14B language model", "l8": "an 8B language model"}
EXEC_CODE = {"q7": 1, "q14": 2, "l8": 3}
BATCH = "WS1"
CFG = {"N_TRAJ": 60, "N_POS": 3, "MIN_LEFT": 5, "N_PROPOSE": 2, "BLOCK": 100, "STOP_FAILS": 70,
       "OBS_LOG": 300, "OBS_PAGE": 3000}
PROFILES = {
    "A": {"deployer": "q7", "seed": 4001, "max_goals": 400, "prefix": "a", "out": "out_ws_A",
          "executors": {"q7": 10, "l8": 10, "q14": 20}, "proposers": ["q7", "l8", "qflash"],
          "tests": [("P1w", "matched-greedy", "Match − 7B greedy"),
                    ("H1w", "eq_minus_l8", "Same thing 5 runs：7B select using the deployed executor − by Llama-8B Selection"),
                    ("P2w", "eq_minus_q14", "Same thing 5 runs：7B select using the deployed executor − by 14B Selection")]},
    "B": {"deployer": "l8", "seed": 4002, "max_goals": 300, "prefix": "b", "out": "out_ws_B",
          "executors": {"l8": 10, "q7": 10}, "proposers": ["q7", "l8", "qflash"],
          "tests": [("H2w", "eq_minus_q7", "Same thing 5 runs：Llama-8B select using the deployed executor − by 7B Selection"),
                    ("H3w", "matched-greedy", "Match − Llama-8B greedy")]},
    "S": {"deployer": "q7", "seed": 4101, "max_goals": 30, "prefix": "s", "out": "out_ws_S",
          "executors": {"q7": 2}, "proposers": ["q7", "qflash"], "tests": [],
          "n_traj": 3, "block": 30, "stop_fails": 3},
}
FAMILY = ["P1w", "H1w", "P2w", "H2w", "H3w"]      # Holm，One side α = 0.05，m = 5（missing results use p = 1）
SMOKE_GATES = {"max_invalid_frac": 0.20, "min_valid_rate": 0.50}
STATE = {"mock": False, "profile": "A", "out": None, "python": None, "webshop_dir": None, "stubs": None,
         "over": {}, "stop_after": None}


# ================================================================ Foundation
def prof():
    p = dict(PROFILES[STATE["profile"]])
    p.update(STATE["over"])
    return p


def C(key):
    p = prof()
    return p.get({"N_TRAJ": "n_traj", "BLOCK": "block", "STOP_FAILS": "stop_fails"}.get(key, "_"), CFG.get(key))


def out():
    return STATE["out"] or ROOT / prof()["out"]


def P(name, x=None):
    o = out()
    return {"episodes": o / "episodes.jsonl", "meta": o / "episodes_meta.json", "positions": o / "positions.json",
            "proposals": o / f"proposals_{x}.jsonl", "pcache": o / f"propose_cache_{x}.jsonl", "design": o / "design.json",
            "evals": o / f"evals_{x}.jsonl", "timing": o / f"timing_{x}.json", "analysis": o / "analysis.json",
            "report": o / "report.md", "log": o / "ws_formal.log", "workers": o / "worker_logs",
            "summary": ROOT / "out_ws_summary"}[name]


def log(msg):
    print(msg, flush=True)
    try:
        out().mkdir(parents=True, exist_ok=True)
        with open(P("log"), "a", encoding="utf-8") as f:
            f.write(time.strftime("[%m-%d %H:%M:%S] ") + msg + "\n")
    except OSError:
        pass


def fp(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def jdump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")
    tmp = Path(f"{path}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if tmp.read_bytes() != data:
        raise OSError(f"Write {path} Read later inconsistently（The disks full？）")
    os.replace(tmp, path)


def jload(path):
    raw = Path(path).read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RuntimeError(f"{path} incomplete JSON（{len(raw)} Bytes，Start {raw[:40]!r}）：{e}") from None


def jl_read(path):
    out_ = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out_.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                               # Half the last line at the break
    return out_


def jl_write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def derive_seed(*parts):
    return ws_agent.derive_seed(*parts)


def max_active():
    """Number of repeat strips advanced at the same time：WS_MAX_ACTIVE Changeable（It only affects speed；Each step of a seed sampled by（continuation，Step）Sure）。"""
    return int(os.environ.get("WS_MAX_ACTIVE", wcfg.MAX_ACTIVE))


def code_sha():
    # The code that decides to repeat the result（ws_formal.py Not in itself：analysis-code changes do not require new continuations；Seeds and prefixes for renewal are different design_id Fixed）
    fs = ["webshop/ws_agent.py", "webshop/ws_rollout.py", "webshop/ws_worker.py", "webshop/ws_pool.py", "webshop/wcfg.py",
          "webshop/ws_data.py"]
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16] for f in fs}


def settings():
    """All Settings for Determining Results（Uncoded）：If you change it, you have to change the output directory。"""
    p = prof()
    return {"profile": STATE["profile"], "deployer": p["deployer"], "seed": p["seed"], "max_goals": p["max_goals"],
            "n_traj": C("N_TRAJ"), "n_pos": CFG["N_POS"], "min_left": CFG["MIN_LEFT"], "n_propose": CFG["N_PROPOSE"],
            "proposers": p["proposers"], "block": C("BLOCK"), "stop_fails": C("STOP_FAILS"),
            "max_steps": wcfg.MAX_STEPS, "num_products": wcfg.NUM_PRODUCTS, "env_seed": wcfg.ENV_SEED,
            "obs": [wcfg.OBS_MODE, wcfg.OBS_CHARS_CURRENT, wcfg.OBS_CHARS_HISTORY], "temperature": acfg.TEMPERATURE,
            "mock": STATE["mock"]}


def take_lock():
    import fcntl
    o = out()
    env_key = "WS_FORMAL_LOCK"
    if os.environ.get(env_key) == str(o):
        return None
    o.mkdir(parents=True, exist_ok=True)
    f = open(o / ".ws_formal.lock", "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.seek(0)
        return f.read().strip() or "unknown"
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()}：{' '.join(sys.argv[1:])}（{time.strftime('%m-%d %H:%M:%S')} Action）")
    f.flush()
    STATE["_lock"] = f
    os.environ[env_key] = str(o)
    return None


def _ds():
    """ds_alfw：Model Backend、Awaiting、Release vLLM、Proposer。Its logbook is written in this output directory ds.log。"""
    import ds_alfw as DS
    if STATE.get("_ds_out") != out():
        DS.use_out(out())
        DS.STATE["mock"] = STATE["mock"]
        STATE["_ds_out"] = out()
    return DS


def make_pool(n=None):
    return ws_pool.WSPool(n_workers=n, log_dir=P("workers"), python=STATE["python"], webshop_dir=STATE["webshop_dir"],
                          stubs=STATE["stubs"])


def make_backend(name):
    if STATE["mock"]:
        return ws_agent.MockBackend()
    acfg.MODEL_CONTEXT = wcfg.MODEL_CONTEXT          # WebShop use 16,384（6.13 Revision 2），Change values only in this process
    return _ds().make_exec_backend(name)


def release(obj):
    if not STATE["mock"] and obj is not None:
        _ds().release_vllm(obj)


def gpu_ready(stage):
    return STATE["mock"] or _ds().wait_gpu(stage)


# ================================================================ Allocation of objectives（independent of collection order、Sure）
MOCK_SPLIT = {"smoke": 10, "A": 40, "B": 30, "S": 30}     # synthetic data for self-tests only（120 Target）


def goal_split(n_goals):
    """smoke test 60 items → A Order of the（Random(4001)，Prior 400）→ B Order of the（Random(4002)，Prior 300）→ S（Random(4101)，before 30）。
    independent of collection progress、Absolutely：There is no overlap between the objectives of the configurations，also exclude smoke-test goals。"""
    n_smoke = MOCK_SPLIT["smoke"] if STATE["mock"] else wcfg.SMOKE["N_GOALS"]
    smoke = set(random.Random(wcfg.SMOKE["SEED"]).sample(range(n_goals), min(n_smoke, n_goals)))
    rest = [g for g in range(n_goals) if g not in smoke]
    order = {}
    for name in ("A", "B", "S"):
        pr = PROFILES[name]
        r = list(rest)
        random.Random(pr["seed"]).shuffle(r)
        order[name] = r[:MOCK_SPLIT[name] if STATE["mock"] else pr["max_goals"]]
        used = set(order[name])
        rest = [g for g in rest if g not in used]
    return order


# ================================================================ check
def stage_check():
    ok = [True]

    def line(c, msg):
        ok[0] = ok[0] and bool(c)
        print(("  ✓ " if c else "  ✗ ") + msg, flush=True)
    p = prof()
    print(f"[ws_formal check] Configure {STATE['profile']}：deployed executor {LABEL[p['deployer']]}；executor "
          f"{ {LABEL[k]: v for k, v in p['executors'].items()} }；proposer {[LABEL[x] for x in p['proposers']]}；"
          f"Top of the game {wcfg.MAX_STEPS} steps；output {out()}", flush=True)
    try:
        with make_pool(1) as pl:
            h = pl.hello()[0]
        order = goal_split(h["n_goals"])
        line(True, f"WebShop：{h['n_goals']} Target、{h['n_products']} Commodities；The target of this configuration {len(order[STATE['profile']])} items"
                   f"（Searching fingerprints {h['search_fp']['hash']}）")
    except Exception as e:                           # noqa: BLE001
        line(False, f"WebShop Subprocess start failed：{e}")
    if not STATE["mock"]:
        DS = _ds()
        for m in sorted(set(p["executors"]) | {p["deployer"]} | {x for x in p["proposers"] if x != "qflash"}):
            d = DS.model_dir(m)
            line(Path(d).exists() or m == "q14", f"model {LABEL[m]}：{d}" + ("（14B Late，at 14B The servers running）" if m == "q14" else ""))
        if "qflash" in p["proposers"]:
            line(bool(os.environ.get("QWEN_API_KEY")), "flash of API Key（Environmental variables QWEN_API_KEY；Do Not Print Contents）")
        print(f"  · Portfolio free memory ratio {DS.gpu_free_frac()}", flush=True)
    print("[ws_formal check] " + ("ready" if ok[0] else "issues found，see above ✗"), flush=True)
    return 0 if ok[0] else 1


# ================================================================ collect：Deployors are running on new targets 1 episodes
def _ep_row(e):
    pol = [s for s in e["steps"]]
    return {"rid": e["rid"], "goal": e["goal"], "seed": e["seed"], "instruction": e["instruction"],
            "cmds": [s["cmd"] for s in pol], "obs": [s["obs"] for s in pol], "kinds": [s["kind"] for s in pol],
            "valid": [bool(s["valid"]) for s in pol], "pages": e["pages"], "done": e["done"], "reward": e["reward"],
            "won": e["won"], "n_actions": e["n_actions"]}


def stage_collect():
    p = prof()
    D = p["deployer"]
    with make_pool(1) as pl:
        n_goals = pl.hello()[0]["n_goals"]
    goals = goal_split(n_goals)[STATE["profile"]]
    specs = [{"rid": f"{p['prefix']}{i:03d}", "goal": g, "seed": derive_seed("WS-COLLECT", STATE["profile"], g)}
             for i, g in enumerate(goals)]
    key = fp({"specs": [[s["rid"], s["goal"], s["seed"]] for s in specs], "s": settings()})
    meta = jload(P("meta")) if P("meta").exists() else {}
    if meta and meta.get("key") != key:
        sys.exit("[collect] The collection is different from the current settings；Please change the output directory")
    if meta.get("complete"):
        log(f"[collect] Reuse already available {meta['n_collected']} episodes")
        return 0
    rows = sorted(jl_read(P("episodes")), key=lambda r: r["rid"])
    done = len(rows)
    if [r["rid"] for r in rows] != [s["rid"] for s in specs[:done]]:
        sys.exit("[collect] Existing stations are not prefixes to the sampling order；Please change the output directory")
    timings = meta.get("timings", [])
    log(f"[collect] deployed executor {LABEL[D]}；Available objectives {len(goals)} items（The smoke and other configuration targets have been removed）；Each piece {C('BLOCK')} items，"
        f"Failed to {C('STOP_FAILS')} No more rules（Existing {done} episodes）")
    be = None
    try:
        with make_pool() as pl:
            while sum(not r["won"] for r in rows) < C("STOP_FAILS") and done < len(specs):
                batch = specs[done:done + C("BLOCK")]
                if be is None:
                    if not gpu_ready(f"collect-{D}"):
                        return 1
                    be = make_backend(D)
                got = {}
                tm = ws_rollout.run_stream(be, pl, batch, acfg.TEMPERATURE, lambda e: got.__setitem__(e["rid"], e),
                                           max_active=max_active(), max_steps=wcfg.MAX_STEPS, log=log)
                rows += [_ep_row(got[s["rid"]]) for s in batch]
                done += len(batch)
                timings.append({k: tm[k] for k in ("wall", "policy_steps", "gen", "env", "max_prompt_tokens", "retries")})
                jl_write(P("episodes"), rows)
                jdump(P("meta"), {"key": key, "complete": False, "timings": timings, "n_collected": done})
                n_f = sum(not r["won"] for r in rows)
                log(f"[collect] Picked {done} episodes：Success {done - n_f}，failed {n_f}（{n_f / done:.0%}）")
    finally:
        release(be)
    jdump(P("meta"), {"key": key, "complete": True, "timings": timings, "n_collected": done})
    n_f = sum(not r["won"] for r in rows)
    log(f"[collect] complete：{done} episodes，Success {done - n_f}，failed {n_f}" +
        ("" if n_f >= C("STOP_FAILS") else f"（Not yet {C('STOP_FAILS')} records：no available goals remain）"))
    return 0


def load_episodes():
    return {e["rid"]: e for e in jl_read(P("episodes"))}


# ================================================================ positions：Failed in order，Organisation（Dont look at any repair signals）
def _strata(elig, k, rng):
    if len(elig) <= k:
        return sorted(elig)
    parts = np.array_split(np.array(sorted(elig)), k)
    return sorted(int(rng.choice(list(x))) for x in parts if len(x))


def stage_positions():
    if P("positions").exists():
        d = jload(P("positions"))
        if d.get("settings") == settings():
            log(f"[positions] Reuse already available {len(d['trajectories'])} trajectories")
            return 0
        sys.exit("[positions] The result is different from the current settings；Please change the output directory")
    meta = jload(P("meta")) if P("meta").exists() else {}
    if not meta.get("complete"):
        sys.exit("[positions] collection is incomplete（Run first collect）")
    eps = load_episodes()
    fails = [eps[k] for k in sorted(eps) if not eps[k]["won"]]
    tmax = wcfg.MAX_STEPS - CFG["MIN_LEFT"]
    log(f"[positions] failed {len(fails)} records；Take the front in order {C('N_TRAJ')} records（Dont look at any repair signals）；position t ≤ {tmax}")
    trajs, excluded, items, keys = [], [], [], []
    for e in fails:
        if len(trajs) >= C("N_TRAJ"):
            break
        elig = [t for t in range(1, len(e["cmds"]) + 1) if t <= tmax]
        if not elig:
            excluded.append({"traj_id": e["rid"], "reason": "no eligible position"})
            continue
        pos = _strata(elig, CFG["N_POS"], random.Random(derive_seed("WS-POS", e["rid"])))
        trajs.append({"traj_id": e["rid"], "goal": e["goal"], "T": len(e["cmds"]), "n_eligible": len(elig),
                      "reward": e["reward"], "done": e["done"], "positions": [{"t": t} for t in pos]})
        for t in pos:
            items.append({"goal": e["goal"], "prefix": e["cmds"][:t - 1], "priv": True})
            keys.append((e["rid"], t))
    with make_pool() as pl:
        res = dict(zip(keys, pl.probe(items)))
    n_mis = 0
    for tr in trajs:
        e = eps[tr["traj_id"]]
        for p in tr["positions"]:
            r = res[(tr["traj_id"], p["t"])]
            st, t = r["state"], p["t"]
            ok = True if t == 1 else ws_agent.clean_obs(st["obs"], r["instruction"]) == e["obs"][t - 2]
            n_mis += int(not ok)
            p.update(orig=e["cmds"][t - 1], orig_valid=e["valid"][t - 1], page=st["page"], search=bool(st["search"]),
                     clicks=list(st["clicks"]), obs=ws_agent.clean_obs(st["obs"], r["instruction"]),
                     priv=r.get("priv"), replay_ok=ok)
    d = {"trajectories": trajs, "excluded": excluded, "n_fail": len(fails), "n_episodes": len(eps),
         "settings": settings(), "n_replay_mismatch": n_mis}
    d["positions_id"] = fp({"t": [{k: v for k, v in tr.items()} for tr in trajs], "s": d["settings"]})
    jdump(P("positions"), d)
    log(f"[positions] complete {d['positions_id'][:12]}：{len(trajs)} trajectories（Exclude {len(excluded)}），position "
        f"{sum(len(t['positions']) for t in trajs)} items；Replaying does not match the page collected {n_mis} Branch（For 0）")
    return 0 if n_mis == 0 else 1


# ================================================================ propose
PROPOSE_SYSTEM = ("You are an expert debugger of AI agents. You will read the log of a failed run of an agent in a "
                  "web shop and propose alternative actions for one step of the run.")
_PROP_RE = re.compile(r"^\s*\**\s*action\s*([12])\s*\**\s*[:：]\s*\**[ \t]*(.*)$", re.I | re.M)


def _cut(s, n):
    return s if len(s) <= n else s[:n].rstrip() + " ..."


def outcome_text(e):
    if e["done"]:
        return (f"the agent bought a product, which only partly matched the instruction (reward {e['reward']:.2f} "
                "out of 1.00)")
    return "the agent ran out of steps without buying anything"


def propose_messages(e, p, k, deployer):
    lines = [f"Step {i}: > {c or '(no action)'}\n{_cut(o, CFG['OBS_LOG'])}"
             for i, (c, o) in enumerate(zip(e["cmds"], e["obs"]), start=1)]
    t, orig = p["t"], p["orig"]
    acts = (["search[<keywords>]  (replace <keywords> with your own search query)"] if p["search"] else []) + \
        [f"click[{c}]" for c in p["clicks"]]
    adm = "\n".join(f"- {a}" for a in acts)
    user = ("The agent shops in a text-based web store. At every step it sees the instruction, its history of actions "
            "with the resulting pages, the current page, and the list of actions possible right now, and it picks one of "
            "them. \"search[...]\" is possible only on the search page; \"click[...]\" clicks a button, a product ID or a "
            "product option (size, color, ...). \"click[buy now]\" buys the product with the options selected so far and "
            "ends the run; the run succeeds only if the product matches the instruction, with the right options and within "
            f"the price limit. An action that is not possible does nothing but uses up a step. The step budget is "
            f"{wcfg.MAX_STEPS} steps.\n\n"
            f"The agent is {MODEL_DESC[deployer]} that samples its actions at temperature 0.7.\n\n"
            f"Instruction: {e['instruction']}\n\n"
            f"Log of the failed run ({len(lines)} steps; in the end {outcome_text(e)}):\n" + "\n".join(lines)
            + f"\n\nWe consider replacing the agent's action at step {t}, which was: \"{orig}\".\n"
              f"The page at step {t}:\n{_cut(p['obs'], CFG['OBS_PAGE'])}\n\n"
              f"Possible actions at step {t}:\n{adm}\n\n"
              f"Propose the {k} actions (other than \"{orig}\") that, if taken at step {t} instead, are most likely to let "
              "the SAME agent succeed when it continues on its own from there with the remaining steps. Copy click actions "
              "exactly as written in the list; for a search, write search[your keywords]. Think briefly, then end with "
              "exactly these lines:\n" + "\n".join(f"Action {i + 1}: <action>" for i in range(k)))
    return [{"role": "system", "content": PROPOSE_SYSTEM}, {"role": "user", "content": user}]


def parse_proposals(text, p, k):
    """Take the last one for each number Action rows；Must be enforceable on this page、Different from the original action、Dont repeat it。"""
    last = {}
    for m in _PROP_RE.finditer(text or ""):
        last[int(m.group(1))] = m.group(2)
    st = {"search": p["search"], "clicks": p["clicks"]}
    acts, bad = [], []
    for i in range(1, k + 1):
        raw = last.get(i)
        if raw is None:
            bad.append({"i": i, "why": "missing"})
            continue
        r = ws_agent.parse_action("Action: " + raw, st)
        if r["kind"] not in ("valid", "alias"):
            bad.append({"i": i, "why": "not_possible", "raw": raw[:80]})
        elif r["cmd"] == p["orig"]:
            bad.append({"i": i, "why": "same_as_original"})
        elif r["cmd"] in acts:
            bad.append({"i": i, "why": "duplicate"})
        else:
            acts.append(r["cmd"])
    return acts, bad


class MockProposer:
    """Offline test：Search page for words in command，Click page to select by torrent from the clickable entry；Small Output Shortcut、Non-executable or repeated。"""

    def __init__(self, name):
        self.name = name
        self.settings = {"backend": "mock", "name": name}

    def ask(self, msgs_list, on_done=None):
        res = []
        for m in msgs_list:
            text = m[-1]["content"]
            rng = random.Random(hashlib.sha1((self.name + text).encode()).hexdigest())
            seg = text.split("Possible actions at step", 1)[1]
            acts = re.findall(r"^- (click\[.*\])$", seg, re.M)
            instr = re.search(r"Instruction: (.+)", text).group(1).lower()
            words = [w for w in re.findall(r"[a-z]+", instr) if len(w) > 3]
            if "search[<keywords>]" in seg:
                picks = [f"search[{' '.join(rng.sample(words, min(3, len(words))) or ['item'])}]" for _ in range(2)]
            else:
                picks = [rng.choice(acts) for _ in range(2)] if acts else ["click[nothing]"] * 2
            u = rng.random()
            body = (f"Action 1: {picks[0]}" if u < 0.08 else f"Action 1: {picks[0]}\nAction 2: click[the moon]" if u < 0.14
                    else f"Action 1: {picks[0]}\nAction 2: {picks[1]}")
            res.append({"text": "Reasoning (mock).\n" + body, "finish_reason": "stop", "error": None})
            if on_done:
                on_done(len(res) - 1, res[-1])
        return res


def make_proposer(name):
    if STATE["mock"]:
        return MockProposer(name)
    return _ds().make_proposer(name)


def stage_propose(name):
    import judge_backends as JB
    pos = jload(P("positions"))
    eps = load_episodes()
    D = prof()["deployer"]
    items = [{"traj_id": t["traj_id"], "t": p["t"], "pos": p,
              "messages": propose_messages(eps[t["traj_id"]], p, CFG["N_PROPOSE"], D)}
             for t in pos["trajectories"] for p in t["positions"]]
    keys = [f"{it['traj_id']}|t{it['t']}" for it in items]
    old = jl_read(P("proposals", name))
    if [f"{r['traj_id']}|t{r['t']}" for r in old] == keys and all(
            r["positions_id"] == pos["positions_id"] and not r["error"] for r in old):
        log(f"[propose {name}] Reuse already available {len(old)} Organisation")
        return 0
    if name != "qflash" and not gpu_ready(f"propose-{name}"):
        return 1
    log(f"[propose {name}] {len(items)} Organisation")
    pr = make_proposer(name)
    try:
        outs = JB.ask_resumable(pr, keys, [it["messages"] for it in items], P("pcache", name))
    finally:
        release(pr)
    rows, n_valid, why = [], 0, Counter()
    for it, o in zip(items, outs):
        acts, bad = parse_proposals((o or {}).get("text"), it["pos"], CFG["N_PROPOSE"])
        n_valid += len(acts)
        why.update(b["why"] for b in bad)
        rows.append({"traj_id": it["traj_id"], "t": it["t"], "proposer": name, "actions": acts, "bad": bad,
                     "text": (o or {}).get("text"), "error": (o or {}).get("error"), "settings": pr.settings,
                     "positions_id": pos["positions_id"]})
    jl_write(P("proposals", name), rows)
    n_err = sum(1 for r in rows if r["error"])
    tot = max(1, len(items) * CFG["N_PROPOSE"])
    log(f"[propose {name}] Valid proposal {n_valid}/{tot}（{n_valid / tot:.0%}）；reason for invalidity {dict(why)}；Error calling {n_err}")
    return 1 if n_err else 0


def load_proposals(name):
    return {(r["traj_id"], r["t"]): r for r in jl_read(P("proposals", name))}


# ================================================================ design：Freezing of candidatures（enumeration ∪ proposal ∪ privileged ∪ random）
def _random_query(instr, rng):
    words = [w for w in re.findall(r"[a-z0-9]+", (instr or "").lower()) if len(w) > 2] or ["item"]
    return "search[" + " ".join(rng.sample(words, min(len(words), rng.randint(2, 3)))) + "]"


def stage_design():
    pos = jload(P("positions"))
    p_ = prof()
    props = {}
    for n in p_["proposers"]:
        if not P("proposals", n).exists():
            sys.exit(f"[design] Missing proposals_{n}.jsonl")
        props[n] = load_proposals(n)
        if {r["positions_id"] for r in props[n].values()} != {pos["positions_id"]}:
            sys.exit(f"[design] proposals_{n}.jsonl From another location file；Please make a new proposal")
    if P("design").exists():
        d = jload(P("design"))
        if d.get("positions_id") == pos["positions_id"] and d.get("settings") == settings():
            log(f"[design] Reuse already designed {d['design_id'][:12]}")
            return 0
        sys.exit("[design] The design is different from the current input；Please change the output directory")
    eps = load_episodes()
    trajs, branches = [], {}
    for tr in pos["trajectories"]:
        e = eps[tr["traj_id"]]
        plist = []
        for p in tr["positions"]:
            t, orig = p["t"], p["orig"]
            cands = {}

            def add(a, src, rank=None):
                c = cands.setdefault(a, {"proposers": [], "planner": False, "random": False, "exhaustive": False,
                                         "flash_rank": None})
                if src in ("planner", "random", "exhaustive"):
                    c[src] = True
                else:
                    if src not in c["proposers"]:
                        c["proposers"].append(src)
                    if src == "qflash" and c["flash_rank"] is None:
                        c["flash_rank"] = rank
            if p["priv"] and p["priv"] != orig:
                add(p["priv"], "planner")
            for n in p_["proposers"]:
                r = props[n].get((tr["traj_id"], t))
                for i, a in enumerate(r["actions"] if r else []):
                    add(a, n, i)
            rng = random.Random(derive_seed("WS-RAND", tr["traj_id"], t))
            clicks = [f"click[{c}]" for c in p["clicks"]]
            if p["search"]:
                for _ in range(20):
                    q = _random_query(e["instruction"], rng)
                    if q != orig and q not in cands:
                        add(q, "random")
                        break
            else:
                pool_ = sorted(a for a in clicks if a != orig and a not in cands)
                if pool_:
                    add(rng.choice(pool_), "random")
            for a in clicks:
                if a != orig:
                    add(a, "exhaustive")
            oid = f"{tr['traj_id']}|t{t}|orig"
            branches[oid] = {"branch_id": oid, "traj_id": tr["traj_id"], "goal": tr["goal"], "t": t,
                             "action": "orig:" + orig, "prefix": e["cmds"][:t], "orig": True}
            ids = []
            for a, meta in cands.items():
                bid = f"{tr['traj_id']}|t{t}|{a}"
                branches[bid] = {"branch_id": bid, "traj_id": tr["traj_id"], "goal": tr["goal"], "t": t, "action": a,
                                 "prefix": e["cmds"][:t - 1] + [a], "orig": False, **meta,
                                 "oracle_F": 1.0 if meta["planner"] else 0.0, "oracle_slack": 0.0}
                ids.append(bid)
            plist.append({"t": t, "orig": oid, "alts": ids, "page": p["page"]})
        trajs.append({"traj_id": tr["traj_id"], "goal": tr["goal"], "T": tr["T"], "positions": plist})
    bids = sorted(branches)
    log(f"[design] {len(trajs)} trajectories、{len(branches)} branches；Replay prefixes one by one（The value of buying the branch is certain）")
    with make_pool() as pl:
        pr = pl.probe([{"goal": branches[b]["goal"], "prefix": branches[b]["prefix"]} for b in bids])
    bad = []
    for b, r in zip(bids, pr):
        br, st = branches[b], r["state"]
        if not br["orig"] and not r["valid"][-1]:
            bad.append(b)
        br["oracle_F"] = br.get("oracle_F", 0.0)
        br["oracle_slack"] = br.get("oracle_slack", 0.0)
        br["deterministic"] = (1.0 if st["won"] else 0.0) if st["done"] else None
        br["det_reward"] = float(st["reward"]) if st["done"] else None
    if bad:
        sys.exit(f"[design] {len(bad)} An alternative action cannot be performed when replayed（for example {bad[0]}）：implementation error，Please send the logbook Claude")
    d = {"kind": "ws_formal", "profile": STATE["profile"], "trajectories": trajs, "branches": branches,
         "positions_id": pos["positions_id"], "proposers": p_["proposers"], "settings": settings()}
    d["design_id"] = fp({"t": trajs, "b": {k: {kk: v.get(kk) for kk in ("prefix", "proposers", "planner", "random",
                                                                           "exhaustive", "deterministic")}
                                           for k, v in branches.items()}, "s": d["settings"]})
    jdump(P("design"), d)
    na = [len(p["alts"]) for t in trajs for p in t["positions"]]
    nst = sum(1 for b in branches.values() if b["deterministic"] is None)
    log(f"[design] complete {d['design_id'][:12]}：Random Branch {nst} items、Define Branch {len(branches) - nst} items；Candidature for each position "
        f"{min(na)}–{max(na)} items，Average {np.mean(na):.1f}")
    return 0


# ================================================================ eval：The executor continues（Every Random Branch n runs + greedy 1 runs）
def stochastic(d):
    return sorted(b for b, x in d["branches"].items() if x["deterministic"] is None)


def load_results(path):
    import runner
    return runner.load_results(path) if Path(path).exists() else {}


def missing(d, name, n):
    have = load_results(P("evals", name))
    return [(c, b) for c in (f"{name}_T07", f"{name}_greedy") for b in stochastic(d) if (c, BATCH, b) not in have]


def _run_cond(be_holder, pl, d, name, cond, n, temp):
    import runner
    path = P("evals", name)
    runner._repair_tail(path)
    done = load_results(path)
    sid = fp({"design": d["design_id"], "cond": cond, "batch": BATCH, "n": n, "T": temp, "code": code_sha(),
              "mock": STATE["mock"]})
    for (c, b_, bid), rec in done.items():
        if c == cond and b_ == BATCH and rec.get("spec_id") != sid:
            sys.exit(f"[{cond}] Different settings signed for existing results（Code or design changed）；Please change the output directory")
    todo = [b for b in stochastic(d) if (cond, BATCH, b) not in done]
    log(f"[{cond}] Random Branch {len(stochastic(d))} items，Completed {len(stochastic(d)) - len(todo)}，Stay {len(todo)} × {n} runs")
    if not todo:
        return None
    be = be_holder()
    buf = {}
    f = open(path, "a", encoding="utf-8")

    def specs():
        for b in todo:
            br = d["branches"][b]
            for k in range(n):
                yield {"rid": f"{b}#{k}", "goal": br["goal"], "prefix": br["prefix"], "seed": derive_seed(cond, BATCH, b, k)}

    def on_done(e):
        b, k = e["rid"].rsplit("#", 1)
        st = ws_rollout.episode_stats(e)
        buf.setdefault(b, {})[int(k)] = (bool(e["won"]), float(e["reward"]), st["n"], st["invalid"] + st["no_action"],
                                         st["retries"], st["trunc"], st["drop"])
        if len(buf[b]) == n:
            rows = [buf[b][i] for i in range(n)]
            rec = {"condition": cond, "batch": BATCH, "branch_id": b, "spec_id": sid, "n": n,
                   "results": [r[0] for r in rows], "successes": sum(r[0] for r in rows),
                   "rewards": [r[1] for r in rows], "steps": [r[2] for r in rows], "invalid": [r[3] for r in rows],
                   "retries": [r[4] for r in rows], "trunc": [r[5] for r in rows], "drop": [r[6] for r in rows]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            del buf[b]
            if STATE.get("stop_after") is not None:          # for self-tests only：Simulation stop
                STATE["stop_after"] -= 1
                if STATE["stop_after"] <= 0:
                    raise KeyboardInterrupt("selftest interrupt")
    try:
        tm = ws_rollout.run_stream(be, pl, specs(), temp, on_done, max_active=max_active(), max_steps=wcfg.MAX_STEPS,
                                   log=log, log_every=50)
    finally:
        f.close()
    return dict(tm, rollouts=len(todo) * n)


def stage_eval(name):
    d = jload(P("design"))
    pos = jload(P("positions"))
    if pos["positions_id"] != d["positions_id"]:
        sys.exit("[eval] design.json and positions.json different")
    n = prof()["executors"][name]
    holder = {}

    def backend():
        if "b" not in holder:
            if not gpu_ready(f"eval-{name}"):
                raise SystemExit(1)
            holder["b"] = make_backend(name)
        return holder["b"]
    try:
        with make_pool() as pl:
            for cond, nn, temp in ((f"{name}_T07", n, acfg.TEMPERATURE), (f"{name}_greedy", 1, 0.0)):
                tm = _run_cond(backend, pl, d, name, cond, nn, temp)
                if tm:
                    old = jload(P("timing", name)) if P("timing", name).exists() else {"runs": []}
                    old["runs"].append({"cond": f"{cond}/{BATCH}", "wall": tm["wall"], "policy_steps": tm["policy_steps"],
                                        "rollouts": tm["rollouts"], "gen_s": tm["gen"], "env_s": tm["env"],
                                        "max_prompt_tokens": tm["max_prompt_tokens"],
                                        "gpu": None if STATE["mock"] else _ds().gpu_info(),
                                        "time": time.strftime("%Y-%m-%d %H:%M")})
                    jdump(P("timing", name), old)
    finally:
        release(holder.get("b"))
    left = missing(d, name, n)
    log(f"[eval {name}] " + ("All branches completed" if not left else f"also {len(left)} items（Condition, branch）incomplete，The same order will be replaced by the same order"))
    return 0 if not left else 1


# ================================================================ analyze（Core statistics continue ds_alfw.analyze_core，and ALFWorld 6.9 / 6.10 Same code）
METRICS = ("reward", "success")     # 6.13 Revision 3：The main goal is reward（WebShop Scores），success rate（reward = 1）For secondary readings


def _traj_arrays(d, t, ev, name, n, metric="reward"):
    """metric = reward：The prize for the renewal（Buying the branch thats about to end for that reward）；success：Success（reward = 1）。"""
    cands = [(a, p["orig"], i) for i, p in enumerate(t["positions"]) for a in p["alts"]]
    bids = sorted({b for c in cands for b in c[:2]})
    col = {b: j for j, b in enumerate(bids)}
    br = d["branches"]
    key = "rewards" if metric == "reward" else "results"
    dv = lambda b: br[b]["det_reward"] if metric == "reward" else br[b]["deterministic"]
    det = np.array([np.nan if br[b]["deterministic"] is None else float(dv(b)) for b in bids])
    rand = np.flatnonzero(np.isnan(det))
    S = np.array([ev[(f"{name}_T07", BATCH, bids[j])][key][:n] for j in rand], float).reshape(len(rand), n)
    G = det.copy()
    for j in rand:
        G[j] = float(ev[(f"{name}_greedy", BATCH, bids[j])][key][0])
    return {"cands": cands, "bids": bids, "col": col, "det": det, "rand": rand, "S": S, "G": G,
            "ia": np.array([col[a] for a, _, _ in cands]), "io": np.array([col[o] for _, o, _ in cands])}


def _complete(d, name):
    n = prof()["executors"].get(name)
    return n is not None and P("evals", name).exists() and not missing(d, name, n)


def _subset_readouts(DS, d, keep, have, D, others_names, props, metric):
    d2, empty = DS.restrict_design(d, keep)
    nD = prof()["executors"][D]
    TD2 = [_traj_arrays(d2, t, have[D], D, nD, metric) for t in d2["trajectories"]]
    oth = {o: [_traj_arrays(d2, t, have[o], o, prof()["executors"][o], metric) for t in d2["trajectories"]]
           for o in others_names}
    meta = lambda ti, b: d["branches"][b]
    res = {"n_traj": len(TD2), "n_empty": len(empty)}
    for obj in ("net", "post"):
        R2 = DS.analyze_core(TD2, oth, meta, obj=obj, proposers=props)
        pad = [0.0] * len(empty) if obj == "net" else []
        x = {"matched": DS.boot([r["matched"] for r in R2] + pad),
             "PC": DS.boot([r["matched"] - r["random"] for r in R2] + pad),
             "matched-greedy": DS.boot([r["matched"] - r["greedy"] for r in R2] + pad),
             "frac_best_gt_0.10": float(np.mean([r["best_in_sample"] > 0.10 for r in R2] + [False] * len(empty)))}
        for o in oth:
            x[f"eq_minus_{o}"] = DS.boot([r.get(f"eq_minus_{o}", np.nan) for r in R2] + pad)
        res[obj] = x
    return res


def _readouts(DS, d, have, D, others, props, metric):
    """A target（Incentives or success rates）All readings down：trajectories statistics、Test、Budget curve、Methodology、proposer × executor、Two subsets。"""
    p_ = prof()
    TD = [_traj_arrays(d, t, have[D], D, p_["executors"][D], metric) for t in d["trajectories"]]
    TO = {o: [_traj_arrays(d, t, have[o], o, p_["executors"][o], metric) for t in d["trajectories"]] for o in others}
    meta = lambda ti, b: d["branches"][b]
    res = {}
    for obj in ("net", "post"):
        res[obj] = {"rows": DS.analyze_core(TD, TO, meta, obj=obj, proposers=props)}
    R = res["net"]["rows"]
    diff = lambda a, b: [r[a] - r[b] for r in R]
    tests = {"PC": DS.boot(diff("matched", "random")), "S1w": DS.boot(diff("matched", "planner_earliest"))}
    tests["PC"]["pass"] = tests["PC"]["ci"][0] > 0.10
    for tid, stat, lab in p_["tests"]:
        v = DS.stat_values(R, stat)
        if v is not None:
            tests[tid] = dict(DS.boot(v), label=lab, stat=stat)
    res["tests"] = tests
    curves = {}
    for obj in ("net", "post"):
        RR = res[obj]["rows"]
        c = {f"self_k{k}": DS.boot([r[f"self_k{k}"] for r in RR if f"self_k{k}" in r]) for k in (1, 2, 3, 5)}
        for o in others:
            c.update({f"{o}_k{k}": DS.boot([r[f"{o}_k{k}"] for r in RR if f"{o}_k{k}" in r]) for k in (1, 2, 3, 5, 10, 20)})
            c[f"eq_minus_{o}"] = DS.boot([r.get(f"eq_minus_{o}", np.nan) for r in RR])
            c[f"greedy_{o}"] = DS.boot([r[f"greedy_{o}"] for r in RR])
        c["greedy"] = DS.boot([r["greedy"] for r in RR])
        c["matched"] = DS.boot([r["matched"] for r in RR])
        curves[obj] = c
    res["curves"] = curves
    meth = {}
    for tag in ("SLpl", "SLfl"):
        for k in (1, 2, 3, 5):
            if all(f"{tag}_k{k}" in r for r in R):
                meth[f"{tag}_k{k}"] = {"loss": DS.boot(diff("matched", f"{tag}_k{k}")),
                                       "budget": float(np.mean([k * r[f"budget_{tag}"] for r in R]))}
        meth[f"recall_{tag}"] = float(np.mean([r[f"recall_{tag}"] for r in R]))
    meth["greedy"] = {"loss": DS.boot(diff("matched", "greedy")), "budget": float(np.mean([r["budget_all"] for r in R]))}
    res["method"] = meth
    srcs = list(props) + ["planner", "random"]
    prop = {x: {"self": DS.boot([r[f"prop_self_{x}"] for r in R]),
                **{o: DS.boot([r[f"prop_{o}_{x}"] for r in R]) for o in others}} for x in srcs}
    prop["best_is"] = {x: float(np.mean([r["best_is"][x] for r in R])) for x in srcs}
    for o in others:
        if o in props and D in props:
            prop[f"interaction_{o}"] = DS.boot([(r[f"prop_self_{D}"] - r[f"prop_self_{o}"]) -
                                                (r[f"prop_{o}_{D}"] - r[f"prop_{o}_{o}"]) for r in R])
    res["proposer_x_executor"] = prop
    res["alfworld_style"] = _subset_readouts(DS, d, lambda b: bool(b.get("proposers")) or b.get("planner") or b.get("random"),
                                             have, D, others, props, metric)
    res["model_only"] = _subset_readouts(DS, d, lambda b: bool(b.get("proposers")), have, D, others, props, metric)
    res["frac_traj_best_gain_gt_0.10"] = float(np.mean([r["best_in_sample"] > 0.10 for r in R]))
    for obj in ("net", "post"):
        for r in res[obj]["rows"]:
            r.pop("best_is", None)
    return res


def stage_analyze():
    DS = _ds()
    d = jload(P("design"))
    p_ = prof()
    D = p_["deployer"]
    if not _complete(d, D):
        log(f"[analyze] deployed executor {LABEL[D]} continuations are incomplete")
        return 1
    others = [o for o in p_["executors"] if o != D and _complete(d, o)]
    for o in p_["executors"]:
        if o != D and o not in others:
            log(f"[analyze] {LABEL[o]} or follow-up is missing or incomplete：skip comparisons requiring that executor")
    have = {n: load_results(P("evals", n)) for n in [D] + others}
    props = tuple(p_["proposers"])
    res = {"design_id": d["design_id"], "profile": STATE["profile"], "deployer": D, "others": others,
           "proposers": list(props), "n_traj": len(d["trajectories"]), "mock": STATE["mock"], "primary": "reward"}
    for metric in METRICS:
        res[metric] = _readouts(DS, d, have, D, others, props, metric)
    res["tests"] = res["reward"]["tests"]                     # primary tests（reward）；the corresponding success-metric test is in res["success"]["tests"]
    res["describe"] = describe(d, have)
    sec = {}
    for n in [D] + others:
        if P("timing", n).exists():
            runs = [r for r in jload(P("timing", n))["runs"] if r["cond"] == f"{n}_T07/{BATCH}" and r["rollouts"]]
            if runs:
                sec[n] = {"sec": sum(r["wall"] for r in runs) / sum(r["rollouts"] for r in runs), "gpu": runs[-1]["gpu"]}
    res["sec_per_rollout"] = sec
    jdump(P("analysis"), res)
    P("report").write_text(report(res), encoding="utf-8")
    log(f"[analyze] -> {P('report')}")
    stage_summary()
    return 0


def describe(d, have):
    pos = jload(P("positions"))
    eps = load_episodes()
    out_ = {"n_episodes": pos["n_episodes"], "n_fail": pos["n_fail"], "excluded": len(pos["excluded"]),
            "success_rate": float(np.mean([e["won"] for e in eps.values()])),
            "n_replay_mismatch": pos.get("n_replay_mismatch")}
    na = [len(p["alts"]) for t in d["trajectories"] for p in t["positions"]]
    out_["cands_per_position"] = [int(min(na)), float(np.mean(na)), int(max(na))]
    br = [b for b in d["branches"].values() if not b["orig"]]
    out_["n_branches"] = len(d["branches"])
    out_["n_deterministic"] = sum(1 for b in d["branches"].values() if b["deterministic"] is not None)
    out_["sources"] = {"exhaustive_only": sum(1 for b in br if b["exhaustive"] and not b["proposers"] and not b["planner"]
                                              and not b["random"]),
                       "proposed": sum(1 for b in br if b["proposers"]), "planner": sum(1 for b in br if b["planner"]),
                       "random": sum(1 for b in br if b["random"]),
                       "search_query": sum(1 for b in br if b["action"].startswith("search["))}
    out_["pages"] = dict(Counter(p["page"] for t in d["trajectories"] for p in t["positions"]))
    val = {}
    for n in prof()["proposers"]:
        rows = jl_read(P("proposals", n))
        val[n] = float(np.mean([len(r["actions"]) / CFG["N_PROPOSE"] for r in rows])) if rows else None
    out_["proposal_valid_rate"] = val
    for n, ev in have.items():
        s = [np.mean(v["results"]) for (c, _, _), v in ev.items() if c == f"{n}_T07"]
        rw = [np.mean(v["rewards"]) for (c, _, _), v in ev.items() if c == f"{n}_T07"]
        steps = sum(sum(v["steps"]) for (c, _, _), v in ev.items() if c == f"{n}_T07")
        inv = sum(sum(v["invalid"]) for (c, _, _), v in ev.items() if c == f"{n}_T07")
        out_[f"branch_success_{n}"] = float(np.mean(s)) if s else None
        out_[f"branch_reward_{n}"] = float(np.mean(rw)) if rw else None
        out_[f"invalid_per_step_{n}"] = inv / max(1, steps)
    return out_


def report(res):
    f = lambda b: f"{b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]" if b.get("n") else "—"
    D, others = res["deployer"], res["others"]
    dl = LABEL[D]
    L = [f"# WebShop Official experiments {res['profile']}：Deployment {dl}（analysis plan.md 6.13）", ""]
    if res.get("mock") or res["profile"] == "S":
        L += ["> **Simulate or smoke run**：The numbers dont mean anything。", ""]
    L += [f"Other implementers {[LABEL[o] for o in others] or 'none'}；design {res['design_id'][:12]}；{res['n_traj']} trajectories；Top of the game {wcfg.MAX_STEPS} steps。",
          f"description：{json.dumps(res['describe'], ensure_ascii=False)}", "",
          f"**The main goal is reward**（WebShop Scores，6.13 Revision 3），success rate（reward = 1）a secondary readout。Value = Select the person to be rehabilitated in the deployment {dl} Net gain above"
          f"（Alternative Branch − Original Action Branch；90% trajectory bootstrap interval）；Match = {dl} Myself 5 Subselection、Other 5 Commentary（Crossing）。",
          "The main test Holm decision（Together, two pools，m = 5）See out_ws_summary/report_ws.md。", ""]
    for metric, title in (("reward", "reward（primary）"), ("success", "success rate（secondary）")):
        x = res[metric]
        t = x["tests"]
        L += [f"## {title}", "", "| Read | Contrast | estimate [90%] | Annotations |", "|---|---|---|---|",
              f"| positive control | Match − random | {f(t['PC'])} | {'passed' if t['PC']['pass'] else 'failed'} |"]
        for tid, _, lab in prof()["tests"]:
            L.append(f"| {tid} | {lab} | {f(t[tid]) if tid in t else 'results pending'} | " +
                     (f"p = {t[tid]['p']:.4f}" if tid in t else "—") + " |")
        L += [f"| S1w | Match − Pre-position with privilege action | {f(t['S1w'])} | {'confirmed' if t['S1w']['ci'][0] > 0 else 'not confirmed'} |", ""]
        for key, lab in (("alfworld_style", "ALFWorld Candidates（Model proposal + privileged + random）"), ("model_only", "model-proposed candidates only")):
            y = x[key]
            parts = [f"Match {f(y['net']['matched'])}", f"positive control {f(y['net']['PC'])}",
                     f"Match − greedy {f(y['net']['matched-greedy'])}"]
            parts += [f"Same thing 5 runs {dl} − {LABEL[o]} {f(y['net'][f'eq_minus_{o}'])}" for o in others]
            parts += [f"Net gains available > 0.10 trajectories of Candidates {y['net']['frac_best_gt_0.10']:.2f}"]
            L += [f"**{lab}**（{y['n_traj']} trajectories have candidates，{y['n_empty']} records without、Based on non-repair）：" + "；".join(parts), ""]
        for obj, lab in (("net", "net gain"), ("post", "Corrected value")):
            c = x["curves"][obj]
            L += [f"**Budget curve（{lab}；_Other Organiser {dl} Value）**", "", "| Selection | Value [90%] |", "|---|---|"]
            for k in (1, 2, 3, 5):
                if c[f"self_k{k}"].get("n"):
                    L.append(f"| {dl} {k} runs | {f(c[f'self_k{k}'])} |")
            L.append(f"| {dl} greedy | {f(c['greedy'])} |")
            for o in others:
                for k in (1, 2, 3, 5, 10, 20):
                    if c.get(f"{o}_k{k}", {}).get("n"):
                        L.append(f"| by {LABEL[o]} {k} runs | {f(c[f'{o}_k{k}'])} |")
                L.append(f"| by {LABEL[o]} greedy | {f(c[f'greedy_{o}'])} |")
                L.append(f"| Poor：{dl} 5 runs − {LABEL[o]} 5 runs | {f(c[f'eq_minus_{o}'])} |")
            L.append("")
        m = x["method"]
        L += ["**Freezing SL Methodology**（SLpl = Only privileged actions for each location，SLfl = Leave every position flash First valid proposal）", "",
              "| Modalities | budget | Relatively matched loss [90%] |", "|---|---:|---|"]
        for key in sorted(k for k in m if isinstance(m[k], dict)):
            L.append(f"| {key} | {m[key]['budget']:.0f} | {f(m[key]['loss'])} |")
        pr = x["proposer_x_executor"]
        hdr = [f"at {dl} on"] + [f"at {LABEL[o]} on" for o in others]
        L += ["", "**proposer × executor**（Average net gains from candidates from various sources；exploratory）", "",
              "| Source | " + " | ".join(hdr) + f" | yes {dl} Proportion of best candidates |", "|---|" + "---|" * len(hdr) + "---:|"]
        for src in res["proposers"] + ["planner", "random"]:
            L.append(f"| {LABEL.get(src, src)} | " + " | ".join([f(pr[src]["self"])] + [f(pr[src][o]) for o in others]) +
                     f" | {pr['best_is'][src]:.2f} |")
        for o in others:
            if f"interaction_{o}" in pr:
                L += ["", f"Interactive（whether self-proposed candidates suit the proposer better）{dl} × {LABEL[o]}：{f(pr[f'interaction_{o}'])}"]
        L += ["", f"Net gains available > 0.10 Proportion of trajectories to be selected：{x['frac_traj_best_gain_gt_0.10']:.2f}", ""]
    sec = res.get("sec_per_rollout", {})
    if sec:
        L += ["Surviving costs：" + "；".join(f"{LABEL[n]} {v['sec']:.3f} seconds / records" for n, v in sec.items())]
    return "\n".join(L) + "\n"


def stage_summary():
    """The primary tester（P1w、H1w、P2w、H2w、H3w）Cross two pools Holm（m = 5）：Read it A、B Two pool analysis.json。"""
    import ds_alfw as DS
    got = {}
    for name in ("A", "B"):
        fpath = ROOT / PROFILES[name]["out"] / "analysis.json"
        if STATE["profile"] in ("A", "B") and STATE.get("out") and name == STATE["profile"]:
            fpath = P("analysis")
        if fpath.exists():
            a = jload(fpath)
            if a.get("mock") != STATE["mock"]:
                continue
            for tid in FAMILY:
                if tid in a["tests"]:
                    got[tid] = dict(a["tests"][tid], pool=name, pc_pass=bool(a["tests"]["PC"]["pass"]))
                    ts = a["success"]["tests"]
                    if tid in ts:
                        got[tid]["success"] = dict(ts[tid], pc_pass=bool(ts["PC"]["pass"]))
    if not got:
        return 0
    DS.holm(got, sorted(got), len(FAMILY))
    o = STATE.get("summary_out") or P("summary")
    o.mkdir(parents=True, exist_ok=True)
    jdump(o / "tests_ws.json", got)
    f = lambda b: f"{b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]"
    L = ["# WebShop Main Check Summary（Main objective：reward；Holm，One side α = 0.05，m = 5；missing results use p = 1）", ""]
    if STATE["mock"]:
        L += ["> **Simulate Run**：The numbers dont mean anything。", ""]
    L += ["| Test | Ji! | Contrast | estimate [90%] | p | decision |", "|---|---|---|---|---|---|"]
    for tid in FAMILY:
        if tid in got:
            x = got[tid]
            v = "insufficient discriminative signal（the pools positive control failed）" if not x["pc_pass"] else ("confirmed" if x["holm_reject"] else "not confirmed")
            L.append(f"| {tid} | {x['pool']} | {x['label']} | {f(x)} | {x['p']:.4f} | {v} |")
        else:
            L.append(f"| {tid} | — | — | results pending | — | — |")
    L += ["", "Secondary readings：The same test is at the rate of success（reward = 1）Yes（Dont do it Holm，Only for comparison）", "",
          "| Test | estimate [90%] | p | positive control for the pools success metric |", "|---|---|---|---|"]
    for tid in FAMILY:
        if tid in got and "success" in got[tid]:
            y = got[tid]["success"]
            L.append(f"| {tid} | {f(y)} | {y['p']:.4f} | {'passed' if y['pc_pass'] else 'failed'} |")
    (o / "report_ws.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"[summary] -> {o / 'report_ws.md'}（Existing {sorted(got)}）")
    return 0


# ================================================================ smoke-test decision（S）
def smoke_verdict():
    d = jload(P("design"))
    D = prof()["deployer"]
    ev = load_results(P("evals", D))
    steps = sum(sum(r["steps"]) for r in ev.values())
    inv = sum(sum(r["invalid"]) for r in ev.values())
    val = {}
    for n in prof()["proposers"]:
        rows = jl_read(P("proposals", n))
        val[n] = {"valid_rate": float(np.mean([len(r["actions"]) / CFG["N_PROPOSE"] for r in rows])),
                  "errors": sum(1 for r in rows if r["error"])}
    res = {"invalid_per_step": inv / max(1, steps), "proposers": val, "n_traj": len(d["trajectories"]),
           "n_branches": len(d["branches"]), "n_stochastic": len(stochastic(d))}
    res["pass"] = res["invalid_per_step"] <= SMOKE_GATES["max_invalid_frac"] and all(
        v["valid_rate"] >= SMOKE_GATES["min_valid_rate"] and v["errors"] == 0 for v in val.values()) and res["n_traj"] > 0
    jdump(out() / "smoke_verdict.json", res)
    log(f"[smoke] It cant be carried out {res['invalid_per_step']:.1%} / steps；proposer {val}；{res['n_traj']} trajectories、"
        f"{res['n_branches']} branches → {'passed' if res['pass'] else 'failed'}（Look at processes and formats only，Do Not Look at Effects）")
    return 0 if res["pass"] else 1


# ================================================================ all：Every one GPU Phase one subprocess（Release between visible existence）
def sub(stage, extra=()):
    cmd = [sys.executable, str(Path(__file__).resolve()), stage, "--profile", STATE["profile"]] + list(extra)
    if STATE["out"]:
        cmd += ["--out-dir", str(STATE["out"])]
    if STATE["mock"]:
        cmd += ["--mock", "--ws-python", STATE["python"], "--ws-dir", STATE["webshop_dir"], "--stubs", STATE["stubs"]]
    log(f"----- Subprocess：{' '.join(cmd[2:])}")
    rc = subprocess.call(cmd, env=dict(os.environ, PYTHONHASHSEED=acfg.HASH_SEED))
    if rc != 0:
        log(f"----- Subprocess Failed（rc={rc}）：{stage}；after fixing the issue, rerun the same command to resume here")
    return rc


def steps_of(profile):
    p = PROFILES[profile]
    st = [("collect", []), ("positions", [])]
    st += [("propose", ["--model", n]) for n in p["proposers"]]
    st += [("design", [])]
    st += [("eval", ["--model", n]) for n in p["executors"] if n != "q14"]
    st += [("smoke", [])] if profile == "S" else [("analyze", [])]
    return st


def stage_all():
    for stage, extra in steps_of(STATE["profile"]):
        rc = sub(stage, extra)
        if rc:
            return rc
    return 0


# ================================================================ selftest（Claude Workspace：Synthetic Data + mock model）
def selftest():
    from webshop import ws_smoke
    ok = [True]

    def check(c, msg):
        print(("  ✓ " if c else "  ✗ ") + msg, flush=True)
        ok[0] = ok[0] and bool(c)
    tmp = Path(tempfile.mkdtemp(prefix="ws_formal_"))
    saved = dict(STATE)
    try:
        wd = tmp / "WebShop"
        ws_smoke._synthetic_webshop(wd, n_products=120, seed=11)
        os.environ["WS_SEARCH"] = "bm25"
        base = dict(mock=True, python=sys.executable, webshop_dir=str(wd), stubs=str(wcfg.STUBS))
        # ---- Targeting：mutually disjoint、Sure（Officially run figures）
        STATE["mock"] = False
        o1, o2 = goal_split(1021), goal_split(1021)
        sm = set(random.Random(wcfg.SMOKE["SEED"]).sample(range(1021), wcfg.SMOKE["N_GOALS"]))
        check(o1 == o2 and not (set(o1["A"]) & set(o1["B"])) and not (set(o1["A"]) & set(o1["S"]))
              and not (set(o1["B"]) & set(o1["S"])) and not (sm & (set(o1["A"]) | set(o1["B"]) | set(o1["S"])))
              and [len(o1[k]) for k in "ABS"] == [400, 300, 30], "Targeting：smoke test、A、B、S Do not overlap and are certain")
        # ---- parse
        p = {"t": 3, "orig": "click[next >]", "search": False, "clicks": ["back to search", "next >", "b0abc12345", "red"]}
        a, bad = parse_proposals("x\nAction 1: click[B0ABC12345]\nAction 2: click[next >]\nAction 1: click[red]", p, 2)
        check(a == ["click[red]"] and bad[0]["why"] == "same_as_original", "Suggested Parsing：Take one last time、Get rid of the action")
        ps = {"t": 1, "orig": "search[red shoes]", "search": True, "clicks": []}
        a, bad = parse_proposals("Action 1: search[Blue Shoes]\nAction 2: search[red shoes]", ps, 2)
        check(a == ["search[blue shoes]"] and bad[0]["why"] == "same_as_original", "Proposals for search pages：Query lowercase、Remove the original query")
        a, bad = parse_proposals("Action 1: search[<keywords>]\nAction 2: click[red]", ps, 2)
        check(a == [] and [b["why"] for b in bad] == ["not_possible", "not_possible"], "Click on a placeholder or search page：Dismissed")
        # ---- Configure A Small-scale full-scale process（3 trajectories，executor 7B 4 runs、Llama 2 runs）
        STATE.update(base, profile="A", out=tmp / "outA", over={"n_traj": 4, "block": 20, "stop_fails": 6,
                                                                  "executors": {"q7": 4, "l8": 2}})
        STATE["summary_out"] = tmp / "summary"
        check(stage_check() == 0, "check")
        check(stage_collect() == 0, "collect")
        eps = load_episodes()
        n_f = sum(not e["won"] for e in eps.values())
        check(len(eps) % 20 == 0 and n_f >= 6 and all(len(e["cmds"]) <= wcfg.MAX_STEPS for e in eps.values()),
              f"Well collect it by block until weve had enough failure（{len(eps)} episodes、failed {n_f}）")
        check(stage_collect() == 0 and len(load_episodes()) == len(eps), "collect Rerun directly")
        check(stage_positions() == 0, "positions（Replay as page by word）")
        pos = jload(P("positions"))
        tmax = wcfg.MAX_STEPS - CFG["MIN_LEFT"]
        check(len(pos["trajectories"]) == 4 and all(1 <= q["t"] <= tmax for t in pos["trajectories"] for q in t["positions"])
              and all(len(t["positions"]) == min(3, t["n_eligible"]) for t in pos["trajectories"]),
              "Prior 4 Article Failed；Every morning / middle / One position at a time of night，t ≤ 20")
        fails = [k for k in sorted(eps) if not eps[k]["won"]][:4]
        check([t["traj_id"] for t in pos["trajectories"]] == fails, "Failed in order，Dont look at any repair signals")
        check(any(q["priv"] for t in pos["trajectories"] for q in t["positions"]), "privileged action")
        for n in PROFILES["A"]["proposers"]:
            check(stage_propose(n) == 0, f"propose {n}（mock）")
        check(stage_design() == 0, "design")
        d = jload(P("design"))
        br = d["branches"]
        okc = True
        for t in d["trajectories"]:
            ptr = {q["t"]: q for q in next(x for x in pos["trajectories"] if x["traj_id"] == t["traj_id"])["positions"]}
            for q in t["positions"]:
                pp = ptr[q["t"]]
                acts = {br[b]["action"] for b in q["alts"]}
                want = {f"click[{c}]" for c in pp["clicks"]} - {pp["orig"]}
                okc &= want <= acts and pp["orig"] not in acts
                okc &= all(br[b]["prefix"] == eps[t["traj_id"]]["cmds"][:q["t"] - 1] + [br[b]["action"]] for b in q["alts"])
                okc &= br[q["orig"]]["prefix"] == eps[t["traj_id"]]["cmds"][:q["t"]]
                okc &= sum(br[b]["random"] for b in q["alts"]) <= 1
                if pp["search"]:
                    okc &= any(br[b]["planner"] for b in q["alts"]) or pp["priv"] == pp["orig"]
        check(okc, "candidate = enumeration ∪ proposal ∪ privileged ∪ random；correct prefixes；A random candidate for each location")
        nbuy = [b for b in br.values() if b["action"] == "click[buy now]"]
        check(all(b["deterministic"] is not None for b in nbuy) and len(nbuy) > 0, "buy now The value of the branch is certain")
        # continuation：resumed results match an uninterrupted run byte for byte
        STATE["stop_after"] = 3
        try:
            stage_eval("q7")
        except KeyboardInterrupt:
            pass
        STATE["stop_after"] = None
        n_part = len(jl_read(P("evals", "q7")))
        check(stage_eval("q7") == 0, f"eval 7B：resume after interruption（Already at break {n_part} records）")
        recs = sorted(json.dumps({k: v for k, v in r.items()}, sort_keys=True) for r in jl_read(P("evals", "q7")))
        Path(P("evals", "q7")).rename(tmp / "evals_q7_first.jsonl")
        check(stage_eval("q7") == 0, "eval 7B：One run at a time")
        recs2 = sorted(json.dumps({k: v for k, v in r.items()}, sort_keys=True) for r in jl_read(P("evals", "q7")))
        check(recs == recs2, "resumed results match an uninterrupted run byte for byte（Same torrent、Same environment）")
        check(stage_eval("l8") == 0, "eval Llama")
        check(stage_analyze() == 0, "analyze（P2w of 14B Not yet）")
        res = jload(P("analysis"))
        check("P1w" in res["tests"] and "H1w" in res["tests"] and "P2w" not in res["tests"] and res["others"] == ["l8"],
              "Test：P1w、H1w results available，P2w etc 14B")
        check(res["reward"]["alfworld_style"]["n_traj"] >= 1 and "eq_minus_l8" in res["reward"]["alfworld_style"]["net"]
              and "P1w" in res["success"]["tests"], "ALFWorld Subset；Secondary reading of success rate")
        # The array of rewards and success rates：reward = 1 Success is success
        okm = True
        for t in d["trajectories"]:
            Ar = _traj_arrays(d, t, load_results(P("evals", "q7")), "q7", 4, "reward")
            As = _traj_arrays(d, t, load_results(P("evals", "q7")), "q7", 4, "success")
            okm &= bool(np.all((Ar["S"] >= 1.0 - 1e-9) == (As["S"] > 0.5))) and bool(np.all(Ar["S"] >= As["S"] - 1e-9))
        check(okm, "The reward array matches the successful array（reward = 1 ⇔ Success）")
        rep = P("report").read_text(encoding="utf-8")
        check("positive control" in rep and "H1w" in rep and "mock" in rep, "report")
        sm_ = jload(tmp / "summary" / "tests_ws.json")
        check(set(sm_) == {"P1w", "H1w"} and all("holm_threshold" in v for v in sm_.values()), "Cross-pool summary（Holm m = 5）")
        # ---- known-answer test：An alternative branch was all worked out、The rest of them all failed、Mistakes → Match − greedy = 1，Match − Youll have to choose = 1
        DS = _ds()
        dk = json.loads(json.dumps(d))
        evk, evo = {}, {}
        for t in dk["trajectories"]:
            for bb in [x for q in t["positions"] for x in [q["orig"]] + q["alts"]]:
                dk["branches"][bb]["deterministic"] = None
            good = t["positions"][0]["alts"][0]
            wrong = t["positions"][0]["alts"][1] if len(t["positions"][0]["alts"]) > 1 else None
            for q in t["positions"]:
                for bb in [q["orig"]] + q["alts"]:
                    win = bb == good
                    evk[("q7_T07", BATCH, bb)] = {"results": [win] * 10, "rewards": [float(win)] * 10}
                    evk[("q7_greedy", BATCH, bb)] = {"results": [bb == wrong], "rewards": [float(bb == wrong)]}
                    evo[("l8_T07", BATCH, bb)] = {"results": [bb == wrong] * 10, "rewards": [float(bb == wrong)] * 10}
                    evo[("l8_greedy", BATCH, bb)] = {"results": [bb == wrong], "rewards": [float(bb == wrong)]}
        TDk = [_traj_arrays(dk, t, evk, "q7", 10) for t in dk["trajectories"]]
        TOk = {"l8": [_traj_arrays(dk, t, evo, "l8", 10) for t in dk["trajectories"]]}
        Rk = DS.analyze_core(TDk, TOk, lambda ti, b: dk["branches"][b], obj="net", proposers=("q7", "l8", "qflash"))
        check(all(abs(r["matched"] - 1) < 1e-9 and abs(r["greedy"]) < 1e-9 and abs(r["eq_minus_l8"] - 1) < 1e-9
                  for r in Rk), "known-answer test：Match = 1、greedy = 0、Youll have to choose = 0")
        # ---- Lock
        os.environ.pop("WS_FORMAL_LOCK", None)
        check(take_lock() is None, "Get a lock for the output folder")
        r2 = subprocess.run([sys.executable, str(Path(__file__).resolve()), "summary", "--profile", "A", "--out-dir",
                             str(out()), "--mock"], env={k: v for k, v in os.environ.items() if k != "WS_FORMAL_LOCK"},
                            capture_output=True, text=True)
        check(r2.returncode == 2 and "The other one ws_formal" in r2.stdout, "The same output folder cannot run two at once")
        lk = STATE.pop("_lock", None)
        if lk:
            lk.close()
        os.environ.pop("WS_FORMAL_LOCK", None)
        # ---- Smoke Configuration S：all Get out of the way（Subprocess）
        STATE.update(base, profile="S", out=tmp / "outS", over={"block": 30})
        rc = stage_all()
        check(rc == 0 and (tmp / "outS" / "smoke_verdict.json").exists(), "Smoke Configuration S：all（Subprocess）complete the pipeline and report its decision")
    finally:
        lk = STATE.get("_lock")
        if lk:
            lk.close()
        STATE.clear()
        STATE.update(saved)
        os.environ.pop("WS_SEARCH", None)
        os.environ.pop("WS_FORMAL_LOCK", None)
        if os.environ.get("WS_KEEP"):
            print(f"[ws_formal selftest] Keep temporary directory {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ws_formal selftest] {'all passed' if ok[0] else 'failures detected'}")
    return 0 if ok[0] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "collect", "positions", "propose", "design", "eval", "analyze", "summary",
                                    "smoke", "all", "selftest"])
    ap.add_argument("--profile", choices=sorted(PROFILES), default="A")
    ap.add_argument("--model", choices=["q7", "q14", "l8", "qflash"])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--ws-python", default=None)
    ap.add_argument("--ws-dir", default=None)
    ap.add_argument("--stubs", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    STATE.update(mock=a.mock, profile=a.profile, out=a.out_dir.resolve() if a.out_dir else None, python=a.ws_python,
                 webshop_dir=a.ws_dir, stubs=a.stubs)
    if a.cmd == "check":
        return stage_check()
    who = take_lock()
    if who:
        print(f"[ws_formal {a.cmd}] The other one ws_formal Working on it {out()}（{who}）；wait for completion，Or use it pgrep -af ws_formal inspect", flush=True)
        return 2
    log(f"===== ws_formal {a.cmd} --profile {a.profile}{' --model ' + a.model if a.model else ''}  output {out()} =====")
    if a.cmd in ("propose", "eval") and not a.model:
        sys.exit(f"{a.cmd} Yes --model")
    if a.cmd == "eval" and a.model not in prof()["executors"]:
        sys.exit(f"Configure {a.profile} executor is {list(prof()['executors'])}")
    if a.cmd == "propose" and a.model not in prof()["proposers"]:
        sys.exit(f"Configure {a.profile} proposer is {prof()['proposers']}")
    fn = {"collect": stage_collect, "positions": stage_positions, "design": stage_design, "analyze": stage_analyze,
          "summary": stage_summary, "smoke": smoke_verdict, "all": stage_all}
    if a.cmd in fn:
        return fn[a.cmd]()
    if a.cmd == "propose":
        return stage_propose(a.model)
    return stage_eval(a.model)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != acfg.HASH_SEED:
        os.environ["PYTHONHASHSEED"] = acfg.HASH_SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.exit(main())
