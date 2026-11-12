"""第 5 篇先导 P0 的数据阶段（论文方案.md 5.2–5.4、5.6）：

  games      划分游戏：评测局 E（6.9 的 220 局 + valid_seen + valid_unseen + train 另抽 400）、训练局 1,000、冒烟 50
  collect    7B 在训练局各跑 1 次（保存每步原始输出，供锚样本）；7B 在 E 上另跑 1 次（接管来源）       [GPU]
  positions  训练局的全部失败轨迹 / 接管来源的前 200 条失败轨迹：逐步规划，早 / 中 / 晚各取 1 个位置     [CPU]
  propose    --proposer q7（GPU）| qflash（API）：在训练决策点上各提 2 个替代动作
  design     候选 = 规划器 + 7B 2 + flash 2 + 随机 1；逐分支规划（检测已达成的分支）                     [CPU]
  label      7B 对每个分支（含原动作）续跑 K = 3 次                                                     [GPU]
  select     训练集：S（7B 按 ĝ 取前 N 个点）、U（随机 N 点 + 规划器动作）、R（同 U 的点 + 随机候选）、DPO 对、锚样本
  fresh      S 的目标与原动作各用 7B 新鲜续跑 3 次（q̄_S）                                                [GPU]
  thoughts   冻结 7B 在"给出目标动作"的提示下写 Thought（含提示用语的重写，至多 3 次，再不行用模板）     [GPU]
  export     导出各组训练文件（提示 = 部署协议的完整提示；助手 = Thought + Action）                      [CPU]

每个阶段都可断点续跑；同一输出目录里设置或输入变了会拒绝混用。
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from alfw import acfg
from rt_common import (CFG, HASH_69, STATE, P, add_timing, derive_seed, file_sha256, fp, hkey, jdump, jl_read,
                       jl_write, jload, log, make_backend, make_pool, release_vllm, repair_tail, run_specs, settings,
                       shuffle_with, split_games, task_type)
import rt_common as C


# ================================================================ games
def _keep(g):
    return task_type(g) not in CFG["EXCLUDE_TYPES"]


def games_69(train):
    """与上一篇 ds_alfw.stage_collect 完全相同的抽样（先排除阶段 A / B 用过的局）。"""
    from alfw import games
    a = set(games.sample(train, acfg.A2_N_GAMES, acfg.A2_SEED)) | set(games.sample(train, acfg.A4_N_TRAIN, acfg.A4_SEED))
    cand_b = [g for g in train if g not in a and _keep(g)]
    b = set(games.sample(cand_b, acfg.B_N_COLLECT, acfg.B_SEED))
    cand = [g for g in train if g not in (a | b) and _keep(g)]
    return games.sample(cand, CFG["N_69"], CFG["SEED_69"])


def stage_games():
    from alfw import games
    if P("games.json").exists():
        d = jload(P("games.json"))
        if d["settings"] == settings():
            log(f"[games] 复用：评测局 {len(d['E'])}，训练局 {len(d['train'])}，格式检查局 {len(d['fmt'])}")
            return 0
        sys.exit("[games] 已有 games.json 的设置不同；请换输出目录")
    train = split_games("train")
    vs = [g for g in split_games("valid_seen") if _keep(g)]
    vu = [g for g in split_games("valid_unseen") if _keep(g)]
    g69 = games_69(train)
    h = C.hashlib.sha256("\n".join(sorted(g69)).encode()).hexdigest()
    if not STATE["mock"] and h != HASH_69:
        sys.exit(f"[games] 重新抽出的 6.9 游戏与上一篇不同（sha256 {h[:12]} ≠ {HASH_69[:12]}）："
                 f"检查 ALFWORLD_DATA 与 alfw/acfg.py 是否与上一篇相同")
    pool_t = [g for g in train if _keep(g) and g not in set(g69)]
    extra = games.sample(pool_t, CFG["N_E_EXTRA"], CFG["SEED_E_EXTRA"])
    rest = [g for g in pool_t if g not in set(extra)]
    n_tr = 1000 if STATE["profile"] == "smoke" else CFG["N_TRAIN"]     # 冒烟也按登记的 1,000 局划分，冒烟局与 P0 相同
    tr = games.sample(rest, n_tr, CFG["SEED_TRAIN"])
    rest2 = [g for g in rest if g not in set(tr)]
    smoke = games.sample(rest2, CFG["N_SMOKE"], CFG["SEED_SMOKE"])
    n_left = len(rest2) - len(smoke)
    E = [{"g": g, "src": "ds69"} for g in g69] + [{"g": g, "src": "valid_seen"} for g in vs] + \
        [{"g": g, "src": "valid_unseen"} for g in vu] + [{"g": g, "src": "train_extra"} for g in extra]
    if STATE["profile"] == "smoke":       # 冒烟：只用冒烟局（30 训练 + 16 评测 + 最后 6 局格式检查）；正式的局一局不碰
        tr = smoke[:CFG["N_TRAIN"]]
        E = [{"g": g, "src": "smoke"} for g in smoke[CFG["N_TRAIN"]:CFG["N_TRAIN"] + CFG["N_E_SMOKE"]]]
        fmt = smoke[-CFG["N_FORMAT"]:]
    else:
        fmt = smoke
    E = shuffle_with(sorted(E, key=lambda x: x["g"]), CFG["SEED_E_ORDER"])
    d = {"E": E, "train": tr, "fmt": fmt, "hash69": h, "n_train_left_for_main": n_left,
         "counts": {"ds69": len(g69), "valid_seen": len(vs), "valid_unseen": len(vu), "train_extra": len(extra),
                    "train_total_kept": len([g for g in train if _keep(g)])},
         "settings": settings()}
    jdump(P("games.json"), d)
    log(f"[games] 评测局 {len(E)}（{d['counts']}），训练局 {len(tr)}，格式检查局 {len(fmt)}；"
        f"train 中还剩 {n_left} 局可供主实验")
    return 0


def load_games():
    return jload(P("games.json"))


# ================================================================ collect（训练局 + 接管来源）
def _ep_row(e, keep_text):
    pol = e["steps"]
    r = {"rid": e["rid"], "gamefile": e["gamefile"], "seed": e["seed"], "task": e["task"], "desc": e["desc"],
         "won": bool(e["won"]), "won_step": e["won_step"], "task_type": task_type(e["gamefile"]),
         "cmds": [s["cmd"] for s in pol], "obs": [s["obs"] for s in pol], "kinds": [s["kind"] for s in pol],
         "first_kinds": [s.get("first_kind") for s in pol], "retries": [s.get("retries", 0) for s in pol]}
    if keep_text:
        r["texts"] = [s.get("text") for s in pol]
    return r


def _collect_set(name, specs, keep_text, backend_fn, pl, chunk=128):
    from alfw import rollout
    path = P(f"episodes_{name}.jsonl")
    key = fp({"specs": [[s["rid"], s["gamefile"], s["seed"]] for s in specs], "s": settings(),
              "code": [file_sha256(C.ROOT / "alfw" / f) for f in ("agent.py", "rollout.py")]})
    meta_p = P(f"episodes_{name}_meta.json")
    if meta_p.exists() and jload(meta_p)["key"] != key:
        sys.exit(f"[collect {name}] 已有结果的设置不同；请换输出目录")
    jdump(meta_p, {"key": key})
    repair_tail(path)
    have = {r["rid"] for r in jl_read(path)}
    todo = [s for s in specs if s["rid"] not in have]
    log(f"[collect {name}] {len(specs)} 局，已有 {len(have)}，本次 {len(todo)}")
    t0 = time.time()
    for i in range(0, len(todo), chunk):
        eps, tm = rollout.run_episodes(backend_fn(), pl, todo[i:i + chunk], temperature=acfg.TEMPERATURE,
                                       keep_text=keep_text, want_key=False, log=None)
        with open(path, "a", encoding="utf-8") as f:
            for e in eps:
                f.write(json.dumps(_ep_row(e, keep_text), ensure_ascii=False) + "\n")
        rows = jl_read(path)
        log(f"[collect {name}] 已完成 {len(rows)}/{len(specs)}：成功 {sum(r['won'] for r in rows)}，"
            f"失败 {sum(not r['won'] for r in rows)}；本块 {tm['wall'] / 60:.1f} 分钟")
    return time.time() - t0


def stage_collect():
    G = load_games()
    tr = [{"rid": f"t{i:04d}", "gamefile": g, "seed": derive_seed("RT1", g)} for i, g in enumerate(G["train"])]
    tk = [{"rid": f"e{i:04d}", "gamefile": x["g"], "seed": derive_seed("TK", x["g"])} for i, x in enumerate(G["E"])]
    holder = {}

    def backend():
        if "b" not in holder:
            holder["b"] = make_backend()
        return holder["b"]
    try:
        with make_pool() as pl:
            w1 = _collect_set("train", tr, True, backend, pl)
            w2 = _collect_set("tksrc", tk, False, backend, pl)
    finally:
        release_vllm(holder.get("b"))
    add_timing("collect", w1 + w2, {"gpu_stage": True, "n_train": len(tr), "n_tksrc": len(tk)})
    return 0


def load_eps(name):
    return {r["rid"]: r for r in jl_read(P(f"episodes_{name}.jsonl"))}


# ================================================================ positions
def _strata(eligible, k, rng):
    if len(eligible) <= k:
        return sorted(eligible)
    parts = np.array_split(np.array(sorted(eligible)), k)
    return sorted(int(rng.choice(list(p))) for p in parts if len(p))


def probe_all(pl, items, label):
    out, B, t0 = [], 256, time.perf_counter()
    for j in range(0, len(items), B):
        out += pl.probe(items[j:j + B])
        log(f"  [{label}] 规划 {min(j + B, len(items))}/{len(items)}，用时 {time.perf_counter() - t0:.0f}s，"
            f"超时 {sum(r['timeout'] for r in out)}")
    return out


def stage_positions(which):
    """which = train：全部失败轨迹；tk：按评测局顺序取前 TK_N_TRAJ 条有可选位置的失败轨迹。"""
    out = P(f"positions_{which}.json")
    src = "train" if which == "train" else "tksrc"
    if out.exists():
        d = jload(out)
        if d["settings"] == settings():
            log(f"[positions {which}] 复用 {len(d['trajectories'])} 条轨迹")
            return 0
        sys.exit(f"[positions {which}] 已有结果的设置不同；请换输出目录")
    eps = load_eps(src)
    fails = [eps[k] for k in sorted(eps) if not eps[k]["won"]]
    cap = CFG["TK_N_TRAJ"] if which == "tk" else (CFG.get("MAX_POINTS", 10 ** 9) + 2) // 3
    seed_tag = "RT2" if which == "train" else "TK2"
    log(f"[positions {which}] 失败 {len(fails)} 条；最多取 {min(cap, len(fails))} 条有可选位置的（不看任何修复信号）")
    t0 = time.time()
    trajs, excluded = [], []
    with make_pool() as pl:
        i = 0
        while len(trajs) < cap and i < len(fails):
            batch = fails[i:i + max(8, min(64, cap - len(trajs)))]
            i += len(batch)
            items, keys = [], []
            for e in batch:
                for t in range(len(e["cmds"])):
                    items.append({"gamefile": e["gamefile"], "prefix": e["cmds"][:t], "timeout": CFG["PLAN_TIMEOUT"],
                                  "want_adm": True})
                    keys.append((e["rid"], t))
            pr = dict(zip(keys, probe_all(pl, items, f"positions {which}")))
            for e in batch:
                if len(trajs) >= cap:
                    break
                T = len(e["cmds"])
                L = {t: (None if pr[(e["rid"], t)]["timeout"] else len(pr[(e["rid"], t)]["plan"])) for t in range(T)}
                elig = [t for t in range(T) if L[t] is not None and (acfg.MAX_STEPS - t) - L[t] >= CFG["MIN_SLACK"]]
                if not elig:
                    excluded.append({"traj_id": e["rid"], "reason": "没有余量 ≥ 5 的位置"})
                    continue
                pos = _strata(elig, CFG["N_POS"], random.Random(derive_seed(seed_tag, e["rid"])))
                plist = []
                for t in pos:
                    r = pr[(e["rid"], t)]
                    plist.append({"t": t, "orig_action": e["cmds"][t], "admissible": list(r["admissible"]),
                                  "planner": r["plan"][0] if r["plan"] else None, "L": L[t],
                                  "slack": (acfg.MAX_STEPS - t) - L[t]})
                trajs.append({"traj_id": e["rid"], "gamefile": e["gamefile"], "task_type": e["task_type"], "T": T,
                              "n_eligible": len(elig), "positions": plist})
    d = {"trajectories": trajs, "excluded": excluded, "n_fail": len(fails), "n_episodes": len(eps),
         "settings": settings()}
    d["positions_id"] = fp({"t": trajs, "s": d["settings"]})
    jdump(out, d)
    add_timing(f"positions_{which}", time.time() - t0, {"gpu_stage": False})
    log(f"[positions {which}] 完成：{len(trajs)} 条轨迹（排除 {len(excluded)}），位置 {sum(len(t['positions']) for t in trajs)} 个")
    return 0


def train_points():
    """训练决策点：规划器给出下一步、且不同于原动作、且可执行（5.3）。"""
    pos = jload(P("positions_train.json"))
    pts = []
    for tr in pos["trajectories"]:
        for p in tr["positions"]:
            if p["planner"] is not None and p["planner"] != p["orig_action"] and p["planner"] in p["admissible"]:
                pts.append({"pid": f"{tr['traj_id']}|t{p['t']}", "traj_id": tr["traj_id"], "gamefile": tr["gamefile"],
                            **p})
    if "MAX_POINTS" in CFG:
        pts = pts[:CFG["MAX_POINTS"]]
    return pos, pts


# ================================================================ propose（提示与解析与 ds_alfw 相同；去掉 14B 提议者）
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
    def __init__(self, name):
        self.name = name
        self.settings = {"backend": "mock", "name": name}

    def ask(self, msgs_list, on_done=None):
        import hashlib
        out = []
        for m in msgs_list:
            text = m[-1]["content"]
            adm = re.findall(r"^- (.+)$", text.split("Possible actions at step", 1)[1], re.M)
            orig = re.search(r'which was: "(.*?)"', text).group(1)
            rng = random.Random(hashlib.sha1((self.name + text).encode()).hexdigest())
            pool_ = [a for a in adm if a != orig] or adm
            picks = [rng.choice(pool_) for _ in range(2)]
            body = (f"Action 1: {picks[0]}" if rng.random() < 0.1 else f"Action 1: {picks[0]}\nAction 2: {picks[1]}")
            out.append({"text": "Reasoning (mock).\n" + body, "finish_reason": "stop", "error": None})
            if on_done:
                on_done(len(out) - 1, out[-1])
        return out


def make_proposer(name):
    import judge_backends as JB
    if STATE["mock"]:
        return MockProposer(name)
    if name == "q7":
        C.BACKENDS["n"] += 1
        j = JB.VLLMJudge(C.Q7_DIR, context=16384, temperature=0.0, max_tokens=1024, name="q7")
    elif name == "qflash":
        j = JB.make_judge("api", model="qwen3.8-flash", thinking=False)
        j.name = "qflash"
    else:
        raise ValueError(name)
    j.settings["role"] = "proposer"
    return j


def stage_propose(name):
    import judge_backends as JB
    pos, pts = train_points()
    eps = load_eps("train")
    keys = [p["pid"] for p in pts]
    msgs = [propose_messages(eps[p["traj_id"]], p, CFG["N_PROPOSE"]) for p in pts]
    path = P(f"proposals_{name}.jsonl")
    old = jl_read(path)
    if [r["pid"] for r in old] == keys and all(r["positions_id"] == pos["positions_id"] and not r["error"] for r in old):
        log(f"[propose {name}] 复用已有的 {len(old)} 个决策点的提议")
        return 0
    log(f"[propose {name}] {len(pts)} 个决策点")
    t0 = time.time()
    prop = make_proposer(name)
    try:
        outs = JB.ask_resumable(prop, keys, msgs, P("propose_cache") / f"{name}.jsonl")
    finally:
        release_vllm(prop)
    rows, n_valid, why = [], 0, {}
    for p, o in zip(pts, outs):
        acts, bad = parse_proposals((o or {}).get("text"), p["admissible"], p["orig_action"], CFG["N_PROPOSE"])
        n_valid += len(acts)
        for b in bad:
            why[b["why"]] = why.get(b["why"], 0) + 1
        rows.append({"pid": p["pid"], "proposer": name, "actions": acts, "bad": bad, "text": (o or {}).get("text"),
                     "error": (o or {}).get("error"), "usage": (o or {}).get("usage"), "settings": prop.settings,
                     "positions_id": pos["positions_id"]})
    jl_write(path, rows)
    n_err = sum(1 for r in rows if r["error"])
    add_timing(f"propose_{name}", time.time() - t0, {"gpu_stage": name == "q7"})
    log(f"[propose {name}] 有效提议 {n_valid}/{len(pts) * CFG['N_PROPOSE']}；无效原因 {why}；调用出错 {n_err}"
        + ("（出错的没有写入缓存，重跑同一命令会补上）" if n_err else ""))
    return 1 if n_err else 0


# ================================================================ design
def stage_design():
    pos, pts = train_points()
    props = {}
    for n in CFG["PROPOSERS"]:
        rows = jl_read(P(f"proposals_{n}.jsonl"))
        if not rows:
            sys.exit(f"[design] 缺少 proposals_{n}.jsonl")
        if {r["positions_id"] for r in rows} != {pos["positions_id"]} or any(r["error"] for r in rows):
            sys.exit(f"[design] proposals_{n}.jsonl 来自另一份位置文件或有出错的调用；请重跑 propose")
        props[n] = {r["pid"]: r for r in rows}
    if P("design.json").exists():
        d = jload(P("design.json"))
        if d["positions_id"] == pos["positions_id"] and d["settings"] == settings():
            log(f"[design] 复用 {d['design_id'][:12]}")
            return 0
        sys.exit("[design] 已有设计与当前输入不同；请换输出目录")
    eps = load_eps("train")
    points, branches = [], {}
    for p in pts:
        e = eps[p["traj_id"]]
        t, orig, adm, pid = p["t"], p["orig_action"], p["admissible"], p["pid"]
        cands = {}

        def add(a, src):
            c = cands.setdefault(a, {"proposers": [], "planner": False, "random": False})
            if src == "planner":
                c["planner"] = True
            elif src == "random":
                c["random"] = True
            elif src not in c["proposers"]:
                c["proposers"].append(src)
        add(p["planner"], "planner")
        for n in CFG["PROPOSERS"]:
            for a in props[n][pid]["actions"]:
                add(a, n)
        rng = random.Random(derive_seed("RT4", pid))
        pool_ = sorted(a for a in adm if a != orig and a not in cands)
        for a in rng.sample(pool_, min(CFG["N_RANDOM"], len(pool_))):
            add(a, "random")
        oid = f"{pid}|orig"
        branches[oid] = {"bid": oid, "pid": pid, "action": orig, "is_orig": True, "gamefile": p["gamefile"],
                         "prefix": e["cmds"][:t + 1], "planner": False, "proposers": [], "random": False}
        alts = []
        for a, meta in cands.items():
            bid = f"{pid}|{a}"
            branches[bid] = {"bid": bid, "pid": pid, "action": a, "is_orig": False, "gamefile": p["gamefile"],
                             "prefix": e["cmds"][:t] + [a], **meta}
            alts.append(bid)
        points.append({"pid": pid, "traj_id": p["traj_id"], "t": t, "gamefile": p["gamefile"], "orig": oid,
                       "alts": alts, "admissible": adm, "orig_action": orig, "planner_action": p["planner"]})
    log(f"[design] {len(points)} 个决策点、{len(branches)} 个分支；逐分支规划（检测已达成目标的分支）")
    t0 = time.time()
    bids = sorted(branches)
    with make_pool() as pl:
        pr = probe_all(pl, [{"gamefile": branches[b]["gamefile"], "prefix": branches[b]["prefix"],
                             "timeout": CFG["PLAN_TIMEOUT"]} for b in bids], "design")
    for b, r in zip(bids, pr):
        rem = acfg.MAX_STEPS - len(branches[b]["prefix"])
        if r["won"]:
            branches[b].update(deterministic=1.0, oracle_L=0)
        else:
            branches[b].update(deterministic=None, oracle_L=None if r["timeout"] else len(r["plan"]),
                               oracle_F=None if r["timeout"] else float(len(r["plan"]) <= rem))
    d = {"points": points, "branches": branches, "positions_id": pos["positions_id"], "settings": settings()}
    d["design_id"] = fp({"p": points, "b": {k: [v["prefix"], v["deterministic"]] for k, v in branches.items()},
                         "s": d["settings"]})
    jdump(P("design.json"), d)
    add_timing("design", time.time() - t0, {"gpu_stage": False})
    na = [len(p["alts"]) for p in points]
    log(f"[design] 完成 {d['design_id'][:12]}：每点候选 {min(na)}–{max(na)} 个，平均 {np.mean(na):.2f}；"
        f"已达成目标的分支 {sum(b['deterministic'] is not None for b in branches.values())}")
    return 0


def stochastic(d):
    return sorted(b for b, x in d["branches"].items() if x["deterministic"] is None)


def _code_sig():
    return {f: file_sha256(C.ROOT / "alfw" / f)[:16] for f in ("agent.py", "rollout.py", "pool.py", "twenv.py")}


# ================================================================ label / fresh（7B 续跑）
def _branch_specs(d, bids, n, tag):
    out = []
    for b in bids:
        br = d["branches"][b]
        for k in range(n):
            out.append({"rid": f"{b}#{tag}{k}", "bid": b, "k": k, "gamefile": br["gamefile"], "prefix": br["prefix"],
                        "seed": derive_seed(tag, b, k)})
    return out


def stage_label(stop_after=None):
    d = jload(P("design.json"))
    specs = _branch_specs(d, stochastic(d), CFG["K_LABEL"], "LAB")
    sig = fp({"design": d["design_id"], "k": CFG["K_LABEL"], "T": acfg.TEMPERATURE, "code": _code_sig()})
    return _run_gpu("label", specs, P("labels_q7.jsonl"), sig, stop_after)


def _run_gpu(stage, specs, path, sig, stop_after=None, backend_kw=None):
    holder = {}

    def backend():
        if "b" not in holder:
            holder["b"] = make_backend(**(backend_kw or {}))
        return holder["b"]
    t0 = time.time()
    try:
        with make_pool() as pl:
            n, tm = run_specs(backend, pl, specs, path, sig, stop_after=stop_after, label=stage)
    finally:
        release_vllm(holder.get("b"))
    if n:
        add_timing(stage, time.time() - t0, {"gpu_stage": True, "rollouts": n,
                                              "policy_steps": tm["policy_steps"] if tm else None})
    rows = {r["rid"] for r in jl_read(path)}
    left = [s["rid"] for s in specs if s["rid"] not in rows]
    log(f"[{stage}] " + ("全部完成" if not left else f"还有 {len(left)} 条没完成，重跑同一命令会补上"))
    return 0 if not left else 1


def outcomes(d, path, tag, n):
    """{bid: [0/1, ...]}；已达成目标的分支为 n 个 1。"""
    res = {}
    for r in jl_read(path):
        res.setdefault(r["bid"], {})[r["k"]] = float(r["won"])
    out = {}
    for b, x in d["branches"].items():
        if x["deterministic"] is not None:
            out[b] = [1.0] * n
        elif b in res and len(res[b]) == n:
            out[b] = [res[b][k] for k in range(n)]
    return out


# ================================================================ select
def stage_select():
    d = jload(P("design.json"))
    K = CFG["K_LABEL"]
    lab = outcomes(d, P("labels_q7.jsonl"), "LAB", K)
    miss = [b for b in d["branches"] if b not in lab]
    if miss:
        sys.exit(f"[select] 还有 {len(miss)} 个分支没有完整标签；先跑完 label")
    pts = d["points"]
    D = len(pts)
    N = int(round(D * CFG["FRAC_N"]))
    sh = lambda b: float(np.mean(lab[b]))
    info = {}
    for p in pts:
        so = sh(p["orig"])
        sc = {b: sh(b) - so for b in p["alts"]}
        best = max(p["alts"], key=lambda b: (sc[b], derive_seed("RT3", p["pid"], d["branches"][b]["action"])))
        info[p["pid"]] = {"ghat": sc[best], "target": best, "s_orig": so, "scores": sc}
    order = sorted(pts, key=lambda p: (-info[p["pid"]]["ghat"], -derive_seed("RT3", p["pid"])))
    S = [{"pid": p["pid"], "bid": info[p["pid"]]["target"]} for p in order[:N]]
    pids = sorted(p["pid"] for p in pts)
    upts = sorted(random.Random(CFG["SEED_U"]).sample(pids, N))
    byp = {p["pid"]: p for p in pts}
    U = [{"pid": q, "bid": next(b for b in byp[q]["alts"] if d["branches"][b]["planner"])} for q in upts]
    # 注：S-plan 在下面构造（需要 S）；A（只用锚样本）不需要修复样本
    rr = random.Random(CFG["SEED_R"])
    R = [{"pid": q, "bid": rr.choice(sorted(byp[q]["alts"]))} for q in upts]
    planner_bid = lambda q: next(b for b in byp[q]["alts"] if d["branches"][b]["planner"])
    SP = [{"pid": x["pid"], "bid": planner_bid(x["pid"])} for x in S]       # 5.12：S 的点 + 规划器动作
    DPO = [{"pid": x["pid"], "chosen": x["bid"], "rejected": byp[x["pid"]]["orig"]} for x in S]
    # 锚样本：训练局里成功局的、第一次就可执行且没有重答的步（目标 = 7B 当时自己的原始输出）
    eps = load_eps("train")
    cand = []
    for rid in sorted(eps):
        e = eps[rid]
        if not e["won"]:
            continue
        for j, (k, fk, nr, tx) in enumerate(zip(e["kinds"], e["first_kinds"], e["retries"], e["texts"])):
            if k == "valid" and fk == "valid" and nr == 0 and tx:
                cand.append((rid, j))
    if len(cand) < N:
        sys.exit(f"[select] 可作锚样本的步只有 {len(cand)} 个，少于 N = {N}")
    anchors = [{"rid": r, "t": j} for r, j in random.Random(CFG["SEED_ANCHOR"]).sample(cand, N)]

    def val(lst):          # 用标签估计的目标净收益（U、R 无偏；S 是选择用过的续跑，有偏，只作描述）
        return float(np.mean([sh(x["bid"]) - info[x["pid"]]["s_orig"] for x in lst])) if lst else float("nan")
    stats = {"D": D, "N": N, "q_label": {"S_in_sample": val(S), "S-plan_in_sample": val(SP), "U": val(U), "R": val(R)},
             "planner_frac": {g: float(np.mean([d["branches"][x["bid"]]["planner"] for x in L]))
                              for g, L in (("S", S), ("U", U), ("R", R), ("S-plan", SP))},
             "S_target_is_planner_on_S_points": float(np.mean([x["bid"] == y["bid"] for x, y in zip(S, SP)])),
             "overlap_S_U_points": len({x["pid"] for x in S} & set(upts)) / max(N, 1),
             "ghat_cut": info[order[N - 1]["pid"]]["ghat"] if N else None, "n_anchor_candidates": len(cand)}
    ts = {"groups": {"S": S, "U": U, "R": R, "S-plan": SP}, "dpo": DPO, "anchors": anchors, "stats": stats,
          "design_id": d["design_id"], "settings": settings()}
    ts["trainsets_id"] = fp({k: ts[k] for k in ("groups", "dpo", "anchors", "design_id")})
    if P("trainsets.json").exists() and jload(P("trainsets.json"))["trainsets_id"] != ts["trainsets_id"]:
        sys.exit("[select] 已有 trainsets.json 与这次算出的不同（输入变了？）；请换输出目录")
    jdump(P("trainsets.json"), ts)
    log(f"[select] S 的点上 7B 目标恰是规划器动作的比例 {stats['S_target_is_planner_on_S_points']:.2f}")
    log(f"[select] D = {D}，N = {N}；标签估计的目标净收益 {stats['q_label']}；目标是规划器动作 {stats['planner_frac']}；"
        f"S 与 U 点集重合 {stats['overlap_S_U_points']:.2f}；入选的 ĝ 下限 {stats['ghat_cut']}")
    return 0


def stage_fresh(stop_after=None):
    d = jload(P("design.json"))
    ts = jload(P("trainsets.json"))
    orig = {p["pid"]: p["orig"] for p in d["points"]}
    bids = sorted({x["bid"] for g in ("S", "S-plan") for x in ts["groups"][g]} |
                  {orig[x["pid"]] for x in ts["groups"]["S"]})
    bids = [b for b in bids if d["branches"][b]["deterministic"] is None]
    specs = _branch_specs(d, bids, CFG["N_FRESH"], "FRESH")
    sig = fp({"design": d["design_id"], "ts": ts["trainsets_id"], "n": CFG["N_FRESH"], "code": _code_sig()})
    return _run_gpu("fresh", specs, P("fresh_q7.jsonl"), sig, stop_after)


# ================================================================ thoughts
THOUGHT_SUFFIX = ("\n\nThe next action is: {a}\nWrite a short Thought (one or two sentences) explaining, from the "
                  "information above, why this is a good next step. Do not mention that the action was given to you. "
                  "Answer with one line that starts with \"Thought:\".")
LEAK_RE = re.compile(r"(given action|action (?:that )?(?:was|is|has been) given|the given|provided action|"
                     r"\bsuggest(?:ed|ion)|\bhint|told to|been told|was told|instruct|"
                     r"next action is|as specified|specified action|i (?:am|was) asked|we are told|"
                     r"the next action|this action was|action provided)", re.I)


def parse_thought(text):
    """返回 Thought 内容或 None（无法解析 / 太长 / 含提示用语）。"""
    m = re.search(r"thought\s*[:：]\s*(.+)", text or "", re.I)
    if not m:
        return None, "unparsable"
    th = re.split(r"\n|\baction\s*[:：]", m.group(1), flags=re.I)[0].strip().strip('"').strip()
    if not th or len(th) > 400:
        return None, "empty_or_long"
    if LEAK_RE.search(th):
        return None, "leak"
    return th, None


def point_messages(ep, t, adm):
    from alfw.agent import build_messages
    hist = [{"cmd": c, "obs": o} for c, o in zip(ep["cmds"][:t], ep["obs"][:t])]
    return build_messages(ep["task"], ep["desc"], hist, adm, acfg.MAX_STEPS)


def needed_pairs(d, ts):
    pairs = set()
    for g in ts["groups"]:
        for x in ts["groups"][g]:
            pairs.add(x["bid"])
    for x in ts["dpo"]:
        pairs.add(x["rejected"])
    return sorted(pairs)


def thoughts_sig(ts):
    if CFG["THOUGHT_MODE"] == "template":
        return fp({"ts": ts["trainsets_id"], "mode": "template", "tpl": TEMPLATE})
    return fp({"ts": ts["trainsets_id"], "suffix": THOUGHT_SUFFIX, "leak": LEAK_RE.pattern, "tries": CFG["THOUGHT_TRIES"]})


TEMPLATE = "I will {a}."


def load_thoughts(ts):
    sig = thoughts_sig(ts)
    return {r["bid"]: r for r in jl_read(P("thoughts.jsonl")) if r["sig"] == sig}


def stage_thoughts():
    d = jload(P("design.json"))
    ts = jload(P("trainsets.json"))
    eps = load_eps("train")
    byp = {p["pid"]: p for p in d["points"]}
    bids = needed_pairs(d, ts)
    path = P("thoughts.jsonl")
    sig = thoughts_sig(ts)
    have = load_thoughts(ts)
    todo = [b for b in bids if b not in have]
    log(f"[thoughts] 方式 {CFG['THOUGHT_MODE']}；需要 {len(bids)} 个（决策点, 动作），已有 {len(have)}，本次 {len(todo)}")
    if not todo:
        return 0
    if CFG["THOUGHT_MODE"] == "template":         # 5.14：不调用模型，各组修复样本的 Thought 只差动作本身
        repair_tail(path)
        with open(path, "a", encoding="utf-8") as f:
            for b in todo:
                a = d["branches"][b]["action"]
                f.write(json.dumps({"bid": b, "action": a, "thought": TEMPLATE.format(a=a), "template": True,
                                    "n_tries": 0, "tries": [], "sig": sig}, ensure_ascii=False) + "\n")
        log(f"[thoughts] 模板 Thought 写入 {len(todo)} 个")
        return 0
    t0 = time.time()
    be = make_backend()
    try:
        prompts = {}
        for b in todo:
            br = d["branches"][b]
            p = byp[br["pid"]]
            msgs = point_messages(eps[p["traj_id"]], p["t"], p["admissible"])
            msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + THOUGHT_SUFFIX.format(a=br["action"])}
            prompts[b] = be.render(msgs)
        res = {b: {"bid": b, "action": d["branches"][b]["action"], "tries": [], "thought": None} for b in todo}
        left = list(todo)
        for r in range(CFG["THOUGHT_TRIES"] + 1):
            if not left:
                break
            temp = 0.0 if r == 0 else acfg.TEMPERATURE
            outs = be.generate([prompts[b] for b in left], temp, [derive_seed("TH", b, r) for b in left], 160,
                               hints=[byp[d["branches"][b]["pid"]]["admissible"] for b in left])
            nxt = []
            for b, o in zip(left, outs):
                th, why = parse_thought(o.text)
                res[b]["tries"].append({"text": o.text[:600], "why": why})
                if th is None:
                    nxt.append(b)
                else:
                    res[b]["thought"] = th
            left = nxt
        with open(path, "a", encoding="utf-8") as f:
            for b in todo:
                x = res[b]
                x["template"] = x["thought"] is None
                if x["template"]:
                    x["thought"] = f"I will {x['action']}."
                x["n_tries"] = len(x["tries"])
                x["sig"] = sig
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    finally:
        release_vllm(be)
    rows = list(load_thoughts(ts).values())
    add_timing("thoughts", time.time() - t0, {"gpu_stage": True, "n": len(todo)})
    log(f"[thoughts] 完成：用模板的 {sum(r['template'] for r in rows)}/{len(rows)}，"
        f"第一次就可用的 {sum(r['n_tries'] == 1 and not r['template'] for r in rows)}")
    return 0


# ================================================================ export
def completion(thought, action):
    return f"Thought: {thought}\nAction: {action}"


def stage_export():
    d = jload(P("design.json"))
    ts = jload(P("trainsets.json"))
    eps = load_eps("train")
    byp = {p["pid"]: p for p in d["points"]}
    th = load_thoughts(ts)
    miss = [b for b in needed_pairs(d, ts) if b not in th]
    if miss:
        sys.exit(f"[export] 还缺 {len(miss)} 个 Thought；先跑完 thoughts")
    # 锚样本的可执行列表：在环境里重放到该步读出
    with make_pool() as pl:
        items = [{"rid": f"anc{i}", "gamefile": eps[a["rid"]]["gamefile"], "prefix": eps[a["rid"]]["cmds"][:a["t"]],
                  "want_key": False} for i, a in enumerate(ts["anchors"])]
        st = {}
        for j in range(0, len(items), 128):
            st.update(pl.open(items[j:j + 128]))
            pl.close([it["rid"] for it in items[j:j + 128]])
    anchors = []
    for i, a in enumerate(ts["anchors"]):
        e = eps[a["rid"]]
        adm = st[f"anc{i}"]["state"]["admissible"]
        if e["cmds"][a["t"]] not in adm:
            sys.exit(f"[export] 锚样本 {a} 重放后动作不可执行：环境重放不一致")
        anchors.append({"kind": "anchor", "key": f"{a['rid']}|t{a['t']}",
                        "messages": point_messages(e, a["t"], adm), "completion": e["texts"][a["t"]].strip()})

    def repair_row(x, bid_key="bid"):
        p = byp[x["pid"]]
        msgs = point_messages(eps[p["traj_id"]], p["t"], p["admissible"])
        b = x[bid_key]
        return msgs, completion(th[b]["thought"], d["branches"][b]["action"])

    def slots(rep):
        rep = sorted(rep, key=lambda r: C.hkey("SLOT", r["key"]))
        out = []
        for i, (r, a) in enumerate(zip(rep, anchors)):
            out += [dict(r, slot=2 * i), dict(a, slot=2 * i + 1)]
        return out
    files = {}
    for g in ts["groups"]:
        rep = []
        for x in ts["groups"][g]:
            m, c = repair_row(x)
            rep.append({"kind": "repair", "key": x["pid"], "bid": x["bid"], "messages": m, "completion": c})
        files[g] = slots(rep)
    rep = []
    for x in ts["dpo"]:
        m, c = repair_row(x, "chosen")
        _, r = repair_row(x, "rejected")
        rep.append({"kind": "pair", "key": x["pid"], "bid": x["chosen"], "rejected_bid": x["rejected"], "messages": m,
                    "completion": c, "rejected": r})
    files["DPO-S"] = slots(rep)
    files["A"] = [dict(a, slot=2 * i + 1) for i, a in enumerate(anchors)]     # 5.12：只用锚样本（占其他组的锚样本槽位）
    man = {}
    n_slots = 2 * len(anchors)
    for g, rows in files.items():
        jl_write(P(f"train_{g}.jsonl"), rows)
        man[g] = {"n": len(rows), "n_slots": n_slots, "sha256": file_sha256(P(f"train_{g}.jsonl"))}
    jdump(P("train_manifest.json"), {"trainsets_id": ts["trainsets_id"], "files": man})
    log(f"[export] 训练文件：{ {g: v['n'] for g, v in man.items()} }（每组修复样本 + 同一组锚样本，交替排成槽位；A 只有锚样本）")
    return 0


# ================================================================ factcheck（5.12：Thought 的事实依据，按组报告）
FACT_SYSTEM = ("You check whether an agent's stated reasoning is supported by what the agent has actually seen. "
               "Be strict about facts and lenient about guesses.")
FACT_USER = ("Below is exactly what the agent saw before acting: its task, the room description, the history of its "
             "actions with the environment's feedback, and the actions possible now.\n\n=== WHAT THE AGENT SAW ===\n"
             "{seen}\n=== END ===\n\nThe agent then wrote:\nThought: {thought}\nAction: {action}\n\n"
             "Does the Thought state as a fact anything about the environment (where an object is, what a receptacle "
             "contains, what the agent holds, what has already been done) that is NOT supported by what the agent saw? "
             "Statements framed as guesses or possibilities (\"might\", \"likely\", \"may\", \"check whether\") count as "
             "supported. General household knowledge (e.g. apples are often in fridges) counts as supported. "
             "Answer in one or two sentences, then end with exactly one line: Verdict: SUPPORTED or Verdict: UNSUPPORTED")
_VERDICT_RE = re.compile(r"verdict\s*[:：]\s*\**\s*(supported|unsupported)", re.I)


def parse_verdict(text):
    m = _VERDICT_RE.findall(text or "")
    return m[-1].upper() if m else None


class MockFactChecker:
    """模拟：Thought 里出现"is on / is in"这类断言且断言的物体不在可见文本里时判 UNSUPPORTED，其余 SUPPORTED。"""

    def __init__(self):
        self.name = "mock_fact"
        self.settings = {"backend": "mock"}

    def ask(self, msgs_list, on_done=None):
        out = []
        for m in msgs_list:
            u = m[-1]["content"]
            seen = u.split("=== WHAT THE AGENT SAW ===", 1)[1].split("=== END ===", 1)[0]
            th = u.split("Thought: ", 1)[1].split("\nAction:", 1)[0]
            bad = any(x not in seen for x in re.findall(r"([a-z]+ \d+) is (?:on|in)", th))
            out.append({"text": f"Mock.\nVerdict: {'UNSUPPORTED' if bad else 'SUPPORTED'}", "finish_reason": "stop",
                        "error": None})
            if on_done:
                on_done(len(out) - 1, out[-1])
        return out


def fact_items(d, ts, eps):
    """要核查的 Thought：各组修复目标、DPO 的被拒动作（改写的）与锚样本（7B 原始输出）。返回 [(key, 可见文本, thought, action)]。"""
    byp = {p["pid"]: p for p in d["points"]}
    th = load_thoughts(ts)
    items = []
    for b in ([] if CFG["THOUGHT_MODE"] == "template" else needed_pairs(d, ts)):   # 模板 Thought 没有事实断言，不核查
        p = byp[d["branches"][b]["pid"]]
        msgs = point_messages(eps[p["traj_id"]], p["t"], p["admissible"])
        items.append((b, msgs[-1]["content"], th[b]["thought"], d["branches"][b]["action"]))
    for row in jl_read(P("train_A.jsonl")):            # 锚样本：直接用导出的提示与原始输出
        c = row["completion"]
        m = re.search(r"thought\s*[:：]\s*(.+?)(?:\n|$)", c, re.I)
        a = re.search(r"action\s*[:：]\s*(.+?)(?:\n|$)", c, re.I)
        items.append((f"anchor|{row['key']}", row["messages"][-1]["content"], m.group(1).strip() if m else c,
                      a.group(1).strip() if a else ""))
    return items


def fact_summary(ver, ts, keys, thr):
    """按组的"无依据断言"比例（按样本计：同一个 (点, 动作) 在组里出现几次算几次）与超过阈值的组。"""
    def rate(keys_):
        v = [ver.get(k) for k in keys_]
        ok = [x for x in v if x is not None]
        return {"n": len(v), "n_parsed": len(ok),
                "unsupported": (sum(x == "UNSUPPORTED" for x in ok) / len(ok)) if ok else None}
    summ = {g: rate([x["bid"] for x in ts["groups"][g]]) for g in ts["groups"]}
    summ["DPO-S_rejected"] = rate([x["rejected"] for x in ts["dpo"]])
    summ["anchor"] = rate([k for k in keys if k.startswith("anchor|")])
    base = summ["S"]["unsupported"]
    flag = {g: v["unsupported"] for g, v in summ.items()
            if g not in ("S", "anchor") and v["unsupported"] is not None and base is not None
            and v["unsupported"] - base > thr}
    return summ, flag


def stage_factcheck():
    import judge_backends as JB
    d = jload(P("design.json"))
    ts = jload(P("trainsets.json"))
    eps = load_eps("train")
    if not P("train_A.jsonl").exists():
        sys.exit("[factcheck] 先跑 export")
    items = fact_items(d, ts, eps)
    msgs = [[{"role": "system", "content": FACT_SYSTEM},
             {"role": "user", "content": FACT_USER.format(seen=seen, thought=t, action=a)}] for _, seen, t, a in items]
    keys = [k for k, *_ in items]
    t0 = time.time()
    judge = MockFactChecker() if STATE["mock"] else JB.make_judge("api", model="qwen3.8-flash", thinking=False)
    outs = JB.ask_resumable(judge, keys, msgs, P("factcheck_cache.jsonl"))
    rows, n_err = [], 0
    for k, o in zip(keys, outs):
        v = parse_verdict((o or {}).get("text"))
        n_err += bool((o or {}).get("error"))
        rows.append({"key": k, "verdict": v, "text": (o or {}).get("text"), "error": (o or {}).get("error"),
                     "usage": (o or {}).get("usage")})
    jl_write(P("factcheck.jsonl"), rows)
    summ, flag = fact_summary({r["key"]: r["verdict"] for r in rows}, ts, keys, CFG["FACT_FLAG"])
    summ["thought_mode"] = CFG["THOUGHT_MODE"]
    jdump(P("factcheck.json"), {"by_group": summ, "flag_vs_S": flag, "threshold": CFG["FACT_FLAG"],
                                "n_api_errors": n_err, "time": time.strftime("%Y-%m-%d %H:%M")})
    add_timing("factcheck", time.time() - t0, {"gpu_stage": False, "n": len(items)})
    log(f"[factcheck] Thought 方式 {CFG['THOUGHT_MODE']}；无依据断言的比例：" + "，".join(
        f"{g} {v['unsupported']:.2f}（{v['n_parsed']}/{v['n']}）" for g, v in summ.items()
        if isinstance(v, dict) and v["unsupported"] is not None)
        + (f"；比 S 高出 > {CFG['FACT_FLAG']:.2f} 的组：{flag}" if flag else "；没有组比 S 高出阈值")
        + (f"；API 出错 {n_err} 条（重跑同一命令会补上）" if n_err else ""))
    return 1 if n_err else 0
