"""Batch Interlacation rollout：All Unfinished rollout Each round produces one step together（and gridworld of run_rollouts Same idea）。

spec: {"rid", "gamefile", "seed", "prefix": [command…]（Optional，For running from the middle of a trajectory）}
number j steps（Absolute Step，from 0 Get up）Seeds for sampling = derive_seed(spec.seed, j)。
"""
import time

from . import acfg
from .agent import MAX_RETRIES, RETRY_MSG, build_messages, derive_seed, parse_action, parse_intro


def fit_prompt(backend, task, desc, hist, adm, max_new_tokens, max_steps):
    """When the context is not adequate, it is omitted from the earliest history（Every time 5 steps）。Back (prompt, token Count, Number of omitted steps)。"""
    limit = backend.context - max_new_tokens - 8
    n_drop = 0
    while True:
        p = backend.render(build_messages(task, desc, hist, adm, max_steps, n_drop))
        if len(p) <= limit:             # Every one token At least one character：No count is required when the number of characters does not exceed the maximum limit
            return p, None, n_drop
        n = backend.count(p)
        if n <= limit or n_drop >= len(hist):
            return p, n, n_drop
        n_drop = min(len(hist), n_drop + 5)


def run_episodes(backend, pool, specs, temperature, max_steps=None, max_new_tokens=None,
                 keep_text=False, want_key=True, log=None, chunk=None):
    max_steps = max_steps or acfg.MAX_STEPS
    max_new_tokens = max_new_tokens or acfg.MAX_NEW_TOKENS
    chunk = chunk or acfg.CHUNK
    episodes, timing = [], {"wall": 0.0, "gen": 0.0, "env": 0.0, "open": 0.0, "waves": 0, "policy_steps": 0,
                            "gen_tokens": 0, "max_prompt_tokens": 0, "retries": 0}
    for i in range(0, len(specs), chunk):
        eps, t = _run_chunk(backend, pool, specs[i:i + chunk], temperature, max_steps, max_new_tokens,
                            keep_text, want_key, log)
        episodes += eps
        for k in timing:
            timing[k] = max(timing[k], t[k]) if k == "max_prompt_tokens" else timing[k] + t[k]
    return episodes, timing


def _new_episode(s, o, want_key):
    task, desc = parse_intro(o["init_obs"])
    steps = [{"cmd": c, "obs": ob, "kind": "prefix"} for c, ob in zip(s.get("prefix", []), o["prefix_obs"])]
    won_at = next((j + 1 for j, w in enumerate(o["prefix_won"]) if w), None)
    return {"rid": s["rid"], "gamefile": s["gamefile"], "seed": s["seed"], "task": task,
            "desc": desc, "prefix_len": len(steps), "steps": steps,
            "won": bool(o["state"]["won"]), "won_step": won_at,
            "adm": o["state"]["admissible"], "keys": [o["key"]] if want_key else []}


def _wave(backend, pool, eps, alive, temperature, max_steps, max_new_tokens, keep_text, want_key, timing):
    """All Unfinished rollout advance each by one step（Include non-executable rebuttals）。Return to unending rid List。"""
    prompts, metas = [], []
    for rid in alive:
        e = eps[rid]
        p, n, drop = fit_prompt(backend, e["task"], e["desc"], e["steps"], e["adm"], max_new_tokens, max_steps)
        prompts.append(p)
        metas.append((n, drop))
    seeds = [derive_seed(eps[rid]["seed"], len(eps[rid]["steps"])) for rid in alive]
    t0 = time.perf_counter()
    outs = backend.generate(prompts, temperature, seeds, max_new_tokens, hints=[eps[r]["adm"] for r in alive])
    timing["gen"] += time.perf_counter() - t0
    acts = []
    for rid, out in zip(alive, outs):
        a = parse_action(out.text, eps[rid]["adm"])
        acts.append(a)
        timing["gen_tokens"] += out.n_gen
        timing["max_prompt_tokens"] = max(timing["max_prompt_tokens"], out.n_prompt)
    first_kind = [a["kind"] for a in acts]
    n_retry = [0] * len(alive)
    outs = list(outs)
    bad = [i for i, a in enumerate(acts) if a["kind"] in ("invalid", "no_action")]
    for r in range(1, MAX_RETRIES + 1):          # not executable：Bring back the feedback，Non-consumption of step
        if not bad:
            break
        rp = []
        for i in bad:
            e = eps[alive[i]]
            msgs = build_messages(e["task"], e["desc"], e["steps"], e["adm"], max_steps, metas[i][1])
            msgs += [{"role": "assistant", "content": outs[i].text},
                     {"role": "user", "content": RETRY_MSG.format(cmd=acts[i]["cmd"] or "(no action)")}]
            rp.append(backend.render(msgs))
        seeds_r = [derive_seed(eps[alive[i]]["seed"], len(eps[alive[i]]["steps"]), "retry", r) for i in bad]
        t0 = time.perf_counter()
        outs_r = backend.generate(rp, temperature, seeds_r, max_new_tokens, hints=[eps[alive[i]]["adm"] for i in bad])
        timing["gen"] += time.perf_counter() - t0
        for i, o in zip(bad, outs_r):
            outs[i] = o
            acts[i] = parse_action(o.text, eps[alive[i]]["adm"])
            n_retry[i] += 1
            timing["gen_tokens"] += o.n_gen
            timing["retries"] = timing.get("retries", 0) + 1
        bad = [i for i in bad if acts[i]["kind"] in ("invalid", "no_action")]
    t0 = time.perf_counter()
    res = pool.step([{"rid": rid, "cmd": a["cmd"], "want_key": want_key} for rid, a in zip(alive, acts)])
    timing["env"] += time.perf_counter() - t0
    timing["waves"] += 1
    timing["policy_steps"] += len(alive)
    nxt = []
    for rid, a, out, (n, drop), fk, nr in zip(alive, acts, outs, metas, first_kind, n_retry):
        e = eps[rid]
        r = res[rid]
        st = {"cmd": a["cmd"], "obs": r["state"]["obs"], "kind": a["kind"], "fr": out.finish_reason,
              "ntok": out.n_prompt, "drop": drop, "first_kind": fk, "retries": nr}
        if keep_text:
            st["text"] = out.text
        e["steps"].append(st)
        e["adm"] = r["state"]["admissible"]
        if want_key:
            e["keys"].append(r["key"])
        if r["state"]["won"]:
            e["won"] = True
            e["won_step"] = len(e["steps"])
        elif len(e["steps"]) < max_steps:
            nxt.append(rid)
    return nxt


def _new_timing():
    return {"gen": 0.0, "env": 0.0, "open": 0.0, "waves": 0, "policy_steps": 0, "gen_tokens": 0,
            "max_prompt_tokens": 0, "retries": 0}


def _run_chunk(backend, pool, specs, temperature, max_steps, max_new_tokens, keep_text, want_key, log):
    t_start = time.perf_counter()
    timing = _new_timing()
    t0 = time.perf_counter()
    opened = pool.open([{"rid": s["rid"], "gamefile": s["gamefile"], "prefix": list(s.get("prefix", [])),
                         "want_key": want_key} for s in specs])
    timing["open"] = time.perf_counter() - t0
    eps = {s["rid"]: _new_episode(s, opened[s["rid"]], want_key) for s in specs}
    alive = [s["rid"] for s in specs if not eps[s["rid"]]["won"] and len(eps[s["rid"]]["steps"]) < max_steps]
    while alive:
        alive = _wave(backend, pool, eps, alive, temperature, max_steps, max_new_tokens, keep_text, want_key, timing)
        if log and timing["waves"] % 10 == 0:
            log(f"  number {timing['waves']} Wheel，Not completed {len(alive)}，Success {sum(e['won'] for e in eps.values())}/{len(eps)}")
    pool.close([s["rid"] for s in specs])
    timing["wall"] = time.perf_counter() - t_start
    out = []
    for s in specs:
        e = eps[s["rid"]]
        e.pop("adm", None)
        out.append(e)
    return out, timing


def run_stream(backend, pool, specs, temperature, on_done, max_active=None, max_steps=None, max_new_tokens=None,
               want_key=True, log=None, log_every=50):
    """Routine follow-up：Keep at most max_active The rule is advancing at the same time，present rollout replace finished jobs immediately（To avoid the slowest stalling in the batch GPU）。

    specs It could be an alternator；Call at the end of each article on_done(episode)。episode Without Text Output，Released after the end。
    Gradual results versus run_episodes exactly identical（Same spec、Same torrent），only scheduling order differs。
    """
    max_steps = max_steps or acfg.MAX_STEPS
    max_new_tokens = max_new_tokens or acfg.MAX_NEW_TOKENS
    max_active = max_active or acfg.CHUNK
    timing = _new_timing()
    t_start = time.perf_counter()
    it = iter(specs)
    eps, alive, exhausted, n_done = {}, [], False, 0

    def finish(rid):
        nonlocal n_done
        e = eps.pop(rid)
        e.pop("adm", None)
        n_done += 1
        on_done(e)

    while True:
        # Filling
        if not exhausted and len(alive) < max_active:
            new = []
            while len(alive) + len(new) < max_active:
                try:
                    new.append(next(it))
                except StopIteration:
                    exhausted = True
                    break
            if new:
                t0 = time.perf_counter()
                opened = pool.open([{"rid": s["rid"], "gamefile": s["gamefile"], "prefix": list(s.get("prefix", [])),
                                     "want_key": want_key} for s in new])
                timing["open"] += time.perf_counter() - t0
                done_now = []
                for s in new:
                    e = _new_episode(s, opened[s["rid"]], want_key)
                    eps[s["rid"]] = e
                    if e["won"] or len(e["steps"]) >= max_steps:
                        done_now.append(s["rid"])
                    else:
                        alive.append(s["rid"])
                if done_now:
                    pool.close(done_now)
                    for rid in done_now:
                        finish(rid)
        if not alive:
            if exhausted:
                break
            continue
        nxt = _wave(backend, pool, eps, alive, temperature, max_steps, max_new_tokens, False, want_key, timing)
        ended = [rid for rid in alive if rid not in set(nxt)]
        if ended:
            pool.close(ended)
            for rid in ended:
                finish(rid)
        alive = nxt
        if log and timing["waves"] % log_every == 0:
            el = time.perf_counter() - t_start
            log(f"  number {timing['waves']} Wheel：Completed {n_done} records，Ongoing {len(alive)}，Time {el / 60:.1f} minutes，"
                f"{timing['policy_steps'] / max(el, 1e-9):.1f} steps/seconds")
    timing["wall"] = time.perf_counter() - t_start
    return timing


def episode_stats(e):
    """One-stop diagnostics（count only steps taken by the policy itself，Without Prefix）。"""
    pol = [s for s in e["steps"] if s["kind"] != "prefix"]
    n = len(pol)
    kinds = {k: sum(1 for s in pol if s["kind"] == k) for k in ("valid", "alias", "invalid", "no_action")}
    kinds["first_bad"] = sum(1 for s in pol if s.get("first_kind", s["kind"]) in ("invalid", "no_action"))
    kinds["retries"] = sum(s.get("retries", 0) for s in pol)
    keys = e.get("keys") or []
    revisit = 0
    if keys:
        seen = set(keys[:e["prefix_len"] + 1])
        for k in keys[e["prefix_len"] + 1:]:
            revisit += int(k in seen)
            seen.add(k)
    repeat = sum(1 for a, b in zip(pol, pol[1:]) if a["cmd"] == b["cmd"])
    return {"n": n, **kinds, "trunc": sum(1 for s in pol if s.get("fr") == "length"),
            "drop": sum(1 for s in pol if s.get("drop")), "revisit": revisit, "repeat": repeat,
            "max_ntok": max([s.get("ntok") or 0 for s in pol] or [0])}
