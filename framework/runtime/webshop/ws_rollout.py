"""WebShop streaming continuations（and alfw/rollout.run_stream Same idea：Keep at most max_active The rule is advancing at the same time，Finish one more one）。

spec: {"rid", "goal", "seed", "prefix": [action…]（Optional，For running from the middle of a trajectory）}
number j steps（Absolute Step，from 0 Get up）Seeds for sampling = derive_seed(spec.seed, j)；retry seeds additionally include "retry" with number。
One game is over = Buy the merchandise（done）or run out of steps。
"""
import time

from . import wcfg
from .ws_agent import MAX_RETRIES, RETRY_MSG, build_messages, clean_obs, derive_seed, parse_action


def fit_prompt(backend, instruction, hist, state, max_new_tokens, max_steps):
    limit = backend.context - max_new_tokens - 8
    n_drop = 0
    while True:
        p = backend.render(build_messages(instruction, hist, state, max_steps, n_drop))
        if len(p) <= limit:
            return p, None, n_drop
        n = backend.count(p)
        if n <= limit or n_drop >= len(hist):
            return p, n, n_drop
        n_drop = min(len(hist), n_drop + 3)


def _new_episode(s, o):
    instr = o["instruction"]
    steps = [{"cmd": c, "obs": clean_obs(p["state"]["obs"], instr), "kind": "prefix", "valid": p["valid"]}
             for c, p in zip(s.get("prefix", []), o["prefix"])]
    st = o["state"]
    return {"rid": s["rid"], "goal": s["goal"], "seed": s["seed"], "instruction": instr, "prefix_len": len(steps),
            "steps": steps, "done": st["done"], "reward": st["reward"], "won": st["won"], "state": st,
            "keys": [st["key"]], "n_actions": [len(st["clicks"]) + int(st["search"])], "pages": [st["page"]]}


def _wave(backend, pool, eps, alive, temperature, max_steps, max_new_tokens, timing):
    prompts, metas = [], []
    for rid in alive:
        e = eps[rid]
        p, n, drop = fit_prompt(backend, e["instruction"], e["steps"], e["state"], max_new_tokens, max_steps)
        prompts.append(p)
        metas.append((n, drop))
    seeds = [derive_seed(eps[rid]["seed"], len(eps[rid]["steps"])) for rid in alive]
    t0 = time.perf_counter()
    outs = list(backend.generate(prompts, temperature, seeds, max_new_tokens, hints=[eps[r]["state"] for r in alive]))
    timing["gen"] += time.perf_counter() - t0
    acts = [parse_action(o.text, eps[rid]["state"]) for rid, o in zip(alive, outs)]
    for o in outs:
        timing["gen_tokens"] += o.n_gen
        timing["max_prompt_tokens"] = max(timing["max_prompt_tokens"], o.n_prompt)
    first_kind = [a["kind"] for a in acts]
    n_retry = [0] * len(alive)
    bad = [i for i, a in enumerate(acts) if a["kind"] in ("invalid", "no_action")]
    for r in range(1, MAX_RETRIES + 1):
        if not bad:
            break
        rp = []
        for i in bad:
            e = eps[alive[i]]
            msgs = build_messages(e["instruction"], e["steps"], e["state"], max_steps, metas[i][1])
            msgs += [{"role": "assistant", "content": outs[i].text},
                     {"role": "user", "content": RETRY_MSG.format(cmd=acts[i]["cmd"] or "(no action)")}]
            rp.append(backend.render(msgs))
        seeds_r = [derive_seed(eps[alive[i]]["seed"], len(eps[alive[i]]["steps"]), "retry", r) for i in bad]
        t0 = time.perf_counter()
        outs_r = backend.generate(rp, temperature, seeds_r, max_new_tokens, hints=[eps[alive[i]]["state"] for i in bad])
        timing["gen"] += time.perf_counter() - t0
        for i, o in zip(bad, outs_r):
            outs[i] = o
            acts[i] = parse_action(o.text, eps[alive[i]]["state"])
            n_retry[i] += 1
            timing["gen_tokens"] += o.n_gen
            timing["retries"] += 1
        bad = [i for i in bad if acts[i]["kind"] in ("invalid", "no_action")]
    t0 = time.perf_counter()
    res = pool.step([{"rid": rid, "cmd": a["cmd"]} for rid, a in zip(alive, acts)])
    timing["env"] += time.perf_counter() - t0
    timing["waves"] += 1
    timing["policy_steps"] += len(alive)
    nxt = []
    for rid, a, out, (n, drop), fk, nr in zip(alive, acts, outs, metas, first_kind, n_retry):
        e = eps[rid]
        r = res[rid]
        st = r["state"]
        e["steps"].append({"cmd": a["cmd"], "obs": clean_obs(st["obs"], e["instruction"]), "kind": a["kind"],
                           "valid": r["valid"], "fr": out.finish_reason, "ntok": out.n_prompt, "drop": drop,
                           "first_kind": fk, "retries": nr})
        e["state"] = st
        e["keys"].append(st["key"])
        e["pages"].append(st["page"])
        e["n_actions"].append(len(st["clicks"]) + int(st["search"]))
        e["done"], e["reward"], e["won"] = st["done"], st["reward"], st["won"]
        if not st["done"] and len(e["steps"]) < max_steps:
            nxt.append(rid)
    return nxt


def _new_timing():
    return {"gen": 0.0, "env": 0.0, "open": 0.0, "waves": 0, "policy_steps": 0, "gen_tokens": 0,
            "max_prompt_tokens": 0, "retries": 0}


def run_stream(backend, pool, specs, temperature, on_done, max_active=None, max_steps=None, max_new_tokens=None,
               log=None, log_every=20):
    max_steps = max_steps or wcfg.MAX_STEPS
    max_new_tokens = max_new_tokens or wcfg.MAX_NEW_TOKENS
    max_active = max_active or wcfg.MAX_ACTIVE
    timing = _new_timing()
    t_start = time.perf_counter()
    it = iter(specs)
    eps, alive, exhausted, n_done = {}, [], False, 0

    def finish(rid):
        nonlocal n_done
        e = eps.pop(rid)
        e.pop("state", None)
        n_done += 1
        on_done(e)

    while True:
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
                opened = pool.open([{"rid": s["rid"], "goal": s["goal"], "prefix": list(s.get("prefix", []))}
                                    for s in new])
                timing["open"] += time.perf_counter() - t0
                done_now = []
                for s in new:
                    e = _new_episode(s, opened[s["rid"]])
                    eps[s["rid"]] = e
                    if e["done"] or len(e["steps"]) >= max_steps:
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
        nxt = _wave(backend, pool, eps, alive, temperature, max_steps, max_new_tokens, timing)
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
    pol = [s for s in e["steps"] if s["kind"] != "prefix"]
    n = len(pol)
    kinds = {k: sum(1 for s in pol if s["kind"] == k) for k in ("valid", "alias", "invalid", "no_action")}
    kinds["first_bad"] = sum(1 for s in pol if s.get("first_kind", s["kind"]) in ("invalid", "no_action"))
    kinds["retries"] = sum(s.get("retries", 0) for s in pol)
    keys = e.get("keys") or []
    seen, revisit = set(keys[:e["prefix_len"] + 1]), 0
    for k in keys[e["prefix_len"] + 1:]:
        revisit += int(k in seen)
        seen.add(k)
    return {"n": n, **kinds, "trunc": sum(1 for s in pol if s.get("fr") == "length"),
            "drop": sum(1 for s in pol if s.get("drop")), "revisit": revisit,
            "searches": sum(1 for s in pol if s["cmd"].startswith("search[") and s.get("valid")),
            "max_ntok": max([s.get("ntok") or 0 for s in pol] or [0])}
