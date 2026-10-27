"""Offline audit of the environment and a saved trajectory; no model/GPU needed.

The search is an environment oracle, NOT a replacement for the agent. It runs
the real GridWorld.step and checks whether the action budget permits success.
"""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import re
import statistics

import config
from env import GridWorld


SEARCH_ACTIONS = [
    {"tool": "move", "arg": direction}
    for direction in ("up", "down", "left", "right")
] + [{"tool": "pick"}, {"tool": "open"}]


def physical_key(snap):
    """For BFS, reaching an identical physical state earlier dominates later."""
    return (
        tuple(snap["pos"]), snap["has_key"], snap["has_gem"],
        snap["door_open"], tuple(tuple(row) for row in snap["layout"]),
    )


def shortest_solution(snap, require_open=False, forbid_open=False):
    """Return shortest success, optionally requiring open BEFORE gem pickup."""
    if snap["done"]:
        return [] if snap["success"] and (not require_open or snap["door_open"]) else None
    queue = deque([(snap, [])])
    seen = {physical_key(snap)}
    for_actions = [a for a in SEARCH_ACTIONS if not (forbid_open and a["tool"] == "open")]
    while queue:
        current, path = queue.popleft()
        for action in for_actions:
            row, col = current["pos"]
            if (require_open and action["tool"] == "pick"
                    and current["layout"][row][col] == "g" and not current["door_open"]):
                continue
            env = GridWorld.restore(current)
            env.step(action)
            if env.success and (not require_open or env.door_open):
                return path + [action]
            if env.done:
                continue
            following = env.snapshot()
            key = physical_key(following)
            if key in seen:
                continue
            seen.add(key)
            queue.append((following, path + [action]))
    return None


def verify_solution(snap, actions):
    if actions is None:
        return False
    env = GridWorld.restore(snap)
    for action in actions:
        env.step(action)
    assert env.done and env.success, "Oracle path failed when replayed in GridWorld.step"
    return True


def read_trajectory(path):
    """Allow a pasted JSON fragment whose complete states/actions arrays survive."""
    source = Path(path).read_text(encoding="utf-8-sig").strip()
    try:
        return json.loads(source), False
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        result = {}
        for field in ("traj_id", "success", "fail_reason", "T", "states", "actions"):
            match = re.search(r'"' + field + r'"\s*:', source)
            if match is None:
                raise ValueError(f"Missing complete field: {field}")
            value, _ = decoder.raw_decode(source[match.end():].lstrip())
            result[field] = value
        return result, True


def audit_trajectory(path):
    record, fragment = read_trajectory(path)
    states, actions = record["states"], record["actions"]
    assert len(states) == len(actions) + 1
    env = GridWorld.restore(states[0])
    mismatches, no_effect, key_steps, gem_steps = [], [], [], []
    for t, action in enumerate(actions, 1):
        before = env.snapshot()
        env.step(action)
        after = env.snapshot()
        unequal = [key for key, value in states[t].items() if after.get(key) != value]
        if unequal:
            mismatches.append({"t": t, "fields": unequal})
        if physical_key(before) == physical_key(after):
            no_effect.append({"t": t, "pos": before["pos"], "action": action})
        if not before["has_key"] and after["has_key"]:
            key_steps.append(t)
        if not before["has_gem"] and after["has_gem"]:
            gem_steps.append(t)
    initial = states[0]
    best = shortest_solution(initial)
    with_open = shortest_solution(initial, require_open=True)
    bypass = shortest_solution(initial, forbid_open=True)
    for solution in (best, with_open, bypass):
        verify_solution(initial, solution)
    viable = [t for t, state in enumerate(states) if shortest_solution(state) is not None]
    return {
        "traj_id": record["traj_id"],
        "input_is_json_fragment": fragment,
        "n_actions": len(actions),
        "state_replay_mismatches": mismatches,
        "success": env.success, "fail_reason": env.fail_reason,
        "key_picked_at": key_steps, "gem_picked_at": gem_steps,
        "no_effect_count": len(no_effect), "no_effect_actions": no_effect,
        "shortest_steps": len(best) if best is not None else None,
        "shortest_with_open_steps": len(with_open) if with_open is not None else None,
        "shortest_without_open_steps": len(bypass) if bypass is not None else None,
        "verified_solution_with_open": with_open,
        "last_oracle_recoverable_state_t": max(viable) if viable else None,
        "state_after_last_key_pick": states[key_steps[-1]] if key_steps else None,
        "observation_after_failed_open": GridWorld.restore(states[7]).observe() if len(states) > 7 else None,
    }


def audit_layouts(n_episodes):
    rows = []
    for seed in range(n_episodes):
        layout = GridWorld.random_layout_checked(
            n=config.GRID, n_traps=config.N_TRAPS, n_walls=config.N_WALLS, seed=seed,
        )
        snap = GridWorld(layout, max_steps=config.MAX_STEPS, seed=seed).snapshot()
        best = shortest_solution(snap)
        with_open = shortest_solution(snap, require_open=True)
        bypass = shortest_solution(snap, forbid_open=True)
        for solution in (best, with_open, bypass):
            verify_solution(snap, solution)
        rows.append({
            "seed": seed, "shortest": len(best) if best is not None else None,
            "with_open": len(with_open) if with_open is not None else None,
            "without_open": len(bypass) if bypass is not None else None,
        })
    def stats(field):
        values = [r[field] for r in rows if r[field] is not None]
        return {"solvable": len(values), "min": min(values) if values else None,
                "median": statistics.median(values) if values else None,
                "max": max(values) if values else None,
                "histogram": dict(sorted(Counter(values).items()))}
    return {"n_episodes": n_episodes, "any_solution": stats("shortest"),
            "require_open": stats("with_open"), "forbid_open": stats("without_open"),
            "rows": rows}


def audit_policy_contract():
    """Exercise actual policy methods with fake inference; never load weights."""
    import model
    from types import SimpleNamespace
    prompts = []
    def generate(batch, params, **kwargs):
        prompts.extend(batch)
        return [SimpleNamespace(outputs=[SimpleNamespace(text='{"tool":"give_up"}')])
                for _ in batch]
    policy = model.AgentPolicy.__new__(model.AgentPolicy)
    policy.temperature = config.TEMPERATURE
    policy._make_params = lambda **kwargs: SimpleNamespace(**kwargs)
    policy.llm = SimpleNamespace(generate=generate)
    policy.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: messages)

    env = GridWorld([list(".."), list(".g")], max_steps=10)
    for action in [{"tool": "move", "arg": "right"}, {"tool": "move", "arg": "down"},
                   {"tool": "pick"}, {"tool": "move", "arg": "up"},
                   {"tool": "move", "arg": "left"}]:
        env.step(action)
    assert env.success
    terminal_results, _ = policy.run_rollouts([env.snapshot()])
    nonterminal = GridWorld([list(".."), list(".g")], max_steps=10)
    action = {"tool": "move", "arg": "right"}
    nonterminal.step(action)
    collection_messages = policy.build_messages(nonterminal.observe(), nonterminal.history)
    policy.run_rollouts([nonterminal.snapshot()])
    continuation_messages = prompts[-1]
    return {
        "inference_is_fake": True,
        "terminal_success_expected": [True],
        "terminal_success_actual": terminal_results,
        "collection_and_continuation_same_messages": collection_messages == continuation_messages,
        "collection_user_message": collection_messages[-1]["content"],
        "continuation_user_message": continuation_messages[-1]["content"],
        "system_requires_last_action_result": 'RESULT OF YOUR LAST ACTION' in model.SYSTEM_PROMPT,
        "observation_provides_last_action_result": 'RESULT OF YOUR LAST ACTION' in nonterminal.observe(),
    }


def audit_saved_episodes(path):
    """Audit actual saved generations and transitions, without loading weights."""
    import model
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    actions, events, finish, empty_picks = Counter(), Counter(), Counter(), Counter()
    counters = Counter()
    mismatches = []
    for rec in records:
        env = GridWorld.restore(rec["states"][0])
        for t, action in enumerate(rec["actions"]):
            state = rec["states"][t]
            row, col = state["pos"]
            counters["decision_states"] += 1
            counters["inside_top_left_3x3"] += int(row <= 2 and col <= 2)
            counters["column_above_2"] += int(col > 2)
            actions[action["tool"]] += 1
            generations = rec.get("generations", [])
            if t < len(generations):
                gen = generations[t]
                finish[str(gen.get("finish_reason"))] += 1
                if model._extract_action(gen.get("text", "")) != action:
                    mismatches.append([rec["traj_id"], t + 1, "parser"])
            if action["tool"] == "pick" and state["layout"][row][col] not in ("k", "g"):
                empty_picks[f"({row},{col})"] += 1
            if t and state["history"][-1]["event"] in ("boundary", "wall", "locked_door"):
                counters["after_blocked"] += 1
                counters["same_action_after_blocked"] += int(action == rec["actions"][t - 1])
            env.step(action)
            if env.snapshot() != rec["states"][t + 1]:
                mismatches.append([rec["traj_id"], t + 1, "state"])
            events[env.history[-1]["event"]] += 1
        counters["ever_key"] += int(any(s["has_key"] for s in rec["states"]))
        counters["ever_door_open"] += int(any(s["door_open"] for s in rec["states"]))
        counters["ever_gem"] += int(any(s["has_gem"] for s in rec["states"]))
    return {"n_episodes": len(records), "successes": sum(r["success"] for r in records),
            "actions": dict(actions), "events": dict(events), "finish_reasons": dict(finish),
            "counts": dict(counters), "empty_pick_positions": dict(empty_picks), "mismatches": mismatches}


def policy_probe_cases():
    """Synthetic valid 5x5 states. Oracle answers are used ONLY for scoring."""
    def make(name, pos, key=(1, 2), has_key=False, has_gem=False, door_open=False,
             walls=(), traps=(), blocked_history=False):
        layout = [list(".....") for _ in range(5)]
        layout[4][3], layout[4][4], layout[3][4] = "D", "g", "#"
        layout[key[0]][key[1]] = "k"
        for row, col in walls:
            layout[row][col] = "#"
        for row, col in traps:
            layout[row][col] = "T"
        env = GridWorld(layout, max_steps=30)
        env.pos = pos
        env.has_key, env.has_gem, env.door_open = has_key, has_gem, door_open
        if has_key or door_open:
            env.layout[key[0]][key[1]] = "."
        if door_open:
            env.layout[4][3] = "."
        if has_gem:
            env.layout[4][4] = "."
        if blocked_history:
            env.step({"tool": "move", "arg": "right"})
            env.step({"tool": "move", "arg": "right"})
        return {"name": name, "snapshot": env.snapshot()}

    cases = [
        make("pick_key_underfoot", (1, 2)),
        make("open_adjacent_with_key", (3, 3), has_key=True),
        make("pick_gem_underfoot", (4, 4), door_open=True),
        make("return_home_one_step", (0, 1), has_gem=True, door_open=True),
        make("key_right_column_4", (0, 3), key=(0, 4)),
        make("key_left_row_4", (4, 1), key=(4, 0)),
        make("key_down_row_3", (2, 1), key=(3, 1)),
        make("key_up_row_3", (4, 1), key=(3, 1)),
        make("detour_around_wall", (0, 0), key=(0, 2), walls=((0, 1),)),
        make("detour_around_trap", (0, 0), key=(0, 2), traps=((0, 1),)),
        make("correct_after_wall_feedback", (0, 0), key=(0, 2), walls=((0, 1),), blocked_history=True),
        make("demo_key_cell_now_empty", (1, 0), key=(1, 1)),
        make("door_adjacent_without_key", (3, 3), key=(3, 2)),
    ]
    for case in cases:
        snap = case["snapshot"]
        best = shortest_solution(snap)
        assert best is not None, case["name"]
        accepted = []
        for action in SEARCH_ACTIONS:
            env = GridWorld.restore(snap)
            env.step(action)
            rest = shortest_solution(env.snapshot())
            if rest is not None and 1 + len(rest) == len(best):
                accepted.append(action)
        assert accepted, case["name"]
        case["expected_actions"] = accepted
        case["remaining_optimal_steps"] = len(best)
    return cases


def run_policy_probes(policy, temperature=0.0):
    """Compare prompting conditions on matched states; never alters the baseline."""
    import model
    cases = policy_probe_cases()
    no_demo = model.SYSTEM_PROMPT.replace(model._EXAMPLE, "").replace(
        "Follow the plan in the example:", "Task prerequisites:")
    modes = ("current", "no_demo", "local_facts", "reasoned")
    rows = []
    for mode in modes:
        messages_batch, prompts = [], []
        max_tokens = 256 if mode == "reasoned" else config.MAX_NEW_TOKENS
        for case in cases:
            env = GridWorld.restore(case["snapshot"])
            messages = policy.build_messages(env.observe(), env.history)
            if mode != "current":
                messages[0]["content"] = no_demo
            if mode == "local_facts":
                row, col = env.pos
                cells = [f"cell under you = {env.layout[row][col]}"]
                for direction, (dr, dc) in {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}.items():
                    rr, cc = row + dr, col + dc
                    cell = env.layout[rr][cc] if 0 <= rr < env.n and 0 <= cc < env.n else "outside map"
                    cells.append(f"{direction} leads to (row {rr}, col {cc}): {cell}")
                messages[1]["content"] += "\nLocal cells (same map information):\n" + "\n".join(cells)
            elif mode == "reasoned":
                messages[0]["content"] = messages[0]["content"].replace(
                    "At each step output EXACTLY ONE tool call as a JSON object, nothing else.",
                    "Briefly reason about the current map and goal, then end with exactly ONE JSON tool call.")
                messages[0]["content"] = messages[0]["content"].replace(
                    "Output only the JSON object. No explanation, no markdown, no code fence.",
                    "Write a brief analysis without JSON objects. End with one final JSON tool call.")
                messages[1]["content"] = messages[1]["content"].replace(
                    "Output the next tool call as JSON.", "Briefly reason, then output one final JSON tool call.")
            messages_batch.append(messages)
            prompts.append(policy.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        params = policy._make_params(temperature=temperature, top_p=config.TOP_P, max_tokens=max_tokens)
        outputs = policy.llm.generate(prompts, params, use_tqdm=False)
        for case, messages, prompt, output in zip(cases, messages_batch, prompts, outputs):
            generated = output.outputs[0]
            action = model._extract_action(generated.text)
            def comparable(a):
                return (a.get("tool"), a.get("arg") if a.get("tool") == "move" else None)
            rows.append({"case": case["name"], "mode": mode,
                         "correct": comparable(action) in [comparable(a) for a in case["expected_actions"]],
                         "action": action, "expected_actions": case["expected_actions"],
                         "text": generated.text, "finish_reason": getattr(generated, "finish_reason", None),
                         "messages": messages, "max_tokens": max_tokens,
                         "input_tokens": len(policy.tokenizer.encode(prompt)) if hasattr(policy.tokenizer, "encode") else None})
    checkpoint = {}
    path = config.MODEL_DIR / "config.json"
    if path.is_file():
        source = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = {key: source.get(key) for key in ("_name_or_path", "model_type", "architectures", "hidden_size", "num_hidden_layers", "vocab_size")}
    return {"run_id": config.run_id(), "policy_id": getattr(policy, "cache_id", None),
            "temperature": temperature, "cases_per_mode": len(cases),
            "model_dir": str(config.MODEL_DIR), "checkpoint_config": checkpoint,
            "tokenizer_class": type(policy.tokenizer).__name__,
            "chat_template": getattr(policy.tokenizer, "chat_template", None),
            "notes": "Matched synthetic one-step states, not episode success rates. Local facts add representation help; reasoned permits more tokens. Temperature 0 is deterministic diagnostic, not the temperature-0.7 rollout policy.",
            "scores": {mode: sum(r["correct"] for r in rows if r["mode"] == mode) for mode in modes}, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--maps", type=int, default=120)
    parser.add_argument("--check-policy", action="store_true")
    parser.add_argument("--episodes-jsonl", type=Path, help="audit saved model outputs; does not load weights")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "config": {name: getattr(config, name) for name in ("GRID", "MAX_STEPS", "N_TRAPS", "N_WALLS")},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("env.py", "model.py", "config.py", "episodes.py", "vcurve.py")},
        "layouts": audit_layouts(args.maps),
    }
    if args.trajectory:
        report["trajectory"] = audit_trajectory(args.trajectory)
    if args.check_policy:
        report["policy_contract"] = audit_policy_contract()
    if args.episodes_jsonl:
        report["episodes"] = audit_saved_episodes(args.episodes_jsonl)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {key: value for key, value in report.items() if key not in ("trajectory", "layouts", "policy_contract")}
    compact["layouts"] = {key: value for key, value in report["layouts"].items() if key != "rows"}
    if "trajectory" in report:
        compact["trajectory"] = {key: value for key, value in report["trajectory"].items()
                                 if key not in ("state_after_last_key_pick", "observation_after_failed_open", "verified_solution_with_open")}
    if "policy_contract" in report:
        compact["policy_contract"] = {key: value for key, value in report["policy_contract"].items()
                                      if not key.endswith("user_message")}
    print(json.dumps(compact, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
