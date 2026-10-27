"""Frame Check：7B Quality of step-by-step decision-making，limited by model capability，or limited by prompt design？

at out_120 In real decision-making（Sampling by mission phase，default 300 items），Generate only“Next Actions”，
Dont run the whole game，approximately 5 minutes。comparison：

  deployed          Conditions of deployment：Existing prompts（Thought approximately 25 Words），T=0.7，Every State 4 runs
  deployed_greedy   Ibid，greedy decoding，Every State 1 runs
  reasoned          Let go of the length of the reasoning（At most 512 token），The rest of the information is exactly the same，T=0.7，Every State 4 runs
  --model-dir run only the other models batch other_deployed（Existing prompts，T=0.7，Every State 4 runs），Results written to a separate directory；
              Two in the same process vLLM The model will be deficient，therefore run in two invocations

Indicators（Only on budget resolution）：
  Best      Actions oracle One of the actions on the shortest path（No budget consideration，i.e“One step towards success”）
  Read Diagram Error Rate  selected an action explicitly described in the observation as wall / outside / locked door / TRAP Direction
  Trap Rate      selected an action described in the observation as TRAP Direction
  Cut Rate      Output achieved token End of cap

The rules of interpretation written in advance：
  - if reasoned Relative deployed Best rate of improvement ≥ 15 Percentage points，And the reading error rate is at least halved：
    Demonstrate that the existing framework is clearly limited 7B，“Weak strategy”partly caused by the design；the paper must distinguish“Strategic ability”As a variable
    （For example, with reasoned Rerun as a second deployment strategy quick），the existing setup alone cannot settle this question。
  - otherwise：7B One-step capacity on this mission itself is limited，Existing frameworks are not the main bottlenecks。

  python harness_check.py                     # Results written out_harness/
  python harness_check.py --model-dir /path --out-dir out_harness_other   # Another model（Same prompts）
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import config
import oracle
from env import GridWorld

OUT = config.ROOT / "out_harness"
SEED = 2026
N_STATES = 300
K = 4
REASON_TOKENS = 512

_INTRO_REASONED = """You control an agent in a gridworld. Each turn you read one observation, think it through, and end your answer with exactly one line:
Action: {"tool": "<name>", "arg": "<optional>"}"""

_RULES_REASONED = """Rules:
- First think step by step about the observation: which stage you are in, where the target is, and what each candidate move leads to. You may write several sentences.
- Then end with exactly one line: Action: {"tool": "<name>", "arg": "<optional>"}. No markdown, no code fence.
- You have limited steps. Waste none.
- After a blocked or unsuccessful action, use the actual position and the result to correct your next action."""


def stage_of(s):
    if s["has_gem"]:
        return "gem->home"
    if s["door_open"]:
        return "door_open->gem"
    if s["has_key"]:
        return "key->door"
    return "->key"


def sample_states(n, seed):
    recs = [json.loads(l) for l in open(config.SOURCE_EPISODES, encoding="utf-8") if l.strip()]
    by_stage = defaultdict(list)
    for r in recs:
        for t, s in enumerate(r["states"][:-1]):
            if not s["done"] and oracle.feasible(s):
                by_stage[stage_of(s)].append({"traj_id": r["traj_id"], "t": t, "snapshot": s})
    rng = random.Random(seed)
    per = n // len(by_stage)
    out = []
    for st in sorted(by_stage):
        rows = by_stage[st][:]
        rng.shuffle(rows)
        out += rows[:per]
    return out


def move_label(obs, direction):
    m = re.search(rf"^- {direction}: (.*)$", obs, re.M)
    return m.group(1) if m else ""


def score(state, action):
    s = state["snapshot"]
    tab = oracle.table_for(s)
    key = lambda a: (a.get("tool"), a.get("arg") if a.get("tool") == "move" else None)
    opt = {key(a) for a in tab.optimal_actions(s)}
    obs = GridWorld.restore(s).observe()
    reading_err = trap = False
    if action.get("tool") == "move" and action.get("arg") in ("up", "down", "left", "right"):
        lab = move_label(obs, action["arg"])
        trap = lab.startswith("TRAP")
        reading_err = trap or lab.startswith(("wall", "outside", "locked door"))
    return {"optimal": key(action) in opt, "reading_error": reading_err, "trap": trap,
            "malformed": action.get("tool") == "malformed"}


def build_prompts(policy, states, variant):
    import model
    prompts = []
    for st in states:
        env = GridWorld.restore(st["snapshot"])
        msgs = model.AgentPolicy.build_messages(env.observe(), env.history)
        if variant == "reasoned":
            msgs[0]["content"] = msgs[0]["content"].replace(model._RULES_THINK, _RULES_REASONED).replace(
                model._INTRO_THINK, _INTRO_REASONED)
            msgs[1]["content"] = msgs[1]["content"].replace(
                "Write the Thought line, then the Action line.",
                "Think step by step, then end with the Action line.")
            assert _RULES_REASONED in msgs[0]["content"] and _INTRO_REASONED in msgs[0]["content"], "Subscribe Failed"
        prompts.append(policy.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return prompts


def run_variant(policy, states, variant, temperature, k, max_tokens):
    from model import _extract_action
    prompts = build_prompts(policy, states, variant)
    params = [policy._make_params(n=k, temperature=temperature, top_p=config.TOP_P, max_tokens=max_tokens,
                                  **({"seed": SEED + i} if temperature > 0 else {}))
              for i in range(len(prompts))]
    outs = policy.llm.generate(prompts, params, use_tqdm=False)
    rows = []
    for st, out in zip(states, outs):
        for o in out.outputs:
            act = _extract_action(o.text)
            r = score(st, act)
            r.update({"variant": variant, "stage": stage_of(st["snapshot"]), "traj_id": st["traj_id"], "t": st["t"],
                      "truncated": getattr(o, "finish_reason", None) == "length",
                      "n_tokens": len(getattr(o, "token_ids", []) or []), "action": act})
            rows.append(r)
    return rows


def summarize(rows):
    out = {}
    for v in sorted({r["variant"] for r in rows}):
        rs = [r for r in rows if r["variant"] == v]
        f = lambda key, sub=rs: sum(r[key] for r in sub) / len(sub)
        out[v] = {"n": len(rs), "optimal": f("optimal"), "reading_error": f("reading_error"), "trap": f("trap"),
                  "malformed": f("malformed"), "truncated": f("truncated"),
                  "mean_tokens": sum(r["n_tokens"] for r in rs) / len(rs),
                  "by_stage_optimal": {st: f("optimal", [r for r in rs if r["stage"] == st])
                                       for st in sorted({r["stage"] for r in rs})}}
    return out


def report(summary):
    L = ["# Frame Check：Quality of step-by-step decision-making\n",
         "| Condition | Sample | Best | Read Diagram Error Rate | Trap Rate | Cut Rate | Average Output token | Best for all stages |",
         "|---|---:|---:|---:|---:|---:|---:|---|"]
    for v, s in summary.items():
        stages = "; ".join(f"{k} {x:.2f}" for k, x in s["by_stage_optimal"].items())
        L.append(f"| {v} | {s['n']} | {s['optimal']:.3f} | {s['reading_error']:.3f} | {s['trap']:.3f} | "
                 f"{s['truncated']:.3f} | {s['mean_tokens']:.0f} | {stages} |")
    if "deployed" in summary and "reasoned" in summary:
        d, r = summary["deployed"], summary["reasoned"]
        gain = r["optimal"] - d["optimal"]
        halved = r["reading_error"] <= d["reading_error"] / 2
        verdict = ("Existing frameworks are clearly limited 7B（Best rate of improvement ≥ 15 percentage points with at least half of the error in reading）"
                   if gain >= 0.15 and halved else "Existing frameworks are not the main bottlenecks（Failure to achieve pre-defined improvements）")
        L.append(f"\n change in optimal-action rate when reasoning is enabled {gain:+.3f}，Read Diagram Error Rate {d['reading_error']:.3f} → {r['reading_error']:.3f}。")
        L.append(f"By the pre-rule：**{verdict}**。")
        if r["truncated"] > 0.05:
            L.append(f"Attention：reasoned Cut Rate {r['truncated']:.2f}，Interrupted output scores by parsing results，Probably underestimating his abilities。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="Another model（Use the same prompts）")
    ap.add_argument("--n-states", type=int, default=N_STATES)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    states = sample_states(args.n_states, SEED)
    print(f"Draw {len(states)} _Other Organiser：", dict(Counter(stage_of(s["snapshot"]) for s in states)))

    from model import AgentPolicy
    rows = []
    if args.model_dir:
        other = AgentPolicy(model_dir=args.model_dir, temperature=0.7)
        for r in run_variant(other, states, "deployed", 0.7, K, config.MAX_NEW_TOKENS):
            r["variant"] = "other_deployed"
            rows.append(r)
    else:
        pol = AgentPolicy(temperature=0.7)
        rows += run_variant(pol, states, "deployed", 0.7, K, config.MAX_NEW_TOKENS)
        rows += run_variant(pol, states, "deployed_greedy", 0.0, 1, config.MAX_NEW_TOKENS)
        rows += run_variant(pol, states, "reasoned", 0.7, K, REASON_TOKENS)

    summary = summarize(rows)
    (args.out_dir / "harness_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (args.out_dir / "harness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = report(summary)
    (args.out_dir / "harness_report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
