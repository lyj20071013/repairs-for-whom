"""Gridworld：Only judgment、repair selection without execution（Corresponding Who&When / FALAT One by one LLM Reading Log Logging Practices）。

The judges read:：Environmental rules、Initial map、Every step of a failed trajectory（position、Holdings、action、Environmental feedback）、Reasons for Failure，And the confirmation of the experiment
Exactly the same candidate for restoration（4 Organisation × Replace All Actions）。must select“Most likely the same executor（7B，temperature 0.7）Success within the remaining steps”
A fix。shuffle candidates with a fixed per-trajectory seed（Avoid position preferences），Numbering C1…Cn。

    python judge_gw.py --judge vllm                 # 7B use the deployed model as judge（GPU）
    python judge_gw.py --judge api                  # A stronger judge：Qwen3.8-flash Think about it（default；Environment variable required QWEN_API_KEY）
    python judge_gw.py --judge api --model deepseek-flash  # DeepSeek（Yes DEEPSEEK_API_KEY；2026-10-23 no longer used in primary results from this point）
    python judge_gw.py --judge api --model qwen3.8-flash   # Qwen judge（Environment variable required QWEN_API_KEY）
    python judge_gw.py --judge vllm --dry-run       # Print only one prompts
output out_confirm/judge_<name>.jsonl；Evaluation methods_gw.py。
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import config
from env import GridWorld
from judge_backends import ask_resumable, make_judge, parse_choice, write_jsonl

OUT = config.ROOT / "out_confirm"

SYSTEM = ("You are an expert debugger of AI agents. You will read the log of a failed run of an agent in a small "
          "gridworld and decide which single one-step fix is most likely to make the run succeed.")

_FAIL = {"trap_step": "the agent stepped on a trap", "timeout_return": "the agent ran out of steps while carrying "
         "the gem back to the start", "timeout_before_door": "the agent ran out of steps before opening the door",
         "open_without_key": "the agent tried to open the door without the key"}


def _act(a):
    if isinstance(a, str):
        tool, _, arg = a.partition(":")
    else:
        tool, arg = a.get("tool"), a.get("arg")
    return f"{tool}({arg})" if arg else f"{tool}()"


def render(rec, cands, executor_info=True):
    first = rec["states"][0]
    grid = "\n".join(f"row {i}: " + " ".join(r) for i, r in enumerate(first["initial_layout"]))
    last = rec["states"][-1]
    lines = []
    for h in last["history"]:
        t = h["step"]
        s = rec["states"][t]
        lines.append(f"Step {t}: at (row {h['pos_before'][0]}, col {h['pos_before'][1]}), key={s['has_key']}, "
                     f"door_open={s['door_open']}, gem={s['has_gem']} | action {_act(h['action'])} -> {h['result']}")
    cl = [f"C{i + 1}: at step {c['t']}, replace {_act(c['orig'])} with {_act(c['alt'])}" for i, c in enumerate(cands)]
    user = (f"{GridWorld.TOOL_SPEC}\n\nLegend: '.' empty, '#' wall, 'T' trap, 'k' key, 'D' locked door, 'g' gem. "
            f"Row 0 is the top row; the start cell is (row 0, col 0). The step budget is {first['max_steps']} steps.\n\n"
            f"Initial map:\n{grid}\n\n"
            + ("The agent is a 7B language model that samples its actions at temperature 0.7. It sees the map, its position "
               "and the result of its last action at every step.\n\n" if executor_info else "")
            + f"Log of the failed run ({len(lines)} steps; failure: {_FAIL.get(rec['fail_reason'], rec['fail_reason'])}):\n"
            + "\n".join(lines)
            + "\n\nCandidate fixes. Each replaces exactly one action of the log; the environment executes the new action, "
              "and then the SAME agent continues on its own from there with the remaining steps:\n"
            + "\n".join(cl)
            + "\n\nWhich single fix is most likely to make this agent's run succeed? Think step by step briefly, "
              "then end with one line: Choice: C<number>")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def build(out_dir, executor_info=True):
    d = json.loads((out_dir / "main_design.json").read_text(encoding="utf-8"))
    recs = {}
    with open(out_dir / "episodes.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["traj_id"]] = r
    items = []
    for traj in d["trajectories"]:
        rec = recs[traj["traj_id"]]
        cands = []
        for p in traj["positions"]:
            orig = rec["actions"][p["t"]]
            for b in p["alts"]:
                cands.append({"branch_id": b, "t": p["t"], "orig": orig, "alt": d["branches"][b]["action"]})
        rng = random.Random(hashlib.sha256(f"judge|{traj['traj_id']}".encode()).hexdigest())
        rng.shuffle(cands)
        items.append({"traj_id": traj["traj_id"], "cands": cands, "messages": render(rec, cands, executor_info)})
    return d, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=["vllm", "api", "mock"])
    ap.add_argument("--model", default=None,
                    help="API Model Name：qwen3.8-flash（default）、deepseek-flash etc（Automatically select a server by name）")
    ap.add_argument("--effort", default=None, help="The intensity of thinking（By default of the service provider：DeepSeek high，Qwen3.8 xhigh）")
    ap.add_argument("--no-thinking", action="store_true", help="API Judge closes the mode of thinking")
    ap.add_argument("--no-executor-info", action="store_true",
                    help="Take it easy：Get rid of the prompts“The implementer is the temperature 0.7 of 7B model、What can you see?”this sentence，The rest is exactly the same")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None, help="Overwrite the default output ceiling（Think mode default 8192）")
    ap.add_argument("--retry-invalid", action="store_true",
                    help="Question only the existence of trajectories that do not give a valid choice in the outcome document（if reasoning is truncated），Overwrite original file after consolidation")
    args = ap.parse_args()
    d, items = build(args.out_dir, executor_info=not args.no_executor_info)
    if args.dry_run:
        m = items[0]["messages"]
        print(m[0]["content"] + "\n\n" + m[1]["content"])
        print(f"\n[{len(items)} trajectories，Every single one {len(items[0]['cands'])} A candidate]")
        return 0
    judge = make_judge(args.judge, model=args.model, model_dir=config.MODEL_DIR, thinking=not args.no_thinking,
                       effort=args.effort)
    if args.no_executor_info:
        judge.name += "_noexec"
        judge.settings["executor_info"] = False
    if args.max_tokens:
        if hasattr(judge, "max_tokens"):
            judge.max_tokens = args.max_tokens
        judge.settings["max_tokens"] = args.max_tokens
    path = args.out_dir / f"judge_{judge.name}.jsonl"
    old = {}
    if args.retry_invalid:
        if not path.exists():
            sys.exit(f"no existing results {path}")
        old = {r["traj_id"]: r for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}
        items = [it for it in items if old.get(it["traj_id"], {}).get("branch_id") is None]
        print(f"[judge] Again {len(items)} The rule is not answered")
        if getattr(judge, "seed", None) is not None:      # The same prompts will be given when the torrent is fixed：Ask again for a new seed
            judge.seed += 1
            judge.settings["seed"] = judge.seed
    outs = ask_resumable(judge, [it["traj_id"] for it in items], [it["messages"] for it in items],
                         args.out_dir / "judge_cache" / f"{judge.name}.jsonl")
    rows = []
    for it, o in zip(items, outs):
        k = parse_choice(o["text"], len(it["cands"]))
        rows.append({"traj_id": it["traj_id"], "judge": judge.name, "settings": judge.settings,
                     "design_id": d["design_id"], "choice": None if k is None else k + 1,
                     "branch_id": None if k is None else it["cands"][k]["branch_id"],
                     "text": o["text"], "finish_reason": o.get("finish_reason"), "error": o.get("error"),
                     "usage": o.get("usage")})
    if args.retry_invalid:
        for r in rows:
            r["retried"] = True
            old[r["traj_id"]] = r
        rows = [old[it_id] for it_id in sorted(old)]
    write_jsonl(path, rows)
    ok = sum(r["branch_id"] is not None for r in rows)
    print(f"[judge] {judge.name}：{ok}/{len(rows)} The bar gives us a valid choice -> {path}")
    errs = [r["error"] for r in rows if r["error"]]
    if errs:
        print(f"[judge] Call Error {len(errs)} records，for example：{errs[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
