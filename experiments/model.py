"""
vLLM Encapsulation + Interlacing batch processing rollout。

Key design：all rollout Synchronize Progress。Every step“Unfinished”all rollout Generate together，
Not one line。7200 runs rollout x 12 If the steps are called by a string, 8 More than a million times，
a single GPU requires more than ten hours；Batch down to date 12 Generation of batches，2-3 The hours over。

Policy shape：One word at a time Thought，One more JSON Tool Call（Parsing last JSON）。
system prompt contains one record 5x5 worked example（Different layouts，Show them four stages
Thought→Action Mode），Help 7B Create“Read observations→The sentencing phase→Select Action”loop。
example The direction is different from the actual map，Reduce the side effects of a copy-opening。
"""
import json
import os
import re
from typing import Any, Dict, List

# —— You have to import vllm Previous Settings ——
# FlashInfer of sampler kernel and part CUDA toolchain Incompatible，causes vLLM Failed to start。
# Turn it off, it only affects the sampling tossing（Fallback PyTorch Native top-p kernel），Do not change sampling distribution，
# It doesnt matter V(h_t) syntax of the syntax，affects speed only。
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import config
from env import GridWorld

_INTRO_THINK = """You control an agent in a gridworld. Each turn you read one observation and answer in exactly this format:
Thought: <one short sentence: your current stage, where its target is, and what your chosen move leads to>
Action: {"tool": "<name>", "arg": "<optional>"}"""

_INTRO_JSON = """You control an agent in a gridworld. At each step output EXACTLY ONE tool call as a JSON object, nothing else.
Format: {"tool": "<name>", "arg": "<optional>"}"""

_READING = """How to read the observation:
- 'RESULT OF YOUR LAST ACTION' tells whether your previous action worked. If it was blocked, you did not move.
- 'Targets relative to you' gives the direction and distance from you to each remaining target. Use it instead of counting cells on the map.
- 'What each move leads to' tells what each direction would do this turn: a wall, the map edge or a locked door blocks you; a TRAP ends the episode."""

_TASK = """Task order. Check 'Holding key', 'Door open' and 'Holding gem' every turn to see which stage you are in:
1. Door locked and no key: walk to the key, then pick() while standing on it.
2. Holding the key: walk until the door shows 'within reach of open()', then open().
3. Door open and no gem: walk to the gem, then pick() while standing on it.
4. Holding the gem: walk back to the start cell (row 0, col 0).
To walk toward a target, choose a direction named in 'Targets relative to you' whose move leads to an empty cell or to the target.
If every such move is blocked or a trap, step sideways around the obstacle. Do not walk back and forth between two cells."""

_RULES_THINK = """Rules:
- Write exactly the two lines above: one short Thought (at most about 25 words), then the Action JSON. No markdown, no code fence.
- You have limited steps. Waste none.
- After a blocked or unsuccessful action, use the actual position and the result to correct your next action."""

_RULES_JSON = """Rules:
- Output only the JSON object. No explanation, no markdown, no code fence.
- You have limited steps. Waste none.
- After a blocked or unsuccessful action, use the actual position and the result to correct your next action."""

# 5x5 worked example. Key location differs from actual maps to prevent rote copying.
_EXAMPLE = """Example trajectory (key at top-right; your actual map will differ — learn the PATTERN, not the exact moves):

=== Turn 1 — Stage 1: Need key ===
Position: row 0, col 0 | Holding key: False | Door open: False | Holding gem: False
RESULT OF YOUR LAST ACTION: Episode just started.
Targets relative to you:
- key at (row 0, col 3): 3 columns right
- locked door at (row 4, col 3): 4 rows down and 3 columns right — too far to open()
What each move leads to:
- right: empty | down: empty

Thought: Stage 1—need key. Key is 3 right. Move right toward key.
Action: {"tool": "move", "arg": "right"}

[Turn 2-3: continue moving right...]

=== Turn 4 — Pick up key ===
Position: row 0, col 3 | Holding key: False
RESULT OF YOUR LAST ACTION: Moved right to (row 0, col 3).
Under your feet: key (call pick() while standing here)

Thought: Standing on key. Call pick() to grab it.
Action: {"tool": "pick"}

=== Turn 5 — Stage 2: Go to door ===
Position: row 0, col 3 | Holding key: True | Door open: False
RESULT OF YOUR LAST ACTION: Picked up the key. You are holding the key.
Targets relative to you:
- locked door at (row 4, col 3): 4 rows down — too far to open()

Thought: Stage 2—have key, door 4 down. Move down toward door.
Action: {"tool": "move", "arg": "down"}

[Turn 6-8: continue moving down...]

=== Turn 9 — Open door ===
Position: row 3, col 3 | Holding key: True | Door open: False
RESULT OF YOUR LAST ACTION: Moved down to (row 3, col 3).
Targets relative to you:
- locked door at (row 4, col 3): 1 row down — within reach of open()
What each move leads to:
- down: locked door (blocks movement; open() can remove it)

Thought: Door in reach and I hold key. Call open().
Action: {"tool": "open"}

=== Turn 10 — Stage 3: Go to gem ===
Position: row 3, col 3 | Holding key: False | Door open: True | Holding gem: False
RESULT OF YOUR LAST ACTION: Opened the door successfully. The key was consumed.
Targets relative to you:
- gem at (row 4, col 4): 1 row down and 1 column right
What each move leads to:
- down: empty | right: empty

Thought: Stage 3—door open, gem 1 down 1 right. Move down first.
Action: {"tool": "move", "arg": "down"}

=== Turn 11 ===
Position: row 4, col 3 | Door open: True | Holding gem: False
RESULT OF YOUR LAST ACTION: Moved down to (row 4, col 3).
Targets relative to you:
- gem at (row 4, col 4): 1 column right

Thought: Gem 1 right. Move right.
Action: {"tool": "move", "arg": "right"}

=== Turn 12 — Pick up gem ===
Position: row 4, col 4 | Door open: True | Holding gem: False
RESULT OF YOUR LAST ACTION: Moved right to (row 4, col 4).
Under your feet: gem (call pick() while standing here)

Thought: Standing on gem. Call pick().
Action: {"tool": "pick"}

=== Turn 13 — Stage 4: Return home ===
Position: row 4, col 4 | Door open: True | Holding gem: True
RESULT OF YOUR LAST ACTION: Picked up the gem. Carry it back to (row 0, col 0).
Targets relative to you:
- start cell (row 0, col 0): 4 rows up and 4 columns left
What each move leads to:
- up: wall (blocked) | left: empty

Thought: Stage 4—have gem, need start at 4 up 4 left. Up blocked, go left.
Action: {"tool": "move", "arg": "left"}

[Continue moving left and up, avoiding obstacles, until reaching (row 0, col 0) for success]"""


def build_system_prompt(think: bool = True) -> str:
    """think=False For use only --stage probe of the United Nations：The observations and rules are the same，Just take it off Thought。"""
    return "\n\n".join([
        _INTRO_THINK if think else _INTRO_JSON,
        GridWorld.TOOL_SPEC,
        _READING,
        _TASK,
        _EXAMPLE,
        _RULES_THINK if think else _RULES_JSON,
    ])


SYSTEM_PROMPT = build_system_prompt(think=True)

_JSON_RE = re.compile(r"\{[^{}]*\}")


# White list of tools and common spelling variants
_TOOL_ALIASES = {
    "move": "move", "walk": "move", "go": "move",
    "open": "open", "unlock": "open",
    "pick": "pick", "take": "pick", "grab": "pick", "collect": "pick",
    "look": "look", "observe": "look", "inspect": "look",
    "give_up": "give_up", "giveup": "give_up", "quit": "give_up",
    "abort": "give_up", "stop": "give_up", "finish": "give_up",
}
# arg Values Priority：Clear direction/Parameter Key Priority，"action"/"name" finally。
# otherwise {"action": "unlock"} may mistake the tool name for arg。
_ARG_KEYS = ("arg", "direction", "dir", "target", "value", "action", "name")


def _norm_arg(tool: str, arg):
    """convert the models common form arg Writing to formats that the environment understands。"""
    if arg is None:
        return None
    a = str(arg).strip().lower()
    if tool == "move":
        mapping = {
            "up": "up", "north": "up", "n": "up", "u": "up",
            "down": "down", "south": "down", "s": "down", "d": "down",
            "left": "left", "west": "left", "w": "left", "l": "left",
            "right": "right", "east": "right", "e": "right", "r": "right",
        }
        return mapping.get(a, a)
    return a


def _extract_action(text: str) -> Dict[str, Any]:
    """Pleasant Parsing。Parsing failed to return malformed，Use it as a step-by-step invalid。

    Take the first solution from the back、With a tool name JSON Object：Thought In front、Action In the back，
    Thought mentioned incidentally in JSON or {key} These braids dont look like moves。
    Pure JSON Output only one object，Act exactly the same as before。
    Known limits：if Action Walking max_tokens truncation，may still extract Thought Yeah JSON，
    Use in such cases generations Yeah finish_reason=length Check。
    """
    if not text:
        return {"tool": "malformed"}
    for candidate in reversed(_JSON_RE.findall(text)):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        raw_tool = obj.get("tool", obj.get("action", obj.get("name")))
        if raw_tool is None:
            continue
        tool = _TOOL_ALIASES.get(str(raw_tool).strip().lower(), str(raw_tool).strip().lower())

        arg = None
        for k in _ARG_KEYS:
            if k in obj and obj[k] is not None:
                arg = obj[k]
                break
        # If you get it "arg" is actually a tool name（The model wrote the tool name twice），Get rid of it
        if arg is not None and str(arg).strip().lower() in _TOOL_ALIASES:
            arg = None
        return {"tool": tool, "arg": _norm_arg(tool, arg)}
    return {"tool": "malformed", "raw": text[-80:]}


class AgentPolicy:
    """freeze Qwen2.5-7B-Instruct，As a fixed strategy。"""

    def __init__(self, model_dir=None, temperature=None, max_model_len=None):
        from vllm import LLM, SamplingParams
        model_dir = str(model_dir or config.MODEL_DIR)
        max_model_len = max_model_len or config.MODEL_CONTEXT
        self.temperature = config.TEMPERATURE if temperature is None else temperature
        self._make_params = SamplingParams
        self.cache_id = config.fingerprint({"run": config.run_id(), "model_dir": model_dir,
                                            "temperature": self.temperature, "context": max_model_len})
        self.llm = LLM(
            model=model_dir,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=max_model_len,
            enforce_eager=False,
            disable_log_stats=True,
            seed=config.MODEL_SEED,
        )
        self.tokenizer = self.llm.get_tokenizer()

    # ---------- prompt Assembly ----------
    @staticmethod
    def build_messages(obs: str, history: List[Dict[str, Any]], max_hist: int = 8,
                       think: bool = True):
        """Collection and continuation of shared actions and results history，Just for the most part max_hist records。

        History contains only actions and results of execution，It doesnt contain a model Thought：absent from the snapshot Thought，
        If you take it with you, you dont follow it，Same h_t The two ends of the input will be different。
        """
        tail = history[-max_hist:] if max_hist > 0 else []
        lines = []
        for h in tail:
            act = h["action"]
            arg = act.get("arg")
            s = act.get("tool", "")
            call = f"{s}({arg})" if arg else f"{s}()"
            lines.append(f"{call} -> {h.get('result', '(result not recorded)')}")
        hist_block = "\n".join(lines) if lines else "(none)"
        user = (
            f"Recent actions and their results:\n{hist_block}\n\n"
            f"Current observation:\n{obs}\n\n"
            + ("Write the Thought line, then the Action line." if think
               else "Output the next tool call as JSON.")
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT if think else build_system_prompt(think=False)},
            {"role": "user", "content": user},
        ]

    def _render(self, obs, history, think=True):
        msgs = self.build_messages(obs, history, think=think)
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def _params(self, temperature=None, max_new_tokens=None):
        temp = self.temperature if temperature is None else temperature
        return self._make_params(
            temperature=temp,
            top_p=config.TOP_P,
            max_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
        )

    # ---------- Bulk Step：collect、continuation、probe Shared ----------
    def act_batch(self, obs_list, histories, temperature=None, max_new_tokens=None, think=True):
        """A batch (obs, history) One generation。Back (actions, generations)，The order is the same as the input。"""
        if not obs_list:
            return [], []
        prompts = [self._render(o, h, think=think) for o, h in zip(obs_list, histories)]
        outs = self.llm.generate(prompts, self._params(temperature, max_new_tokens), use_tqdm=False)
        actions, generations = [], []
        for out in outs:
            gen = out.outputs[0]
            actions.append(_extract_action(gen.text))
            generations.append({"text": gen.text,
                                "finish_reason": getattr(gen, "finish_reason", None)})
        return actions, generations

    # ---------- Batch processing rollout ----------
    def run_rollouts(self, env_snaps, temperature=None, max_new_tokens=None):
        """
        Well start with multiple environmental snapshots at the same time，advance jobs in interleaved order until all finish。
        env_snaps: List[snapshot dict]
        Back: (results: List[bool], envs: List[GridWorld])
        ----------
        Attention：Sampling randomity comes entirely from vLLM of temperature，Dont pass it seed。
        Recoverability vLLM its internal within-batch randomness，does not guarantee per- token Recurrence——
        this is why the reported quantity is a Monte Carlo estimate V Not a single path。
        """
        envs = [GridWorld.restore(s) for s in env_snaps]
        obs = [e.observe() for e in envs]
        # Closed snapshots are out of circulation，It has to go straight to its final outcome（Otherwise, success will be counted False）
        results = [e.success for e in envs]
        alive = [i for i, e in enumerate(envs) if not e.done]

        while alive:
            acts, _ = self.act_batch([obs[i] for i in alive], [envs[i].history for i in alive],
                                     temperature=temperature, max_new_tokens=max_new_tokens)
            nxt_alive = []
            for i, act in zip(alive, acts):
                obs[i], finished = envs[i].step(act)
                if finished:
                    results[i] = envs[i].success
                else:
                    nxt_alive.append(i)
            alive = nxt_alive

        return results, envs

    # ---------- One step ----------
    def act(self, obs, history, temperature=None):
        """One-step reasoning。Returned after parsing action dict，Original output in last_generation。"""
        actions, generations = self.act_batch([obs], [history], temperature=temperature)
        self.last_generation = generations[0]
        return actions[0]
