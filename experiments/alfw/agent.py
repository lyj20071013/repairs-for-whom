"""ALFWorld agent: prompts, action parsing and generation backends.

Intervention history contains actions and environment feedback, never past
Thought text. Replacing an action therefore yields the same subsequent history
as choosing that alternative originally, without contradictory old reasoning.

Prompts list executable commands and request one verbatim action. Invalid
commands produce "Nothing happens." and consume a step, as in the standard
environment. The legacy alias "put X in/on Y" maps to "move X to Y" only when
the mapped command is executable.

The 2026-10-23 protocol correction followed the pre-specified feasibility rules.
Initial A4 runs had 36% non-executable first responses, mostly state-tracking
errors (including redundant movement and actions at the wrong location).
The correction exposes location and held-object state reconstructed only from
visible feedback, states action constraints, and permits up to MAX_RETRIES
replies for invalid or missing actions before sending the final action to the
environment. Retries do not consume environment steps. Their immediate context
contains the rejected response, but later history still contains actions and
feedback only.
"""
import hashlib
import random
import re
from types import SimpleNamespace

from . import acfg

SYSTEM_PROMPT = """You are an agent in a text-based household environment. Each turn you see your task, the steps you have taken so far with what happened, and the list of actions that are possible right now. Choose exactly one of them.

How the environment works:
- "go to X" takes you to receptacle X and shows what is on or in it. A closed receptacle (cabinet, drawer, fridge, microwave, safe, ...) must be opened with "open X" before you can see or take what is inside.
- "take A from X" picks up object A; you can hold only one object at a time. "move A to X" puts the object you are holding in or on receptacle X; you must be at X.
- To heat an object, hold it, go to a microwave and use "heat A with microwave N". To cool it, use "cool A with fridge N" at the fridge. To clean it, use "clean A with sinkbasin N" at the sinkbasin.
- To look at an object under a lamp, hold the object, go to where the lamp is and "use desklamp N" (or "use floorlamp N").
- If the task says "two", put two such objects in the target receptacle, one at a time.
- Only the actions listed under "Actions possible now" can be done in the current state. For example, you cannot "go to" the place where you already are; you can take, move, heat, cool, clean or use things only at your current location; you can take something only when your hands are empty; and you can heat, cool, clean or move only the object you are holding. If what you want to do is not in the list, first do what makes it possible (for example, go to the right place).
- An action that is not in the list does nothing ("Nothing happens.") but still uses up a step. You have at most {max_steps} steps.

Answer in exactly this format:
Thought: <one or two short sentences: what you know so far and what to do next>
Action: <one action copied exactly from the list>"""

_WELCOME_RE = re.compile(r"-=\s*Welcome to TextWorld, ALFRED!\s*=-\s*", re.I)
_TASK_RE = re.compile(r"Your task is to:\s*(.+)", re.I | re.S)


def parse_intro(obs):
    """Initial observations -> (Tasks, Description of the room)。"""
    text = _WELCOME_RE.sub("", obs).strip()
    m = _TASK_RE.search(text)
    if not m:
        return "", text
    task = m.group(1).strip()
    desc = text[:m.start()].strip()
    return task, desc


MAX_RETRIES = 2
RETRY_MSG = ('"{cmd}" is not possible right now. Choose exactly one action from the list "Actions possible now" above. '
             "Write the Thought line, then the Action line.")

_ARRIVE_RE = re.compile(r"You arrive at (.+?)\.")
_PICK_RE = re.compile(r"(?:^|: )You pick up the (.+?)(?: from the .+?)?\.")


def track_state(history):
    """Rolled out by environmental feedback text (Current Location, Handhelds)。Use only the information that the intelligence itself sees。"""
    loc, hold = None, None
    for h in history:
        o = h.get("obs") or ""
        m = _ARRIVE_RE.match(o)
        if m:
            loc = m.group(1)
            continue
        m = _PICK_RE.search(o)
        if m:
            hold = m.group(1)
            continue
        if o.startswith("You move the") or "You put the" in o:
            hold = None
    return loc, hold


def build_messages(task, desc, history, admissible, max_steps=None, n_drop=0):
    """history: [{"cmd", "obs"}]。n_drop>0 The first one was omitted n_drop steps（Use only when there is insufficient context）。

    Part of the structure that changes the number of steps（Number of steps used、Executable List）Put it at the end，System prompts in front、The mission and history are two steps from one side to the other
    Keep as same prefix，Easy vLLM Prefix Cache。
    """
    max_steps = max_steps or acfg.MAX_STEPS
    lines = []
    if n_drop > 0:
        lines.append(f"(steps 1-{n_drop} omitted)")
    for i, h in enumerate(history[n_drop:], start=n_drop + 1):
        lines.append(f"{i}. > {h['cmd'] or '(no action)'}\n{h['obs']}")
    hist = "\n".join(lines) if lines else "(none yet)"
    acts = "\n".join(admissible)
    loc, hold = track_state(history)
    user = (f"Your task is to: {task}\n\n{desc}\n\nSteps so far:\n{hist}\n\n"
            f"You are now at: {loc or 'the middle of the room'}. You are holding: {hold or 'nothing'}.\n"
            f"You have used {len(history)} of {max_steps} steps.\n"
            f"Actions possible now:\n{acts}\n\n"
            "Write the Thought line, then the Action line.")
    return [{"role": "system", "content": SYSTEM_PROMPT.format(max_steps=max_steps)},
            {"role": "user", "content": user}]


_ACTION_RE = re.compile(r"^\s*\**\s*action\s*\**\s*[:：]\s*\**[ \t]*(.*)$", re.I | re.M)
_ACTION_LOOSE_RE = re.compile(r"\baction\s*\**\s*[:：]\s*\**[ \t]*(.*)$", re.I | re.M)
_PUT_RE = re.compile(r"^put (.+?) (?:in/on|into|in|on) (.+)$")


def normalize(cmd):
    c = cmd.strip()
    c = c.strip("`'\"*[]<> ")
    c = re.sub(r"[.。!]+$", "", c).strip()
    if c.startswith(">"):
        c = c[1:].strip()
    return re.sub(r"\s+", " ", c).lower()


def parse_action(text, admissible):
    """Back {"cmd": Orders issued to the environment, "kind": valid|alias|invalid|no_action}。Take the first one Action rows。"""
    m = _ACTION_RE.search(text or "") or _ACTION_LOOSE_RE.search(text or "")
    if not m:
        return {"cmd": "", "kind": "no_action"}
    raw = m.group(1)
    if not normalize(raw):              # “Action:” Write orders after line change
        rest = [ln for ln in (text[m.end():].splitlines()) if ln.strip()]
        raw = rest[0] if rest else ""
    cmd = normalize(raw)
    if not cmd:
        return {"cmd": "", "kind": "no_action"}
    adm = set(admissible)
    if cmd in adm:
        return {"cmd": cmd, "kind": "valid"}
    p = _PUT_RE.match(cmd)
    if p:
        alt = f"move {p.group(1)} to {p.group(2)}"
        if alt in adm:
            return {"cmd": alt, "kind": "alias"}
    return {"cmd": cmd, "kind": "invalid"}


def derive_seed(*parts):
    """and policies.derive_seed Same definition：Any field is definitely mapd to [0, 2^31)。"""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


# ---------------------------------------------------------------- Backend
class VLLMBackend:
    """freeze Qwen2.5-7B-Instruct。Each request is separate SamplingParams（Seed-to-Request）；temperature 0 For greedy。"""
    name = "vllm"

    def __init__(self, model_dir=None, context=None):
        import os
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams
        self._SP = SamplingParams
        self.context = context or acfg.MODEL_CONTEXT
        self.llm = LLM(model=str(model_dir or acfg.MODEL_DIR), dtype="bfloat16", gpu_memory_utilization=0.90,
                       max_model_len=self.context, enable_prefix_caching=True, disable_log_stats=True,
                       seed=acfg.MODEL_SEED)
        self.tok = self.llm.get_tokenizer()

    def render(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count(self, prompt):
        return len(self.tok(prompt, add_special_tokens=False).input_ids)

    def generate(self, prompts, temperature, seeds, max_new_tokens, hints=None):
        params = []
        for s in seeds:
            kw = dict(temperature=temperature, top_p=acfg.TOP_P, max_tokens=max_new_tokens, stop=acfg.STOP)
            if temperature > 0:
                kw["seed"] = s
            params.append(self._SP(**kw))
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        res = []
        for o in outs:
            g = o.outputs[0]
            res.append(SimpleNamespace(text=g.text, finish_reason=getattr(g, "finish_reason", None),
                                       n_prompt=len(o.prompt_token_ids or []), n_gen=len(g.token_ids or [])))
        return res


class MockBackend:
    """Offline test：does not load a model。Actions are selected by torrent from the executable list attached to the prompts，Preferences take/move/heat；
    A small output is intentionally written out of nothing Action rows、Old Syntax: put、Or an unlawful order，To overwrite the parse branch。"""
    name = "mock"

    def __init__(self, context=None):
        self.context = context or acfg.MODEL_CONTEXT

    def render(self, messages):
        return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages) + "\n<assistant>\n"

    def count(self, prompt):
        return len(prompt) // 4

    @staticmethod
    def _choose(prompt, adm, rng):
        return _mock_choose(prompt, adm, rng)

    def generate(self, prompts, temperature, seeds, max_new_tokens, hints=None):
        res = []
        for i, (p, s) in enumerate(zip(prompts, seeds)):
            adm = hints[i] if hints else []
            rng = random.Random(s if temperature > 0 else derive_seed(p))
            u = rng.random()
            fr = "stop"
            if u < 0.03:
                text = "Thought: I am not sure what to do."
            elif u < 0.06:
                text = "Thought: try something.\nAction: fly to the moon"
            elif u < 0.08:
                text = "Thought: I keep thinking " + "and thinking " * 200
                fr = "length"
            else:
                a = self._choose(p, adm, rng)
                if a.startswith("move ") and rng.random() < 0.3:
                    o, r = a[5:].split(" to ")
                    a = f"put {o} in/on {r}"
                text = f"Thought: mock step.\nAction: {a}"
            res.append(SimpleNamespace(text=text, finish_reason=fr, n_prompt=self.count(p),
                                       n_gen=len(text) // 4))
        return res


def _mock_choose(prompt, adm, rng):
    """Read the object and target container in the mission：Get the target、（When needed）Heated、Put it in the target；Or Ill walk randomly/Open it。"""
    m = re.search(r"Your task is to: (?:put |heat )?(?:some|a) (\w+)(?: and put it)? in (\w+)", prompt)
    heat = "Your task is to: heat" in prompt
    if m and rng.random() < 0.8:
        obj, goal = m.group(1), m.group(2)
        hot = f"You heat the {obj}" in prompt
        for a in adm:
            if a.startswith(f"take {obj} "):
                return a
            if heat and not hot and a.startswith(f"heat {obj} "):
                return a
            if a.startswith(f"move {obj} ") and a.endswith(f" to {goal} 1") and (hot or not heat):
                return a
        holding = any(a.startswith(f"move {obj} ") or a.startswith(f"heat {obj} ") for a in adm)
        if holding:
            want = "microwave 1" if heat and not hot else f"{goal} 1"
            for a in adm:
                if a == f"go to {want}":
                    return a
    opens = [a for a in adm if a.startswith("open ")]
    if opens and rng.random() < 0.6:
        return rng.choice(opens)
    gos = [a for a in adm if a.startswith("go to ")]
    if gos and rng.random() < 0.8:
        return rng.choice(gos)
    return rng.choice(adm) if adm else "look"


def make_backend(kind, model_dir=None):
    if kind == "mock":
        return MockBackend()
    return VLLMBackend(model_dir=model_dir)
