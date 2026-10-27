"""WebShop Up strategy：prompt、Action Parsing、Simulation backend for offline testing。Generate backend with ALFWorld identical（alfw.agent.VLLMBackend）。

Interference agreements and ALFWorld identical：History in the prompts contains only“action + Environmental feedback”，It doesnt contain a model Thought；
Unexecutable or with feedback when no action is taken，At most MAX_RETRIES runs，Non-consumption of step；send to the environment only if still not executable after retries（Environmental non-implementation、Expend one step）。
"""
import hashlib
import random
import re
from types import SimpleNamespace

from . import wcfg

SYSTEM_PROMPT = """You are shopping in a web store. You are given an instruction that describes a product to buy. Each turn you see the instruction, the steps you have taken so far with the resulting pages, the current page, and the actions that are possible now. Choose exactly one of them.

How the store works:
- "search[keywords]" searches the store and is possible only on the search page. Replace keywords with your own query.
- "click[...]" clicks something on the current page: a button such as "back to search", "next >", "< prev", "description", "features", "reviews" or "buy now"; a product ID in the search results, which opens the product page; or an option on a product page (size, color, ...). Copy the text inside the brackets exactly from the list.
- On a product page, select every option the instruction asks for before buying. "buy now" buys the product with the options selected so far and ends the episode. Buying a product that matches the instruction, with the right options and within the price limit, is a success; anything else is a failure.
- An action that is not in the list does nothing but still uses up a step. You have at most {max_steps} steps; if you have not bought anything by then, you fail.

Answer in exactly this format:
Thought: <one or two short sentences: what you know so far and what to do next>
Action: <one action from the list>"""

MAX_RETRIES = 2
RETRY_MSG = ('"{cmd}" is not possible right now. Choose exactly one action from the list "Actions possible now" above. '
             "Write the Thought line, then the Action line.")
SEARCH_ITEM = "search[<keywords>]"


def clean_obs(obs, instruction):
    """Remove the name and instructions that are repeated at the beginning of each page（The prompts starts with an order）；Keep the rest as it is。"""
    lines = (obs or "").split("\n")
    out, skipped_instr = [], False
    for ln in lines:
        s = ln.strip()
        if s in ("WebShop", "Instruction:", "Instruction: "):
            continue
        if not skipped_instr and instruction and s == instruction.strip():
            skipped_instr = True
            continue
        out.append(ln.rstrip())
    text = "\n".join(x for x in out if x.strip())
    return text


def _cut(text, n):
    return text if len(text) <= n else text[:n].rstrip() + " ..."


def action_list(state):
    acts = [SEARCH_ITEM] if state.get("search") else []
    return acts + [f"click[{c}]" for c in state.get("clicks", [])]


def build_messages(instruction, history, state, max_steps=None, n_drop=0):
    """history: [{"cmd", "obs"}]（obs already clean_obs）；state：Current status（Ham obs Original）。"""
    max_steps = max_steps or wcfg.MAX_STEPS
    lines = []
    if n_drop > 0:
        lines.append(f"(steps 1-{n_drop} omitted)")
    for i, h in enumerate(history[n_drop:], start=n_drop + 1):
        lines.append(f"{i}. > {h['cmd'] or '(no action)'}\n{_cut(h['obs'], wcfg.OBS_CHARS_HISTORY)}")
    hist = "\n".join(lines) if lines else "(none yet)"
    cur = _cut(clean_obs(state["obs"], instruction), wcfg.OBS_CHARS_CURRENT)
    acts = "\n".join(action_list(state))
    user = (f"Instruction: {instruction}\n\nSteps so far:\n{hist}\n\nCurrent page:\n{cur}\n\n"
            f"You have used {len(history)} of {max_steps} steps.\n"
            f"Actions possible now:\n{acts}\n\n"
            "Write the Thought line, then the Action line.")
    return [{"role": "system", "content": SYSTEM_PROMPT.format(max_steps=max_steps)},
            {"role": "user", "content": user}]


_ACTION_RE = re.compile(r"^\s*\**\s*action\s*\**\s*[:：]\s*\**[ \t]*(.*)$", re.I | re.M)
_ACTION_LOOSE_RE = re.compile(r"\baction\s*\**\s*[:：]\s*\**[ \t]*(.*)$", re.I | re.M)
_CALL_RE = re.compile(r"^(search|click)\s*\[(.*)\]$", re.I | re.S)
_SPECIAL = ("<r>", "<a>", "<c>", "<q>")


def _norm(s):
    s = s.strip().strip("`'\"*").strip()
    s = re.sub(r"[.。]+$", "", s).strip()
    return re.sub(r"\s+", " ", s)


def parse_action(text, state):
    """Back {"cmd": Rationale for distribution to the environment, "kind": valid|alias|invalid|no_action}。Take the first one Action rows。"""
    m = _ACTION_RE.search(text or "") or _ACTION_LOOSE_RE.search(text or "")
    if not m:
        return {"cmd": "", "kind": "no_action"}
    raw = m.group(1)
    if not _norm(raw):
        rest = [ln for ln in text[m.end():].splitlines() if ln.strip()]
        raw = rest[0] if rest else ""
    raw = _norm(raw)
    if not raw:
        return {"cmd": "", "kind": "no_action"}
    clicks = set(state.get("clicks", []))
    c = _CALL_RE.match(raw)
    if c:
        name, arg = c.group(1).lower(), _norm(c.group(2))
        if name == "search":
            cmd = f"search[{arg.lower()}]"
            ok = bool(state.get("search")) and bool(arg) and arg != "<keywords>" and arg.split()[0].lower() not in _SPECIAL
            return {"cmd": cmd, "kind": "valid" if ok else "invalid"}
        cmd = f"click[{arg.lower()}]"
        return {"cmd": cmd, "kind": "valid" if arg.lower() in clicks else "invalid"}
    low = raw.lower()                     # missing square brackets：It just happens to be a clickable item to click on
    if low.startswith("click "):
        low = low[6:].strip()
    if low in clicks:
        return {"cmd": f"click[{low}]", "kind": "alias"}
    return {"cmd": raw, "kind": "invalid"}


def derive_seed(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


class MockBackend:
    """Offline test：does not load a model。Read the command and executable list in the prompts：Well search the words in the order first，Point the first commodity，Options mentioned in the point command，Ill buy it again；
    A small output is intentionally written out of nothing Action rows、Invalid action or absence of square brackets，To overwrite the parse branch。"""
    name = "mock"

    def __init__(self, context=8192):
        self.context = context

    def render(self, messages):
        return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages) + "\n<assistant>\n"

    def count(self, prompt):
        return len(prompt) // 4

    def generate(self, prompts, temperature, seeds, max_new_tokens, hints=None):
        res = []
        for i, (p, s) in enumerate(zip(prompts, seeds)):
            st = hints[i] if hints else {"search": True, "clicks": []}
            rng = random.Random(s if temperature > 0 else derive_seed(p))
            u = rng.random()
            if u < 0.03:
                text = "Thought: not sure."
            elif u < 0.06:
                text = "Thought: try.\nAction: click[the moon]"
            else:
                a = _mock_choose(p, st, rng)
                if a.startswith("click[") and rng.random() < 0.05:
                    a = a[6:-1]                     # missing square brackets
                text = f"Thought: mock step.\nAction: {a}"
            res.append(SimpleNamespace(text=text, finish_reason="stop", n_prompt=self.count(p), n_gen=len(text) // 4))
        return res


def _mock_choose(prompt, st, rng):
    m = re.search(r"Instruction: (.+)", prompt)
    instr = (m.group(1) if m else "").lower()
    clicks = st.get("clicks", [])
    if st.get("search"):
        words = [w for w in re.findall(r"[a-z]+", instr) if len(w) > 3][:4]
        return f"search[{' '.join(words) or 'product'}]"
    prods = [c for c in clicks if re.fullmatch(r"b[0-9a-z]{9}", c)]
    if prods:
        if rng.random() < 0.15 and "next >" in clicks:
            return "click[next >]"
        return f"click[{prods[0] if rng.random() < 0.7 else rng.choice(prods)}]"
    opts = [c for c in clicks if c not in ("back to search", "< prev", "next >", "description", "features", "reviews",
                                           "buy now", "attributes")]
    want = [o for o in opts if o in instr and f"click[{o}]" not in prompt.split("Current page:")[0]]
    if want and rng.random() < 0.8:
        return f"click[{want[0]}]"
    if "buy now" in clicks and rng.random() < 0.6:
        return "click[buy now]"
    return f"click[{rng.choice(clicks)}]" if clicks else "search[product]"
