"""real ALFWorld（TextWorld PDDL Backend）Encapsulation。

Basis alfworld 0.4.x / textworld 1.6–1.7 Source（2026-10 Check）：
- environment = PddlEnv + AlfredDemangler(shuffle=False) + AlfredInfos，and alfworld Consistency of packaging at evaluation（Non-expert packaging）；
- Back every step admissible_commands Sorted；The first rule of syntax for text generation（No Random）；
- The order of item listing is from Python Gather all over the place，Dependence PYTHONHASHSEED（So it has to be fixed）；
- planner：PddlState.replan() Call from current status fast-downward（g greedy with respect to the value = Unanimous cost search，Cost：Most Actions 1，
  heat/cool/clean/use is 5），Returns the template command。Well just call it，Not dependent AlfredExpert Packaging。

Two optional acceleration patches（stage A2 verify that outputs remain unchanged）：
- translate：cache fast_downward.pddl2sas（reset translation is repeated on every call PDDL）；
- grammar：Parse result of cached text syntax（Reconstruct the template for every executable action at every step）。
"""
import copy
import hashlib
import time
from collections import OrderedDict

_PATCHES = {}


def _cached_parser(orig):
    """tatsu Parsing results cache by parameters；Caller may rewrite the returned object（# Like a symbol # context），therefore return a deep copy each time。"""
    cache = {}

    def cached(*args, **kwargs):
        if kwargs.get("trace"):
            return orig(*args, **kwargs)
        try:
            key = (args, tuple(sorted(kwargs.items())))
            hit = cache.get(key)
        except TypeError:                        # Non-Hashis Arguments：No Cache
            return orig(*args, **kwargs)
        if hit is None:
            hit = orig(*args, **kwargs)
            cache[key] = hit
        return copy.deepcopy(hit)

    return cached


def _cached_translate(orig, maxsize):
    """PDDL -> SAS Translating results by（domain, problem, optimize）cache。Returned task Objects read only。"""
    cache = OrderedDict()

    def cached_pddl2sas(domain, problem, verbose=False, optimize=False):
        key = hashlib.sha1(f"{optimize}\0{domain}\0{problem}".encode()).hexdigest()
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            return hit
        out = orig(domain, problem, verbose=verbose, optimize=optimize)
        cache[key] = out
        if len(cache) > maxsize:
            cache.popitem(last=False)
        return out

    return cached_pddl2sas


def install_patches(translate=True, grammar=True, maxsize=64):
    """installation / Unmount Accelerating Patches。Return to Current Status。"""
    import fast_downward
    from textworld.envs.pddl import textgen

    if translate and "translate" not in _PATCHES:
        orig_translate = fast_downward.pddl2sas
        fast_downward.pddl2sas = _cached_translate(orig_translate, maxsize)
        _PATCHES["translate"] = (fast_downward, "pddl2sas", orig_translate)
    elif not translate and "translate" in _PATCHES:
        mod, name, orig = _PATCHES.pop("translate")
        setattr(mod, name, orig)

    if grammar and "grammar" not in _PATCHES:
        import textworld.logic as twlogic
        saved = []
        for mod in (textgen, twlogic):           # Text Template Parsing、Conditional Expression Resolution（Every move is called）
            orig_parse = mod._parse_and_convert
            mod._parse_and_convert = _cached_parser(orig_parse)
            saved.append((mod, "_parse_and_convert", orig_parse))
        _PATCHES["grammar"] = saved
    elif not grammar and "grammar" in _PATCHES:
        for mod, name, orig in _PATCHES.pop("grammar"):
            setattr(mod, name, orig)
    return sorted(_PATCHES)


def _py313_compat():
    """Python ≥ 3.13（PEP 667）under locals().update() It doesnt matter anymore eval，textworld of EvalSymbol.derive Ill report it NameError。
    only in 3.13+ Replace with Equivalent：Expression in Module Global + Rendering in Template Variables（Templates refer only to these variables，e.g r.name）。"""
    import sys
    if sys.version_info < (3, 13):
        return False
    from textworld.envs.pddl import textgen
    if getattr(textgen.EvalSymbol, "_rt_compat", False):
        return True

    def derive(self, context=None):
        context = context or self.context
        value = eval(self.expression, vars(textgen), dict(context["variables"]))
        return [textgen.TerminalSymbol(value)]

    textgen.EvalSymbol.derive = derive
    textgen.EvalSymbol._rt_compat = True
    return True


def use_data_disk_tmp():
    """Pointing this process to a temporary directory acfg.TMP（Data Disk）。Keep default if directory cannot be created。Returns the directory that was actually used。"""
    import tempfile
    from . import acfg
    try:
        acfg.TMP.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(acfg.TMP)
    except OSError:
        pass
    return tempfile.gettempdir()


def lib_size_mb():
    """libdownward.so Size（One for each environmental example）。"""
    import os
    try:
        from fast_downward.interface import DOWNWARD_LIB_PATH
        return os.path.getsize(DOWNWARD_LIB_PATH) / 1e6
    except Exception:
        return None


def pack(state):
    return {"obs": state["feedback"], "admissible": list(state["admissible_commands"]),
            "won": bool(state["won"])}


class AlfEnv:
    """An example of an environment that can be reused：load(Game) → reset() → step(cmd)…；Reuse the same game fast-downward Copy of Library。"""
    kind = "tw"

    def __init__(self):
        import textworld
        from textworld.envs.pddl import PddlEnv
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        _py313_compat()
        use_data_disk_tmp()
        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        self._base = PddlEnv(infos)                       # One copy of each example libdownward.so
        self.env = AlfredInfos(AlfredDemangler(shuffle=False)(self._base))
        self.gamefile = None
        self.n_loads = 0

    def load(self, gamefile):
        if gamefile != self.gamefile:
            self.env.load(gamefile)
            self.gamefile = gamefile
            self.n_loads += 1

    def reset(self):
        return pack(self.env.reset())

    def step(self, cmd):
        st, _score, _done = self.env.step(cmd)
        return pack(st)

    def plan(self):
        """Current status to target planner command sequence；Return when the target has been achieved or no solution has been reached []。"""
        p = self._base._pddl_state
        return list(p.replan(self._base._entity_infos))

    def state_key(self):
        """The state of the whole world（All the facts）short hash，To determine whether or not to return to the status quo ante。"""
        facts = sorted(str(f) for f in self._base._pddl_state.facts)
        return hashlib.sha1("\n".join(facts).encode()).hexdigest()[:16]


def plan_cost(plan):
    """Cost calibre of the planners：heat/cool/clean/use is 5，The rest is 1。"""
    return sum(5 if c.split(" ", 1)[0] in ("heat", "cool", "clean", "use", "slice") else 1 for c in plan)


def make_env(kind):
    if kind == "tw":
        return AlfEnv()
    if kind == "mock":
        from .mockenv import MockEnv
        return MockEnv()
    raise ValueError(kind)


def transcript_hash(st):
    """Fingerprints to be exported one by one：Observation、List of executable commands（Order）、Yes or no。"""
    h = hashlib.sha1()
    h.update(st["obs"].encode())
    h.update(b"\x1f")
    h.update("\x1e".join(st["admissible"]).encode())
    h.update(b"\x1f1" if st["won"] else b"\x1f0")
    return h.hexdigest()[:16]


def replay(env, gamefile, cmds, want_key=False, plan_at=()):
    """load + reset + Execute in turn cmds。Back Gradual Fingerprints（Include initial status）、Status Keys、Timed with plan_at Planning of the Office。"""
    t0 = time.perf_counter()
    env.load(gamefile)
    t1 = time.perf_counter()
    st = env.reset()
    t2 = time.perf_counter()
    hashes = [transcript_hash(st)]
    keys = [env.state_key()] if want_key else []
    plans = {}
    step_times = []
    for i, c in enumerate(cmds):
        if i in plan_at:
            plans[i] = env.plan()
        ts = time.perf_counter()
        st = env.step(c)
        step_times.append(time.perf_counter() - ts)
        hashes.append(transcript_hash(st))
        if want_key:
            keys.append(env.state_key())
    if len(cmds) in plan_at:
        plans[len(cmds)] = env.plan()
    return {"hashes": hashes, "keys": keys, "plans": plans, "t_load": t1 - t0, "t_reset": t2 - t1,
            "step_times": step_times, "final": st}
