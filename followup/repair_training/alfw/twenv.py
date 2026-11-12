"""真实 ALFWorld（TextWorld PDDL 后端）封装。

依据 alfworld 0.4.x / textworld 1.6–1.7 源码（2026-10 核对）：
- 环境 = PddlEnv + AlfredDemangler(shuffle=False) + AlfredInfos，与 alfworld 评测时的包装一致（不加专家包装）；
- 每步返回的 admissible_commands 已排序；文本生成总是取语法的第一条规则（无随机）；
- 物品列举顺序来自 Python 集合迭代，依赖 PYTHONHASHSEED（因此必须固定）；
- 规划器：PddlState.replan() 从当前状态调用 fast-downward（g 值贪心 = 一致代价搜索，代价：多数动作 1，
  heat/cool/clean/use 为 5），返回模板化命令。我们直接调用它，不依赖 AlfredExpert 包装。

两个可选加速补丁（阶段 A2 检验它们不改变任何输出）：
- translate：缓存 fast_downward.pddl2sas（reset 每次都会重新翻译 PDDL）；
- grammar：缓存文本语法的解析结果（每步对每个可执行动作都要重新解析模板）。
"""
import copy
import hashlib
import time
from collections import OrderedDict

_PATCHES = {}


def _cached_parser(orig):
    """tatsu 解析结果按参数缓存；调用方可能改写返回的对象（如符号的 context），所以每次返回深拷贝。"""
    cache = {}

    def cached(*args, **kwargs):
        if kwargs.get("trace"):
            return orig(*args, **kwargs)
        try:
            key = (args, tuple(sorted(kwargs.items())))
            hit = cache.get(key)
        except TypeError:                        # 不可哈希的参数：不缓存
            return orig(*args, **kwargs)
        if hit is None:
            hit = orig(*args, **kwargs)
            cache[key] = hit
        return copy.deepcopy(hit)

    return cached


def _cached_translate(orig, maxsize):
    """PDDL -> SAS 翻译结果按（domain, problem, optimize）缓存。返回的 task 对象只被读取。"""
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
    """安装 / 卸载加速补丁。返回当前状态。"""
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
        for mod in (textgen, twlogic):           # 文本模板解析、条件表达式解析（每步每个动作都会调用）
            orig_parse = mod._parse_and_convert
            mod._parse_and_convert = _cached_parser(orig_parse)
            saved.append((mod, "_parse_and_convert", orig_parse))
        _PATCHES["grammar"] = saved
    elif not grammar and "grammar" in _PATCHES:
        for mod, name, orig in _PATCHES.pop("grammar"):
            setattr(mod, name, orig)
    return sorted(_PATCHES)


def _py313_compat():
    """Python ≥ 3.13（PEP 667）下 locals().update() 不再影响 eval，textworld 的 EvalSymbol.derive 会报 NameError。
    只在 3.13+ 替换为等价写法：表达式在模块全局 + 模板变量中求值（模板只引用这些变量，如 r.name）。"""
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
    """把本进程的临时目录指向 acfg.TMP（数据盘）。目录无法创建时保持默认。返回实际使用的目录。"""
    import tempfile
    from . import acfg
    try:
        acfg.TMP.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(acfg.TMP)
    except OSError:
        pass
    return tempfile.gettempdir()


def lib_size_mb():
    """libdownward.so 的大小（每个环境实例占用一份）。"""
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
    """一个可重复使用的环境实例：load(游戏) → reset() → step(cmd)…；换游戏时复用同一个 fast-downward 库副本。"""
    kind = "tw"

    def __init__(self):
        import textworld
        from textworld.envs.pddl import PddlEnv
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        _py313_compat()
        use_data_disk_tmp()
        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        self._base = PddlEnv(infos)                       # 每个实例会复制一份 libdownward.so
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
        """当前状态到目标的规划器命令序列；已达成目标或无解时返回 []。"""
        p = self._base._pddl_state
        return list(p.replan(self._base._entity_infos))

    def state_key(self):
        """完整世界状态（全部事实）的短哈希，用于判断是否回到已到过的状态。"""
        facts = sorted(str(f) for f in self._base._pddl_state.facts)
        return hashlib.sha1("\n".join(facts).encode()).hexdigest()[:16]


def plan_cost(plan):
    """规划器的代价口径：heat/cool/clean/use 为 5，其余为 1。"""
    return sum(5 if c.split(" ", 1)[0] in ("heat", "cool", "clean", "use", "slice") else 1 for c in plan)


def make_env(kind):
    if kind == "tw":
        return AlfEnv()
    if kind == "mock":
        from .mockenv import MockEnv
        return MockEnv()
    raise ValueError(kind)


def transcript_hash(st):
    """一步输出的指纹：观测、可执行命令列表（有序）、是否达成。"""
    h = hashlib.sha1()
    h.update(st["obs"].encode())
    h.update(b"\x1f")
    h.update("\x1e".join(st["admissible"]).encode())
    h.update(b"\x1f1" if st["won"] else b"\x1f0")
    return h.hexdigest()[:16]


def replay(env, gamefile, cmds, want_key=False, plan_at=()):
    """load + reset + 依次执行 cmds。返回逐步指纹（含初始状态）、状态键、计时与 plan_at 处的规划。"""
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
