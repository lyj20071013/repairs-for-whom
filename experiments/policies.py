"""A follow-up strategy。Unified interface：

    policy.run_rollouts(snapshots, seeds) -> List[dict]
        Every single one rollout From the snapshot to the end，Back {"success": bool, "steps": int, "malformed": int}

seeds and snapshots one-to-one correspondence。Same (Condition, Batch, branch, number k runs) always receives the same seed，
Seeds vary from batch to batch，therefore“Repeat with condition”and“Independent Validation”uses a different random stream。
"""
import hashlib
import random

import config
import oracle
from env import GridWorld

# Noise oracle Random actions to assemble（Does not include give_up，Consistency with candidate action）。
_RANDOM_ACTIONS = config.CANDIDATE_ACTIONS


def derive_seed(*parts):
    """Map any field with certainty to [0, 2^31) Integer Feeds。"""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def physical_state(env):
    """To determine the physical state of the transfer（No Step & History），detect whether a continuation revisits a prior state。"""
    return (tuple(env.pos), env.has_key, env.has_gem, env.door_open, tuple("".join(r) for r in env.layout))


def trace_of(env, start_steps):
    """The step-by-step record of the renewal：[Tools, parameters, Environmental events, Position after action]。For testing“Repeat Error”Mechanisms etc。"""
    out = []
    for h in env.history:
        if h.get("step", -1) >= start_steps:
            a = h.get("action", {})
            out.append([a.get("tool"), a.get("arg"), h.get("event"), h.get("pos_after")])
    return out


class OraclePolicy:
    """environment-optimal continuation：Determination，success if and only if F(h)=1。Real execution to check oracle table。"""
    stochastic = False

    def run_rollouts(self, snaps, seeds=None):
        out = []
        for snap in snaps:
            env = GridWorld.restore(snap)
            tab = oracle.table_for(snap)
            while not env.done:
                acts = tab.optimal_actions(env.snapshot())
                env.step(acts[0] if acts else {"tool": "look"})
            assert env.success == tab.feasible(snap), "oracle Tables are inconsistent with actual implementation"
            out.append({"success": env.success, "steps": env.steps - snap["steps"], "malformed": 0,
                        "trace": trace_of(env, snap["steps"])})
        return out


class NoisyOraclePolicy:
    """By eps Random Actions with Equal Probability，Otherwise, one of the best actions is randomly chosen（Ignore the remaining budget）。

    CPU You can run on it；serves both as“By comparison 7B More or less、Modifiable”Controlled assessors，
    and as a substitute during offline tests for 7B Simulation strategy。
    """
    stochastic = True

    def __init__(self, eps):
        if not 0.0 <= eps <= 1.0:
            raise ValueError("eps You have to [0, 1]")
        self.eps = eps

    def run_rollouts(self, snaps, seeds):
        out = []
        for snap, seed in zip(snaps, seeds):
            rng = random.Random(seed)
            env = GridWorld.restore(snap)
            tab = oracle.table_for(snap)
            while not env.done:
                acts = tab.optimal_actions(env.snapshot())
                if not acts or rng.random() < self.eps:
                    env.step(dict(rng.choice(_RANDOM_ACTIONS)))
                else:
                    env.step(dict(rng.choice(acts)))
            out.append({"success": env.success, "steps": env.steps - snap["steps"], "malformed": 0,
                        "trace": trace_of(env, snap["steps"])})
        return out


def make_seeded_llm_class():
    """Delay Import model（will import vllm），none GPU construct this test only when needed。"""
    from model import AgentPolicy, _extract_action

    class SeededAgentPolicy(AgentPolicy):
        """at pilot of AgentPolicy Add request-by-requestseeds。prompt、Reconstructing the historical agreement completely with the original realization。

        Every single one rollout For the first time in my life, j Step derive_seed(rollout seed, j)。Temperature 0 uses greedy decoding when，does not use seeds。
        vLLM There may be very small differences in values under the composition of different batches，# So go # token exact reproduction is not guaranteed；
        But the random streams of different batches are independent of each other，Thats guaranteed by seeds。
        """
        stochastic = True

        # escape temperature：Not None when，If the current state of physics had already appeared in this follow-up,，Well use the temperature to sample this step（Experimental use of mechanisms）
        escape_temperature = None

        def __init__(self, temperature, **kwargs):
            super().__init__(temperature=temperature, **kwargs)

        def _step_params(self, seed, step, escape=False):
            temp = self.escape_temperature if escape else self.temperature
            kwargs = dict(temperature=temp, top_p=config.TOP_P, max_tokens=config.MAX_NEW_TOKENS)
            if temp > 0:
                kwargs["seed"] = derive_seed(seed, step, "escape") if escape else derive_seed(seed, step)
            return self._make_params(**kwargs)

        def run_rollouts(self, snaps, seeds):
            envs = [GridWorld.restore(s) for s in snaps]
            obs = [e.observe() for e in envs]
            stats = [{"success": e.success, "steps": 0, "malformed": 0, "escapes": 0} for e in envs]
            alive = [i for i, e in enumerate(envs) if not e.done]
            step = {i: 0 for i in alive}
            visited = {i: set() for i in alive}
            while alive:
                prompts = [self._render(obs[i], envs[i].history) for i in alive]
                params = []
                for i in alive:
                    escape = False
                    if self.escape_temperature is not None:
                        key = physical_state(envs[i])
                        escape = key in visited[i]
                        visited[i].add(key)
                        stats[i]["escapes"] += int(escape)
                    params.append(self._step_params(seeds[i], step[i], escape))
                outs = self.llm.generate(prompts, params, use_tqdm=False)
                nxt = []
                for i, out in zip(alive, outs):
                    act = _extract_action(out.outputs[0].text)
                    stats[i]["malformed"] += int(act.get("tool") == "malformed")
                    obs[i], finished = envs[i].step(act)
                    step[i] += 1
                    stats[i]["steps"] += 1
                    if finished:
                        stats[i]["success"] = envs[i].success
                    else:
                        nxt.append(i)
                alive = nxt
            for st, env, snap in zip(stats, envs, snaps):
                st["trace"] = trace_of(env, snap["steps"])
            return stats

    return SeededAgentPolicy


class PolicyPool:
    """Construct policies according to conditions；7B Load only once，different temperatures share the same vLLM Example。

    mock=True when，llm Conditional noise oracle Replace（eps as temperature increases），For offline testing process only，
    It turns out it doesnt have any scientific meaning。
    """

    def __init__(self, mock=False):
        self.mock = mock
        self._llm = None

    def get(self, cond_name):
        spec = config.CONDITIONS[cond_name]
        kind = spec["kind"]
        if kind == "oracle":
            return OraclePolicy()
        if kind == "noisy_oracle":
            return NoisyOraclePolicy(spec["eps"])
        if kind != "llm":
            raise ValueError(f"Unknown Condition Type {kind}")
        if self.mock:
            return NoisyOraclePolicy(min(1.0, 0.25 + 0.5 * spec["temperature"]))
        if self._llm is None:
            cls = make_seeded_llm_class()
            self._llm = cls(temperature=spec["temperature"])
        self._llm.temperature = spec["temperature"]
        self._llm.escape_temperature = spec.get("escape_temperature")
        return self._llm
