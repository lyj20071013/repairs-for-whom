"""Gridworld：use API model（e.g Qwen3.8-flash）as a continuation executor。prompt、Historical protocols and actions to resolve and 7B exactly identical（model.AgentPolicy）。

    pol = APIAgentPolicy("qwen3.8-flash", temperature=0.7)
    stats = pol.run_rollouts(snaps, seeds)      # and policies.SeededAgentPolicy Same interface
    pol.usage                                   # Cumulative number of calls、token、Cache hit、Errors

- Close Thinking（enable_thinking=false）：The executor is a strategy to act directly at every step，and 7B Same thing；The cost of thinking and the delay is one measure higher。
- Sample：temperature 0.7 / top_p 0.95 / Up each step 160 token（and 7B Same conditions for deployment）；number j Seeds of Step derive_seed(rollout seed, j)。
  temperature 0 For greedy。API does not guarantee per- token Organisation，Seeds are only different rollout The random stream is independent of each other。
- Every single one rollout advance sequentially within one thread（Call every step），Multiple rollout Together（QWEN_EXEC_WORKERS，default 32）。
- Call Failed：429 / 5xx / Network error avoids retrying（At most 8 runs）；If you still fail, you should rollout Tag error，**Do not think of it as a failure**，
  Caller should be discarded error the branch of the、And then run again。400 / 401 / 403 / 404 indicates a parameter or API-key problem，Direct Abort。
Key with address judge_backends（QWEN_API_KEY / QWEN_BASE_URL）。
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import config
from api_client import ChatClient, FatalAPIError, cost_yuan  # noqa: F401（cost_yuan For ladder_gw Use）
from env import GridWorld
from model import AgentPolicy, _extract_action
from policies import derive_seed, trace_of


class APIAgentPolicy:
    stochastic = True

    def __init__(self, model, temperature=0.7, workers=None, max_tokens=None, timeout=120):
        self.client = ChatClient(model, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.escape_temperature = None          # and SeededAgentPolicy Interfaces are consistent（Do not support escape mechanisms）
        self.workers = workers or int(os.environ.get("QWEN_EXEC_WORKERS", "32"))
        self.max_tokens = max_tokens or config.MAX_NEW_TOKENS
        self.usage = self.client.usage
        self.samples = self.client.samples
        self._lock = self.client.lock
        self.cache_id = config.fingerprint({"api": self.model, "base": self.client.base, "top_p": config.TOP_P,
                                            "max_tokens": self.max_tokens, "thinking": False,
                                            "prompt": config.file_sha256(config.ROOT / "model.py")})

    # ---------------- Single Call ----------------
    def _call(self, msgs, temperature, seed):
        return self.client.call(msgs, temperature, config.TOP_P, self.max_tokens, seed=seed)[0]

    # ---------------- One rollout ----------------
    def _one(self, snap, seed):
        env = GridWorld.restore(snap)
        st = {"success": env.success, "steps": 0, "malformed": 0, "error": None}
        obs = env.observe()
        step = 0
        temp = self.temperature
        try:
            while not env.done:
                msgs = AgentPolicy.build_messages(obs, env.history)
                text = self._call(msgs, temp, derive_seed(seed, step))
                act = _extract_action(text)
                st["malformed"] += int(act.get("tool") == "malformed")
                obs, finished = env.step(act)
                step += 1
                st["steps"] += 1
                if finished:
                    st["success"] = env.success
        except FatalAPIError:
            raise
        except Exception as e:
            st["error"] = repr(e)[:300]
            with self._lock:
                self.usage["errors"] += 1
        st["trace"] = trace_of(env, snap["steps"])
        return st

    def preflight(self):
        """before the formal run, issue one request at each temperature 0.7（with a seed）And temperature 0 A very short request，Confirmed parameters accepted；invalid parameters or API keys fail here immediately。"""
        msgs = [{"role": "user", "content": 'Reply with exactly: {"tool": "look"}'}]
        out = []
        for temp in (self.temperature, 0.0):
            out.append(self._call(msgs, temp, 12345))
        return out

    def run_rollouts(self, snaps, seeds):
        """And theyre running a bunch of them rollout；each QWEN_EXEC_HEARTBEAT seconds（default 60）Print this batch progress and accumulated costs。"""
        done = [0]
        n = len(snaps)
        t0 = time.time()
        stop = threading.Event()

        def one(snap, seed):
            st = self._one(snap, seed)
            with self._lock:
                done[0] += 1
            return st

        def beat():
            while not stop.wait(float(os.environ.get("QWEN_EXEC_HEARTBEAT", "60"))):
                u = self.usage
                print(f"    [This instalment {done[0]}/{n} records rollout，{(time.time() - t0) / 60:.1f} minutes] Cumulative call {u['calls']}，"
                      f"Cost estimates ¥{cost_yuan(u):.1f}，Try again {u['retries']}，There was a mistake rollout {u['errors']}", flush=True)
        th = threading.Thread(target=beat, daemon=True)
        th.start()
        try:
            with ThreadPoolExecutor(self.workers) as ex:
                return list(ex.map(one, snaps, seeds))
        finally:
            stop.set()

