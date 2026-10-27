"""API shared by continuation executors /chat/completions Client（gridworld of api_policy.APIAgentPolicy and ALFWorld of alfw/apibackend.APIBackend）。

- Service provider with key judge_backends.PROVIDERS（Qwen：QWEN_API_KEY / QWEN_BASE_URL）。
- The executor doesnt think：Qwen3 Series enable_thinking=false；Qwen2.5 Do not send this parameter。temperature > 0 Time with Feeds。
- 429 / 5xx / Network error avoids retrying（At most 8 runs，QWEN_RETRY_BASE Controls the retreat base）；400 / 401 / 403 / 404 indicates a parameter or API-key problem，
  Throw FatalAPIError，the caller should stop immediately。Retry Exhaustion RuntimeError（Caller decides whether to abandon the branch or stop，It cant be used as a strategy failure）。
- usage Cumulative number of calls、token、Cache hit、Try again、Length Cut；cost_yuan Estimated costs at unit cost（Based on platform billing）。
"""
import os
import threading
import time

from judge_backends import PROVIDERS, provider_of


class FatalAPIError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, model, timeout=120):
        import requests
        self._rq = requests
        self.model = model
        self.provider = provider_of(model)
        pv = PROVIDERS[self.provider]
        self.key = next((os.environ[e] for e in pv["key_env"] if os.environ.get(e)), None)
        if not self.key:
            raise RuntimeError(f"No environmental variables found {' or '.join(pv['key_env'])}")
        self.base = (os.environ.get(pv["base_env"]) or pv["base"]).rstrip("/")
        self.timeout = timeout
        self.usage = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
                      "retries": 0, "errors": 0, "length_cut": 0, "seconds_in_calls": 0.0}
        self.lock = threading.Lock()
        self.samples = []                       # Previous original output，Check it out when you smoke

    def settings(self):
        return {"api": self.model, "base": self.base, "thinking": False}

    def call(self, msgs, temperature, top_p, max_tokens, seed=None, stop=None):
        """Back (Text, finish_reason, usage Dictionary)。"""
        body = {"model": self.model, "messages": msgs, "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens, "stream": False}
        if stop:
            body["stop"] = list(stop)
        if self.provider == "qwen":
            if "qwen3" in self.model.lower():      # only Qwen3 the series supports a thinking switch；Qwen2.5 Do not send this parameter
                body["enable_thinking"] = False
            if temperature > 0 and seed is not None:
                body["seed"] = int(seed)
        else:
            body["thinking"] = {"type": "disabled"}
        last = None
        for attempt in range(8):
            t0 = time.time()
            try:
                r = self._rq.post(f"{self.base}/chat/completions", json=body, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                dt = time.time() - t0
                if r.status_code == 200:
                    j = r.json()
                    ch = j["choices"][0]
                    u = j.get("usage") or {}
                    text = ch["message"].get("content") or ""
                    with self.lock:
                        self.usage["calls"] += 1
                        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                        self.usage["cached_tokens"] += ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                        self.usage["completion_tokens"] += u.get("completion_tokens", 0) or 0
                        self.usage["length_cut"] += int(ch.get("finish_reason") == "length")
                        self.usage["seconds_in_calls"] += dt
                        if len(self.samples) < 5:
                            self.samples.append(text)
                    return text, ch.get("finish_reason"), u
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 403, 404):
                    raise FatalAPIError(last)
            except FatalAPIError:
                raise
            except Exception as e:             # Network Error
                last = repr(e)
            with self.lock:
                self.usage["retries"] += 1
            time.sleep(min(60, float(os.environ.get("QWEN_RETRY_BASE", "1")) * 2 ** attempt))
        raise RuntimeError(last)


def cost_yuan(usage, price_in=None, price_cached=None, price_out=None):
    """Press every million token Estimated unit cost（CNY）。Default unit price available environment variable overwrite：
    QWEN_PRICE_IN（default 0.8）、QWEN_PRICE_CACHED（Hit Cache Inputs，Default at input price 20%）、QWEN_PRICE_OUT（default 2.7）。
    In a thousand years AI Platform billed，used only for cost estimation here。"""
    pi = float(price_in if price_in is not None else os.environ.get("QWEN_PRICE_IN", "0.8"))
    pc = float(price_cached if price_cached is not None else os.environ.get("QWEN_PRICE_CACHED", str(pi * 0.2)))
    po = float(price_out if price_out is not None else os.environ.get("QWEN_PRICE_OUT", "2.7"))
    cached = usage["cached_tokens"]
    return ((usage["prompt_tokens"] - cached) * pi + cached * pc + usage["completion_tokens"] * po) / 1e6
