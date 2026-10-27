"""“Only judgment、Not implemented”for the repair of the LLM Backend（gridworld and ALFWorld Shared）。

    VLLMJudge   Local vLLM（default Qwen2.5-7B-Instruct，The executors own judgment）
    APIJudge    OpenAI Compatibility Interface。Select the service provider by model（Keys read from environment variables only）：
                  deepseek-*  DeepSeek Official API，DEEPSEEK_API_KEY（Optional DEEPSEEK_BASE_URL）
                  qwen*       Qwen AI Platform，QWEN_API_KEY or DASHSCOPE_API_KEY；
                              QWEN_BASE_URL default https://maas.qianwenaiapi.com/compatible-mode/v1
                              （The key to Ali Yunburi https://dashscope.aliyuncs.com/compatible-mode/v1）
    MockJudge   Offline test

Unified interface：judge.ask(list_of_messages_lists) -> list of {"text", "finish_reason", "error"}。

Connectivity check（a very short request，Print Answer、Length and Usage of Thinking）：
    python judge_backends.py --ping qwen3.8-flash
    python judge_backends.py --ping qwen3.8-flash --no-thinking
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CHOICE_RE = re.compile(r"choice\s*[:：]\s*\**\s*C\s*(\d+)", re.I)


def parse_choice(text, n):
    """Take the last one “Choice: C<k>”；k Over 1..n treat as invalid。Back 0 Subscript or None。"""
    ms = CHOICE_RE.findall(text or "")
    if not ms:
        return None
    k = int(ms[-1])
    return k - 1 if 1 <= k <= n else None


class VLLMJudge:
    def __init__(self, model_dir, context=8192, max_tokens=1024, temperature=0.0, name="qwen7b"):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams
        self.name = name
        self.model = str(model_dir)
        self.params = SamplingParams(temperature=temperature, top_p=1.0 if temperature == 0 else 0.95,
                                     max_tokens=max_tokens, seed=0 if temperature > 0 else None)
        self.llm = LLM(model=self.model, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=context,
                       disable_log_stats=True, seed=0)
        self.tok = self.llm.get_tokenizer()
        self.settings = {"backend": "vllm", "model": self.model, "temperature": temperature, "max_tokens": max_tokens}

    def ask(self, msgs_list, on_done=None):
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list]
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        outs = [{"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason, "error": None} for o in outs]
        for i, o in enumerate(outs):
            if on_done:
                on_done(i, o)
        return outs


# Service provider preset。The argument is based on the question AI Platform Document（platform.qianwenai.com/docs/api-reference/chat/openai-chat，2026-10-23 Chai）：
# enable_thinking Switches to think；reasoning_effort Yeah Qwen3.8 Take low / medium / xhigh（default xhigh），Cant be with thinking_budget Same as；
# The upper limit of the corresponding thinking chain low 4096、medium 16384、xhigh 262144 token。default xhigh One judgment at a time, maybe ten minutes，Default here medium
# （and DeepSeek the thinking-enabled version on gridworld The length of the chain of thought actually used is the same as the length of the chain：First 8k Ceiling Interruption 7/34，32k All caps are complete）；
# Thinking mode temperature < 0.6 automatically adjusted to 0.6（Text Default 1.0），So use it seed Fixed Randomity；
# Divide qwen3.8-max Out，max_tokens Restricted answers only、It doesnt contain the chain of thought；reasoning content is in reasoning_content。
PROVIDERS = {
    "deepseek": {"base_env": "DEEPSEEK_BASE_URL", "base": "https://api.deepseek.com", "key_env": ("DEEPSEEK_API_KEY",)},
    "qwen": {"base_env": "QWEN_BASE_URL", "base": "https://maas.qianwenaiapi.com/compatible-mode/v1",
             "key_env": ("QWEN_API_KEY", "DASHSCOPE_API_KEY")},
}
DEFAULT_EFFORT = {"deepseek": "high", "qwen": "medium"}
DEFAULT_WORKERS = {"deepseek": 8, "qwen": 16}


def provider_of(model):
    return "qwen" if model.lower().startswith("qwen") else "deepseek"


class APIJudge:
    """OpenAI Compatible /chat/completions。The service provider is determined by the name of the model（See PROVIDERS）。

    DeepSeek The request was exactly the same as before（Non-fluorescent；thinking={"type": ...}；reasoning_effort）。
    Qwen：enable_thinking；Think mode default flow request（The chain of thought can be long，Avoid breaking the gateway for long periods of time without responding），seed Fixed。
    """

    def __init__(self, model, base_url=None, key_env=None, temperature=0.0, max_tokens=1024,
                 workers=None, name=None, thinking=None, effort=None, seed=None, stream=None, provider=None):
        import requests
        self._rq = requests
        self.provider = provider or provider_of(model)
        pv = PROVIDERS[self.provider]
        envs = (key_env,) if key_env else pv["key_env"]
        self.key = next((os.environ[e] for e in envs if os.environ.get(e)), None)
        if not self.key:
            raise RuntimeError(f"No environmental variables found {' or '.join(envs)}（API Keys read from environment variables only）")
        self.base = (base_url or os.environ.get(pv["base_env"]) or pv["base"]).rstrip("/")
        self.model = model
        self.name = name or model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.workers = workers or int(os.environ.get("JUDGE_WORKERS", DEFAULT_WORKERS[self.provider]))
        self.thinking = thinking
        self.live = {}                       # streaming request in progress：Subscript -> reasoning content received so far / Number of characters answered（Progress Display Usage）
        if self.provider == "deepseek":
            # DeepSeek V4 The series defaults to start thinking（2026-10 Official Documents）；Thinking mode temperature Not effective
            self.effort = effort or DEFAULT_EFFORT["deepseek"]
            self.seed, self.stream = None, False
        else:
            self.effort = effort or DEFAULT_EFFORT["qwen"]
            self.seed = 0 if seed is None else seed
            self.stream = bool(thinking) if stream is None else stream
        self.settings = {"backend": "api", "provider": self.provider, "base_url": self.base, "model": model,
                         "temperature": temperature, "max_tokens": max_tokens, "thinking": thinking,
                         "reasoning_effort": self.effort if thinking else None}
        if self.provider == "qwen":
            self.settings.update(seed=self.seed, stream=self.stream,
                                 note="Thinking mode temperature in practice no lower than 0.6；Random by seed Fixed")

    def _body(self, msgs):
        body = {"model": self.model, "messages": msgs, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "stream": self.stream}
        if self.provider == "deepseek":
            if self.thinking is not None:
                body["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
                if self.thinking:
                    body["reasoning_effort"] = self.effort
        else:
            if self.thinking is not None:
                body["enable_thinking"] = bool(self.thinking)
            if self.thinking and self.effort:
                body["reasoning_effort"] = self.effort
            if self.seed is not None:
                body["seed"] = self.seed
            if self.stream:
                body["stream_options"] = {"include_usage": True}
        return body

    def _read_stream(self, r, i=None):
        text, reasoning, finish, usage, model = [], [], None, None, None
        nr = nt = 0
        for raw in r.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            model = j.get("model") or model
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    text.append(d["content"])
                    nt += len(d["content"])
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                    nr += len(d["reasoning_content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if i is not None:
                self.live[i] = (nr, nt)
        return {"text": "".join(text), "finish_reason": finish, "error": None, "model_reported": model,
                "usage": usage, "reasoning_chars": len("".join(reasoning))}

    def _one(self, msgs, i=None):
        body = self._body(msgs)
        last = None
        for attempt in range(6):
            try:
                r = self._rq.post(f"{self.base}/chat/completions", json=body, stream=self.stream,
                                  timeout=(30, 600) if self.stream else 300,
                                  headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                if r.status_code == 200:
                    if self.stream:
                        out = self._read_stream(r, i)
                        if out["finish_reason"] is None and not out["text"]:
                            raise RuntimeError("The flow response is over without content")
                        return out
                    j = r.json()
                    ch = j["choices"][0]
                    return {"text": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
                            "error": None, "model_reported": j.get("model"), "usage": j.get("usage"),
                            "reasoning_chars": len(ch["message"].get("reasoning_content") or "")}
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 402, 403, 404):
                    break                       # parameters、Key or balance problem，retrying will not help
            except Exception as e:              # Network error or stream interruption：Stay away from the retry
                last = repr(e)
            time.sleep(min(60, 2 ** attempt))
        return {"text": "", "finish_reason": None, "error": last}

    def ask(self, msgs_list, on_done=None):
        """# And send a request #；Every call completed on_done(Subscript, results)，print received reasoning length for pending requests once per minute。"""
        n, outs, t0 = len(msgs_list), [None] * len(msgs_list), time.time()
        stop = threading.Event()

        def beat():
            while not stop.wait(float(os.environ.get("JUDGE_HEARTBEAT", "60"))):
                live = dict(self.live)
                done = sum(o is not None for o in outs)
                if live:
                    r = sorted(v[0] for v in live.values())
                    print(f"  [{(time.time() - t0) / 60:.1f} minutes] Completed {done}/{n}；Ongoing {len(live)} records，"
                          f"reasoning content received {r[0]}–{r[-1]} Character（Centre {r[len(r) // 2]}）", flush=True)
                else:
                    print(f"  [{(time.time() - t0) / 60:.1f} minutes] Completed {done}/{n}，Waiting for the answer", flush=True)
        th = threading.Thread(target=beat, daemon=True)
        th.start()
        try:
            with ThreadPoolExecutor(self.workers) as ex:
                futs = {ex.submit(self._one, m, i): i for i, m in enumerate(msgs_list)}
                for f in as_completed(futs):
                    i = futs[f]
                    outs[i] = f.result()
                    self.live.pop(i, None)
                    if on_done:
                        on_done(i, outs[i])
        finally:
            stop.set()
        return outs


class MockJudge:
    def __init__(self, name="mock"):
        self.name = name
        self.settings = {"backend": "mock"}

    def ask(self, msgs_list, on_done=None):
        out = []
        for m in msgs_list:
            text = m[-1]["content"]
            n = len(re.findall(r"^C\d+:", text, re.M))
            rng = random.Random(hashlib.sha1(text.encode()).hexdigest())
            out.append({"text": f"Reasoning (mock).\nChoice: C{rng.randint(1, max(1, n))}", "finish_reason": "stop",
                        "error": None})
            if on_done:
                on_done(len(out) - 1, out[-1])
        return out


def make_judge(kind, model=None, model_dir=None, temperature=0.0, thinking=True, effort=None):
    if kind == "mock":
        return MockJudge()
    if kind == "vllm":
        return VLLMJudge(model_dir, temperature=temperature, name="qwen7b_self")
    if kind == "api":
        m = model or "qwen3.8-flash"       # 2026-10-23 All the judges will have to ask questions；DeepSeek Available --model deepseek-flash
        prov = provider_of(m)
        if prov == "qwen":
            # Qwen（Qwen3.8 Not max）of max_tokens Restricted answers only，It doesnt contain the chain of thought；The reasoning is written in the answer when you dont think，Same thing 4096
            mt = 4096
        else:
            mt = 8192 if thinking else 1024     # DeepSeek：A chain of thought
        eff = effort or DEFAULT_EFFORT[prov]
        suffix = ("_think" if thinking else "") + (f"_{eff}" if thinking and eff != DEFAULT_EFFORT[prov] else "")
        return APIJudge(m, temperature=temperature, thinking=thinking, max_tokens=mt, effort=eff, name=m + suffix)
    raise ValueError(kind)


def _msgs_hash(msgs, settings):
    keep = {k: v for k, v in settings.items() if k not in ("note",)}
    return hashlib.sha256(json.dumps([msgs, keep], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def ask_resumable(judge, keys, msgs_list, cache_path):
    """Questions, article by article：write each completed record immediately cache_path（jsonl），after interruption, request only missing results（A prompts or a changed set will be asked again）。
    Error calling（error Non-empty）Uncached。returns with msgs_list Aligning Results List。"""
    cache_path = Path(cache_path)
    have = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                        # Half the lines at the break
            have[(r["key"], r["hash"])] = r["out"]
    hs = [_msgs_hash(m, judge.settings) for m in msgs_list]
    outs = [have.get((k, h)) for k, h in zip(keys, hs)]
    todo = [i for i, o in enumerate(outs) if o is None]
    n = len(msgs_list)
    print(f"[judge] {judge.name}：Total {n} records，Results already available {n - len(todo)} records，This question {len(todo)} records"
          f"（Together {getattr(judge, 'workers', 1)}；save each completed record to {cache_path.name}）", flush=True)
    if not todo:
        return outs
    lock, t0, cnt = threading.Lock(), time.time(), [0]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():                     # Half the last line that was written when the break was cut off，Make sure the extra records start with the new line
        raw = cache_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            cache_path.write_bytes(raw[:raw.rfind(b"\n") + 1])
    with cache_path.open("a", encoding="utf-8") as f:
        def done(j, o):
            i = todo[j]
            outs[i] = o
            with lock:
                cnt[0] += 1
                if not o.get("error"):
                    f.write(json.dumps({"key": keys[i], "hash": hs[i], "out": o}, ensure_ascii=False) + "\n")
                    f.flush()
                u = o.get("usage") or {}
                print(f"  {cnt[0]}/{len(todo)} complete（{(time.time() - t0) / 60:.1f} minutes）{keys[i]}："
                      + (f"Error {o['error'][:120]}" if o.get("error") else
                         f"The thought chain {o.get('reasoning_chars', 0)} Character，Answer {len(o.get('text') or '')} Character，"
                         f"output token {u.get('completion_tokens', '?')}，Quantity {o.get('finish_reason')}"), flush=True)
        judge.ask([msgs_list[i] for i in todo], on_done=done)
    return outs


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _ping():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", required=True, help="Model Name，e.g qwen3.8-flash or deepseek-flash")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--effort", default=None)
    args = ap.parse_args()
    j = make_judge("api", model=args.ping, thinking=not args.no_thinking, effort=args.effort)
    print("Settings：", json.dumps(j.settings, ensure_ascii=False))
    t0 = time.time()
    out = j.ask([[{"role": "user", "content": "Which is larger, 17*23 or 19*21? Answer briefly, then end with "
                   "a line 'Choice: C1' if the first is larger, otherwise 'Choice: C2'."}]])[0]
    print(f"Time {time.time() - t0:.1f}s")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    k = parse_choice(out["text"], 2)
    ok = out["error"] is None and k is not None
    print("Connectivity：" + ("Normal" if ok else "issues found") + (f"（Parsing to C{k + 1}；Correct answer C2）" if k is not None else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_ping())
