"""“只判断、不执行”的修复选择所用的 LLM 后端（gridworld 与 ALFWorld 共用）。

    VLLMJudge   本地 vLLM（默认 Qwen2.5-7B-Instruct，即执行者自己当裁判）
    APIJudge    OpenAI 兼容接口。按模型名选服务商（密钥只从环境变量读取）：
                  deepseek-*  DeepSeek 官方 API，DEEPSEEK_API_KEY（可选 DEEPSEEK_BASE_URL）
                  qwen*       千问AI平台，QWEN_API_KEY 或 DASHSCOPE_API_KEY；
                              QWEN_BASE_URL 默认 https://maas.qianwenaiapi.com/compatible-mode/v1
                              （阿里云百炼的密钥用 https://dashscope.aliyuncs.com/compatible-mode/v1）
    MockJudge   离线测试

统一接口：judge.ask(list_of_messages_lists) -> list of {"text", "finish_reason", "error"}。

连通性检查（一次很短的调用，打印回答、思考长度与用量）：
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
    """取最后一个 “Choice: C<k>”；k 超出 1..n 视为无效。返回 0 起的下标或 None。"""
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


# 服务商预设。千问的参数依据千问AI平台文档（platform.qianwenai.com/docs/api-reference/chat/openai-chat，2026-10-23 查）：
# enable_thinking 开关思考；reasoning_effort 对 Qwen3.8 取 low / medium / xhigh（默认 xhigh），不能与 thinking_budget 同设；
# 对应思维链上限 low 4096、medium 16384、xhigh 262144 token。默认 xhigh 时一次判断可能想十几分钟，这里默认用 medium
# （与 DeepSeek 思考版在 gridworld 上实际用到的思维链长度同一量级：首次 8k 上限截断 7/34，32k 上限全部完成）；
# 思考模式下 temperature < 0.6 会被自动调到 0.6（文本默认 1.0），因此用 seed 固定随机性；
# 除 qwen3.8-max 外，max_tokens 只限制回答、不含思维链；思维链在 reasoning_content。
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
    """OpenAI 兼容的 /chat/completions。服务商由模型名决定（见 PROVIDERS）。

    DeepSeek 的请求与之前完全相同（非流式；thinking={"type": ...}；reasoning_effort）。
    千问：enable_thinking；思考模式默认流式请求（思维链可能很长，避免网关在长时间无响应时断开），seed 固定。
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
            raise RuntimeError(f"没有找到环境变量 {' 或 '.join(envs)}（API 密钥只从环境变量读取）")
        self.base = (base_url or os.environ.get(pv["base_env"]) or pv["base"]).rstrip("/")
        self.model = model
        self.name = name or model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.workers = workers or int(os.environ.get("JUDGE_WORKERS", DEFAULT_WORKERS[self.provider]))
        self.thinking = thinking
        self.live = {}                       # 流式请求进行中：下标 -> 已收到的思维链 / 回答字符数（进度显示用）
        if self.provider == "deepseek":
            # DeepSeek V4 系列默认开启思考（2026-10 官方文档）；思考模式下 temperature 不生效
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
                                 note="思考模式下 temperature 实际不低于 0.6；随机性由 seed 固定")

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
                            raise RuntimeError("流式响应没有内容就结束了")
                        return out
                    j = r.json()
                    ch = j["choices"][0]
                    return {"text": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
                            "error": None, "model_reported": j.get("model"), "usage": j.get("usage"),
                            "reasoning_chars": len(ch["message"].get("reasoning_content") or "")}
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 402, 403, 404):
                    break                       # 参数、密钥或余额问题，重试无用
            except Exception as e:              # 网络错误或流中断：退避重试
                last = repr(e)
            time.sleep(min(60, 2 ** attempt))
        return {"text": "", "finish_reason": None, "error": last}

    def ask(self, msgs_list, on_done=None):
        """并发请求；每完成一条调用 on_done(下标, 结果)，并每分钟打印一次进行中的请求收到了多少思维链。"""
        n, outs, t0 = len(msgs_list), [None] * len(msgs_list), time.time()
        stop = threading.Event()

        def beat():
            while not stop.wait(float(os.environ.get("JUDGE_HEARTBEAT", "60"))):
                live = dict(self.live)
                done = sum(o is not None for o in outs)
                if live:
                    r = sorted(v[0] for v in live.values())
                    print(f"  [{(time.time() - t0) / 60:.1f} 分钟] 已完成 {done}/{n}；进行中 {len(live)} 条，"
                          f"已收到思维链 {r[0]}–{r[-1]} 字符（中位 {r[len(r) // 2]}）", flush=True)
                else:
                    print(f"  [{(time.time() - t0) / 60:.1f} 分钟] 已完成 {done}/{n}，等待回答中", flush=True)
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
        m = model or "qwen3.8-flash"       # 2026-10-23 起全部裁判用千问；DeepSeek 仍可用 --model deepseek-flash
        prov = provider_of(m)
        if prov == "qwen":
            # 千问（Qwen3.8 非 max）的 max_tokens 只限制回答，不含思维链；不思考时推理写在回答里，同样给 4096
            mt = 4096
        else:
            mt = 8192 if thinking else 1024     # DeepSeek：含思维链
        eff = effort or DEFAULT_EFFORT[prov]
        suffix = ("_think" if thinking else "") + (f"_{eff}" if thinking and eff != DEFAULT_EFFORT[prov] else "")
        return APIJudge(m, temperature=temperature, thinking=thinking, max_tokens=mt, effort=eff, name=m + suffix)
    raise ValueError(kind)


def _msgs_hash(msgs, settings):
    keep = {k: v for k, v in settings.items() if k not in ("note",)}
    return hashlib.sha256(json.dumps([msgs, keep], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def ask_resumable(judge, keys, msgs_list, cache_path):
    """逐条落盘的提问：每完成一条立即写入 cache_path（jsonl），中断后重跑只问还没有结果的（提示或设置变了的会重问）。
    调用出错（error 非空）的不缓存。返回与 msgs_list 对齐的结果列表。"""
    cache_path = Path(cache_path)
    have = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                        # 中断时写了一半的行
            have[(r["key"], r["hash"])] = r["out"]
    hs = [_msgs_hash(m, judge.settings) for m in msgs_list]
    outs = [have.get((k, h)) for k, h in zip(keys, hs)]
    todo = [i for i, o in enumerate(outs) if o is None]
    n = len(msgs_list)
    print(f"[judge] {judge.name}：共 {n} 条，已有结果 {n - len(todo)} 条，本次提问 {len(todo)} 条"
          f"（并发 {getattr(judge, 'workers', 1)}；每条完成即保存到 {cache_path.name}）", flush=True)
    if not todo:
        return outs
    lock, t0, cnt = threading.Lock(), time.time(), [0]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():                     # 截掉中断时写了一半的最后一行，保证追加的记录从新行开始
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
                print(f"  {cnt[0]}/{len(todo)} 完成（{(time.time() - t0) / 60:.1f} 分钟）{keys[i]}："
                      + (f"出错 {o['error'][:120]}" if o.get("error") else
                         f"思维链 {o.get('reasoning_chars', 0)} 字符，回答 {len(o.get('text') or '')} 字符，"
                         f"输出 token {u.get('completion_tokens', '?')}，结束原因 {o.get('finish_reason')}"), flush=True)
        judge.ask([msgs_list[i] for i in todo], on_done=done)
    return outs


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _ping():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", required=True, help="模型名，如 qwen3.8-flash 或 deepseek-flash")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--effort", default=None)
    args = ap.parse_args()
    j = make_judge("api", model=args.ping, thinking=not args.no_thinking, effort=args.effort)
    print("设置：", json.dumps(j.settings, ensure_ascii=False))
    t0 = time.time()
    out = j.ask([[{"role": "user", "content": "Which is larger, 17*23 or 19*21? Answer briefly, then end with "
                   "a line 'Choice: C1' if the first is larger, otherwise 'Choice: C2'."}]])[0]
    print(f"用时 {time.time() - t0:.1f}s")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    k = parse_choice(out["text"], 2)
    ok = out["error"] is None and k is not None
    print("连通性：" + ("正常" if ok else "有问题") + (f"（解析到 C{k + 1}；正确答案 C2）" if k is not None else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_ping())
