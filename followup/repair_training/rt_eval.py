"""评测（论文方案.md 5.5 格式检查、5.6 读数）。一个进程只建一个 vLLM 引擎（enable_lora），各模型依次换 adapter。

    python rt.py --stage formatcheck     每个模型在格式检查局上各跑 1 次，只看不可执行 / 无法解析动作占步骤的比例
    python rt.py --stage evaluate        端到端（E 的每局 1 次）+ 接管（P 个点 × m 次）；要求格式检查已通过

共同随机数：同一局 / 同一接管点上，各模型用相同的请求种子序列（种子只由局或点决定，不含模型名）。
基础模型 B 也在同一个开了 LoRA 的引擎里评测（不带 adapter），与 LoRA 模型的数值路径一致。
"""
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from alfw import acfg, agent
import rt_common as C
from rt_common import (CFG, P, add_timing, adapter_dir, all_models, derive_seed, file_sha256, fp, jdump, jl_read,
                       jload, log, make_backend, make_pool, release_vllm, run_specs)


# ---------------------------------------------------------------- 后端
class LoraVLLMBackend(agent.VLLMBackend):
    """与 agent.VLLMBackend 相同的生成参数；额外开 LoRA，generate 时带当前 adapter。"""

    def __init__(self, model_dir):
        import os
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        self._SP = SamplingParams
        self._LR = LoRARequest
        self.context = acfg.MODEL_CONTEXT
        self.llm = LLM(model=str(model_dir), dtype="bfloat16", gpu_memory_utilization=0.90,
                       max_model_len=self.context, enable_prefix_caching=True, disable_log_stats=True,
                       seed=acfg.MODEL_SEED, enable_lora=True, max_lora_rank=CFG["LORA_R"], max_loras=1)
        self.tok = self.llm.get_tokenizer()
        self.lora = None
        self._ids = {}

    def set_lora(self, name, path):
        if path is None:
            self.lora = None
            return
        if name not in self._ids:
            self._ids[name] = len(self._ids) + 1
        self.lora = self._LR(name, self._ids[name], str(path))

    def generate(self, prompts, temperature, seeds, max_new_tokens, hints=None):
        params = []
        for s in seeds:
            kw = dict(temperature=temperature, top_p=acfg.TOP_P, max_tokens=max_new_tokens, stop=acfg.STOP)
            if temperature > 0:
                kw["seed"] = s
            params.append(self._SP(**kw))
        outs = self.llm.generate(prompts, params, use_tqdm=False, lora_request=self.lora)
        return [SimpleNamespace(text=o.outputs[0].text, finish_reason=getattr(o.outputs[0], "finish_reason", None),
                                n_prompt=len(o.prompt_token_ids or []), n_gen=len(o.outputs[0].token_ids or []))
                for o in outs]


class MockLoraBackend(agent.MockBackend):
    """模拟：adapter 的"能力" skill = 以该概率走模拟环境里的聪明策略，否则随机可执行动作（基础模型 0.55）。"""

    def __init__(self):
        super().__init__()
        self.skill = 0.55

    def set_lora(self, name, path):
        self.skill = 0.55 if path is None else jload(Path(path) / "mock_adapter.json")["skill"]

    def generate(self, prompts, temperature, seeds, max_new_tokens, hints=None):
        res = []
        for i, (p, s) in enumerate(zip(prompts, seeds)):
            adm = hints[i] if hints else []
            rng = random.Random(s if temperature > 0 else agent.derive_seed(p))
            a = agent._mock_choose(p, adm, rng) if rng.random() < self.skill else (rng.choice(adm) if adm else "look")
            if rng.random() < 0.03:
                a = "fly to the moon"
            text = f"Thought: mock.\nAction: {a}"
            res.append(SimpleNamespace(text=text, finish_reason="stop", n_prompt=len(p) // 4, n_gen=len(text) // 4))
        return res


def adapter_path(model):
    if model == "B":
        return None
    g, s = model.rsplit("_s", 1)
    d = adapter_dir(g, int(s))
    if not (d / "done.json").exists():
        sys.exit(f"[eval] {model} 还没训练完（缺 {d / 'done.json'}）")
    return d


def adapter_sig(model):
    d = adapter_path(model)
    if d is None:
        return "base"
    return fp({f.name: file_sha256(f) for f in sorted(d.iterdir()) if f.is_file() and f.name != "train_log.json"})


def _eval_models(stage, models, spec_fn, path, stop_after=None):
    holder = {}

    def backend():
        if "b" not in holder:
            holder["b"] = make_backend(lora=True)
        return holder["b"]
    t0 = time.time()
    n_tot = 0
    try:
        with make_pool() as pl:
            for m in models:
                specs = spec_fn(m)
                sig = fp({"stage": stage, "model": m, "adapter": adapter_sig(m), "code": _code(),
                          "n": len(specs)})
                have = {r["rid"] for r in jl_read(path)}
                if all(s["rid"] in have for s in specs):
                    log(f"[{stage}] {m}：已完成 {len(specs)} 条")
                    continue
                backend().set_lora(m, adapter_path(m))
                t1 = time.time()
                n, tm = run_specs(backend, pl, specs, path, sig, stop_after=stop_after, label=f"{stage} {m}")
                n_tot += n
                add_timing(f"{stage}_{m}", time.time() - t1, {"gpu_stage": True, "rollouts": n})
    finally:
        release_vllm(holder.get("b"))
    log(f"[{stage}] 本次共 {n_tot} 条，用时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


def _code():
    return {f: file_sha256(C.ROOT / "alfw" / f)[:16] for f in ("agent.py", "rollout.py")} | \
        {"rt_eval.py": file_sha256(Path(__file__))[:16]}


# ---------------------------------------------------------------- 格式检查（5.5）
def stage_formatcheck(models=None, stop_after=None):
    G = jload(P("games.json"))
    models = models or all_models()

    def specs(m):
        return [{"rid": f"{m}|fmt|{g}", "model": m, "gamefile": g, "prefix": [], "seed": derive_seed("FMT", g)}
                for g in G["fmt"]]
    _eval_models("fmt", models, specs, P("eval_fmt.jsonl"), stop_after)
    rows = jl_read(P("eval_fmt.jsonl"))
    rate = {}
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        rate[m] = sum(r["first_bad"] for r in rs) / max(1, sum(r["n_steps"] for r in rs))
    base = rate["B"]
    bad = {m: v for m, v in rate.items() if m != "B" and v > base + CFG["FORMAT_MAX_EXCESS"]}
    res = {"rate_first_bad": rate, "base": base, "max_excess": CFG["FORMAT_MAX_EXCESS"], "fail": bad,
           "pass": not bad, "time": time.strftime("%Y-%m-%d %H:%M"), "models": models}
    jdump(P("formatcheck.json"), res)
    log("[formatcheck] 第一次回答就不可执行 / 无法解析的步的比例：" +
        "，".join(f"{m} {v:.3f}" for m, v in rate.items()) +
        ("；通过" if not bad else f"；不通过：{bad}（允许按 5.5 改一次超参，所有组一起改，先登记）"))
    return 0 if not bad else 3


# ---------------------------------------------------------------- 读数（5.6）
def takeover_points():
    pos = jload(P("positions_tk.json"))
    eps = {r["rid"]: r for r in jl_read(P("episodes_tksrc.jsonl"))}
    pts = []
    for tr in pos["trajectories"]:
        for p in tr["positions"]:
            pts.append({"pid": f"{tr['traj_id']}|t{p['t']}", "traj_id": tr["traj_id"], "gamefile": tr["gamefile"],
                        "t": p["t"], "prefix": eps[tr["traj_id"]]["cmds"][:p["t"]]})
    return pts


def stage_evaluate(models=None, force=False, stop_after=None):
    if not force:
        fc = jload(P("formatcheck.json")) if P("formatcheck.json").exists() else None
        if not fc or not fc["pass"]:
            sys.exit("[evaluate] 格式检查还没跑或没通过（见 formatcheck.json）；先处理，确认后可加 --force")
    G = jload(P("games.json"))
    pts = takeover_points()
    models = models or all_models()
    src = {x["g"]: x["src"] for x in G["E"]}

    def e2e(m):
        return [{"rid": f"{m}|e2e|{x['g']}", "model": m, "gamefile": x["g"], "src": src[x["g"]], "prefix": [],
                 "seed": derive_seed("E2E", x["g"])} for x in G["E"]]

    def tk(m):
        return [{"rid": f"{m}|tk|{p['pid']}#{k}", "model": m, "pid": p["pid"], "traj_id": p["traj_id"], "k": k,
                 "gamefile": p["gamefile"], "prefix": p["prefix"], "seed": derive_seed("TKR", p["pid"], k)}
                for p in pts for k in range(CFG["TK_M"])]

    def both(m):
        return tk(m) + e2e(m)
    # 同一模型的接管与端到端放在一次流式续跑里（一次 adapter 切换），结果按 rid 前缀分到两个文件
    path = P("eval_rollouts.jsonl")
    _eval_models("eval", models, both, path, stop_after)
    rows = jl_read(path)
    from collections import Counter
    cnt = Counter((r["model"], "tk" if "|tk|" in r["rid"] else "e2e") for r in rows)
    log("[evaluate] 已有结果：" + "，".join(f"{m} 接管 {cnt[(m, 'tk')]} / 端到端 {cnt[(m, 'e2e')]}" for m in models))
    return 0
