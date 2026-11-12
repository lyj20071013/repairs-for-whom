"""vLLM 带 LoRA 与 transformers + PEFT 在真实 τ² 提示上是否一致（2026-11-01 写；第 6 篇 τ² 探针，T1 的加严核对）。

为什么要做：T1 原来的 LoRA 检查（train_tau2.py probe-lora）只用了 3 个很短的提示、一个随机 adapter，
比的只是第一个 token。真实的 τ² 提示有 6–17k token（系统提示 + 工具 schema + 多轮历史），Qwen3.5 又是
混合结构（线性注意力 + 完整注意力，完整注意力的 q_proj 带输出门）。若 vLLM 应用 LoRA 的方式与 PEFT 不同
（例如层的打包 / 切片、缩放），S1 和 S2 的退化都可能是部署造成的，S3 也不会好。

做法（只用 GPU，不需要任何 vLLM 服务；先停掉 8001 / 8003）：
  1. 从训练样本（真实 τ² 请求，提示 token 已与 vLLM /tokenize 核对一致）里按长度分层选 N 个，工具调用目标与文字目标各半。
  2. transformers + PEFT（训练时用的同一套）：每个变体（基础模型、各 adapter、S2 只留注意力 / 只留 MLP 的消融）
     对目标（助手部分）逐 token 求 log p（教师强制）；再贪心生成 --gen 个 token。
  3. 释放显存，进程内起 vLLM（enable_lora，与 serve_lora.sh 同版本），同样的 token id 用 prompt_logprobs 求同样的
     逐 token log p，同样贪心生成。
  4. 比较。主读数是"adapter 效应比" r = (vLLM 基础 NLL − vLLM 带 adapter NLL) / (HF 基础 NLL − HF 带 adapter NLL)：
     LoRA 应用正确时 r ≈ 1；没生效 ≈ 0；缩放或切片错了会明显偏离 1。按提示长度分桶报。

判定（事先写定，在看结果之前；每个 adapter 分别判）：
  通过   ：r 的中位数在 [0.8, 1.25]（全部样本、且提示 ≥ 10k token 的桶都满足），带 adapter 的 |NLL 差| 中位数 ≤ 0.05 nat/token，
           且基础模型的 |NLL 差| 中位数 ≤ 0.05。
  不通过 ：任何一个长度桶里 r 的中位数 < 0.5 或 > 2，或带 adapter 的 |NLL 差| 中位数 > 0.2。→ 部署不一致，S3 之前先修。
  不确定 ：其余情况。看分桶与消融再说。
  基础模型本身就差很多（|NLL 差| 中位数 > 0.05）时另报：那是引擎之间的数值差，不是 LoRA 的问题，但同样说明训练与部署看到的函数不同。

附带读数（不参与判定）：各 adapter 在训练样本上的 HF NLL（修复 / 锚分开）——说明 S1、S2 各自把训练集背到什么程度。

用法（14b2，τ² 的 venv，tau2_pilot 下）：见 run_probe.sh 的 parity 步。
"""
import argparse
import gc
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/root/autodl-tmp/cache")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_tau2 as TT  # noqa: E402  渲染、编码、加载模型都与训练完全相同

ATTN = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj)\.")
MLP = re.compile(r"\.(gate_proj|up_proj|down_proj)\.")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def pick(rows_enc, n):
    """工具调用目标与文字目标各 n/2 个，各自按提示长度等距取（含最长的）。"""
    out = []
    for key in ("tool", "text"):
        g = sorted([x for x in rows_enc if x["key"] == key], key=lambda x: len(x["p"]))
        k = min(len(g), n // 2 if key == "tool" else n - n // 2)
        if k == 0:
            continue
        idx = sorted({round(i * (len(g) - 1) / max(1, k - 1)) for i in range(k)})
        out += [g[i] for i in idx]
    return sorted(out, key=lambda x: len(x["p"]))


def make_ablations(src, out_dir, base_name):
    """把 adapter 的一部分 lora_B 置零：只留注意力（-attn）、只留 MLP（-mlp）。"""
    import shutil
    from safetensors.torch import load_file, save_file
    W = load_file(str(Path(src) / "adapter_model.safetensors"))
    res = {}
    for tag, drop in (("attn", MLP), ("mlp", ATTN)):
        d = Path(out_dir) / f"ablate_{base_name}_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(src) / "adapter_config.json", d / "adapter_config.json")
        W2 = {k: (v.clone().zero_() if ("lora_B" in k and drop.search(k)) else v) for k, v in W.items()}
        n0 = sum(1 for k in W if "lora_B" in k and drop.search(k))
        save_file(W2, str(d / "adapter_model.safetensors"))
        res[f"{base_name}-{tag}"] = str(d)
        log(f"消融 {base_name}-{tag}：置零 {n0} 个 lora_B → {d}")
    return res


# ---------------------------------------------------------------- transformers
def hf_phase(a, samples, variants):
    import torch
    from peft import PeftModel
    model, cls = TT.load_model(a.model_dir)
    model.to("cuda")
    model.eval()
    names = list(variants)
    pm = PeftModel.from_pretrained(model, variants[names[0]], adapter_name=names[0])
    for n in names[1:]:
        pm.load_adapter(variants[n], adapter_name=n)
    pm.eval()
    log(f"HF：加载类 {cls}，adapter {names}；快速实现 {TT.fast_path_info()}")
    R = {}

    def tf_logp(p, c):
        ids = torch.tensor([p + c], device="cuda")
        k = len(c) + 1
        try:
            lg = pm(input_ids=ids, logits_to_keep=k, use_cache=False).logits[0, :-1]
        except TypeError:
            lg = pm(input_ids=ids, use_cache=False).logits[0, -k:-1]
        lsm = torch.log_softmax(lg.float(), dim=-1)
        tgt = torch.tensor(c, device="cuda")
        lp = lsm.gather(1, tgt[:, None])[:, 0]
        top1 = (lsm.argmax(-1) == tgt)
        return lp.tolist(), top1.tolist()

    def greedy(p):
        x = torch.tensor([p], device="cuda")
        return pm.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=a.gen, do_sample=False)[0, len(p):].tolist()

    t0 = time.time()
    with torch.no_grad():
        for i, s in enumerate(samples):
            r = {}
            for v in ["base"] + names:
                if v == "base":
                    with pm.disable_adapter():
                        lp, t1 = tf_logp(s["p"], s["c"])
                        g = greedy(s["p"]) if a.gen else []
                else:
                    pm.set_adapter(v)
                    lp, t1 = tf_logp(s["p"], s["c"])
                    g = greedy(s["p"]) if (a.gen and v in a.gen_variants) else []
                r[v] = {"lp": lp, "top1": t1, "gen": g}
            R[s["slot"]] = r
            log(f"  HF {i + 1}/{len(samples)} 槽位 {s['slot']}（提示 {len(s['p'])}、目标 {len(s['c'])} token）"
                f" NLL 基础 {-sum(r['base']['lp']) / len(s['c']):.3f} | "
                + " ".join(f"{v} {-sum(r[v]['lp']) / len(s['c']):.3f}" for v in names)
                + f"；累计 {time.time() - t0:.0f} 秒")
    del pm, model
    gc.collect()
    torch.cuda.empty_cache()
    return R


# ---------------------------------------------------------------- vLLM
def vllm_phase(a, samples, variants):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    names = list(variants)
    max_len = max(len(s["p"]) + len(s["c"]) for s in samples) + a.gen + 64
    llm = LLM(model=a.model_dir, enable_lora=True, max_lora_rank=16, max_loras=len(names), max_model_len=max_len,
              gpu_memory_utilization=a.vllm_util, max_num_seqs=4, max_num_batched_tokens=a.chunk,
              enable_prefix_caching=False)
    import vllm
    log(f"vLLM {vllm.__version__} 已起（max_model_len {max_len}，前缀缓存关、分块 {a.chunk}）")
    lreq = {n: LoRARequest(n, i + 1, variants[n]) for i, n in enumerate(names)}
    sp_lp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    sp_g = SamplingParams(temperature=0.0, max_tokens=max(1, a.gen))
    R = {s["slot"]: {} for s in samples}
    t0 = time.time()
    for v in ["base"] + names:
        lr = None if v == "base" else lreq[v]
        inputs = [{"prompt_token_ids": s["p"] + s["c"]} for s in samples]
        outs = llm.generate(inputs, sp_lp, lora_request=lr, use_tqdm=False)
        for s, o in zip(samples, outs):
            pl = o.prompt_logprobs
            lp, top1 = [], []
            for j, tid in enumerate(s["c"]):
                d = pl[len(s["p"]) + j]
                e = d.get(tid) if d is not None else None
                if e is None and d is not None:
                    e = d.get(str(tid))
                lp.append(float(e.logprob) if e is not None else float("nan"))
                top1.append(bool(e is not None and getattr(e, "rank", None) == 1))
            R[s["slot"]][v] = {"lp": lp, "top1": top1, "gen": []}
        if a.gen and (v == "base" or v in a.gen_variants):
            outs = llm.generate([{"prompt_token_ids": s["p"]} for s in samples], sp_g, lora_request=lr, use_tqdm=False)
            for s, o in zip(samples, outs):
                R[s["slot"]][v]["gen"] = list(o.outputs[0].token_ids)
        log(f"  vLLM {v} 完成；累计 {time.time() - t0:.0f} 秒")
    return R


# ---------------------------------------------------------------- 比较
def compare(samples, H, V, names, adapters):
    rows = []
    for s in samples:
        n = len(s["c"])
        r = {"slot": s["slot"], "kind": s["kind"], "key": s["key"], "domain": s["domain"], "p_len": len(s["p"]), "c_len": n}
        for v in ["base"] + names:
            h, w = H[s["slot"]][v], V[s["slot"]][v]
            r[f"nll_hf_{v}"] = -sum(h["lp"]) / n
            r[f"nll_vl_{v}"] = -sum(w["lp"]) / n
            r[f"dtok_{v}"] = sum(abs(x - y) for x, y in zip(h["lp"], w["lp"])) / n
            r[f"dmax_{v}"] = max(abs(x - y) for x, y in zip(h["lp"], w["lp"]))
            r[f"top1_hf_{v}"] = sum(h["top1"]) / n
            r[f"top1_vl_{v}"] = sum(w["top1"]) / n
            if h["gen"] and w["gen"]:
                k = next((i for i, (x, y) in enumerate(zip(h["gen"], w["gen"])) if x != y), min(len(h["gen"]), len(w["gen"])))
                r[f"gen_prefix_{v}"] = k
        for v in names:
            eh = r["nll_hf_base"] - r[f"nll_hf_{v}"]
            ev = r["nll_vl_base"] - r[f"nll_vl_{v}"]
            r[f"eff_hf_{v}"], r[f"eff_vl_{v}"] = eh, ev
            r[f"ratio_{v}"] = ev / eh if abs(eh) >= 0.05 else None
        rows.append(r)
    buckets = [("<6k", 0, 6000), ("6–10k", 6000, 10000), ("≥10k", 10000, 10 ** 9)]
    verdict = {}
    base_d = med([abs(r["nll_vl_base"] - r["nll_hf_base"]) for r in rows])
    for v in adapters:
        d = med([abs(r[f"nll_vl_{v}"] - r[f"nll_hf_{v}"]) for r in rows])
        rb = {b: med([r[f"ratio_{v}"] for r in rows if lo <= r["p_len"] < hi]) for b, lo, hi in buckets}
        rall = med([r[f"ratio_{v}"] for r in rows])
        big = rb.get("≥10k")
        ok = (rall is not None and 0.8 <= rall <= 1.25 and (big is None or 0.8 <= big <= 1.25)
              and d is not None and d <= 0.05 and base_d is not None and base_d <= 0.05)
        bad = (d is not None and d > 0.2) or any(x is not None and (x < 0.5 or x > 2) for x in list(rb.values()) + [rall])
        verdict[v] = {"result": "通过" if ok else ("不通过" if bad else "不确定"), "ratio_median": rall,
                      "ratio_by_len": rb, "abs_dnll_median": d, "base_abs_dnll_median": base_d,
                      "n_ratio": sum(r[f"ratio_{v}"] is not None for r in rows)}
    return rows, verdict, buckets


def fmt(x, k=3):
    return "—" if x is None else f"{x:.{k}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="名字=目录；第一个以外也可以有多个（例 4B-S2=… 4B-S1=…）")
    ap.add_argument("--ablate", default="", help="对这个 adapter 做注意力 / MLP 消融（例 4B-S2）；空 = 不做")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--gen", type=int, default=48, help="贪心生成的 token 数（0 = 不生成）")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--vllm-util", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=4096, help="vLLM 的 max_num_batched_tokens（限制 prompt_logprobs 的显存）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adapters = dict(x.split("=", 1) for x in a.adapters)
    for n, d in adapters.items():
        if not (Path(d) / "adapter_model.safetensors").exists():
            log(f"[停下] {d} 里没有 adapter_model.safetensors")
            return 2
    variants = dict(adapters)
    if a.ablate:
        variants.update(make_ablations(adapters[a.ablate], Path(adapters[a.ablate]).parent, a.ablate))
    a.gen_variants = set(adapters)                 # 消融变体不做贪心生成，只比 log p
    names = list(variants)

    tok = TT.load_tok(a.model_dir)
    enc = []
    for r in TT.jl(a.samples):
        p, c, _, _, mode = TT.encode(tok, r)
        if len(p) + len(c) > a.max_len:
            continue
        enc.append({"slot": r["slot"], "kind": r["kind"], "domain": r["domain"], "p": p, "c": c,
                    "key": "tool" if r["target"].get("tool_calls") else "text"})
    samples = pick(enc, a.n)
    log(f"样本 {len(enc)} 个可用，选 {len(samples)} 个：提示长度 {[len(s['p']) for s in samples]}；"
        f"工具 / 文字目标 {sum(s['key'] == 'tool' for s in samples)} / {sum(s['key'] == 'text' for s in samples)}")

    t0 = time.time()
    H = hf_phase(a, samples, variants)
    log(f"HF 阶段完成（{(time.time() - t0) / 60:.1f} 分钟）")
    V = vllm_phase(a, samples, variants)
    log(f"vLLM 阶段完成（共 {(time.time() - t0) / 60:.1f} 分钟）")

    rows, verdict, buckets = compare(samples, H, V, names, list(adapters))
    print("\n== 逐样本（NLL = 目标部分平均每 token 的负 log p，nat；r = vLLM 的 adapter 效应 / HF 的 adapter 效应）")
    for r in rows:
        line = f"槽位 {r['slot']:>3} {r['kind']:<6} {r['key']:<4} {r['domain']:<7} 提示 {r['p_len']:>5} 目标 {r['c_len']:>3} | 基础 HF {fmt(r['nll_hf_base'])} vLLM {fmt(r['nll_vl_base'])}"
        for v in names:
            line += f" | {v} HF {fmt(r['nll_hf_' + v])} vLLM {fmt(r['nll_vl_' + v])} r {fmt(r['ratio_' + v], 2)}"
        print(line)
    print("\n== 训练集拟合（HF，带 adapter 的 NLL 中位数；不参与判定）")
    for v in ["base"] + list(adapters):
        for kind in ("repair", "anchor"):
            print(f"  {v:<8} {kind:<6} {fmt(med([r['nll_hf_' + v] for r in rows if r['kind'] == kind]))}"
                  f"  目标 token 为 top-1 的比例 {fmt(med([r['top1_hf_' + v] for r in rows if r['kind'] == kind]), 2)}")
    import math
    print("\n== 决策 token：目标第一个 token 的概率（HF，中位数；第一个 token 决定回文字还是调工具、调哪个；不参与判定）")
    for v in ["base"] + list(adapters):
        print(f"  {v:<8} " + "  ".join(
            f"{kind} {fmt(med([math.exp(H[s['slot']][v]['lp'][0]) for s in samples if s['kind'] == kind]), 3)}"
            f"（其中 <0.5 的 {sum(math.exp(H[s['slot']][v]['lp'][0]) < 0.5 for s in samples if s['kind'] == kind)}"
            f"/{sum(s['kind'] == kind for s in samples)}）" for kind in ("repair", "anchor")))
    if a.gen:
        print("\n== 贪心生成：HF 与 vLLM 从第几个 token 开始不同（中位数；满 = 全部相同）")
        for v in ["base"] + list(adapters):
            ks = [r.get("gen_prefix_" + v) for r in rows]
            print(f"  {v:<8} 中位 {med(ks)}，全部相同 {sum(k == a.gen for k in ks if k is not None)}/{len(ks)}")
    print("\n== 分桶（r 的中位数）")
    for v in names:
        print(f"  {v:<12} " + "  ".join(
            f"{b}: {fmt(med([r['ratio_' + v] for r in rows if lo <= r['p_len'] < hi]), 2)}"
            f"（n={sum(1 for r in rows if lo <= r['p_len'] < hi)}）" for b, lo, hi in buckets)
            + f"  | 带 adapter |ΔNLL| 中位 {fmt(med([abs(r['nll_vl_' + v] - r['nll_hf_' + v]) for r in rows]))}"
            + f"  逐 token |Δlog p| 中位 {fmt(med([r['dtok_' + v] for r in rows]))}")
    print(f"  基础模型 |ΔNLL| 中位 {fmt(med([abs(r['nll_vl_base'] - r['nll_hf_base']) for r in rows]))}"
          f"  逐 token |Δlog p| 中位 {fmt(med([r['dtok_base'] for r in rows]))}")
    print("\n== 判定（事先写定的规则，见文件开头）")
    for v, x in verdict.items():
        print(f"  {v}：{x['result']}（r 中位 {fmt(x['ratio_median'], 2)}，按长度 "
              + "、".join(f"{b} {fmt(y, 2)}" for b, y in x["ratio_by_len"].items())
              + f"；带 adapter |ΔNLL| 中位 {fmt(x['abs_dnll_median'])}；基础 {fmt(x['base_abs_dnll_median'])}；有效 r {x['n_ratio']} 个）")
    rep = {"time": time.strftime("%Y-%m-%d %H:%M"), "args": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(a).items()},
           "variants": variants, "verdict": verdict, "rows": rows,
           "raw": {str(s["slot"]): {"hf": H[s["slot"]], "vllm": V[s["slot"]], "p_len": len(s["p"]), "c": s["c"]} for s in samples}}
    out.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    log(f"写出 {out}")
    res = {x["result"] for x in verdict.values()}
    print("== 通过" if res == {"通过"} else ("== 不通过：见上" if "不通过" in res else "== 不确定：见上"))
    return 0 if res == {"通过"} else (1 if "不通过" in res else 3)


if __name__ == "__main__":
    sys.exit(main())
