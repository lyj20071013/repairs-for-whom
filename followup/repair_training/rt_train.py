"""LoRA 训练（论文方案.md 5.5）：SFT（S / U / R）与 DPO（DPO-S，次要）。不用 TRL：peft + 自写损失。

    python rt_train.py --group S --seed 11            训练一个模型 -> out_rt/adapters/S_s11/
    python rt_train.py --selftest                     服务器上的训练自检（极小的随机初始化 Qwen2 + 真实分词器；CPU 即可）

要点
- 样本 = 部署协议的完整提示（系统 + 用户，apply_chat_template + add_generation_prompt）+ 助手文本 + <|im_end|>。
  提示与助手分别分词再拼接（与 vLLM 生成时一致：提示按字符串分词，之后的 token 由模型生成），只在助手部分算损失。
- 长于 MAX_LEN 的样本丢弃并计数。
- 槽位：export 把修复样本与锚样本交替排成 2N 个槽位；每轮的顺序 = 由 (种子, 轮次) 决定的槽位排列，各组相同。
- LoRA 初始化与 dropout 由种子决定（torch.manual_seed 在 get_peft_model 之前），各组相同。
- DPO：参考模型 = 关掉 adapter 的基础模型；参考对数概率在训练开始前算一次（eval 模式）。损失 = 修复对上的 DPO + 锚样本上的 SFT。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import rt_common as C   # noqa: I001  先于 numpy / torch 导入：rt_common 在导入时修正线程环境变量
import numpy as np
from rt_common import CFG, P, add_timing, adapter_dir, file_sha256, jdump, jl_read, jload, log, model_name

END = "<|im_end|>"


# ---------------------------------------------------------------- 纯数据部分（模拟自检也用）
def slot_order(slots, seed, epoch):
    """各组相同的槽位排列：只依赖槽位编号集合、种子与轮次。"""
    s = sorted(slots)
    perm = np.random.default_rng([int(seed), int(epoch), 20261028]).permutation(len(s))
    return [s[i] for i in perm]


def micro_schedule(slots, seed, epochs, batch, n_slots=None):
    """返回 [[槽位, ...], ...]：每个内层列表是一次优化步的梯度累积组。

    n_slots：全部槽位数（2N）。给出时，先对 0..n_slots−1 做与其他组相同的排列，再只保留本组有的槽位
    （A 组只有锚样本槽位；个别样本因过长被丢弃时也一样），这样各组的共同槽位顺序完全相同。"""
    have = set(slots)
    uni = range(n_slots) if n_slots else sorted(have)
    seq = []
    for ep in range(epochs):
        seq += [s for s in slot_order(uni, seed, ep) if s in have]
    return [seq[i:i + batch] for i in range(0, len(seq), batch)]


def n_slots_of(group):
    try:
        return jload(P("train_manifest.json"))["files"][group].get("n_slots")
    except Exception:
        return None


def lr_at(step, total, base, warmup_frac):
    w = max(1, math.ceil(warmup_frac * total))
    if step < w:
        return base * (step + 1) / w
    prog = (step - w) / max(1, total - w)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def group_lr(group):
    return CFG["LR_DPO"] if group.startswith("DPO") else CFG["LR"]


def train_file(group):
    return P(f"train_{group}.jsonl")


def run_key(group, seed):
    man = jload(P("train_manifest.json"))
    keys = ("LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "LORA_TARGETS", "LR", "LR_DPO", "WARMUP", "EPOCHS", "BATCH",
            "WEIGHT_DECAY", "GRAD_CLIP", "DPO_BETA", "MAX_LEN")
    return C.fp({"group": group, "seed": seed, "file": man["files"][group]["sha256"], "cfg": {k: CFG[k] for k in keys},
                 "model": C.Q7_DIR, "code": file_sha256(Path(__file__))})


def is_done(group, seed):
    f = adapter_dir(group, seed) / "done.json"
    return f.exists() and jload(f).get("key") == run_key(group, seed)


# ---------------------------------------------------------------- 模拟训练（selftest：不加载任何模型）
MOCK_SKILL = {"S": 0.85, "U": 0.75, "R": 0.35, "DPO-S": 0.8, "S-plan": 0.82, "A": 0.6}


def mock_train(group, seed):
    rows = jl_read(train_file(group))
    sched = micro_schedule([r["slot"] for r in rows], seed, CFG["EPOCHS"], CFG["BATCH"], n_slots_of(group))
    out = adapter_dir(group, seed)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    jdump(out / "mock_adapter.json", {"skill": MOCK_SKILL[group] + float(rng.normal(0, 0.02)), "group": group})
    jdump(out / "done.json", {"key": run_key(group, seed), "n_rows": len(rows), "n_steps": len(sched),
                              "order_head": [s for g in sched[:2] for s in g], "mock": True})
    return 0


# ---------------------------------------------------------------- 真实训练
def load_tok(model_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    if END not in tok.get_vocab():
        raise RuntimeError(f"分词器里没有 {END}（不是 Qwen2.5-Instruct？）")
    return tok


def render(tok, messages):
    """与 alfw.agent.VLLMBackend.render 完全相同的调用。"""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode(tok, messages, completion):
    p = tok(render(tok, messages), add_special_tokens=False)["input_ids"]
    c = tok(completion + END, add_special_tokens=False)["input_ids"]
    return list(p), list(c)


def load_model(model_dir, dtype_name="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM
    dt = getattr(torch, dtype_name)
    kw = dict(attn_implementation="sdpa")
    try:
        m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dt, **kw)
    return m


def add_lora(model, seed):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(int(seed))
    cfg = LoraConfig(r=CFG["LORA_R"], lora_alpha=CFG["LORA_ALPHA"], lora_dropout=CFG["LORA_DROPOUT"],
                     target_modules=list(CFG["LORA_TARGETS"]), bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def seq_logp(model, p_ids, c_ids, device, mean=False):
    """助手部分的对数概率之和（或每 token 平均）。只取最后 len(c)+1 个位置的 logits（logits_to_keep），省显存。"""
    import torch
    ids = torch.tensor([p_ids + c_ids], device=device)
    k = len(c_ids) + 1
    try:
        out = model(input_ids=ids, logits_to_keep=k, use_cache=False)
        logits = out.logits[0, :-1]                             # 预测 c_ids 的 k-1 个位置
    except TypeError:
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0, -k:-1]
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = torch.tensor(c_ids, device=device)
    tok_lp = lp.gather(1, tgt[:, None])[:, 0]
    return tok_lp.mean() if mean else tok_lp.sum()


def ckpt_steps(group, seed, n_steps):
    """5.15：要额外保存中间检查点的优化步（从 1 起计；第 k 步做完后保存）。"""
    if model_name(group, seed) not in CFG.get("CKPT_RUNS", []):
        return set()
    return {max(1, int(round(f * n_steps))) for f in CFG.get("CKPT_FRACS", [])} - {n_steps}


def train(group, seed, model_dir=None, out=None, rows=None, max_steps=None, log_every=10, device=None, n_slots=None,
          ckpts=None):
    import torch
    model_dir = model_dir or C.Q7_DIR
    out = Path(out or adapter_dir(group, seed))
    t0 = time.time()
    if rows is None:
        rows = jl_read(train_file(group))
        n_slots = n_slots or n_slots_of(group)
    tok = load_tok(model_dir)
    enc, dropped = {}, []
    for r in rows:
        p, c = encode(tok, r["messages"], r["completion"])
        x = {"kind": r["kind"], "p": p, "c": c}
        n = len(p) + len(c)
        if r["kind"] == "pair":
            _, cr = encode(tok, r["messages"], r["rejected"])
            x["cr"] = cr
            n = max(n, len(p) + len(cr))
        if n > CFG["MAX_LEN"]:
            dropped.append(r["slot"])
            continue
        enc[r["slot"]] = x
    sched = micro_schedule(list(enc), seed, CFG["EPOCHS"], CFG["BATCH"], n_slots)
    if max_steps:
        sched = sched[:max_steps]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_dir, "bfloat16" if device == "cuda" else "float32").to(device)
    model.config.use_cache = False
    if device == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = add_lora(model, seed)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[train {group} s{seed}] 样本 {len(enc)}（丢弃过长 {len(dropped)}），优化步 {len(sched)}，"
        f"可训练参数 {ntr / 1e6:.1f}M，设备 {device}")
    ckpt_set = set(ckpts) if ckpts is not None else ckpt_steps(group, seed, len(sched))
    is_dpo = group.startswith("DPO")
    ref = {}
    if is_dpo:                                   # 参考对数概率：关掉 adapter、eval 模式，训练前算一次
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            for s, x in enc.items():
                if x["kind"] == "pair":
                    ref[s] = (float(seq_logp(model, x["p"], x["c"], device)), float(seq_logp(model, x["p"], x["cr"], device)))
        log(f"[train {group} s{seed}] 参考对数概率 {len(ref)} 对")
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=group_lr(group),
                            weight_decay=CFG["WEIGHT_DECAY"])
    hist, ntok, t1 = [], 0, time.time()
    beta = CFG["DPO_BETA"]
    for step, grp in enumerate(sched):
        lr = lr_at(step, len(sched), group_lr(group), CFG["WARMUP"])
        for g in opt.param_groups:
            g["lr"] = lr
        tot, parts = 0.0, {"sft": [], "dpo": [], "acc": []}
        for s in grp:
            x = enc[s]
            if x["kind"] == "pair":
                pc = seq_logp(model, x["p"], x["c"], device)
                pr = seq_logp(model, x["p"], x["cr"], device)
                rc, rr = ref[s]
                z = beta * ((pc - rc) - (pr - rr))
                loss = -torch.nn.functional.logsigmoid(z)
                parts["dpo"].append(float(loss.detach()))
                parts["acc"].append(float(z > 0))
                ntok += len(x["p"]) * 2 + len(x["c"]) + len(x["cr"])
            else:
                loss = -seq_logp(model, x["p"], x["c"], device, mean=True)
                parts["sft"].append(float(loss.detach()))
                ntok += len(x["p"]) + len(x["c"])
            (loss / len(grp)).backward()
            tot += float(loss.detach()) / len(grp)
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CFG["GRAD_CLIP"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) in ckpt_set:                     # 只存 adapter，不影响训练（不碰随机数、优化器状态）
            cdir = out / f"ckpt_step{step + 1:04d}"
            cdir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(cdir))
            jdump(cdir / "ckpt.json", {"step": step + 1, "n_steps": len(sched), "frac": (step + 1) / len(sched),
                                       "group": group, "seed": seed})
            log(f"[train {group} s{seed}] 中间检查点：第 {step + 1}/{len(sched)} 步 -> {cdir.name}")
        rec = {"step": step, "lr": lr, "loss": tot, "grad_norm": float(gn),
               "sft": float(np.mean(parts["sft"])) if parts["sft"] else None,
               "dpo": float(np.mean(parts["dpo"])) if parts["dpo"] else None,
               "dpo_acc": float(np.mean(parts["acc"])) if parts["acc"] else None}
        hist.append(rec)
        if step % log_every == 0 or step == len(sched) - 1:
            el = time.time() - t1
            log(f"[train {group} s{seed}] 步 {step + 1}/{len(sched)} loss {tot:.4f} "
                + (f"sft {rec['sft']:.4f} " if rec["sft"] is not None else "")
                + (f"dpo {rec['dpo']:.4f} acc {rec['dpo_acc']:.2f} " if rec["dpo"] is not None else "")
                + f"lr {lr:.2e} |g| {rec['grad_norm']:.2f}；{ntok / max(el, 1e-9):.0f} token/秒，"
                  f"预计还要 {el / (step + 1) * (len(sched) - step - 1) / 60:.1f} 分钟")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    wall = time.time() - t0
    import peft
    import transformers
    meta = {"group": group, "seed": seed, "n_rows": len(rows), "n_used": len(enc), "dropped_slots": dropped,
            "n_steps": len(sched), "wall_s": wall, "tokens": ntok, "trainable": ntr, "device": device,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
            "history": hist, "time": time.strftime("%Y-%m-%d %H:%M")}
    jdump(out / "train_log.json", meta)
    return meta, model


def stage_train(group, seed):
    if is_done(group, seed):
        log(f"[train {model_name(group, seed)}] 已完成（设置与训练文件未变），跳过")
        return 0
    if C.STATE["mock"]:
        return mock_train(group, seed)
    meta, _ = train(group, seed)
    out = adapter_dir(group, seed)
    files = {f.name: file_sha256(f) for f in sorted(out.iterdir()) if f.name.startswith("adapter_")}
    jdump(out / "done.json", {"key": run_key(group, seed), "adapter_files": files, "wall_s": meta["wall_s"],
                              "n_used": meta["n_used"], "n_dropped": len(meta["dropped_slots"])})
    add_timing(f"train_{model_name(group, seed)}", meta["wall_s"], {"gpu_stage": True, "tokens": meta["tokens"]})
    log(f"[train {model_name(group, seed)}] 完成：{meta['wall_s'] / 60:.1f} 分钟 -> {out}")
    return 0


# ---------------------------------------------------------------- 服务器上的训练自检（极小模型）
def selftest_tiny():
    import tempfile
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from alfw.agent import build_messages

    def ok(c, m):
        if not c:
            raise AssertionError(m)
        print("  OK ", m, flush=True)
    print("[rt_train 自检] 真实分词器 + 极小随机 Qwen2（CPU）")
    tok = load_tok(C.Q7_DIR)
    cfg = AutoConfig.from_pretrained(C.Q7_DIR)
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, max_window_layers=2).items():
        setattr(cfg, k, v)
    if hasattr(cfg, "layer_types"):
        cfg.layer_types = ["full_attention"] * 2
    torch.manual_seed(0)
    tiny = AutoModelForCausalLM.from_config(cfg).float()
    tmp = Path(tempfile.mkdtemp(prefix="rt_tiny_"))
    tiny.save_pretrained(tmp / "base")
    tok.save_pretrained(tmp / "base")
    msgs = build_messages("put a mug in shelf.", "You are in the middle of a room.",
                          [{"cmd": "go to desk 1", "obs": "You arrive at desk 1. On the desk 1, you see a mug 1."}],
                          ["take mug 1 from desk 1", "go to shelf 1", "look"], 50)
    comp = "Thought: The mug is here.\nAction: take mug 1 from desk 1"
    p, c = encode(tok, msgs, comp)
    ok(tok.decode(c) == comp + END, f"助手部分解码 = 文本 + {END}（{len(c)} 个 token 算损失）")
    ok(tok.decode(p).endswith("<|im_start|>assistant\n"), "提示以 <|im_start|>assistant\\n 结尾（add_generation_prompt）")
    full = tok(render(tok, msgs) + comp + END, add_special_tokens=False)["input_ids"]
    print(f"  注：整体分词与分开分词{'相同' if full == p + c else '在边界处不同（训练用分开分词，与生成时一致）'}")
    m = load_model(str(tmp / "base"), "float32")
    a = float(seq_logp(m, p, c, "cpu"))
    import torch.nn.functional as F
    with torch.no_grad():
        lg = m(input_ids=torch.tensor([p + c]), use_cache=False).logits[0]
        ref = F.log_softmax(lg.float(), -1)[len(p) - 1:len(p) + len(c) - 1].gather(1, torch.tensor(c)[:, None]).sum()
    ok(abs(a - float(ref)) < 1e-3, f"logits_to_keep 的对数概率与全量 logits 一致（{a:.4f} vs {float(ref):.4f}）")
    m1 = add_lora(load_model(str(tmp / "base"), "float32"), 11)
    m2 = add_lora(load_model(str(tmp / "base"), "float32"), 11)
    m3 = add_lora(load_model(str(tmp / "base"), "float32"), 12)
    A = lambda mm: next(v for k, v in mm.named_parameters() if "lora_A" in k).detach().clone()
    ok(torch.equal(A(m1), A(m2)) and not torch.equal(A(m1), A(m3)), "同一种子的 LoRA 初始化相同，不同种子不同")
    cfgd = m1.peft_config["default"]
    ok(cfgd.r == CFG["LORA_R"] and set(cfgd.target_modules) == set(CFG["LORA_TARGETS"]), "LoRA 秩与作用层与登记相同")
    m1.eval()
    with torch.no_grad():
        la = float(seq_logp(m1, p, c, "cpu"))
        with m1.disable_adapter():
            lb = float(seq_logp(m1, p, c, "cpu"))
    ok(abs(la - lb) < 1e-5, "初始化时 LoRA 的 B = 0：带 / 不带 adapter 的对数概率相同（DPO 初始损失 = ln 2）")
    rows = []
    for i in range(4):
        rows.append({"slot": 2 * i, "kind": "repair", "messages": msgs, "completion": comp})
        rows.append({"slot": 2 * i + 1, "kind": "anchor", "messages": msgs,
                     "completion": "Thought: Go to the shelf.\nAction: go to shelf 1"})
    old = dict(CFG)
    CFG.update(LR=5e-3, LR_DPO=5e-3, BATCH=2, EPOCHS=4)
    try:
        meta, mt = train("S", 11, model_dir=str(tmp / "base"), out=tmp / "ad", rows=rows, log_every=100, device="cpu",
                         ckpts=[])
        meta_c, mt_c = train("S", 11, model_dir=str(tmp / "base"), out=tmp / "adc", rows=rows, log_every=100,
                             device="cpu", ckpts=[4, 8])
        ok(sorted(p.name for p in (tmp / "adc").glob("ckpt_step*")) == ["ckpt_step0004", "ckpt_step0008"]
           and (tmp / "adc" / "ckpt_step0004" / "adapter_config.json").exists(), "中间检查点按指定的步保存（5.15）")
        ok([h["loss"] for h in meta_c["history"]] == [h["loss"] for h in meta["history"]],
           "保存中间检查点不改变训练过程（逐步损失完全相同）")
        h = meta["history"]
        # 极小随机模型：词表 15 万、嵌入与输出层冻结，只有 LoRA 能动，16 步里损失只能降一点；
        # 这里只检查梯度方向对（下降 > 0.2 nat），不检查降多少（10-28 18:37 服务器上：11.891 -> 10.975）
        ok(h[-1]["sft"] < h[0]["sft"] - 0.2, f"SFT 损失下降（{h[0]['sft']:.3f} -> {h[-1]['sft']:.3f}）")
        pr = [dict(r, kind="pair", rejected="Thought: Look around.\nAction: look") if r["kind"] == "repair" else r
              for r in rows]
        meta2, _ = train("DPO-S", 11, model_dir=str(tmp / "base"), out=tmp / "ad2", rows=pr, log_every=100, device="cpu")
        d0 = meta2["history"][0]["dpo"]
        ok(abs(d0 - math.log(2)) < 1e-3, f"DPO 第一步损失 = ln 2（{d0:.5f}）")
        ok(meta2["history"][-1]["dpo"] < d0, f"DPO 损失下降（{d0:.3f} -> {meta2['history'][-1]['dpo']:.3f}）")
    finally:
        CFG.clear()
        CFG.update(old)
    from peft import PeftModel
    mt.eval()
    re_m = PeftModel.from_pretrained(load_model(str(tmp / "base"), "float32"), str(tmp / "ad")).eval()
    with torch.no_grad():
        x1, x2 = float(seq_logp(mt, p, c, "cpu")), float(seq_logp(re_m, p, c, "cpu"))
    ok(abs(x1 - x2) < 1e-3, f"保存再加载的 adapter 输出相同（{x1:.4f} vs {x2:.4f}）")
    acfg_ = json.loads((tmp / "ad" / "adapter_config.json").read_text())
    ok(acfg_.get("r") == CFG["LORA_R"], "adapter_config.json 可读，秩正确（vLLM 用它加载）")
    o1 = micro_schedule(list(range(10)), 11, 2, 4)
    ok(o1 == micro_schedule(list(range(10)), 11, 2, 4) and o1 != micro_schedule(list(range(10)), 12, 2, 4),
       "槽位顺序由种子决定")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("[rt_train 自检] 全部通过", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true", help="依次训练登记的全部 (组, 种子)（每个在单独的子进程里）")
    ap.add_argument("--profile", default="p0", choices=["p0", "smoke", "mock"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest_tiny()
    C.apply_profile(a.profile)
    C.use_out(a.out_dir or (C.ROOT / {"p0": "out_rt", "smoke": "out_rt_smoke", "mock": "out_rt_mock"}[a.profile]))
    if a.all:
        import subprocess
        rc = 0
        for g, s in C.wanted_runs():
            if is_done(g, s):
                log(f"[train {model_name(g, s)}] 已完成，跳过")
                continue
            cmd = [sys.executable, __file__, "--group", g, "--seed", str(s), "--profile", a.profile,
                   "--out-dir", str(C.STATE["out"])]
            r = subprocess.run(cmd).returncode
            if r:
                log(f"[train {model_name(g, s)}] 失败（退出码 {r}）；后面的不跑了")
                return r
        return rc
    return stage_train(a.group, a.seed)


if __name__ == "__main__":
    sys.exit(main())
