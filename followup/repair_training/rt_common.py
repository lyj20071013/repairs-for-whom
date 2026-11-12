"""第 5 篇（repair_training）共用：登记的设置、路径、日志、文件读写、游戏划分、续跑缓存、计时。

设置的权威来源是 论文方案.md 第 5 节（先导 P0，2026-10-28 17:12 登记）；这里的数值逐项对应，改动要先在方案里登记。
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def _sanitize_thread_env():
    """线程相关环境变量的保护（10-28 18:51 14b2 挂卡后 vLLM 启动失败：libgomp 报 OMP_NUM_THREADS 的值不合法，
    mkl-service 报 MKL_THREADING_LAYER=INTEL 与 libgomp 不兼容）。必须在 import numpy / torch / vllm 之前执行；
    原值记在 ENV_FIX 里写进日志。不影响任何结果（线程数只影响 CPU 上的速度）。"""
    fix = {}
    o = os.environ.get("OMP_NUM_THREADS")
    if o is not None and not (o.strip().isdigit() and int(o.strip()) > 0):
        fix["OMP_NUM_THREADS"] = (o, "4")
        os.environ["OMP_NUM_THREADS"] = "4"
    m = os.environ.get("MKL_THREADING_LAYER")
    if m != "GNU":
        fix["MKL_THREADING_LAYER"] = (m, "GNU")
        os.environ["MKL_THREADING_LAYER"] = "GNU"
    return fix


ENV_FIX = _sanitize_thread_env()

from alfw import acfg  # noqa: E402

ROOT = Path(__file__).resolve().parent
STATE = {"mock": False, "out": ROOT / "out_rt", "profile": "p0"}

# ---------------------------------------------------------------- 登记的设置（论文方案.md 5.2–5.7）
CFG = {
    # 5.2 游戏
    "SEED_69": 2028, "N_69": 220,             # 6.9 的 220 局（与上一篇完全相同的抽样）
    "SEED_E_EXTRA": 2031, "N_E_EXTRA": 400,   # 评测局：train 中另抽
    "SEED_TRAIN": 2030, "N_TRAIN": 1000,      # 先导训练局
    "SEED_SMOKE": 2033, "N_SMOKE": 50,        # 冒烟与格式检查
    "SEED_E_ORDER": 2037,                     # 评测局的固定随机顺序（接管来源按此顺序取失败轨迹；5.11 实现补充）
    "EXCLUDE_TYPES": ["pick_two_obj_and_place"],
    # 5.3 决策点、候选、标签
    "N_POS": 3, "MIN_SLACK": 5, "PLAN_TIMEOUT": 60,
    "N_PROPOSE": 2, "N_RANDOM": 1, "PROPOSERS": ["q7", "qflash"],
    "K_LABEL": 3,
    # 5.4 训练集
    "FRAC_N": 0.5, "SEED_U": 2034, "SEED_R": 2035, "SEED_ANCHOR": 2036,
    "THOUGHT_TRIES": 3,
    "THOUGHT_MODE": "template",               # 5.14：修复样本的 Thought 统一用模板 "I will <动作>."（"rewrite" = 原 5.4 的改写，留作敏感性）
    "MAX_LEN": 6144,
    # 5.5 训练
    "LORA_R": 16, "LORA_ALPHA": 32, "LORA_DROPOUT": 0.05,
    "LORA_TARGETS": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "LR": 1e-4, "LR_DPO": 2e-5, "WARMUP": 0.03, "EPOCHS": 2, "BATCH": 16, "WEIGHT_DECAY": 0.0, "GRAD_CLIP": 1.0,
    "DPO_BETA": 0.1,
    "RUNS": [["S", 11], ["S", 12], ["S", 13], ["U", 11], ["U", 12], ["R", 11], ["R", 12], ["DPO-S", 11],
             ["S-plan", 11], ["S-plan", 12], ["A", 11], ["A", 12]],          # S-plan、A：5.12 增补（10-28）
    "FORMAT_MAX_EXCESS": 0.05,
    "FACT_FLAG": 0.10,
    # 5.15：中间检查点（只多存文件，不改变训练本身与 P0 的任何读数；供第 6 篇的剂量—反应分析）
    "CKPT_RUNS": ["S_s11", "A_s11"], "CKPT_FRACS": [0.25, 0.5, 0.75],                        # 5.12：某组"无依据断言"比例比 S 高出这么多就标记
    # 5.6 读数
    "N_FRESH": 3, "TK_N_TRAJ": 200, "TK_M": 3,
    # 5.7 统计
    "N_BOOT": 2000, "SEED_BOOT": 0,
    # 成本
    "PRICE_PER_GPU_HOUR": 5.5,
}
# 6.9 的 220 局游戏文件（排序后以换行连接）的 sha256：games 阶段核对重新抽出的列表与上一篇完全相同
HASH_69 = "7160cbb8ec00171c98e327462d007a6d03b8dff7692c4d1ac2fd85a848e5b74f"
Q7_DIR = os.environ.get("RT_Q7", str(acfg.MODEL_DIR))
PATCHES_ON = {"translate": True, "grammar": True}

# 冒烟（真实模型、极小规模，另存 out_rt_smoke/，结果不进入任何读数）与模拟（selftest）
SMOKE = {"N_TRAIN": 30, "N_E_SMOKE": 16, "TK_N_TRAJ": 4, "TK_M": 1, "K_LABEL": 2, "N_FRESH": 1, "MAX_POINTS": 12,
         "EPOCHS": 1, "RUNS": [["S", 11], ["DPO-S", 11], ["A", 11]], "N_FORMAT": 6}
MOCK = {"N_69": 20, "N_E_EXTRA": 20, "N_TRAIN": 80, "N_SMOKE": 10, "TK_N_TRAJ": 10, "TK_M": 2, "N_BOOT": 300}


def apply_profile(name):
    """p0：登记的设置；smoke：真实模型的极小规模；mock：模拟环境 + 模拟模型（selftest）。"""
    STATE["profile"] = name
    if name == "smoke":
        CFG.update(json.loads(json.dumps(SMOKE)))
    elif name == "mock":
        CFG.update(json.loads(json.dumps(MOCK)))
        STATE["mock"] = True


def use_out(out):
    STATE["out"] = Path(out).resolve()
    STATE["out"].mkdir(parents=True, exist_ok=True)


def P(name):
    return STATE["out"] / name


# ---------------------------------------------------------------- 日志与文件
_ENV_LOGGED = [False]


def log(msg):
    if not _ENV_LOGGED[0] and ENV_FIX:
        _ENV_LOGGED[0] = True
        log("[env] 线程环境变量已修正（原值 -> 新值）：" + "，".join(f"{k} {v[0]!r} -> {v[1]!r}" for k, v in ENV_FIX.items()))
    line = time.strftime("[%m-%d %H:%M:%S] ") + msg
    print(line, flush=True)
    try:
        with open(P("rt.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def fp(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jl_read(path):
    """读 jsonl；末尾写了一半的行忽略，其他位置损坏报错。"""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines) - 1:
                continue
            raise ValueError(f"损坏的文件 {path}（第 {i + 1} 行）")
    return out


def jl_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def repair_tail(path):
    """追加写之前截掉末尾未写完的半行。"""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        return
    with path.open("r+b") as f:
        f.truncate(raw.rfind(b"\n") + 1)


def derive_seed(*parts):
    from alfw.agent import derive_seed as ds
    return ds(*parts)


def hkey(*parts):
    """确定性的排序键（并列打破、样本槽位排序用）。"""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


# ---------------------------------------------------------------- 环境与模型
def kind():
    return "mock" if STATE["mock"] else "tw"


def make_pool():
    from alfw import pool
    return pool.make_pool(kind(), acfg.n_workers(), None if STATE["mock"] else PATCHES_ON)


def split_games(split):
    from alfw import games
    if STATE["mock"]:
        n, off = {"train": (2000, 0), "valid_seen": (14, 5000), "valid_unseen": (12, 6000)}[split]
        return [f"{games.MOCK_PREFIX}{off + i}" for i in range(n)]
    return games.cached_collect(split)


def task_type(g):
    from alfw import games
    return games.task_type_of(g)


TRAIN_EVAL_KEYS = ("LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "LORA_TARGETS", "LR", "LR_DPO", "WARMUP", "EPOCHS", "BATCH",
                   "WEIGHT_DECAY", "GRAD_CLIP", "DPO_BETA", "RUNS", "FORMAT_MAX_EXCESS", "FACT_FLAG", "TK_M", "N_BOOT", "SEED_BOOT",
                   "THOUGHT_MODE", "CKPT_RUNS", "CKPT_FRACS",
                   "PRICE_PER_GPU_HOUR", "N_FORMAT")


def settings():
    """数据阶段的设置签名：不含训练 / 评测 / 分析的设置，也不含只影响 thoughts 之后阶段的设置（THOUGHT_*；
    这些阶段有自己的签名）。按 5.5 改超参、按 5.14 改 Thought 方式都不会让已采的数据失效。"""
    s = {k: v for k, v in CFG.items() if k not in TRAIN_EVAL_KEYS}
    s.update(MAX_STEPS=acfg.MAX_STEPS, TEMPERATURE=acfg.TEMPERATURE, TOP_P=acfg.TOP_P,
             MAX_NEW_TOKENS=acfg.MAX_NEW_TOKENS, MODEL=Q7_DIR, mock=STATE["mock"], profile=STATE["profile"])
    return s


def release_vllm(obj):
    """显式关闭 vLLM 引擎（与 ds_alfw.release_vllm 相同）。"""
    llm = getattr(obj, "llm", None)
    if llm is None:
        return
    for fn in (lambda: llm.llm_engine.engine_core.shutdown(), lambda: llm.shutdown()):
        try:
            fn()
            break
        except Exception:
            pass
    try:
        del obj.llm
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


BACKENDS = {"n": 0}


def make_backend(lora=False):
    """每个进程只能建一个 vLLM 引擎（显存）；自检核对这个计数。"""
    BACKENDS["n"] += 1
    if BACKENDS["n"] > 1:
        raise RuntimeError("同一进程里第二次建 vLLM 引擎；每个 GPU 阶段只能有一个（先 release_vllm 再换阶段）")
    from alfw import agent
    if STATE["mock"]:
        if lora:
            import rt_eval
            return rt_eval.MockLoraBackend()
        return agent.MockBackend()
    if lora:
        import rt_eval
        return rt_eval.LoraVLLMBackend(Q7_DIR)
    return agent.VLLMBackend(model_dir=Q7_DIR)


def gpu_info():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return [x.strip() for x in r.stdout.splitlines() if x.strip()]
    except Exception:
        return None


def add_timing(stage, wall, extra=None):
    """记录每个阶段的用时（卡时与按单价折算的费用；CPU 阶段也记，标明是否用 GPU）。"""
    t = jload(P("timing.json")) if P("timing.json").exists() else {"runs": []}
    rec = {"stage": stage, "wall_s": round(wall, 1), "time": time.strftime("%Y-%m-%d %H:%M"), "gpu": gpu_info()}
    rec.update(extra or {})
    t["runs"].append(rec)
    jdump(P("timing.json"), t)


# ---------------------------------------------------------------- 流式续跑（逐条落盘、断点续跑、签名核对）
def run_specs(backend_fn, pool, specs, path, sig, temperature=None, on_row=None, stop_after=None, label=""):
    """specs：[{"rid", "gamefile", "prefix", "seed", 其他字段原样写入}]；每条 rollout 结束写一行。

    已有结果按 rid 跳过；签名 sig 不同的旧结果拒绝混用。返回 (本次完成数, 计时)。
    stop_after：只用于自检（写够这么多行后模拟中断）。
    """
    from alfw import rollout
    path = Path(path)
    repair_tail(path)
    have = {}
    mine = {s["rid"] for s in specs}
    for r in jl_read(path):
        if r["rid"] not in mine:          # 同一文件里别的模型 / 阶段的结果
            continue
        if r.get("sig") != sig:
            sys.exit(f"[{label}] {path.name} 里已有结果的设置签名不同（代码、设计或设置改过）；请换输出目录")
        have[r["rid"]] = r
    todo = [s for s in specs if s["rid"] not in have]
    log(f"[{label}] 共 {len(specs)} 条，已完成 {len(specs) - len(todo)}，本次 {len(todo)}")
    if not todo:
        return 0, None
    temperature = acfg.TEMPERATURE if temperature is None else temperature
    backend = backend_fn()
    meta = {s["rid"]: s for s in todo}
    cnt = [0]
    f = open(path, "a", encoding="utf-8")

    def on_done(e):
        st = rollout.episode_stats(e)
        s = meta[e["rid"]]
        row = {k: v for k, v in s.items() if k not in ("prefix",)}
        row.update(sig=sig, won=bool(e["won"]), won_step=e["won_step"], n_steps=st["n"],
                   first_bad=st["first_bad"], bad=st["invalid"] + st["no_action"], retries=st["retries"],
                   trunc=st["trunc"])
        pol = [x for x in e["steps"] if x["kind"] != "prefix"]
        row["first_cmd"] = pol[0]["cmd"] if pol else None
        if on_row:
            on_row(row, e)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        cnt[0] += 1
        if stop_after is not None and cnt[0] >= stop_after:
            raise KeyboardInterrupt("selftest interrupt")
    try:
        tm = rollout.run_stream(backend, pool, iter(todo), temperature, on_done, max_active=acfg.B_MAX_ACTIVE,
                                log=log, want_key=False)
        log(f"[{label}] 完成 {cnt[0]} 条：{tm['wall'] / 60:.1f} 分钟，{tm['policy_steps']} 步，"
            f"{tm['policy_steps'] / max(tm['wall'], 1e-9):.1f} 步/秒")
        return cnt[0], tm
    finally:
        f.close()


def wanted_runs():
    """[(组, 种子)]；模型名 = 组_s种子。"""
    return [(g, int(s)) for g, s in CFG["RUNS"]]


def model_name(group, seed):
    return f"{group}_s{seed}"


def adapter_dir(group, seed):
    return P("adapters") / model_name(group, seed)


def all_models():
    return ["B"] + [model_name(g, s) for g, s in wanted_runs()]


def shuffle_with(seq, seed):
    x = list(seq)
    random.Random(seed).shuffle(x)
    return x
