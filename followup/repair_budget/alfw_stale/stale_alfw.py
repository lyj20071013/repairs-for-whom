"""第 6 篇 A 层：ALFWorld 上的标签过期实验——续跑部分（GPU）。登记见 D:\\Paper\\repair_budget\\登记_ALFWorld过期_20261030.md
（第 1–7 节 10-30 14:16；第 8 节补充一 15:00）。分析在 stale_analyze.py（只用 CPU）。

只读复用第 5 篇 P0 的代码（RT_CODE，默认 /root/paper/repair_training：alfw/、rt_common.py、rt_eval.py），不改它的任何文件；
P0 的 adapter 与检查点从 RT_OUT 读（默认 /root/autodl-tmp/rt_runs/out_rt，没有就用 RT_CODE/out_rt）。
未训练的位置 = 第 4 篇 6.9 池：data/design_69.json、data/evals_q7_69.jsonl（从 D:\\Paper\\repair_transfer\\out_ds_alfw 原样复制，sha256 见 DATA_SHA）。

阶段（一个进程只建一个开了 LoRA 的 vLLM 引擎，各模型依次换 adapter；与 P0 的评测相同）：
  check            不用 GPU：核对数据、adapter、检查点，列出各阶段的续跑数
  drift            登记 4.1：每个模型在 180 个位置各采样 20 个第一步（含不可执行时的重答），B 另采 20 个作噪声底
  free_b           B 在 180 个位置的自由选择，各 10 次
  full             --models（默认 S_s11,S_s12）：1,286 个分支 + 180 个自由选择，各 10 次
  reverse          --models（默认 S_s11）：π0 与 π1 用全部 10 次选出的最好修复不同的位置（至多 60 个），两个候选各 30 次
  two              --models（默认 A_s11,R_s11,U_s11,S-plan_s11,DPO-S_s11）：π0 的最好修复与自由选择，各 10 次（补充一 8.3）
用法：python stale_alfw.py check
      python stale_alfw.py run --stages drift,free_b,full,reverse      （优先级 1）
      python stale_alfw.py run --stages two                            （优先级 2）
      --profile mock：模拟环境 + 模拟模型（工作区自检；先 python stale_alfw.py mocksetup）
种子（共同随机数，不含模型名）：分支续跑 derive_seed("STALE", 分支 id, i)；自由选择 derive_seed("STALE", 位置 id + "|free", i)；
  反转检查 derive_seed("STALE-REV", 分支 id, i)；漂移 derive_seed("DRIFT", 位置 id, 模型, i)（漂移不用共同随机数，见 specs_drift）。每一步的采样种子 = derive_seed(种子, 绝对步号)（P0 / 第 4 篇的协议）。
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RT_CODE = Path(os.environ.get("RT_CODE", "/root/paper/repair_training"))
sys.path.insert(0, str(RT_CODE))
os.environ.setdefault("PYTHONHASHSEED", "0")

import rt_common as C  # noqa: E402  只读复用
from rt_common import derive_seed, fp, file_sha256, jdump, jl_read, jload, log  # noqa: E402

DATA_SHA = {"design_69.json": "abed43d474e23894d6ec25651e542f46999083b7ae213d04362651b3b070c390",
            "evals_q7_69.jsonl": "76b4a64babca6a923f76c69855db793dce7c7b467c1b3c54556ebd5a042b70c4"}
K = 10                 # 每个分支、每个自由选择的续跑次数（登记 4.2）
M_DRIFT = 20           # 漂移：每个位置采样数（登记 4.1）
K_REV = 30             # 反转检查（登记 4.3）
N_REV = 60
FULL_DEFAULT = ["S_s11", "S_s12"]                                   # 补充一 8.2
TWO_DEFAULT = ["A_s11", "R_s11", "U_s11", "S-plan_s11", "DPO-S_s11"]
REV_DEFAULT = ["S_s11"]
FINALS = ["S_s11", "S_s12", "S_s13", "U_s11", "U_s12", "R_s11", "R_s12", "DPO-S_s11", "S-plan_s11", "S-plan_s12",
          "A_s11", "A_s12"]
STATE = {"mock": False, "data": HERE / "data", "rt_out": None, "out": None}


# ---------------------------------------------------------------- 路径与数据
def setup(profile, out):
    if profile == "mock":
        C.apply_profile("mock")
        STATE["mock"] = True
        STATE["data"] = Path(out) / "mock_data"
        STATE["rt_out"] = Path(out) / "mock_rt_out"
    else:
        cand = [os.environ.get("RT_OUT"), "/root/autodl-tmp/rt_runs/out_rt", str(RT_CODE / "out_rt")]
        STATE["rt_out"] = next((Path(c) for c in cand if c and (Path(c) / "adapters").is_dir()), None)
    STATE["out"] = Path(out)
    C.use_out(out)                      # rt_common 的日志写到本目录的 rt.log


def P(name):
    return STATE["out"] / name


def load_design():
    d = STATE["data"]
    if not STATE["mock"]:
        for f, h in DATA_SHA.items():
            got = file_sha256(d / f)
            if got != h:
                sys.exit(f"[stale] {d / f} 的 sha256 不对（{got}）；应与 repair_transfer/out_ds_alfw 的原文件相同")
    D = jload(d / "design_69.json")
    ev = {}
    for r in jl_read(d / "evals_q7_69.jsonl"):
        if r["condition"] == "q7_T07":
            ev[r["branch_id"]] = [bool(x) for x in r["results"]]
    pos = []
    for tr in D["trajectories"]:
        for p in tr["positions"]:
            ob = D["branches"][p["orig"]]
            pos.append({"pos_id": f"{tr['traj_id']}|t{p['t']}", "traj_id": tr["traj_id"], "t": p["t"],
                        "gamefile": tr["gamefile"], "orig": p["orig"], "alts": list(p["alts"]),
                        "free_prefix": list(ob["prefix"][:-1])})
    return D, ev, pos


def best_alt(p, scores):
    """最好的修复候选（不含原动作）；并列用 derive_seed("STALE-TIE", 位置, 动作) 较大者（登记第 5 节）。"""
    D = DESIGN[0]
    return max(p["alts"], key=lambda b: (scores[b], derive_seed("STALE-TIE", p["pos_id"], D["branches"][b]["action"])))


DESIGN = [None]


# ---------------------------------------------------------------- 模型与 adapter
def adapter_path(model):
    if model in ("B", "B#2"):
        return None
    base, _, step = model.partition("@")
    d = STATE["rt_out"] / "adapters" / base
    if step:
        d = d / f"ckpt_step{int(step):04d}"
        if not d.is_dir():
            sys.exit(f"[stale] 找不到检查点 {d}")
        return d
    if not (d / "done.json").exists():
        sys.exit(f"[stale] {model} 没有训练完（缺 {d / 'done.json'}）")
    return d


def checkpoints():
    out = []
    for base in C.CFG.get("CKPT_RUNS", ["S_s11", "A_s11"]):
        d = STATE["rt_out"] / "adapters" / base
        for c in sorted(d.glob("ckpt_step*")):
            if c.is_dir():
                out.append(f"{base}@{int(c.name.replace('ckpt_step', ''))}")
    return out


def drift_models():
    return ["B"] + FINALS + checkpoints()


def adapter_sig(model):
    d = adapter_path(model)
    if d is None:
        return "base"
    return fp({f.name: file_sha256(f) for f in sorted(d.iterdir()) if f.is_file() and f.name != "train_log.json"})


def code_sig():
    from alfw import acfg
    s = {"stale_alfw.py": file_sha256(Path(__file__))[:16],
         "acfg": [acfg.MAX_STEPS, acfg.TEMPERATURE, acfg.TOP_P, acfg.MAX_NEW_TOKENS]}
    for f in ("agent.py", "rollout.py"):
        s[f] = file_sha256(RT_CODE / "alfw" / f)[:16] if not STATE["mock"] else "mock"
    return s


# ---------------------------------------------------------------- 各阶段的续跑清单
def specs_free(model, pos):
    return [{"rid": f"{model}|{p['pos_id']}|free#{i}", "model": model, "pos_id": p["pos_id"], "branch": "free", "i": i,
             "gamefile": p["gamefile"], "prefix": p["free_prefix"], "seed": derive_seed("STALE", p["pos_id"] + "|free", i)}
            for p in pos for i in range(K)]


def specs_branch(model, branches, k=K, tag="STALE"):
    D = DESIGN[0]
    out = []
    for b in branches:
        x = D["branches"][b]
        for i in range(k):
            out.append({"rid": f"{model}|{b}#{i}", "model": model, "pos_id": f"{x['traj_id']}|t{x['t']}", "branch": b,
                        "i": i, "gamefile": x["gamefile"], "prefix": list(x["prefix"]), "seed": derive_seed(tag, b, i)})
    return out


def plan_full(model, pos):
    return specs_branch(model, [b for p in pos for b in [p["orig"]] + p["alts"]]) + specs_free(model, pos)


def plan_two(model, pos, ev):
    best = [best_alt(p, {b: sum(ev[b]) / len(ev[b]) for b in p["alts"]}) for p in pos]
    return specs_branch(model, best) + specs_free(model, pos)


def plan_reverse(model, pos, ev):
    rows = [r for r in jl_read(P("full.jsonl")) if r["model"] == model]
    need = len(plan_full(model, pos))
    if len(rows) < need:
        sys.exit(f"[reverse] {model} 的 full 续跑还没跑完（{len(rows)}/{need}）")
    res = {}
    for r in rows:
        res.setdefault(r["branch"], {})[r["i"]] = r["won"]
    dis = []
    for p in pos:
        b0 = best_alt(p, {b: sum(ev[b]) / len(ev[b]) for b in p["alts"]})
        b1 = best_alt(p, {b: sum(res[b].values()) / len(res[b]) for b in p["alts"]})
        if b0 != b1:
            dis.append((derive_seed("STALE-REV", p["pos_id"]), p["pos_id"], b0, b1))
    dis.sort(reverse=True)
    dis = dis[:N_REV]
    jdump(P(f"reverse_positions_{model}.json"), [{"pos_id": d[1], "b_pi0": d[2], "b_pi1": d[3]} for d in dis])
    log(f"[reverse] {model}：最好修复不同的位置 {len(dis)} 个（至多取 {N_REV}）")
    return specs_branch(model, [b for d in dis for b in (d[2], d[3])], k=K_REV, tag="STALE-REV")


def specs_drift(model, pos):
    # 漂移用各模型独立的种子（不用共同随机数）：同一种子会让两个相近的分布抽出同一个动作，TV 被系统性压低，
    # 也就没法和"B 与 B 自己的另一组 20 个样本"（B#2，噪声底）比较。B 用 0–19，B#2 用 20–39。
    tag, base = ("B", M_DRIFT) if model == "B#2" else (model, 0)
    return [{"rid": f"{model}|{p['pos_id']}|drift#{i}", "model": model, "pos_id": p["pos_id"], "i": i,
             "gamefile": p["gamefile"], "prefix": p["free_prefix"],
             "seed": derive_seed("DRIFT", p["pos_id"], tag, base + i)} for p in pos for i in range(M_DRIFT)]


# ---------------------------------------------------------------- 运行
def run_drift(backend, pool, model, pos):
    """每个 spec 只走一步（含不可执行时至多 2 次重答，与部署协议相同），记下第一步的命令与种类。"""
    from alfw import acfg, rollout
    path = P("drift.jsonl")
    sig = fp({"stage": "drift", "model": model, "adapter": adapter_sig(model if model != "B#2" else "B"),
              "code": code_sig()})
    have = set()
    for r in jl_read(path):
        if r["model"] == model:
            if r["sig"] != sig:
                sys.exit(f"[drift] drift.jsonl 里 {model} 的签名不同（代码或 adapter 改过）；换输出目录")
            have.add(r["rid"])
    todo = [s for s in specs_drift(model, pos) if s["rid"] not in have]
    log(f"[drift {model}] 共 {len(pos) * M_DRIFT}，已完成 {len(pos) * M_DRIFT - len(todo)}，本次 {len(todo)}")
    if not todo:
        return 0
    t0 = time.time()
    C.repair_tail(path)
    with open(path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), acfg.CHUNK):
            chunk = todo[i:i + acfg.CHUNK]
            opened = pool.open([{"rid": s["rid"], "gamefile": s["gamefile"], "prefix": s["prefix"], "want_key": False}
                                for s in chunk])
            eps = {s["rid"]: rollout._new_episode(s, opened[s["rid"]], False) for s in chunk}
            alive = [s["rid"] for s in chunk if not eps[s["rid"]]["won"]]
            if alive:
                rollout._wave(backend, pool, eps, alive, acfg.TEMPERATURE, acfg.MAX_STEPS, acfg.MAX_NEW_TOKENS,
                              False, False, rollout._new_timing())
            pool.close([s["rid"] for s in chunk])
            for s in chunk:
                e = eps[s["rid"]]
                st = e["steps"][-1] if len(e["steps"]) > e["prefix_len"] else None
                row = {k: v for k, v in s.items() if k != "prefix"}
                row.update(sig=sig, cmd=st["cmd"] if st else None, kind=st["kind"] if st else "already_won",
                           first_kind=st.get("first_kind") if st else None)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    C.add_timing(f"drift_{model}", time.time() - t0, {"gpu_stage": True, "n": len(todo)})
    return 0


def run_rollouts(backend_fn, pool, stage, model, specs):
    path = P(f"{stage}.jsonl")
    sig = fp({"stage": stage, "model": model, "adapter": adapter_sig(model), "code": code_sig(), "n": len(specs)})
    t0 = time.time()
    n, _ = C.run_specs(backend_fn, pool, specs, path, sig, label=f"{stage} {model}")
    C.add_timing(f"{stage}_{model}", time.time() - t0, {"gpu_stage": True, "rollouts": n})
    return n


def stage_run(stages, models_arg):
    D, ev, pos = load_design()
    DESIGN[0] = D
    holder = {}

    def backend():
        if "b" not in holder:
            holder["b"] = C.make_backend(lora=True)
        return holder["b"]

    def use(model):
        backend().set_lora(model, adapter_path("B" if model == "B#2" else model))
        return backend()

    t_all = time.time()
    try:
        with C.make_pool() as pl:
            for st in stages:
                if st == "drift":
                    for m in (models_arg or drift_models()) + ["B#2"]:
                        run_drift(use(m), pl, m, pos)
                elif st == "free_b":
                    use("B")
                    run_rollouts(backend, pl, "free_b", "B", specs_free("B", pos))
                elif st == "full":
                    for m in models_arg or FULL_DEFAULT:
                        use(m)
                        run_rollouts(backend, pl, "full", m, plan_full(m, pos))
                elif st == "reverse":
                    for m in models_arg or REV_DEFAULT:
                        sp = plan_reverse(m, pos, ev)
                        use(m)
                        run_rollouts(backend, pl, "reverse", m, sp)
                elif st == "two":
                    for m in models_arg or TWO_DEFAULT:
                        use(m)
                        run_rollouts(backend, pl, "two", m, plan_two(m, pos, ev))
                else:
                    sys.exit(f"未知阶段 {st}")
                log(f"===== 阶段 {st} 完成 =====")
    finally:
        C.release_vllm(holder.get("b"))
    log(f"[run] 全部完成：{(time.time() - t_all) / 3600:.2f} 小时")
    return 0


def stage_check():
    D, ev, pos = load_design()
    DESIGN[0] = D
    ok = True
    n_br = sum(1 + len(p["alts"]) for p in pos)
    trajs = {p["traj_id"] for p in pos}
    log(f"[check] 6.9 池：{len(trajs)} 条轨迹、{len(pos)} 个位置、{n_br} 个分支；π0 续跑覆盖 {len(ev)} 个分支")
    for b in D["branches"].values():
        if len(b["prefix"]) != b["t"] + 1:
            log(f"[check] 分支 {b['branch_id']} 的前缀长度不是 t + 1")
            ok = False
    miss = [b for p in pos for b in [p["orig"]] + p["alts"] if b not in ev or len(ev[b]) != K]
    if miss:
        log(f"[check] {len(miss)} 个分支缺 π0 的 10 次续跑，例如 {miss[:3]}")
        ok = False
    if not STATE["mock"] and (len(trajs), len(pos), n_br) != (60, 180, 1286):
        log("[check] 数量与登记（60 / 180 / 1,286）不符")
        ok = False
    if os.environ.get("PYTHONHASHSEED") != "0":
        log("[check] PYTHONHASHSEED 不是 0（ALFWorld 的观测顺序依赖它；用 run_stale.sh 运行会自动设置）")
        ok = False
    if STATE["rt_out"] is None:
        log("[check] 找不到 P0 的输出目录（设环境变量 RT_OUT）")
        return 1
    log(f"[check] P0 输出目录：{STATE['rt_out']}")
    for m in drift_models():
        try:
            d = adapter_path(m)
        except SystemExit as e:
            log(str(e))
            ok = False
            continue
        if d is not None and not STATE["mock"] and not (d / "adapter_config.json").exists():
            log(f"[check] {m}：{d} 里没有 adapter_config.json")
            ok = False
        else:
            log(f"[check] {m}：{d if d else '基础模型'}")
    if len(checkpoints()) != 6:
        log(f"[check] 检查点数 {len(checkpoints())}（登记 5.15 为 6 个）：{checkpoints()}")
    nf = len(plan_full("x", pos))
    nt = len(plan_two("x", pos, ev))
    log(f"[check] 续跑数：drift {len(drift_models()) + 1} 个模型 × {len(pos) * M_DRIFT} 个单步；free_b {len(pos) * K}；"
        f"full 每个模型 {nf}；two 每个模型 {nt}；reverse 至多 {N_REV * 2 * K_REV}")
    log(f"[check] {'通过' if ok else '有问题，见上'}")
    return 0 if ok else 1


# ---------------------------------------------------------------- 模拟数据（只用于工作区自检）
def mocksetup(out):
    """模拟版 6.9 池：mock 游戏上 B（技能 0.55）收集失败 → 每条 3 个位置 → 原动作 + 规划器 + 至多 4 个随机可执行动作；
    B 每分支 10 次续跑写成 evals 格式；模拟 adapter（技能不同）、检查点与 P0 的接管结果。"""
    from alfw import acfg, rollout
    md, ro = STATE["data"], STATE["rt_out"]
    md.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    be = C.make_backend(lora=True)
    be.set_lora("B", None)
    with C.make_pool() as pl:
        games = [f"mock://{i}" for i in range(60)]
        eps, _ = rollout.run_episodes(be, pl, [{"rid": g, "gamefile": g, "seed": derive_seed("MOCKC", g)} for g in games],
                                      acfg.TEMPERATURE, want_key=False)
        fails = [e for e in eps if not e["won"]][:12]
        trajs, branches = [], {}
        for k, e in enumerate(fails):
            tid = f"n{k:03d}"
            cmds = [s["cmd"] for s in e["steps"]]
            ts = sorted(rng.sample(range(2, len(cmds) - 6), 3))
            P_ = []
            for t in ts:
                rid = f"{tid}|{t}"
                o = pl.open([{"rid": rid, "gamefile": e["gamefile"], "prefix": cmds[:t], "want_key": False}])[rid]
                adm = o["state"]["admissible"]
                plan = pl.plan([rid])[rid]["plan"] or []
                pl.close([rid])
                planner = plan[0] if plan else None
                alts = []
                if planner and planner != cmds[t]:
                    alts.append((planner, True))
                for a in rng.sample(adm, min(len(adm), 6)):
                    if a != cmds[t] and a not in [x[0] for x in alts] and len(alts) < 5:
                        alts.append((a, False))
                ob = f"{tid}|t{t}|orig"
                branches[ob] = {"branch_id": ob, "traj_id": tid, "t": t, "action": "orig:" + cmds[t],
                                "gamefile": e["gamefile"], "prefix": cmds[:t + 1], "planner": None, "expert": False}
                ids = []
                for a, pf in alts:
                    bid = f"{tid}|t{t}|{a}"
                    branches[bid] = {"branch_id": bid, "traj_id": tid, "t": t, "action": a, "gamefile": e["gamefile"],
                                     "prefix": cmds[:t] + [a], "planner": pf, "expert": pf}
                    ids.append(bid)
                P_.append({"t": t, "orig": ob, "alts": ids})
            trajs.append({"traj_id": tid, "gamefile": e["gamefile"], "positions": P_})
        jdump(md / "design_69.json", {"trajectories": trajs, "branches": branches})
        DESIGN[0] = jload(md / "design_69.json")
        sp = specs_branch("B0", list(branches))
        tmp = md / "_evals_raw.jsonl"
        C.run_specs(lambda: be, pl, sp, tmp, "mock", label="mock evals")
        rows = {}
        for r in jl_read(tmp):
            rows.setdefault(r["branch"], {})[r["i"]] = r["won"]
        with open(md / "evals_q7_69.jsonl", "w") as f:
            for b, d in rows.items():
                f.write(json.dumps({"condition": "q7_T07", "branch_id": b, "n": K,
                                    "results": [d[i] for i in range(K)]}) + "\n")
    skill = {"S_s11": 0.9, "S_s12": 0.8, "S_s13": 0.8, "U_s11": 0.85, "U_s12": 0.8, "R_s11": 0.7, "R_s12": 0.65,
             "DPO-S_s11": 0.75, "S-plan_s11": 0.85, "S-plan_s12": 0.8, "A_s11": 0.57, "A_s12": 0.56}
    for m, s in skill.items():
        d = ro / "adapters" / m
        d.mkdir(parents=True, exist_ok=True)
        jdump(d / "mock_adapter.json", {"skill": s})
        jdump(d / "done.json", {"mock": True})
    for base, steps, s in (("S_s11", (35, 70, 105), (0.65, 0.75, 0.85)), ("A_s11", (18, 35, 52), (0.55, 0.56, 0.57))):
        for st, sk in zip(steps, s):
            d = ro / "adapters" / base / f"ckpt_step{st:04d}"
            d.mkdir(parents=True, exist_ok=True)
            jdump(d / "mock_adapter.json", {"skill": sk})
    with open(ro / "eval_rollouts.jsonl", "w") as f:
        for m, s in [("B", 0.55)] + list(skill.items()):
            for j in range(300):
                f.write(json.dumps({"rid": f"{m}|tk|p{j}#0", "model": m, "won": rng.random() < (s - 0.5)}) + "\n")
    log(f"[mocksetup] {len(trajs)} 条轨迹、{sum(len(t['positions']) for t in trajs)} 个位置、{len(branches)} 个分支")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run", "mocksetup"])
    ap.add_argument("--stages", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--profile", default="real", choices=["real", "mock"])
    ap.add_argument("--out", default=os.environ.get("STALE_OUT", "/root/autodl-tmp/stale_runs/out_stale"))
    a = ap.parse_args()
    setup(a.profile, a.out)
    if a.cmd == "mocksetup":
        return mocksetup(a.out)
    if a.cmd == "check":
        return stage_check()
    if stage_check() != 0 and a.profile != "mock":
        sys.exit("[run] check 没通过，不运行")
    return stage_run([s for s in a.stages.split(",") if s], [m for m in a.models.split(",") if m])


if __name__ == "__main__":
    sys.exit(main())
