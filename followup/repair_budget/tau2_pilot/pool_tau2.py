"""τ²-bench 上的"第 4 篇式"修复池（2026-10-29 写；第 6 篇；同时可作第 5 篇在 τ² 上的标签来源）。

两步：
  design  ：从部署者（默认 4B）的失败对话里选失败、选位置、生成候选，写设计文件（每个位置一行 JSON）。
            候选 = 原动作 + LLM 提议（提议者在该位置用温度 1.0 采样 --n-sample 次，去重后最多留 --max-llm 个）
                   + 金标下一步（任务 evaluation_criteria.actions 里还没做到的第一个部署者动作）
                   + 随机工具调用（随机选一个部署者工具，参数取前缀里出现过的同名参数值；负对照）。
            同一个动作被几种来源同时提出时只留一个候选，kinds 里记下所有来源。
            每个候选都会真的构造分叉前缀并初始化一次环境（不调用模型），检查能恢复。
  rollout ：对设计文件里每个候选、每个执行者（默认 4B 与 9B）、各续跑 k 次，写结果（每次续跑一行 JSON；可断点续跑）。
            10-31 起设计文件每个位置另有一行自由选择（cand_id = "free"，msg = null：执行者从前缀自己接着说）。

分叉前缀 = 原对话第 t 条部署者消息之前的全部消息 + 候选消息；候选若是工具调用，先在由前缀恢复的环境里执行它，
把工具结果也放进前缀（τ² 的 set_state 要求工具调用后面紧跟结果）。原动作也走同一条路（重新执行，结果应与原对话相同，
design 时核对并记在 orig_result_same）。这与 branch_tau2.py 的"原动作分支"等价。
剩余步数 = --max-steps − 前缀消息数（至少 10）。任务 id 不变，所以用户 persona 不变。

失败的选择（design）：奖励 0；去掉基础设施错误；默认去掉 max_steps 终止的对话（--keep-maxsteps 保留）——
先导 1 里 telecom 的 max_steps 有一半按最终状态其实已解决，不能当失败。每个任务最多 --per-task 条（默认 2），
随机顺序种子 2027。位置：部署者消息（去掉开场白与最后一条），按早 / 中 / 晚三段各取 1 个（与 branch_tau2.py 相同）。

服务（serve_tau.sh 起）：4B=8001，9B=8002，Qwen3.8-27B-FP8=8003。用户默认 q27（8003）；--user 9B 用本地 9B。

CPU：tau2 每次续跑光建环境、恢复、判分就要约 2–3 秒 CPU（工作区实测），单进程会被 GIL 卡住，
所以 rollout 用多进程（--procs，每个进程里 --workers/--procs 个线程）。design 每个位置约 10–20 秒 CPU（工作区），
可以开 --workers 让等模型的时间重叠，但 CPU 部分仍是单进程；位置多时可以按源文件分几次跑。

用法（tau 服务器，tau 环境，在 /root/paper/repair_budget/tau2_pilot/tau2-bench 目录下）：
  # 1) 设计（需要提议者的服务；--no-llm 时不调用模型，只有原动作 / 金标 / 随机，用来先检查）
  /root/autodl-tmp/envs/tau/bin/python ../pool_tau2.py design --src data/simulations/<标签>/results.json \
      --domain airline --n-traj 60 --proposer 4B --max-steps 200 --out ../pool/design_air.jsonl
  # 2) 续跑（--dry-run 20 先检查前 20 个续跑的恢复，不调用模型）
  /root/autodl-tmp/envs/tau/bin/python ../pool_tau2.py rollout --design ../pool/design_air.jsonl \
      --executors 4B,9B --k 10 --user q27 --max-steps 200 --procs 8 --workers 48 --out ../pool/rollout_air.jsonl
  # 贪心各 1 次（第 4 篇的"贪心"读数）另写一个文件：--temp 0 --k 1 --out ../pool/rollout_air_greedy.jsonl
10-31 修正（见 先导3-4结果_tau2.md 第 4 节）：τ² 自带的判分把前缀（message_history）里的写操作也放进金标数据库，
retail / airline 的分叉续跑因此有偏。现在续跑后用不带前缀的原任务重判整段对话，作 reward；τ² 原判分记为 reward_hist（对照）。
续跑的新消息另存 <out>.msgs.jsonl.gz。位置默认只取第一次错误写之前（--pos-rule before_bad；firstbad.py；telecom 不受影响）。
旧结果行（没有 score=orig_task）续跑时不算完成。

正常日志：design 每个位置一行"[i/N] 位置 id（任务…，位置 t/总长；提议 8 条、出错 0）：候选 n 个 [来源…]"，最后"恢复失败 0"与"== 完成"；
rollout 每 50 次一行"完成 x/y（出错 e）；已用…，预计还要…"，最后"== 完成"。中断后重跑同一条命令会跳过已完成的。
"""
import argparse
import copy
import os
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")   # 不去 GitHub 拉价格表（国内连不上，会打印一条警告）
import gzip
import hashlib
import multiprocessing as mp
import queue
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def quiet_logs():
    """10-30：只显示 WARNING 及以上，并去掉两类无害的刷屏：
    litellm 价格表里没有本地模型（get_response_cost 的 ERROR 与 get_cost 的"has no cost"警告，只影响"费用"一栏）；
    airline 取消订单时 τ² 自带的"Seats release not implemented"警告（重放前缀时每次都会打印，不影响判分）。"""
    try:
        from loguru import logger
    except Exception:
        return
    noise = ("Seats release not implemented", "isn't mapped yet", "has no cost")

    def keep(r):
        return r["function"] not in ("get_response_cost", "get_cost") and not any(x in r["message"] for x in noise)
    logger.remove()
    logger.add(sys.stderr, level="WARNING", filter=keep)


quiet_logs()

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import Results, TerminationReason, TextRunConfig
from tau2.data_model.tasks import InitialState
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.runner.batch import run_single_task
from tau2.runner.build import build_text_orchestrator

from firstbad import WRITE_TOOLS, eligible_positions

MODELS = {"4B": ("Qwen3.5-4B", 8001), "9B": ("Qwen3.5-9B", 8002), "q27": ("Qwen3.8-27B-FP8", 8003),
          "2B": ("Qwen3.5-2B", 8004)}                # 10-30：2B 部署者（先导 3）
# 10-31：探针里 π1 是带 LoRA 的 vLLM 服务（serve_lora.sh），用 adapter 名当模型名。环境变量 TAU2_EXTRA_MODELS="名字@端口,…"
for _x in filter(None, os.environ.get("TAU2_EXTRA_MODELS", "").split(",")):
    _n, _p = _x.strip().rsplit("@", 1)
    MODELS[_n] = (_n, int(_p))
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
SPLIT_SEED = 2027


def sseed(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def llm_args(name, temp):
    model, port = MODELS[name]
    return f"hosted_vllm/{model}", {"api_base": f"http://127.0.0.1:{port}/v1", "api_key": "EMPTY", "temperature": temp,
                                    "max_tokens": 2048, "timeout": 180, "extra_body": NOTHINK}


def make_cfg(domain, agent, user, max_steps, agent_temp=0.7):
    am, aa = llm_args(agent, agent_temp)
    um, ua = llm_args(user, 0.0)
    return TextRunConfig(domain=domain, seed=300, max_steps=max_steps, llm_agent=am, llm_args_agent=aa,
                         llm_user=um, llm_args_user=ua, communicate_scoring=CommunicateScoring(scorer="substring"))


def with_history(task, msgs):
    t = task.model_copy(deep=True)
    ist = task.initial_state
    t.initial_state = InitialState(initialization_data=ist.initialization_data if ist else None,
                                   initialization_actions=ist.initialization_actions if ist else None,
                                   message_history=[m.model_copy(deep=True) for m in msgs])
    return t


# ---------- 候选的规范形式（用来去重） ----------
def canon(msg):
    if msg.tool_calls:
        return "T:" + ";".join(f"{c.name}({json.dumps(c.arguments, sort_keys=True, ensure_ascii=False)})"
                               for c in msg.tool_calls)
    return "X:" + re.sub(r"\s+", " ", (msg.content or "").strip().lower())


def clean(msg, prefix_id):
    """去掉用量、原始响应、时间戳；工具调用 id 换成确定的值。"""
    m = AssistantMessage(role="assistant", content=msg.content if msg.content and msg.content.strip() else None,
                         tool_calls=[ToolCall(id=f"{prefix_id}_{i}", name=c.name, arguments=c.arguments, requestor="assistant")
                                     for i, c in enumerate(msg.tool_calls)] if msg.tool_calls else None)
    m.validate()
    return m


def tool_call_msg(name, args, cid):
    return AssistantMessage(role="assistant", content=None,
                            tool_calls=[ToolCall(id=cid, name=name, arguments=args, requestor="assistant")])


# ---------- 分叉前缀 ----------
def branch_prefix(domain, task, prefix, cand):
    """返回 (分叉前缀, 工具结果列表, 环境 DB 哈希)。工具调用候选在由前缀恢复的环境里执行一次。"""
    if not cand.tool_calls:
        return prefix + [cand], [], None
    cfg = make_cfg(domain, "4B", "4B", 10)
    orch = build_text_orchestrator(cfg, with_history(task, prefix), seed=0)
    orch.initialize()                                   # 只恢复环境，不调用模型
    results = [orch.environment.get_response(c) for c in cand.tool_calls]
    return prefix + [cand] + results, results, orch.environment.get_db_hash()


def check_restore(domain, task, msgs, max_steps):
    cfg = make_cfg(domain, "4B", "4B", max(10, max_steps - len(msgs)))
    orch = build_text_orchestrator(cfg, with_history(task, msgs), seed=0)
    orch.initialize()
    return str(orch.to_role), orch.environment.get_db_hash()


# ---------- 选失败、选位置 ----------
def select_failures(results_list, n, per_task, keep_maxsteps):
    pool = []
    for src, R in results_list:
        for s in R.simulations:
            if s.reward_info is None or s.reward_info.reward != 0.0 or not s.messages:
                continue
            if "infra" in str(s.termination_reason).lower():
                continue
            if s.termination_reason == TerminationReason.MAX_STEPS and not keep_maxsteps:
                continue
            pool.append((src, s))
    pool.sort(key=lambda x: (x[1].task_id, x[0], x[1].trial))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(pool)
    cnt, out = {}, []
    for src, s in pool:
        if cnt.get(s.task_id, 0) >= per_task:
            continue
        cnt[s.task_id] = cnt.get(s.task_id, 0) + 1
        out.append((src, s))
        if len(out) >= n:
            break
    return out, len(pool)


def cand_positions(sim, task, domain, rule, max_frac=0.8):
    """可选的部署者位置。rule=before_bad（retail / airline 默认）：只取第一次错误写之前（含）的位置，见 firstbad.py；
    rule=any：全部部署者位置（去掉开场白与最后一条）。两种规则都只留 t < max_frac × 总长（10-31 加，默认 0.8，去掉对话末尾）。
    返回 (位置列表, 第一次错误写下标或 None)。"""
    msgs = sim.messages
    if rule == "any" or domain not in WRITE_TOOLS:
        return [i for i, m in enumerate(msgs) if m.role == "assistant" and 0 < i < len(msgs) - 1
                and i < max_frac * len(msgs)], None
    ec = task.evaluation_criteria
    idx, e, _ = eligible_positions(msgs, ec.actions if ec else None, domain, max_frac)
    return idx, e


def positions(idx, rng):
    """在可选位置里按早 / 中 / 晚三段各取 1 个。"""
    if len(idx) < 3:
        return idx
    thirds = [idx[: len(idx) // 3], idx[len(idx) // 3: 2 * len(idx) // 3], idx[2 * len(idx) // 3:]]
    return [rng.choice(t) for t in thirds if t]


# ---------- 候选来源 ----------
def gold_next(task, prefix):
    """evaluation_criteria.actions 里部署者的动作中，前缀里还没有匹配的工具调用的第一个。"""
    ec = task.evaluation_criteria
    if ec is None or not ec.actions:
        return None
    done = [c for m in prefix if m.role == "assistant" and m.tool_calls for c in m.tool_calls]
    used = set()
    for a in ec.actions:
        if a.requestor != "assistant":
            continue
        hit = next((j for j, c in enumerate(done) if j not in used and a.compare_with_tool_call(c)), None)
        if hit is None:
            return a.name, dict(a.arguments)
        used.add(hit)
    return None


def random_call(env_tools, prefix, rng, taken):
    seen = {}
    for m in prefix:
        for c in (getattr(m, "tool_calls", None) or []):
            for k, v in c.arguments.items():
                seen.setdefault(k, [])
                if v not in seen[k]:
                    seen[k].append(v)
    tools = sorted(env_tools, key=lambda t: t.name)
    for _ in range(50):
        t = rng.choice(tools)
        params = t.openai_schema["function"].get("parameters", {}) or {}
        req = params.get("required", []) or []
        if any(not seen.get(p) for p in req):
            continue
        args = {p: rng.choice(seen[p]) for p in req}
        key = "T:" + f"{t.name}({json.dumps(args, sort_keys=True, ensure_ascii=False)})"
        if key not in taken:
            return t.name, args
    return None


def propose(domain, task, prefix, proposer, n_sample, temp, pos_key, user):
    """在前缀末尾让提议者以部署者身份生成 n_sample 次（同一系统提示与工具），返回消息列表与出错次数。"""
    out, errs = [], 0
    cfg = make_cfg(domain, proposer, user, 10, agent_temp=temp)
    orch = build_text_orchestrator(cfg, with_history(task, prefix), seed=0)   # 只建一次；每次采样复制部署者状态、换种子
    orch.initialize()
    assert str(orch.to_role).lower().endswith("agent"), f"前缀之后该说话的不是部署者：{orch.to_role}"
    for j in range(n_sample):
        try:
            orch.agent.llm_args["seed"] = sseed("PROP", pos_key, j)
            msg, _ = orch.agent.generate_next_message(orch.message.model_copy(deep=True), copy.deepcopy(orch.agent_state))
            msg.validate()
            out.append(msg)
        except Exception as e:
            errs += 1
            print(f"    提议出错（{pos_key} 第 {j} 次）：{repr(e)[:200]}", flush=True)
    return out, errs


# ---------- design ----------
def design_position(a, src, s, task, t):
    pos_id = f"{s.id[:12]}_{t}"
    prefix = [m.model_copy(deep=True) for m in s.messages[:t]]
    cands = {}                                   # 规范形式 -> 候选

    def add(msg, kind):
        k = canon(msg)
        if k in cands:
            if kind not in cands[k]["kinds"]:
                cands[k]["kinds"].append(kind)
            return
        cands[k] = {"cand_id": f"c{len(cands)}", "kinds": [kind], "msg": msg}

    orig = clean(s.messages[t], f"{pos_id}_orig")
    add(orig, "orig")
    n_llm_raw = n_llm_err = 0
    if not a.no_llm:
        props, n_llm_err = propose(a.domain, task, prefix, a.proposer, a.n_sample, a.temp, pos_id, a.user)
        n_llm_raw = len(props)
        kept = 0
        for j, p in enumerate(props):
            if kept >= a.max_llm:
                break
            try:
                pm = clean(p, f"{pos_id}_llm{j}")
            except Exception:
                continue
            if canon(pm) not in cands:
                kept += 1
            add(pm, "llm")                       # 与已有候选（含原动作）相同时只在 kinds 里加 "llm"
    g = gold_next(task, prefix)
    if g is not None:
        add(tool_call_msg(g[0], g[1], f"{pos_id}_gold"), "gold")
    if not a.no_random:
        env_tools = build_text_orchestrator(make_cfg(a.domain, "4B", "4B", 10), with_history(task, prefix),
                                            seed=0).environment.get_tools()
        r = random_call(env_tools, prefix, random.Random(sseed("RAND", pos_id)), set(cands))
        if r is not None:
            add(tool_call_msg(r[0], r[1], f"{pos_id}_rand"), "rand")
    rows, n_err = [], 0
    for k, c in cands.items():                   # 每个候选：构造分叉前缀、初始化一次（不调用模型）
        try:
            bp, res, _ = branch_prefix(a.domain, task, prefix, c["msg"])
            role, _ = check_restore(a.domain, task, bp, a.max_steps)
            ok = True
        except Exception as e:
            bp, res, role, ok = None, [], None, False
            n_err += 1
            print(f"    恢复失败 {pos_id} {c['cand_id']}：{repr(e)[:200]}", flush=True)
        row = {"cand_id": c["cand_id"], "kinds": c["kinds"], "canon": k[:300],
               "msg": c["msg"].model_dump(mode="json", exclude_none=True),
               "tool_results": [r.content[:500] for r in res], "tool_error": any(r.error for r in res),
               "n_prefix": len(bp) if bp else None, "next_role": role, "restore_ok": ok}
        if "orig" in c["kinds"] and c["msg"].tool_calls:
            j = t + 1
            old = []
            while j < len(s.messages) and s.messages[j].role == "tool":
                old.append(s.messages[j].content)
                j += 1
            row["orig_result_same"] = old == [r.content for r in res]
        rows.append(row)
    if not a.no_free:                            # 10-31：自由选择（不放候选，执行者自己从前缀接着说）；点价值 G 的基线，不当作修复候选
        try:
            role, _ = check_restore(a.domain, task, prefix, a.max_steps)
            ok = True
        except Exception as ex:
            role, ok = None, False
            n_err += 1
            print(f"    恢复失败 {pos_id} free：{repr(ex)[:200]}", flush=True)
        rows.append({"cand_id": "free", "kinds": ["free"], "canon": "FREE", "msg": None, "tool_results": [],
                     "tool_error": False, "n_prefix": len(prefix), "next_role": role, "restore_ok": ok})
    idx, e = cand_positions(s, task, a.domain, a.pos_rule, a.max_frac)
    return {"pos_id": pos_id, "src": src, "sim_id": s.id, "task_id": s.task_id, "trial": s.trial, "t": t,
            "n_msgs_src": len(s.messages), "domain": a.domain, "deployer": a.deployer,
            "pos_rule": a.pos_rule, "max_frac": a.max_frac, "first_bad": e, "n_eligible": len(idx),
            "proposer": None if a.no_llm else a.proposer, "n_sample": 0 if a.no_llm else a.n_sample,
            "llm_raw": n_llm_raw, "llm_err": n_llm_err, "cands": rows}, n_err


def cmd_design(a):
    results_list = [(src, Results.load(Path(src))) for src in a.src]
    tasks = {t.id: t for _, R in results_list for t in R.tasks}
    if a.task_file:                              # 10-31：只用这些任务的失败（训练 / 评价按任务分开）
        keep = {l.strip() for l in Path(a.task_file).read_text(encoding="utf-8").splitlines() if l.strip()}
        for _, R in results_list:
            R.simulations = [x for x in R.simulations if x.task_id in keep]
        print(f"任务清单 {a.task_file}：{len(keep)} 个任务")
    fails, n_pool = select_failures(results_list, a.n_traj, a.per_task, a.keep_maxsteps)
    print(f"领域 {a.domain}；符合条件的失败 {n_pool} 条，选 {len(fails)} 条（每任务最多 {a.per_task} 条，随机种子 {SPLIT_SEED}）")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["pos_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo, n_none = [], 0
    for src, s in fails:
        prng = random.Random(sseed("POS", s.task_id, s.trial, s.id))
        idx, _ = cand_positions(s, tasks[s.task_id], a.domain, a.pos_rule, a.max_frac)
        n_none += not idx
        for t in positions(idx, prng):
            if f"{s.id[:12]}_{t}" not in done:
                todo.append((src, s, t))
    if a.positions:                              # 11-05（主实验）：只做 main_pool.py select 选定的位置
        want = {r["pos_id"] for r in (json.loads(l) for l in Path(a.positions).read_text(encoding="utf-8").splitlines() if l.strip())
                if r["domain"] == a.domain}
        found = {f"{s.id[:12]}_{t}" for _, s, t in todo} | done
        miss = sorted(want - found)
        if miss:
            print(f"[停下] --positions 里有 {len(miss)} 个位置在这里选不出来（例 {miss[:3]}）：--per-task / --n-traj / --src 与 select 不一致？")
            return 1
        todo = [(src, s, t) for src, s, t in todo if f"{s.id[:12]}_{t}" in want]
        print(f"--positions：只做选定的 {len(want)} 个位置（已有 {len(want & done)}）")
    print(f"位置规则 {a.pos_rule}，只留 t < {a.max_frac} × 总长；没有可选位置的失败 {n_none} 条")
    print(f"位置共 {len(todo) + len(done)} 个；已有 {len(done)}，待做 {len(todo)}；并发 {a.workers}", flush=True)
    n_pos = n_cand = n_err = 0
    stats = {}
    with ThreadPoolExecutor(a.workers) as ex, open(out, "a", encoding="utf-8") as f:
        futs = [ex.submit(design_position, a, src, s, tasks[s.task_id], t) for src, s, t in todo]
        for fu in as_completed(futs):
            rec, e = fu.result()
            n_err += e
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_pos += 1
            n_cand += len(rec["cands"])
            for r in rec["cands"]:
                for kd in r["kinds"]:
                    stats[kd] = stats.get(kd, 0) + 1
            print(f"  {time.strftime('%H:%M:%S')} [{n_pos}/{len(todo)}] {rec['pos_id']}（任务 {rec['task_id'][:40]}，"
                  f"位置 {rec['t']}/{rec['n_msgs_src']}；提议 {rec['llm_raw']} 条、出错 {rec['llm_err']}）："
                  f"候选 {len(rec['cands'])} 个 {[r['kinds'] for r in rec['cands']]}", flush=True)
    print(f"本次新写 {n_pos} 个位置、{n_cand} 个候选；各来源计数 {stats}；恢复失败 {n_err}")
    print("== 完成" if n_err == 0 else "== 有恢复失败，见上")
    return 0 if n_err == 0 else 1


# ---------- rollout ----------
def tasks_by_domain(design, cache):
    """任务表按 (领域, 任务 id) 建。11-07 修：原来按任务 id 建，两个领域的 results.json 放在同一个设计文件里时，
    后读的领域会覆盖先读的同名任务（retail 与 airline 有 31 个任务 id 重名），续跑拿到别的领域的任务（用户指令、初始状态、金标都错）。
    每个结果文件只能属于一个领域；同一 (领域, id) 在两个文件里出现时要求是同一个任务，否则停下。"""
    src_dom = {}
    for d in design:
        if src_dom.setdefault(d["src"], d["domain"]) != d["domain"]:
            raise SystemExit(f"[停下] 结果文件 {d['src']} 在设计文件里同时属于 {src_dom[d['src']]} 和 {d['domain']}")
    out = {}
    for src, R in cache.items():
        for t in R.tasks:
            k = (src_dom[src], t.id)
            if k in out and out[k].model_dump_json() != t.model_dump_json():
                raise SystemExit(f"[停下] {k} 在两个结果文件里是不同的任务")
            out[k] = t
    miss = {(d["domain"], d["task_id"]) for d in design} - set(out)
    if miss:
        raise SystemExit(f"[停下] 设计文件里有任务找不到：{sorted(miss)[:5]}")
    for d in design:                                    # 位置的原对话必须属于这个任务
        s = _CTX["sims"].get(d["sim_id"])
        if s is not None and s.task_id != d["task_id"]:
            raise SystemExit(f"[停下] 位置 {d['pos_id']} 的原对话属于任务 {s.task_id}，设计文件写的是 {d['task_id']}")
    return out


_CTX = {}          # 在 fork 之前填好：sims、tasks、max_steps；子进程共享（写时复制）


def _bp_cached(cache, lock, j):
    key = (j["pos_id"], j["cand_id"])
    with lock:
        if key in cache:
            return cache[key]
    s, task = _CTX["sims"][j["sim_id"]], _CTX["tasks"][(j["domain"], j["task_id"])]
    prefix = [m.model_copy(deep=True) for m in s.messages[: j["t"]]]
    if j["msg"] is None:                     # takeover：不放候选，执行者从这里自己接着说
        bp = prefix
    else:
        bp, _, _ = branch_prefix(j["domain"], task, prefix, AssistantMessage.model_validate(j["msg"]))
    with lock:
        cache[key] = bp
    return bp


def rescore(sim, task, domain):
    """10-31：用不带前缀的原任务重判整段对话（前缀 + 续跑）。
    τ² 自己的判分把 initial_state.message_history 也重放进金标数据库（evaluator_env.py：gold = 初始状态 + 历史 + 金标动作），
    所以前缀里的写操作——错的也好、对的也好——都会进金标：错误写之后的续跑几乎必然判对，前缀里有不可重复的正确写
    （如订票）时几乎必然判错。retail / airline（判分含 DB）受影响；telecom（只看环境断言）不受影响。
    sim.messages 已含前缀（orchestrator 把历史放进 trajectory），所以用原任务重判 = "整段对话是否成功"。"""
    return evaluate_simulation(simulation=sim, task=task, evaluation_type=EvaluationType.ALL, solo_mode=False,
                               domain=domain, communicate_scoring=CommunicateScoring(scorer="substring")).reward


def repair_gz(path):
    """读得出的记录原样保留；读到写了一半的 gzip 块就截断，重写成一个完整文件（原文件留 .bad 备份）。"""
    path = Path(path)
    if not path.exists():
        return
    from gzrobust import read_records            # 11-01：坏块可能在中间（被杀后又追加过），按块头分段读
    recs, ok = read_records(path)
    if ok:
        return
    bak = Path(str(path) + ".bad")
    k = 1
    while bak.exists():
        bak = Path(f"{path}.bad{k}")
        k += 1
    path.replace(bak)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    print(f"[修复] {path} 有写了一半的块：保留 {len(recs)} 条完整记录，原文件改名 {bak.name}", flush=True)


def _run_one(j, cache, lock):
    task = _CTX["tasks"][(j["domain"], j["task_id"])]
    t0 = time.time()
    err = None
    for attempt in range(3):
        try:
            bp = _bp_cached(cache, lock, j)
            cfg = make_cfg(j["domain"], j["executor"], j["user"], max(10, _CTX["max_steps"] - len(bp)), agent_temp=j["temp"])
            sim = run_single_task(cfg, with_history(task, bp), seed=j["seed"])
            if "infra" in str(sim.termination_reason).lower():
                raise RuntimeError(f"基础设施错误：{sim.termination_reason}")
            rew_hist = sim.reward_info.reward if sim.reward_info is not None else None
            rew = rescore(sim, task, j["domain"])      # 主读数：原任务判分
            new = [m.model_dump(mode="json", exclude_none=True) for m in sim.messages[len(bp):]]
            return dict({k: v for k, v in j.items() if k != "msg"}, reward=rew, reward_hist=rew_hist, score="orig_task",
                        termination=str(sim.termination_reason), n_msgs=len(sim.messages), n_prefix=len(bp),
                        secs=round(time.time() - t0, 1), attempts=attempt + 1, _msgs=new)
        except Exception as e:                   # 基础设施错误：重试；三次都失败记 reward=None，不当失败
            err = repr(e)[:300]
    return dict({k: v for k, v in j.items() if k != "msg"}, reward=None, termination="error: " + err,
                n_msgs=None, n_prefix=None, secs=round(time.time() - t0, 1), attempts=3)


def _proc_main(jobs, n_threads, q, repeat_stop):
    from tau2_ext import install_repeat_stop
    install_repeat_stop(repeat_stop)
    quiet_logs()
    cache, lock = {}, threading.Lock()
    with ThreadPoolExecutor(n_threads) as ex:
        for fu in as_completed([ex.submit(_run_one, j, cache, lock) for j in jobs]):
            q.put(fu.result())


def cmd_rollout(a):
    from tau2_ext import install_repeat_stop
    install_repeat_stop(a.repeat_stop)
    design = [json.loads(l) for l in Path(a.design).read_text(encoding="utf-8").splitlines() if l.strip()]
    design.sort(key=lambda d: d["pos_id"])
    if a.limit_pos:
        design = design[: a.limit_pos]
    cache = {}
    for d in design:
        if d["src"] not in cache:
            cache[d["src"]] = Results.load(Path(d["src"]))
    _CTX["sims"] = {s.id: s for R in cache.values() for s in R.simulations}
    _CTX["tasks"] = tasks_by_domain(design, cache)      # 11-07 修：任务按 (领域, 任务 id) 取（retail 与 airline 的任务 id 有 31 个重名）
    _CTX["max_steps"] = a.max_steps
    execs = a.executors.split(",")
    jobs = []
    for d in design:
        for c in d["cands"]:
            if not c["restore_ok"]:
                continue
            for e in execs:
                for k in range(a.k):
                    jobs.append(dict(pos_id=d["pos_id"], cand_id=c["cand_id"], kinds=c["kinds"], executor=e, k=k,
                                     temp=a.temp, user=a.user, max_steps=a.max_steps, repeat_stop=a.repeat_stop,
                                     domain=d["domain"], sim_id=d["sim_id"], task_id=d["task_id"], t=d["t"], msg=c["msg"],
                                     seed=sseed("RO", d["pos_id"], c["cand_id"], e, a.temp, k)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    key = lambda r: (r["pos_id"], r["cand_id"], r["executor"], r["temp"], r["k"])
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("score") != "orig_task":       # 10-31 以前的行用的是 τ² 带前缀的判分（有偏），不算完成
                    continue
                if r.get("reward") is not None or not a.retry_errors:
                    done.add(key(r))
    todo = [j for j in jobs if key(j) not in done]
    n_c = sum(1 for d in design for c in d["cands"] if c["restore_ok"])
    print(f"位置 {len(design)} 个，候选 {n_c} 个；执行者 {execs}，每候选每执行者 {a.k} 次（温度 {a.temp}）；用户 {a.user}；"
          f"步数上限 {a.max_steps}；用户重复终止 {'N=' + str(a.repeat_stop) if a.repeat_stop > 0 else '关'}")
    print(f"续跑共 {len(jobs)} 次；已完成 {len(jobs) - len(todo)}，待跑 {len(todo)}；"
          f"{a.procs} 个进程 × {max(1, a.workers // a.procs)} 个线程", flush=True)
    if a.dry_run:
        c2, l2 = {}, threading.Lock()
        for j in todo[: a.dry_run]:
            bp = _bp_cached(c2, l2, j)
            role, _ = check_restore(j["domain"], _CTX["tasks"][(j["domain"], j["task_id"])], bp, a.max_steps)
            print(f"  {j['pos_id']} {j['cand_id']} {j['kinds']} {j['executor']} k={j['k']}：前缀 {len(bp)} 条，下一个 {role}")
        print(f"dry-run：检查了前 {min(a.dry_run, len(todo))} 个续跑的恢复，没有调用模型")
        return 0
    if not todo:
        print("== 没有要跑的")
        return 0
    # 按候选分片（同一候选的续跑落在同一进程，分叉前缀只算一次）；多进程是为了绕开 GIL（每次续跑建环境约 1 秒 CPU）
    shards = [[] for _ in range(a.procs)]
    for j in todo:
        shards[sseed(j["pos_id"], j["cand_id"]) % a.procs].append(j)
    mpctx = mp.get_context("fork")
    q = mpctx.Queue()
    procs = [mpctx.Process(target=_proc_main, args=(sh, max(1, a.workers // a.procs), q, a.repeat_stop))
             for sh in shards if sh]
    for p in procs:
        p.start()
    n = n_err = 0
    t_start = time.time()
    mpath = Path(str(out) + ".msgs.jsonl.gz")          # 续跑的新消息（不含前缀），供离线重判；gzip 追加写，可断点续跑
    repair_gz(mpath)                                    # 11-01：上次被杀时可能留下写了一半的 gzip 块，先修好，否则后面追加的都读不出来
    with open(out, "a", encoding="utf-8") as f, gzip.open(mpath, "at", encoding="utf-8") as fm:
        while n < len(todo):
            try:
                r = q.get(timeout=30)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    print(f"[停下] 子进程都已退出，但只收到 {n}/{len(todo)} 个结果（退出码 {[p.exitcode for p in procs]}）")
                    break
                continue
            msgs = r.pop("_msgs", None)
            if msgs is not None:
                fm.write(json.dumps({k: r[k] for k in ("pos_id", "cand_id", "executor", "temp", "k", "seed")}
                                    | {"msgs": msgs}, ensure_ascii=False) + "\n")
                fm.flush()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            n_err += r["reward"] is None
            if n % 50 == 0 or n == len(todo):
                el = time.time() - t_start
                print(f"  {time.strftime('%H:%M:%S')} 完成 {n}/{len(todo)}（出错 {n_err}）；"
                      f"已用 {el / 60:.1f} 分钟，预计还要 {el / n * (len(todo) - n) / 60:.1f} 分钟", flush=True)
    for p in procs:
        p.join(timeout=60)
    if n < len(todo):
        print(f"== 没跑完：{n}/{len(todo)}；重跑同一条命令会接着跑")
        return 1
    print("== 完成" if n_err == 0 else f"== 完成，但有 {n_err} 次出错（reward=None；加 --retry-errors 重跑这些）")
    return 0


# ---------- drift（10-31 探针）：各模型在设计文件的位置上采样第一步，看动作分布变了多少 ----------
def _drift_one(a, d, task, model, rep_tag):
    s = _CTX["sims"][d["sim_id"]]
    prefix = [m.model_copy(deep=True) for m in s.messages[: d["t"]]]
    key = f"DRIFT|{d['pos_id']}|{model}|{rep_tag}"
    msgs, errs = propose(d["domain"], task, prefix, model, a.m, a.temp, key, a.user)
    out = []
    for i, m in enumerate(msgs):
        try:
            c = canon(clean(m, f"{d['pos_id']}_d{i}"))
        except Exception:
            c = "INVALID"
        out.append({"pos_id": d["pos_id"], "task_id": d["task_id"], "model": model, "rep": rep_tag, "i": i, "canon": c[:300],
                    "is_tool": c.startswith("T:")})
    return out, errs


def cmd_drift(a):
    """对设计文件里每个位置、每个模型，用部署温度采样 --m 个第一步（不续跑）。--models 里写 "4B#2" 表示同一模型再独立抽一次（噪声底）。"""
    design = [json.loads(l) for l in Path(a.design).read_text(encoding="utf-8").splitlines() if l.strip()]
    cache = {src: Results.load(Path(src)) for src in sorted({d["src"] for d in design})}   # 11-07 修：原写法对每一行都重读一次结果文件（800 行 × 上百 MB）
    _CTX["sims"] = {s.id: s for R in cache.values() for s in R.simulations}
    tasks = tasks_by_domain(design, cache)                # 11-07 修：同上
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {(r["pos_id"], r["model"], r["rep"]) for r in (json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()
                                                             if l.strip())}
    jobs = []
    for spec in a.models.split(","):
        model, rep_tag = (spec.split("#") + ["1"])[:2]
        for d in design:
            if (d["pos_id"], model, rep_tag) not in done:
                jobs.append((d, model, rep_tag))
    print(f"位置 {len(design)} 个；模型 {a.models}；每位置 {a.m} 个样本（温度 {a.temp}）；待做 {len(jobs)} 组", flush=True)
    n = n_err = 0
    with ThreadPoolExecutor(a.workers) as ex, open(out, "a", encoding="utf-8") as f:
        futs = [ex.submit(_drift_one, a, d, tasks[(d["domain"], d["task_id"])], m, r) for d, m, r in jobs]
        for fu in as_completed(futs):
            rows, e = fu.result()
            n_err += e
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0 or n == len(jobs):
                print(f"  {time.strftime('%H:%M:%S')} 完成 {n}/{len(jobs)} 组（生成出错 {n_err}）", flush=True)
    print("== 完成" if n_err == 0 else f"== 完成，生成出错 {n_err} 次（缺的样本不补，分析时按实际样本数算）")
    return 0


# ---------- takeover（10-30 先导 3）：从别的部署者的失败状态出发，让执行者自己接着做 ----------
def cmd_takeover(a):
    """从 --src 里的失败对话（默认 2B 的）中选至多 --n-pos 条，每条随机取 1 个部署者位置 t；
    前缀 = 第 t 条部署者消息之前的全部消息（不含 2B 在 t 的动作），执行者（--executors）从这里自己接着说，各续跑 --k 次。
    写一个只有"free"候选的设计文件（<out>.design.jsonl），然后走与 rollout 相同的多进程续跑。"""
    results_list = [(src, Results.load(Path(src))) for src in a.src]
    fails, n_pool = select_failures(results_list, 10 ** 9, a.per_task, False)   # 全部按随机顺序排好，再依次取到 --n-pos 个位置
    tasks = {t.id: t for _, R in results_list for t in R.tasks}
    dpath = Path(str(a.out) + ".design.jsonl")
    dpath.parent.mkdir(parents=True, exist_ok=True)
    rows, n_skip = [], 0
    for src, s in fails:
        if len(rows) >= a.n_pos:
            break
        idx, e = cand_positions(s, tasks[s.task_id], a.domain, a.pos_rule, a.max_frac)
        if not idx:
            n_skip += 1
            continue
        t = random.Random(sseed("TK", s.id)).choice(idx)
        role, _ = check_restore(a.domain, tasks[s.task_id], s.messages[:t], a.max_steps)
        rows.append({"pos_id": f"{s.id[:12]}_{t}", "src": src, "sim_id": s.id, "task_id": s.task_id, "trial": s.trial,
                     "t": t, "n_msgs_src": len(s.messages), "domain": a.domain, "deployer": a.deployer,
                     "pos_rule": a.pos_rule, "max_frac": a.max_frac, "first_bad": e, "n_eligible": len(idx),
                     "cands": [{"cand_id": "free", "kinds": ["free"], "msg": None, "restore_ok": True, "next_role": role}]})
    print(f"源对话中符合条件的失败 {n_pool} 条（每任务最多 {a.per_task} 条，随机种子 {SPLIT_SEED}）；位置规则 {a.pos_rule}，只留 t < {a.max_frac} × 总长；"
          f"跳过没有可选位置的 {n_skip} 条")
    dpath.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"位置 {len(rows)} 个（每条失败 1 个，前缀恢复都检查过），设计写在 {dpath}")
    a.design, a.limit_pos = str(dpath), 0
    return cmd_rollout(a)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design")
    d.add_argument("--src", nargs="+", required=True, help="一个或多个 results.json（部署者的失败对话来源）")
    d.add_argument("--domain", choices=["airline", "telecom", "retail"], required=True)
    d.add_argument("--deployer", default="4B", help="只作记录：src 里的部署者")
    d.add_argument("--n-traj", type=int, default=60)
    d.add_argument("--per-task", type=int, default=2)
    d.add_argument("--keep-maxsteps", action="store_true")
    d.add_argument("--proposer", choices=list(MODELS), default="4B")
    d.add_argument("--user", choices=["9B", "q27"], default="q27", help="提议时只初始化，不会调用用户")
    d.add_argument("--n-sample", type=int, default=8)
    d.add_argument("--temp", type=float, default=1.0)
    d.add_argument("--max-llm", type=int, default=3)
    d.add_argument("--no-llm", action="store_true", help="不调用模型：只有原动作、金标、随机")
    d.add_argument("--no-random", action="store_true")
    d.add_argument("--no-free", action="store_true", help="不加自由选择分支（10-31 起默认加）")
    d.add_argument("--task-file", default=None, help="只用这些任务 id 的失败（每行一个）")
    d.add_argument("--max-steps", type=int, default=100)
    d.add_argument("--workers", type=int, default=8, help="同时处理的位置数（提议要等模型）")
    d.add_argument("--pos-rule", choices=["before_bad", "any"], default="before_bad",
                   help="before_bad：只取第一次错误写之前（含）的位置（retail / airline；telecom 自动用 any）")
    d.add_argument("--max-frac", type=float, default=0.8, help="只留 t < max_frac × 总长（去掉对话末尾）；1.0 = 不去掉")
    d.add_argument("--positions", default=None, help="11-05：main_pool.py select 写的 positions.jsonl；只做其中本领域的位置")
    d.add_argument("--out", required=True)
    r = sub.add_parser("rollout")
    r.add_argument("--design", required=True)
    r.add_argument("--executors", default="4B,9B")
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--temp", type=float, default=0.7)
    r.add_argument("--user", choices=["9B", "q27"], default="q27")
    r.add_argument("--max-steps", type=int, default=100)
    r.add_argument("--repeat-stop", type=int, default=0, help="用户连续 N 次同一句就当用户结束；0=关（默认）")
    r.add_argument("--workers", type=int, default=48, help="总并发（同时在跑的续跑数）")
    r.add_argument("--procs", type=int, default=4, help="进程数；并发平均分到各进程")
    r.add_argument("--limit-pos", type=int, default=0, help="只跑前 N 个位置（小试）")
    r.add_argument("--retry-errors", action="store_true")
    r.add_argument("--dry-run", type=int, default=0, help="只检查前 N 个续跑的恢复，不调用模型")
    r.add_argument("--out", required=True)
    t = sub.add_parser("takeover")
    t.add_argument("--src", nargs="+", required=True)
    t.add_argument("--domain", choices=["airline", "telecom", "retail"], required=True)
    t.add_argument("--deployer", default="2B", help="只作记录：src 里的部署者")
    t.add_argument("--n-pos", type=int, default=20)
    t.add_argument("--pos-rule", choices=["before_bad", "any"], default="before_bad")
    t.add_argument("--max-frac", type=float, default=0.8, help="只留 t < max_frac × 总长；1.0 = 不去掉末尾（先导 5 的设定）")
    t.add_argument("--per-task", type=int, default=2)
    t.add_argument("--executors", default="4B")
    t.add_argument("--k", type=int, default=4)
    t.add_argument("--temp", type=float, default=0.7)
    t.add_argument("--user", choices=["9B", "q27"], default="q27")
    t.add_argument("--max-steps", type=int, default=200)
    t.add_argument("--repeat-stop", type=int, default=0)
    t.add_argument("--workers", type=int, default=16)
    t.add_argument("--procs", type=int, default=4)
    t.add_argument("--retry-errors", action="store_true")
    t.add_argument("--dry-run", type=int, default=0)
    t.add_argument("--out", required=True)
    dr = sub.add_parser("drift")
    dr.add_argument("--design", required=True)
    dr.add_argument("--models", required=True, help="逗号分隔；同一模型的第二次独立抽样写 4B#2")
    dr.add_argument("--m", type=int, default=10)
    dr.add_argument("--temp", type=float, default=0.7)
    dr.add_argument("--user", choices=["9B", "q27"], default="q27", help="只初始化，不调用用户")
    dr.add_argument("--workers", type=int, default=8)
    dr.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"design": cmd_design, "rollout": cmd_rollout, "takeover": cmd_takeover, "drift": cmd_drift}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
