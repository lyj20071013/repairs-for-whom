"""τ²-bench 先导 Q5：从失败对话的中间位置分叉续跑（2026-10-29 写；探索性，只检验流程与方差）。

对每条失败对话（奖励 0、不是基础设施错误），在部署者的回合中按早 / 中 / 晚三段各取 1 个位置 t，
前缀 = 对话到第 t 条部署者消息为止（强制保留原动作；若它是工具调用，连同其工具结果一起放进前缀），
然后用同一部署者与同一用户模拟器续跑 K 次（每次不同种子），记录奖励。
这就是第 4 篇里"原动作分支"的续跑；修复候选的分支以后用同一函数，只是把第 t 条消息换掉。

恢复方式：把前缀写进任务的 initial_state.message_history，τ² 的 Orchestrator 用 set_state 重放工具调用恢复数据库，
用户模拟器与部署者都从这段历史接着说。任务 id 不变，所以语言包分配的用户 persona 与原对话相同（已在工作区核对）。
剩余步数 = 100 − 前缀消息数（原对话的步数上限是 100），与第 4 篇"剩余预算"的做法一致。

用法（tau 环境，在 tau2-bench 目录下运行；需要 8001 / 8002 两个服务）：
  /root/autodl-tmp/envs/tau/bin/python ../branch_tau2.py --src data/simulations/p_4b_air_local/results.json \
      --domain airline --dep 4B --n-traj 10 --k 5 --out ../logs/branch_p_4b_air.jsonl
  加 --dry-run：只构造前缀并初始化（检查恢复），不调用模型。
输出：每个续跑一行 JSON（源任务、位置、续跑序号、种子、奖励、终止原因、消息数、耗时）。已有的行会跳过（可断点续跑）。
"""
import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import Results, TextRunConfig
from tau2.data_model.tasks import InitialState
from tau2.runner.batch import run_single_task
from tau2.runner.build import build_text_orchestrator

MAX_STEPS = 100
PORT = {"4B": 8001, "9B": 8002}
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def sseed(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def positions(sim, rng):
    """部署者消息的下标（去掉开场白、最后一条），按早 / 中 / 晚三段各取 1 个。"""
    msgs = sim.messages
    idx = [i for i, m in enumerate(msgs) if m.role == "assistant" and i > 0 and i < len(msgs) - 1]
    if len(idx) < 3:
        return idx
    thirds = [idx[: len(idx) // 3], idx[len(idx) // 3: 2 * len(idx) // 3], idx[2 * len(idx) // 3:]]
    return [rng.choice(t) for t in thirds if t]


def prefix_at(sim, i):
    j = i + 1
    while j < len(sim.messages) and sim.messages[j].role == "tool":   # 工具调用连同其结果
        j += 1
    return [m.model_copy(deep=True) for m in sim.messages[:j]]


def branched_task(task, prefix):
    t = task.model_copy(deep=True)
    ist = task.initial_state
    t.initial_state = InitialState(initialization_data=ist.initialization_data if ist else None,
                                   initialization_actions=ist.initialization_actions if ist else None,
                                   message_history=prefix)
    return t


def make_cfg(domain, dep, max_steps):
    return TextRunConfig(
        domain=domain, seed=300, max_steps=max_steps,
        llm_agent=f"hosted_vllm/Qwen3.5-{dep}",
        llm_args_agent={"api_base": f"http://127.0.0.1:{PORT[dep]}/v1", "api_key": "EMPTY", "temperature": 0.7,
                        "max_tokens": 2048, "timeout": 180, "extra_body": NOTHINK},
        llm_user="hosted_vllm/Qwen3.5-9B",
        llm_args_user={"api_base": "http://127.0.0.1:8002/v1", "api_key": "EMPTY", "temperature": 0.0,
                       "max_tokens": 2048, "timeout": 180, "extra_body": NOTHINK},
        communicate_scoring=CommunicateScoring(scorer="substring"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--domain", choices=["airline", "telecom"], required=True)
    ap.add_argument("--dep", choices=["4B", "9B"], required=True)
    ap.add_argument("--n-traj", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    R = Results.load(Path(a.src))
    domain = a.domain
    tasks = {t.id: t for t in R.tasks}
    fails = [s for s in R.simulations if s.reward_info is not None and s.reward_info.reward == 0.0
             and "infra" not in str(s.termination_reason).lower() and len(s.messages) > 0]
    fails.sort(key=lambda s: (s.task_id, s.trial))
    rng = random.Random(2027)
    rng.shuffle(fails)
    fails = fails[: a.n_traj]
    print(f"领域 {domain}；源文件中失败对话 {len(fails)} 条（取前 {a.n_traj} 条，随机顺序种子 2027）")

    jobs = []
    for s in fails:
        prng = random.Random(sseed("POS", s.task_id, s.trial))
        for i in positions(s, prng):
            pre = prefix_at(s, i)
            for k in range(a.k):
                jobs.append(dict(task_id=s.task_id, trial=s.trial, src_sim=s.id, pos=i, n_prefix=len(pre),
                                 k=k, seed=sseed("BR", s.task_id, s.trial, i, k), prefix=pre))
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["task_id"], r["trial"], r["pos"], r["k"]))
    todo = [j for j in jobs if (j["task_id"], j["trial"], j["pos"], j["k"]) not in done]
    print(f"续跑共 {len(jobs)} 次（{len(jobs) // max(a.k, 1)} 个位置 × {a.k}）；已完成 {len(jobs) - len(todo)}，待跑 {len(todo)}")

    if a.dry_run:
        ok = 0
        for j in {(x["task_id"], x["trial"], x["pos"]): x for x in jobs}.values():
            cfg = make_cfg(domain, a.dep, max(10, MAX_STEPS - j["n_prefix"]))
            orch = build_text_orchestrator(cfg, branched_task(tasks[j["task_id"]], j["prefix"]), seed=j["seed"])
            orch.initialize()
            ok += 1
            print(f"  任务 {j['task_id'][:40]} 试次 {j['trial']} 位置 {j['pos']}：前缀 {j['n_prefix']} 条，"
                  f"下一个发言 {orch.to_role}，DB {orch.environment.get_db_hash()[:10]}")
        print(f"dry-run：{ok} 个位置都恢复成功，没有调用模型")
        return 0

    def run(j):
        cfg = make_cfg(domain, a.dep, max(10, MAX_STEPS - j["n_prefix"]))
        t0 = time.time()
        for attempt in range(3):
            try:
                sim = run_single_task(cfg, branched_task(tasks[j["task_id"]], j["prefix"]), seed=j["seed"])
                rew = sim.reward_info.reward if sim.reward_info is not None else None
                return dict({k: v for k, v in j.items() if k != "prefix"}, reward=rew,
                            termination=str(sim.termination_reason), n_msgs=len(sim.messages),
                            secs=round(time.time() - t0, 1), attempts=attempt + 1)
            except Exception as e:                      # 基础设施错误：重试，三次都失败就记下，不当作失败
                err = repr(e)[:300]
        return dict({k: v for k, v in j.items() if k != "prefix"}, reward=None, termination="error: " + err,
                    n_msgs=None, secs=round(time.time() - t0, 1), attempts=3)

    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(out, "a", encoding="utf-8") as f:
        futs = [ex.submit(run, j) for j in todo]
        for fu in as_completed(futs):
            r = fu.result()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0 or n == len(todo):
                print(f"  {time.strftime('%H:%M:%S')} 完成 {n}/{len(todo)}", flush=True)
    print("== 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
