"""τ²-bench 对话的两条事后规则（2026-10-29 写；第 6 篇；只用 CPU，不调用模型）。

1) --cap M：模拟"步数上限 M"。按 τ² 的计步方式（每条部署者 / 用户消息 1 步，一组工具结果 1 步；
   刚发出工具调用、等环境回复时不检查上限）走一遍对话；若在自然结束前步数到了 M，就当作 max_steps 终止、奖励 0。
   步数上限只决定对话在哪里停，停之前的对话不受影响，所以用上限 200 跑出来的对话可以事后得到"上限 100 会怎样"。
   在上限 100 跑出的先导 1 数据上用 --cap 100 应当原样复现原判分（用来核对这里的计步）。
2) --repeat N：用户连续 N 条文本消息（去首尾空白）相同 → 当作用户在第 N 条处结束（USER_STOP），截断后按环境断言 / DB 重判。
   与 tau2_ext.py 的运行时开关等价（触发前的对话完全一样）。
两条都开时：先看哪个先发生。

用法：PYTHONPATH=… python posthoc_tau2.py <simulations 目录> <标签> … [--cap 100] [--repeat 3]
"""
import argparse
import collections
import sys
from pathlib import Path

from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import Results, TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation


def _is_user_text(m):
    return m.role == "user" and not getattr(m, "tool_calls", None)


def cap_index(msgs, cap):
    """返回 τ² 在上限 cap 下会停在哪条消息之后（下标）；不会因上限停下则返回 None。msgs[0] 是开场白，不计步。"""
    step, i = 0, 1
    while i < len(msgs):
        m = msgs[i]
        if m.role == "tool":
            while i + 1 < len(msgs) and msgs[i + 1].role == "tool":
                i += 1
        step += 1
        waiting_env = m.role != "tool" and bool(getattr(m, "tool_calls", None))
        if step >= cap and not waiting_env and i < len(msgs) - 1:
            return i
        i += 1
    return None


def repeat_index(msgs, n):
    last, run = None, 0
    for i, m in enumerate(msgs):
        if not _is_user_text(m):
            continue
        c = (m.content or "").strip()
        run = run + 1 if c and c == last else 1
        last = c
        if run >= n:
            return i
    return None


def regrade(s, task, dom, msgs):
    s2 = s.model_copy(deep=True)
    s2.messages = msgs
    s2.termination_reason = TerminationReason.USER_STOP
    return evaluate_simulation(simulation=s2, task=task, evaluation_type=EvaluationType.ALL, solo_mode=False,
                               domain=dom, communicate_scoring=CommunicateScoring(scorer="substring")).reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=0)
    a = ap.parse_args()
    for tag in a.tags:
        R = Results.load(Path(a.root) / tag / "results.json")
        T = {t.id: t for t in R.tasks}
        dom = "telecom" if "tel" in tag else "airline"
        c = collections.Counter()
        old = new = n = 0
        for s in R.simulations:
            if s.reward_info is None or "infra" in str(s.termination_reason).lower():
                c["跳过（基础设施错误或无判分）"] += 1
                continue
            n += 1
            r0 = s.reward_info.reward
            ms = s.termination_reason == TerminationReason.MAX_STEPS
            ic = cap_index(s.messages, a.cap) if a.cap else None
            ir = repeat_index(s.messages, a.repeat) if a.repeat else None
            if ir is not None and (ic is None or ir <= ic):
                try:
                    r1 = regrade(s, T[s.task_id], dom, s.messages[: ir + 1])
                except Exception as e:
                    c["判分出错:" + type(e).__name__] += 1
                    r1 = r0
                c[f"重复规则触发（原{'max_steps' if ms else '正常结束'}）：原 {r0:.0f} → {r1:.0f}"] += 1
            elif ic is not None:
                r1 = 0.0
                c[f"步数上限触发（原{'max_steps' if ms else '正常结束'}）：原 {r0:.0f} → 0"] += 1
            else:
                r1 = r0
                c[f"未触发（原{'max_steps' if ms else '正常结束'}）"] += 1
            old += r0 == 1.0
            new += r1 == 1.0
        rules = "，".join(x for x in [f"上限 {a.cap}" if a.cap else "", f"重复 {a.repeat}" if a.repeat else ""] if x) or "无"
        print(f"== {tag}（{dom}；规则：{rules}）：成功率 原 {old}/{n} = {old / n:.2f} → {new}/{n} = {new / n:.2f}")
        for k in sorted(c):
            print(f"   {k}: {c[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
