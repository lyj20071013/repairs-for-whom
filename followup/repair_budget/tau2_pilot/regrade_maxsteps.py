# 2026-10-29 13:55 北京时间写；用法：PYTHONPATH=… python regrade_maxsteps.py p_4b_tel_local …（只用环境断言 / DB 重新判分，不调用模型）
"""把 max_steps 终止的对话当作正常结束重新判分（只用环境断言 / DB，不调用模型），看有多少其实已经解决。"""
import sys, json, collections
from pathlib import Path
from tau2.data_model.simulation import Results, TerminationReason
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
import inspect

root = Path("/mnt/user-data/uploads/repair_budget/tau2_pilot/tau2-bench/data/simulations")
for tag in sys.argv[1:]:
    R = Results.load(root / tag / "results.json")
    T = {t.id: t for t in R.tasks}
    dom = "telecom" if "tel" in tag else "airline"
    c = collections.Counter()
    for s in R.simulations:
        if s.termination_reason != TerminationReason.MAX_STEPS:
            continue
        s2 = s.model_copy(deep=True)
        s2.termination_reason = TerminationReason.USER_STOP
        try:
            ri = evaluate_simulation(simulation=s2, task=T[s.task_id], evaluation_type=EvaluationType.ALL, solo_mode=False,
                                     domain=dom, communicate_scoring=CommunicateScoring(scorer="substring"))
            c["解决" if ri.reward == 1.0 else "未解决"] += 1
        except Exception as e:
            c["判分出错:" + type(e).__name__] += 1
    print(tag, dict(c))
