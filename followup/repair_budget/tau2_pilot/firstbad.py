"""找 τ² 失败对话里"第一次错误写操作"（2026-10-31 写；第 6 篇 τ² 层正式建池用；只读 JSON，不依赖 tau2）。

定义（事先写定）：
  写操作   = 部署者调用的、τ² 标为 ToolType.WRITE 的工具（下面 WRITE_TOOLS，取自 tau2-bench@122e243 的 tools.py）。
  生效     = 该调用的工具结果没有 error（报错的写不改数据库，不算）。
  错误写   = 生效的写操作，且和金标动作（evaluation_criteria.actions 里 requestor=assistant 的）都对不上。
             比对规则与 τ² 的 Action.compare_with_tool_call 相同：名字相同；compare_args 为空时比全部参数，否则只比这些参数；
             每个金标动作只能配一次（按对话顺序贪心配对），所以同一个正确的写做两遍，第二遍算错误写。
  第一次错误写的位置 e = 含这次调用的部署者消息的下标。
可选位置 = 部署者消息下标 t，满足 0 < t < 总长 − 1，且 t ≤ e（t = e 时前缀里还没有这次错误写，执行者可以不做它）。
没有错误写的失败（漏做写操作、沟通没满足等）：e = None，所有部署者位置都可选。
10-31 加（用户定）：去掉对话末尾的位置，即只留 t < max_frac × 总长（默认 0.8）。先导 5 里 t / 总长 ≥ 0.8 的 4 个位置
2B / 4B / 9B 全为 0（用户快结束对话），放进池子只浪费续跑。
"""
import json

WRITE_TOOLS = {
    "retail": {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
               "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
               "return_delivered_order_items"},
    "airline": {"book_reservation", "cancel_reservation", "send_certificate", "update_reservation_baggages",
                "update_reservation_flights", "update_reservation_passengers"},
}


def _get(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def _args(o):
    a = _get(o, "arguments") or {}
    return dict(a)


def match(action, call):
    """τ² Action.compare_with_tool_call 的同一规则（action、call 可以是 dict 或 tau2 对象）。"""
    if _get(action, "name") != _get(call, "name"):
        return False
    ca = _get(action, "compare_args")
    targs, aargs = _args(call), _args(action)
    keys = list(targs.keys()) if ca is None else list(ca)
    if not keys:
        return True
    return {k: v for k, v in targs.items() if k in keys} == {k: v for k, v in aargs.items() if k in keys}


def write_calls(messages, domain):
    """按顺序返回部署者的生效写调用：[(消息下标, call)]。"""
    W = WRITE_TOOLS[domain]
    res_err = {}
    for m in messages:
        if _get(m, "role") == "tool":
            res_err[_get(m, "id")] = bool(_get(m, "error"))
    out = []
    for i, m in enumerate(messages):
        if _get(m, "role") != "assistant":
            continue
        for c in (_get(m, "tool_calls") or []):
            if _get(c, "name") in W and not res_err.get(_get(c, "id"), False):
                out.append((i, c))
    return out


def first_bad_write(messages, gold_actions, domain):
    """返回 (e, 信息)。e = 第一次错误写所在的部署者消息下标；没有错误写时 e=None。"""
    gold = [a for a in (gold_actions or []) if _get(a, "requestor", "assistant") == "assistant"]
    used = set()
    n_ok = 0
    for i, c in write_calls(messages, domain):
        hit = next((j for j, a in enumerate(gold) if j not in used and match(a, c)), None)
        if hit is None:
            return i, {"bad_call": _get(c, "name"), "bad_args": _args(c), "n_good_writes_before": n_ok}
        used.add(hit)
        n_ok += 1
    return None, {"bad_call": None, "n_good_writes_before": n_ok}


def eligible_positions(messages, gold_actions, domain, max_frac=0.8):
    e, info = first_bad_write(messages, gold_actions, domain)
    n = len(messages)
    idx = [i for i, m in enumerate(messages) if _get(m, "role") == "assistant" and 0 < i < n - 1 and i < max_frac * n]
    if e is not None:
        idx = [i for i in idx if i <= e]
    return idx, e, info


if __name__ == "__main__":       # 小检查：python firstbad.py <results.json> <domain>
    import sys
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    tasks = {t["id"]: t for t in d["tasks"]}
    for s in d["simulations"]:
        ri = s.get("reward_info") or {}
        if ri.get("reward", 1) >= 1:
            continue
        ec = tasks[s["task_id"]].get("evaluation_criteria") or {}
        idx, e, info = eligible_positions(s["messages"], ec.get("actions"), sys.argv[2])
        print(s["id"][:12], s["task_id"], len(s["messages"]), e, len(idx), json.dumps(info, ensure_ascii=False)[:150])
