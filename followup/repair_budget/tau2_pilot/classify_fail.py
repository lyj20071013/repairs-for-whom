"""τ² 失败对话的粗分类与导出（2026-10-30 写；先导 3；只用 CPU，只读 results.json，不需要 tau2）。

规则（启发式，事先写定；之后由 Claude 读导出的对话核对）：
  max_steps          τ² 判跑满步数
  格式/工具调用      部署者的工具调用里，报错（error=true）的 ≥ 3 次，或占部署者工具调用的 ≥ 30%
  部署者漏做写操作    金标动作里 requestor=assistant 的写操作有没做到的（且不属于上两类）
  用户侧动作没完成    只有 requestor=user 的动作没做到（部署者没引导到位，或用户没照做）
  其他               以上都不是（例如环境断言没满足但动作检查都过了）
用法：python classify_fail.py <results.json> [更多 results.json …] [--dump 导出.md] [--n-dump 30]
"""
import argparse
import collections
import json
from pathlib import Path


def classify(s):
    term = str(s.get("termination_reason"))
    msgs = s.get("messages") or []
    a_calls = [c for m in msgs if m["role"] == "assistant" for c in (m.get("tool_calls") or [])]
    a_err = sum(1 for m in msgs if m["role"] == "tool" and m.get("requestor") == "assistant" and m.get("error"))
    ri = s.get("reward_info") or {}
    miss = [c["action"] for c in (ri.get("action_checks") or []) if not c.get("action_match")]
    miss_a = [x for x in miss if x.get("requestor") == "assistant"]
    miss_u = [x for x in miss if x.get("requestor") == "user"]
    env_fail = [e["env_assertion"]["func_name"] for e in (ri.get("env_assertions") or []) if not e.get("met")]
    info = {"n_msgs": len(msgs), "agent_calls": len(a_calls), "agent_tool_errors": a_err,
            "missed_assistant": [x["name"] for x in miss_a], "missed_user": [x["name"] for x in miss_u],
            "env_failed": env_fail}
    if "max_steps" in term:
        return "max_steps", info
    if a_err >= 3 or (a_calls and a_err / len(a_calls) >= 0.3):
        return "格式/工具调用", info
    if miss_a:
        return "部署者漏做写操作", info
    if miss_u:
        return "用户侧动作没完成", info
    return "其他", info


def short(m):
    if m.get("tool_calls"):
        return "调用 " + "; ".join(f"{c['name']}({json.dumps(c.get('arguments'), ensure_ascii=False)[:80]})" for c in m["tool_calls"])
    c = (m.get("content") or "").replace("\n", " ")
    return c[:220]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--n-dump", type=int, default=30)
    a = ap.parse_args()
    dump = []
    for f in a.files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        S = d["simulations"]
        ev = [s for s in S if s.get("reward_info")]
        infra = sum(1 for s in S if "infra" in str(s.get("termination_reason")).lower())
        fails = [s for s in ev if s["reward_info"]["reward"] < 1]
        cnt = collections.Counter()
        print(f"== {f}\n   对话 {len(S)}，基础设施错误 {infra}，判分 {len(ev)}，成功 {len(ev) - len(fails)}"
              f"（{(len(ev) - len(fails)) / max(1, len(ev)):.2f}），失败 {len(fails)}")
        err_all = sum(classify(s)[1]["agent_tool_errors"] for s in ev)
        calls_all = sum(classify(s)[1]["agent_calls"] for s in ev)
        print(f"   全部对话里部署者工具调用报错 {err_all}/{calls_all}（{err_all / max(1, calls_all):.2f}）")
        for s in fails:
            k, info = classify(s)
            cnt[k] += 1
            dump.append((f, s, k, info))
        for k, v in cnt.most_common():
            print(f"   {k}: {v}")
    if a.dump:
        L = [f"# 失败对话导出（classify_fail.py；每个文件最多 {a.n_dump} 段）", ""]
        per = collections.Counter()
        for f, s, k, info in dump:
            if per[f] >= a.n_dump:
                continue
            per[f] += 1
            L += [f"## {Path(f).parent.name} | {s['task_id'][:90]} | 试次 {s.get('trial')} | {k}", "",
                  f"- {json.dumps(info, ensure_ascii=False)}", ""]
            for i, m in enumerate(s.get("messages") or []):
                who = {"assistant": "A", "user": "U", "tool": "T"}.get(m["role"], m["role"])
                err = " [ERROR]" if m["role"] == "tool" and m.get("error") else ""
                L.append(f"{i:3d} {who}{err}: {short(m)}")
            L.append("")
        Path(a.dump).write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"导出 {sum(per.values())} 段失败对话 → {a.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
