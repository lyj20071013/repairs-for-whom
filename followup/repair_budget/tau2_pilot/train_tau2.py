"""τ² 上训练 π1（Qwen3.5 + LoRA SFT）及其检查（2026-10-31 写；第 6 篇 τ² 探针，也给主实验用）。

子命令：
  export      （τ² 的 venv，只用 CPU）由设计文件 + π0 的续跑选出修复标签，导出训练样本；同时抽锚样本（π0 成功对话里的部署者消息）。
              每个样本的提示 = τ² 部署者在该位置**实际发给模型的请求**（messages + tools）：在 litellm 的 completion 处截获，不调用模型。
  check       （CPU；有 vLLM 服务时更好）用真实分词器把样本渲染成"提示 + 助手部分"，逐条检查；
              --vllm-port 时再用 vLLM 的 /tokenize 核对：训练用的提示 token 与服务实际看到的完全相同。
  probe-lora  （GPU，约 15–20 分钟）Qwen3.5 是多模态结构（Qwen3_5ForConditionalGeneration），先确认：
              ① 语言模型里有哪些线性层、LoRA 作用在哪些层；② 一次前向 + 反向的速度与显存；
              ③ 存出的 adapter 能被 vLLM 加载，且 vLLM 带 adapter 的贪心输出与 transformers 一致（层名映射对）。
  train       （GPU）LoRA SFT，只在助手部分算损失。

样本格式（export 写，train / check 读；每行一个）：
  {"slot", "kind": "repair"|"anchor", "domain", "task_id", "pos_id", "cand_id", "kinds", "g_pi0",
   "messages": [litellm 格式，含系统提示], "tools": [OpenAI 工具 schema], "target": {litellm 格式的助手消息}}
渲染（与 vLLM 的 OpenAI 服务一致）：工具调用的 arguments 由 JSON 字符串转成 dict；
  提示 = apply_chat_template(messages, tools, add_generation_prompt=True, enable_thinking=False)；
  助手部分 = apply_chat_template(messages + [target], …, add_generation_prompt=False) 去掉提示前缀，到 <|im_end|> 为止。
"""
import argparse
import collections
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

# 10-31：直接在 Python 里起 vLLM 时，关掉要现场编译的 FlashInfer 采样核（14b2 的系统 nvcc 太旧；同 serve_tau.sh）
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/root/autodl-tmp/cache")
HERE = Path(__file__).resolve().parent
END = "<|im_end|>"
THINK_OFF = "<think>\n\n</think>\n\n"
# 只作用在语言模型上（排除视觉塔 visual.*）；Qwen3.5 的线性注意力层没有 q/k/v_proj，只有完整注意力层与 MLP 会被选中
TARGETS = {
    "attn_mlp": r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    "mlp": r"^(?!.*visual).*\.(gate_proj|up_proj|down_proj)$",
}


def sseed(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def jl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ====================================================================== export
def capture_request(P, domain, task, prefix):
    """在 τ² 部署者生成下一条消息时截获发给 litellm 的 (messages, tools)，不调用模型。"""
    import tau2.utils.llm_utils as LU

    class _Cap(Exception):
        pass
    box = {}

    def fake_completion(model=None, messages=None, tools=None, tool_choice=None, **kw):
        box["messages"] = copy.deepcopy(messages)
        box["tools"] = copy.deepcopy(tools)
        box["tool_choice"] = tool_choice
        raise _Cap("__CAPTURE__")
    cfg = P.make_cfg(domain, "4B", "q27", 10)
    orch = P.build_text_orchestrator(cfg, P.with_history(task, prefix), seed=0)
    orch.initialize()
    assert str(orch.to_role).lower().endswith("agent"), f"前缀之后该说话的不是部署者：{orch.to_role}"
    old = LU.completion
    LU.completion = fake_completion
    try:
        orch.agent.generate_next_message(orch.message.model_copy(deep=True), copy.deepcopy(orch.agent_state))
    except _Cap:
        pass
    finally:
        LU.completion = old
    if "messages" not in box:
        raise RuntimeError("没有截获到请求")
    return box["messages"], box["tools"]


def target_dict(P, msg, tag):
    from tau2.utils.llm_utils import to_litellm_messages
    m = P.clean(msg, tag)                      # 去掉用量等、工具调用 id 换成确定值
    return to_litellm_messages([m])[0]


# ---------------------------------------------------------------- 11-01：给不带文字的修复标签补前置文字（登记补充三第 2 节）
PRE_LEAK = re.compile(r"(given|suggest|hint|been told|was told|told to|instruct|next action|specified|provided|"
                      r"i (?:am|was) asked|asked to (?:call|use|make)|note for this step|as requested by the system)", re.I)
PRE_ASK = ("[Note for this step only] Your next action will be to call the tool `{name}` with these arguments: {args}\n"
           "Write only the short message (one to three sentences) that you would write in this conversation right before "
           "making that call, in your usual style. Do not call any tool. Do not mention that the action was given, "
           "suggested or decided for you.")


from adm_tau2 import vllm_chat  # 11-05：与可执行性判定共用（adm_tau2.py；原来在这里，内容不变）


def target_needs_preamble(tgt):
    return bool(tgt.get("tool_calls")) and not (tgt.get("content") or "").strip()


# 11-01 补充五：模板兜底（同 P0 登记 5.14 的模板 Thought）；只用于可执行的标签
TEMPLATES = {
    "find_user_id_by_email": "Let me look up your account using your email address.",
    "find_user_id_by_name_zip": "Let me look up your account using your name and zip code.",
    "get_user_details": "Let me look up your account details.",
    "get_order_details": "Let me look up the details of order {order_id}.",
    "get_product_details": "Let me look up the product details.",
    "list_all_product_types": "Let me check the list of available product types.",
    "get_reservation_details": "Let me look up the details of reservation {reservation_id}.",
    "list_all_airports": "Let me check the list of airports.",
    "search_direct_flight": "Let me search for direct flights from {origin} to {destination} on {date}.",
    "search_onestop_flight": "Let me search for one-stop flights from {origin} to {destination} on {date}.",
    "get_flight_status": "Let me check the status of flight {flight_number}.",
    "calculate": "Let me calculate that.",
}


def template_text(name, args, is_write):
    if is_write:
        return "I'll go ahead with that now."
    t = TEMPLATES.get(name, "Let me look that up.")
    try:
        return t.format(**{k: v for k, v in (args or {}).items()})
    except (KeyError, IndexError, ValueError):
        return re.sub(r"\s*\{[^}]*\}", "", t).replace(" .", ".")


def fill_preamble(port, served, msgs, tools, tgt, key, n_nat=8, n_given=3, okfn=None, template=False):
    """返回 (新目标或 None, 方式, 记录)。方式：natural（π0 自然采样，先写文字、调用同名工具）/ given（告诉 π0 动作后让它写）/ dropped。"""
    names = [tc["function"]["name"] for tc in tgt["tool_calls"]]
    rec = {"natural": [], "given": []}
    try:
        ch = vllm_chat(port, served, msgs, tools, 0.7, sseed("PRE_NAT", key), n=n_nat)
    except Exception as e:
        ch = []
        rec["natural_error"] = repr(e)[:300]
    for c in ch:
        m = c.get("message") or {}
        txt = (m.get("content") or "").strip()
        got = [tc["function"]["name"] for tc in (m.get("tool_calls") or [])]
        rec["natural"].append({"text": txt[:300], "calls": got})
        if txt and got and got[0] == names[0] and len(txt) <= 1500 and not txt.startswith("<"):
            ok, why = okfn(txt) if okfn else (True, None)
            rec["natural"][-1]["why"] = why
            if ok:
                return dict(tgt, content=txt), "natural", rec
    args = json.dumps(json.loads(tgt["tool_calls"][0]["function"]["arguments"])
                      if isinstance(tgt["tool_calls"][0]["function"]["arguments"], str)
                      else tgt["tool_calls"][0]["function"]["arguments"], ensure_ascii=False)
    ask = msgs + [{"role": "user", "content": PRE_ASK.format(name=names[0], args=args)}]
    for j in range(n_given):
        try:
            c = vllm_chat(port, served, ask, tools, 0.0 if j == 0 else 0.7, sseed("PRE_GIV", key, j), tool_choice="none",
                          max_tokens=300)[0]
        except Exception as e:
            rec["given"].append({"error": repr(e)[:300]})
            continue
        m = c.get("message") or {}
        txt = (m.get("content") or "").strip().strip('"').strip()
        why = None
        if m.get("tool_calls"):
            why = "called_tool"
        elif not txt or len(txt) > 600 or "<tool_call>" in txt:
            why = "empty_or_long"
        elif PRE_LEAK.search(txt):
            why = "leak"
        elif okfn:
            ok, why = okfn(txt)
        rec["given"].append({"text": txt[:300], "why": why})
        if why is None:
            return dict(tgt, content=txt), "given", rec
    if template:
        n0, g0 = call_list(tgt)[0]
        return dict(tgt, content=template_text(n0, g0, n0 in write_names())), "template", rec
    return None, "dropped", rec


# ---------------------------------------------------------------- 11-01：可执行性过滤与一致性检查（登记补充四）
# 11-05：判定的代码抽到 adm_tau2.py（design 之后的标注、export、分析共用）；这里只留补文字的一致性检查。
from adm_tau2 import (JUDGE_CONFIRM, Admissibility, call_list, ctx_text, judge, load_verdicts,  # noqa: F401
                      missing_args, scalars, transcript, write_names)


JUDGE_CONSIST = ("Below is the end of a conversation between a customer-service agent and a user. In its next turn the agent "
                 "writes the message shown and, in the same turn, immediately makes the tool call shown.\n\n=== Conversation "
                 "(end) ===\n{conv}\n\n=== Agent's next message ===\n{msg}\n\n=== Tool call made right after it ===\n{act}\n\n"
                 "Is the message consistent with making exactly this tool call right now? It is NOT consistent if it asks the "
                 "user to confirm or to provide information first, says it will do a different action first, or claims "
                 "something happened that has not happened. Answer with exactly YES or NO.")
PRE_BAD = re.compile(r"\?|please confirm|confirm (?:with|the|that|if|whether)|before i (?:can )?proceed|could you|"
                     r"can you (?:please )?(?:provide|tell|confirm|share|give)|please (?:provide|share|tell)", re.I)
PRE_BAD_WRITE = re.compile(r"(?:need to|let me|i'll|i will|first,? i)\s+(?:first\s+)?(?:check|verify|look up|search|get|confirm|find)", re.I)


def preamble_ok(text, calls, msgs, a, key):
    """补出的文字能否与紧接着的调用放在一起：规则 + 27B 判断（a.judge_port 为 0 时只用规则）。"""
    if PRE_BAD.search(text):
        return False, "rule_ask"
    if calls and calls[0][0] in write_names() and PRE_BAD_WRITE.search(text):
        return False, "rule_other_action"
    if not getattr(a, "judge_port", 0):
        return True, None
    act = "; ".join(f"{n}({json.dumps(g, ensure_ascii=False)})" for n, g in calls)
    ok, raw = judge(a.judge_port, a.judge_name, JUDGE_CONSIST.format(conv=transcript(msgs, n=8), msg=text, act=act),
                    sseed("JC", key))
    return ok, None if ok else f"judge_{raw}"


def rates_from_rollouts(paths, executor):
    """{(pos_id, cand_id): [reward, …]}（按 k 排序；同一 k 多行时取 reward 不为 None 的最后一行）。"""
    R = {}
    for p in paths:
        for r in jl(p):
            if r["executor"] != executor:
                continue
            key = (r["pos_id"], r["cand_id"], r["k"], r["temp"])
            if r.get("score") != "orig_task":
                continue
            if r.get("reward") is not None or key not in R:
                R[key] = r
    out = {}
    for (pos, cand, k, temp), r in sorted(R.items()):
        if r.get("reward") is not None:
            out.setdefault((pos, cand), []).append(float(r["reward"]))
    return out


EXCLUDE = set()          # 11-01：--exclude-kinds 给出的来源；只来自这些来源的候选不作修复标签（例如 rand）


def is_repair(c):
    """可作修复标签的候选：不是自由选择，不是只有"原动作"来源，也不是只来自 --exclude-kinds 的来源。"""
    if "free" in c["kinds"] or c["kinds"] == ["orig"]:
        return False
    return not (EXCLUDE and set(c["kinds"]) <= EXCLUDE)


def choose_labels(design, rates, thr, min_n, admissible=None):
    """admissible(d, c) -> bool（11-01 补充四：只在可执行的修复候选里选）；None = 不过滤。"""
    rows = []
    for d in design:
        cs = [c for c in d["cands"] if c.get("restore_ok")]
        free = rates.get((d["pos_id"], "free"))
        rep = [(c, rates.get((d["pos_id"], c["cand_id"]))) for c in cs if is_repair(c)
               and (admissible is None or admissible(d, c))]
        rep = [(c, v) for c, v in rep if v]
        if not free or not rep:
            continue
        pf = sum(free) / len(free)
        best = max(rep, key=lambda cv: (sum(cv[1]) / len(cv[1]), sseed("TIE", d["pos_id"], cv[0]["cand_id"])))
        pb = sum(best[1]) / len(best[1])
        rows.append({"d": d, "c": best[0], "g": pb - pf, "p_best": pb, "p_free": pf, "n": len(best[1])})
    rows.sort(key=lambda r: (-r["g"], sseed("ORDER", r["d"]["pos_id"])))
    keep = [r for r in rows if r["g"] >= thr - 1e-9]
    rule = f"G ≥ {thr}"
    if len(keep) < min_n:                        # 事先定的补足规则：按 G 从大到小补到 min_n 个（只取 G > 0 的）
        keep = [r for r in rows if r["g"] > 1e-9][:min_n]
        rule = f"G ≥ {thr} 只有 {sum(r['g'] >= thr - 1e-9 for r in rows)} 个，按 G 补到 {len(keep)} 个（G > 0）"
    return keep, rows, rule


def cmd_export(a):
    sys.path.insert(0, str(HERE))
    import pool_tau2 as P
    from loguru import logger
    from tau2.data_model.simulation import Results
    logger.remove()
    logger.add(sys.stderr, level="WARNING", filter=lambda r: "__CAPTURE__" not in str(r["message"])
               and r["function"] not in ("get_response_cost", "get_cost"))
    EXCLUDE.update(a.exclude_kinds or [])
    design = [d for f in a.design for d in jl(f)]
    rates = rates_from_rollouts(a.rollouts, a.executor)
    srcs = {}

    def R(src):
        if src not in srcs:
            srcs[src] = Results.load(Path(src))
        return srcs[src]
    adm_fn, adm_rows = None, []
    if a.admissible:                               # 11-01 补充四：可执行性过滤（11-05：判定代码在 adm_tau2.py）
        if not a.judge_port:                       # 补文字的一致性检查也要 27B，所以有 --adm-file 时同样要求
            log("[停下] --admissible 需要 --judge-port（27B 服务）")
            return 2
        seen = set()
        if a.adm_file:                             # 11-05：用 adm_tau2.py annotate 事先写好的判定（标签与池上读数共用同一份）
            V = {(r["pos_id"], r["cand_id"]): r for f in a.adm_file for r in jl(f)}
            log(f"可执行性判定来自 {a.adm_file}（{len(V)} 行；规则 {sorted({str(r.get('rule', 'v1')) for r in V.values()})}）")

            def adm_fn(d, c):
                k = (d["pos_id"], c["cand_id"])
                if k not in V:
                    raise KeyError(f"--adm-file 里没有 {k} 的判定（先对这个设计文件跑 adm_tau2.py annotate）")
                if k not in seen:
                    seen.add(k)
                    adm_rows.append(V[k])
                return bool(V[k]["ok"])
        else:                                      # 与 11-01 的做法相同：现场判（--adm-rule 默认 v1 = 探针原样）
            A = Admissibility(a.adm_rule, a.judge_port, a.judge_name)

            def adm_fn(d, c):
                k = (d["pos_id"], c["cand_id"])
                row = A.check(d, c)
                if k not in seen:
                    seen.add(k)
                    adm_rows.append(row)
                return row["ok"]
    keep0, _, _ = choose_labels(design, rates, a.thr, a.min_n) if a.admissible else (None, None, None)
    keep, allrows, rule = choose_labels(design, rates, a.thr, a.min_n, admissible=adm_fn)
    log(f"位置 {len(design)} 个，有 π0（{a.executor}）读数的 {len(allrows)} 个；标签：{rule} → {len(keep)} 个")
    if a.admissible:
        by = collections.Counter(("+".join(r["kinds"]), r["why"] or "ok") for r in adm_rows)
        log("可执行性（修复候选）：" + "；".join(f"{k}/{w} {n}" for (k, w), n in sorted(by.items())))
        kk = lambda rows: dict(collections.Counter("+".join(r["c"]["kinds"]) for r in rows))
        log(f"重选前标签来源 {kk(keep0)}（{len(keep0)} 个）→ 重选后 {kk(keep)}（{len(keep)} 个）；"
            f"重选后与原标签相同的位置 {sum(1 for r in keep if any(r0['d']['pos_id'] == r['d']['pos_id'] and r0['c']['cand_id'] == r['c']['cand_id'] for r0 in keep0))}")
        Path(str(a.out) + ".admissible.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in adm_rows),
                                                         encoding="utf-8")
    out, n_err, n_drop = [], 0, []
    if a.preamble and not a.vllm_port:
        log("[停下] --preamble 需要 --vllm-port（4B 服务）")
        return 2
    for i, r in enumerate(keep):
        d, c = r["d"], r["c"]
        Rs = R(d["src"])
        s = next(x for x in Rs.simulations if x.id == d["sim_id"])
        task = next(t for t in Rs.tasks if t.id == d["task_id"])
        prefix = [m.model_copy(deep=True) for m in s.messages[: d["t"]]]
        try:
            msgs, tools = capture_request(P, d["domain"], task, prefix)
            tgt = target_dict(P, P.AssistantMessage.model_validate(c["msg"]), f"{d['pos_id']}_lab")
        except Exception as e:
            n_err += 1
            log(f"  导出失败 {d['pos_id']}：{repr(e)[:200]}")
            continue
        pre_mode = "orig_text" if not target_needs_preamble(tgt) else "bare"
        pre_rec = None
        if a.preamble and pre_mode == "bare":
            okfn = (lambda txt, _c=call_list(tgt), _m=msgs, _k=d["pos_id"]: preamble_ok(txt, _c, _m, a, _k)) \
                if a.admissible else None
            tgt2, pre_mode, pre_rec = fill_preamble(a.vllm_port, a.served_name, msgs, tools, tgt, d["pos_id"], okfn=okfn,
                                                    template=a.template and a.admissible)
            log(f"  补文字 {d['pos_id']}（{'+'.join(c['kinds'])}，{tgt['tool_calls'][0]['function']['name']}）：{pre_mode}"
                + (f" → {tgt2['content'][:100]!r}" if tgt2 else ""))
            if tgt2 is None:
                n_drop.append(d["pos_id"])
                continue
            tgt = tgt2
        out.append({"kind": "repair", "domain": d["domain"], "task_id": d["task_id"], "pos_id": d["pos_id"],
                    "cand_id": c["cand_id"], "kinds": c["kinds"], "g_pi0": round(r["g"], 4), "p_best": r["p_best"],
                    "p_free": r["p_free"], "messages": msgs, "tools": tools, "target": tgt,
                    "preamble": pre_mode, "preamble_rec": pre_rec})
    n_rep = len(out) + len(n_drop)                 # 11-01：锚样本数按"补文字之前"的修复标签数算，锚与 S1 完全相同
    # 锚样本：π0 自己成功对话里的部署者消息（不含开场白），只取训练任务；与修复样本数相同
    tasks_ok = None
    if a.task_file:
        # 11-05 修正：retail 与 airline 的任务 id 会重号（都有 "12" 等），只按 id 过滤会把另一领域的同号任务也放进来。
        # 给了 --task-domain（与 --task-file 一一对应）时按 (领域, 任务 id) 过滤；没给时沿用旧做法并提示。
        if a.task_domain:
            if len(a.task_domain) != len(a.task_file):
                log("[停下] --task-domain 要与 --task-file 一一对应")
                return 2
            tasks_ok = {(dm, l.strip()) for f, dm in zip(a.task_file, a.task_domain)
                        for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()}
        else:
            log("注意：没有 --task-domain，锚样本只按任务 id 过滤（两个领域的同号任务会混进来）")
            tasks_ok = {l.strip() for f in a.task_file for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()}
    cand = []
    for src in a.anchor_src:
        Rs = R(src)
        dom = a.anchor_domain[a.anchor_src.index(src)]
        for s in Rs.simulations:
            if s.reward_info is None or s.reward_info.reward < 1 or not s.messages:
                continue
            if tasks_ok is not None and (((dom, s.task_id) not in tasks_ok) if a.task_domain else (s.task_id not in tasks_ok)):
                continue
            for t, m in enumerate(s.messages):
                if m.role == "assistant" and 0 < t:
                    cand.append((src, dom, s, t))
    rng = random.Random(sseed("ANCHOR", a.seed))
    rng.shuffle(cand)
    by_sim, order = {}, []
    for x in cand:                                # 轮流取：先每段对话 1 条，再第 2 条……
        by_sim.setdefault(x[2].id, []).append(x)
    rounds = list(by_sim.values())
    j = 0
    n_want = int(round(a.anchor_mult * n_rep))      # 11-01：锚样本数 = 倍数 × 修复样本数（默认 1，同 P0）
    while any(rounds) and len(order) < n_want:
        for lst in rounds:
            if lst and len(order) < n_want:
                order.append(lst.pop(0))
        j += 1
    n_anchor = 0
    for src, dom, s, t in order:
        task = next(x for x in R(src).tasks if x.id == s.task_id)
        try:
            msgs, tools = capture_request(P, dom, task, [m.model_copy(deep=True) for m in s.messages[:t]])
            tgt = target_dict(P, s.messages[t], f"{s.id[:12]}_{t}_anc")
        except Exception as e:
            n_err += 1
            log(f"  锚样本导出失败 {s.id[:12]}_{t}：{repr(e)[:200]}")
            continue
        out.append({"kind": "anchor", "domain": dom, "task_id": s.task_id, "pos_id": f"{s.id[:12]}_{t}", "cand_id": None,
                    "kinds": ["anchor"], "g_pi0": None, "messages": msgs, "tools": tools, "target": tgt})
        n_anchor += 1
    rep_rows = [x for x in out if x["kind"] == "repair"]
    anc_rows = [x for x in out if x["kind"] == "anchor"]
    final = [dict(x, slot=i) for i, x in enumerate(rep_rows + anc_rows)]   # 槽位只作编号；训练顺序由种子决定的排列给出
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in final), encoding="utf-8")
    kinds = {}
    for x in rep_rows:
        kk = "+".join(x["kinds"])
        kinds[kk] = kinds.get(kk, 0) + 1
    summ = {"time": time.strftime("%Y-%m-%d %H:%M"), "design": a.design, "rollouts": a.rollouts, "executor": a.executor,
            "rule": rule, "n_positions": len(design), "n_with_rates": len(allrows), "n_repair": len(rep_rows),
            "n_anchor": len(anc_rows), "n_export_errors": n_err, "label_kinds": kinds,
            "g_all": [round(r["g"], 4) for r in allrows], "is_tool_target": sum(bool(x["target"].get("tool_calls")) for x in final),
            "anchor_src": a.anchor_src, "task_file": a.task_file, "exclude_kinds": sorted(EXCLUDE), "anchor_mult": a.anchor_mult,
            "preamble": a.preamble, "preamble_modes": dict(collections.Counter(x.get("preamble") for x in rep_rows)),
            "preamble_dropped": n_drop, "admissible": a.admissible,
            "admissible_counts": {f"{k}/{w}": n for (k, w), n in collections.Counter(
                ("+".join(r["kinds"]), r["why"] or "ok") for r in adm_rows).items()},
            "tool_target_with_text": {k: [sum(1 for x in final if x["kind"] == k and x["target"].get("tool_calls")
                                              and (x["target"].get("content") or "").strip()),
                                          sum(1 for x in final if x["kind"] == k and x["target"].get("tool_calls"))]
                                      for k in ("repair", "anchor")}}
    Path(str(a.out) + ".summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"写出 {len(final)} 个样本（修复 {len(rep_rows)}、锚 {len(anc_rows)}；目标是工具调用的 {summ['is_tool_target']}）"
        f"→ {a.out}；标签来源 {kinds}；导出失败 {n_err}")
    if a.preamble:
        log(f"补文字：{summ['preamble_modes']}；丢弃 {len(n_drop)} 个；工具调用目标中带文字的（修复 / 锚）："
            f"{summ['tool_target_with_text']['repair']} / {summ['tool_target_with_text']['anchor']}")
    if a.admissible:                               # 补充四第 4 节：最终修复样本 ≥ --min-final 才训练；补充五：--min-final 0 = 照样训练
        too_many_drop = len(rep_rows) < a.min_final
        if too_many_drop:
            log(f"[停下] 最终修复样本 {len(rep_rows)} 个 < {a.min_final}（登记补充四：停下交用户）")
        elif len(rep_rows) < 30:
            log(f"注意：最终修复样本 {len(rep_rows)} 个 < 30；按登记补充五第 2 节第 4 条照样训练，结果里注明 N = {len(rep_rows)}")
    else:
        too_many_drop = a.preamble and len(n_drop) > len(keep) / 3
        if too_many_drop:
            log(f"[停下] 补文字丢弃 {len(n_drop)} 个，超过标签数的 1/3（登记补充三：停下交用户）")
    ok = n_err == 0 and len(rep_rows) > 0 and not too_many_drop
    print("== 完成" if ok else "== 有问题，见上")
    return 0 if ok else 1


# ====================================================================== 渲染
def to_template_msgs(msgs):
    """与 vLLM 的 OpenAI 服务一样：工具调用 arguments 的 JSON 字符串转成 dict；content 为 None 时用空串。"""
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                f = dict(tc["function"])
                if isinstance(f.get("arguments"), str):
                    try:
                        f["arguments"] = json.loads(f["arguments"])
                    except Exception:
                        pass
                tcs.append({"id": tc.get("id"), "type": "function", "function": f})
            m["tool_calls"] = tcs
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def render(tok, row):
    msgs = to_template_msgs(row["messages"])
    tgt = to_template_msgs([row["target"]])[0]
    kw = dict(tools=row["tools"], tokenize=False, enable_thinking=False)
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, **kw)
    full = tok.apply_chat_template(msgs + [tgt], add_generation_prompt=False, **kw)
    if full.startswith(prompt):
        comp, mode = full[len(prompt):], "prefix"
    else:                                         # 退路：取最后一个助手块，去掉空的思考块（提示里已经有了）
        i = full.rfind("<|im_start|>assistant\n")
        comp = full[i + len("<|im_start|>assistant\n"):]
        comp = re.sub(r"^<think>\n.*?\n</think>\n\n", "", comp, count=1, flags=re.S)
        mode = "fallback"
    j = comp.find(END)
    if j < 0:
        raise ValueError("助手部分里没有 <|im_end|>")
    return prompt, comp[: j + len(END)], mode


def encode(tok, row):
    prompt, comp, mode = render(tok, row)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(comp, add_special_tokens=False)["input_ids"]
    return list(p), list(c), prompt, comp, mode


def load_tok(model_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    if END not in tok.get_vocab():
        raise RuntimeError(f"分词器里没有 {END}")
    return tok


# ====================================================================== check
def cmd_check(a):
    tok = load_tok(a.model_dir)
    rows = jl(a.samples)
    bad, modes, lens, n_think = 0, {}, [], 0
    ex = {}
    enc = []
    for r in rows:
        try:
            p, c, prompt, comp, mode = encode(tok, r)
        except Exception as e:
            bad += 1
            log(f"  [失败] 槽位 {r['slot']}：{repr(e)[:200]}")
            continue
        modes[mode] = modes.get(mode, 0) + 1
        n_think += prompt.endswith("<|im_start|>assistant\n" + THINK_OFF)
        if tok.decode(c) != comp:
            bad += 1
            log(f"  [失败] 槽位 {r['slot']}：助手部分 decode 后与文本不同")
        lens.append(len(p) + len(c))
        key = "tool" if r["target"].get("tool_calls") else "text"
        ex.setdefault(key, (r, comp))
        enc.append((r, p))
    lens.sort()
    q = lambda f: lens[min(len(lens) - 1, int(f * len(lens)))] if lens else None
    log(f"样本 {len(rows)} 个：渲染失败 {bad}；助手部分取法 {modes}；提示以空思考块结尾的 {n_think}")
    log(f"长度（token）：中位 {q(0.5)}、90% {q(0.9)}、最大 {q(1.0)}；超过 --max-len {a.max_len} 的 {sum(x > a.max_len for x in lens)} 个")
    for k, (r, comp) in ex.items():
        print(f"--- 例（{k}，槽位 {r['slot']}，{r['kind']}）助手部分：\n{comp[:600]!r}")
    if a.vllm_port:
        import urllib.request
        n_ok = n_try = 0
        for r, p in enc[: a.n_tokenize]:
            body = {"model": a.served_name, "messages": r["messages"], "tools": r["tools"], "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{a.vllm_port}/tokenize", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            n_try += 1
            try:
                toks = json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"]
            except Exception as e:
                log(f"  /tokenize 出错：{repr(e)[:300]}")
                continue
            if toks == p:
                n_ok += 1
            else:
                k = next((i for i, (x, y) in enumerate(zip(toks, p)) if x != y), min(len(toks), len(p)))
                log(f"  [不同] 槽位 {r['slot']}：vLLM {len(toks)} 个 token、本地 {len(p)} 个；第 {k} 个起不同："
                    f"vLLM {tok.decode(toks[max(0, k - 5):k + 15])!r} | 本地 {tok.decode(p[max(0, k - 5):k + 15])!r}")
        log(f"vLLM /tokenize 核对：{n_ok}/{n_try} 个样本的提示 token 完全相同")
        if n_ok < n_try:
            bad += 1
    print("== 通过" if bad == 0 else "== 有问题，见上")
    return 0 if bad == 0 else 1


# ====================================================================== 模型
def load_model(model_dir, dtype="bfloat16"):
    import torch
    import transformers as T
    dt = getattr(torch, dtype)
    errs = []
    for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            m = getattr(T, cls).from_pretrained(model_dir, dtype=dt, attn_implementation="sdpa")
            return m, cls
        except Exception as e:
            errs.append(f"{cls}: {repr(e)[:300]}")
    raise RuntimeError("加载模型失败：" + " | ".join(errs))


def add_lora(model, seed, targets, r, alpha, dropout):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(int(seed))
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS[targets], bias="none")
    pm = get_peft_model(model, cfg)
    n = sum(1 for k, _ in pm.named_modules() if k.endswith(".lora_A"))
    return pm, n


def seq_logp(model, p, c, device, mean=False):
    import torch
    ids = torch.tensor([p + c], device=device)
    k = len(c) + 1
    try:
        logits = model(input_ids=ids, logits_to_keep=k, use_cache=False).logits[0, :-1]
    except TypeError:
        logits = model(input_ids=ids, use_cache=False).logits[0, -k:-1]
    lp = torch.log_softmax(logits.float(), dim=-1).gather(1, torch.tensor(c, device=device)[:, None])[:, 0]
    return lp.mean() if mean else lp.sum()


def fast_path_info():
    info = {}
    try:
        import fla  # noqa: F401
        info["fla"] = getattr(fla, "__version__", "有")
    except Exception:
        info["fla"] = None
    try:
        import importlib
        M = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        info["is_fast_path_available"] = getattr(M, "is_fast_path_available", "未知")
    except Exception as e:
        info["is_fast_path_available"] = f"导入失败 {repr(e)[:100]}"
    return info


# ====================================================================== probe-lora
def cmd_probe_lora(a):
    import gc
    import torch
    import transformers as T
    from collections import Counter
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"time": time.strftime("%Y-%m-%d %H:%M"), "model_dir": a.model_dir, "targets": a.targets,
           "versions": {"torch": torch.__version__, "transformers": T.__version__}, "fast_path": fast_path_info()}
    try:
        import peft
        import vllm
        rep["versions"].update(peft=peft.__version__, vllm=vllm.__version__)
    except Exception as e:
        log(f"[停下] 缺少 peft 或 vllm：{e!r}（τ² 的 venv 里 pip install peft）")
        return 2
    log(f"版本 {rep['versions']}；线性注意力快速实现 {rep['fast_path']}")
    # ① 线性层清单（真实加载，CPU 也可，但直接上 GPU 省时间）
    model, cls = load_model(a.model_dir)
    rep["load_class"] = cls
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    suf = Counter(("visual" if "visual" in n else "lm") + ":" + n.split(".")[-1] for n in names)
    pat = re.compile(TARGETS[a.targets])
    hit = [n for n in names if pat.match(n)]
    rep["linear_suffix_counts"] = dict(sorted(suf.items()))
    rep["n_target_modules"] = len(hit)
    rep["target_examples"] = hit[:6] + hit[-3:]
    log(f"加载类 {cls}；线性层（视觉 / 语言 × 名字）：{dict(sorted(suf.items()))}")
    log(f"LoRA 目标（{a.targets}）匹配 {len(hit)} 个，例：{hit[:3]} … {hit[-2:]}")
    if not hit:
        log("[停下] LoRA 一个层都没匹配上")
        return 1
    model.to("cuda")
    tok = load_tok(a.model_dir)
    pm, n = add_lora(model, 1, a.targets, 16, 32, 0.0)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():                        # 让 adapter 真的改变输出（B 默认是 0）
        for k, p in pm.named_parameters():
            if "lora_B" in k:
                p.copy_((torch.randn(p.shape, generator=g) * a.b_std).to(p.dtype))
    pm.save_pretrained(str(out / "probe_adapter"))
    rep["adapter_files"] = sorted(x.name for x in (out / "probe_adapter").iterdir())
    keys = []
    try:
        from safetensors import safe_open
        with safe_open(str(out / "probe_adapter" / "adapter_model.safetensors"), "pt") as f:
            keys = list(f.keys())
    except Exception as e:
        log(f"  读 adapter 的键失败：{e!r}")
    rep["adapter_key_examples"] = keys[:4]
    log(f"adapter 已存（{n} 个 LoRA 层）；键名例：{keys[:2]}")
    # ② 速度与显存：一次前向 + 反向（长度 --bench-len）
    pm.train()
    pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        pm.enable_input_require_grads()
    except Exception:
        pass
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(100, 50000, (1, a.bench_len), device="cuda")
    t0 = time.time()
    for _ in range(2):
        loss = seq_logp(pm, ids[0, :-200].tolist(), ids[0, -200:].tolist(), "cuda", mean=True)
        (-loss).backward()
        pm.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    rep["bench"] = {"len": a.bench_len, "s_per_step": round((time.time() - t0) / 2, 2),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    log(f"速度：长度 {a.bench_len} 的前向 + 反向每次 {rep['bench']['s_per_step']} 秒，显存峰值 {rep['bench']['peak_gb']} GB")
    pm.gradient_checkpointing_disable()
    pm.eval()
    # ③ transformers 的贪心（带 / 不带 adapter）
    prompts = [[{"role": "user", "content": x}] for x in
               ["Hi! I want to return an item from my last order.", "What is 17 times 23?",
                "Write one sentence about the airline baggage policy."]]
    pids = [tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False, enable_thinking=False) for p in prompts]
    pids = [tok(x, add_special_tokens=False)["input_ids"] for x in pids]
    hf = {"base": [], "lora": []}
    with torch.no_grad():
        for p in pids:
            x = torch.tensor([p], device="cuda")
            with pm.disable_adapter():
                hf["base"].append(pm.generate(input_ids=x, max_new_tokens=a.n_tok, do_sample=False)[0, len(p):].tolist())
            hf["lora"].append(pm.generate(input_ids=x, max_new_tokens=a.n_tok, do_sample=False)[0, len(p):].tolist())
    del pm, model
    gc.collect()
    torch.cuda.empty_cache()
    # ④ vLLM 加载同一个 adapter
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    try:
        llm = LLM(model=a.model_dir, enable_lora=True, max_lora_rank=16, max_loras=1, max_model_len=4096,
                  gpu_memory_utilization=a.vllm_util, max_num_seqs=8)
    except Exception as e:
        rep["vllm_error"] = repr(e)[:2000]
        log(f"[停下] vLLM 开 LoRA 起不来：{repr(e)[:600]}")
        (out / "probe_lora.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1
    sp = SamplingParams(temperature=0.0, max_tokens=a.n_tok)
    inputs = [{"prompt_token_ids": p} for p in pids]
    vl = {"base": [o.outputs[0].token_ids for o in llm.generate(inputs, sp)]}
    try:
        lr = LoRARequest("probe", 1, str(out / "probe_adapter"))
        vl["lora"] = [o.outputs[0].token_ids for o in llm.generate(inputs, sp, lora_request=lr)]
    except Exception as e:
        rep["vllm_lora_error"] = repr(e)[:2000]
        log(f"[停下] vLLM 加载 adapter 出错：{repr(e)[:600]}（可试 --targets mlp）")
        (out / "probe_lora.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1
    res = []
    for i in range(len(pids)):
        b_hf, l_hf, b_v, l_v = hf["base"][i], hf["lora"][i], list(vl["base"][i]), list(vl["lora"][i])
        res.append({"base_first_same": b_hf[:1] == b_v[:1], "lora_first_same": l_hf[:1] == l_v[:1],
                    "base_prefix_same": sum(1 for x, y in zip(b_hf, b_v) if x == y) if b_hf[:1] == b_v[:1] else 0,
                    "lora_vs_base_differs_vllm": l_v != b_v, "lora_vs_base_differs_hf": l_hf != b_hf,
                    "hf_lora": tok.decode(l_hf), "vllm_lora": tok.decode(l_v), "vllm_base": tok.decode(b_v)})
    rep["compare"] = res
    ok = all(r["base_first_same"] and r["lora_first_same"] for r in res) and any(r["lora_vs_base_differs_vllm"] for r in res)
    rep["ok"] = ok
    for i, r in enumerate(res):
        log(f"  提示 {i}：第一个 token 基础模型 HF=vLLM {r['base_first_same']}、带 adapter HF=vLLM {r['lora_first_same']}；"
            f"adapter 改变了 vLLM 输出 {r['lora_vs_base_differs_vllm']}（HF {r['lora_vs_base_differs_hf']}）")
        log(f"     vLLM 带 adapter：{r['vllm_lora'][:80]!r} | HF 带 adapter：{r['hf_lora'][:80]!r}")
    (out / "probe_lora.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("== 通过：vLLM 能加载 adapter，且与 transformers 一致" if ok else
          "== 不通过：见上（adapter 没生效或与 transformers 不一致；把 probe_lora.json 发给 Claude）")
    return 0 if ok else 1


# ====================================================================== train
def schedule(slots, seed, epochs, batch):
    import numpy as np
    seq = []
    for ep in range(epochs):
        s = sorted(slots)
        perm = np.random.default_rng([int(seed), int(ep), 20261031]).permutation(len(s))
        seq += [s[i] for i in perm]
    return [seq[i:i + batch] for i in range(0, len(seq), batch)]


def lr_at(step, total, base, warmup):
    w = max(1, math.ceil(warmup * total))
    if step < w:
        return base * (step + 1) / w
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - w) / max(1, total - w))))


def cmd_train(a):
    import torch
    out = Path(a.out)
    if (out / "done.json").exists() and not a.force:
        log(f"{out} 已有 done.json，跳过（--force 重训）")
        return 0
    t0 = time.time()
    tok = load_tok(a.model_dir)
    rows = jl(a.samples)
    enc, dropped, modes = {}, [], {}
    for r in rows:
        p, c, _, _, mode = encode(tok, r)
        modes[mode] = modes.get(mode, 0) + 1
        if len(p) + len(c) > a.max_len:
            dropped.append(r["slot"])
            continue
        enc[r["slot"]] = (p, c, r["kind"])
    sched = schedule(list(enc), a.seed, a.epochs, a.batch)
    if a.max_steps:
        sched = sched[: a.max_steps]
    log(f"样本 {len(rows)}（修复 {sum(1 for x in enc.values() if x[2] == 'repair')}、锚 {sum(1 for x in enc.values() if x[2] == 'anchor')}；"
        f"丢弃过长 {len(dropped)}；助手部分取法 {modes}）；{a.epochs} 轮、每步 {a.batch} 个 → {len(sched)} 步")
    model, cls = load_model(a.model_dir)
    model.to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    pm, n_lora = add_lora(model, a.seed, a.targets, a.rank, a.alpha, a.dropout)
    ntr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    log(f"加载类 {cls}；LoRA 层 {n_lora} 个（{a.targets}），可训练参数 {ntr / 1e6:.1f}M；快速实现 {fast_path_info()}")
    pm.train()
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    hist, ntok, t1 = [], 0, time.time()
    for step, grp in enumerate(sched):
        lr = lr_at(step, len(sched), a.lr, a.warmup)
        for gpar in opt.param_groups:
            gpar["lr"] = lr
        tot, parts = 0.0, {"repair": [], "anchor": []}
        for s in grp:
            p, c, kind = enc[s]
            loss = -seq_logp(pm, p, c, "cuda", mean=True)
            (loss / len(grp)).backward()
            tot += float(loss.detach()) / len(grp)
            parts[kind].append(float(loss.detach()))
            ntok += len(p) + len(c)
        gn = torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step, "lr": lr, "loss": tot, "grad_norm": float(gn),
               "repair": sum(parts["repair"]) / len(parts["repair"]) if parts["repair"] else None,
               "anchor": sum(parts["anchor"]) / len(parts["anchor"]) if parts["anchor"] else None}
        hist.append(rec)
        if step % a.log_every == 0 or step == len(sched) - 1:
            el = time.time() - t1
            log(f"步 {step + 1}/{len(sched)} loss {tot:.4f}（修复 {rec['repair']}、锚 {rec['anchor']}）lr {lr:.2e} "
                f"|g| {rec['grad_norm']:.2f}；{ntok / max(el, 1e-9):.0f} token/秒，预计还要 {el / (step + 1) * (len(sched) - step - 1) / 60:.1f} 分钟")
    out.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(str(out))
    import peft
    import transformers
    meta = {"time": time.strftime("%Y-%m-%d %H:%M"), "samples": a.samples,
            "samples_sha256": hashlib.sha256(Path(a.samples).read_bytes()).hexdigest(), "model_dir": a.model_dir,
            "load_class": cls, "seed": a.seed, "targets": a.targets, "rank": a.rank, "alpha": a.alpha, "dropout": a.dropout,
            "lr": a.lr, "epochs": a.epochs, "batch": a.batch, "warmup": a.warmup, "max_len": a.max_len, "n_used": len(enc),
            "dropped": dropped, "n_steps": len(sched), "n_lora_layers": n_lora, "trainable": ntr, "wall_s": time.time() - t0,
            "tokens": ntok, "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                         "peft": peft.__version__}, "history": hist}
    (out / "train_log.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "done.json").write_text(json.dumps({"time": meta["time"], "n_steps": len(sched)}), encoding="utf-8")
    log(f"完成：{meta['wall_s'] / 60:.1f} 分钟 → {out}")
    print("== 完成")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--design", nargs="+", required=True, help="训练位置的设计文件（可多个，例如 retail 与 airline 各一个）")
    e.add_argument("--rollouts", nargs="+", required=True, help="π0 在这些位置上的续跑文件")
    e.add_argument("--executor", default="4B", help="π0 的名字（续跑文件里的 executor）")
    e.add_argument("--thr", type=float, default=0.20)
    e.add_argument("--min-n", type=int, default=40)
    e.add_argument("--anchor-src", nargs="*", default=[], help="π0 自己的完整对话 results.json（取成功对话作锚样本）")
    e.add_argument("--anchor-domain", nargs="*", default=[], help="与 --anchor-src 一一对应的领域")
    e.add_argument("--task-file", nargs="*", default=None, help="锚样本只取这些任务（训练任务清单，可多个）")
    e.add_argument("--task-domain", nargs="*", default=None, help="11-05：与 --task-file 一一对应的领域（按 (领域, 任务 id) 过滤锚样本）")
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--exclude-kinds", nargs="*", default=[], help="只来自这些来源的候选不作修复标签（例：rand）")
    e.add_argument("--anchor-mult", type=float, default=1.0, help="锚样本数 = 倍数 × 修复样本数")
    e.add_argument("--preamble", action="store_true", help="11-01：给不带文字的修复标签补前置文字（需要 --vllm-port 的 4B 服务）")
    e.add_argument("--vllm-port", type=int, default=0)
    e.add_argument("--served-name", default="Qwen3.5-4B")
    e.add_argument("--admissible", action="store_true", help="11-01 补充四：只在可执行的候选里选标签，并检查补出文字的一致性")
    e.add_argument("--judge-port", type=int, default=0, help="27B 服务（判断写操作是否已获确认、文字与调用是否一致）")
    e.add_argument("--judge-name", default="Qwen3.8-27B-FP8")
    e.add_argument("--min-final", type=int, default=30)
    e.add_argument("--adm-file", nargs="*", default=[], help="11-05：adm_tau2.py annotate 写的判定（给了就不现场判）")
    e.add_argument("--adm-rule", choices=["v1", "v2", "v3"], default="v1", help="11-05：现场判时的规则（见 adm_tau2.py）")
    e.add_argument("--template", action="store_true", help="11-01 补充五：补文字的第三级用模板（只用于可执行的标签）")
    e.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--samples", required=True)
    c.add_argument("--model-dir", required=True)
    c.add_argument("--max-len", type=int, default=16384)
    c.add_argument("--vllm-port", type=int, default=0)
    c.add_argument("--served-name", default="Qwen3.5-4B")
    c.add_argument("--n-tokenize", type=int, default=8)
    pl = sub.add_parser("probe-lora")
    pl.add_argument("--model-dir", required=True)
    pl.add_argument("--targets", choices=list(TARGETS), default="attn_mlp")
    pl.add_argument("--b-std", type=float, default=0.02)
    pl.add_argument("--bench-len", type=int, default=12000)
    pl.add_argument("--n-tok", type=int, default=12)
    pl.add_argument("--vllm-util", type=float, default=0.5)
    pl.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--samples", required=True)
    t.add_argument("--model-dir", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--targets", choices=list(TARGETS), default="attn_mlp")
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--alpha", type=int, default=32)
    t.add_argument("--dropout", type=float, default=0.05)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--warmup", type=float, default=0.03)
    t.add_argument("--max-len", type=int, default=16384)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--log-every", type=int, default=5)
    t.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "export" and len(a.anchor_src) != len(a.anchor_domain):
        ap.error("--anchor-src 与 --anchor-domain 要一一对应")
    return {"export": cmd_export, "check": cmd_check, "probe-lora": cmd_probe_lora, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
