"""tau 服务器：检查一个 vLLM 服务能否在"关思考"下正确做工具调用（先导 Q1 的一部分；2026-10-29 写）。
用法（tau 环境）：/root/autodl-tmp/envs/tau/bin/python probe_tools.py <端口> <模型名> 2>&1 | tee logs/probe_<端口>.log
每项打印 PASS / FAIL；最后一行 "全部通过" 或 "有失败"。"""
import json
import sys
import time

import httpx

port, name = sys.argv[1], sys.argv[2]
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
TOOLS = [{"type": "function", "function": {
    "name": "get_reservation_details",
    "description": "Get the details of a reservation.",
    "parameters": {"type": "object", "properties": {
        "reservation_id": {"type": "string", "description": "The reservation id, such as '8JX2WO'."}},
        "required": ["reservation_id"]}}}]
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def ask(messages, tools=None, n=1, temperature=0.7, seed=None):
    body = {"model": name, "messages": messages, "temperature": temperature, "max_tokens": 512, "n": n, **NOTHINK}
    if tools:
        body["tools"] = tools
    if seed is not None:
        body["seed"] = seed
    t0 = time.time()
    r = httpx.post(URL, json=body, timeout=120)
    r.raise_for_status()
    return r.json(), time.time() - t0


fails = 0


def check(label, cond, detail=""):
    global fails
    print(f"{'PASS' if cond else 'FAIL'}  {label}  {detail}")
    fails += 0 if cond else 1


sys_msg = {"role": "system", "content": "You are an airline customer service agent. Use tools when needed."}
user = {"role": "user", "content": "Hi, can you look up my reservation 8JX2WO?"}
# 1. 工具调用能否被解析成 tool_calls（20 次采样）
ok = think = 0
args_ok = 0
lat = []
for s in range(20):
    out, dt = ask([sys_msg, user], TOOLS, seed=s)
    lat.append(dt)
    m = out["choices"][0]["message"]
    tc = m.get("tool_calls") or []
    content = m.get("content") or ""
    think += ("<think>" in content) or bool(m.get("reasoning_content"))
    if tc and tc[0]["function"]["name"] == "get_reservation_details":
        ok += 1
        try:
            args_ok += json.loads(tc[0]["function"]["arguments"]).get("reservation_id") == "8JX2WO"
        except Exception:
            pass
check("工具调用被解析（≥ 19/20）", ok >= 19, f"{ok}/20")
check("参数正确（≥ 19/20）", args_ok >= 19, f"{args_ok}/20")
check("关思考：输出里没有思考内容", think == 0, f"含思考 {think}/20")
# 2. 工具结果之后能否用自然语言回答
tc_msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
    "name": "get_reservation_details", "arguments": json.dumps({"reservation_id": "8JX2WO"})}}]}
tool_msg = {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"reservation_id": "8JX2WO", "origin": "JFK",
                                                                        "destination": "SFO", "cabin": "economy"})}
out, _ = ask([sys_msg, user, tc_msg, tool_msg], TOOLS, temperature=0.0)
txt = out["choices"][0]["message"].get("content") or ""
check("工具结果之后用文字回答，提到 SFO", "SFO" in txt, repr(txt[:120]))
# 3. 每请求种子是否让采样可复现（同种子两次是否相同；vLLM 批处理下不保证逐位一致，只作描述）
a, _ = ask([sys_msg, {"role": "user", "content": "Tell me a random 6-letter word."}], seed=123)
b, _ = ask([sys_msg, {"role": "user", "content": "Tell me a random 6-letter word."}], seed=123)
print(f"INFO  同种子两次输出相同：{a['choices'][0]['message'].get('content') == b['choices'][0]['message'].get('content')}（只作描述）")
print(f"INFO  单请求延迟中位数 {sorted(lat)[len(lat) // 2]:.2f} 秒")
print("全部通过" if fails == 0 else f"有失败（{fails} 项）")
sys.exit(0 if fails == 0 else 1)
