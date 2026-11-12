#!/usr/bin/env bash
# tau 服务器：用本地 vLLM 服务跑 τ²-bench（2026-10-29 写；03:50 改为第三版）。需要先用 serve_tau.sh 起好服务：
#   Qwen3.5-4B 在 8001，Qwen3.5-9B 在 8002（9B 同时当本地用户模拟器）。
# 用法：bash run_tau2.sh <部署者 2B|4B|9B> <领域 airline|telecom|retail> <每任务次数> <标签> <用户 local|q27|flash> <任务列表文件>
#   用户 local = Qwen3.5-9B（端口 8002）；q27 = Qwen3.8-27B-FP8（端口 8003，10-29 先导 2 加）；flash = qwen3.8-flash API。
#   并发默认 16，可用环境变量 CONC 改（例：CONC=48 bash run_tau2.sh …）。
#   步数上限默认 100，可用环境变量 MAXSTEPS 改（10-29 加；先导 2 用 200，见下面第 2 条）。
#   用户重复终止默认关；REPEAT_STOP=3 时用 tau2_ext.py 包装 tau2（用户连续 3 次同一句话 = 用户结束；偏离 τ²，见 tau2_ext.py）。
#   例：bash run_tau2.sh 4B airline 2 p_4b_air_local local ids_airline.txt
# 结果：/root/paper/repair_budget/tau2_pilot/tau2-bench/data/simulations/<标签>/results.json
# 日志：/root/paper/repair_budget/tau2_pilot/logs/run_<标签>.log
#
# 与 τ² 默认设置的不同（冒烟后定下；都要在论文里写明）：
#  1) --communicate-scorer substring：默认用 gpt-5.5 做语义判定，服务器连不上 OpenAI（冒烟任务 3 卡 30 分钟的原因）。
#     改用 τ² 原版的子串匹配，确定、不调用模型。
#  2) --max-steps 100（默认 200）：冒烟中正常对话 14–42 条消息；任务 1 是部署者与贪心用户互相重复到 200 步。
#     10-29 更正：先导 1 里 telecom 的成功对话有不少接近 100 步（flash 用户 41 个成功里 11 个 ≥ 80 条消息，最长 96），
#     100 步对 telecom 太紧；先导 2 改用 MAXSTEPS=200（= τ² 默认），事后用 posthoc_tau2.py --cap 100 还能得到上限 100 的结果。
#  3) 每次生成最多 2048 token、单次请求 180 秒超时。
#  4) 部署者温度 0.7（与第 4、5 篇一致），用户贪心（温度 0）；两者都关思考。
#  5) 用户 flash（只作 Q3 的参照）：qwen3.8-flash，关思考。密钥从环境变量 QWEN_API_KEY 读，脚本里只传给 litellm，不打印。
set -u
[ $# -eq 6 ] || { echo "用法：bash run_tau2.sh <2B|4B|9B|名字@端口> <airline|telecom|retail> <每任务次数> <标签> <local|q27|flash> <任务列表文件>"; exit 2; }
CONC=${CONC:-16}
MAXSTEPS=${MAXSTEPS:-100}
REPEAT_STOP=${REPEAT_STOP:-0}
SEED=${SEED:-300}          # 10-31 加：τ² 的 --seed。先导 1–5 都是 300；探针收新失败用 301，正式池用 302（同一种子会重演同样的对话）
DEP=$1; DOM=$2; TR=$3; TAG=$4; USR=$5; IDS=$6
HERE=/root/paper/repair_budget/tau2_pilot
ENV=/root/autodl-tmp/envs/tau
case $DEP in
  4B) AMODEL=Qwen3.5-4B; APORT=8001 ;;
  9B) AMODEL=Qwen3.5-9B; APORT=8002 ;;
  2B) AMODEL=Qwen3.5-2B; APORT=8004 ;;       # 10-30 加（先导 3）
  *@*) AMODEL=${DEP%@*}; APORT=${DEP#*@} ;;   # 11-05 加（主实验端到端）：带 LoRA 的服务，写成 <adapter 名>@<端口>，例 4B-A@8001
  *) echo "[停下] 部署者只能是 2B、4B、9B 或 <adapter 名>@<端口>"; exit 2 ;;
esac
[ -f "$HERE/$IDS" ] || { echo "[停下] 找不到任务列表 $HERE/$IDS"; exit 2; }
mapfile -t TASK_IDS < <(grep -v '^\s*$' "$HERE/$IDS")
[ ${#TASK_IDS[@]} -gt 0 ] || { echo "[停下] 任务列表是空的"; exit 2; }
curl -s -m 3 "http://127.0.0.1:$APORT/v1/models" >/dev/null || { echo "[停下] 端口 $APORT 上没有 vLLM 服务"; exit 2; }
NOTHINK='"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}'
AARGS="{\"api_base\": \"http://127.0.0.1:$APORT/v1\", \"api_key\": \"EMPTY\", \"temperature\": 0.7, \"max_tokens\": 2048, \"timeout\": 180, $NOTHINK}"
case $USR in
  local)
    curl -s -m 3 "http://127.0.0.1:8002/v1/models" >/dev/null || { echo "[停下] 端口 8002 上没有 vLLM 服务（本地用户）"; exit 2; }
    UMODEL="hosted_vllm/Qwen3.5-9B"
    UARGS="{\"api_base\": \"http://127.0.0.1:8002/v1\", \"api_key\": \"EMPTY\", \"temperature\": 0.0, \"max_tokens\": 2048, \"timeout\": 180, $NOTHINK}" ;;
  q27)
    curl -s -m 3 "http://127.0.0.1:8003/v1/models" >/dev/null || { echo "[停下] 端口 8003 上没有 vLLM 服务（Qwen3.8-27B-FP8 用户）"; exit 2; }
    UMODEL="hosted_vllm/Qwen3.8-27B-FP8"
    UARGS="{\"api_base\": \"http://127.0.0.1:8003/v1\", \"api_key\": \"EMPTY\", \"temperature\": 0.0, \"max_tokens\": 2048, \"timeout\": 180, $NOTHINK}" ;;
  flash)
    [ -n "${QWEN_API_KEY:-}" ] || { echo "[停下] 没有设置环境变量 QWEN_API_KEY"; exit 2; }
    export OPENAI_API_KEY="$QWEN_API_KEY"      # 只给 litellm 用；不打印、不写进命令行参数
    UMODEL="openai/qwen3.8-flash"
    UARGS="{\"api_base\": \"https://maas.qianwenaiapi.com/compatible-mode/v1\", \"temperature\": 0.0, \"max_tokens\": 2048, \"timeout\": 180, \"extra_body\": {\"enable_thinking\": false}}" ;;
  *) echo "[停下] 用户只能是 local、q27 或 flash"; exit 2 ;;
esac
mkdir -p "$HERE/logs"
cd "$HERE/tau2-bench" || exit 2
echo "部署者 $AMODEL（端口 $APORT，温度 0.7）；用户 $UMODEL（温度 0）；$DOM，${#TASK_IDS[@]} 个任务 × $TR 次；并发 $CONC；步数上限 $MAXSTEPS；用户重复终止 $REPEAT_STOP（0=关）；种子 $SEED；标签 $TAG"
if [ "$REPEAT_STOP" -gt 0 ]; then
  [ -f "$HERE/tau2_ext.py" ] || { echo "[停下] REPEAT_STOP>0 需要 $HERE/tau2_ext.py"; exit 2; }
  TAU2=("$ENV/bin/python" "$HERE/tau2_ext.py" --repeat-stop "$REPEAT_STOP" --)
else
  TAU2=("$ENV/bin/tau2")
fi
TZ=Asia/Shanghai date
"${TAU2[@]}" run --domain "$DOM" \
  --agent-llm "hosted_vllm/$AMODEL" --agent-llm-args "$AARGS" \
  --user-llm "$UMODEL" --user-llm-args "$UARGS" \
  --task-ids "${TASK_IDS[@]}" --num-trials "$TR" --max-concurrency "$CONC" --seed "$SEED" \
  --max-steps "$MAXSTEPS" --communicate-scorer substring \
  --save-to "$TAG" 2>&1 | tee "$HERE/logs/run_$TAG.log"
rc=${PIPESTATUS[0]}
TZ=Asia/Shanghai date
echo "退出码 $rc；结果文件："; ls -la "$HERE/tau2-bench/data/simulations/$TAG/" 2>&1 | head
exit $rc
