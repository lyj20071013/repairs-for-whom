#!/usr/bin/env bash
# 第 6 篇 A 层（ALFWorld 标签过期）：后台运行（setsid + nohup，关掉 SSH 不影响）。14b2 上，在 /root/paper/repair_budget/alfw_stale 下运行。
#
#   bash run_stale.sh check      不用 GPU：核对 6.9 池数据、P0 的 adapter 与检查点，列出续跑数（前台运行，几秒）
#   bash run_stale.sh p1         优先级 1：drift → free_b → full（S_s11、S_s12）→ reverse（S_s11）；约 13 卡时
#   bash run_stale.sh p2         优先级 2：two（A_s11、R_s11、U_s11、S-plan_s11、DPO-S_s11）；约 6.5 卡时
#   bash run_stale.sh all        p1 跑完接着跑 p2（同一个后台进程；10-31 加）；约 20 卡时
#   bash run_stale.sh analyze    只用 CPU：写 $STALE_OUT/report_stale.md、analysis_stale.json（前台运行）
#   bash run_stale.sh status | log | kill
#
# Python：P0 用的 conda 环境 llm（vLLM 的 LoRA 引擎在 P0 已验证）；不要用 τ² 的 venv。
# 代码只读复用 /root/paper/repair_training（RT_CODE）；P0 的 adapter 从 RT_OUT 读（默认 /root/autodl-tmp/rt_runs/out_rt）。
# 输出：$STALE_OUT（默认 /root/autodl-tmp/stale_runs/out_stale）；日志 logs/_stale.log。每个阶段都可断点续跑：重跑同一命令接着跑。
set -uo pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p logs
PY="${PY:-/root/miniconda3/envs/llm/bin/python}"
[ -x "$PY" ] || PY="$(command -v python)"
LOG=logs/_stale.log
PIDF=logs/_stale.pid
is_running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
export PYTHONHASHSEED=0 TZ=Asia/Shanghai
# 10-31 加（双卡）：A 层固定用 ALFW_GPU 这张卡（默认 0）；τ² 用另一张（TAU_GPU=1）
GPU="${ALFW_GPU:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export RT_CODE="${RT_CODE:-/root/paper/repair_training}"
export STALE_OUT="${STALE_OUT:-/root/autodl-tmp/stale_runs/out_stale}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
export ALFW_TMP="${ALFW_TMP:-/root/autodl-tmp/alfw_tmp}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/autodl-tmp/cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/autodl-tmp/cache/inductor}"
case "${OMP_NUM_THREADS-4}" in ''|0|*[!0-9]*) export OMP_NUM_THREADS=4 ;; esac
export MKL_THREADING_LAYER=GNU

case "${1:-}" in
status)
    if is_running; then echo "运行中，pid=$(cat "$PIDF")"; else echo "未运行"; fi
    echo "--- 最近日志（$LOG）---"; tail -n 25 "$LOG" 2>/dev/null || echo "（无日志）"
    echo "--- GPU ---"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo "（没有 GPU）"
    exit 0 ;;
log) exec tail -n 40 -f "$LOG" ;;
kill)
    if is_running; then P=$(cat "$PIDF"); pkill -TERM -g "$P" 2>/dev/null || kill -TERM "$P"; rm -f "$PIDF"; echo "已停止进程组 $P"
    else echo "没有正在运行的任务"; fi
    exit 0 ;;
check)   exec "$PY" stale_alfw.py check ;;
analyze) exec "$PY" stale_analyze.py ;;
p1) ARGS="run --stages drift,free_b,full,reverse" ;;
p2) ARGS="run --stages two" ;;
all) ARGS="run --stages drift,free_b,full,reverse,two" ;;
*) sed -n '2,13p' "$0"; exit 2 ;;
esac
if is_running; then echo "已有任务在运行（pid=$(cat "$PIDF")）。bash run_stale.sh status 查看。"; exit 1; fi
if nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "[停下] 第 $GPU 张卡上还有别的进程（例如 τ² 的 vLLM 服务）："; nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv
    echo "先停掉（τ²：kill \$(cat /root/paper/repair_budget/tau2_pilot/logs/vllm_8001.pid) 等），再运行。"; exit 1
fi
mkdir -p "$STALE_OUT"
echo "" >> "$LOG"
setsid nohup bash -c "echo \"=== \$(date '+%F %T') 开始：$1（Python：$PY；输出 $STALE_OUT；GPU $GPU）===\"; \"$PY\" stale_alfw.py $ARGS; rc=\$?; echo \"=== \$(date '+%F %T') 结束：$1，退出码 \$rc ===\"" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "已在后台启动 $1（pid $(cat "$PIDF")）。bash run_stale.sh log 看日志，bash run_stale.sh status 看状态。"
