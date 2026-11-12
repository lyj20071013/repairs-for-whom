#!/usr/bin/env bash
# 14b2：起一个带 LoRA 的 vLLM 服务（2026-10-31 写；第 6 篇 τ² 探针的 π1）。需要挂卡。与 serve_tau.sh 相同，只多了 LoRA。
# 用法：bash serve_lora.sh <基础模型名 Qwen3.5-4B> <端口> <显存比例> <adapter名>=<adapter目录> [<adapter名>=<目录> …]
#   例：bash serve_lora.sh Qwen3.5-4B 8001 0.30 4B-S1=/root/autodl-tmp/tau_runs/probe/adapter_s1
# 请求时用 adapter 名当模型名（pool_tau2.py 里设环境变量 TAU2_EXTRA_MODELS="4B-S1@8001"）；基础模型名照样可用（= π0）。
# 日志：/root/paper/repair_budget/tau2_pilot/logs/vllm_<端口>.log；停止：kill $(cat logs/vllm_<端口>.pid)
set -u
[ $# -ge 4 ] || { echo "用法：bash serve_lora.sh <基础模型名> <端口> <显存比例> <名字>=<目录> …"; exit 2; }
NAME=$1; PORT=$2; UTIL=$3; shift 3
HERE=/root/paper/repair_budget/tau2_pilot
ENV=/root/autodl-tmp/envs/tau
MODEL=/root/autodl-tmp/models/$NAME
LOG=$HERE/logs/vllm_$PORT.log
mkdir -p "$HERE/logs"
[ -f "$MODEL/config.json" ] || { echo "[停下] 找不到模型 $MODEL"; exit 2; }
LORA=()
for x in "$@"; do
  d=${x#*=}
  [ -f "$d/adapter_config.json" ] || { echo "[停下] $d 里没有 adapter_config.json"; exit 2; }
  LORA+=("$x")
done
if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then echo "[停下] 端口 $PORT 已被占用（已有服务？）"; exit 2; fi
# 10-31 加（双卡）：τ² 的服务固定用 TAU_GPU 这张卡（默认 0）；这张卡上若有 A 层（stale_alfw.py）就不启动
GPU="${TAU_GPU:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
for p in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  if ps -o args= -p "$p" 2>/dev/null | grep -q stale_alfw; then
    echo "[停下] 第 $GPU 张卡上在跑 A 层（pid $p）。双卡时 τ² 用另一张：先 export TAU_GPU=1"; exit 2
  fi
done
echo "τ² 服务用第 $GPU 张卡（TAU_GPU=${TAU_GPU:-未设置，默认 0}）"
export TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache VLLM_CACHE_ROOT=/root/autodl-tmp/cache/vllm TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
case "${OMP_NUM_THREADS-}" in ''|*[!0-9]*|0) export OMP_NUM_THREADS=4 ;; esac
export MKL_THREADING_LAYER=GNU VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=/root/autodl-tmp/cache
nohup "$ENV/bin/vllm" serve "$MODEL" --served-model-name "$NAME" --port "$PORT" \
  --gpu-memory-utilization "$UTIL" --max-model-len 32768 --enable-prefix-caching --max-num-seqs 256 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-lora --max-lora-rank 16 --max-loras "${#LORA[@]}" --lora-modules "${LORA[@]}" > "$LOG" 2>&1 &
PID=$!; echo $PID > "$HERE/logs/vllm_$PORT.pid"
echo "已启动 vLLM（PID $PID，基础模型 $NAME，端口 $PORT，LoRA：${LORA[*]}），等待就绪（最多 10 分钟）……"
for i in $(seq 1 120); do
  if ! kill -0 $PID 2>/dev/null; then echo "[停下] vLLM 进程已退出，日志最后 40 行："; tail -40 "$LOG"; exit 1; fi
  M=$(curl -s -m 3 "http://127.0.0.1:$PORT/v1/models")
  if echo "$M" | grep -q "\"$NAME\""; then
    for x in "${LORA[@]}"; do n=${x%%=*}; echo "$M" | grep -q "\"$n\"" || { echo "[停下] 服务起来了，但模型列表里没有 adapter $n"; exit 1; }; done
    echo "就绪（约 $((i*5)) 秒）；模型列表：$(echo "$M" | grep -o '"id":"[^"]*"' | tr '\n' ' ')"; exit 0
  fi
  sleep 5
done
echo "[停下] 10 分钟内没有就绪，日志最后 40 行："; tail -40 "$LOG"; exit 1
