#!/usr/bin/env bash
# tau 服务器：在后台起一个 vLLM 服务并等到可用（2026-10-29 写）。需要挂卡。
# 用法：bash serve_tau.sh <模型名 Qwen3.5-4B|Qwen3.5-9B> <端口> <显存比例> <工具调用解析器>
#   例：bash serve_tau.sh Qwen3.5-9B 8002 0.55 qwen3_coder
#        bash serve_tau.sh Qwen3.5-4B 8001 0.25 qwen3_coder
# 日志：/root/paper/repair_budget/tau2_pilot/logs/vllm_<端口>.log；停止：kill $(cat logs/vllm_<端口>.pid)
# 10-29：加 --max-num-seqs 256（默认 1024 超过 4B 在 0.25 显存下的 Mamba 缓存块数 491，启动报错；先导并发不超过 64，256 足够）
set -u
[ $# -eq 4 ] || { echo "用法：bash serve_tau.sh <模型名> <端口> <显存比例> <解析器>"; exit 2; }
NAME=$1; PORT=$2; UTIL=$3; PARSER=$4
HERE=/root/paper/repair_budget/tau2_pilot
ENV=/root/autodl-tmp/envs/tau
MODEL=/root/autodl-tmp/models/$NAME
LOG=$HERE/logs/vllm_$PORT.log
mkdir -p "$HERE/logs"
[ -f "$MODEL/config.json" ] || { echo "[停下] 找不到模型 $MODEL"; exit 2; }
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
export TMPDIR=/root/autodl-tmp/tmp
# 10-29 加：系统盘只剩约 3 GB，vLLM / triton / torch 的编译缓存默认写在 ~/.cache，改到数据盘
export XDG_CACHE_HOME=/root/autodl-tmp/cache VLLM_CACHE_ROOT=/root/autodl-tmp/cache/vllm TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
# 10-29 加：AutoDL 设置的 OMP_NUM_THREADS 不合法，vLLM 里 torch.set_num_threads 会报错退出（第 5 篇在 14b2 上遇到过同一问题）。
# 不是正整数就改成 4；MKL 线程层设为 GNU。只影响 CPU 线程数，不影响结果。原值打印出来备查。
echo "原 OMP_NUM_THREADS='${OMP_NUM_THREADS-（未设置）}'，原 MKL_THREADING_LAYER='${MKL_THREADING_LAYER-（未设置）}'"
case "${OMP_NUM_THREADS-}" in ''|*[!0-9]*|0) export OMP_NUM_THREADS=4 ;; esac
export MKL_THREADING_LAYER=GNU
echo "现 OMP_NUM_THREADS=$OMP_NUM_THREADS，MKL_THREADING_LAYER=$MKL_THREADING_LAYER"
# 10-29 加：FlashInfer 的 top-k/top-p 采样核要现场用 nvcc 编译，而系统的 /usr/local/cuda/bin/nvcc 太旧
# （报 "Unknown option '--compress-mode=size'"）。关掉 FlashInfer 采样器，改用 PyTorch 实现（同一采样分布，只是实现不同）。
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_WORKSPACE_BASE=/root/autodl-tmp/cache      # 万一还有别的 FlashInfer 编译，缓存也放数据盘
echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER；系统 nvcc：$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' || echo 无)"
"$ENV/bin/python" -c "import torch, sys; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else '无', '| torch', torch.__version__); sys.exit(0 if ok else 1)" \
  || { echo "[停下] torch 看不到 GPU（没挂卡，或驱动版本太旧不支持 cu130；把 nvidia-smi 的输出发给 Claude）"; exit 2; }
nohup "$ENV/bin/vllm" serve "$MODEL" --served-model-name "$NAME" --port "$PORT" \
  --gpu-memory-utilization "$UTIL" --max-model-len 32768 --enable-prefix-caching --max-num-seqs 256 \
  --enable-auto-tool-choice --tool-call-parser "$PARSER" > "$LOG" 2>&1 &
PID=$!; echo $PID > "$HERE/logs/vllm_$PORT.pid"
echo "已启动 vLLM（PID $PID，模型 $NAME，端口 $PORT，解析器 $PARSER），等待就绪（最多 10 分钟）……"
for i in $(seq 1 120); do
  if ! kill -0 $PID 2>/dev/null; then echo "[停下] vLLM 进程已退出，日志最后 40 行："; tail -40 "$LOG"; exit 1; fi
  if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "$NAME"; then echo "就绪（约 $((i*5)) 秒）"; exit 0; fi
  sleep 5
done
echo "[停下] 10 分钟内没有就绪，日志最后 40 行："; tail -40 "$LOG"; exit 1
