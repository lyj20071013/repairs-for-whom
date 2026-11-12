#!/usr/bin/env bash
# 第 5 篇先导 P0：后台运行（setsid + nohup，关掉 SSH 或本地电脑都不影响）。在项目根 /root/paper/repair_training 下运行。
#
#   bash run_bg.sh selftest        无 GPU 自检（模拟环境 + 模拟模型；几分钟）
#   bash run_bg.sh trainselftest   训练自检（真实分词器 + 极小随机模型，CPU；无卡模式也能跑）
#   bash run_bg.sh apicheck        flash 接口连通性（不打印密钥）
#   bash run_bg.sh smoke           冒烟：真实 7B、极小规模走完全流程（out_rt_smoke/，结果不用）
#   bash run_bg.sh data            P0 数据阶段：games -> collect -> positions -> propose -> design -> label -> select
#                                  -> fresh -> thoughts -> export（out_rt/）
#   bash run_bg.sh train           依次训练登记的 8 个模型（每个在单独的子进程里）
#   bash run_bg.sh formatcheck     格式检查（不通过会停下，见 out_rt/formatcheck.json）
#   bash run_bg.sh evaluate        端到端 + 接管（要求格式检查已通过）
#   bash run_bg.sh analyze         分析 -> out_rt/report.md、out_rt/analysis.json
#   bash run_bg.sh p0              data -> train -> formatcheck -> evaluate -> analyze 一口气跑完
#   bash run_bg.sh status | log | kill
#
# 其余参数原样传给 rt.py（例如 bash run_bg.sh evaluate --force）。任何一步失败，后面的步骤都不会执行；
# 每个阶段都可断点续跑：重跑同一命令会从断点接着跑。
set -uo pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p logs
PY="$(command -v python || command -v python3)"
LOG=logs/_rt.log
PIDF=logs/_rt.pid
is_running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

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
""|-h|--help) sed -n '2,20p' "$0"; exit 0 ;;
esac

STAGE="$1"; shift
case "$STAGE" in
selftest)      CMD="\"$PY\" selftest.py" ;;
trainselftest) CMD="\"$PY\" rt_train.py --selftest" ;;
apicheck|smoke|data|train|formatcheck|evaluate|analyze|p0) CMD="\"$PY\" rt.py --stage $STAGE" ;;
*) echo "未知阶段：$STAGE"; sed -n '2,20p' "$0"; exit 2 ;;
esac
if [ "$#" -gt 0 ]; then printf -v EXTRA ' %q' "$@"; CMD="$CMD$EXTRA"; fi
if is_running; then echo "已有任务在运行（pid=$(cat "$PIDF")）。bash run_bg.sh status 查看，或 bash run_bg.sh kill 停止。"; exit 1; fi
case "$STAGE" in apicheck|data|p0|smoke)
    if [ -z "${QWEN_API_KEY:-}" ] && [ -z "${DASHSCOPE_API_KEY:-}" ]; then
        echo "没有设置 QWEN_API_KEY（flash 提议要用）。先 export QWEN_API_KEY=...（不要写进任何文件）再运行。"; exit 1; fi ;;
esac

RUNNER=logs/_rt_runner.sh
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
cd "$(pwd)" || exit 1
export PYTHONHASHSEED=0
export TZ=Asia/Shanghai
export ALFWORLD_DATA="\${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
export ALFW_TMP="\${ALFW_TMP:-/root/autodl-tmp/alfw_tmp}"
export HF_HOME="\${HF_HOME:-/root/autodl-tmp/hf_home}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-/root/autodl-tmp/pip_cache}"
export TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="\${VLLM_CACHE_ROOT:-/root/autodl-tmp/cache/vllm}"
export TRITON_CACHE_DIR="\${TRITON_CACHE_DIR:-/root/autodl-tmp/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="\${TORCHINDUCTOR_CACHE_DIR:-/root/autodl-tmp/cache/inductor}"
echo "[env] 原来的 OMP_NUM_THREADS='\${OMP_NUM_THREADS-（未设置）}' MKL_THREADING_LAYER='\${MKL_THREADING_LAYER-（未设置）}'"
case "\${OMP_NUM_THREADS-4}" in ''|0|*[!0-9]*) export OMP_NUM_THREADS=4 ;; esac
export MKL_THREADING_LAYER=GNU
echo "=== \$(TZ=Asia/Shanghai date '+%F %T') 开始：$STAGE（Python：$PY）==="
$CMD
rc=\$?
echo "=== \$(TZ=Asia/Shanghai date '+%F %T') 结束：$STAGE，退出码 \$rc ==="
exit \$rc
EOF
chmod +x "$RUNNER"
echo "" >> "$LOG"
setsid nohup bash "$RUNNER" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "已在后台启动 $STAGE（pid=$(cat "$PIDF")）。查看：bash run_bg.sh log（Ctrl-C 只退出查看）；状态：bash run_bg.sh status"
