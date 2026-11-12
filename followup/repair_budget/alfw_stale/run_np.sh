#!/usr/bin/env bash
# 第 6 篇实验 B（ALFWorld 无规划器后继 S-np）：后台运行。14b2 上，在 /root/paper/repair_budget/alfw_stale 下运行。
# 登记：D:\Paper\repair_budget\登记_ALFWorld无规划器后继_20261110.md
#
#   bash run_np.sh labels    前台、只用 CPU：np_labels.py（两条自检 + 写 S-np 训练文件；几分钟）
#   bash run_np.sh all       后台：训练 S-np_s11 → S-np_s12 → drift → full（两个模型）；约 8 卡时
#   bash run_np.sh status | log | kill
#
# Python：conda 环境 llm（与 P0、A 层相同）。代码只读复用 /root/paper/repair_training。
# 训练输出：$NP_RT_OUT（默认 /root/autodl-tmp/rt_runs/out_np；adapters/S-np_s11、S-np_s12）
# 续跑输出：$NP_STALE_OUT（默认 /root/autodl-tmp/stale_runs/out_np；drift.jsonl、full.jsonl）
# 任何一步失败，后面的不跑；每一步都可断点续跑（训练已完成会跳过；续跑从断点接着跑）：重跑 bash run_np.sh all。
set -uo pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p logs
PY="${PY:-/root/miniconda3/envs/llm/bin/python}"
[ -x "$PY" ] || PY="$(command -v python)"
LOG=logs/_np.log
PIDF=logs/_np.pid
is_running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
export PYTHONHASHSEED=0 TZ=Asia/Shanghai
GPU="${ALFW_GPU:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export RT_CODE="${RT_CODE:-/root/paper/repair_training}"
export NP_RT_OUT="${NP_RT_OUT:-/root/autodl-tmp/rt_runs/out_np}"
export NP_STALE_OUT="${NP_STALE_OUT:-/root/autodl-tmp/stale_runs/out_np}"
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
    echo "--- GPU ---"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null
    exit 0 ;;
log) exec tail -n 40 -f "$LOG" ;;
kill)
    if is_running; then P=$(cat "$PIDF"); pkill -TERM -g "$P" 2>/dev/null || kill -TERM "$P"; rm -f "$PIDF"; echo "已停止进程组 $P"
    else echo "没有正在运行的任务"; fi
    exit 0 ;;
labels) exec "$PY" np_labels.py --out "$NP_RT_OUT" ;;
all) ;;
*) sed -n '2,12p' "$0"; exit 2 ;;
esac
if is_running; then echo "已有任务在运行（pid=$(cat "$PIDF")）。bash run_np.sh status 查看。"; exit 1; fi
if [ ! -f "$NP_RT_OUT/train_S-np.jsonl" ]; then echo "[停下] 还没有 $NP_RT_OUT/train_S-np.jsonl；先 bash run_np.sh labels"; exit 1; fi
if nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "[停下] 第 $GPU 张卡上还有别的进程："; nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv
    echo "换一张空卡（ALFW_GPU=1 bash run_np.sh all），或先停掉那个进程。"; exit 1
fi
mkdir -p "$NP_STALE_OUT"
RUNNER=logs/_np_runner.sh
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
step() { echo "=== \$(date '+%F %T') \$1 ==="; }
step "开始（Python $PY；GPU $GPU；训练输出 $NP_RT_OUT；续跑输出 $NP_STALE_OUT）"
for s in 11 12; do
    step "训练 S-np_s\$s"
    (cd "$RT_CODE" && "$PY" rt_train.py --group S-np --seed \$s --out-dir "$NP_RT_OUT"); rc=\$?
    if [ \$rc -ne 0 ]; then step "【停下】训练 S-np_s\$s 失败，退出码 \$rc"; exit \$rc; fi
done
step "把 P0 的 adapter 以软链接放进 $NP_RT_OUT/adapters（stale_alfw.py 的 check 要求 P0 的 adapter 都在同一目录；只读，不复制）"
P0A="\${P0_RT_OUT:-/root/autodl-tmp/rt_runs/out_rt}/adapters"
if [ ! -d "\$P0A" ]; then step "【停下】找不到 \$P0A"; exit 1; fi
for d in "\$P0A"/*; do
    n=\$(basename "\$d")
    if [ ! -e "$NP_RT_OUT/adapters/\$n" ]; then ln -s "\$d" "$NP_RT_OUT/adapters/\$n"; fi
done
ls -l "$NP_RT_OUT/adapters"
step "续跑：drift 与 full（S-np_s11、S-np_s12）"
RT_OUT="$NP_RT_OUT" "$PY" stale_alfw.py run --stages drift,full --models S-np_s11,S-np_s12 --out "$NP_STALE_OUT"; rc=\$?
if [ \$rc -ne 0 ]; then step "【停下】续跑失败，退出码 \$rc；重跑 bash run_np.sh all 会接着跑"; exit \$rc; fi
step "全部完成：下一步拉回 $NP_STALE_OUT/full.jsonl、drift.jsonl 与 $NP_RT_OUT/trainsets_np.json、adapters/*/train_log.json"
EOF
chmod +x "$RUNNER"
echo "" >> "$LOG"
setsid nohup bash "$RUNNER" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "已在后台启动（pid $(cat "$PIDF")）。bash run_np.sh log 看日志，bash run_np.sh status 看状态。"
