#!/usr/bin/env bash
# 第 6 篇 τ² 探针（2026-10-31 写；登记 D:\Paper\repair_budget\登记_τ²探针_20261031.md）。14b2 上，在 /root/paper/repair_budget/tau2_pilot 下运行。
# Python：τ² 的 venv /root/autodl-tmp/envs/tau。小文件写在 tau2_pilot/probe/，adapter 写在 /root/autodl-tmp/tau_runs/probe/。
# 续跑的并发：环境变量 WORKERS（默认 16）、PROCS（默认 4）。10-31 实测 airline 对话很长、GPU 很闲，用 WORKERS=24 PROCS=6。
# 每一步都可以重跑（已完成的会跳过）。长的步骤在后台跑，日志 probe/logs/<步>.log，结尾打印 "=== 步 … 结束，退出码 …"。
#
#   bash run_probe.sh lora      （GPU，前台约 15–20 分钟）LoRA 能否训练、vLLM 能否加载；不需要任何 vLLM 服务
#   bash run_probe.sh collect   （后台）新的 2B 失败：retail 74 × 2、airline 50 × 2，种子 301。需要 27B（8003）与 2B（8004）
#   bash run_probe.sh design    （后台）任务对半分 → 训练 / 评价位置的设计文件（4B 提议）。需要 4B（8001）
#   bash run_probe.sh pi0       （后台）π0 = 4B：训练位置每分支 3 次、评价位置每分支 6 次、漂移（4B、4B#2）。需要 27B 与 4B
#   bash run_probe.sh pi0more   （后台）补充五：训练位置上可执行候选 + 自由选择补跑到每分支 6 次。需要 27B 与 4B
#   bash run_probe.sh export    （前台，CPU）选标签、导出训练样本；再用 4B 服务的 /tokenize 核对提示。需要 4B 服务还开着
#   bash run_probe.sh train     （后台，GPU）训 π1（LoRA）。要先停掉所有 vLLM 服务
#   bash run_probe.sh pi1       （后台）π1 先采漂移（几分钟），再在评价位置每分支 6 次。需要 27B 与带 adapter 的 4B（serve_lora.sh，8001）
#   bash run_probe.sh pi1check  （前台）漂移采完后看 π1 的输出是否正常（无效输出、工具调用比例）
#   bash run_probe.sh parity    （后台，GPU，约 15–25 分钟）vLLM 带 LoRA 与 transformers 在真实 τ² 提示上是否一致。要先停掉所有 vLLM 服务
#   bash run_probe.sh diag      （前台）π1 与 π0 的续跑行为比较（终止原因、工具错误、重复循环；不读 reward）
#   bash run_probe.sh gate      （前台，CPU）闸门"π1 变好"：评价位置上自由接手 π1 − π0（只读 free 分支）。全量续跑之后、analyze 之前
#   bash run_probe.sh analyze   （前台，CPU）写 probe/report_probe.md
#   bash run_probe.sh status    看各步进度
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY=/root/autodl-tmp/envs/tau/bin/python
OUT=probe
RUNS=/root/autodl-tmp/tau_runs/probe
M4=/root/autodl-tmp/models/Qwen3.5-4B
PI1="${PI1:-4B-S1}"                     # 11-01：π1 的名字、样本与 adapter 的标签可以改（重训时 PI1=4B-S2 TAG=s2）
TAG="${TAG:-s1}"
EPOCHS="${EPOCHS:-3}"
EXCLUDE="${EXCLUDE:-}"                  # export：只来自这些来源的候选不作标签（例：EXCLUDE=rand）
ANCHOR_MULT="${ANCHOR_MULT:-1}"         # export：锚样本数 = 倍数 × 修复样本数
SAMPLES="${SAMPLES:-$OUT/samples_$TAG.jsonl}"   # train / check 用的样本文件（S2 沿用 S1 的样本：SAMPLES=probe/samples_s1.jsonl）
LIMIT="${LIMIT:-0}"
PRE_ARGS=""                             # 11-01（补充三）：PREAMBLE=1 时 export 给不带文字的修复标签补前置文字（用 8001 上的 4B 服务）
[ "${PREAMBLE:-0}" = 1 ] && PRE_ARGS="--preamble --vllm-port 8001 --served-name Qwen3.5-4B"
# 11-01（补充四）：ADMISSIBLE=1 时只在可执行的候选里选标签、检查补出文字的一致性（用 8003 上的 27B 判断）
[ "${ADMISSIBLE:-0}" = 1 ] && PRE_ARGS="$PRE_ARGS --admissible --judge-port 8003 --judge-name Qwen3.8-27B-FP8 --min-final ${MIN_FINAL:-30}"
[ "${TEMPLATE:-0}" = 1 ] && PRE_ARGS="$PRE_ARGS --template"   # 11-01（补充五）：补文字第三级用模板
RETRY="${RETRY:-1}"                     # 0 = 不重跑出错的续跑（行为检查时用：超上下文的错误重跑也一样）                     # pi1：只跑每个评价设计文件按位置 id 排序后的前 LIMIT 个位置（0 = 全部；行为检查用）
mkdir -p "$OUT/logs" "$RUNS"
GPU="${TAU_GPU:-0}"                     # 10-31 加（双卡）：lora / train 两步直接用 GPU，固定在 TAU_GPU 这张卡
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
# 10-31 加：与 serve_tau.sh 相同的环境（lora 一步在 Python 里直接起 vLLM；不设的话 FlashInfer 采样核要现场用太旧的 nvcc 编译，引擎启动失败）
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=/root/autodl-tmp/cache
export TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache VLLM_CACHE_ROOT=/root/autodl-tmp/cache/vllm TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
case "${OMP_NUM_THREADS-}" in ''|*[!0-9]*|0) export OMP_NUM_THREADS=4 ;; esac
export MKL_THREADING_LAYER=GNU
need() { for p in "$@"; do curl -s -m 3 "http://127.0.0.1:$p/v1/models" >/dev/null || { echo "[停下] 端口 $p 上没有 vLLM 服务"; exit 2; }; done; }
bg() {  # bg <步名> <命令…>：后台运行，日志 probe/logs/<步名>.log
  local name=$1; shift
  nohup bash -c "echo \"=== \$(date '+%F %T') 步 $name 开始\"; $*; rc=\$?; echo \"=== \$(date '+%F %T') 步 $name 结束，退出码 \$rc\"" \
    >> "$OUT/logs/$name.log" 2>&1 < /dev/null &
  echo "已在后台启动 $name（pid $!）。看日志：tail -f $OUT/logs/$name.log"
}
TB=tau2-bench/data/simulations
case "${1:-}" in
lora)
  if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then echo "[停下] 第 $GPU 张卡上还有进程（双卡时先 export TAU_GPU=1）"; nvidia-smi; exit 1; fi
  exec "$PY" train_tau2.py probe-lora --model-dir "$M4" --targets "${TARGETS:-attn_mlp}" --out "$RUNS/lora_probe" ;;
collect)
  need 8003 8004
  bg collect "env CONC=20 MAXSTEPS=200 SEED=301 bash run_tau2.sh 2B retail 2 probe_2b_ret q27 ids_retail_det.txt; env CONC=20 MAXSTEPS=200 SEED=301 bash run_tau2.sh 2B airline 2 probe_2b_air q27 ids_airline.txt" ;;
design)
  need 8001
  "$PY" split_tasks.py PROBE retail ids_retail_det.txt "$OUT" || exit 1
  "$PY" split_tasks.py PROBE airline ids_airline.txt "$OUT" || exit 1
  C="cd tau2-bench; P=../pool_tau2.py; PY=$PY"
  bg design "$C; \
\$PY \$P design --src data/simulations/p4_2b_ret_q27/results.json data/simulations/probe_2b_ret/results.json --domain retail --deployer 2B --task-file ../$OUT/train_retail.txt --n-traj 24 --proposer 4B --max-steps 200 --workers 8 --out ../$OUT/design_train_ret.jsonl; \
\$PY \$P design --src data/simulations/p4_2b_air_q27/results.json data/simulations/probe_2b_air/results.json --domain airline --deployer 2B --task-file ../$OUT/train_airline.txt --n-traj 16 --proposer 4B --max-steps 200 --workers 8 --out ../$OUT/design_train_air.jsonl; \
\$PY \$P design --src data/simulations/p4_2b_ret_q27/results.json data/simulations/probe_2b_ret/results.json --domain retail --deployer 2B --task-file ../$OUT/eval_retail.txt --n-traj 12 --proposer 4B --max-steps 200 --workers 8 --out ../$OUT/design_eval_ret.jsonl; \
\$PY \$P design --src data/simulations/p4_2b_air_q27/results.json data/simulations/probe_2b_air/results.json --domain airline --deployer 2B --task-file ../$OUT/eval_airline.txt --n-traj 8 --proposer 4B --max-steps 200 --workers 8 --out ../$OUT/design_eval_air.jsonl" ;;
pi0)
  need 8003 8001
  C="cd tau2-bench; P=../pool_tau2.py; PY=$PY"
  R="--executors 4B --user q27 --max-steps 200 --procs ${PROCS:-4} --workers ${WORKERS:-16} --retry-errors"
  bg pi0 "$C; \
for d in train_ret train_air; do \$PY \$P rollout --design ../$OUT/design_\$d.jsonl --k 3 $R --out ../$OUT/ro_\$d.jsonl; done; \
for d in eval_ret eval_air; do \$PY \$P rollout --design ../$OUT/design_\$d.jsonl --k 6 $R --out ../$OUT/ro_\$d.jsonl; done; \
cat ../$OUT/design_eval_ret.jsonl ../$OUT/design_eval_air.jsonl > ../$OUT/design_eval_all.jsonl; \
\$PY \$P drift --design ../$OUT/design_eval_all.jsonl --models 4B,4B#2 --m 10 --out ../$OUT/drift_eval.jsonl" ;;
pi0more)    # 11-01 加（补充五）：训练位置上"可执行的修复候选 + 自由选择"补跑到每分支 6 次（已有的 k = 0–2 跳过）。需要 27B 与 4B
  need 8003 8001
  ADM="${ADM_FROM:-$OUT/samples_s4b.jsonl.admissible.jsonl}"
  [ -f "$ADM" ] || { echo "[停下] 找不到 $ADM"; exit 2; }
  for d in ret air; do "$PY" adm_design.py --design $OUT/design_train_$d.jsonl --adm "$ADM" --out $OUT/design_train_${d}_adm.jsonl || exit 1; done
  C="cd tau2-bench; P=../pool_tau2.py; PY=$PY"
  R="--executors 4B --user q27 --max-steps 200 --procs ${PROCS:-4} --workers ${WORKERS:-16} --retry-errors"
  bg pi0more "$C; for d in ret air; do \$PY \$P rollout --design ../$OUT/design_train_\${d}_adm.jsonl --k 6 $R --out ../$OUT/ro_train_\$d.jsonl; done" ;;
export)
  need 8001
  [ "${ADMISSIBLE:-0}" = 1 ] && need 8003
  cd tau2-bench || exit 1
  "$PY" ../train_tau2.py export --design ../$OUT/design_train_ret.jsonl ../$OUT/design_train_air.jsonl \
      --rollouts ../$OUT/ro_train_ret.jsonl ../$OUT/ro_train_air.jsonl --executor 4B --thr 0.20 --min-n ${MIN_N:-40} \
      --anchor-src data/simulations/p4_4b_ret_q27/results.json data/simulations/p2_4b_air_q27/results.json \
      --anchor-domain retail airline --task-file ../$OUT/train_retail.txt ../$OUT/train_airline.txt \
      --seed 1 --exclude-kinds $EXCLUDE --anchor-mult $ANCHOR_MULT $PRE_ARGS --out ../$OUT/samples_$TAG.jsonl 2>&1 | tee ../$OUT/logs/export_$TAG.log
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "[停下] export 没有通过，见 $OUT/logs/export_$TAG.log"; exit 1; }
  cd .. || exit 1
  "$PY" train_tau2.py check --samples $OUT/samples_$TAG.jsonl --model-dir "$M4" --max-len 16384 --vllm-port 8001 --served-name Qwen3.5-4B 2>&1 | tee -a $OUT/logs/export_$TAG.log
  exit "${PIPESTATUS[0]}" ;;
train)
  if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then echo "[停下] 第 $GPU 张卡上还有进程（先停掉这张卡上的 vLLM 服务；双卡时确认 TAU_GPU=1）"; nvidia-smi; exit 1; fi
  bg train "$PY train_tau2.py train --samples $SAMPLES --model-dir $M4 --out $RUNS/adapter_$TAG --seed 1 --targets ${TARGETS:-attn_mlp} --epochs $EPOCHS --batch 8 --lr ${LR:-1e-4}" ;;
pi1)
  need 8003 8001
  curl -s -m 3 http://127.0.0.1:8001/v1/models | grep -q "\"$PI1\"" || { echo "[停下] 8001 上没有 adapter $PI1（用 serve_lora.sh 起）"; exit 2; }
  C="cd tau2-bench; P=../pool_tau2.py; PY=$PY; export TAU2_EXTRA_MODELS=$PI1@8001"
  R="--executors $PI1 --user q27 --max-steps 200 --procs ${PROCS:-4} --workers ${WORKERS:-16}"
  [ "$RETRY" = 1 ] && R="$R --retry-errors"
  bg pi1 "$C; \
\$PY \$P drift --design ../$OUT/design_eval_all.jsonl --models $PI1 --m 10 --out ../$OUT/drift_eval.jsonl; \
for d in eval_ret eval_air; do \$PY \$P rollout --design ../$OUT/design_\$d.jsonl --k 6 $R --limit-pos $LIMIT --out ../$OUT/ro_\$d.jsonl; done" ;;
parity)     # 11-01 15:xx 加：vLLM 带 LoRA 与 transformers + PEFT 在真实 τ² 提示（训练样本）上是否一致；T1 的加严核对。要先停掉这张卡上所有 vLLM 服务
  if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then echo "[停下] 第 $GPU 张卡上还有进程（先停掉 8001 / 8003）"; nvidia-smi; exit 1; fi
  # 11-01 16:xx：PARITY_ADAPTERS / PARITY_ABLATE（空 = 不做消融）/ PARITY_GEN / PARITY_OUT 可改；默认 = 第一次核对（S2、S1）
  PA="${PARITY_ADAPTERS:-4B-S2=$RUNS/adapter_s2 4B-S1=$RUNS/adapter_s1}"; ABL="${PARITY_ABLATE-4B-S2}"
  bg parity "$PY lora_parity.py --samples $OUT/samples_s1.jsonl --model-dir $M4 --adapters $PA ${ABL:+--ablate $ABL} --n ${N:-16} --gen ${PARITY_GEN:-48} --out ${PARITY_OUT:-$OUT/lora_parity.json}" ;;
diag)       # 11-01 加：π1 与 π0 的行为比较（不读 reward）；LIMIT>0 时只比较两边都有的 (位置, 候选)
  exec "$PY" pi1_diag.py --rollouts $OUT/ro_eval_ret.jsonl $OUT/ro_eval_air.jsonl --models 4B $PI1 --same-pos ;;
pi1check)   # 10-31 加：π1 的漂移跑完后（几分钟）先看一眼：无效输出比例、工具调用比例、与 π0 的差别；不看任何续跑结果
  exec "$PY" -c "
import json, collections
R=[json.loads(l) for l in open('$OUT/drift_eval.jsonl') if l.strip()]
for m in sorted({(r['model'], r['rep']) for r in R}):
    x=[r for r in R if (r['model'], r['rep'])==m]
    print(m, '样本', len(x), '无效', round(sum(r['canon']=='INVALID' for r in x)/len(x),3), '工具调用', round(sum(r['is_tool'] for r in x)/len(x),3))
" ;;
gate)       # 11-01 加（补充三）：闸门"π1 变好"——只读自由选择分支的 reward；π1 全量续跑完成后运行
  exec "$PY" gate_free.py --rollouts $OUT/ro_eval_ret.jsonl $OUT/ro_eval_air.jsonl --pi0 4B --pi1 $PI1 --out $OUT/gate_$PI1.json ;;
analyze)
  exec "$PY" probe_analyze.py --design $OUT/design_eval_ret.jsonl $OUT/design_eval_air.jsonl \
      --rollouts $OUT/ro_eval_ret.jsonl $OUT/ro_eval_air.jsonl --pi0 4B --pi1 $PI1 --k 6 --drift $OUT/drift_eval.jsonl \
      --out $OUT/report_probe.md ;;
status)
  for f in $OUT/logs/*.log; do [ -f "$f" ] && { echo "--- $f"; grep -a "=== \|完成 [0-9]*/\|== 完成\|停下\|出错" "$f" | tail -4; }; done
  ls -la $OUT/*.jsonl 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>/dev/null ;;
*) sed -n '2,22p' "$0"; exit 2 ;;
esac
