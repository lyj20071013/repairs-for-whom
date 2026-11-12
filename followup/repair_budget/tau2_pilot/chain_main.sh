#!/usr/bin/env bash
# 主实验串联（2026-11-07 写；第 6 篇；登记 D:\Paper\repair_budget\登记_τ²主实验_20261105.md 补充三）。14b2 上，/root/paper/repair_budget/tau2_pilot 下运行。
# 等 π0 跑完 → 核对齐全（不齐就补跑，最多 2 次）→ 漂移样本里受任务重名影响的位置补做（不当闸门）→ w0 → 导出两折标签（含 check）
# → 停 vLLM → 训练 3 个 adapter → parity → 起 27B 与带 LoRA 的 4B → 行为检查 → CHAIN_PI1=A 时接着跑 π1 的池上续跑，否则停下。
# 任何一步失败或闸门不通过：在 pool/logs/chain.log 写明原因并停下，不往下跑。chain.log 只写步骤、时间、通过与否，不写任何 reward 读数；
# 各步的详细输出在各自的日志（pool/logs/<步>.log）。
#
# 启动（后台；断开 SSH 也会继续）：
#   cd /root/paper/repair_budget/tau2_pilot
#   CHAIN_PI1=B nohup bash chain_main.sh > /dev/null 2>&1 &        # B = 行为检查通过后停下；A = 接着跑 π1
#   tail -f pool/logs/chain.log
# 挂上去之前先核对（只做第 0 步的检查、列出各步，不等待、不运行任何步骤）：CHAIN_PLAN=1 bash chain_main.sh; tail -20 pool/logs/chain.log
# 各外部命令可用环境变量替换（本地测试用）：CHAIN_PY、CHAIN_RUN_MAIN、CHAIN_SERVE_TAU、CHAIN_SERVE_LORA、CHAIN_RUNS、CHAIN_POLL
set -uo pipefail
cd "$(dirname "$0")" || exit 1
OUT=pool
LOG=$OUT/logs/chain.log
mkdir -p "$OUT/logs"
PY=${CHAIN_PY:-/root/autodl-tmp/envs/tau/bin/python}
RUN_MAIN=${CHAIN_RUN_MAIN:-"bash run_main.sh"}
SERVE_TAU=${CHAIN_SERVE_TAU:-"bash serve_tau.sh"}
SERVE_LORA=${CHAIN_SERVE_LORA:-"bash serve_lora.sh"}
RUNS=${CHAIN_RUNS:-/root/autodl-tmp/tau_runs/main}
POLL=${CHAIN_POLL:-300}
PI1=${CHAIN_PI1:-B}
GPU="${TAU_GPU:-0}"
export FG=1                                   # run_main.sh 的各步在前台跑、返回退出码

say()  { echo "$(TZ=Asia/Tokyo date '+%m-%d %H:%M') 东京 | $*" >> "$LOG"; }
stop() { say "【停下】$*"; say "串联结束（未完成）。处理后可以重新启动 chain_main.sh：已完成的步骤会跳过。"; exit 1; }
step() {  # step <名字> <一行说明> <命令…>：前台运行，写开始 / 结束与耗时
  local name=$1 what=$2; shift 2
  local t0=$(date +%s)
  say "▶ $name 开始：$what"
  "$@"
  local rc=$?
  say "■ $name 结束：退出码 $rc，用时 $(( ($(date +%s) - t0) / 60 )) 分钟"
  return $rc
}
gate() { "$PY" gate_main.py "$@"; }
gpu_busy() { nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; }
up() { curl -s -m 3 "http://127.0.0.1:$1/v1/models" >/dev/null; }
stop_services() {
  for p in 8001 8003 8004; do
    [ -f "logs/vllm_$p.pid" ] && kill "$(cat logs/vllm_$p.pid)" 2>/dev/null
  done
  for i in $(seq 1 60); do gpu_busy || return 0; sleep 5; done
  pkill -f "vllm serve" 2>/dev/null; sleep 20
  gpu_busy && return 1 || return 0
}
done_mark() { [ -f "$OUT/logs/chain_done_$1" ]; }
mark() { touch "$OUT/logs/chain_done_$1"; }

say "================ 串联启动（CHAIN_PI1=$PI1：行为检查通过后$([ "$PI1" = A ] && echo '接着跑 π1' || echo '停下等用户')）"

# ---- 0. 开跑前的核对（不读任何 reward）
gate taskmix --rollouts $OUT/ro_pi0.jsonl >> "$LOG" 2>&1 || stop "任务重名 bug 的修正没有生效（pool_tau2.py 未更新，或 ro_pi0 的受影响行没挪出）"
for f in $OUT/design_run.jsonl $OUT/design_run_A.jsonl $OUT/design_run_B.jsonl $OUT/adm_v2.jsonl $OUT/adm_v3.jsonl $OUT/folds.json \
         $OUT/tasks_A_retail.txt $OUT/tasks_A_airline.txt $OUT/tasks_B_retail.txt $OUT/tasks_B_airline.txt \
         tau2-bench/data/simulations/main_4b_ret/results.json tau2-bench/data/simulations/main_4b_air/results.json main_pool.py gate_main.py; do
  [ -s "$f" ] || stop "找不到 $f"
done
grep -q "design_pi1check_" main_pool.py || stop "main_pool.py 是旧版（没有 design_pi1check）：先推送 11-07 的版本"
for d in adapter_A adapter_B adapter_A2; do
  if [ -e "$RUNS/$d" ] && ! done_mark train; then stop "$RUNS/$d 已存在但串联没训练过（避免覆盖）；确认后删掉或改名再启动"; fi
done

if [ -n "${CHAIN_PLAN:-}" ]; then
  say "【核对模式】第 0 步的检查都通过。将依次：等 π0 → 齐全核对（最多补跑 2 次）→ 漂移补做 → w0 → 导出 → 停服务 → 训练 → parity → 起 27B 与 LoRA → 行为检查 → $([ "$PI1" = A ] && echo 'π1 续跑' || echo '停下')"
  say "  已完成的步骤：$(ls $OUT/logs/chain_done_* 2>/dev/null | sed 's#.*chain_done_##' | tr '\n' ' ')；π0 进程：$(pgrep -f 'python.*pool_tau2.py rollout.*ro_pi0' | tr '\n' ' ')；服务：8003 $(up 8003 && echo 在 || echo 无)、8001 $(up 8001 && echo 在 || echo 无)"
  exit 0
fi

# ---- 1. 等 π0 结束
if ! done_mark pi0; then
  say "等 π0 的续跑进程结束（每 $((POLL / 60)) 分钟看一次）"
  n=0
  while pgrep -f "python.*pool_tau2.py rollout.*ro_pi0" >/dev/null 2>&1; do
    sleep "$POLL"; n=$((n + 1))
    [ $((n * POLL % 3600)) -lt "$POLL" ] && say "  π0 仍在跑（ro_pi0.jsonl 现有 $(wc -l < $OUT/ro_pi0.jsonl) 行）"
  done
  say "π0 的续跑进程已结束"
  # ---- 2. 齐全？不齐就补跑，最多 2 次
  for try in 0 1 2; do
    if gate complete --design $OUT/design_run.jsonl --rollouts $OUT/ro_pi0.jsonl --executor 4B --k 6 >> "$LOG" 2>&1; then
      say "π0 读数齐全（设计里每个分支 × 6 次都有 reward；不含具体数值）"; break
    fi
    [ "$try" = 2 ] && stop "补跑 2 次后 π0 仍不齐全（按要求不用 ALLOW_INCOMPLETE）；见 $OUT/logs/pi0.log"
    up 8003 && up 8001 || stop "π0 不齐全，但 8003 / 8001 上没有服务，没法补跑"
    step "pi0 补跑 $((try + 1))" "带 --retry-errors 重跑缺的与出错的续跑" $RUN_MAIN pi0 || stop "π0 补跑的命令失败；见 $OUT/logs/pi0.log"
  done
  mark pi0
fi

# ---- 3. 漂移样本：受任务重名影响的位置挪出后补做（描述性读数，不当闸门）
if ! done_mark drift0; then
  if gate driftfix --design $OUT/design_run.jsonl --drift $OUT/drift.jsonl >> "$LOG" 2>&1 && up 8001; then
    if step drift0 "π0 的第一步漂移样本（4B、4B#2；只补缺的）" $RUN_MAIN drift0; then mark drift0
    else say "（提示）drift0 失败，漂移是描述性读数，不当闸门，串联继续；之后单独补"; fi
  else
    say "（提示）漂移样本没处理（driftfix 出错或 8001 没服务），串联继续；之后单独补"
  fi
fi

# ---- 4. w0（读 π0 的 reward；登记补充三已在这之前定稿）
if ! done_mark w0; then
  step w0 "π0 值得修的位置、一半内部值得修的位置、核对样本、反复报错的位置、π1 的续跑设计（详见 w0.log）" \
    bash -c "$RUN_MAIN w0 > $OUT/logs/w0.log 2>&1" || stop "w0 失败；见 $OUT/logs/w0.log"
  for f in w0.json design_pi1_free_A.jsonl design_pi1_free_B.jsonl; do
    [ -s "$OUT/$f" ] || stop "w0 之后找不到（或是空的）$OUT/$f"
  done
  for f in design_pi1_full_A.jsonl design_pi1_full_B.jsonl design_pi1s2_A.jsonl design_pi1check_A.jsonl design_pi1check_B.jsonl; do
    [ -e "$OUT/$f" ] || stop "w0 之后找不到 $OUT/$f"
    [ -s "$OUT/$f" ] || say "（提示）$OUT/$f 是空的"
  done
  mark w0
fi

# ---- 5. 导出两折的训练样本（8001 基础 4B + 8003 27B）
if ! done_mark export; then
  up 8003 && up 8001 || stop "导出需要 8003（27B）与 8001（基础 4B）的服务"
  step export "两折训练样本（π1ᴬ 用 B 折标签、π1ᴮ 用 A 折）+ 分词核对（详见 export_A.log / export_B.log）" $RUN_MAIN export \
    || stop "导出或分词核对没有通过；见 $OUT/logs/export_A.log、export_B.log"
  [ -s $OUT/samples_A.jsonl ] && [ -s $OUT/samples_B.jsonl ] || stop "导出之后找不到 samples_A / samples_B"
  mark export
fi

# ---- 6. 停服务 → 训练 → parity
if ! done_mark train; then
  step stopvllm "停掉 vLLM 服务（训练要整张卡）" stop_services || stop "停服务之后卡上还有进程；见 nvidia-smi"
  step train "训练 π1ᴬ（种子 1）、π1ᴮ（种子 1）、π1ᴬ#2（种子 2）（详见 train.log）" $RUN_MAIN train || stop "训练失败；见 $OUT/logs/train.log"
  for d in adapter_A adapter_B adapter_A2; do [ -f "$RUNS/$d/adapter_config.json" ] || stop "训练之后 $RUNS/$d 里没有 adapter_config.json"; done
  mark train
fi
if ! done_mark parity; then
  gpu_busy && { step stopvllm "停掉 vLLM 服务（parity 要整张卡）" stop_services || stop "卡上还有进程"; }
  step parity "vLLM 带 LoRA 与 transformers 的一致性（3 个 adapter；详见 parity.log）" $RUN_MAIN parity
  rc=$?
  [ $rc = 0 ] || stop "parity 没有通过（退出码 $rc：1 = 不通过，3 = 不确定，其他 = 运行出错）；见 $OUT/logs/parity.log。按登记，不通过时先诊断，不跑 π1"
  mark parity
fi

# ---- 7. 起服务 → 行为检查
if ! done_mark diag; then
  if ! up 8003; then
    step serve27b "起 27B（8003）" $SERVE_TAU Qwen3.8-27B-FP8 8003 0.62 qwen3_coder || stop "27B 起不来；见 logs/vllm_8003.log"
  fi
  if ! curl -s -m 3 http://127.0.0.1:8001/v1/models | grep -q '"4B-A2"'; then
    up 8001 && { [ -f logs/vllm_8001.pid ] && kill "$(cat logs/vllm_8001.pid)" 2>/dev/null; sleep 30; }
    step serve4b "起带 LoRA 的 4B（8001：4B-A、4B-B、4B-A2）" $SERVE_LORA Qwen3.5-4B 8001 0.30 \
      4B-A=$RUNS/adapter_A 4B-B=$RUNS/adapter_B 4B-A2=$RUNS/adapter_A2 || stop "带 LoRA 的 4B 起不来；见 logs/vllm_8001.log"
  fi
  step pi1check "行为检查续跑：每折每个领域前 6 个位置（详见 pi1check.log）" $RUN_MAIN pi1check || stop "行为检查的续跑失败；见 $OUT/logs/pi1check.log"
  step diag "行为检查闸门：问题终止率、重复循环率 ≤ π0 + 0.05（每折每个领域；详见 diag.log）" $RUN_MAIN diag
  rc=$?
  [ $rc = 0 ] || stop "行为检查没有通过（退出码 $rc：1 = 超过 π0 + 0.05，2 = 消息记录不全）；见 $OUT/logs/diag.log。按登记先诊断，不跑 π1"
  mark diag
fi
say "闸门都通过（π0 齐全、导出、训练、parity、行为检查）"

# ---- 8. π1
if [ "$PI1" != A ]; then
  say "按选择 B 停下，等用户回来。要接着跑 π1：bash run_main.sh pi1（或 CHAIN_PI1=A 重新启动本脚本）"
  exit 0
fi
if ! done_mark pi1; then
  step pi1 "π1 的池上续跑（两折全部修复候选与自由选择 + π1ᴬ#2；详见 pi1.log）" $RUN_MAIN pi1 || stop "π1 续跑失败；见 $OUT/logs/pi1.log（重跑会接着跑）"
  mark pi1
fi
say "π1 的池上续跑结束。下一步（用户回来后）：核对齐全、drift1、e2e、analyze"
exit 0
