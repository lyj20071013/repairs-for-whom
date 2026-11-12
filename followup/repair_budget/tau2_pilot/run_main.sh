#!/usr/bin/env bash
# 第 6 篇 τ² 主实验（2026-11-05 写；登记 D:\Paper\repair_budget\登记_τ²主实验_20261105.md）。14b2 上，在 /root/paper/repair_budget/tau2_pilot 下运行。
# Python：τ² 的 venv /root/autodl-tmp/envs/tau。小文件写在 tau2_pilot/pool/，adapter 写在 /root/autodl-tmp/tau_runs/main/。
# 长的步骤在后台跑，日志 pool/logs/<步>.log，结尾打印 "=== … 步 … 结束，退出码 …"。每一步都可以重跑（已完成的会跳过）。
# 续跑并发：WORKERS（默认 24）、PROCS（默认 6）。
#
# 顺序（括号里是需要的 vLLM 服务；显存：27B 0.62 + 4B 0.30，或 27B 0.62 + 2B 0.12；不要三个同开）：
#   bash run_main.sh folds       （CPU）两折 → pool/folds.json、pool/tasks_<折>_<领域>.txt
#   bash run_main.sh collect2b   （后台；8003 + 8004）2B 每任务 20 次，种子 302
#   bash run_main.sh collect4b   （后台；8003 + 8001）4B 每任务 2 次，种子 302（锚样本来源）
#   bash run_main.sh select      （CPU）选失败与位置，取 800 个 → pool/positions.jsonl
#   bash run_main.sh design      （后台；8001）4B 提议、金标、原动作、自由选择（不生成随机候选）
#   bash run_main.sh adm         （后台；8003）可执行性判定 v2、v3 → pool/adm_v2.jsonl、pool/adm_v3.jsonl
#   bash run_main.sh branches    （CPU）续跑用的设计文件 → pool/design_run.jsonl（以及 _A / _B）。之后把设计文件与 v2 判定交第 4 篇
#   bash run_main.sh pi0         （后台；8003 + 8001）π0 = 4B 在全部位置、全部分支上每分支 6 次 → pool/ro_pi0.jsonl
#   bash run_main.sh drift0      （后台；8001）π0 的第一步漂移样本（4B、4B#2）
#   bash run_main.sh w0          （CPU）π0 值得修的位置、π1 的续跑设计（不读任何 π1 数据）
#   bash run_main.sh export      （前台；8001 + 8003）两折的训练样本（π1ᴬ 用 B 折的标签，π1ᴮ 用 A 折的）+ /tokenize 核对
#   bash run_main.sh train       （后台；要先停掉所有 vLLM 服务）训 π1ᴬ（种子 1）、π1ᴮ（种子 1）、π1ᴬ#2（种子 2）
#   bash run_main.sh parity      （后台；要先停掉所有 vLLM 服务）vLLM 带 LoRA 与 transformers 的一致性
#   然后起服务：bash serve_tau.sh Qwen3.8-27B-FP8 8003 0.62 qwen3_coder
#               bash serve_lora.sh Qwen3.5-4B 8001 0.30 4B-A=/root/autodl-tmp/tau_runs/main/adapter_A 4B-B=/root/autodl-tmp/tau_runs/main/adapter_B 4B-A2=/root/autodl-tmp/tau_runs/main/adapter_A2
#   bash run_main.sh pi1check    （后台；8003 + LoRA 8001）每折前 6 个位置的行为检查续跑；完了用 bash run_main.sh diag 看
#   bash run_main.sh diag        （前台，CPU）π1 与 π0 的行为比较（不读 reward）
#   bash run_main.sh pi1         （后台；8003 + LoRA 8001）π1 的全部池上续跑（两折）+ π1ᴬ#2
#   bash run_main.sh drift1      （后台；LoRA 8001）π1 的第一步漂移样本
#   bash run_main.sh e2e         （后台；8003 + LoRA 8001）端到端：π0 全部任务，π1ᴬ A 折任务，π1ᴮ B 折任务，每任务 4 次，种子 304
#   bash run_main.sh analyze     （前台，CPU）写 pool/report_main.md
#   bash run_main.sh status      看各步进度
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY=/root/autodl-tmp/envs/tau/bin/python
OUT=pool
RUNS=/root/autodl-tmp/tau_runs/main
M4=/root/autodl-tmp/models/Qwen3.5-4B
W="${WORKERS:-24}"; P="${PROCS:-6}"
mkdir -p "$OUT/logs" "$RUNS"
GPU="${TAU_GPU:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=/root/autodl-tmp/cache
export TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache VLLM_CACHE_ROOT=/root/autodl-tmp/cache/vllm TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
case "${OMP_NUM_THREADS-}" in ''|*[!0-9]*|0) export OMP_NUM_THREADS=4 ;; esac
export MKL_THREADING_LAYER=GNU
need() { for p in "$@"; do curl -s -m 3 "http://127.0.0.1:$p/v1/models" >/dev/null || { echo "[停下] 端口 $p 上没有 vLLM 服务"; exit 2; }; done; }
needfile() { for f in "$@"; do [ -s "$f" ] || { echo "[停下] 找不到 $f（前一步还没做完？）"; exit 2; }; done; }
needlora() { for m in "$@"; do curl -s -m 3 http://127.0.0.1:8001/v1/models | grep -q "\"$m\"" || { echo "[停下] 8001 上没有 adapter $m（用 serve_lora.sh 起）"; exit 2; }; done; }
nogpu() { if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then echo "[停下] 第 $GPU 张卡上还有进程（先停掉 vLLM 服务）"; nvidia-smi; exit 1; fi; }
bg() {  # bg <步名> <命令…>：后台运行，日志 pool/logs/<步名>.log。FG=1 时前台运行并返回退出码（chain_main.sh 用）
  local name=$1; shift
  if [ -n "${FG:-}" ]; then
    bash -c "echo \"=== \$(TZ=Asia/Shanghai date '+%F %T') 步 $name 开始（前台）\"; ( $* ); rc=\$?; echo \"=== \$(TZ=Asia/Shanghai date '+%F %T') 步 $name 结束，退出码 \$rc\"; exit \$rc" \
      >> "$OUT/logs/$name.log" 2>&1 < /dev/null
    return $?
  fi
  nohup bash -c "echo \"=== \$(TZ=Asia/Shanghai date '+%F %T') 步 $name 开始\"; $*; rc=\$?; echo \"=== \$(TZ=Asia/Shanghai date '+%F %T') 步 $name 结束，退出码 \$rc\"" \
    >> "$OUT/logs/$name.log" 2>&1 < /dev/null &
  echo "已在后台启动 $name（pid $!）。看日志：tail -f $OUT/logs/$name.log"
}
SIM=tau2-bench/data/simulations
C="cd tau2-bench; PY=$PY"
ROARGS="--user q27 --max-steps 200 --procs $P --workers $W --retry-errors --k 6"
case "${1:-}" in
folds)
  exec "$PY" main_pool.py folds --ids-retail ids_retail_det.txt --ids-airline ids_airline.txt --out-dir $OUT ;;
collect2b)
  need 8003 8004
  bg collect2b "env CONC=20 MAXSTEPS=200 SEED=302 bash run_tau2.sh 2B retail 20 main_2b_ret q27 ids_retail_det.txt; env CONC=20 MAXSTEPS=200 SEED=302 bash run_tau2.sh 2B airline 20 main_2b_air q27 ids_airline.txt" ;;
collect4b)
  need 8003 8001
  bg collect4b "env CONC=20 MAXSTEPS=200 SEED=302 bash run_tau2.sh 4B retail 2 main_4b_ret q27 ids_retail_det.txt; env CONC=20 MAXSTEPS=200 SEED=302 bash run_tau2.sh 4B airline 2 main_4b_air q27 ids_airline.txt" ;;
select)
  needfile $OUT/folds.json $SIM/main_2b_ret/results.json $SIM/main_2b_air/results.json
  cd tau2-bench || exit 1
  "$PY" ../main_pool.py select --src-retail data/simulations/main_2b_ret/results.json --src-airline data/simulations/main_2b_air/results.json \
      --folds ../$OUT/folds.json --per-task 4 --n 800 --out ../$OUT/positions.jsonl 2>&1 | tee ../$OUT/logs/select.log
  exit "${PIPESTATUS[0]}" ;;
design)
  need 8001; needfile $OUT/positions.jsonl
  D="--deployer 2B --n-traj 1000000 --per-task 4 --no-random --proposer 4B --max-steps 200 --workers 8 --positions ../$OUT/positions.jsonl"
  bg design "$C; \$PY ../pool_tau2.py design --src data/simulations/main_2b_ret/results.json --domain retail $D --out ../$OUT/design_ret.jsonl; \
\$PY ../pool_tau2.py design --src data/simulations/main_2b_air/results.json --domain airline $D --out ../$OUT/design_air.jsonl" ;;
adm)
  need 8003; needfile $OUT/design_ret.jsonl $OUT/design_air.jsonl
  bg adm "$C; \$PY ../adm_tau2.py annotate --design ../$OUT/design_ret.jsonl ../$OUT/design_air.jsonl --rule v2 --judge-port 8003 --out ../$OUT/adm_v2.jsonl; \
\$PY ../adm_tau2.py annotate --design ../$OUT/design_ret.jsonl ../$OUT/design_air.jsonl --rule v3 --judge-port 8003 --cache ../$OUT/adm_v2.jsonl --out ../$OUT/adm_v3.jsonl" ;;
branches)
  needfile $OUT/adm_v2.jsonl $OUT/adm_v3.jsonl
  exec "$PY" main_pool.py branches --design $OUT/design_ret.jsonl $OUT/design_air.jsonl --positions $OUT/positions.jsonl --adm-v2 $OUT/adm_v2.jsonl --out $OUT/design_run.jsonl ;;
pi0)
  need 8003 8001; needfile $OUT/design_run.jsonl
  bg pi0 "$C; \$PY ../pool_tau2.py rollout --design ../$OUT/design_run.jsonl --executors 4B $ROARGS --out ../$OUT/ro_pi0.jsonl" ;;
drift0)
  need 8001; needfile $OUT/design_run.jsonl
  bg drift0 "$C; \$PY ../pool_tau2.py drift --design ../$OUT/design_run.jsonl --models 4B,4B#2 --m 10 --workers 16 --out ../$OUT/drift.jsonl" ;;
w0)
  needfile $OUT/ro_pi0.jsonl
  exec "$PY" main_pool.py w0 --design $OUT/design_run.jsonl --rollouts $OUT/ro_pi0.jsonl --adm-v2 $OUT/adm_v2.jsonl --adm-v3 $OUT/adm_v3.jsonl \
      --check-frac 0.15 --loop-frac 0.25 --seed2-fold A ${ALLOW_INCOMPLETE:+--allow-incomplete} --out-dir $OUT ;;
export)
  need 8001 8003; needfile $OUT/w0.json $SIM/main_4b_ret/results.json $SIM/main_4b_air/results.json
  cd tau2-bench || exit 1
  # π1ᴬ 用 B 折位置上的标签、B 折任务上的锚（--seed 1）；π1ᴮ 用 A 折的（--seed 2）。登记补充一
  for pair in "A B 1" "B A 2"; do
    set -- $pair; M=$1; SRC=$2; S=$3
    "$PY" ../train_tau2.py export --design ../$OUT/design_run_$SRC.jsonl --rollouts ../$OUT/ro_pi0.jsonl --executor 4B --thr 0.20 --min-n 0 \
        --anchor-src data/simulations/main_4b_ret/results.json data/simulations/main_4b_air/results.json --anchor-domain retail airline \
        --task-file ../$OUT/tasks_${SRC}_retail.txt ../$OUT/tasks_${SRC}_airline.txt --task-domain retail airline --seed $S --anchor-mult 1 \
        --preamble --vllm-port 8001 --served-name Qwen3.5-4B --admissible --adm-file ../$OUT/adm_v3.jsonl \
        --judge-port 8003 --judge-name Qwen3.8-27B-FP8 --template --min-final 0 --out ../$OUT/samples_$M.jsonl 2>&1 | tee ../$OUT/logs/export_$M.log
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "[停下] export $M 没有通过，见 $OUT/logs/export_$M.log"; exit 1; }
  done
  cd .. || exit 1
  for M in A B; do
    "$PY" train_tau2.py check --samples $OUT/samples_$M.jsonl --model-dir "$M4" --max-len 16384 --vllm-port 8001 --served-name Qwen3.5-4B 2>&1 | tee -a $OUT/logs/export_$M.log
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "[停下] check $M 没有通过"; exit 1; }
  done ;;
train)
  nogpu; needfile $OUT/samples_A.jsonl $OUT/samples_B.jsonl
  T="--model-dir $M4 --targets attn_mlp --epochs 3 --batch 8 --lr 1e-4 --rank 16 --alpha 32 --dropout 0.05 --max-len 16384"
  bg train "$PY train_tau2.py train --samples $OUT/samples_A.jsonl $T --seed 1 --out $RUNS/adapter_A || exit 1; \
$PY train_tau2.py train --samples $OUT/samples_B.jsonl $T --seed 1 --out $RUNS/adapter_B || exit 1; \
$PY train_tau2.py train --samples $OUT/samples_A.jsonl $T --seed 2 --out $RUNS/adapter_A2 || exit 1" ;;
parity)
  nogpu
  bg parity "$PY lora_parity.py --samples $OUT/samples_A.jsonl --model-dir $M4 --adapters 4B-A=$RUNS/adapter_A 4B-A2=$RUNS/adapter_A2 --n 16 --gen 48 --out $OUT/lora_parity_A.json || exit \$?; \
$PY lora_parity.py --samples $OUT/samples_B.jsonl --model-dir $M4 --adapters 4B-B=$RUNS/adapter_B --n 16 --gen 48 --out $OUT/lora_parity_B.json || exit \$?" ;;
pi1check)
  need 8003; needlora 4B-A 4B-B; needfile $OUT/design_pi1check_A.jsonl $OUT/design_pi1check_B.jsonl
  bg pi1check "$C; export TAU2_EXTRA_MODELS=4B-A@8001,4B-B@8001; \
for M in A B; do \$PY ../pool_tau2.py rollout --design ../$OUT/design_pi1check_\$M.jsonl --executors 4B-\$M --user q27 --max-steps 200 --procs $P --workers $W --retry-errors --k 6 --out ../$OUT/ro_pi1_\$M.jsonl || exit 1; done" ;;
diag)
  # 11-07：闸门（每折、每个领域分开；不读 reward）。pi1_diag.py 按文件名猜领域，主实验的文件名里没有领域，所以改用 gate_main.py
  "$PY" gate_main.py behavior --check A=$OUT/design_pi1check_A.jsonl B=$OUT/design_pi1check_B.jsonl \
      --rollouts $OUT/ro_pi0.jsonl $OUT/ro_pi1_A.jsonl $OUT/ro_pi1_B.jsonl --pi0 4B --pi1 A=4B-A B=4B-B 2>&1 | tee $OUT/logs/diag.log
  exit "${PIPESTATUS[0]}" ;;
pi1)
  need 8003; needlora 4B-A 4B-B 4B-A2
  bg pi1 "$C; export TAU2_EXTRA_MODELS=4B-A@8001,4B-B@8001,4B-A2@8001; \
for M in A B; do \$PY ../pool_tau2.py rollout --design ../$OUT/design_pi1_full_\$M.jsonl --executors 4B-\$M $ROARGS --out ../$OUT/ro_pi1_\$M.jsonl || exit 1; \
\$PY ../pool_tau2.py rollout --design ../$OUT/design_pi1_free_\$M.jsonl --executors 4B-\$M $ROARGS --out ../$OUT/ro_pi1_\$M.jsonl || exit 1; done; \
\$PY ../pool_tau2.py rollout --design ../$OUT/design_pi1s2_A.jsonl --executors 4B-A2 $ROARGS --out ../$OUT/ro_pi1s2_A.jsonl || exit 1" ;;
drift1)
  needlora 4B-A 4B-B 4B-A2
  bg drift1 "$C; export TAU2_EXTRA_MODELS=4B-A@8001,4B-B@8001,4B-A2@8001; \
\$PY ../pool_tau2.py drift --design ../$OUT/design_run_A.jsonl --models 4B-A,4B-A2 --m 10 --workers 16 --out ../$OUT/drift.jsonl; \
\$PY ../pool_tau2.py drift --design ../$OUT/design_run_B.jsonl --models 4B-B --m 10 --workers 16 --out ../$OUT/drift.jsonl" ;;
e2e)
  need 8003; needlora 4B-A 4B-B
  E="env CONC=20 MAXSTEPS=200 SEED=304 bash run_tau2.sh"
  # 11-08：任何一段失败就停（原来用 ; 连接会继续往下跑）；已有结果的段跳过（重跑时不重复跑）
  bg e2e "for spec in '4B retail main_e2e_pi0_ret ids_retail_det.txt' '4B airline main_e2e_pi0_air ids_airline.txt' \
'4B-A@8001 retail main_e2e_A_ret $OUT/tasks_A_retail.txt' '4B-A@8001 airline main_e2e_A_air $OUT/tasks_A_airline.txt' \
'4B-B@8001 retail main_e2e_B_ret $OUT/tasks_B_retail.txt' '4B-B@8001 airline main_e2e_B_air $OUT/tasks_B_airline.txt'; do \
set -- \$spec; [ -s $SIM/\$3/results.json ] && { echo \"已有 \$3，跳过\"; continue; }; $E \$1 \$2 4 \$3 q27 \$4 || exit 1; done" ;;
analyze)
  E2E=""
  [ -s $SIM/main_e2e_pi0_ret/results.json ] && E2E="--e2e-pi0 $SIM/main_e2e_pi0_ret/results.json $SIM/main_e2e_pi0_air/results.json --e2e-pi1 A=$SIM/main_e2e_A_ret/results.json,$SIM/main_e2e_A_air/results.json B=$SIM/main_e2e_B_ret/results.json,$SIM/main_e2e_B_air/results.json --folds $OUT/folds.json"
  exec "$PY" main_analyze.py --design $OUT/design_run.jsonl --adm-v2 $OUT/adm_v2.jsonl --adm-v3 $OUT/adm_v3.jsonl --w0 $OUT/w0.json \
      --rollouts $OUT/ro_pi0.jsonl $OUT/ro_pi1_A.jsonl $OUT/ro_pi1_B.jsonl $OUT/ro_pi1s2_A.jsonl --pi0 4B --pi1 A=4B-A B=4B-B --pi1-seed2 A=4B-A2 \
      --rule v3 --boot 2000 --drift $OUT/drift.jsonl $E2E --out $OUT/report_main.md ;;
status)
  for f in $OUT/logs/*.log; do [ -f "$f" ] && { echo "--- $f"; grep -a "=== \|完成 [0-9]*/\|== 完成\|停下\|出错" "$f" | tail -4; }; done
  ls -la $OUT/*.jsonl 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>/dev/null ;;
*) sed -n '2,32p' "$0"; exit 2 ;;
esac
