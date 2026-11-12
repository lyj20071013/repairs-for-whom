#!/usr/bin/env bash
# 第二段串联（2026-11-08 写；第 6 篇；登记补充五之后）：π1 池上续跑 → 核对齐全（最多补跑 2 次）→ 端到端 → 停下。
# 14b2，/root/paper/repair_budget/tau2_pilot 下运行。需要 27B（8003）与带 LoRA 的 4B（8001：4B-A、4B-B、4B-A2）都在。
# 日志：pool/logs/chain.log（与第一段同一个文件，接在后面）；只写步骤、时间、通过与否，不写任何 reward 读数。
# 齐全的标准：设计里每个分支 × 6 次都有 reward；补跑 2 次后仍缺、且缺的全是上下文超长（ContextWindowExceeded）的，
# 按登记第 7 节与补充四按可用次数算、继续；缺的里有别的原因就停下。
#
# 启动：
#   cd /root/paper/repair_budget/tau2_pilot
#   CHAIN_PLAN=1 bash chain_pi1.sh; tail -3 pool/logs/chain.log        # 只核对，不运行
#   nohup bash chain_pi1.sh > pool/logs/chain_pi1_nohup.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")" || exit 1
OUT=pool
LOG=$OUT/logs/chain.log
mkdir -p "$OUT/logs"
PY=${CHAIN_PY:-/root/autodl-tmp/envs/tau/bin/python}
RUN_MAIN=${CHAIN_RUN_MAIN:-"bash run_main.sh"}
export FG=1
say()  { echo "$(TZ=Asia/Tokyo date '+%m-%d %H:%M') 东京 | $*" >> "$LOG"; }
stop() { say "【停下】$*"; say "第二段串联结束（未完成）。处理后可以重新启动 chain_pi1.sh：已完成的步骤会跳过。"; exit 1; }
step() {
  local name=$1 what=$2; shift 2
  local t0=$(date +%s)
  say "▶ $name 开始：$what"
  "$@"; local rc=$?
  say "■ $name 结束：退出码 $rc，用时 $(( ($(date +%s) - t0) / 60 )) 分钟"
  return $rc
}
up() { curl -s -m 3 "http://127.0.0.1:$1/v1/models" >/dev/null; }
lora() { curl -s -m 3 http://127.0.0.1:8001/v1/models | grep -q "\"$1\""; }
done_mark() { [ -f "$OUT/logs/chain_done_$1" ]; }
mark() { touch "$OUT/logs/chain_done_$1"; }
complete() {  # complete [--cw-ok]：三个执行者都齐全才返回 0
  local ok=0
  "$PY" gate_main.py complete --design $OUT/design_pi1_full_A.jsonl $OUT/design_pi1_free_A.jsonl --rollouts $OUT/ro_pi1_A.jsonl --executor 4B-A --k 6 "$@" >> "$LOG" 2>&1 || ok=1
  "$PY" gate_main.py complete --design $OUT/design_pi1_full_B.jsonl $OUT/design_pi1_free_B.jsonl --rollouts $OUT/ro_pi1_B.jsonl --executor 4B-B --k 6 "$@" >> "$LOG" 2>&1 || ok=1
  "$PY" gate_main.py complete --design $OUT/design_pi1s2_A.jsonl --rollouts $OUT/ro_pi1s2_A.jsonl --executor 4B-A2 --k 6 "$@" >> "$LOG" 2>&1 || ok=1
  return $ok
}

say "================ 第二段串联启动（π1 池上续跑 → 核对齐全 → 端到端 → 停下）"
for f in $OUT/w0.json $OUT/design_pi1_full_A.jsonl $OUT/design_pi1_full_B.jsonl $OUT/design_pi1_free_A.jsonl $OUT/design_pi1_free_B.jsonl \
         $OUT/design_pi1s2_A.jsonl $OUT/tasks_A_retail.txt $OUT/tasks_A_airline.txt $OUT/tasks_B_retail.txt $OUT/tasks_B_airline.txt gate_main.py; do
  [ -s "$f" ] || stop "找不到 $f"
done
grep -q -- "--cw-ok" gate_main.py || stop "gate_main.py 是旧版（没有 --cw-ok）：先推送 11-08 的版本"
grep -q "已有 " run_main.sh || stop "run_main.sh 是旧版（端到端失败不会停）：先推送 11-08 的版本"
grep -q 'for src in sorted' pool_tau2.py || stop "pool_tau2.py 是旧版（漂移读文件的修正没有推送）"
done_mark diag || stop "第一段串联的闸门还没通过（没有 chain_done_diag）"
up 8003 || stop "8003 上没有 27B 服务"
for m in 4B-A 4B-B 4B-A2; do lora "$m" || stop "8001 上没有 adapter $m（用 serve_lora.sh 起）"; done
if pgrep -f "python.*pool_tau2.py rollout" >/dev/null 2>&1; then stop "已有续跑进程在跑（$(pgrep -f 'python.*pool_tau2.py rollout' | tr '\n' ' ')），不重复启动"; fi
if [ -n "${CHAIN_PLAN:-}" ]; then
  say "【核对模式】检查都通过：文件齐、代码是新版、8003 与 8001（4B-A、4B-B、4B-A2）都在、没有别的续跑进程。已完成：$(ls $OUT/logs/chain_done_* 2>/dev/null | sed 's#.*chain_done_##' | tr '\n' ' ')"
  exit 0
fi

# ---- 1. π1 池上续跑 + 齐全核对（最多补跑 2 次）
if ! done_mark pi1; then
  step pi1 "π1 的池上续跑（两折全部修复候选与自由选择 + π1ᴬ#2；详见 pi1.log）" $RUN_MAIN pi1 || stop "π1 续跑的命令失败；见 $OUT/logs/pi1.log（重跑会接着跑）"
  for try in 1 2; do
    complete && break
    step "pi1 补跑 $try" "带 --retry-errors 重跑缺的与出错的续跑" $RUN_MAIN pi1 || stop "π1 补跑的命令失败；见 $OUT/logs/pi1.log"
  done
  if complete; then
    say "π1 读数齐全（不含具体数值）"
  elif complete --cw-ok; then
    say "π1 补跑 2 次后仍缺几次，都是上下文超长；按登记第 7 节与补充四按可用次数算（缺的明细见上）"
  else
    stop "补跑 2 次后 π1 仍不齐全，且缺的里有上下文超长以外的原因；见上面的明细与 $OUT/logs/pi1.log"
  fi
  mark pi1
fi

# ---- 2. 端到端
if ! done_mark e2e; then
  step e2e "端到端：π0 全部任务，π1ᴬ A 折、π1ᴮ B 折，每任务 4 段（详见 e2e.log 与 logs/run_main_e2e_*.log）" $RUN_MAIN e2e \
    || stop "端到端失败；见 $OUT/logs/e2e.log。重跑前先把失败那一段的结果目录发给 Claude 核对（可能写了一半）"
  mark e2e
fi
say "第二段串联完成：π1 池上续跑与端到端都结束。下一步：拉回数据，运行 analyze"
exit 0
