#!/usr/bin/env bash
# 先导 5（2026-10-31 写）：判分修正后，重跑"共享失败状态"——2B 的失败状态，位置只取第一次错误写之前，2B / 4B / 9B 各自接着做。
# 14b2 上，在 /root/paper/repair_budget/tau2_pilot 下运行；Python 用 τ² 的 venv（/root/autodl-tmp/envs/tau）。
#   bash run_tk5.sh 2B      需要 27B（8003）和 2B（8004）已经起好；后台跑 retail（20 个位置）再跑 airline（15 个），各 ×4 次
#   bash run_tk5.sh 4B      同上，4B（8001）
#   bash run_tk5.sh 9B      同上，9B（8002）
# 输出：logs/tk5_{ret,air}_<模型>.jsonl（每次续跑一行；reward = 原任务判分，reward_hist = τ² 带前缀的旧判分，只作对照）
#       同名 .design.jsonl（位置；三个模型的位置完全相同）与 .msgs.jsonl.gz（续跑的新消息，供离线重判）
# 日志：logs/tk5_<模型>.out。正常：每个领域先打印"位置规则 before_bad""位置 N 个"，再"续跑共 M 次"，每 50 次一行进度，
#       最后"== 完成"；两个领域都完成后打印"=== tk5 <模型> 结束"。中断后重跑同一条命令会接着跑。
set -uo pipefail
cd "$(dirname "$0")" || exit 1
M="${1:-}"
case "$M" in
  2B) PORT=8004 ;; 4B) PORT=8001 ;; 9B) PORT=8002 ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
PY=/root/autodl-tmp/envs/tau/bin/python
for p in 8003 $PORT; do
  curl -s -m 3 "http://127.0.0.1:$p/v1/models" >/dev/null || { echo "[停下] 端口 $p 上没有 vLLM 服务"; exit 2; }
done
m=$(echo "$M" | tr 'A-Z' 'a-z')
OUT=logs/tk5_$m.out
nohup bash -c "
cd tau2-bench
echo \"=== \$(date '+%F %T') tk5 $M retail\"
$PY ../pool_tau2.py takeover --src data/simulations/p4_2b_ret_q27/results.json --domain retail --n-pos 20 --executors $M --k 4 --user q27 --max-steps 200 --workers 16 --procs 4 --retry-errors --out ../logs/tk5_ret_$m.jsonl
echo \"=== \$(date '+%F %T') tk5 $M airline\"
$PY ../pool_tau2.py takeover --src data/simulations/p4_2b_air_q27/results.json --domain airline --n-pos 15 --executors $M --k 4 --user q27 --max-steps 200 --workers 16 --procs 4 --retry-errors --out ../logs/tk5_air_$m.jsonl
echo \"=== \$(date '+%F %T') tk5 $M 结束\"
" >> "$OUT" 2>&1 < /dev/null &
echo "已在后台启动（pid $!）。看日志：tail -f $OUT"
