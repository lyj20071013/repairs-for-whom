#!/usr/bin/env bash
# 14b2：第 4 篇 H1 / H2 的第一步动作采样（登记_策略差异与损失_20261111.md 第 4 节）。只生成一步，不续跑、不读任何 reward。
# 用第 6 篇的 pool_tau2.py drift 与 classify_drift.py（只读，不改）。输出都在 /root/paper/repair_budget/tau2_pilot/pool/：
#   first_p4.jsonl         2B 第 1–4 组、4B 第 3–4 组，每组每位置 10 个（4B 第 1–2 组用已有的 pool/drift.jsonl）
#   first_p4_greedy.jsonl  2B 温度 0 的第一步（rep = g）
#   first_cls.jsonl        文字回复的类别（27B 盲判）
# 显存：27B(0.62) + 2B(0.12) 或 27B + 4B(0.30)，三个不能同开。所以分三步，按顺序各跑一次：
#   bash run_first_p4.sh 2b     # 需要 2B=8004
#   bash run_first_p4.sh 4b     # 先停 2B、起 4B=8001（见下面的提示）
#   bash run_first_p4.sh cls    # 需要 27B=8003；可在 2b / 4b 之后任何时候跑，重跑只补没判的
set -u
T=/root/paper/repair_budget/tau2_pilot
D=/root/paper/repair_transfer/out_tau_pool/design_p4sub.jsonl
PY=/root/autodl-tmp/envs/tau/bin/python
O=$T/pool
need() { curl -s -m 3 "http://127.0.0.1:$1/v1/models" >/dev/null || { echo "[停下] 端口 $1 上没有 vLLM 服务"; exit 2; }; }
[ "$(md5sum $D | cut -d' ' -f1)" = "1de74ad8f93b33fdb31752928118dd38" ] || { echo "[停下] design_p4sub.jsonl 的 md5 不对"; exit 2; }
cd $T/tau2-bench || exit 2
case "${1:-}" in
2b)
  need 8004
  $PY ../pool_tau2.py drift --design $D --models "2B,2B#2,2B#3,2B#4" --m 10 --temp 0.7 --workers 16 --out $O/first_p4.jsonl || exit 1
  $PY ../pool_tau2.py drift --design $D --models "2B#g" --m 1 --temp 0 --workers 16 --out $O/first_p4_greedy.jsonl || exit 1
  echo "下一步：kill \$(cat $T/logs/vllm_8004.pid)；等半分钟；cd $T && bash serve_tau.sh Qwen3.5-4B 8001 0.30 qwen3_coder；然后 bash run_first_p4.sh 4b" ;;
4b)
  need 8001
  $PY ../pool_tau2.py drift --design $D --models "4B#3,4B#4" --m 10 --temp 0.7 --workers 16 --out $O/first_p4.jsonl || exit 1 ;;
cls)
  need 8003
  cd $T && $PY classify_drift.py classify --drift pool/first_p4.jsonl --port 8003 --out pool/first_cls.jsonl || exit 1
  $PY classify_drift.py classify --drift pool/first_p4_greedy.jsonl --port 8003 --out pool/first_cls.jsonl || exit 1 ;;
*) echo "用法：bash run_first_p4.sh 2b|4b|cls"; exit 2 ;;
esac
wc -l $O/first_p4.jsonl $O/first_p4_greedy.jsonl 2>/dev/null
echo "== ${1} 完成（应有：2b 后 first_p4 8,040 行、greedy 201 行；4b 后 first_p4 12,060 行）"
