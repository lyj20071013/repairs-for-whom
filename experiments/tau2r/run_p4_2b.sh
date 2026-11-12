#!/usr/bin/env bash
# 14b2 服务器：第 4 篇 6.14 修订 3 / 4 的 2B 读数（子集 201 个位置、733 个分支；2B 每分支 10 次 + 贪心 1 次，共 8,063 次续跑）。
# 用第 6 篇的 pool_tau2.py rollout（只读它，不改）；输出全部写在 /root/paper/repair_transfer/out_tau_pool/。
# 需要的服务（用第 6 篇的 serve_tau.sh 起；不要与 4B 服务同开）：27B=8003（0.62）+ 2B=8004（0.12）。
# 用法：cd /root/paper/repair_transfer/tau2r && nohup bash run_p4_2b.sh > ../out_tau_pool/run_p4_2b.log 2>&1 &
#       然后 tail -f ../out_tau_pool/run_p4_2b.log
# 顺序：核对设计文件 md5 → dry-run（只检查前 20 个续跑的恢复，不调用模型）→ 2B 温度 0.7 每分支 10 次 → 2B 贪心 1 次。
# 任何一步失败就停下；中断后重跑本脚本会接着跑（pool_tau2.py 跳过已完成的续跑）。
set -u
D=/root/paper/repair_transfer/out_tau_pool
PY=/root/autodl-tmp/envs/tau/bin/python
WANT_MD5=1de74ad8f93b33fdb31752928118dd38
P="${PROCS:-6}"; W="${WORKERS:-24}"
ts() { TZ=Asia/Shanghai date '+%m-%d %H:%M:%S'; }
cd /root/paper/repair_budget/tau2_pilot/tau2-bench || { echo "[停下] 找不到 tau2-bench 目录"; exit 2; }
for p in 8003 8004; do
  curl -s -m 3 "http://127.0.0.1:$p/v1/models" >/dev/null || { echo "[停下] 端口 $p 上没有 vLLM 服务"; exit 2; }
done
GOT=$(md5sum "$D/design_p4sub.jsonl" | cut -d' ' -f1)
[ "$GOT" = "$WANT_MD5" ] || { echo "[停下] design_p4sub.jsonl 的 md5 是 $GOT，应为 $WANT_MD5"; exit 2; }
COMMON="--design $D/design_p4sub.jsonl --executors 2B --user q27 --max-steps 200 --procs $P --workers $W --retry-errors"
echo "===== $(ts) dry-run"
$PY ../pool_tau2.py rollout $COMMON --k 10 --temp 0.7 --dry-run 20 --out $D/ro_2b.jsonl || { echo "[停下] dry-run 失败"; exit 1; }
echo "===== $(ts) 2B 温度 0.7，每分支 10 次（7,330 次）"
$PY ../pool_tau2.py rollout $COMMON --k 10 --temp 0.7 --out $D/ro_2b.jsonl
rc=$?; echo "===== $(ts) 2B T07 结束，退出码 $rc"; [ $rc -eq 0 ] || { echo "[停下] 重跑本脚本会补跑出错的续跑"; exit $rc; }
echo "===== $(ts) 2B 贪心，每分支 1 次（733 次）"
$PY ../pool_tau2.py rollout $COMMON --k 1 --temp 0 --out $D/ro_2b_greedy.jsonl
rc=$?; echo "===== $(ts) 2B 贪心结束，退出码 $rc"; [ $rc -eq 0 ] || { echo "[停下] 重跑本脚本会补跑出错的续跑"; exit $rc; }
wc -l $D/ro_2b.jsonl $D/ro_2b_greedy.jsonl
echo "===== $(ts) 全部完成"
