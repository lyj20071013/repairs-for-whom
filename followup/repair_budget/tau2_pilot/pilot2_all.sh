#!/usr/bin/env bash
# tau 服务器：先导 2——换更强的本地用户 Qwen3.8-27B-FP8（2026-10-29 写；13:59 北京时间改第二版）。
# 前提：8001 上是 Qwen3.5-4B（部署者），8003 上是 Qwen3.8-27B-FP8（用户）；9B 不用开（显存给 27B）。
# 用法（在 /root/paper/repair_budget/tau2_pilot 下）：nohup bash pilot2_all.sh > logs/pilot2_all.log 2>&1 &
# 各步状态写在 logs/pilot2_status.txt；CPU 采样写在 logs/pilot2_cpu.log。
#
# 第二版的改动（都是为了让先导 2 能回答问题，见 补充_先导1再分析.md）：
#  1) 步数上限 200（= τ² 默认；先导 1 是 100）。先导 1 里 telecom 正常成功的对话常到 80–96 条消息，
#     100 步本身就会截掉一部分会成功的对话，和"用户不结束"混在一起。200 跑完后用 posthoc_tau2.py --cap 100
#     可以事后得到上限 100 的结果（步数上限只决定在哪里停），所以不丢可比性。
#  2) 两步同时跑（两个 tau2 进程）。工作区里测得 tau2 每段对话光建环境、判分就要约 2–3 秒 CPU，
#     单个 Python 进程受 GIL 限制，先导 1 里"GPU 没跑满"很可能是这个原因而不是并发不够（未验证）。
#     每分钟记一次两个进程的 CPU 占用：如果一直在 100% 左右，就是 CPU 瓶颈。
#  3) 用户重复终止仍然关（REPEAT_STOP=0），事后用 posthoc_tau2.py --repeat 3 看开了会怎样。
set -u
HERE=/root/paper/repair_budget/tau2_pilot
cd "$HERE" || exit 2
ST=$HERE/logs/pilot2_status.txt
CPU=$HERE/logs/pilot2_cpu.log
mkdir -p logs
step() {
  local name=$1; shift
  echo "== $(TZ=Asia/Shanghai date '+%m-%d %H:%M:%S') 开始 $name"
  "$@"; local rc=$?
  echo "$(TZ=Asia/Shanghai date '+%m-%d %H:%M:%S')  $name  退出码 $rc" | tee -a "$ST"
}
echo "== 先导 2 开始 $(TZ=Asia/Shanghai date)" | tee -a "$ST"
# 与先导 1 同样的任务、同样的部署者、同样的种子；换用户、步数上限 200。telecom 2 次（与本地 9B 用户对照），airline 1 次（与 flash 对照）
step p2_4b_tel_q27 env CONC=32 MAXSTEPS=200 bash run_tau2.sh 4B telecom 2 p2_4b_tel_q27 q27 ids_telecom.txt > logs/pilot2_tel.out 2>&1 &
P1=$!
step p2_4b_air_q27 env CONC=32 MAXSTEPS=200 bash run_tau2.sh 4B airline 1 p2_4b_air_q27 q27 ids_airline.txt > logs/pilot2_air.out 2>&1 &
P2=$!
echo "两步已在后台启动（PID $P1、$P2）；各自的输出在 logs/pilot2_tel.out、logs/pilot2_air.out"
while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do
  { echo "-- $(TZ=Asia/Shanghai date '+%H:%M:%S')"; ps -eo pid,pcpu,rss,etime,args | grep -E "bin/tau2 run|tau2_ext.py" | grep -v grep | cut -c1-120; } >> "$CPU"
  sleep 60
done
wait $P1; wait $P2
cat "$ST"
echo "== 先导 2 结束 $(TZ=Asia/Shanghai date)" | tee -a "$ST"
