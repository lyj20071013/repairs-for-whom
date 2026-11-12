#!/usr/bin/env bash
# tau 服务器：τ²-bench 正式先导（Q2–Q5）一次跑完（2026-10-29 写）。需要挂卡，并先用 serve_tau.sh 起好 8001（4B）与 8002（9B）。
# 用法（在 /root/paper/repair_budget/tau2_pilot 下）：nohup bash pilot_all.sh > logs/pilot_all.log 2>&1 &
#   然后 tail -f logs/pilot_all.log 看进度；每一步的状态另写在 logs/pilot_status.txt。
# 各步互相独立：一步失败会记下退出码，接着跑下一步（失败的步可以单独重跑，tau2 会从断点续）。
# flash 用户的两步需要环境变量 QWEN_API_KEY；没设置就跳过并记下。
set -u
HERE=/root/paper/repair_budget/tau2_pilot
ENV=/root/autodl-tmp/envs/tau
cd "$HERE" || exit 2
ST=$HERE/logs/pilot_status.txt
mkdir -p logs
step() {   # step <名字> <命令…>
  local name=$1; shift
  echo "== $(TZ=Asia/Shanghai date '+%m-%d %H:%M:%S') 开始 $name"
  "$@"; local rc=$?
  echo "$(TZ=Asia/Shanghai date '+%m-%d %H:%M:%S')  $name  退出码 $rc" | tee -a "$ST"
}
echo "== 先导开始 $(TZ=Asia/Shanghai date)" | tee -a "$ST"
# Q2–Q4：本地用户（9B 贪心），两个部署者 × 两个领域 × 50 个任务 × 2 次
step p_4b_air_local bash run_tau2.sh 4B airline 2 p_4b_air_local local ids_airline.txt
step p_4b_tel_local bash run_tau2.sh 4B telecom 2 p_4b_tel_local local ids_telecom.txt
step p_9b_air_local bash run_tau2.sh 9B airline 2 p_9b_air_local local ids_airline.txt
step p_9b_tel_local bash run_tau2.sh 9B telecom 2 p_9b_tel_local local ids_telecom.txt
# Q3 参照：flash 用户，只配 4B，每任务 1 次
if [ -n "${QWEN_API_KEY:-}" ]; then
  step p_4b_air_flash bash run_tau2.sh 4B airline 1 p_4b_air_flash flash ids_airline.txt
  step p_4b_tel_flash bash run_tau2.sh 4B telecom 1 p_4b_tel_flash flash ids_telecom.txt
else
  echo "QWEN_API_KEY 未设置：跳过两步 flash 用户" | tee -a "$ST"
fi
# Q5：从 4B 的失败对话中间分叉，原动作续跑（每领域 5 条失败 × 3 个位置 × 5 次）
cd "$HERE/tau2-bench" || exit 2
step br_4b_air "$ENV/bin/python" ../branch_tau2.py --src data/simulations/p_4b_air_local/results.json \
  --domain airline --dep 4B --n-traj 5 --k 5 --out ../logs/branch_p_4b_air.jsonl
step br_4b_tel "$ENV/bin/python" ../branch_tau2.py --src data/simulations/p_4b_tel_local/results.json \
  --domain telecom --dep 4B --n-traj 5 --k 5 --out ../logs/branch_p_4b_tel.jsonl
echo "== 先导结束 $(TZ=Asia/Shanghai date)" | tee -a "$ST"
