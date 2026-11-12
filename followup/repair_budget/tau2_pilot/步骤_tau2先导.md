# τ²-bench 先导：操作步骤（2026-10-29 01:44 北京时间写）

服务器：**tau 服务器**（连接信息已在公开版中删去，¥5.5 / 卡时）。项目根 `/root/paper/repair_budget`；本篇文件都在 `tau2_pilot/` 下。
Python 环境：conda 环境 `/root/autodl-tmp/envs/tau`（第 2 步建；放数据盘）。模型：`/root/autodl-tmp/models/`。
本地 ↔ 服务器传文件一律用 `python D:\Paper\repair_budget\tools\sync.py …`（在你的电脑上运行）。

## A. 无卡模式（不花 GPU 钱）
**A1. 上传**（你的电脑）：
```
python D:\Paper\repair_budget\tools\sync.py push tau tau2_pilot/setup_check_tau.sh tau2_pilot/install_tau.sh tau2_pilot/download_models.py tau2_pilot/serve_tau.sh tau2_pilot/probe_tools.py tau2_pilot/run_tau2.sh tau2_pilot/tau2-bench-122e243-trim.tar.gz
```
正常：7 个文件都是"新建"，最后"完成，服务器上的每个目标都核对过 md5"。

**A2. 清点 + 安装 + 下载**（tau 服务器，无卡模式，系统 python 即可）：
```
cd /root/paper/repair_budget/tau2_pilot
mkdir -p logs
bash setup_check_tau.sh 2>&1 | tee logs/setup_check_tau.log
bash install_tau.sh 2>&1 | tee logs/install_tau.log
/root/autodl-tmp/envs/tau/bin/python download_models.py 2>&1 | tee logs/download_models.log
```
- install 约 10–30 分钟（主要是 vLLM 与 torch 的下载）。正常的结尾是 `== 第 5 步：导入检查` 下面打印 torch / vllm / transformers / tau2 / litellm / modelscope 的版本、"cuda 可用：False（无卡模式下……正常）"、tau2 领域列表，最后 `== 完成`。中途失败会打印 `[停下] 第 N 步`。
- download 约 25 GB（4B 约 9 GB、9B 约 19 GB，以实际为准）。正常：每个模型一行"config.json 有；safetensors N 个，共 X GB"，最后 `== 完成`。
- 三条命令前后独立，前一条失败就不要跑后一条，把日志拉回来给我看。

**A3. 拉回日志**（你的电脑）：
```
python D:\Paper\repair_budget\tools\sync.py pull tau tau2_pilot/logs/setup_check_tau.log tau2_pilot/logs/install_tau.log tau2_pilot/logs/download_models.log
```

## B. 挂卡（A 都正常之后；我看过 A 的日志再确认）
**B1. 起服务**（tau 服务器）：
```
cd /root/paper/repair_budget/tau2_pilot
bash serve_tau.sh Qwen3.5-9B 8002 0.55 qwen3_coder
bash serve_tau.sh Qwen3.5-4B 8001 0.25 qwen3_coder
```
正常：各打印"就绪（约 N 秒）"。如果 vLLM 报不认识解析器或启动失败，日志最后 40 行会直接打印出来；换 `hermes` 再试（先 `kill $(cat logs/vllm_8002.pid)`）。**Qwen3.5 该用哪个解析器我没法在工作区验证**，这一步就是用来确认的。

**B2. 工具调用探针**（Q1）：
```
/root/autodl-tmp/envs/tau/bin/python probe_tools.py 8001 Qwen3.5-4B 2>&1 | tee logs/probe_8001.log
/root/autodl-tmp/envs/tau/bin/python probe_tools.py 8002 Qwen3.5-9B 2>&1 | tee logs/probe_8002.log
```
正常：每项 PASS，最后"全部通过"。有 FAIL 就停，拉回日志。

**B3. τ² 冒烟**（5 个任务，各 1 次）：
```
bash run_tau2.sh 4B airline 5 1 smoke_4b_air
```
正常：tau2 打印进度与每个任务的奖励，最后"退出码 0"并列出 `results.json`。

**B4. 拉回**（你的电脑）：
```
python D:\Paper\repair_budget\tools\sync.py pull tau tau2_pilot/logs/vllm_8001.log tau2_pilot/logs/vllm_8002.log tau2_pilot/logs/probe_8001.log tau2_pilot/logs/probe_8002.log tau2_pilot/logs/run_smoke_4b_air.log tau2_pilot/tau2-bench/data/simulations/smoke_4b_air/results.json
```
**B 做完就关机或切回无卡模式**，正式先导（Q2–Q5）的命令我看过冒烟结果再写。B 预计 < 0.5 卡时（≈ ¥3）。

## C. 正式先导（Q2–Q5；10-29 03:55 加）
冒烟后改过的设置见 `run_tau2.sh` 开头的注释（子串判定、步数上限 100、生成上限 2048、超时 180 秒）。

**C0. 上传**（你的电脑）：
```
python D:\Paper\repair_budget\tools\sync.py push tau tau2_pilot/run_tau2.sh tau2_pilot/branch_tau2.py tau2_pilot/pilot_all.sh tau2_pilot/ids_airline.txt tau2_pilot/ids_telecom.txt
```
**C1. 起服务**（挂卡；两个都用同一套参数，所以 9B 也要重起）：
```
cd /root/paper/repair_budget/tau2_pilot
bash serve_tau.sh Qwen3.5-9B 8002 0.55 qwen3_coder
bash serve_tau.sh Qwen3.5-4B 8001 0.25 qwen3_coder
```
**C2. 分叉检查**（不调用模型，几秒）：
```
cd /root/paper/repair_budget/tau2_pilot/tau2-bench
/root/autodl-tmp/envs/tau/bin/python ../branch_tau2.py --src data/simulations/smoke_4b_air/results.json --domain airline --dep 4B --n-traj 5 --k 5 --out ../logs/branch_dryrun.jsonl --dry-run
```
正常：打印 3 个位置"前缀 N 条，下一个发言 …，DB …"，最后"dry-run：3 个位置都恢复成功，没有调用模型"。
**C3. 跑先导**（后台；flash 两步需要先 `export QWEN_API_KEY=…`，不设置就自动跳过）：
```
cd /root/paper/repair_budget/tau2_pilot
nohup bash pilot_all.sh > logs/pilot_all.log 2>&1 &
tail -f logs/pilot_all.log
```
预计约 1–1.5 卡时。各步结束时 `logs/pilot_status.txt` 记一行退出码。
**C4. 拉回**（你的电脑）：
```
python D:\Paper\repair_budget\tools\sync.py pull tau tau2_pilot/logs/pilot_status.txt tau2_pilot/logs/pilot_all.log tau2_pilot/tau2-bench/data/simulations/p_4b_air_local/results.json tau2_pilot/tau2-bench/data/simulations/p_4b_tel_local/results.json tau2_pilot/tau2-bench/data/simulations/p_9b_air_local/results.json tau2_pilot/tau2-bench/data/simulations/p_9b_tel_local/results.json tau2_pilot/logs/branch_p_4b_air.jsonl tau2_pilot/logs/branch_p_4b_tel.jsonl tau2_pilot/logs/vllm_8001.log tau2_pilot/logs/vllm_8002.log
```
flash 跑了的话再拉 `tau2_pilot/tau2-bench/data/simulations/p_4b_air_flash/results.json` 与 `p_4b_tel_flash/results.json`。拉完关卡；分析（`analyze_pilot.py`）由 Claude 做。

## D. 先导 2：用户换成 Qwen3.8-27B-FP8（10-29 加；14:03 北京时间改第二版）
起因：先导 1 中本地 9B 用户在 telecom 的 max_steps 很多（见 `先导结果_tau2.md`）。第二版改动的原因见 `补充_先导1再分析.md`：
步数上限改 200（100 对 telecom 太紧），两步同时跑并记录 CPU（检查是不是 CPU 瓶颈）。

> **10-30 15:00 加：tau 服务器的卡租满，D 节改在 14b2 上做。** 14b2 上用与 tau 服务器相同的布局，下面的命令就不用改：项目根 `/root/paper/repair_budget`、τ² 的 venv `/root/autodl-tmp/envs/tau`（**不要装进 P0 用的 conda 环境 `llm`**）、模型 `/root/autodl-tmp/models/`。传文件把命令里的 `tau` 换成 `14b2`（`tools/sync.py` 已把 14b2 的项目根改成 `/root/paper/repair_budget`）。
> 14b2 第一次部署按 A 节做：上传 `setup_check_tau.sh`、`install_tau.sh`、`download_models.py`、`serve_tau.sh`、`probe_tools.py`、`run_tau2.sh`、`tau2-bench-122e243-trim2.tar.gz`（`install_tau.sh` 已改用 trim2 包，10-30 14:21 也已加上 14b2 的 Python 路径），再下载 Qwen3.5-4B 与 Qwen3.8-27B-FP8（`download_models.py Qwen/Qwen3.5-4B Qwen/Qwen3.8-27B-FP8`；9B 先导 2 用不到，可以先不下）。数据盘先 `df -h /root/autodl-tmp`，约需 40 GB。
> 先导 1 的 `results.json` 不在 14b2 上；D1b 的第一条检查（`posthoc_tau2.py`）要先把 `tau2_pilot/tau2-bench/data/simulations/p_4b_tel_local/results.json` 与 `p_4b_air_local/results.json` 传上去。

**D0. 上传**（你的电脑）：
```
python D:\Paper\repair_budget\tools\sync.py push tau tau2_pilot/run_tau2.sh tau2_pilot/pilot2_all.sh tau2_pilot/tau2_ext.py tau2_pilot/posthoc_tau2.py tau2_pilot/pool_tau2.py
```
正常：5 个文件（run_tau2.sh、pilot2_all.sh 是"替换"，另外 3 个是"新建"），最后"完成，服务器上的每个目标都核对过 md5"。

**D1. 下载模型**（tau 服务器，无卡模式；10-29 数据盘已腾到 37 GB）：
```
cd /root/paper/repair_budget/tau2_pilot
/root/autodl-tmp/envs/tau/bin/python download_models.py Qwen/Qwen3.8-27B-FP8 2>&1 | tee logs/download_27b.log
```
- 正常：先打印"数据盘剩余 37.x GB"，最后"config.json 有；safetensors N 个，共约 28–30 GB"和"== 完成"。
- 下载途中另开终端看 `df -h /root/autodl-tmp` 与 `du -sh /root/autodl-tmp/modelscope_cache`：缓存目录也在涨（下两份）就 Ctrl-C，把输出发给 Claude。

**D1b. 不挂卡的检查**（tau 服务器，几秒；确认新脚本在服务器的 tau2 版本上能用）：
```
cd /root/paper/repair_budget/tau2_pilot/tau2-bench
/root/autodl-tmp/envs/tau/bin/python ../posthoc_tau2.py data/simulations p_4b_tel_local p_4b_air_local --cap 100
/root/autodl-tmp/envs/tau/bin/python ../pool_tau2.py design --src data/simulations/p_4b_air_local/results.json --domain airline --n-traj 3 --no-llm --max-steps 200 --out ../logs/design_check.jsonl
/root/autodl-tmp/envs/tau/bin/python ../pool_tau2.py rollout --design ../logs/design_check.jsonl --dry-run 10 --out ../logs/rollout_check.jsonl
```
- 第一条正常：两行"成功率 原 51/100 = 0.51 → 51/100 = 0.51"和"原 67/100 = 0.67 → 67/100 = 0.67"（与工作区结果相同）。
- 第二条正常：9 个位置各一行"候选 n 个 [['orig'], ['gold'], ['rand']]"之类，最后"恢复失败 0"和"== 完成"。
- 第三条正常：10 行"前缀 N 条，下一个 Role.AGENT / Role.USER"，最后"dry-run：检查了前 10 个续跑的恢复，没有调用模型"。
- 第二、三条输出里会夹着 tau2 打印的数据库内容，不用管。
- 有报错就停，把输出发给 Claude。

**D2. 起服务**（挂卡；先停掉 9B，把显存给 27B）：
```
cd /root/paper/repair_budget/tau2_pilot
kill $(cat logs/vllm_8002.pid)
bash serve_tau.sh Qwen3.8-27B-FP8 8003 0.60 qwen3_coder
bash serve_tau.sh Qwen3.5-4B 8001 0.25 qwen3_coder
/root/autodl-tmp/envs/tau/bin/python probe_tools.py 8003 Qwen3.8-27B-FP8 2>&1 | tee logs/probe_8003.log
```
- 9B 没开着的话 `kill` 会报 No such process，不用管。
- A800 没有原生 FP8 计算，vLLM 会用"只量化权重"的方式跑（可能比 bf16 慢一些）；能否加载、用哪个工具调用解析器都要看这一步。27B 起不来就把 `logs/vllm_8003.log` 拉回来。
- 探针要"全部通过"（telecom 里用户也要调工具，所以用户的工具调用解析同样要过）。

**D2b. 小冒烟**（挂卡，约 5 分钟；14:12 加）：探针只检查工具调用能否解析，这一步检查 27B 当用户能把整段对话跑完。
```
cd /root/paper/repair_budget/tau2_pilot
awk 'NR%10==1' ids_telecom.txt > ids_tel_smoke5.txt
MAXSTEPS=200 bash run_tau2.sh 4B telecom 1 smoke2_4b_tel_q27 q27 ids_tel_smoke5.txt
/root/autodl-tmp/envs/tau/bin/python -c "import json; d=json.load(open('tau2-bench/data/simulations/smoke2_4b_tel_q27/results.json')); [print(s['task_id'][:45], s['termination_reason'], '消息', len(s['messages']), '用户工具调用', sum(1 for m in s['messages'] if m['role']=='user' and m.get('tool_calls')), '奖励', (s.get('reward_info') or {}).get('reward')) for s in d['simulations']]"
```
- 取第 1、11、21、31、41 个任务（前 5 个都是同一类 mms 难题，不取）。
- 第二条正常：开头一行"部署者 Qwen3.5-4B…用户 hosted_vllm/Qwen3.8-27B-FP8…步数上限 200；用户重复终止 0（0=关）"，tau2 的进度，最后"退出码 0"并列出 `results.json`。
- 第三条正常：5 行，终止原因大多是 user_stop / agent_stop，"用户工具调用"大多 > 0（telecom 里用户要自己在手机上操作）。
- **停下、把 `logs/run_smoke2_4b_tel_q27.log` 拉回来**的情况：退出码不是 0；有 infrastructure_error；5 段的用户工具调用都是 0；5 段里 ≥ 3 段 max_steps。

**D3. 跑**：
```
nohup bash pilot2_all.sh > logs/pilot2_all.log 2>&1 &
tail -f logs/pilot2_all.log
```
- telecom 50 任务 × 2 次、airline 50 × 1 次，**两步同时跑**，各自并发 32，步数上限 200。预计 1–1.5 卡时（未验证；27B 比 9B 慢、上限 200 会让长对话更长）。
- `logs/pilot2_all.log` 只有开始 / 结束几行；进度看 `tail -f logs/pilot2_tel.out` 或 `logs/pilot2_air.out`（与先导 1 的 run 日志一样）。
- 跑的过程中顺便看一眼 `top`：两个 `tau2 run` 进程的 %CPU 如果一直在 100 左右，就是 CPU 瓶颈（`logs/pilot2_cpu.log` 也会每分钟记一次）。

**D4. 拉回**（你的电脑），然后关卡：
```
python D:\Paper\repair_budget\tools\sync.py pull tau tau2_pilot/logs/pilot2_status.txt tau2_pilot/logs/pilot2_all.log tau2_pilot/logs/pilot2_cpu.log tau2_pilot/logs/pilot2_tel.out tau2_pilot/logs/pilot2_air.out tau2_pilot/logs/probe_8003.log tau2_pilot/logs/vllm_8001.log tau2_pilot/logs/vllm_8003.log tau2_pilot/tau2-bench/data/simulations/p2_4b_tel_q27/results.json tau2_pilot/tau2-bench/data/simulations/p2_4b_air_q27/results.json tau2_pilot/tau2-bench/data/simulations/smoke2_4b_tel_q27/results.json tau2_pilot/logs/run_smoke2_4b_tel_q27.log
```
分析（Claude 做）：成功率；max_steps 比例；`posthoc_tau2.py --cap 100` 与先导 1 对比（同上限下 27B 与 9B 用户比）；`--repeat 3`；与 flash 的 0.82 / 0.72 比；CPU 记录。

## E. 先导 3：2B 部署者 + 共享失败状态（10-30 20:10 北京时间加；全部在 14b2）
目的（用户 10-30 20:07 认可）：① Qwen3.5-2B 在 telecom 上的失败率与失败类型；② 4B、9B 从 2B 的失败状态出发自己接着做，成功率是否在 30–70%（"共享失败状态池"能否成立）；③ 27B 用户在显存给多、并发降低后的速度。
27B 的显存一律 0.62，部署者 2B 0.12 / 4B 0.25 / 9B 0.30（同时只开 27B 和一个部署者）。

**E0. 上传**（你的电脑）：
```
python D:\Paper\repair_budget\tools\sync.py push 14b2 tau2_pilot/run_tau2.sh tau2_pilot/pool_tau2.py tau2_pilot/classify_fail.py
```
**E1. 换模型**（14b2，无卡即可）。先看清楚，再记下 Qwen2.5 的校验和（以后下回来要逐个核对，P0 的 adapter 必须配同一份 7B），拉回本地后再删：
```
ls /root/autodl-tmp/models
du -sh /root/autodl-tmp/models/* /root/autodl-tmp/modelscope_cache
df -h /root/autodl-tmp
cd /root/autodl-tmp/models/Qwen2.5-7B-Instruct
find . -type f | sort | xargs sha256sum > /root/paper/repair_budget/tau2_pilot/logs/sha_qwen25_7b.txt
```
14B 同样做一次（目录名按 `ls` 的结果写），输出到 `logs/sha_qwen25_14b.txt`。拉回本地：
```
python D:\Paper\repair_budget\tools\sync.py pull 14b2 tau2_pilot/logs/sha_qwen25_7b.txt tau2_pilot/logs/sha_qwen25_14b.txt
```
拉回成功后再删除两个目录（`rm -rf /root/autodl-tmp/models/<目录名>`，目录名照 `ls` 抄），然后下载：
```
cd /root/paper/repair_budget/tau2_pilot
/root/autodl-tmp/envs/tau/bin/python download_models.py Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B 2>&1 | tee logs/download_p3.log
```
正常：每个模型一行"config.json 有；safetensors N 个，共 X GB"（2B 约 4–5 GB、9B 约 19 GB），最后 `== 完成`。

**E2. 2B 跑 telecom**（挂卡）：
```
cd /root/paper/repair_budget/tau2_pilot
nvidia-smi
bash serve_tau.sh Qwen3.8-27B-FP8 8003 0.62 qwen3_coder
bash serve_tau.sh Qwen3.5-2B 8004 0.12 qwen3_coder
/root/autodl-tmp/envs/tau/bin/python probe_tools.py 8004 Qwen3.5-2B 2>&1 | tee logs/probe_8004.log
```
- 探针有 FAIL（2B 工具调用不行）就停，把 `logs/probe_8004.log` 发给 Claude。
```
nohup env CONC=20 MAXSTEPS=200 bash run_tau2.sh 2B telecom 2 p3_2b_tel_q27 q27 ids_telecom.txt > logs/p3_2b_tel.out 2>&1 &
tail -f logs/p3_2b_tel.out
```
- 100 段对话，并发 20。正常结尾"退出码 0"并列出 `results.json`。跑的时候可以 `grep "Avg prompt" logs/vllm_8003.log | tail -3` 看 27B 的 Waiting 是否基本为 0、KV 使用率是否不再贴着 100%。
- 跑完分类（只用 CPU，几秒）：
```
/root/autodl-tmp/envs/tau/bin/python classify_fail.py tau2-bench/data/simulations/p3_2b_tel_q27/results.json --dump logs/fails_p3_2b_tel.md
```
  正常打印成功率、部署者工具调用报错率和各类失败的个数，最后"导出 N 段失败对话"。

**E3. 4B、9B 从 2B 的失败状态接着做**（挂卡；27B 保持开着）：
```
kill $(cat logs/vllm_8004.pid)
bash serve_tau.sh Qwen3.5-4B 8001 0.25 qwen3_coder
cd /root/paper/repair_budget/tau2_pilot/tau2-bench
/root/autodl-tmp/envs/tau/bin/python ../pool_tau2.py takeover --src data/simulations/p3_2b_tel_q27/results.json --domain telecom --n-pos 20 --executors 4B --k 4 --user q27 --max-steps 200 --workers 16 --procs 4 --out ../logs/tk_p3_4b.jsonl
```
- 正常：先打印"源对话中符合条件的失败 N 条，取 20 条""位置 20 个……"，再"续跑共 80 次"，每 50 次一行进度，最后"== 完成"。
```
cd /root/paper/repair_budget/tau2_pilot
kill $(cat logs/vllm_8001.pid)
bash serve_tau.sh Qwen3.5-9B 8002 0.30 qwen3_coder
cd tau2-bench
/root/autodl-tmp/envs/tau/bin/python ../pool_tau2.py takeover --src data/simulations/p3_2b_tel_q27/results.json --domain telecom --n-pos 20 --executors 9B --k 4 --user q27 --max-steps 200 --workers 16 --procs 4 --out ../logs/tk_p3_9b.jsonl
```
- 位置的选法是确定的，9B 与 4B 用的是同样 20 个位置。

**E4. 拉回**（你的电脑），然后停服务或关卡：
```
python D:\Paper\repair_budget\tools\sync.py pull 14b2 tau2_pilot/logs/p3_2b_tel.out tau2_pilot/tau2-bench/data/simulations/p3_2b_tel_q27/results.json tau2_pilot/logs/fails_p3_2b_tel.md tau2_pilot/logs/probe_8004.log tau2_pilot/logs/tk_p3_4b.jsonl tau2_pilot/logs/tk_p3_4b.jsonl.design.jsonl tau2_pilot/logs/tk_p3_9b.jsonl tau2_pilot/logs/tk_p3_9b.jsonl.design.jsonl tau2_pilot/logs/vllm_8003.log tau2_pilot/logs/vllm_8004.log tau2_pilot/logs/download_p3.log
```

## F. 先导 5：判分修正后重跑共享失败状态（10-31 00:13 北京时间加；全部在 14b2）
为什么重跑：先导 4 的接手结果判分有偏（`先导3-4结果_tau2.md` 第 4 节），作废。这次 `pool_tau2.py` 用原任务重判、存续跑消息、位置只取第一次错误写之前。
设计：2B 的失败（`p4_2b_ret_q27`、`p4_2b_air_q27`，不用重跑）→ retail 20 个位置、airline 15 个位置（airline 只有 15 条失败）→ 2B / 4B / 9B 各自接着做，每位置 4 次。共 (20 + 15) × 4 × 3 = 420 段续跑。
显存：27B 0.62；部署者 2B 0.20 / 4B 0.25 / 9B 0.35，同时只开 27B 和一个部署者。旧的 `tk_p4_*` 文件留着不动，这次写新文件 `tk5_*`。

**F0. 上传**（你的电脑）：
```
python D:\Paper\repair_budget\tools\sync.py push 14b2 tau2_pilot/pool_tau2.py tau2_pilot/firstbad.py tau2_pilot/run_tk5.sh tau2_pilot/analyze_tk.py
```
**F1. 2B**（14b2，挂卡；在 `/root/paper/repair_budget/tau2_pilot` 下）：
```
cd /root/paper/repair_budget/tau2_pilot
nvidia-smi
bash serve_tau.sh Qwen3.8-27B-FP8 8003 0.62 qwen3_coder
bash serve_tau.sh Qwen3.5-2B 8004 0.20 qwen3_coder
bash run_tk5.sh 2B
tail -f logs/tk5_2b.out
```
- 正常：retail 先打印"……位置规则 before_bad；跳过没有可选位置的 0 条""位置 20 个""续跑共 80 次"，每 50 次一行进度，"== 完成"；airline 同样（位置 15 个、续跑 60 次）；最后"=== … tk5 2B 结束"。
- 有"出错 n"（reward=None，例如连接被拒）不要紧：再运行一次 `bash run_tk5.sh 2B` 只补出错的那几次。

**F2. 4B**（27B 保持开着）：
```
kill $(cat logs/vllm_8004.pid)
bash serve_tau.sh Qwen3.5-4B 8001 0.25 qwen3_coder
bash run_tk5.sh 4B
tail -f logs/tk5_4b.out
```
**F3. 9B**：
```
kill $(cat logs/vllm_8001.pid)
bash serve_tau.sh Qwen3.5-9B 8002 0.35 qwen3_coder
bash run_tk5.sh 9B
tail -f logs/tk5_9b.out
```
- 位置的选法是确定的，三个模型用的是同一批位置（`analyze_tk.py` 会按位置取交集）。

**F4. 汇总**（只用 CPU，几秒；不需要卡，可以先关服务）：
```
kill $(cat logs/vllm_8002.pid) $(cat logs/vllm_8003.pid)
/root/autodl-tmp/envs/tau/bin/python analyze_tk.py --prefix tk5 --out logs/tk5_summary.md
```
- 正常：打印 retail、airline 两张表（各模型平均成功率、出错数、"旧判分不一致"的续跑数、死位置数、配对差值与区间、逐位置成功率）。

**F5. 拉回**（你的电脑），然后关机：
```
python D:\Paper\repair_budget\tools\sync.py pull 14b2 tau2_pilot/logs/tk5_ret_2b.jsonl tau2_pilot/logs/tk5_ret_2b.jsonl.design.jsonl tau2_pilot/logs/tk5_ret_2b.jsonl.msgs.jsonl.gz tau2_pilot/logs/tk5_air_2b.jsonl tau2_pilot/logs/tk5_air_2b.jsonl.design.jsonl tau2_pilot/logs/tk5_air_2b.jsonl.msgs.jsonl.gz tau2_pilot/logs/tk5_2b.out tau2_pilot/logs/tk5_ret_4b.jsonl tau2_pilot/logs/tk5_ret_4b.jsonl.design.jsonl tau2_pilot/logs/tk5_ret_4b.jsonl.msgs.jsonl.gz tau2_pilot/logs/tk5_air_4b.jsonl tau2_pilot/logs/tk5_air_4b.jsonl.design.jsonl tau2_pilot/logs/tk5_air_4b.jsonl.msgs.jsonl.gz tau2_pilot/logs/tk5_4b.out tau2_pilot/logs/tk5_ret_9b.jsonl tau2_pilot/logs/tk5_ret_9b.jsonl.design.jsonl tau2_pilot/logs/tk5_ret_9b.jsonl.msgs.jsonl.gz tau2_pilot/logs/tk5_air_9b.jsonl tau2_pilot/logs/tk5_air_9b.jsonl.design.jsonl tau2_pilot/logs/tk5_air_9b.jsonl.msgs.jsonl.gz tau2_pilot/logs/tk5_9b.out tau2_pilot/logs/tk5_summary.md
```

## G. τ² 探针（10-31 11:49 北京时间加；登记 `D:\Paper\repair_budget\登记_τ²探针_20261031.md`；全部在 14b2）
所有步骤都在 `/root/paper/repair_budget/tau2_pilot` 下用 `run_probe.sh` 运行（脚本内部用 τ² 的 venv）。小文件写在 `tau2_pilot/probe/`，adapter 写在 `/root/autodl-tmp/tau_runs/probe/`，日志在 `probe/logs/<步>.log`，每步结尾一行 "=== … 步 X 结束，退出码 0"。随时可以 `bash run_probe.sh status` 看进度。
**A 层要先跑完，或者先停掉**（`cd ../alfw_stale; bash run_stale.sh kill`，进度不丢，以后 `bash run_stale.sh all` 接着跑）：单卡，两边不能同时用 GPU。

**G0. 上传**（你的电脑）：
```
python D:\Paper\repair_budget\tools\sync.py push 14b2 tau2_pilot/pool_tau2.py tau2_pilot/firstbad.py tau2_pilot/run_tau2.sh tau2_pilot/split_tasks.py tau2_pilot/train_tau2.py tau2_pilot/serve_lora.sh tau2_pilot/probe_analyze.py tau2_pilot/run_probe.sh
```

**G1. LoRA 探针**（挂卡；GPU 上没有别的进程；前台约 15–20 分钟）。先装 peft（只装缺的，不动 torch / transformers / vLLM）：
```
cd /root/paper/repair_budget/tau2_pilot
/root/autodl-tmp/envs/tau/bin/pip install peft accelerate 2>&1 | tail -3
/root/autodl-tmp/envs/tau/bin/python -c "import torch, transformers, vllm, peft; print(torch.__version__, transformers.__version__, vllm.__version__, peft.__version__)"
mkdir -p probe/logs
bash run_probe.sh lora 2>&1 | tee probe/logs/lora.log
```
- pip 那行如果显示要升级或卸载 torch / transformers / vllm，**先停下**把输出发我。第二行应打印 torch 2.13…、transformers 5.19…、vLLM 0.32.0 和 peft 的版本。
- `lora` 正常：打印线性层清单与"LoRA 目标匹配 N 个"、一次前向 + 反向的秒数与显存、三个提示的对比，最后 **"== 通过：vLLM 能加载 adapter，且与 transformers 一致"**。
- 不通过：把 `probe/logs/lora.log` 和 `/root/autodl-tmp/tau_runs/probe/lora_probe/probe_lora.json` 发我。如果是 vLLM 加载 adapter 出错，先试只作用在 MLP：`TARGETS=mlp bash run_probe.sh lora`。

**G2. 收 2B 的新失败**（约 0.5 小时）：
```
bash serve_tau.sh Qwen3.8-27B-FP8 8003 0.62 qwen3_coder
bash serve_tau.sh Qwen3.5-2B 8004 0.20 qwen3_coder
bash run_probe.sh collect
tail -f probe/logs/collect.log
```
- 正常：retail 与 airline 各一段 τ² 的运行输出，开头一行写"种子 301"，各以"退出码 0"结束，结果在 `tau2-bench/data/simulations/probe_2b_ret/`、`probe_2b_air/`。

**G3. 设计**（换成 4B；约 0.5 小时）：
```
kill $(cat logs/vllm_8004.pid)
bash serve_tau.sh Qwen3.5-4B 8001 0.25 qwen3_coder
bash run_probe.sh design
tail -f probe/logs/design.log
```
- 正常：先打印任务划分（retail 训练 37 / 评价 37，airline 25 / 25），然后四次 design，每次开头"任务清单 …""位置规则 before_bad，只留 t < 0.8 × 总长"，每个位置一行"候选 n 个 [来源…]"，结尾"恢复失败 0"与"== 完成"。

**G4. π0 的续跑**（27B 与 4B 保持开着；约 12 小时，可以挂一夜）：
```
bash run_probe.sh pi0
tail -f probe/logs/pi0.log
```
- 正常：四次 rollout（训练位置每分支 3 次、评价位置 6 次），每 50 次一行进度，各以"== 完成"结束；最后 drift 打印"完成 N/N 组"与"== 完成"。有"出错 n"就再运行一次 `bash run_probe.sh pi0`，只补出错的。

**G5. 导出训练样本并核对提示**（4B 服务还开着；前台几分钟，只用 CPU）：
```
bash run_probe.sh export 2>&1 | tee probe/logs/export.log
```
- 正常：先打印"标签：G ≥ 0.2 → N 个"与"写出 … 个样本（修复 N、锚 N…）"，然后 check 打印长度统计和两个例子，**"vLLM /tokenize 核对：8/8 个样本的提示 token 完全相同"**，最后"== 通过"。不通过就停，把 `probe/logs/export.log` 发我。

**G6. 训练 π1**（先停掉所有 vLLM 服务；约 1 小时）：
```
kill $(cat logs/vllm_8001.pid) $(cat logs/vllm_8003.pid)
bash run_probe.sh train
tail -f probe/logs/train.log
```
- 正常：先打印样本数与步数，每 5 步一行 loss（修复样本的 loss 应明显下降）与"预计还要 … 分钟"，最后"完成 … → /root/autodl-tmp/tau_runs/probe/adapter_s1"。G1 若改成了 MLP，这里也要 `TARGETS=mlp bash run_probe.sh train`。

**G7. π1 的续跑**（约 7 小时）：
```
bash serve_tau.sh Qwen3.8-27B-FP8 8003 0.62 qwen3_coder
bash serve_lora.sh Qwen3.5-4B 8001 0.30 4B-S1=/root/autodl-tmp/tau_runs/probe/adapter_s1
bash run_probe.sh pi1
tail -f probe/logs/pi1.log
```
- `serve_lora.sh` 正常：最后一行"就绪……模型列表："里同时有 `Qwen3.5-4B` 和 `4B-S1`。
- `pi1` 正常：两次 rollout（执行者 4B-S1，每分支 6 次）与一次 drift，各以"== 完成"结束。

**G8. 分析与拉回**（先关服务；分析只用 CPU）：
```
kill $(cat logs/vllm_8001.pid) $(cat logs/vllm_8003.pid)
bash run_probe.sh analyze
```
拉回（你的电脑）：
```
python D:\Paper\repair_budget\tools\sync.py pull 14b2 tau2_pilot/probe/report_probe.md tau2_pilot/probe/report_probe.json tau2_pilot/probe/samples_s1.jsonl.summary.json tau2_pilot/probe/logs/lora.log tau2_pilot/probe/logs/collect.log tau2_pilot/probe/logs/design.log tau2_pilot/probe/logs/pi0.log tau2_pilot/probe/logs/export.log tau2_pilot/probe/logs/train.log tau2_pilot/probe/logs/pi1.log
```
（需要细看时我再请你拉设计与续跑文件。）

