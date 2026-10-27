# ALFWorld 阶段 A：可行性检查报告

判断标准在运行前固定（alfw/acfg.py 的 GATES；论文方案.md 6.3a）。

| 部分 | 内容 | 结果 |
|---|---|---|
| A1 | 安装、数据、单局冒烟 | ✓ 通过 |
| A2 | 重放确定性（四种方式逐步一致）与耗时 | ✓ 通过 |
| A3 | 规划器：可执行率 ≥ 0.95、耗时中位数 ≤ 2.0s、无副作用 | ✓ 通过 |
| A4 | 7B 基线：train 成功率 ∈ [0.15, 0.85]、不合法+无动作 ≤ 0.2、截断 ≤ 0.05 | ✓ 通过 |

## A1 安装与数据

- Python 3.12.3；CPU 112；内存 1007.0 GB；数据盘剩余 29.6 GB；PYTHONHASHSEED=0
- 版本：{'alfworld': '0.4.2', 'textworld': '1.7.0', 'fast-downward-textworld': '20.6.4', 'vllm': '0.32.0', 'torch': '2.13.0', 'transformers': '5.19.0', 'numpy': '2.1.3', 'tatsu': '5.8.3', 'jericho': '3.3.1'}
- 游戏数：{'train': 3553, 'valid_seen': 140, 'valid_unseen': 134}
  - train：{'look_at_obj_in_light': 308, 'pick_and_place_simple': 790, 'pick_clean_then_place_in_recep': 650, 'pick_cool_then_place_in_recep': 533, 'pick_heat_then_place_in_recep': 459, 'pick_two_obj_and_place': 813}
  - valid_seen：{'look_at_obj_in_light': 13, 'pick_and_place_simple': 35, 'pick_clean_then_place_in_recep': 27, 'pick_cool_then_place_in_recep': 25, 'pick_heat_then_place_in_recep': 16, 'pick_two_obj_and_place': 24}
  - valid_unseen：{'look_at_obj_in_light': 18, 'pick_and_place_simple': 24, 'pick_clean_then_place_in_recep': 31, 'pick_cool_then_place_in_recep': 21, 'pick_heat_then_place_in_recep': 23, 'pick_two_obj_and_place': 17}
- 单局（look_at_obj_in_light-AlarmClock-None-DeskLamp-308/trial_T20190908_222917_366542）：加载 0.41s，重置 0.38s，可执行命令 21 条，规划 4 步（0.01s）

```
-= Welcome to TextWorld, ALFRED! =-

You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.

Your task is to: examine the alarmclock with the desklamp.
```

规划：`['go to desk 2', 'take alarmclock 1 from desk 2', 'go to desk 1', 'use desklamp 1']`

## A2 重放确定性与耗时

| 检查 | 一致 |
|---|---|
| 每局新建环境实例重放 | ✓ |
| 同一实例 reset 后再重放 | ✓ |
| 加速补丁（翻译缓存 + 语法缓存） | ✓ |
| 新进程（同 PYTHONHASHSEED） | ✓ |

- 不同 PYTHONHASHSEED 的新进程是否一致（说明用；正式运行固定为 0）：True
- 耗时（秒，未加速）：加载 均值 0.752，中位数 0.744，p95 1.23，最大 1.63（n=20）；重置 均值 0.716，中位数 0.737，p95 1.23，最大 1.68（n=20）；单步 均值 0.0815，中位数 0.0849，p95 0.151，最大 0.219（n=800）
- 耗时（秒，加速后）：重置 均值 0.0595，中位数 0.0607，p95 0.132，最大 0.18（n=20）；单步 均值 0.0248，中位数 0.0253，p95 0.0491，最大 0.068（n=800）
- 可执行命令条数：均值 29.2，中位数 30，p95 48，最大 65（n=820）
- 每个环境实例内存约 5.3 MB；同一实例反复加载每次增长 0.00 MB

## A3 规划器

- 80 个探测点（20 局随机游走的第 [0, 10, 20, 30] 步）：计划可执行且到达目标 **1.000**
- 单次规划耗时（秒）：均值 5.88，中位数 0.675，p95 18.3，最大 98.5（n=80）（未加速 均值 6.14，中位数 0.676，p95 18.7，最大 99.2（n=80））
- 计划长度：均值 6.15，中位数 6，p95 9，最大 10（n=80）；加速补丁前后计划相同比例 1.000
- 调用规划器不改变后续转移：✓
- 代价一致性（最优性旁证）：{'pairs': 200, 'cost_violations': 0, 'length_violations': 0}
- 最慢的规划：[{'game': 'pick_two_obj_and_place-PepperShaker-None-Drawer-15/trial_T20190909_022741_011150', 'step': 20, 'len': 10, 'dt': 98.45943956822157}, {'game': 'pick_two_obj_and_place-PepperShaker-None-Drawer-15/trial_T20190909_022741_011150', 'step': 30, 'len': 10, 'dt': 95.53528225794435}, {'game': 'pick_two_obj_and_place-PepperShaker-None-Drawer-15/trial_T20190909_022741_011150', 'step': 10, 'len': 9, 'dt': 68.95828715711832}, {'game': 'pick_two_obj_and_place-PepperShaker-None-Drawer-15/trial_T20190909_022741_011150', 'step': 0, 'len': 9, 'dt': 61.51453562825918}, {'game': 'pick_clean_then_place_in_recep-Potato-None-Microwave-6/trial_T20190907_182137_254302', 'step': 30, 'len': 8, 'dt': 18.343743309378624}]

## A4 7B 基线（温度 0.7，最多 50 步）

“首答不可执行”= 重答之前；“不合法 / 无动作”= 重答后仍不可执行、实际发给环境并消耗一步的比例。

| 划分 | 局数 | 成功率 | 90% 区间 | 首答不可执行 | 每步重答次数 | 不合法 | 无动作 | 别名 | 截断 | 回到旧状态 | 连续重复 | 最长提示 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 100 | 0.390 | [0.314, 0.472] | 0.154 | 0.204 | 0.023 | 0.000 | 0.000 | 0.000 | 0.454 | 0.054 | 5378 |
| valid_unseen | 134 | 0.567 | [0.496, 0.636] | 0.226 | 0.306 | 0.037 | 0.000 | 0.000 | 0.000 | 0.460 | 0.060 | 3882 |
| all | 234 | 0.491 | [0.438, 0.545] | 0.191 | 0.256 | 0.030 | 0.000 | 0.000 | 0.000 | 0.457 | 0.057 | 5378 |

各任务类型成功率：
- train：look_at_obj_in_light 0.83（6）；pick_and_place_simple 0.55（22）；pick_clean_then_place_in_recep 0.65（17）；pick_cool_then_place_in_recep 0.40（20）；pick_heat_then_place_in_recep 0.20（10）；pick_two_obj_and_place 0.04（25）
- valid_unseen：look_at_obj_in_light 0.89（18）；pick_and_place_simple 0.67（24）；pick_clean_then_place_in_recep 0.58（31）；pick_cool_then_place_in_recep 0.57（21）；pick_heat_then_place_in_recep 0.57（23）；pick_two_obj_and_place 0.06（17）
- 初始状态最优计划长度：成功局 均值 5.28，中位数 6，p95 7，最大 8（n=114）；失败局 均值 6.54，中位数 6，p95 9，最大 9（n=114）

失败的 train 局上，第 t 步时“剩余步数 − 规划长度”（余量；≥ 0 即预算内可完成）：

| t | 局数 | 可完成比例 | 余量中位数 |
|---:|---:|---:|---:|
| 0 | 57 | 1.00 | 44 |
| 5 | 58 | 1.00 | 39.0 |
| 10 | 58 | 1.00 | 34.0 |
| 15 | 58 | 1.00 | 29.0 |
| 20 | 56 | 1.00 | 24.0 |
| 25 | 58 | 1.00 | 19.0 |
| 30 | 56 | 1.00 | 14.0 |
| 35 | 58 | 1.00 | 9.0 |
| 40 | 57 | 1.00 | 4 |
| 45 | 59 | 0.32 | -1 |

- 最后一个可完成的探测步（每 5 步一测）：均值 41.1，中位数 40，p95 45，最大 45（n=59）；从一开始就不可完成的比例 0.00

- 耗时：模型加载 33s；rollout 325s（生成 284s，环境 19s，打开 18s），共 7531 步、50 轮；规划探测 844 次 752s（超时 41 次；单次耗时 均值 4.4，中位数 0.79，p95 23.3，最大 59.2（n=803））
- **算力外推**：每 10 万步约 1.20 小时；名义阶段 B（1,341,900 步）约 16.1 小时（本次同时推进 234 条；阶段 B 同时推进更多、共享前缀更多，实际应更快）
- 判断：{'success_in_range': True, 'bad_action_ok': True, 'trunc_ok': True, 'hist_drop_ok': True}

## 按事先规则的结论

- **全部通过**：可以进入阶段 B 的设计与登记（采集失败轨迹 → 筛选 → 主实验）。
