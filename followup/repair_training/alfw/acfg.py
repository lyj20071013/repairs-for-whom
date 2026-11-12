"""ALFWorld 实验设置。阶段 A 的判断标准在运行前固定（2026-10-22 登记，见 论文方案.md 6.3a）。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_alfw"

# ---------- 数据 ----------
# alfworld-download 也读这个环境变量；AutoDL 上放数据盘，避免占系统盘。
DATA = Path(os.environ.get("ALFWORLD_DATA", "/root/autodl-tmp/alfworld_data")).expanduser()
SPLIT_DIRS = {
    "train": "json_2.1.1/train",
    "valid_seen": "json_2.1.1/valid_seen",
    "valid_unseen": "json_2.1.1/valid_unseen",
}
# ALFWorld 论文与 ReAct 使用的数量（过滤 movable / Sliced 与不可解游戏之后）
EXPECTED_COUNTS = {"train": 3553, "valid_seen": 140, "valid_unseen": 134}
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

# fast-downward 每创建一个环境实例，就把 libdownward.so 复制到临时目录再加载（文件删除后仍占磁盘，直到进程退出）。
# AutoDL 系统盘的 /tmp 配额很小，几百个实例就会 “Disk quota exceeded”，所以临时目录放到数据盘。
TMP = Path(os.environ.get("ALFW_TMP", "/root/autodl-tmp/alfw_tmp")).expanduser()

# ---------- 环境 ----------
MAX_STEPS = 50              # 与 ReAct / 多数 ALFWorld 评测相同
HASH_SEED = "0"             # 观测中物品列举顺序依赖 Python 集合顺序，必须固定 PYTHONHASHSEED

# ---------- 模型 ----------
MODEL_DIR = Path(os.environ.get("ALFW_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"))
MODEL_CONTEXT = 8192
MODEL_SEED = 0
TEMPERATURE = 0.7           # 部署条件
TOP_P = 0.95
MAX_NEW_TOKENS = 256        # 一两句 Thought + 一行 Action
STOP = ["\nObservation", "\nThought"]

# ---------- 并行 ----------
def n_workers():
    v = os.environ.get("ALFW_WORKERS")
    if v:
        return max(1, int(v))
    return max(1, min(32, (os.cpu_count() or 4) - 2))

CHUNK = 256                 # 同时推进的 rollout 数上限（每条占一个环境实例）

# ---------- 阶段 A ----------
A2_N_GAMES = 20             # 重放确定性：train 中抽 20 局
A2_WALK = 40                # 每局随机游走 40 步（约 10% 为不合法命令）
A2_SEED = 0
A3_PROBE_STEPS = (0, 10, 20, 30)
A3_BELLMAN_GAMES = 5        # 逐步规划一致性检查（代价最优性的旁证）
A4_N_TRAIN = 100            # 7B 基线：train 随机 100 局 + valid_unseen 全部
A4_SEED = 0
A4_SLACK_EVERY = 5          # 失败轨迹上每 5 步求一次规划长度，估计余量曲线
PLAN_TIMEOUT = 60           # 单次规划超时（秒）。规划耗时长尾很重（个别状态数分钟）；超时记为“未知”并单独报告

# 事先固定的判断标准（运行前登记，不看结果后修改）
GATES = {
    "a2_identical": "同进程新实例、同实例 reset、加速补丁、子进程（同哈希种子）四种重放与参考逐步完全一致",
    "a3_valid_frac": 0.95,          # 规划器从任意状态给出可执行且到达目标的计划的比例
    "a3_median_latency_s": 2.0,     # 单次规划耗时中位数上限
    "a4_success_min": 0.15,         # train 抽样上的部署成功率区间
    "a4_success_max": 0.85,
    "a4_bad_action_max": 0.20,      # 不合法 + 无法解析动作占步骤比例上限
    "a4_trunc_max": 0.05,           # 生成被截断（finish_reason=length）比例上限
    "a4_hist_drop_max": 0.01,       # 因上下文不足丢弃历史的步骤比例上限
}
# 名义阶段 B 规模（只用于算力外推）：30 条轨迹 × 3 个位置 × 7 个分支 × 每分支 71 次续跑 × 平均 30 步
NOMINAL_B_STEPS = 30 * 3 * 7 * 71 * 30

# ---------- 阶段 B（事先登记：论文方案.md 6.3b，2026-10-23）----------
OUT_B = ROOT / "out_alfw_b"
B_SEED = 2027
B_N_COLLECT = 300               # train 中新抽的游戏数（排除 A2 / A4 用过的游戏与 B_EXCLUDE_TYPES）
B_EXCLUDE_TYPES = ("pick_two_obj_and_place",)   # A4 中成功率 4–6%（地板），事先排除
B_MAX_SCREEN = 100              # 进入筛选的失败轨迹数上限（固定种子随机取）
B_N_POSITIONS = 3               # 每条轨迹的干预位置：余量 ≥ B_MIN_SLACK 的步分早 / 中 / 晚三段，各随机取一个
B_MIN_SLACK = 5
B_N_RANDOM = 5                  # 每个位置：规划器建议的下一步 + 5 条随机可执行命令（建议步等于原动作时补 1 条随机）
B_N_SCREEN = 4                  # 筛选批次：原动作分支与替代分支各 4 次
B_SCREEN_MIN_DIFF = 2           # 有信号：某位置上某替代分支比原动作分支多成功 ≥ 2 次
B_MIN_INFORMATIVE = 20          # 不足则停止
B_N_MAIN_MAX = 30               # 主实验轨迹数上限
B_N_DEPLOY = 30                 # 部署条件每分支 30 次（两折各 15）
B_N_GREEDY = 1
B_PLAN_TIMEOUT = 60             # 规划超时：轨迹位置上超时则该位置不可选；分支上超时按“可完成、余量最低”处理
B_MAX_ACTIVE = 256              # 同时推进的续跑条数
