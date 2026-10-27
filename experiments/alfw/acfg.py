"""ALFWorld Experimental Settings。stage A The standard of judgment is fixed before it runs（2026-10-22 Registration，See analysis plan.md 6.3a）。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_alfw"

# ---------- data ----------
# alfworld-download Read the environment variable, too；AutoDL Up and out of the disk，Avoid taking over the system tray。
DATA = Path(os.environ.get("ALFWORLD_DATA", "/root/autodl-tmp/alfworld_data")).expanduser()
SPLIT_DIRS = {
    "train": "json_2.1.1/train",
    "valid_seen": "json_2.1.1/valid_seen",
    "valid_unseen": "json_2.1.1/valid_unseen",
}
# ALFWorld Articles and ReAct Number used（Filter movable / Sliced After the inexplicable game）
EXPECTED_COUNTS = {"train": 3553, "valid_seen": 140, "valid_unseen": 134}
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

# fast-downward Examples of Every Environment Created，then set libdownward.so Copy to Temporary Directory before Loading（Keep the disk after the file has been deleted，Until the process exits）。
# AutoDL System Disk /tmp the quota is small，Hundreds of examples “Disk quota exceeded”，So the temporary directory is on the data drive。
TMP = Path(os.environ.get("ALFW_TMP", "/root/autodl-tmp/alfw_tmp")).expanduser()

# ---------- environment ----------
MAX_STEPS = 50              # and ReAct / Most ALFWorld Same assessment
HASH_SEED = "0"             # the order of listed objects in observations depends on Python Gather order，It has to be fixed PYTHONHASHSEED

# ---------- model ----------
MODEL_DIR = Path(os.environ.get("ALFW_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"))
MODEL_CONTEXT = 8192
MODEL_SEED = 0
TEMPERATURE = 0.7           # Conditions of deployment
TOP_P = 0.95
MAX_NEW_TOKENS = 256        # One or two Thought + One line Action
STOP = ["\nObservation", "\nThought"]

# ---------- Parallel ----------
def n_workers():
    v = os.environ.get("ALFW_WORKERS")
    if v:
        return max(1, int(v))
    return max(1, min(32, (os.cpu_count() or 4) - 2))

CHUNK = 256                 # advanced concurrently rollout Upper limit of number（Each article takes an environmental example）

# ---------- stage A ----------
A2_N_GAMES = 20             # Replay certainty：train sample from 20 episodes
A2_WALK = 40                # Every game goes randomly 40 steps（approximately 10% Unlawful orders）
A2_SEED = 0
A3_PROBE_STEPS = (0, 10, 20, 30)
A3_BELLMAN_GAMES = 5        # Step-by-step planning for consistency checks（Anecdotal evidence of the most cost-effective）
A4_N_TRAIN = 100            # 7B Baseline：train random 100 episodes + valid_unseen all
A4_SEED = 0
A4_SLACK_EVERY = 5          # Every trajectories of failure 5 Step-by-step planning length，Estimated residual curve
PLAN_TIMEOUT = 60           # Single planning timeout（seconds）。It takes a lot of time to plan（Number of minutes for individual status）；Timeout as“unknown”And report separately

# Pre-established criteria of judgments（Register before running，Change without looking at the result）
GATES = {
    "a2_identical": "New examples of the same process、Same example reset、Accelerating patches、Subprocess（With Hashi seeds）Four resets are gradually fully aligned with the reference",
    "a3_valid_frac": 0.95,          # Proportion of planners giving implementable and target-achieving plans from any state
    "a3_median_latency_s": 2.0,     # One-time median planning ceiling
    "a4_success_min": 0.15,         # train Deployment success rate range on sample
    "a4_success_max": 0.85,
    "a4_bad_action_max": 0.20,      # Invalid + Could not parse action as a step ratio ceiling
    "a4_trunc_max": 0.05,           # Generated cut（finish_reason=length）Scale cap
    "a4_hist_drop_max": 0.01,       # Maximum percentage of steps to discard history due to lack of context
}
# Nominal phase B Size（used only to extrapolate compute requirements）：30 trajectories × 3 Organisation × 7 branches × Each branch 71 Keep going × Average 30 steps
NOMINAL_B_STEPS = 30 * 3 * 7 * 71 * 30

# ---------- stage B（Prior registration：analysis plan.md 6.3b，2026-10-23）----------
OUT_B = ROOT / "out_alfw_b"
B_SEED = 2027
B_N_COLLECT = 300               # train Number of new games drawn（Exclude A2 / A4 Used Game with B_EXCLUDE_TYPES）
B_EXCLUDE_TYPES = ("pick_two_obj_and_place",)   # A4 Medium success rate 4–6%（Floor），Exclusion
B_MAX_SCREEN = 100              # The maximum number of failed trajectories into the filter（Fixed Feed Random）
B_N_POSITIONS = 3               # Intervention position of each trajectory：slack ≥ B_MIN_SLACK split steps into early / middle / Three paragraphs at night，Take one each randomly
B_MIN_SLACK = 5
B_N_RANDOM = 5                  # Every position：Next steps proposed by the planners + 5 A random executable command（Suggested steps are added when the action is equal to the action 1 Random）
B_N_SCREEN = 4                  # Filter Batch：The original action branch is separate from the substitute branch 4 runs
B_SCREEN_MIN_DIFF = 2           # signal detected：An alternative branch in a position is more successful than the original action branch ≥ 2 runs
B_MIN_INFORMATIVE = 20          # If not, stop
B_N_MAIN_MAX = 30               # Maximum number of main experimental trajectories
B_N_DEPLOY = 30                 # Deployment conditions per branch 30 runs（in each of the two folds 15）
B_N_GREEDY = 1
B_PLAN_TIMEOUT = 60             # Planning timeout：Overtime on trajectories position is not optional；Timeout on branch“Completed、Minimum residual”Processing
B_MAX_ACTIVE = 256              # Number of repeat strips advanced at the same time
