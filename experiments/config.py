"""repair_transfer Global Configuration。

Reuse this directory pilot_vshape Environment and strategy：env.py、model.py、diagnose.py Copy as，
selftest Theyll check them with pilot_vshape The version in which the bytes are the same（If that directory exists,）。
model.py / diagnose.py The read constant name remains the same here，Value with pilot identical。
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PILOT_DIR = ROOT.parent / "pilot_vshape"
OUT = ROOT / "out"

# ---------- model（and pilot_vshape identical；model.py Read these names directly） ----------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR = Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
MODEL_CONTEXT = 4096
MODEL_SEED = 0
TEMPERATURE = 0.7          # Temperature of deployment conditions
TOP_P = 0.95
MAX_NEW_TOKENS = 160
MAX_NEW_TOKENS_JSON_ONLY = 24

# ---------- environment（diagnose.py Read） ----------
GRID = 5
MAX_STEPS = 30
N_TRAPS = 1
N_WALLS = 2

# ---------- Freezing the origin of the trajectory ----------
SOURCE_EPISODES = PILOT_DIR / "out_120" / "episodes.jsonl"
# already inspected V The curves trajectory does not enter the official experiment，Avoid selecting samples on the basis of the results。
PRIOR_INSPECTED = [PILOT_DIR / "out_120" / "summary.json",
                   PILOT_DIR / "out_120" / "curves_altT_summary.json"]

# ---------- design（Environment Only oracle And fixed seed decisions，Freeze before running a model） ----------
N_TRAJ = 20                # Number of failed trajectories
TRAJ_SEED = 0
N_POSITIONS = 4            # Number of intervention locations per trajectory（Retrieved from budget deconstructible positions）
POSITION_SEED = 0
# Can not open message：All Legal Actions，Exclude give_up And the original action itself。Do not rely on model proposals。
CANDIDATE_ACTIONS = [{"tool": "move", "arg": d} for d in ("up", "down", "left", "right")] + [
    {"tool": "pick"}, {"tool": "open"}, {"tool": "look"}]

# ---------- Relay terms ----------
# kind: llm（7B，Sample by temperature）| oracle（environment-optimal continuation，success if and only if F=1）
#       | noisy_oracle（By eps Probability Random Actions，otherwise follow the optimal policy；CPU Run）
DEPLOY = "deploy_T07"      # Conditions of deployment c0
CONDITIONS = {
    "deploy_T07": {"kind": "llm", "temperature": 0.7},
    "llm_T03": {"kind": "llm", "temperature": 0.3},
    "llm_greedy": {"kind": "llm", "temperature": 0.0},
    "oracle": {"kind": "oracle"},
    "noisy_oracle_e30": {"kind": "noisy_oracle", "eps": 0.3},
    # Institutional experiments：greedy，but when revisiting a physical state already reached in this continuation，use this temperature for the current step 0.7 Sample
    "llm_greedy_escape": {"kind": "llm", "temperature": 0.0, "escape_temperature": 0.7},
}
# Batch per condition and sample per branch。S = Select Batch；V = Independent validation batch under deployment conditions。
# Terms of deployment S and V use different random seeds，S the selected repair under V Up-to-date evaluation，Composition“Matching Condition Selection”Lower Noise Limit。
BATCHES = {
    "deploy_T07": {"S": 50, "V": 100},
    "llm_T03": {"S": 50},
    "llm_greedy": {"S": 2},          # greedy decoding is pretty much certain；Take 2 Sub-check for certainty
    "oracle": {"S": 1},
    "noisy_oracle_e30": {"S": 50},
}
VALID_BATCH = "V"
# Use only S Previous to Batch k Sub-results，mock“Assessment of deployment conditions for a small sample”；No extra math。
PRECISION_PROXIES = [5, 10]

# ---------- Analysis（Stretch before running model） ----------
OBJECTIVE = "net"          # net: R = p(Substitution) - p(original action)；post: Success rate after restoration p(Substitution)
# Main issues（2026-10-22 Select before running model）：Can you measure it? ≥ DELTA Losses on relocation，and whether it can rule out ≥ DELTA Losses；
# not based on“Equivalent”Target。Efficacy Simulation 20 The area of loss below the trajectories is wide open 0.08–0.14，0.05 equivalence cannot be confirmed。
DELTA = 0.10               # Meaningful migration loss threshold（Probability of success，Absolute value [u]）
ALPHA = 0.05               # Use 1-2*ALPHA（90%）of bootstrap interval，One-sided 5% judgments
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

# ---------- Run ----------
ROLLOUT_CHUNK = 800        # Once run_rollouts How many?（and pilot of VCURVE_BATCH identical）

DESIGN_JSON = OUT / "design.json"
ROLLOUTS_JSONL = OUT / "rollouts.jsonl"
ANALYSIS_JSON = OUT / "analysis.json"
REPORT_MD = OUT / "report.md"


def set_output_dir(path):
    global OUT, DESIGN_JSON, ROLLOUTS_JSONL, ANALYSIS_JSON, REPORT_MD
    OUT = Path(path).expanduser().resolve()
    DESIGN_JSON, ROLLOUTS_JSONL = OUT / "design.json", OUT / "rollouts.jsonl"
    ANALYSIS_JSON, REPORT_MD = OUT / "analysis.json", OUT / "report.md"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_id():
    """Strategic semantic signature：model、Decoding base settings and environment/prompts Source。model.AgentPolicy Do it cache_id。"""
    names = ("MODEL_ID", "MODEL_DIR", "MODEL_CONTEXT", "MODEL_SEED", "GRID", "MAX_STEPS",
             "N_TRAPS", "N_WALLS", "TOP_P", "MAX_NEW_TOKENS")
    return fingerprint({"config": {k: str(globals()[k]) for k in names},
                        "sources": {n: file_sha256(ROOT / n) for n in ("env.py", "model.py")}})
