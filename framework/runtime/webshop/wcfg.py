"""WebShop Experimental Settings（The third environment；For registration see analysis plan.md 6.13）。

On the server WebShop runs in a separate conda In the environment（Java + pyserini + spaCy etc，and vLLM Environmental isolation），
Main Process（conda environment llm）It communicates with it through the sub-process（webshop/ws_worker.py）。The path can be changed by an environmental variable。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------- path ----------
WEBSHOP_DIR = Path(os.environ.get("WS_DIR", "/root/autodl-tmp/webshop/WebShop"))       # Undo it vendor Location of the bag
WS_PYTHON = os.environ.get("WS_PYTHON", "/root/autodl-tmp/envs/webshop/bin/python")   # WebShop Environmental Interpreter（On the disk conda environment）
# Commodities and attributes：Full items_shuffle.json / items_ins_v2.json Front 10 Ten thousand（Local make_100k.py As-Screen Intercept；6.13 Revision 1）
PRODUCT_FILE = "items_shuffle_100k.json"
ATTR_FILE = "items_ins_v2_100k.json"
HUMAN_FILE = "items_human_ins.json"
DATA_FILES = (PRODUCT_FILE, ATTR_FILE, HUMAN_FILE)
DATA_MANIFEST = "data_100k_manifest.json"     # make_100k.py generated manifest（sha256 etc），setup Check with it
VENDOR = ROOT / "webshop" / "vendor" / "webshop_64fa2a5_min.tar.gz"   # princeton-nlp/WebShop@64fa2a5（MIT），remove images and chromedriver
SHIMS = ROOT / "webshop" / "shims"           # pyserini pure-Python fallback when installation fails Python BM25 Substitution（WS_SEARCH=bm25）
COMPAT = ROOT / "webshop" / "compat"         # gym / selenium Minimum replacement for（The text environment doesnt use them），Always Load
STUBS = ROOT / "webshop" / "test_stubs"      # For use only Claude offline workspace self-tests

# ---------- environment ----------
NUM_PRODUCTS = 100000        # WebShop bundled 10 Ten thousand Configurations（The originals for it indexes_100k）；Objective = People on these commodities write orders
INDEX_NAME = {100: "indexes_100", 1000: "indexes_1k", 100000: "indexes_100k", None: "indexes"}[NUM_PRODUCTS]
RESOURCES_NAME = INDEX_NAME.replace("indexes", "resources")
HUMAN_GOALS = 1
OBS_MODE = "text_rich"       # Take it [button] Tag、Identify Selected Options（"You have clicked ..."）Text-observation
ENV_SEED = 0                 # SimServer Staple Global Before Initialisation random（Prices of inter-priced commodities、Target price ceiling）
MAX_STEPS = 25               # 6.13 Revision 2：Original 15 steps（Llama at 15 Timeout in steps 80%）；shared by smoke tests and formal experiments
HASH_SEED = "0"


def search_backend():
    return os.environ.get("WS_SEARCH", "pyserini")        # pyserini（Original）| bm25（Substitution realization，See shims/）


WORKER_GB = 3.0               # 10 Each in 10,000 pieces WebShop approximate memory per worker（smoke report recording live measurements）


def mem_available_gb():
    """Available Memory（GB）：/proc/meminfo of MemAvailable and the maximum memory of the packaging（cgroup）The rest of it，Take the smaller one。"""
    vals = []
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    vals.append(int(line.split()[1]) / 1048576)
    except OSError:
        pass
    for lim, use in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                     ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            a = open(lim).read().strip()
            b = open(use).read().strip()
            if a.isdigit() and int(a) < (1 << 60):
                vals.append((int(a) - int(b)) / 1073741824)
            break
        except OSError:
            continue
    return min(vals) if vals else None


def n_workers():
    """Number of Subprocesses：WS_WORKERS Assign；otherwise min(16, CPU Count / 2, Half Available Memory / Memory of each subprocess)。It only affects speed，Without prejudice to outcome。"""
    v = os.environ.get("WS_WORKERS")
    if v:
        return max(1, int(v))
    n = max(1, min(16, (os.cpu_count() or 4) // 2))
    mem = mem_available_gb()
    if mem is not None:
        n = max(1, min(n, int(mem * 0.5 / WORKER_GB)))
    return n


# ---------- model（and ALFWorld Same decoding settings：alfw.acfg）----------
MODEL_CONTEXT = 16384         # 6.13 Revision 2：25 The history of the steps is not going to fit in ALFWorld of 8,192；only in WebShop Process change
MAX_NEW_TOKENS = 256
OBS_CHARS_CURRENT = 6000      # The current page is given up to so many characters（Commodity description pages can be long）
OBS_CHARS_HISTORY = 300       # Only so many characters remain for each step of history
MAX_ACTIVE = 128

# ---------- smoke test（Registration：analysis plan.md 6.13，2026-10-26）----------
SMOKE = {
    "SEED": 3001,             # goal-sampling seed（Officially experiment with other seeds，And well rule out the targets that smoked）
    "N_GOALS": 60,            # Each model runs here 60 Target、Each 1 episodes
    "MODELS": ["q7", "q14", "l8"],
    "N_DET_GOALS": 20,        # Replay certainty：20 A random swam on a target
    "DET_STEPS": 12,
    "GATES": {"max_invalid_frac": 0.20, "max_trunc_frac": 0.05, "max_drop_frac": 0.01,
              "q7_success": (0.10, 0.85)},
}
