#!/usr/bin/env bash
# ALFWorld（text mode）installation + data download。in the installation containing vLLM the same Python run in the environment：
#
#   bash alfw_setup.sh
#
# 1. pin numpy / torch / vllm / transformers and other installed versions（textworld -> jericho -> spacy may attempt to upgrade numpy，would break vLLM）；
# 2. pip install alfworld（install text-mode dependencies only：textworld[pddl]、fast-downward-textworld etc）；if a pinned version must change，pip report the conflict and stop；
# 3. verify that pinned versions are unchanged、vllm can still be imported；
# 4. download data to $ALFWORLD_DATA（default /root/autodl-tmp/alfworld_data，approximately 1.7GB after extraction）。skip if already present。
set -uo pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p logs

if [ -x ".venv/bin/python" ]; then PY="$(pwd)/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then PY="$(command -v python)"
else PY="$(command -v python3)"; fi
echo "Python: $PY ($($PY --version 2>&1))"

export ALFWORLD_DATA="${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
KEEP=logs/alfw_keep_versions.txt
$PY -m pip freeze 2>/dev/null | grep -iE '^(numpy|torch|torchvision|torchaudio|vllm|transformers|tokenizers|triton|xformers|flashinfer[-_a-z0-9]*|pydantic)==' > "$KEEP"
echo "--- pin these installed versions（$KEEP）---"; cat "$KEEP"

if $PY -c "import alfworld, textworld, fast_downward" 2>/dev/null; then
    echo "alfworld / textworld / fast_downward installed，skip pip。"
else
    echo "--- pip install alfworld（with version constraints）---"
    if ! $PY -m pip install alfworld -c "$KEEP" 2>&1 | tee logs/alfw_pip.log; then
        echo "pip installation failed（likely a conflict with pinned versions）。provide logs/alfw_pip.log the last 40 lines for diagnosis。"; exit 1
    fi
    if grep -qiE "^ERROR|ResolutionImpossible" logs/alfw_pip.log; then
        echo "pip error，provide logs/alfw_pip.log the last 40 lines for diagnosis。"; exit 1
    fi
fi

echo "--- verify pinned versions are unchanged ---"
NOW=logs/alfw_versions_after.txt
$PY -m pip freeze 2>/dev/null | grep -iE '^(numpy|torch|torchvision|torchaudio|vllm|transformers|tokenizers|triton|xformers|flashinfer[-_a-z0-9]*|pydantic)==' > "$NOW"
if ! diff "$KEEP" "$NOW"; then
    echo "!!! the package versions listed above changed。do not start experiments yet，retain this output for diagnosis。"; exit 1
fi
echo "versions unchanged。"
$PY - <<'EOF' || { echo "import check failed，retain the error above for diagnosis。"; exit 1; }
import importlib.metadata as md
import textworld, fast_downward
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
import vllm
print("textworld", md.version("textworld"), "| alfworld", md.version("alfworld"),
      "| fast-downward-textworld", md.version("fast-downward-textworld"), "| vllm", md.version("vllm"))
EOF

if [ "${ALFW_SKIP_DATA:-0}" = "1" ]; then
    echo "ALFW_SKIP_DATA=1：skip data download（use ladder_alfw.py --stage pack bundled 30 games）。"
elif [ -d "$ALFWORLD_DATA/json_2.1.1/train" ] && [ -n "$(find "$ALFWORLD_DATA/json_2.1.1/valid_unseen" -name game.tw-pddl -print -quit 2>/dev/null)" ]; then
    echo "data already exists：$ALFWORLD_DATA，skip download。"
else
    echo "--- download data to $ALFWORLD_DATA ---"
    mkdir -p "$ALFWORLD_DATA"
    # AutoDL access GitHub use an alternative download route if access is slow（Only in this script）
    if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
    ALFDL="$(dirname "$PY")/alfworld-download"
    [ -x "$ALFDL" ] || ALFDL="alfworld-download"
    if ! "$ALFDL" --data-dir "$ALFWORLD_DATA"; then
        echo "Data download failed。But run back to the script（The part that has been downloaded will skip）；if it still fails, retain the error for diagnosis。"; exit 1
    fi
fi
echo "--- complete。Next：bash run_bg.sh alfw-a ---"
