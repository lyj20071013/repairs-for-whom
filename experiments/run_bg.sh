#!/usr/bin/env bash
# Lowest cross-exposure on backstage。use setsid + nohup Departure from the current SSH Session：Turn it off SSH、closing the local computer does not interrupt it。
#
#   bash run_bg.sh ds-alfw --stage smoke|prep7|prep14|run7|analyze  # ALFWorld unscreened failures × Candidates for the model × 7B / 14B（6.9；See README）
#   bash run_bg.sh ds-alfw --profile l8 --stage smoke  /  bash run_bg.sh ds-alfw --stage p610  # 6.10：Second deployment model Llama-3.1-8B（See README）
#   bash run_bg.sh ladder-alfw --evaluator q14  # ALFWorld Followers ladder：14B The running phase B The main experiment is the same branch（approximately 3 hours，out_ladder_alfw/ladder_report.md）
#   bash run_bg.sh ladder-gw --evaluator q14  # gridworld Followers ladder + Reliance pre-screening by the executor：14B Run to confirm the same branch of the experiment（approximately 1–1.5 hours，out_ladder_gw/ladder_report.md）
#   bash run_bg.sh ds-gw --executor q7    # gridworld Datasets：collect/design q7 The pool and evaluation（Single card in order of order q7、q14、q7；Double cards can be added at the same time --wait）
#   bash run_bg.sh alfw-b       # ALFWorld stage B One key：self-test -> smoke test（real 7B Very small，approximately 15–20 minutes）-> And then well just run through the formal experiment（out_alfw_b/report.md）
#   bash run_bg.sh alfw-b-prep  # ALFWorld stage B First paragraph（approximately 3–4 hours）：collect -> design -> screening -> selection（out_alfw_b/screen_result.json）
#   bash run_bg.sh alfw-b-run   # ALFWorld stage B Second paragraph（approximately 5–8 hours）：Main experiment -> Analysis（out_alfw_b/report.md）；Well run again after the break
#   bash run_bg.sh alfw-a       # ALFWorld stage A（approximately 30–60 minutes）：self-test -> installation/data -> Replay certainty -> planner -> 7B Baseline（out_alfw/stageA_report.md）
#   bash run_bg.sh mechanism    # Mechanism follow-up experiments（approximately 30 minutes）：greedy+Run away，Reuse out_confirm data（out_confirm/mechanism_report.md）
#   bash run_bg.sh confirm-prep # Confirm phase one of the experiment（approximately 25 minutes）：self-test -> New trajectories capture -> screening（out_confirm/screen_result.json）
#   bash run_bg.sh confirm-run  # Confirm phase two of the experiment（approximately 3 hours）：Main experiment -> Analysis（out_confirm/confirm_report.md）
#   bash run_bg.sh confirm      # Confirm the experiment（approximately 2.5 hours）：self-test -> New trajectories capture -> screening（Not enough 20 Bar automatically stops）-> Main experiment -> Analysis（out_confirm/）
#   bash run_bg.sh harness      # Frame Check（approximately 5 minutes）：Quality of step-by-step decision-making，Existing prompts vs Let go of the reasoning（Output in out_harness/）
#   bash run_bg.sh quick        # Rapidly confirm the experiment（approximately 30–40 minutes）：Check on the new trajectory H1–H3（Output in out_quick/）
#   bash run_bg.sh trial        # The floor effect is running：self-test -> 5 trajectories Design -> Selection of deployment conditions -> inspect（Output in out_try/）
#   bash run_bg.sh all          # Official experiments：self-test -> design -> plan -> All of them -> analyze（Output in out/）
#   bash run_bg.sh rollouts     # Just keep running（The unfinished part will run again）
#   bash run_bg.sh analyze      # Analysis only
#   bash run_bg.sh selftest | design | plan | inspect
#   bash run_bg.sh status       # Whether or not to run + recent logs + GPU
#   bash run_bg.sh log          # View logs on an ongoing basis（Ctrl-C Exit View，It wont stop the mission）
#   bash run_bg.sh kill         # stop
#
# forward all remaining arguments unchanged to run_cross.py，for example：bash run_bg.sh rollouts --condition llm_T03 --batch S
# if any stage fails，The next steps wont be carried out；The result is down by branch，The same order will run again。

set -uo pipefail
cd "$(dirname "$0")" || exit 1
LOG_DIR="logs"; mkdir -p "$LOG_DIR"

# Interpreter：Item venv > current PATH Medium python（conda environment）> python3
if [ -x ".venv/bin/python" ]; then PY="$(pwd)/.venv/bin/python"
elif [ -x "../pilot_vshape/.venv/bin/python" ]; then PY="$(cd ../pilot_vshape && pwd)/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then PY="$(command -v python)"
else PY="$(command -v python3)"; fi

PIPE_LOG="$LOG_DIR/_cross.log"
PID_FILE="$LOG_DIR/_cross.pid"

is_running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

case "${1:-}" in
status)
    if is_running; then echo "running，pid=$(cat "$PID_FILE")"; else echo "not run"; fi
    echo "--- recent logs ($PIPE_LOG) ---"
    tail -n 25 "$PIPE_LOG" 2>/dev/null || echo "(no logs)"
    echo "--- GPU ---"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>/dev/null || true
    exit 0 ;;
log)
    exec tail -n 40 -f "$PIPE_LOG" ;;
kill)
    if is_running; then
        P=$(cat "$PID_FILE")
        pkill -TERM -g "$P" 2>/dev/null || { pkill -TERM -P "$P" 2>/dev/null; kill -TERM "$P" 2>/dev/null; }
        rm -f "$PID_FILE"; echo "stopped process group $P"
    else
        echo "no task is running"
    fi
    exit 0 ;;
"")
    sed -n '2,28p' "$0"; exit 0 ;;
esac

STAGE="$1"; shift
case "$STAGE" in ds-alfw|ladder-alfw|ladder-gw|ds-gw|alfw-a|alfw-b|alfw-b-prep|alfw-b-run|mechanism|confirm|confirm-prep|confirm-run|harness|quick|trial|all|selftest|design|plan|rollouts|inspect|analyze) ;;
    *) echo "unknown stage: $STAGE"; sed -n '2,28p' "$0"; exit 2 ;; esac
if is_running; then
    echo "a task is already running（pid=$(cat "$PID_FILE")）。first bash run_bg.sh status inspect，or bash run_bg.sh kill stop。"
    exit 1
fi
EXTRA=""
if [ "$#" -gt 0 ]; then printf -v EXTRA ' %q' "$@"; fi

RUNNER="$LOG_DIR/_cross_runner.sh"
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$(pwd)"
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
step() {
    echo "[\$(date '+%F %T')] === START \$* ==="
    "$PY" run_cross.py "\$@"
    local rc=\$?
    echo "[\$(date '+%F %T')] === END \$1 \$2 rc=\$rc ==="
    if [ \$rc -ne 0 ]; then echo "[\$(date '+%F %T')] failed，subsequent stages will not run"; exit \$rc; fi
}
quick_step() {
    echo "[\$(date '+%F %T')] === START quick ==="
    "$PY" quick.py --stage all
    local rc=\$?
    echo "[\$(date '+%F %T')] === END quick rc=\$rc ==="
    if [ \$rc -ne 0 ]; then echo "[\$(date '+%F %T')] failed"; exit \$rc; fi
}
confirm_step() {
    echo "[\$(date '+%F %T')] === START confirm \$1 ==="
    "$PY" confirm.py --stage "\$1"$EXTRA
    local rc=\$?
    echo "[\$(date '+%F %T')] === END confirm \$1 rc=\$rc ==="
    if [ \$rc -ne 0 ]; then echo "[\$(date '+%F %T')] failed，subsequent stages will not run"; exit \$rc; fi
}
case "$STAGE" in
    ds-alfw)
        export PYTHONHASHSEED=0
        export ALFWORLD_DATA="\${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
        echo "[\$(date '+%F %T')] === START ds-alfw$EXTRA (ALFWORLD_DATA=\$ALFWORLD_DATA, LADDER_TP=\${LADDER_TP:-1}) ==="
        "$PY" ds_alfw.py --stage selftest || { echo "[\$(date '+%F %T')] self-test failed，stop"; exit 1; }
        "$PY" ds_alfw.py$EXTRA
        rc=\$?
        echo "[\$(date '+%F %T')] === END ds-alfw rc=\$rc ==="
        exit \$rc ;;
    ladder-alfw)
        export PYTHONHASHSEED=0
        export ALFWORLD_DATA="\${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
        echo "[\$(date '+%F %T')] === START ladder-alfw$EXTRA (LADDER_TP=\${LADDER_TP:-1}) ==="
        "$PY" ladder_alfw.py --stage selftest || { echo "[\$(date '+%F %T')] self-test failed，stop"; exit 1; }
        "$PY" ladder_alfw.py --stage run$EXTRA || { echo "[\$(date '+%F %T')] continuation execution failed，stop（The same order will run again）"; exit 1; }
        "$PY" ladder_alfw.py --stage analyze
        rc=\$?
        echo "[\$(date '+%F %T')] === END ladder-alfw rc=\$rc ==="
        exit \$rc ;;
    ladder-gw)
        echo "[\$(date '+%F %T')] === START ladder-gw$EXTRA (LADDER_TP=\${LADDER_TP:-1}) ==="
        "$PY" ladder_gw.py --stage selftest || { echo "[\$(date '+%F %T')] self-test failed，stop"; exit 1; }
        "$PY" ladder_gw.py --stage run$EXTRA || { echo "[\$(date '+%F %T')] continuation execution failed，stop（The same order will run again）"; exit 1; }
        "$PY" ladder_gw.py --stage analyze
        rc=\$?
        echo "[\$(date '+%F %T')] === END ladder-gw rc=\$rc ==="
        exit \$rc ;;
    ds-gw)
        echo "[\$(date '+%F %T')] === START ds-gw$EXTRA ==="
        "$PY" ds_gw.py --stage selftest || { echo "[\$(date '+%F %T')] self-test failed，stop"; exit 1; }
        "$PY" ds_gw.py --stage all$EXTRA
        rc=\$?
        echo "[\$(date '+%F %T')] === END ds-gw rc=\$rc ==="
        exit \$rc ;;
    alfw-b)
        export PYTHONHASHSEED=0
        export ALFWORLD_DATA="\${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
        echo "[\$(date '+%F %T')] === START alfw-b (ALFWORLD_DATA=\$ALFWORLD_DATA) ==="
        "$PY" alfw_stage_b.py --stage selftest || { echo "[\$(date '+%F %T')] self-test failed，stop"; exit 1; }
        "$PY" alfw_stage_b.py --stage smoke$EXTRA || { echo "[\$(date '+%F %T')] The smoke didnt pass，stop（Look out_alfw_b_smoke/）"; exit 1; }
        echo "[\$(date '+%F %T')] smoke test passed，Start Full Operation"
        "$PY" alfw_stage_b.py --stage all$EXTRA
        rc=\$?
        echo "[\$(date '+%F %T')] === END alfw-b rc=\$rc ==="
        exit \$rc ;;
    alfw-b-prep|alfw-b-run)
        export PYTHONHASHSEED=0
        export ALFWORLD_DATA="\${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
        echo "[\$(date '+%F %T')] === START $STAGE (ALFWORLD_DATA=\$ALFWORLD_DATA) ==="
        "$PY" alfw_stage_b.py --stage selftest
        rc=\$?
        if [ \$rc -ne 0 ]; then echo "[\$(date '+%F %T')] self-test failed，stop"; exit \$rc; fi
        "$PY" alfw_stage_b.py --stage $( [ "$STAGE" = "alfw-b-prep" ] && echo prep || echo run )$EXTRA
        rc=\$?
        echo "[\$(date '+%F %T')] === END $STAGE rc=\$rc ==="
        exit \$rc ;;
    alfw-a)
        export PYTHONHASHSEED=0
        export ALFWORLD_DATA="\${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
        echo "[\$(date '+%F %T')] === START alfw-a (ALFWORLD_DATA=\$ALFWORLD_DATA) ==="
        "$PY" alfw_stage_a.py --part all$EXTRA
        rc=\$?
        echo "[\$(date '+%F %T')] === END alfw-a rc=\$rc ==="
        exit \$rc ;;
    mechanism)
        step --stage selftest
        echo "[\$(date '+%F %T')] === START mechanism ==="
        "$PY" mechanism.py --stage all$EXTRA
        rc=\$?
        echo "[\$(date '+%F %T')] === END mechanism rc=\$rc ==="
        exit \$rc ;;
    confirm-prep)
        step --stage selftest
        confirm_step prep ;;
    confirm-run)
        step --stage selftest
        confirm_step run ;;
    confirm)
        step --stage selftest
        echo "[\$(date '+%F %T')] === START confirm ==="
        "$PY" confirm.py --stage all$EXTRA
        rc=\$?
        echo "[\$(date '+%F %T')] === END confirm rc=\$rc ==="
        exit \$rc ;;
    harness)
        echo "[\$(date '+%F %T')] === START harness ==="
        "$PY" harness_check.py$EXTRA
        rc=\$?
        echo "[\$(date '+%F %T')] === END harness rc=\$rc ==="
        exit \$rc ;;
    quick)
        quick_step ;;
    trial)
        step --stage selftest
        step --stage design   --out-dir out_try --n-traj 5$EXTRA
        step --stage rollouts --out-dir out_try --condition deploy_T07 --batch S$EXTRA
        step --stage inspect  --out-dir out_try ;;
    all)
        step --stage selftest
        step --stage design$EXTRA
        step --stage plan$EXTRA
        step --stage rollouts$EXTRA
        step --stage inspect$EXTRA
        step --stage analyze$EXTRA ;;
    *)
        step --stage $STAGE$EXTRA ;;
esac
echo "[\$(date '+%F %T')] ALL DONE"
EOF
chmod +x "$RUNNER"

# setsid：New Session、New Process Group，and SSH Terminal Decoupling；nohup：Ignore Hangup Signal；Enter From /dev/null。
setsid nohup bash "$RUNNER" >> "$PIPE_LOG" 2>&1 < /dev/null &
echo "$!" > "$PID_FILE"
echo "Started in the backstage [$STAGE]，pid=$(cat "$PID_FILE")"
echo "  View Progress： bash run_bg.sh status    Keep looking at the log： bash run_bg.sh log    stop： bash run_bg.sh kill"
echo "  You can just turn it off now SSH。"
