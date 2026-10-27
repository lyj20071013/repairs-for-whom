#!/usr/bin/env bash
# WebShop Environment（7B Do it on the server once；You can double-run it：Completed steps will skip，Stop that step when you fail and explain why）。
#
#   cd /root/paper/repair_transfer
#   bash webshop/setup_webshop.sh
#
# operations performed（Finish printing each step ✓ or ✗）：
#   1. Created in Data Disk conda environment /root/autodl-tmp/envs/webshop（Python 3.12；and vLLM Environment llm Separate）
#   2. Load Java 21（conda-forge openjdk；pyserini / Lucene Yes）
#   3. Load WebShop Text environment Python Bag（bs4、flask、spaCy、thefuzz etc）
#   4. Load pyserini（Original Retrieval）：install first CPU Version torch（approximately 200 MB），otherwise pip Well wear it CUDA Version torch and nvidia-* How many? GB；
#      It was half loaded last time CUDA Itll be removed first
#   5. Load spaCy of en_core_web_sm model（from GitHub Download；enable first AutoDL Academic acceleration）
#   6. Undo it webshop/vendor/ Yeah WebShop Code（princeton-nlp/WebShop@64fa2a5），Put data in：
#      webshop_data/ in make_100k.py Generated items_shuffle_100k.json.gz、items_ins_v2_100k.json.gz（Press after decompression
#      data_100k_manifest.json Check sha256）and items_human_ins.json
#   7. Generate 10 units of ten thousand products documents.jsonl、Retrieval Index indexes_100k（With original run_indexing.sh Same argument，One Thread）And well try it
# 10 See thousands of configurations analysis plan.md 6.13 Revision 1；Filename & webshop/wcfg.py of PRODUCT_FILE / ATTR_FILE / INDEX_NAME identical。
# pip / conda The download caches and temporary files are on the disk /root/autodl-tmp/cache（System Disk Small，Default ~/.cache/pip and /tmp all located on the system disk）；
# This folder can be deleted when all is finished。
# Path Available Environment Variables：WS_ENV_PREFIX（conda environment）、WS_PARENT（WebShop Where?）、WS_CACHE（cache）、
# WS_TORCH_INDEX（CPU Version torch Download Source，default https://download.pytorch.org/whl/cpu）、WS_PYSERINI（pyserini version，default 2.3.0）。
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
PREFIX=${WS_ENV_PREFIX:-/root/autodl-tmp/envs/webshop}
WS_PARENT=${WS_PARENT:-/root/autodl-tmp/webshop}
WS_DIR=$WS_PARENT/WebShop
PY=$PREFIX/bin/python
DATA_SRC=$ROOT/webshop_data
VENDOR=$ROOT/webshop/vendor/webshop_64fa2a5_min.tar.gz
LOG=$ROOT/logs/setup_webshop.log
CACHE=${WS_CACHE:-/root/autodl-tmp/cache}
TORCH_INDEX=${WS_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}
# Fixed Version 2.3.0：2.4.0 at import must start from raw.githubusercontent.com Download pre-indexed directories，fail if the server cannot connect；
# 2.3.0 The directorys in the bag、import Unconnected，Dependence with 2.4.0 exactly identical。If you dont write a version, pip may fall back to an old version requiring local compilation，may appear stalled。
PYSERINI_VER=${WS_PYSERINI:-2.3.0}
mkdir -p "$ROOT/logs"

step() { echo; echo "===== number $1 steps：$2"; echo "===== number $1 steps：$2 ($(date '+%F %T'))" >> "$LOG"; }
ok()   { echo "  ✓ $1"; }
fail() { echo "  ✗ failed：$1"; echo "    （Full output on $LOG）"; exit 1; }
run()  { "$@" >> "$LOG" 2>&1; }       # Detailed output of the command is written into the log，Just look on the screen ✓ / ✗

find_java_home() {
    for f in "$PREFIX/lib/jvm/lib/server/libjvm.so" "$PREFIX/lib/server/libjvm.so" /usr/lib/jvm/*/lib/server/libjvm.so; do
        if [ -f "$f" ]; then dirname "$(dirname "$(dirname "$f")")"; return 0; fi
    done
    return 1
}

command -v conda >/dev/null 2>&1 || fail "not found conda（first conda activate llm or set conda Add PATH）"

# Cache & Temporary File Disk（The orders in this script only take effect）
export PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp CONDA_PKGS_DIRS=$CACHE/conda_pkgs PYTHONUNBUFFERED=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$CONDA_PKGS_DIRS" || fail "Create Cache Folder $CACHE"
free_gb() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%.1f", $4/1048576}'; }
# Available Memory Not Available $1 GB Just stop（system termination may leave no error in the log，therefore check in advance）
need_mem() {
    local m; m=$(mem_gb)
    [ "$m" = "?" ] && return 0
    awk -v m="$m" -v n="$1" 'BEGIN { exit !(m + 0 < n) }' \
        && fail "Available Memory Only $m GB，This is an offer $1 GB（AutoDL The cardless mode is only about an appointment 2 GB Memory：Shut it down first、Well switch to card mode and start again）"
    return 0
}
mem_gb()  { python3 -c "import sys; sys.path.insert(0, '$ROOT'); from webshop import wcfg; m = wcfg.mem_available_gb(); print('?' if m is None else round(m, 1))" 2>/dev/null || echo "?"; }
echo "Disk Remaining：System Disk / $(free_gb /) GB，Data Disk $CACHE $(free_gb "$CACHE") GB（Cache and Temporary File Disk）"
echo "Disk Remaining：System Disk $(free_gb /) GB，Data Disk $(free_gb "$CACHE") GB ($(date '+%F %T'))" >> "$LOG"

step 1 "conda environment $PREFIX（Python 3.12）"
if [ -x "$PY" ]; then
    ok "Existing：$($PY --version 2>&1)"
else
    run conda create -y -p "$PREFIX" python=3.12 || fail "conda create"
    [ -x "$PY" ] || fail "not found after creation $PY"
    ok "$($PY --version 2>&1)"
fi

BM25=0
[ "${WS_SEARCH:-pyserini}" = "bm25" ] && BM25=1

step 2 "Java 21"
JH=$(find_java_home || true)
if [ "$BM25" = 1 ]; then
    ok "skip（WS_SEARCH=bm25：not used pyserini）"
elif [ -n "$JH" ] && "$JH/bin/java" -version 2>&1 | grep -q 'version "2[1-9]'; then
    ok "Existing：$JH"
else
    run conda install -y -p "$PREFIX" -c conda-forge openjdk=21 || fail "conda install openjdk=21（Yeah apt-get install -y openjdk-21-jdk-headless Rerun Script）"
    JH=$(find_java_home || true)
    [ -n "$JH" ] || fail "not found after installation libjvm.so"
    ok "$("$JH/bin/java" -version 2>&1 | head -1)（$JH）"
fi
[ -n "$JH" ] && export JAVA_HOME=$JH

step 3 "WebShop Text environment Python Bag"
if $PY -c "import bs4, flask, spacy, thefuzz, rank_bm25, cleantext, rich, tqdm, requests" 2>/dev/null; then
    ok "installed"
else
    run $PY -m pip install beautifulsoup4 flask requests "spacy>=3.7,<3.9" thefuzz rank_bm25 cleantext rich tqdm \
        || fail "pip install（Basic dependence）"
    $PY -c "import bs4, flask, spacy, thefuzz, rank_bm25, cleantext, rich, tqdm, requests" 2>>"$LOG" || fail "after installation import failed"
    ok "Set it up"
fi

step 4 "pyserini（Original Retrieval）"
torch_ver() { $PY -c "import torch; print(torch.__version__)" 2>/dev/null; }
ps_ver()    { $PY -c 'import importlib.metadata as m; print(m.version("pyserini"))' 2>/dev/null; }
# Disconnect the net import：You cant rely on the network during the experiment，import fail if network access is attempted
OFFLINE_IMPORT='import socket
def _deny(*a, **k): raise OSError("setup_webshop: pyserini at import attempts network access during")
socket.socket.connect = _deny
from pyserini.search.lucene import LuceneSearcher'
if [ "$BM25" = 1 ]; then
    ok "skip（WS_SEARCH=bm25：use webshop/shims Pure Lee Python BM25）"
elif [ "$(ps_ver)" = "$PYSERINI_VER" ] && $PY -c "$OFFLINE_IMPORT" 2>/dev/null; then
    ok "installed：pyserini $PYSERINI_VER，torch $(torch_ver)（Disconnect the net import passed）"
else
    # pyserini of Lucene Retrieving in import also includes import torch，But not at all GPU。
    # install first CPU Version torch，after pip install pyserini will no longer install CUDA Version（together with nvidia-* How many? GB）。
    awk -v g="$(free_gb "$CACHE")" 'BEGIN { exit !(g + 0 < 3) }' && fail "Disk is not available 3 GB（$CACHE），free disk space first"
    TV=$(torch_ver || true)
    case "$TV" in
        *+cpu)
            ok "CPU Version torch $TV（Existing）" ;;
        *)
            # may have been installed before the previous interruption（Half）CUDA Version：torch、triton、nvidia-* Take it off first
            OLD=$($PY -m pip list --format=freeze 2>/dev/null | grep -i -E '^(torch|triton|nvidia-[a-z0-9_.-]+)==' | cut -d= -f1 | tr '\n' ' ')
            if [ -n "${OLD// /}" ]; then
                run $PY -m pip uninstall -y $OLD || fail "Unload the last one CUDA Version torch（$OLD）"
                ok "Unloaded the last one：$OLD"
            fi
            # Disruption in the middle of decompression，Theyll leave an unregistered half-cooked folder（pip cannot uninstall）；At this point, the bags are no longer in the register，Delete it directly
            SITE=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null)
            for d in torch functorch torchgen triton nvidia; do
                if [ -n "$SITE" ] && [ -d "$SITE/$d" ]; then
                    rm -rf "${SITE:?}/$d" || fail "Delete $SITE/$d"
                    ok "removed the incomplete artifact left by interruption $SITE/$d"
                fi
            done
            echo "  … Load CPU Version torch（A few minutes；Open another terminal tail -f $LOG You can see the progress）"
            if ! run $PY -m pip install "torch>=2.9" --index-url "$TORCH_INDEX"; then
                [ -f /etc/network_turbo ] || fail "Load CPU Version torch（cannot connect $TORCH_INDEX；Yeah export WS_TORCH_INDEX=<Other sources> Rerun，Or change it export WS_SEARCH=bm25）"
                echo "  … Direct $TORCH_INDEX failed，enable the download accelerator and retry"
                ( source /etc/network_turbo; $PY -m pip install "torch>=2.9" --index-url "$TORCH_INDEX" ) >> "$LOG" 2>&1 \
                    || fail "Load CPU Version torch（cannot connect $TORCH_INDEX；Yeah export WS_TORCH_INDEX=<Other sources> Rerun，Or change it export WS_SEARCH=bm25）"
            fi
            TV=$(torch_ver) || fail "CPU Version torch installed but import failed"
            case "$TV" in *+cpu) ;; *) fail "installed version torch is not CPU Version（$TV），check WS_TORCH_INDEX";; esac
            ok "CPU Version torch $TV" ;;
    esac
    echo "  … Load pyserini $PYSERINI_VER and its dependencies（5–15 minutes；Open another terminal tail -f $LOG You can see the progress）"
    run $PY -m pip install --prefer-binary "pyserini==$PYSERINI_VER" \
        || fail "pip install pyserini==$PYSERINI_VER（Whats the last line of the log thats going to write about?；You can use pure Python of BM25：export WS_SEARCH=bm25）"
    case "$(torch_ver)" in *+cpu) ;; *) fail "Load pyserini when torch was replaced by $(torch_ver)（Look at the log）";; esac
    [ "$(ps_ver)" = "$PYSERINI_VER" ] || fail "after installation pyserini yes $(ps_ver)，is not $PYSERINI_VER（Look at the log）"
    $PY -c "$OFFLINE_IMPORT" 2>>"$LOG" || fail "pyserini installed but offline import failed（Look at the last rows of the log：Mostly JAVA_HOME，or import attempts network access during）"
    ok "pyserini $(ps_ver)，torch $(torch_ver)（Disconnect the net import passed）"
fi
echo "  Disk Remaining：System Disk $(free_gb /) GB，Data Disk $(free_gb "$CACHE") GB"

step 5 "spaCy model en_core_web_sm（WebShop used for reward computation）"
if $PY -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    ok "installed"
else
    WHL=$(ls "$DATA_SRC"/en_core_web_sm-*.whl 2>/dev/null | head -1)
    if [ -n "$WHL" ]; then
        run $PY -m pip install "$WHL" || fail "pip install $WHL"
    else
        ( [ -f /etc/network_turbo ] && source /etc/network_turbo; $PY -m spacy download en_core_web_sm ) >> "$LOG" 2>&1 \
            || fail "spacy download（GitHub if downloading fails：download on your local machine en_core_web_sm of whl，Put it in webshop_data/ Rerun，See README）"
    fi
    $PY -c "import spacy; spacy.load('en_core_web_sm')" 2>>"$LOG" || fail "Failed to load after loading"
    ok "Set it up"
fi

step 6 "WebShop Code and Data -> $WS_DIR"
PROD=items_shuffle_100k.json
ATTR=items_ins_v2_100k.json
HUMAN=items_human_ins.json
MAN=data_100k_manifest.json
RES=resources_100k
IDX=indexes_100k
mkdir -p "$WS_PARENT"
if [ ! -f "$WS_DIR/web_agent_site/envs/web_agent_text_env.py" ]; then
    mkdir -p "$WS_DIR" && tar -xzf "$VENDOR" -C "$WS_DIR" || fail "Undo it $VENDOR"
fi
ok "Code：$WS_DIR"
mkdir -p "$WS_DIR/data"
[ -s "$DATA_SRC/$MAN" ] || fail "Not available $DATA_SRC/$MAN（run on your local machine webshop/make_100k.py Generate，Again sync.py push Upload，See README 6.13）"
cp "$DATA_SRC/$MAN" "$WS_DIR/data/$MAN" || fail "Copy $MAN"
man_sha() { $PY -c "import json, sys; m = json.load(open(sys.argv[1])); print(m[sys.argv[2]][sys.argv[3]]['sha256'])" "$DATA_SRC/$MAN" "$1" "$2"; }
file_sha() { sha256sum "$1" | cut -d' ' -f1; }
for f in "$PROD" "$ATTR"; do
    W=$(man_sha outputs "$f") || fail "List $MAN does not contain $f"
    if [ -s "$WS_DIR/data/$f" ] && [ "$(file_sha "$WS_DIR/data/$f")" = "$W" ]; then
        ok "data $f（Existing，$(du -h "$WS_DIR/data/$f" | cut -f1)，sha256 Consistency with the List）"
        continue
    fi
    [ -s "$DATA_SRC/$f.gz" ] || fail "Not available $DATA_SRC/$f.gz（make_100k.py Generate，use sync.py push Upload，See README 6.13）"
    gzip -dc "$DATA_SRC/$f.gz" > "$WS_DIR/data/$f.tmp" || fail "Depressure $f.gz"
    G=$(file_sha "$WS_DIR/data/$f.tmp")
    if [ "$G" != "$W" ]; then
        rm -f "$WS_DIR/data/$f.tmp"
        fail "$f decompressed sha256 Inconsistent with the List（${G:0:16}… ≠ ${W:0:16}…）：Re-upload $f.gz"
    fi
    mv "$WS_DIR/data/$f.tmp" "$WS_DIR/data/$f" || fail "mv $f"
    ok "data $f（$(du -h "$WS_DIR/data/$f" | cut -f1)，sha256 Consistency with the List）"
done
W=$(man_sha inputs "$HUMAN") || fail "List $MAN does not contain $HUMAN"
if [ ! -s "$WS_DIR/data/$HUMAN" ] || [ "$(file_sha "$WS_DIR/data/$HUMAN")" != "$W" ]; then
    [ -s "$DATA_SRC/$HUMAN" ] || fail "Not available $DATA_SRC/$HUMAN（use sync.py push Upload）"
    cp "$DATA_SRC/$HUMAN" "$WS_DIR/data/$HUMAN" || fail "Copy $HUMAN"
fi
[ "$(file_sha "$WS_DIR/data/$HUMAN")" = "$W" ] || fail "$HUMAN of sha256 Inconsistent with the List"
ok "data $HUMAN（sha256 Consistency with the List）"

step 7 "Search for Documents and Indexes（10 10,000 commodities）"
cd "$WS_DIR/search_engine" || fail "cd search_engine"
if [ ! -s "$RES.done" ]; then
    awk -v g="$(free_gb "$WS_DIR")" 'BEGIN { exit !(g + 0 < 4) }' && fail "Disk is not available 4 GB，free disk space first"
    need_mem 2
    echo "  … Generate 10 units of ten thousand products documents.jsonl（approximately 1 minutes，An offer 1.5 GB Memory；Available Memory $(mem_gb) GB）"
    # Always --shims：Convert only with original load_products，Do Not Call Search，You dont have to load it torch and Java（See ws_convert.py Start）
    run $PY "$ROOT/webshop/ws_convert.py" --webshop-dir "$WS_DIR" --product-file "$PROD" --attr-file "$ATTR" \
        --resources "$RES" --compat "$ROOT/webshop/compat" --shims "$ROOT/webshop/shims" \
        || fail "ws_convert.py（Original convert_product_file_format.py）；If the top line says, Killed，indicates insufficient RAM：Look free -g"
fi
ok "documents.jsonl：$(wc -l < "$RES/documents.jsonl") Commodities（$(du -h "$RES/documents.jsonl" | cut -f1)；$(grep -o '"peak_rss_gb": [0-9.]*' "$RES.done" | cut -d' ' -f2) GB Memory）"
if [ "$BM25" = 1 ]; then
    ok "WS_SEARCH=bm25：do not create Lucene Index"
else
    if [ ! -s "$IDX.done" ]; then
        rm -rf "$IDX"                              # Half-section index left from last break
        need_mem 3
        echo "  … Construction Lucene Index $IDX（One Thread，With original run_indexing.sh Same argument；approximately 5–15 minutes）"
        run $PY -m pyserini.index.lucene --collection JsonCollection --input "$RES" --index "$IDX" \
            --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw \
            || fail "pyserini Index"
        date '+%F %T' > "$IDX.done"
    fi
    N=$($PY -c "from pyserini.search.lucene import LuceneSearcher; s=LuceneSearcher('$IDX'); print(len(s.search('shoes for men', k=10)))" 2>>"$LOG") \
        || fail "The search failed"
    ok "Index $IDX Available（$(du -sh "$IDX" | cut -f1)；Check it out 'shoes for men' Back $N records）"
fi
echo "  Disk Remaining：System Disk $(free_gb /) GB，Data Disk $(free_gb "$WS_DIR") GB"
cd "$ROOT" || exit 1

echo
echo "All completed。Next：conda activate llm Run after  python webshop/ws_smoke.py check --models <A model on this server>"
echo "（7B Servers --models q7,l8；14B Servers --models q14）"
echo "（cache $CACHE Now take it $(du -sh "$CACHE" 2>/dev/null | cut -f1)，Make sure you can run when you smoke rm -rf $CACHE Free Data Disk）"
echo "（WebShop interpreter is $PY；if paths were changed，first export WS_PYTHON=$PY WS_DIR=$WS_DIR）"
