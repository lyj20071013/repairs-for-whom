#!/usr/bin/env bash
# tau 服务器：建环境、装 vLLM 与 τ²-bench（2026-10-29 写）。无卡模式即可运行（只下载、安装，不用 GPU）。
# 用法（在 /root/paper/repair_budget/tau2_pilot 下）：bash install_tau.sh 2>&1 | tee logs/install_tau.log
# 每一步失败都会打印"[停下] 第 N 步"并退出；可以修好后重跑，已完成的步会跳过。
set -u
HERE=/root/paper/repair_budget/tau2_pilot
ENV=/root/autodl-tmp/envs/tau                 # conda 环境放数据盘（系统盘小）
TARBALL=$HERE/tau2-bench-122e243-trim2.tar.gz   # 10-29 改：trim 版缺英文语言包，tau2 run 报 no published language pack for 'en'
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HERE/logs"
stop() { echo "[停下] 第 $1 步：$2"; exit 1; }

echo "== 第 1 步：Python 3.12+ 环境"
# 10-29 修改：清华 anaconda 镜像返回 403（conda create 失败）。改为按顺序尝试，任何一种成功就停：
#   a) 机器上已有 Python ≥ 3.12 → 用它建 venv（不需要任何 conda 频道）；
#   b) conda create，忽略 .condarc 里的频道（--override-channels），依次用阿里云镜像、conda-forge 官方频道。
# 失败的半成品目录会先删掉再试下一种。
if [ -x "$ENV/bin/python" ] && [ -x "$ENV/bin/pip" ]; then
  echo "环境已存在：$ENV"
else
  rm -rf "$ENV"; mkdir -p "$(dirname "$ENV")"
  made=0
  for py in python3.13 python3.12 python3 python /root/miniconda3/bin/python /root/miniconda3/envs/llm/bin/python; do   # 10-30：14b2 上 llm 环境是 3.12.9；venv 与它隔离，不改动 llm 环境
    p=$(command -v "$py" 2>/dev/null) || continue
    if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
      echo "a) 用 $p（$("$p" --version 2>&1)）建 venv"
      if "$p" -m venv "$ENV"; then made=1; break; fi
      echo "   venv 失败，换下一种"; rm -rf "$ENV"
    else
      echo "a) $p 是 $("$p" --version 2>&1)，低于 3.12，跳过"
    fi
  done
  if [ $made = 0 ]; then
    CONDA_BASE=$(conda info --base 2>/dev/null) || stop 1 "没有 Python ≥ 3.12，也找不到 conda"
    source "$CONDA_BASE/etc/profile.d/conda.sh" || stop 1 "无法加载 conda.sh"
    for ch in https://mirrors.aliyun.com/anaconda/pkgs/main https://mirrors.aliyun.com/anaconda/cloud/conda-forge conda-forge; do
      echo "b) conda create，频道 $ch"
      if conda create -y -p "$ENV" --override-channels -c "$ch" python=3.12; then made=1; break; fi
      echo "   这个频道失败，换下一个"; rm -rf "$ENV"
    done
  fi
  [ $made = 1 ] || stop 1 "三种方式都没建成环境（把这段日志发给 Claude）"
fi
"$ENV/bin/python" --version || stop 1 "环境里的 python 不能运行"
"$ENV/bin/python" -m pip --version || stop 1 "环境里没有 pip"
"$ENV/bin/python" -m pip install -U pip || stop 1 "升级 pip 失败（pip 镜像不通？）"
echo "pip 镜像：$("$ENV/bin/python" -m pip config list 2>/dev/null | tr '\n' ' ')"

echo "== 第 2 步：vLLM（与 14b2 同版本 0.32.0）"
if "$ENV/bin/python" -c "import vllm" 2>/dev/null; then
  echo "已安装：$("$ENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
else
  "$ENV/bin/pip" install "vllm==0.32.0" || stop 2 "pip 安装 vllm 失败（把上面的报错发给 Claude）"
fi

echo "== 第 3 步：τ²-bench（本地推上来的源码包，不连 GitHub）"
[ -f "$TARBALL" ] || stop 3 "找不到 $TARBALL（先用 sync.py push）"
got=$(md5sum < "$TARBALL" | cut -d' ' -f1); [ "$got" = 97b3d4f644b101954f73a397ec7c00cf ] || stop 3 "源码包 md5 不对：$got"
if [ ! -d "$HERE/tau2-bench/src/tau2" ]; then
  tar -xzf "$TARBALL" -C "$HERE" || stop 3 "解压失败"
fi
"$ENV/bin/pip" install -e "$HERE/tau2-bench" || stop 3 "pip 安装 tau2 失败"

echo "== 第 4 步：modelscope"
"$ENV/bin/pip" install modelscope || stop 4 "pip 安装 modelscope 失败"

echo "== 第 5 步：导入检查"
"$ENV/bin/python" - <<'EOF' || stop 5 "导入失败"
import importlib
for m in ("torch", "vllm", "transformers", "tau2", "litellm", "modelscope"):
    mod = importlib.import_module(m)
    print(m, getattr(mod, "__version__", "?"))
import torch
print("cuda 可用：", torch.cuda.is_available(), "（无卡模式下是 False，正常）")
from tau2.registry import registry
print("tau2 领域：", registry.get_domains())
EOF
echo "== 完成"
