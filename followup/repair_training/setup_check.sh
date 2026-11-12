#!/usr/bin/env bash
# 环境检查（只读，不安装、不改任何东西）。在项目根下、激活 conda 环境 llm 之后运行：bash setup_check.sh
# 检查：Python 与关键库的版本、peft 能否与 transformers 一起用、ALFWorld 数据与游戏数、7B 模型、GPU、磁盘、API 密钥是否已设置。
cd "$(dirname "$0")" || exit 1
export PYTHONHASHSEED=0
export ALFWORLD_DATA="${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
PY="$(command -v python || command -v python3)"
echo "Python：$PY（$($PY --version 2>&1)）"
"$PY" - <<'EOF'
import importlib.metadata as md
for p in ("torch", "transformers", "peft", "vllm", "accelerate", "alfworld", "textworld", "fast-downward-textworld",
          "numpy", "requests"):
    try:
        print(f"  {p:26s} {md.version(p)}")
    except Exception:
        print(f"  {p:26s} （没装）")
try:
    import transformers, peft
    from peft import LoraConfig, get_peft_model
    print("  peft + transformers 导入：正常")
except Exception as e:
    print("  peft + transformers 导入失败：", repr(e)[:300])
try:
    import torch
    print("  CUDA 可用：", torch.cuda.is_available(), "；显卡：", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
except Exception as e:
    print("  torch 导入失败：", repr(e)[:200])
EOF
echo "--- ALFWorld 数据（$ALFWORLD_DATA）---"
"$PY" - <<'EOF'
import os, sys
sys.path.insert(0, ".")
from alfw import games, acfg
for s in ("train", "valid_seen", "valid_unseen"):
    try:
        g, st = games.collect(s)
        print(f"  {s:13s} {len(g)} 局（期望 {acfg.EXPECTED_COUNTS[s]}）")
    except Exception as e:
        print(f"  {s:13s} 读取失败：{e}")
try:
    import fast_downward, textworld
    print("  fast_downward / textworld 导入：正常")
except Exception as e:
    print("  fast_downward / textworld 导入失败：", repr(e)[:200])
d = os.environ.get("RT_Q7", str(acfg.MODEL_DIR))
print(f"  7B 模型 {d}：", "存在" if os.path.isfile(os.path.join(d, "config.json")) else "找不到 config.json")
EOF
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || echo "  （没有 GPU：无卡模式。selftest / trainselftest 可以跑，其余要挂卡）"
echo "--- 磁盘 ---"
df -h / /root/autodl-tmp 2>/dev/null
echo "--- API 密钥 ---"
if [ -n "${QWEN_API_KEY:-}" ]; then echo "  QWEN_API_KEY 已设置（不显示内容）"; else echo "  QWEN_API_KEY 没有设置"; fi
