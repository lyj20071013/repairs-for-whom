"""tau 服务器：用 ModelScope 把 Qwen3.5-4B、Qwen3.5-9B 下到数据盘（2026-10-29 写）。无卡模式即可。
用法（tau 环境）：/root/autodl-tmp/envs/tau/bin/python download_models.py [模型 ID …] 2>&1 | tee logs/download_models.log
  不带参数时下载 Qwen3.5-4B 与 Qwen3.5-9B；10-29 起可以点名，例如 Qwen/Qwen3.8-27B-FP8。
已下载的会跳过（ModelScope 按文件校验）。"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MODELSCOPE_CACHE", "/root/autodl-tmp/modelscope_cache")
from modelscope import snapshot_download

ROOT = Path("/root/autodl-tmp/models")
MODELS = sys.argv[1:] or ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]
free = __import__("shutil").disk_usage("/root/autodl-tmp").free / 1e9
print(f"数据盘剩余 {free:.1f} GB", flush=True)

ok = True
for mid in MODELS:
    dst = ROOT / mid.split("/")[1]
    print(f"== {mid} -> {dst}", flush=True)
    try:
        snapshot_download(mid, local_dir=str(dst))
    except Exception as e:
        print(f"[失败] {mid}: {e!r}（若是找不到模型，把报错发给 Claude，可能是 ModelScope 上的名字不同）")
        ok = False
        continue
    cfg = dst / "config.json"
    st = list(dst.glob("*.safetensors"))
    size = sum(p.stat().st_size for p in st) / 1e9
    arch = json.loads(cfg.read_text()).get("architectures") if cfg.exists() else None
    print(f"  config.json {'有' if cfg.exists() else '没有'}；safetensors {len(st)} 个，共 {size:.1f} GB；architectures = {arch}")
    if not cfg.exists() or not st:
        ok = False
print("== 完成" if ok else "== 有失败，见上")
sys.exit(0 if ok else 1)
