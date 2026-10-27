"""
Model Download，Go from home to home。

Policy：
  1. try first hf-mirror.com（HF Mirror）
  2. Fail and Fall ModelScope
Command Line：
  python download_model.py              # Default Source
  python download_model.py --source ms  # Force ModelScope
  python download_model.py --source hf  # Force hf-mirror
"""
import argparse
import os
import sys
from pathlib import Path

import config


def try_hf_mirror(repo_id: str, local_dir: Path) -> bool:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[hf-mirror] huggingface_hub Not installed", file=sys.stderr)
        return False

    try:
        print(f"[hf-mirror] Start Downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            max_workers=8,
            resume_download=True,
            ignore_patterns=["*.gguf", "*.pth", "original/*"],
        )
        return True
    except Exception as e:
        print(f"[hf-mirror] failed：{type(e).__name__}: {e}", file=sys.stderr)
        return False


def try_modelscope(repo_id: str, local_dir: Path) -> bool:
    try:
        from modelscope import snapshot_download as ms_download
    except ImportError:
        print("[modelscope] modelscope Not installed：pip install modelscope", file=sys.stderr)
        return False
    try:
        print(f"[modelscope] Start Downloading {repo_id} -> {local_dir}")
        ms_download(repo_id, local_dir=str(local_dir))
        return True
    except Exception as e:
        print(f"[modelscope] failed：{type(e).__name__}: {e}", file=sys.stderr)
        return False


def verify(local_dir: Path) -> bool:
    need = ["config.json"]
    if not local_dir.exists():
        print(f"[verify] Directory does not exist：{local_dir}", file=sys.stderr)
        return False
    missing = [f for f in need if not (local_dir / f).exists()]
    if missing:
        print(f"[verify] missing file：{missing}", file=sys.stderr)
        return False
    # Weight Files：.safetensors or .bin At least one
    has_w = list(local_dir.glob("*.safetensors")) + list(local_dir.glob("*.bin"))
    if not has_w:
        print("[verify] No weight files found", file=sys.stderr)
        return False
    total_gb = sum(f.stat().st_size for f in has_w) / 1e9
    print(f"[verify] OK，Weight {len(has_w)} items，Total {total_gb:.1f} GB")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "hf", "ms"], default="auto")
    ap.add_argument("--repo", default=config.MODEL_ID)
    ap.add_argument("--dir", default=str(config.MODEL_DIR))
    args = ap.parse_args()

    local_dir = Path(args.dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if (local_dir / "config.json").exists() and verify(local_dir):
        print("Model already exists，skip download。")
        return

    ok = False
    if args.source in ("auto", "hf"):
        ok = try_hf_mirror(args.repo, local_dir)
    if not ok and args.source in ("auto", "ms"):
        ok = try_modelscope(args.repo, local_dir)

    if not ok:
        print("\n Download Failed。Manual Scheme：", file=sys.stderr)
        print("  1) Set \\$env:HF_ENDPOINT='https://hf-mirror.com' Retry", file=sys.stderr)
        print("  2) modelscope download --model Qwen/Qwen2.5-7B-Instruct "
              f"--local_dir {local_dir}", file=sys.stderr)
        sys.exit(1)

    if not verify(local_dir):
        sys.exit(1)
    print(f"\n complete。config.MODEL_DIR Pointed {local_dir}")


if __name__ == "__main__":
    main()
