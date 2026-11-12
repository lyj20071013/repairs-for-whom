"""按任务把一个领域对半分成"训练任务"与"评价任务"（2026-10-31 写；只用 CPU）。
排序键 = sha256("SPLIT|<标签>|<领域>|<任务 id>") 的前 8 位，前一半（向下取整）为训练，其余为评价。确定、与运行顺序无关。
用法：python split_tasks.py <标签> <领域> <任务清单> <输出目录>
  例：python split_tasks.py PROBE retail ids_retail_det.txt probe     -> probe/train_retail.txt、probe/eval_retail.txt
"""
import hashlib
import sys
from pathlib import Path


def key(tag, dom, tid):
    return int(hashlib.sha256(f"SPLIT|{tag}|{dom}|{tid}".encode()).hexdigest()[:8], 16)


def main():
    tag, dom, ids_file, out = sys.argv[1:5]
    ids = [l.strip() for l in Path(ids_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(ids) == len(set(ids)), "任务清单里有重复"
    order = sorted(ids, key=lambda t: key(tag, dom, t))
    n = len(order) // 2
    Path(out).mkdir(parents=True, exist_ok=True)
    for name, part in [("train", order[:n]), ("eval", order[n:])]:
        Path(out, f"{name}_{dom}.txt").write_text("".join(t + "\n" for t in sorted(part, key=lambda x: (len(x), x))),
                                                  encoding="utf-8")
    print(f"{dom}：{len(ids)} 个任务 -> 训练 {n}、评价 {len(order) - n}（标签 {tag}）；写在 {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
