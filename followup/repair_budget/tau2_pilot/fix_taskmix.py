"""11-07：π0 续跑里"任务 id 跨领域重名"的位置，把已经写下的续跑行挪出去，好让修过的 pool_tau2.py 重跑它们。

背景：pool_tau2.py rollout（11-07 之前的版本）按任务 id 建任务表；design_run.jsonl 里同时有 retail 与 airline 的位置，
后读的 airline 覆盖了 retail 的同名任务（31 个 id 重名）。这些 retail 位置的续跑拿到的是 airline 的任务：
用户模拟器的指令、初始状态、判分用的金标都是错的，整条续跑作废（不论 reward 是多少）。

做法（不看 reward，只按位置挑）：
1. 按旧代码的读法重建"任务 id → 哪个结果文件"：设计文件按 pos_id 排序后，结果文件按第一次出现的顺序读，后读的覆盖先读的。
2. 位置的结果文件 ≠ 这个任务 id 最后归属的结果文件 → 受影响。
3. 默认只打印（dry-run）。加 --apply 才改：原文件先整份备份（.bak_taskmix_<时间>），受影响的行另存到 *.taskmix_removed.*，
   其余行原样写回。msgs.jsonl.gz 同样处理。

用法（服务器上，tau2_pilot 目录；π0 的续跑进程必须已经停下）：
    python fix_taskmix.py --design pool/design_run.jsonl --rollouts pool/ro_pi0.jsonl
    python fix_taskmix.py --design pool/design_run.jsonl --rollouts pool/ro_pi0.jsonl --apply
"""
import argparse
import collections
import gzip
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAU = HERE / "tau2-bench"


def affected_positions(design):
    order = []
    for d in sorted(design, key=lambda d: d["pos_id"]):          # 旧代码：design.sort(key=pos_id)，再按位置顺序读结果文件
        if d["src"] not in order:
            order.append(d["src"])
    owner = {}
    for src in order:                                            # 后读的覆盖先读的
        R = json.loads((TAU / src).read_text(encoding="utf-8"))
        for t in R["tasks"]:
            owner[t["id"]] = src
    aff = {d["pos_id"] for d in design if owner.get(d["task_id"]) != d["src"]}
    return order, owner, aff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    running = subprocess.run(["pgrep", "-f", "pool_tau2.py rollout"], capture_output=True, text=True).stdout.split()
    if running and a.apply:
        raise SystemExit(f"[停下] 还有续跑进程在跑（pid {running}）；先停掉再 --apply")

    design = [json.loads(l) for l in Path(a.design).read_text(encoding="utf-8").splitlines() if l.strip()]
    order, owner, aff = affected_positions(design)
    by = collections.Counter((d["domain"], d.get("fold")) for d in design if d["pos_id"] in aff)
    nb = sum(sum(1 for c in d["cands"] if c.get("restore_ok")) for d in design if d["pos_id"] in aff)
    print(f"结果文件读入顺序：{order}")
    print(f"受影响的位置 {len(aff)} 个（{dict(by)}），分支 {nb} 个，任务 {len({d['task_id'] for d in design if d['pos_id'] in aff})} 个")

    ro = Path(a.rollouts)
    lines = [l for l in ro.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep, drop = [], []
    for l in lines:
        (drop if json.loads(l)["pos_id"] in aff else keep).append(l)
    print(f"{ro.name}：共 {len(lines)} 行；挪出 {len(drop)} 行，保留 {len(keep)} 行")

    mp = Path(str(ro) + ".msgs.jsonl.gz")
    sys.path.insert(0, str(HERE))
    from gzrobust import read_records                # 被杀时可能留下写了一半的块：只读完整的记录
    recs, ok = read_records(mp) if mp.exists() else ([], True)
    mkeep = [r for r in recs if r["pos_id"] not in aff]
    mdrop = [r for r in recs if r["pos_id"] in aff]
    print(f"{mp.name}：共 {len(recs)} 条（文件完整：{ok}）；挪出 {len(mdrop)} 条，保留 {len(mkeep)} 条")

    if not a.apply:
        print("dry-run：没有改任何文件。核对上面的数字后加 --apply。")
        return 0
    tag = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(ro, f"{ro}.bak_taskmix_{tag}")
    if mp.exists():
        shutil.copy2(mp, f"{mp}.bak_taskmix_{tag}")
    Path(f"{ro}.taskmix_removed.jsonl").write_text("".join(l + "\n" for l in drop), encoding="utf-8")
    with gzip.open(f"{ro}.taskmix_removed.msgs.jsonl.gz", "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in mdrop)
    ro.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    with gzip.open(mp, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in mkeep)
    print(f"已改。备份：{ro.name}.bak_taskmix_{tag}、{mp.name}.bak_taskmix_{tag}；挪出的行：{ro.name}.taskmix_removed.*")
    print("下一步：用修过的 pool_tau2.py 重跑 run_main.sh pi0（会只跑缺的续跑）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
