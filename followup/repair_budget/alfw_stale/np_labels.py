"""第 6 篇实验 B：ALFWorld 上训练"不用规划器标签"的后继——标签选择与训练文件导出（CPU，不用 GPU）。
登记：D:\\Paper\\repair_budget\\登记_ALFWorld无规划器后继_20261110.md

只读复用第 5 篇 P0 的数据与代码（RT_CODE，默认 /root/paper/repair_training；RT_OUT，默认 /root/autodl-tmp/rt_runs/out_rt），
不改它们的任何文件；输出全部写到 --out（默认 /root/autodl-tmp/rt_runs/out_np）。

S-np 组 = P0 的 S 组，只把候选集合换成"无规划器流水线"的候选：
  保留有模型提议（q7 / qflash）的候选与随机候选；去掉只来自规划器的候选（与事后分析 A 的主设定相同）。
  其余全部照 S：同一批 1,116 个训练决策点、同一批 3 次标签续跑、同一个 ĝ（最好候选 − 原动作）、同样取前 N = 558 个点、
  同样的并列打破（derive_seed("RT3", …)）、同一批锚样本、模板 Thought "I will <动作>."、同样的槽位排法。

自检（任何一条不过就退出，什么都不写）：
  1. 用本脚本的选择函数、全部候选，重算 S 组 → 必须与 trainsets.json 的 S 组逐项相同；
  2. 用本脚本的导出函数重建 S 组的训练文件 → 必须与 train_S.jsonl 逐行相同（消息、目标、槽位）。
两条都过，说明 S-np 与 S 只差候选集合。

用法（14b2，conda 环境 llm）：
  cd /root/paper/repair_budget/alfw_stale
  python np_labels.py                      # 写 out_np/train_S-np.jsonl、train_manifest.json、trainsets_np.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RT_CODE = Path(os.environ.get("RT_CODE", "/root/paper/repair_training"))
sys.path.insert(0, str(RT_CODE))
os.environ.setdefault("PYTHONHASHSEED", "0")

import rt_common as C  # noqa: E402
import rt_data as D  # noqa: E402
from rt_common import derive_seed, file_sha256, jdump, jl_read, jload  # noqa: E402

GROUP = "S-np"


def keep_np(br):
    """无规划器流水线的候选：有模型提议，或是随机候选（规划器动作若也被模型提议过，保留）。"""
    return bool(br.get("proposers")) or bool(br.get("random"))


def select(d, lab, keep):
    """与 rt_data.stage_select 里 S 组的规则逐行对应，只多一个候选过滤 keep。"""
    pts = d["points"]
    N = int(round(len(pts) * C.CFG["FRAC_N"]))
    sh = lambda b: float(np.mean(lab[b]))  # noqa: E731
    info = {}
    for p in pts:
        alts = [b for b in p["alts"] if keep(d["branches"][b])]
        if not alts:
            continue
        so = sh(p["orig"])
        sc = {b: sh(b) - so for b in alts}
        best = max(alts, key=lambda b: (sc[b], derive_seed("RT3", p["pid"], d["branches"][b]["action"])))
        info[p["pid"]] = {"ghat": sc[best], "target": best, "s_orig": so}
    elig = [p for p in pts if p["pid"] in info]
    order = sorted(elig, key=lambda p: (-info[p["pid"]]["ghat"], -derive_seed("RT3", p["pid"])))
    return [{"pid": p["pid"], "bid": info[p["pid"]]["target"]} for p in order[:N]], info, N, len(elig)


def export_rows(d, eps, sel, anchors):
    """与 rt_data.stage_export 的 repair_row + slots 相同（Thought 用模板，5.14）。"""
    byp = {p["pid"]: p for p in d["points"]}
    rep = []
    for x in sel:
        p = byp[x["pid"]]
        msgs = D.point_messages(eps[p["traj_id"]], p["t"], p["admissible"])
        a = d["branches"][x["bid"]]["action"]
        rep.append({"kind": "repair", "key": x["pid"], "bid": x["bid"], "messages": msgs,
                    "completion": D.completion(D.TEMPLATE.format(a=a), a)})
    rep = sorted(rep, key=lambda r: C.hkey("SLOT", r["key"]))
    out = []
    for i, (r, a) in enumerate(zip(rep, anchors)):
        out += [dict(r, slot=2 * i), dict(a, slot=2 * i + 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt-out", default=os.environ.get("RT_OUT", "/root/autodl-tmp/rt_runs/out_rt"))
    ap.add_argument("--out", default="/root/autodl-tmp/rt_runs/out_np")
    ap.add_argument("--profile", default="p0", choices=["p0", "mock"], help="mock 只用于本地自检")
    a = ap.parse_args()
    src, dst = Path(a.rt_out), Path(a.out)
    C.apply_profile(a.profile)
    C.use_out(src)                                   # 只读：P() 指向 P0 的输出
    if C.CFG["THOUGHT_MODE"] != "template":
        sys.exit("[np] P0 的 THOUGHT_MODE 不是 template，与登记不符")
    d = jload(src / "design.json")
    ts = jload(src / "trainsets.json")
    K = C.CFG["K_LABEL"]
    lab = D.outcomes(d, src / "labels_q7.jsonl", "LAB", K)
    miss = [b for b in d["branches"] if b not in lab]
    if miss:
        sys.exit(f"[np] {len(miss)} 个分支没有完整的 3 次标签续跑")
    eps = D.load_eps("train")
    s_rows = jl_read(src / "train_S.jsonl")
    anchors = [{k: v for k, v in r.items() if k != "slot"} for r in sorted(s_rows, key=lambda r: r["slot"]) if r["kind"] == "anchor"]
    if len(anchors) != len(ts["anchors"]):
        sys.exit(f"[np] train_S.jsonl 的锚样本 {len(anchors)} 个，trainsets.json 里 {len(ts['anchors'])} 个")

    # 自检 1：全部候选 → 必须复现 S 组
    S_re, _, N, _ = select(d, lab, lambda br: True)
    if S_re != ts["groups"]["S"]:
        nd = sum(x != y for x, y in zip(S_re, ts["groups"]["S"]))
        sys.exit(f"[np] 自检 1 不过：重算的 S 组与 trainsets.json 不同（{nd} 处）；不写任何文件")
    print(f"[np] 自检 1 通过：重算的 S 组（{len(S_re)} 个点）与 trainsets.json 逐项相同")
    # 自检 2：重建 S 的训练文件 → 必须与 train_S.jsonl 逐行相同
    rebuilt = export_rows(d, eps, S_re, anchors)
    if len(rebuilt) != len(s_rows) or any(json.dumps(x, sort_keys=True, ensure_ascii=False) != json.dumps(y, sort_keys=True, ensure_ascii=False)
                                          for x, y in zip(rebuilt, s_rows)):
        bad = next((i for i, (x, y) in enumerate(zip(rebuilt, s_rows)) if x != y), None)
        sys.exit(f"[np] 自检 2 不过：重建的 S 训练文件与 train_S.jsonl 不同（第一处在第 {bad} 行）；不写任何文件")
    print(f"[np] 自检 2 通过：重建的 S 训练文件与 train_S.jsonl 逐行相同（{len(s_rows)} 行）")

    # S-np
    sel, info, N2, n_elig = select(d, lab, keep_np)
    if len(sel) != N:
        sys.exit(f"[np] S-np 只选出 {len(sel)} 个点，少于 N = {N}（有候选的点 {n_elig} 个）")
    rows = export_rows(d, eps, sel, anchors)
    br = d["branches"]
    sh = lambda b: float(np.mean(lab[b]))  # noqa: E731
    s_pts = {x["pid"] for x in ts["groups"]["S"]}
    s_bid = {x["pid"]: x["bid"] for x in ts["groups"]["S"]}
    stats = {
        "N": N, "n_points_with_np_cands": n_elig,
        "n_np_cands_per_point": float(np.mean([sum(keep_np(br[b]) for b in p["alts"]) for p in d["points"]])),
        "q_label_in_sample": {"S-np": float(np.mean([sh(x["bid"]) - info[x["pid"]]["s_orig"] for x in sel])),
                              "S": ts["stats"]["q_label"]["S_in_sample"]},
        "ghat_cut": {"S-np": info[sel[-1]["pid"]]["ghat"], "S": ts["stats"]["ghat_cut"]},
        "target_is_planner_action": float(np.mean([br[x["bid"]]["planner"] for x in sel])),
        "target_from_proposer": {n: float(np.mean([n in br[x["bid"]]["proposers"] for x in sel])) for n in C.CFG["PROPOSERS"]},
        "target_is_random": float(np.mean([bool(br[x["bid"]]["random"]) and not br[x["bid"]]["proposers"] for x in sel])),
        "overlap_points_with_S": len({x["pid"] for x in sel} & s_pts) / N,
        "same_target_as_S_on_shared_points": float(np.mean([s_bid[x["pid"]] == x["bid"] for x in sel if x["pid"] in s_bid])),
        "S_target_is_planner": ts["stats"]["planner_frac"]["S"],
    }
    dst.mkdir(parents=True, exist_ok=True)
    C.use_out(dst)
    fpath = dst / f"train_{GROUP}.jsonl"
    if fpath.exists():
        old = jl_read(fpath)
        if [json.dumps(r, sort_keys=True) for r in old] != [json.dumps(r, sort_keys=True) for r in rows]:
            sys.exit(f"[np] {fpath} 已存在且内容不同；换输出目录")
    C.jl_write(fpath, rows)
    jdump(dst / "train_manifest.json", {"trainsets_id": ts["trainsets_id"], "source": str(src),
                                         "files": {GROUP: {"n": len(rows), "n_slots": 2 * len(anchors), "sha256": file_sha256(fpath)}}})
    jdump(dst / "trainsets_np.json", {"group": GROUP, "points": sel, "stats": stats, "design_id": d["design_id"],
                                      "src_trainsets_id": ts["trainsets_id"]})
    print(f"[np] 写入 {fpath}（{len(rows)} 行，sha256 {file_sha256(fpath)[:16]}）")
    print("[np] 统计：" + json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
