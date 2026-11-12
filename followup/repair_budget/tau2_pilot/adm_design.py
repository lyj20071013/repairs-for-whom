"""由设计文件与可执行性记录，写出只含"可执行的修复候选 + 自由选择"的设计文件（2026-11-01 写；登记补充五第 2 节第 1 条）。
用它跑 `pool_tau2.py rollout --k 6 --out <原来的续跑文件>`：已完成的 k = 0–2 会跳过，只补跑 k = 3–5。
用法（tau2_pilot 下，只用 CPU）：
  python adm_design.py --design probe/design_train_ret.jsonl --adm probe/samples_s4b.jsonl.admissible.jsonl --out probe/design_train_ret_adm.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--adm", required=True, help="export --admissible 写出的 *.admissible.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ok = {}
    for l in Path(a.adm).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            ok[(r["pos_id"], r["cand_id"])] = bool(r["ok"])
    out, n_pos, n_c, n_free, n_unknown = [], 0, 0, 0, 0
    for l in Path(a.design).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        keep = []
        for c in d["cands"]:
            if c["cand_id"] == "free":
                keep.append(c)
                continue
            k = (d["pos_id"], c["cand_id"])
            if k not in ok:
                n_unknown += 1                      # 不是修复候选（只有原动作等），或没被判过
                continue
            if ok[k]:
                keep.append(c)
        rep = [c for c in keep if c["cand_id"] != "free"]
        if not rep:
            continue                                # 这个位置没有可执行的修复候选：不加续跑
        n_pos += 1
        n_c += len(rep)
        n_free += sum(1 for c in keep if c["cand_id"] == "free")
        out.append(dict(d, cands=keep))
    Path(a.out).write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in out), encoding="utf-8")
    print(f"{a.design} → {a.out}：位置 {n_pos} 个，可执行的修复候选 {n_c} 个，自由选择 {n_free} 个"
          f"（未判过 / 非修复候选 {n_unknown} 个，不跑）")
    return 0 if n_pos else 1


if __name__ == "__main__":
    raise SystemExit(main())
