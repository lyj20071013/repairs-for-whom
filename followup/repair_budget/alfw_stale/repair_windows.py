"""A 层补充统计（2026-11-06，事后、描述性）：同一条失败轨迹的早 / 中 / 晚三个位置里，有几个"值得修"（G ≥ 0.20）？
π0 用交叉拟合的 G0（各半 5 次）；另报 π0 全部 10 次的 G0_all（有赢家诅咒，偏高）与 π1（S_s11、S_s12）的 G1。
噪声参照：在 π1 = π0 的假设下，同一套算法在两半之间不一致的比例见 stale_analyze 的 N0；这里只做描述。"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import stale_alfw as SA
import stale_analyze as AN
from rt_common import jl_read

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
SA.setup("real", str(OUT))
D, ev, pos = SA.load_design()
SA.DESIGN[0] = D
res = defaultdict(lambda: defaultdict(dict))
for stage in ("full", "free_b"):
    for r in jl_read(OUT / f"{stage}.jsonl"):
        key = r["branch"] if r["branch"] != "free" else r["pos_id"] + "|free"
        res[r["model"]][key][r["i"]] = bool(r["won"])
Bfree = {p["pos_id"]: v for p in pos if (v := AN.lst(res["B"].get(p["pos_id"] + "|free"))) is not None}
TH, mean = AN.TH, AN.mean
out, L = {}, ["# A 层：每条失败有几个值得修的位置（2026-11-06，事后、描述性）", ""]
for m in ("S_s11", "S_s12"):
    pp = AN.per_position(D, ev, pos, res, m, Bfree)
    by = defaultdict(list)
    for k, x in pp.items():
        by[x["traj"]].append((x["p"]["t"], x))
    for field, who in (("G0", "π0（交叉拟合）"), ("G0_all", "π0（全部 10 次，偏高）"), ("G1", f"π1 = {m}（交叉拟合）")):
        if field.startswith("G0") and m != "S_s11":
            continue
        cnt, first, n3 = Counter(), Counter(), 0
        for t, xs in by.items():
            xs.sort(key=lambda z: z[0])
            if len(xs) != 3 or any(x.get(field) is None for _, x in xs):
                continue
            n3 += 1
            w = [x[field] >= TH for _, x in xs]
            cnt[sum(w)] += 1
            if any(w):
                first[["早", "中", "晚"][w.index(True)]] += 1
        dist = {k: cnt[k] for k in range(4)}
        any_ = sum(v for k, v in cnt.items() if k > 0)
        multi = sum(v for k, v in cnt.items() if k >= 2)
        L += [f"## {who}", "", f"- 三个位置都有数据的失败：{n3} 条",
              f"- 值得修的位置数分布（0 / 1 / 2 / 3）：{dist[0]} / {dist[1]} / {dist[2]} / {dist[3]}",
              f"- 至少一个可修的失败里，有 ≥ 2 个可修位置的比例：{multi / any_:.2f}（{multi}/{any_}）" if any_ else "- 没有可修位置",
              f"- 最早的可修位置：早 {first['早']}、中 {first['中']}、晚 {first['晚']}", ""]
        out[f"{m}:{field}"] = {"n": n3, "dist": dist, "multi_share": multi / any_ if any_ else None, "first": dict(first)}
Path("repair_windows.md").write_text("\n".join(L), encoding="utf-8")
Path("repair_windows.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n".join(L))
