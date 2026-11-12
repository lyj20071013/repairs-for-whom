"""6.14 修订 5（2026-11-11）：2B 子集里拿不到 reward 的续跑的处理。只用 CPU；不改任何已有文件。

规则（修订 5，写在看 P1τ / P2τ 之前）：补跑之后仍没有 reward 的续跑（对话超过 32,768 token 的上下文上限，或连续超时）记为失败（reward 0）。
    python fill_missing.py --design D --ro ro_pi0.jsonl ro_2b.jsonl ro_2b_greedy.jsonl --subset p4sub_2b.txt --out-fill ro_2b_fill.jsonl --out-subset p4sub_2b_nomiss.txt
写出：
  ro_2b_fill.jsonl      每个缺失的续跑一行（reward 0.0、score orig_task、filled = 原因）；作为最后一个 --ro 传给 pool_analyze.py（同一键取最后一行）
  p4sub_2b_nomiss.txt   去掉含缺失续跑的失败对话后的子集（敏感性分析）
"""
import argparse, collections, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--design", required=True); ap.add_argument("--ro", nargs="+", required=True)
ap.add_argument("--subset", required=True); ap.add_argument("--out-fill", required=True); ap.add_argument("--out-subset", required=True)
a = ap.parse_args()
rows = [json.loads(l) for p in a.ro for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
des = {(d["pos_id"], c["cand_id"]) for d in map(json.loads, Path(a.design).read_text(encoding="utf-8").splitlines()) for c in d["cands"] if c["cand_id"] != "free"}
rows = [r for r in rows if (r["pos_id"], r["cand_id"]) in des and r["executor"] in ("2B", "4B")]      # 只管子集里的分支；4B 的读数来自第 6 篇的 ro_pi0.jsonl
last = {}
for r in rows:                                       # 同一键：有 reward 的优先，否则取最后一行
    k = (r["pos_id"], r["cand_id"], r["executor"], r["temp"], r["k"])
    if r.get("reward") is not None or k not in last or last[k].get("reward") is None:
        last[k] = r
miss = [r for r in last.values() if r.get("reward") is None]
why = collections.Counter("上下文超长" if "ContextWindow" in str(r.get("termination")) or "input tokens" in str(r.get("termination"))
                          else ("超时" if "Timeout" in str(r.get("termination")) else "其他") for r in miss)
assert why["其他"] == 0, f"有规则之外的缺失原因：{[r['termination'][:200] for r in miss]}"
with open(a.out_fill, "w", encoding="utf-8") as f:
    for r in sorted(miss, key=lambda r: (r["pos_id"], r["cand_id"], r["temp"], r["k"])):
        f.write(json.dumps(dict(r, reward=0.0, score="orig_task", filled=str(r.get("termination"))[:80]), ensure_ascii=False) + "\n")
bad = {r["sim_id"] for r in miss}
keep = [l for l in Path(a.subset).read_text(encoding="utf-8").splitlines() if l.startswith("#") or (l.strip() and l.split()[0] not in bad)]
Path(a.out_subset).write_text("\n".join(keep) + "\n", encoding="utf-8")
print(f"缺失 {len(miss)} 次（{dict(why)}；按执行者 {dict(collections.Counter(r['executor'] for r in miss))}；温度 0 的 {sum(r['temp'] == 0 for r in miss)} 次），涉及 {len({(r['pos_id'], r['cand_id']) for r in miss})} 个分支、{len(bad)} 条失败对话")
print(f"敏感性子集：{sum(1 for l in keep if l.strip() and not l.startswith('#'))} 条失败对话")
