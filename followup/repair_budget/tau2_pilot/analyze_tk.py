"""共享失败状态接手结果的汇总（2026-10-31 写；只用 CPU，只读 jsonl）。
用法（在 tau2_pilot 下）：python analyze_tk.py [--prefix tk5] [--models 2b,4b,9b] [--out logs/tk5_summary.md]
读 logs/<prefix>_{ret,air}_<模型>.jsonl：同一 (位置, k) 有多行时取 reward 不为 None 的最后一行。
输出：各模型平均成功率（先按位置平均，再对位置平均）、死位置数（三个模型全为 0）、按位置配对的差值与 90% 自助法区间
（对位置重抽 10,000 次，种子 0），以及 reward_hist（τ² 带前缀的旧判分）与 reward 不一致的续跑数。
"""
import argparse
import json
import random
from pathlib import Path


def load(path):
    R = {}
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        k = (r["pos_id"], r["k"])
        if r.get("reward") is not None or k not in R:
            R[k] = r
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="tk5")
    ap.add_argument("--models", default="2b,4b,9b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ms = a.models.split(",")
    L = [f"# 共享失败状态接手：{a.prefix}（analyze_tk.py）", ""]
    for ab, dom in [("ret", "retail"), ("air", "airline")]:
        files = {m: Path(f"logs/{a.prefix}_{ab}_{m}.jsonl") for m in ms}
        have = [m for m in ms if files[m].exists()]
        if not have:
            continue
        D = {json.loads(l)["pos_id"]: json.loads(l) for l in
             Path(str(files[have[0]]) + ".design.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        rate, n_err, n_diff, n_run = {}, {}, {}, {}
        for m in have:
            R = load(files[m])
            per = {}
            n_err[m] = sum(r["reward"] is None for r in R.values())
            n_run[m] = sum(r["reward"] is not None for r in R.values())
            n_diff[m] = sum(1 for r in R.values() if r["reward"] is not None and "reward_hist" in r
                            and r["reward_hist"] is not None and r["reward_hist"] != r["reward"])
            for (p, k), r in R.items():
                if r["reward"] is not None:
                    per.setdefault(p, []).append(r["reward"])
            rate[m] = {p: sum(v) / len(v) for p, v in per.items()}
        pos = sorted(set.intersection(*[set(rate[m]) for m in have]) & set(D))
        L += [f"## {dom}：位置 {len(pos)} 个", "",
              "| 模型 | 平均成功率 | 有效续跑 | 出错 | 旧判分不一致 |", "|---|---:|---:|---:|---:|"]
        for m in have:
            L.append(f"| {m.upper()} | {sum(rate[m][p] for p in pos) / max(1, len(pos)):.2f} | {n_run[m]} | {n_err[m]} | {n_diff[m]} |")
        dead = [p for p in pos if max(rate[m][p] for m in have) == 0]
        L += ["", f"死位置（{'/'.join(x.upper() for x in have)} 全为 0）：{len(dead)} / {len(pos)}", ""]
        rng = random.Random(0)
        if len(have) >= 2 and pos:
            L += ["| 配对差值 | 点估计 | 90% 区间 |", "|---|---:|---|"]
            for i in range(len(have)):
                for j in range(i + 1, len(have)):
                    x, y = have[j], have[i]
                    d = [rate[x][p] - rate[y][p] for p in pos]
                    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
                    L.append(f"| {x.upper()} − {y.upper()} | {sum(d) / len(d):+.2f} | [{bs[500]:+.2f}, {bs[9499]:+.2f}] |")
            L.append("")
        L += ["| 位置 | t / 总长 | 第一次错误写 | " + " | ".join(m.upper() for m in have) + " |",
              "|---|---|---|" + "---:|" * len(have)]
        for p in pos:
            d = D[p]
            L.append(f"| {p} | {d['t']} / {d['n_msgs_src']} | {d.get('first_bad')} | "
                     + " | ".join(f"{rate[m][p]:.2f}" for m in have) + " |")
        L.append("")
    txt = "\n".join(L) + "\n"
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
