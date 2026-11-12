"""无 GPU 自检（模拟环境 + 模拟模型 + 模拟训练）。上服务器跑任何 GPU 阶段之前先跑：

    python selftest.py                 全部检查（约几分钟）
    python selftest.py --write-manifest   改了代码后更新 MANIFEST.json

[0] 代码与 MANIFEST.json 一致（防止只同步了部分文件）
[1] 纯函数：Thought 解析与提示用语过滤、槽位顺序、学习率、bootstrap、判定逻辑
[2] 模拟全流程 p0（rt.py --stage p0 --profile mock）：各阶段输出的结构与登记的规则（N = D/2、U 与 R 同点集、
    S 取 ĝ 最高的点、锚样本、槽位交替、各模型评测条数、共同随机数）
[3] 断点续跑：label 中途中断后重跑，与不中断的结果逐条相同；签名不同的旧结果拒绝混用
[4] 分析的已知答案：合成数据（已知各组真值）上点估计接近真值、区间覆盖率合理、判定正确
服务器上另有训练自检（真实分词器 + 极小随机模型）：python rt_train.py --selftest
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.json"


def check(c, msg):
    if not c:
        raise AssertionError(msg)
    print("  OK ", msg, flush=True)


def manifest_files():
    fs = [p.name for p in ROOT.glob("*.py")] + [f"alfw/{p.name}" for p in (ROOT / "alfw").glob("*.py")]
    fs += ["run_bg.sh", "setup_check.sh", "alfw_setup.sh", "tools/sync.py"]
    return sorted(f for f in fs if (ROOT / f).exists())


def write_manifest():
    import rt_common as C
    data = {n: C.file_sha256(ROOT / n) for n in manifest_files()}
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 MANIFEST.json（{len(data)} 个文件）")


def t0_manifest():
    import rt_common as C
    print("[0] 代码与 MANIFEST.json 一致")
    if not MANIFEST.exists():
        raise AssertionError("缺少 MANIFEST.json：请把 rt_code 压缩包完整解开")
    want = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stale = [n for n, h in want.items() if not (ROOT / n).exists() or C.file_sha256(ROOT / n) != h]
    extra = [n for n in manifest_files() if n not in want]
    if stale or extra:
        raise AssertionError(f"与清单不一致：{stale}；清单外的代码文件：{extra}。重新解压完整的代码包；"
                             "若是有意修改，运行 python selftest.py --write-manifest")
    check(True, f"{len(want)} 个文件与清单一致")


def t1_pure():
    print("[1] 纯函数")
    import rt_data as D
    import rt_train as T
    import rt_analyze as A
    th, why = D.parse_thought("Thought: The mug should be on the desk, so I go there.\nAction: go to desk 1")
    check(th == "The mug should be on the desk, so I go there." and why is None, "Thought 解析：取 Thought 行，去掉 Action")
    for bad in ("Thought: As suggested, I go to the desk.", "Thought: The given action takes me to the desk.",
                "Thought: I was told to open the drawer.", "Thought: The next action is to open it."):
        check(D.parse_thought(bad)[1] == "leak", f"提示用语被拦下：{bad[9:40]}…")
    check(D.parse_thought("Action: look")[1] == "unparsable", "没有 Thought 行：无法解析")
    check(D.parse_verdict("The thought claims X.\nVerdict: UNSUPPORTED") == "UNSUPPORTED" and
          D.parse_verdict("Verdict: supported") == "SUPPORTED" and D.parse_verdict("no verdict") is None,
          "事实核查的结论解析（取最后一个 Verdict 行）")
    ts_ = {"groups": {"S": [{"bid": f"s{i}"} for i in range(10)], "U": [{"bid": f"u{i}"} for i in range(10)]},
           "dpo": [{"rejected": "s0"}]}
    ver = {**{f"s{i}": "UNSUPPORTED" if i < 2 else "SUPPORTED" for i in range(10)},
           **{f"u{i}": "UNSUPPORTED" if i < 5 else "SUPPORTED" for i in range(10)}}
    sm, fl = D.fact_summary(ver, ts_, list(ver) + ["anchor|x"], 0.10)
    check(abs(sm["S"]["unsupported"] - 0.2) < 1e-9 and abs(sm["U"]["unsupported"] - 0.5) < 1e-9 and fl == {"U": 0.5, "DPO-S_rejected": 1.0},
          "事实核查汇总：S 0.2、U 0.5、DPO 被拒 1.0 时标记 U 与 DPO 被拒（高出 > 0.10），锚样本不参与标记")
    check(D.parse_thought("Thought: Given that the apple is not here, I check the fridge.")[1] is None,
          "普通的 'Given that' 不算提示用语")
    s1 = T.micro_schedule(list(range(20)), 11, 2, 16)
    check(s1 == T.micro_schedule(list(range(20)), 11, 2, 16) and s1 != T.micro_schedule(list(range(20)), 12, 2, 16),
          "槽位顺序只由种子决定")
    check(sorted(x for g in s1 for x in g) == sorted(list(range(20)) * 2) and len(s1) == 3, "两轮、每个槽位各两次、按 16 分组")
    lrs = [T.lr_at(i, 100, 1e-4, 0.03) for i in range(100)]
    check(abs(lrs[2] - 1e-4) < 1e-12 and lrs[0] < lrs[1] < lrs[2] and lrs[-1] < 1e-6, "学习率：3% 预热、余弦衰减到接近 0")
    rng = np.random.default_rng(0)
    a = rng.random(400) < 0.6
    b = rng.random(400) < 0.5
    est, lo, hi, p = A.boot_diff(a.astype(float), b.astype(float), np.arange(400), 2000, 0)
    check(abs(est - (a.mean() - b.mean())) < 1e-12 and lo < est < hi, "bootstrap：点估计 = 配对差均值，区间包含点估计")
    se = math.sqrt((a.var() + b.var()) / 400)
    check(abs((hi - lo) / 2 / 1.645 - se) / se < 0.25, f"bootstrap 区间宽度与正态近似一致（{(hi - lo) / 2 / 1.645:.4f} vs {se:.4f}）")
    mk = lambda se_, sr: {"diff": {"S-B": {"e2e": se_, "tk": (0, 0, 0, 1)}, "S-R": {"tk": sr, "e2e": (0, 0, 0, 1)}}}
    check(A.decide(mk((0.04, 0.01, 0.07, 0.01), (0.12, 0.05, 0.2, 0.01)))["pass_e2e"], "判定：两项都满足 -> 通过")
    d2 = A.decide(mk((0.02, -0.01, 0.05, 0.1), (0.12, 0.05, 0.2, 0.01)))
    check(not d2["pass_e2e"] and d2["pass_tk"] and "只达到接管" in d2["verdict"], "判定：只有接管满足")
    d3 = A.decide(mk((0.05, 0.01, 0.09, 0.01), (0.09, 0.02, 0.15, 0.01)))
    check(not d3["pass_tk"] and "未达到" in d3["verdict"], "判定：接管 < +0.10 -> 未达到推进条件（不写成“没有效果”）")
    d4 = A.decide(mk((0.05, 0.01, 0.09, 0.01), (0.12, -0.01, 0.25, 0.06)))
    check(not d4["pass_tk"], "判定：接管下界 ≤ 0 -> 不满足")
    d5 = A.decide({"diff": {"S-B": {"e2e": (0.1, 0, 0, 0), "tk": (0.1, 0, 0, 0)}}})
    check(d5["pass_tk"] is None and "不适用" in d5["verdict"], "判定：缺 R 组（冒烟）时不适用、不报错")


def run(cmd, **kw):
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True, **kw)
    if r.returncode:
        print(r.stdout[-4000:], r.stderr[-4000:])
    return r


def t2_pipeline(tmp):
    print("[2] 模拟全流程（rt.py --stage p0 --profile mock）")
    out = tmp / "p0"
    r = run(["rt.py", "--stage", "p0", "--profile", "mock", "--out-dir", str(out)], timeout=3600)
    check(r.returncode == 0, "p0 流水线走完（数据 -> 训练 -> 格式检查 -> 评测 -> 分析）")
    import rt_common as C
    C.apply_profile("mock")
    C.use_out(out)
    G = C.jload(out / "games.json")
    check(len(G["train"]) == C.CFG["N_TRAIN"] and len(G["fmt"]) == C.CFG["N_SMOKE"], "训练局、格式检查局数量与设置相同")
    Eg = [x["g"] for x in G["E"]]
    check(len(set(Eg)) == len(Eg) and not set(Eg) & set(G["train"]) and not set(Eg) & set(G["fmt"])
          and not set(G["train"]) & set(G["fmt"]), "评测局、训练局、格式检查局两两不重叠")
    check({x["src"] for x in G["E"]} == {"ds69", "valid_seen", "valid_unseen", "train_extra"}, "评测局由四部分组成")
    d = C.jload(out / "design.json")
    ts = C.jload(out / "trainsets.json")
    D_, N = ts["stats"]["D"], ts["stats"]["N"]
    check(D_ == len(d["points"]) and N == int(round(D_ * 0.5)), f"N = round(D / 2)（D = {D_}，N = {N}）")
    for g in ("S", "U", "R", "S-plan"):
        check(len(ts["groups"][g]) == N, f"{g} 组 {N} 个样本")
    check([x["pid"] for x in ts["groups"]["S-plan"]] == [x["pid"] for x in ts["groups"]["S"]] and
          all(d["branches"][x["bid"]]["planner"] for x in ts["groups"]["S-plan"]), "S-plan = S 的点 + 规划器动作")
    check([x["pid"] for x in ts["groups"]["U"]] == [x["pid"] for x in ts["groups"]["R"]], "U 与 R 用同一个点集")
    br = d["branches"]
    check(all(br[x["bid"]]["planner"] for x in ts["groups"]["U"]), "U 的目标都是规划器动作")
    check(all(x["bid"] in next(p for p in d["points"] if p["pid"] == x["pid"])["alts"] for x in ts["groups"]["R"]),
          "R 的目标都是该点的候选")
    # S 取 ĝ 最高的点：用标签重算
    import rt_data as Dm
    lab = Dm.outcomes(d, out / "labels_q7.jsonl", "LAB", C.CFG["K_LABEL"])
    gh = {p["pid"]: max(np.mean(lab[b]) - np.mean(lab[p["orig"]]) for b in p["alts"]) for p in d["points"]}
    sp = {x["pid"] for x in ts["groups"]["S"]}
    cut = min(gh[q] for q in sp)
    check(all(gh[q] <= cut + 1e-12 for q in gh if q not in sp), f"S 取的是 ĝ 最高的 N 个点（入选下限 {cut:+.2f}）")
    check(all(np.mean(lab[x["bid"]]) - np.mean(lab[next(p["orig"] for p in d["points"] if p["pid"] == x["pid"])])
              == gh[x["pid"]] for x in ts["groups"]["S"]), "S 的目标是该点估计净收益最大的候选")
    for g in ("S", "U", "R", "S-plan", "DPO-S"):
        rows = C.jl_read(out / f"train_{g}.jsonl")
        check(len(rows) == 2 * N and [r["slot"] for r in rows] == list(range(2 * N)) and
              all(r["kind"] == ("anchor" if r["slot"] % 2 else ("pair" if g == "DPO-S" else "repair")) for r in rows),
              f"train_{g}.jsonl：2N 个槽位，修复 / 锚样本交替")
    anc = [r["completion"] for r in C.jl_read(out / "train_S.jsonl") if r["kind"] == "anchor"]
    check(anc == [r["completion"] for r in C.jl_read(out / "train_U.jsonl") if r["kind"] == "anchor"], "各组锚样本相同")
    A_rows = C.jl_read(out / "train_A.jsonl")
    check(len(A_rows) == N and all(r["kind"] == "anchor" and r["slot"] % 2 == 1 for r in A_rows)
          and [r["completion"] for r in A_rows] == anc, "train_A.jsonl：只有锚样本，占同样的奇数槽位")
    import rt_train as T
    man = C.jload(out / "train_manifest.json")["files"]
    sS = [x for g in T.micro_schedule([r["slot"] for r in C.jl_read(out / "train_S.jsonl")], 11, 2, 16, man["S"]["n_slots"]) for x in g]
    sA = [x for g in T.micro_schedule([r["slot"] for r in A_rows], 11, 2, 16, man["A"]["n_slots"]) for x in g]
    check(sA == [x for x in sS if x % 2 == 1], "A 的样本顺序 = S 的顺序去掉修复槽位（锚样本按同样的顺序出现）")
    fc = C.jload(out / "factcheck.json")
    check(fc["by_group"]["thought_mode"] == "template" and fc["by_group"]["anchor"]["n"] == N
          and fc["by_group"]["S"]["n_parsed"] == 0 and fc["n_api_errors"] == 0,
          "factcheck.json：模板方式下只核查锚样本（修复样本的模板 Thought 没有事实断言）")
    for g in ("S", "U", "R", "S-plan"):
        rows_g = [r for r in C.jl_read(out / f"train_{g}.jsonl") if r["kind"] == "repair"]
        check(all(r["completion"] == f"Thought: I will {d['branches'][r['bid']]['action']}.\nAction: {d['branches'][r['bid']]['action']}"
                  for r in rows_g), f"train_{g}.jsonl：修复样本是模板 Thought（5.14）")
    pr = [r for r in C.jl_read(out / "train_DPO-S.jsonl") if r["kind"] == "pair"]
    check(all(r["rejected"].startswith("Thought: I will ") for r in pr), "DPO 对的被拒一侧也是模板 Thought")
    # 改写方式（留作敏感性）仍可运行
    import rt_data as Dm2
    C.CFG["THOUGHT_MODE"] = "rewrite"
    C.BACKENDS["n"] = 0
    try:
        Dm2.stage_thoughts()
        rw = Dm2.load_thoughts(ts)
        check(len(rw) == len(Dm2.needed_pairs(d, ts)) and not all(r["template"] for r in rw.values()),
              "改写方式（敏感性）的 Thought 生成仍可运行，与模板方式的结果按签名分开")
    finally:
        C.CFG["THOUGHT_MODE"] = "template"
    rr = C.jl_read(out / "train_S.jsonl")[0]
    check(rr["messages"][0]["role"] == "system" and "Actions possible now" in rr["messages"][1]["content"]
          and rr["completion"].startswith("Thought: ") and "\nAction: " in rr["completion"],
          "训练样本 = 部署协议的提示 + Thought / Action")
    check(all("The next action is" not in m["content"] for m in rr["messages"]), "改写 Thought 用的提示用语没有进入训练文本")
    ev = C.jl_read(out / "eval_rollouts.jsonl")
    models = C.all_models()
    ntk = sum(len(t["positions"]) for t in C.jload(out / "positions_tk.json")["trajectories"]) * C.CFG["TK_M"]
    for m in models:
        check(sum(r["model"] == m and "|tk|" in r["rid"] for r in ev) == ntk and
              sum(r["model"] == m and "|e2e|" in r["rid"] for r in ev) == len(G["E"]), f"{m}：接管 {ntk} 条、端到端 {len(G['E'])} 条")
    seeds = {}
    for r in ev:
        seeds.setdefault((r.get("gamefile"), r.get("pid"), r.get("k")), set()).add(r["seed"])
    check(all(len(v) == 1 for v in seeds.values()), "共同随机数：同一局 / 同一接管点上各模型的请求种子相同")
    a = C.jload(out / "analysis.json")
    check("decision" in a and "main" in a and (out / "report.md").exists(), "analysis.json 与 report.md 已生成")
    lv = a["level"]["e2e"]
    check(lv["S"] > lv["B"] > lv["R"], f"模拟数据上的方向合理（端到端 S {lv['S']:.2f} > B {lv['B']:.2f} > R {lv['R']:.2f}；"
          "模拟 adapter 的能力 S 0.85、基础 0.55、R 0.35。模拟环境里从失败轨迹中途接手几乎都救不回，接管读数只检查结构）")
    tkp = {r["pid"] for r in ev if "|tk|" in r["rid"]}
    check(all(r["first_cmd"] for r in ev if "|tk|" in r["rid"]) and len(tkp) * C.CFG["TK_M"] == ntk,
          "接管：每条都从该点的历史开始由模型自己选第一步（记录了第一步动作）")
    r2 = run(["rt.py", "--stage", "select", "--profile", "mock", "--out-dir", str(out)])
    check(r2.returncode == 0, "重跑 select：算出的训练集与已有的相同（可复现）")
    return out


def t3_resume(tmp, ref_out):
    print("[3] 断点续跑与签名")
    import rt_common as C
    import rt_data as D
    out = tmp / "resume"
    out.mkdir()
    for f in ("games.json", "episodes_train.jsonl", "episodes_train_meta.json", "positions_train.json",
              "proposals_q7.jsonl", "proposals_qflash.jsonl", "design.json"):
        shutil.copy(ref_out / f, out / f)
    C.apply_profile("mock")
    C.use_out(out)
    C.BACKENDS["n"] = 0
    try:
        D.stage_label(stop_after=7)
        raise AssertionError("没有按设定中断")
    except KeyboardInterrupt:
        pass
    n1 = len(C.jl_read(out / "labels_q7.jsonl"))
    with open(out / "labels_q7.jsonl", "a") as f:
        f.write('{"rid": "half-writ')                  # 模拟写到一半被杀
    C.BACKENDS["n"] = 0
    D.stage_label()
    a = {r["rid"]: (r["won"], r["n_steps"], r["seed"]) for r in C.jl_read(out / "labels_q7.jsonl")}
    b = {r["rid"]: (r["won"], r["n_steps"], r["seed"]) for r in C.jl_read(ref_out / "labels_q7.jsonl")}
    check(n1 == 7 and a == b, f"中断（已写 {n1} 条 + 半行）后重跑，与不中断的结果逐条相同（{len(a)} 条）")
    rows = C.jl_read(out / "labels_q7.jsonl")
    rows[0]["sig"] = "changed"
    C.jl_write(out / "labels_q7.jsonl", rows)
    r = run(["rt.py", "--stage", "label", "--profile", "mock", "--out-dir", str(out)])
    check(r.returncode != 0 and "签名不同" in (r.stdout + r.stderr), "设置签名不同的旧结果拒绝混用")
    C.BACKENDS["n"] = 0
    C.make_backend()
    try:
        C.make_backend()
        ok = False
    except RuntimeError:
        ok = True
    check(ok, "同一进程第二次建引擎会报错（每个 GPU 阶段只有一个 vLLM 引擎）")


def synth_rows(rng, truth_e2e, truth_tk, seeds, n_games=800, n_traj=200, m=3, seed_sd=0.01):
    """合成评测结果：每个模型 = 组真值 + 种子偏移；每局 / 每点再加局（点）难度的共同随机效应。"""
    rows = []
    gdiff = rng.normal(0, 1.0, n_games)
    pdiff = rng.normal(0, 1.2, n_traj * 3)
    for g, ss in seeds.items():
        for s in ss:
            m_name = "B" if g == "B" else f"{g}_s{s}"
            off_e = 0 if g == "B" else rng.normal(0, seed_sd)
            off_t = 0 if g == "B" else rng.normal(0, seed_sd)
            lo_e = math.log(truth_e2e[g] / (1 - truth_e2e[g]))
            lo_t = math.log(truth_tk[g] / (1 - truth_tk[g]))
            for i in range(n_games):
                pr = 1 / (1 + math.exp(-(lo_e + gdiff[i]))) + off_e
                rows.append({"rid": f"{m_name}|e2e|g{i}", "model": m_name, "gamefile": f"g{i}",
                             "won": bool(rng.random() < pr)})
            for j in range(n_traj * 3):
                pr = 1 / (1 + math.exp(-(lo_t + pdiff[j]))) + off_t
                for k in range(m):
                    rows.append({"rid": f"{m_name}|tk|p{j}#{k}", "model": m_name, "pid": f"p{j}", "traj_id": f"t{j // 3}",
                                 "k": k, "won": bool(rng.random() < pr)})
    return rows


def true_mean(lo, eff):
    """logit 正态随机效应下的边际均值（数值积分）。"""
    z = np.linspace(-6, 6, 2001)
    w = np.exp(-z ** 2 / 2)
    w /= w.sum()
    return float(np.sum(w / (1 + np.exp(-(lo + eff * z)))))


def t4_known_answer():
    print("[4] 分析的已知答案（合成数据）")
    import rt_analyze as A
    import rt_common as C
    C.CFG["TK_M"] = 3
    tE = {"B": 0.60, "S": 0.71, "U": 0.67, "R": 0.57, "DPO-S": 0.69}
    tT = {"B": 0.12, "S": 0.30, "U": 0.25, "R": 0.08, "DPO-S": 0.28}
    seeds = {"B": [0], "S": [11, 12, 13], "U": [11, 12], "R": [11, 12], "DPO-S": [11]}
    mE = {g: true_mean(math.log(p / (1 - p)), 1.0) for g, p in tE.items()}
    mT = {g: true_mean(math.log(p / (1 - p)), 1.2) for g, p in tT.items()}
    cover, errs = {"S-B_e2e": 0, "S-R_tk": 0}, []
    R = 12
    for rep in range(R):
        rng = np.random.default_rng(100 + rep)
        rows = synth_rows(rng, tE, tT, seeds, seed_sd=0.0)
        res = A.analyze_core(rows, n_boot=600, seed=rep)
        e = res["diff"]["S-B"]["e2e"]
        t = res["diff"]["S-R"]["tk"]
        te, tt = mE["S"] - mE["B"], mT["S"] - mT["R"]
        cover["S-B_e2e"] += e[1] <= te <= e[2]
        cover["S-R_tk"] += t[1] <= tt <= t[2]
        errs.append((e[0] - te, t[0] - tt))
        if rep == 0:
            dec = A.decide(res)
    er = np.array(errs)
    check(np.abs(er.mean(0)).max() < 0.012, f"点估计无偏（平均误差 端到端 {er[:, 0].mean():+.4f}、接管 {er[:, 1].mean():+.4f}）")
    check(all(c >= R * 0.6 for c in cover.values()), f"90% 区间覆盖真值：{cover}（共 {R} 次；种子当固定时应约 90%）")
    check(dec["pass_e2e"] and dec["pass_tk"], f"真值 S − B = {mE['S'] - mE['B']:.3f}、S − R = {mT['S'] - mT['R']:.3f} 时判定通过")
    rng = np.random.default_rng(7)
    rows = synth_rows(rng, dict(tE, S=0.60), tT, seeds, seed_sd=0.0)
    d2 = A.decide(A.analyze_core(rows, n_boot=600))
    check(not d2["pass_e2e"] and d2["pass_tk"], "真值 S − B 端到端 = 0 时：只有接管一项通过")
    rng = np.random.default_rng(8)
    rows = synth_rows(rng, tE, tT, seeds, seed_sd=0.03)
    res = A.analyze_core(rows, n_boot=300)
    sd = res["seeds"]["tk"]["S_sd"]
    check(sd is not None and 0.005 < sd < 0.08, f"种子 SD 能被估出量级（真值 0.03 左右，估计 {sd:.3f}）")
    pw, _ = A.power(0.059, 0.02, 0.0, 3)
    check(abs(pw - 0.89) < 0.02, f"功效公式与 prelim/power_calc.py 一致（Γ = 0.2、σ = 0.02、3 种子：{pw:.2f}，原表 0.89）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--skip-manifest", action="store_true")
    ap.add_argument("--keep", action="store_true", help="保留临时输出目录")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    os.environ["PYTHONHASHSEED"] = "0"
    if a.write_manifest:
        write_manifest()
        return 0
    if not a.skip_manifest:
        t0_manifest()
    t1_pure()
    tmp = Path(tempfile.mkdtemp(prefix="rt_selftest_"))
    try:
        ref = t2_pipeline(tmp)
        t3_resume(tmp, ref)
        t4_known_answer()
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"临时输出：{tmp}")
    print("\n全部自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
