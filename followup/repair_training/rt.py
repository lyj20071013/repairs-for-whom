"""第 5 篇先导 P0 的总入口（论文方案.md 第 5 节）。在服务器上一般通过 run_bg.sh 后台运行。

单个阶段：
  python rt.py --stage games|collect|positions|propose|design|label|select|fresh|thoughts|export|factcheck
  python rt.py --stage positions --which train|tk
  python rt.py --stage propose --proposer q7|qflash
  python rt.py --stage train            （= python rt_train.py --all：依次训练登记的 8 个模型，各在子进程里）
  python rt.py --stage formatcheck | evaluate [--force] | analyze
  python rt.py --stage apicheck          （flash 接口连通性；不打印密钥）

流水线（每个阶段在单独的子进程里，显存在阶段之间完全释放；任何一步失败就停；重跑同一命令从断点接着跑）：
  python rt.py --stage data             games -> collect -> positions(train, tk) -> propose(q7, qflash) -> design
                                         -> label -> select -> fresh -> thoughts -> export -> factcheck
  python rt.py --stage p0               data -> train -> formatcheck（不通过就停）-> evaluate -> analyze
  python rt.py --stage smoke            --profile smoke 的 p0（out_rt_smoke/，真实模型极小规模，结果不用）

--profile p0（默认，out_rt/）| smoke（out_rt_smoke/）| mock（out_rt_mock/，模拟环境 + 模拟模型，selftest 用）
"""
import argparse
import os
import subprocess
import sys
import time

import rt_common as C
from rt_common import log

OUT = {"p0": "out_rt", "smoke": "out_rt_smoke", "mock": "out_rt_mock"}
DATA = [("games", []), ("collect", []), ("positions", ["--which", "train"]), ("positions", ["--which", "tk"]),
        ("propose", ["--proposer", "q7"]), ("propose", ["--proposer", "qflash"]), ("design", []), ("label", []),
        ("select", []), ("fresh", []), ("thoughts", []), ("export", []), ("factcheck", [])]


def sub(a, stage, extra=()):
    cmd = [sys.executable, os.path.abspath(__file__), "--stage", stage, "--profile", a.profile,
           "--out-dir", str(C.STATE["out"])] + list(extra)
    log(f"===== {stage} {' '.join(extra)} =====")
    t0 = time.time()
    r = subprocess.run(cmd).returncode
    log(f"===== {stage} {' '.join(extra)} 结束：退出码 {r}，{(time.time() - t0) / 60:.1f} 分钟 =====")
    return r


def pipeline(a, steps):
    for st, ex in steps:
        r = sub(a, st, ex)
        if r:
            log(f"[{a.stage}] {st} 失败或未完成（退出码 {r}）；后面的步骤没有执行。处理后重跑同一命令会从断点接着跑。")
            return r
    return 0


def run_one(a):
    import rt_data as D
    st = a.stage
    if st == "games":
        return D.stage_games()
    if st == "collect":
        return D.stage_collect()
    if st == "positions":
        return D.stage_positions(a.which)
    if st == "propose":
        return D.stage_propose(a.proposer)
    if st == "design":
        return D.stage_design()
    if st == "label":
        return D.stage_label()
    if st == "select":
        return D.stage_select()
    if st == "fresh":
        return D.stage_fresh()
    if st == "thoughts":
        return D.stage_thoughts()
    if st == "export":
        return D.stage_export()
    if st == "factcheck":
        return D.stage_factcheck()
    if st == "train":
        import rt_train
        rc = 0
        for g, s in C.wanted_runs():
            if rt_train.is_done(g, s):
                log(f"[train {C.model_name(g, s)}] 已完成，跳过")
                continue
            if C.STATE["mock"]:
                rt_train.mock_train(g, s)
                continue
            r = subprocess.run([sys.executable, os.path.join(C.ROOT, "rt_train.py"), "--group", g, "--seed", str(s),
                                "--profile", a.profile, "--out-dir", str(C.STATE["out"])]).returncode
            if r:
                log(f"[train {C.model_name(g, s)}] 失败（退出码 {r}）")
                return r
        return rc
    if st == "formatcheck":
        import rt_eval
        return rt_eval.stage_formatcheck()
    if st == "evaluate":
        import rt_eval
        return rt_eval.stage_evaluate(force=a.force)
    if st == "analyze":
        import rt_analyze
        return rt_analyze.stage_analyze()
    if st == "apicheck":
        import judge_backends as JB
        j = JB.make_judge("api", model="qwen3.8-flash", thinking=False)
        out = j.ask([[{"role": "user", "content": "Reply with exactly: Action 1: look"}]])[0]
        ok = out["error"] is None and "look" in (out["text"] or "").lower()
        log(f"[apicheck] {'正常' if ok else '有问题'}：回答 {out['text']!r}，错误 {out['error']}，用量 {out.get('usage')}")
        return 0 if ok else 1
    raise SystemExit(f"未知阶段 {st}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--profile", default="p0", choices=list(OUT))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--which", choices=["train", "tk"])
    ap.add_argument("--proposer", choices=["q7", "qflash"])
    ap.add_argument("--force", action="store_true", help="evaluate：格式检查没通过也继续（只在按 5.5 处理并登记之后用）")
    a = ap.parse_args()
    if a.stage == "smoke":
        a.profile = "smoke"
    C.apply_profile(a.profile)
    C.use_out(a.out_dir or (C.ROOT / OUT[a.profile]))
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if a.stage == "data":
        return pipeline(a, DATA)
    if a.stage in ("p0", "smoke"):
        r = pipeline(a, DATA + [("train", []), ("formatcheck", [])])
        if r:
            return r
        return pipeline(a, [("evaluate", []), ("analyze", [])])
    return run_one(a)


if __name__ == "__main__":
    sys.exit(main())
