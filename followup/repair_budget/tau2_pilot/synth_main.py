"""合成数据：检验 main_pool.py w0 与 main_analyze.py（不需要 τ²）。埋入已知的效应，看读数能否读出来。
场景：null（π1 = π0）、better（π1 自由接手 +0.3）、action（π1 在一半位置上换了最好候选）。"""
import json, random, sys, hashlib
from pathlib import Path
sc, out = sys.argv[1], Path(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = random.Random(seed); out.mkdir(parents=True, exist_ok=True)
design, adm, ro = [], [], []
def runs(model, pos, cand, p):
    for k in range(6):
        ro.append({"pos_id": pos, "cand_id": cand, "executor": model, "k": k, "temp": 0.7, "score": "orig_task",
                   "reward": float(rng.random() < p), "termination": "TerminationReason.USER_STOP"})
folds = {}
for t in range(100):
    dom = "retail" if t < 60 else "airline"; tid = str(t); fold = "A" if t % 2 == 0 else "B"
    folds[f"{dom}:{tid}"] = fold
    task_eff = rng.gauss(0, 0.1)
    for j in range(8):
        pos = f"p{t:03d}_{j}"
        nrep = rng.choice([1, 2, 3])
        pf0 = min(max(rng.choice([0.0, 0.1, 0.3, 0.6, 0.9]) + task_eff, 0), 1)
        cands = [{"cand_id": "c0", "kinds": ["orig"], "msg": {}, "restore_ok": True}]
        pc0 = {}
        for i in range(nrep):
            cid = f"c{i+1}"; cands.append({"cand_id": cid, "kinds": ["llm"], "msg": {}, "restore_ok": True})
            pc0[cid] = min(max(pf0 + rng.choice([-0.2, 0.0, 0.1, 0.4]), 0), 1)
            adm.append({"pos_id": pos, "cand_id": cid, "kinds": ["llm"], "ok": True})
        adm.append({"pos_id": pos, "cand_id": "c0", "kinds": ["orig"], "ok": True})
        cands.append({"cand_id": "free", "kinds": ["free"], "msg": None, "restore_ok": True})
        design.append({"pos_id": pos, "src": "x", "sim_id": "x", "task_id": tid, "t": 2, "domain": dom, "fold": fold, "cands": cands})
        runs("4B", pos, "free", pf0); runs("4B", pos, "c0", pf0)
        for c, p in pc0.items(): runs("4B", pos, c, p)
        # π1
        pf1, pc1 = pf0, dict(pc0)
        if sc in ("better", "action"):
            pf1 = min(1, pf0 + 0.3); pc1 = {c: min(1, p + 0.3) for c, p in pc0.items()}
        if sc == "action" and nrep >= 2 and rng.random() < 0.5:
            best = max(pc0, key=pc0.get); other = [c for c in pc0 if c != best][0]
            pc1[best] = pf1; pc1[other] = min(1, pf1 + 0.5)
        m1 = "4B-" + fold
        runs(m1, pos, "free", pf1)
        for c, p in pc1.items(): runs(m1, pos, c, p)
        if fold == "A":
            runs("4B-A2", pos, "free", pf1)
            for c, p in pc1.items(): runs("4B-A2", pos, c, p)
w = lambda n, rows: (out / n).write_text("".join(json.dumps(r) + "\n" for r in rows))
w("design_run.jsonl", design); w("adm.jsonl", adm); w("ro.jsonl", ro)
(out / "folds.json").write_text(json.dumps(folds))
print(sc, len(design), "位置", len(ro), "次续跑")
