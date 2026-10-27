"""WebShop feasibility smoke checks (analysis plan section 6.13).

Checks installation, data/search fingerprints, prefix-replay consistency,
action parsing and baseline task performance without inspecting repair effects.
Run check, env, run, report or all from the experiment root. Select models with
--models q7,l8 or --models q14; the run stage requires --model.
The selftest stage uses synthetic products and mock inference and is not a
measurement of real environment performance. Compatible interrupted runs resume.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from alfw import acfg                       # noqa: E402
from webshop import wcfg, ws_agent, ws_pool, ws_rollout   # noqa: E402

STATE = {"mock": False, "out": ROOT / ("out_ws_smoke" if wcfg.MAX_STEPS == 15 else f"out_ws_smoke{wcfg.MAX_STEPS}"), "stubs": None, "python": None, "webshop_dir": None,
         "n_goals": None, "models": None}
S = wcfg.SMOKE


def log(msg):
    print(msg, flush=True)
    try:
        STATE["out"].mkdir(parents=True, exist_ok=True)
        with open(STATE["out"] / "ws_smoke.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("[%m-%d %H:%M:%S] ") + msg + "\n")
    except OSError:
        pass


def P(name, x=None):
    o = STATE["out"]
    return {"env": o / "env_check.json", "goals": o / "goals.json", "episodes": o / f"episodes_{x}.jsonl",
            "timing": o / f"timing_{x}.json", "report": o / "smoke_report.md", "summary": o / "smoke_summary.json",
            "workers": o / "worker_logs"}[name]


def jdump(path, obj):
    """Write JSON：Write the processs own provisional document first、Falling Disk（fsync）、Read Back Check，Atoms rename。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")
    tmp = Path(f"{path}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if tmp.read_bytes() != data:
        raise OSError(f"Write {path} Read later inconsistently（The disks full？df -h Take a look）")
    os.replace(tmp, path)


def jload(path):
    raw = Path(path).read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RuntimeError(f"{path} incomplete JSON（{len(raw)} Bytes，Start {raw[:40]!r}）：{e}。"
                           "likely a full disk，or another process ws_smoke Hes also writing this folder") from None


def jl_read(path):
    out = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


LOCK_ENV = "WS_SMOKE_LOCK"


def take_lock():
    """The same output folder allows only one folder at the same time ws_smoke（all and the childs progress will be the parents lock.）。if locking fails, report the current lock owner。"""
    import fcntl
    out = STATE["out"]
    if os.environ.get(LOCK_ENV) == str(out):
        return None
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / ".ws_smoke.lock", "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.seek(0)
        return f.read().strip() or "unknown"
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()}：{' '.join(sys.argv[1:])}（{time.strftime('%m-%d %H:%M:%S')} Action）")
    f.flush()
    STATE["_lock"] = f                             # Automatic release at the end of the process
    os.environ[LOCK_ENV] = str(out)
    return None


def disk_free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def make_pool(n=None):
    return ws_pool.WSPool(n_workers=n, log_dir=P("workers"), python=STATE["python"], webshop_dir=STATE["webshop_dir"],
                          stubs=STATE["stubs"])


def models():
    return STATE["models"] or S["MODELS"]


def _ds():
    """ds_alfw（Model Backend、Awaiting、Time）；It rewrites its log to: out_ws_smoke/ds.log，No writing ALFWorld Output directory for %1。"""
    import ds_alfw as DS
    if STATE.get("_ds_out") != STATE["out"]:
        DS.use_out(STATE["out"])
        STATE["_ds_out"] = STATE["out"]
    return DS


# ================================================================ check
def stage_check():
    ok = True
    py = Path(STATE["python"] or wcfg.WS_PYTHON)
    wd = Path(STATE["webshop_dir"] or wcfg.WEBSHOP_DIR)

    def line(c, msg):
        nonlocal ok
        ok = ok and bool(c)
        print(("  ✓ " if c else "  ✗ ") + msg, flush=True)
    line(py.exists(), f"WebShop Environmental Python：{py}（on mismatch export WS_PYTHON=...）")
    line((wd / "web_agent_site" / "envs" / "web_agent_text_env.py").exists(), f"WebShop Code：{wd}（on mismatch export WS_DIR=...）")
    for f in wcfg.DATA_FILES:
        p = wd / "data" / f
        line(p.exists(), f"data {p}" + (f"（{p.stat().st_size / 1e6:.1f} MB）" if p.exists() else ""))
    backend = wcfg.search_backend()
    if backend == "pyserini":
        idx = wd / "search_engine" / wcfg.INDEX_NAME
        line(idx.exists() and any(idx.iterdir()), f"Retrieval Index {idx}")
        jh = ws_pool.java_home(str(py))
        line(bool(jh), f"Java：{jh}")
    else:
        line((wd / "search_engine" / wcfg.RESOURCES_NAME / "documents.jsonl").exists(), "BM25 required by the fallback search backend documents.jsonl")
    print(f"  · Search backend：{backend}（WS_SEARCH）；{wcfg.NUM_PRODUCTS} Commodity allocation", flush=True)
    mem = wcfg.mem_available_gb()
    print(f"  · Available Memory {mem if mem is None else round(mem, 1)} GB → Default Drive {wcfg.n_workers()} items WebShop Subprocess"
          f"（Every contract {wcfg.WORKER_GB} GB；WS_WORKERS Changeable）", flush=True)
    if ok:
        try:
            with make_pool(1) as pl:
                h = pl.hello()[0]
            print("  ✓ WebShop Subprocess start：" + json.dumps(h, ensure_ascii=False), flush=True)
            need = STATE["n_goals"] or S["N_GOALS"]
            line(h["n_goals"] >= need, f"Objective {h['n_goals']} items（Smoke {need} items）")
        except Exception as e:                     # noqa: BLE001
            line(False, f"WebShop Subprocess start failed：{e}")
    if not STATE["mock"]:
        try:
            import selftest as grid_selftest
            grid_selftest.t0_manifest()
        except AssertionError as e:
            line(False, f"Code List：{e}")
        import ds_alfw as DS                     # check Do not write any files，Do Not Call _ds()
        for m in models():                       # Just check out the models that this server is running（--models）
            d = DS.model_dir(m)
            line(Path(d).exists(), f"model {m}：{d}")
        print(f"  · Portfolio free memory ratio {DS.gpu_free_frac()}", flush=True)
    print("[ws check] " + ("ready" if ok else "issues found，see above ✗"), flush=True)
    return 0 if ok else 1


# ================================================================ env：Consistency and replay certainty
def _walk(pool, goal, rng, n_steps, instr_words):
    """Randomly swimming：Search box page search command 2–3 Words；One random point for other pages（before 8 Cant move buy now），Sometimes you order an item that doesnt exist。"""
    rid = f"walk|{goal}|{rng.random():.12f}"
    o = pool.open([{"rid": rid, "goal": goal, "prefix": []}])[rid]
    st, cmds, hashes, valid = o["state"], [], [o["state"]["hash"]], []
    for j in range(n_steps):
        if st["done"]:
            break
        u = rng.random()
        if u < 0.12:
            c = "click[no such button]"
        elif st["search"]:
            c = "search[" + " ".join(rng.sample(instr_words, min(len(instr_words), rng.randint(2, 3)))) + "]"
        else:
            opts = [x for x in st["clicks"] if j >= 8 or x != "buy now"] or st["clicks"]
            c = f"click[{rng.choice(opts)}]" if opts else "click[back to search]"
        r = pool.step([{"rid": rid, "cmd": c}])[rid]
        st = r["state"]
        cmds.append(c)
        hashes.append(st["hash"])
        valid.append(r["valid"])
    pool.close([rid])
    return {"goal": goal, "cmds": cmds, "hashes": hashes, "valid": valid, "done": st["done"], "reward": st["reward"]}


def stage_env():
    t0 = time.time()
    res = {"time": time.strftime("%Y-%m-%d %H:%M"), "search_backend": wcfg.search_backend(), "checks": {}}
    wd = Path(STATE["webshop_dir"] or wcfg.WEBSHOP_DIR)
    res["data_sha256"] = {f: sha256(wd / "data" / f) for f in wcfg.DATA_FILES if (wd / "data" / f).exists()}
    man = wd / "data" / wcfg.DATA_MANIFEST
    if man.exists():                               # make_100k.py generated manifest：The data on the server is consistent with the local intercept
        outs = jload(man).get("outputs", {})
        res["checks"]["data_matches_manifest"] = all(
            res["data_sha256"].get(f) == outs[f]["sha256"] for f in (wcfg.PRODUCT_FILE, wcfg.ATTR_FILE) if f in outs) \
            and all(f in outs for f in (wcfg.PRODUCT_FILE, wcfg.ATTR_FILE))
    res["mem_available_gb"] = wcfg.mem_available_gb()
    res["n_workers_default"] = wcfg.n_workers()
    with make_pool(2) as pl:
        hello = pl.hello()
        res["hello"] = hello
        same = len({(h["goals_hash"], h["products_hash"], h["n_goals"], h["search_fp"]["hash"]) for h in hello}) == 1
        res["checks"]["workers_same_goals"] = same
        n_goals = hello[0]["n_goals"]
        need = STATE["n_goals"] or S["N_GOALS"]
        res["checks"]["enough_goals"] = n_goals >= need
        log(f"[ws env] {n_goals} Target、{hello[0]['n_products']} Commodities；Target list and search for two subprocesses"
            + ("identical" if same else "different") + f"；Search backend {hello[0]['search_backend']}；"
            f"Every Subprocess {hello[0].get('rss_mb') or 0:.0f} MB、Start {hello[0]['init_s']} seconds")
        if n_goals < need:
            res["pass"] = False
            jdump(P("env"), res)
            log(f"[ws env] only this many goals are available {n_goals} items，Smoke {need} items → failed（do not silently reduce the goal count）")
            return 1
        rng = random.Random(S["SEED"])
        n = need
        goals = sorted(rng.sample(range(n_goals), n))
        info = pl.goal(goals)
        jdump(P("goals"), {"seed": S["SEED"], "n_goals_total": n_goals, "goals": goals,
                           "instructions": {str(g): info[str(g)]["instruction"] for g in goals},
                           "goal_info": info, "goals_hash": hello[0]["goals_hash"]})
        drng = random.Random(S["SEED"] + 1)
        det_goals = sorted(drng.sample(range(n_goals), min(S["N_DET_GOALS"], n_goals)))
        dinfo = pl.goal(det_goals)
        walks = []
        for g in det_goals:
            words = [w for w in dinfo[str(g)]["instruction"].lower().replace(",", " ").split() if len(w) > 2] or ["item"]
            walks.append(_walk(pl, g, drng, S["DET_STEPS"], words))
        items = [{"goal": w["goal"], "prefix": w["cmds"]} for w in walks]
        rep0 = pl.probe(items, worker=0)
        rep1 = pl.probe(items, worker=1)
        stats = pl.stats()
    with make_pool(1) as pl2:                      # New process
        rep2 = pl2.probe(items, worker=0)
    ok = {name: all(r["hashes"] == w["hashes"] and r["valid"] == w["valid"] for r, w in zip(rep, walks))
          for name, rep in (("same_process", rep0), ("other_process", rep1), ("fresh_process", rep2))}
    res["checks"].update({f"replay_{k}": v for k, v in ok.items()})
    n_steps = sum(len(w["cmds"]) for w in walks)
    res["walks"] = {"n": len(walks), "steps": n_steps, "invalid": sum(not v for w in walks for v in w["valid"]),
                    "done": sum(w["done"] for w in walks), "reward_mean": sum(w["reward"] for w in walks) / max(1, len(walks))}
    res["worker_stats"] = stats
    res["pass"] = all(res["checks"].values())
    res["wall_s"] = round(time.time() - t0, 1)
    jdump(P("env"), res)
    log(f"[ws env] Replay Same：Same process {ok['same_process']}、Another process {ok['other_process']}、New process {ok['fresh_process']}"
        f"（{len(walks)} Ill go for a swim、{n_steps} steps，Of which not enforceable {res['walks']['invalid']}）→ " + ("passed" if res["pass"] else "failed"))
    log(f"[ws env] smoke-test goals {len(goals)} items（seed {S['SEED']}）-> {P('goals')}")
    return 0 if res["pass"] else 1


# ================================================================ run：A model
def make_backend(model):
    if STATE["mock"]:
        return ws_agent.MockBackend()
    acfg.MODEL_CONTEXT = wcfg.MODEL_CONTEXT          # WebShop use 16,384（6.13 Revision 2）；Change values only in this process
    return _ds().make_exec_backend(model)


def stage_run(model):
    if not P("goals").exists():
        sys.exit("[ws run] Not yet goals.json（Run first env）")
    if P("env").exists() and not jload(P("env")).get("pass"):
        sys.exit("[ws run] Environmental check failed（env_check.json），No running models")
    gl = jload(P("goals"))
    specs = [{"rid": f"{model}|g{g}", "goal": g, "seed": ws_agent.derive_seed("WS-SMOKE", g)} for g in gl["goals"]]
    have = {e["rid"] for e in jl_read(P("episodes", model))}
    todo = [s for s in specs if s["rid"] not in have]
    log(f"[ws run {model}] {len(specs)} Target，Completed {len(specs) - len(todo)}，Stay {len(todo)}")
    if not todo:
        return 0
    if not STATE["mock"]:
        DS = _ds()
        try:
            import selftest as grid_selftest
            grid_selftest.t0_manifest()
        except AssertionError as e:
            log(f"Code version check failed：{e}")
            return 1
        if not DS.wait_gpu(f"ws-{model}"):
            return 1
    backend = make_backend(model)
    f = open(P("episodes", model), "a", encoding="utf-8")

    def on_done(e):
        e["model"] = model
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
        f.flush()
    try:
        with make_pool() as pl:
            tm = ws_rollout.run_stream(backend, pl, todo, acfg.TEMPERATURE, on_done, max_steps=wcfg.MAX_STEPS,
                                       log=log)
        old = jload(P("timing", model)) if P("timing", model).exists() else {"runs": []}
        gpu = None
        if not STATE["mock"]:
            DS = _ds()
            gpu = DS.gpu_info()
        old["runs"].append({"wall": tm["wall"], "policy_steps": tm["policy_steps"], "episodes": len(todo),
                            "gen_s": tm["gen"], "env_s": tm["env"], "max_prompt_tokens": tm["max_prompt_tokens"],
                            "retries": tm["retries"], "gpu": gpu, "time": time.strftime("%Y-%m-%d %H:%M")})
        jdump(P("timing", model), old)
        log(f"[ws run {model}] complete：{tm['wall'] / 60:.1f} minutes，{tm['policy_steps']} steps（Generate {tm['gen']:.0f} seconds、environment {tm['env']:.0f} seconds）")
    finally:
        f.close()
        if not STATE["mock"]:
            DS = _ds()
            DS.release_vllm(backend)
    left = [s for s in specs if s["rid"] not in {e["rid"] for e in jl_read(P("episodes", model))}]
    return 0 if not left else 1


# ================================================================ report
def summarize(eps):
    st = [ws_rollout.episode_stats(e) for e in eps]
    n_steps = sum(s["n"] for s in st)
    bought = [e for e in eps if e["done"]]
    pages = Counter(p for e in eps for p in e["pages"])
    acts = [a for e in eps for a in e["n_actions"]]
    return {"n": len(eps), "success": sum(e["won"] for e in eps) / max(1, len(eps)),
            "reward_mean": sum(e["reward"] for e in eps) / max(1, len(eps)),
            "bought": len(bought) / max(1, len(eps)),
            "bought_wrong": sum(1 for e in bought if not e["won"]) / max(1, len(eps)),
            "timeout": sum(1 for e in eps if not e["done"]) / max(1, len(eps)),
            "reward_if_bought": (sum(e["reward"] for e in bought) / len(bought)) if bought else None,
            "steps_mean": n_steps / max(1, len(eps)),
            "invalid_frac": sum(s["invalid"] + s["no_action"] for s in st) / max(1, n_steps),
            "first_bad_frac": sum(s["first_bad"] for s in st) / max(1, n_steps),
            "retries_per_step": sum(s["retries"] for s in st) / max(1, n_steps),
            "alias_frac": sum(s["alias"] for s in st) / max(1, n_steps),
            "trunc_frac": sum(s["trunc"] for s in st) / max(1, n_steps),
            "drop_frac": sum(s["drop"] for s in st) / max(1, n_steps),
            "searches_mean": sum(s["searches"] for s in st) / max(1, len(eps)),
            "revisit_mean": sum(s["revisit"] for s in st) / max(1, len(eps)),
            "max_prompt_tokens": max([s["max_ntok"] for s in st] or [0]),
            "actions_per_page_mean": sum(acts) / max(1, len(acts)), "pages": dict(pages)}


ENV_KEYS = ("n_goals", "n_products", "goals_hash", "products_hash", "search_fp", "search_backend", "num_products")


def env_fingerprint(env):
    h = env["hello"][0]
    fp = {k: (h[k]["hash"] if k == "search_fp" else h.get(k)) for k in ENV_KEYS}
    fp.update(data_sha256=env.get("data_sha256"), seed=h.get("seed"))
    return fp


def compare_peers(env):
    """From other servers env_check_<name>.json（Download to the same folder）：The environmental fingerprints are the same as this machine、Same target sample。"""
    out = {}
    if not env:
        return out
    mine = env_fingerprint(env)
    goals = jload(P("goals"))["goals"] if P("goals").exists() else None
    for pth in sorted(STATE["out"].glob("env_check_*.json")):
        name = pth.stem[len("env_check_"):]
        pe = jload(pth)
        fp = env_fingerprint(pe) if pe.get("hello") else {}
        diff = [k for k in mine if fp.get(k) != mine[k]]
        pg = STATE["out"] / f"goals_{name}.json"
        same_goals = pg.exists() and goals is not None and jload(pg)["goals"] == goals
        out[name] = {"pass": bool(pe.get("pass")), "same_env": not diff, "diff": diff, "same_smoke_goals": same_goals,
                     "time": pe.get("time")}
    return out


def stage_report():
    env = jload(P("env")) if P("env").exists() else None
    res = {"env": env, "models": {}, "gates": {}}
    peers = compare_peers(env)
    res["peers"] = peers
    G = S["GATES"]
    for m in models():
        eps = jl_read(P("episodes", m))
        if not eps:
            continue
        s = summarize(eps)
        tm = jload(P("timing", m)) if P("timing", m).exists() else None
        if tm:
            w = sum(r["wall"] for r in tm["runs"])
            s["sec_per_episode"] = w / max(1, sum(r["episodes"] for r in tm["runs"]))
        res["models"][m] = s
        g = {"invalid": s["invalid_frac"] <= G["max_invalid_frac"], "trunc": s["trunc_frac"] <= G["max_trunc_frac"],
             "drop": s["drop_frac"] <= G["max_drop_frac"]}
        if m == "q7":
            lo, hi = G["q7_success"]
            g["success_range"] = lo <= s["success"] <= hi
        res["gates"][m] = g
    peers_ok = all(p["pass"] and p["same_env"] and p["same_smoke_goals"] for p in peers.values())
    res["pass"] = bool(env and env.get("pass")) and peers_ok and all(all(g.values()) for g in res["gates"].values()) \
        and set(res["models"]) == set(models())
    jdump(P("summary"), res)
    L = ["# WebShop Feasibility smoke（analysis plan.md 6.13）", "", f"Generated in {time.strftime('%Y-%m-%d %H:%M')}。Just see the process、Format and Success Rate，Dont look at any restoration effects。", ""]
    if env:
        h = env["hello"][0]
        L += ["## environment", "",
              f"- {h['n_goals']} Target（People write orders）、{h['n_products']} Commodities；Search backend `{h['search_backend']}`；"
              f"WebShop environment Python {h['python']}；version {json.dumps(h['versions'], ensure_ascii=False)}",
              f"- Matching target lists between subprocesses：{env['checks']['workers_same_goals']}（Target Hash `{h['goals_hash']}`，Commodities and prices `{h['products_hash']}`）",
              f"- Replaying Progressively：Same process {env['checks']['replay_same_process']}、Another process {env['checks']['replay_other_process']}、"
              f"New process {env['checks']['replay_fresh_process']}（{env['walks']['n']} random walks、{env['walks']['steps']} steps）",
              f"- Searching fingerprints `{h['search_fp']['hash']}`（{h['search_fp']['queries']} Query、{h['search_fp']['hits']} Outcome）；"
              f"Every Subprocess {h.get('rss_mb') or 0:.0f} MB、Start {h['init_s']} seconds；The memory is available on this machine {env.get('mem_available_gb')} GB",
              f"- data sha256：" + "；".join(f"`{k}` {v[:16]}…" for k, v in env.get("data_sha256", {}).items())
              + (f"（and make_100k Same list：{env['checks']['data_matches_manifest']}）" if "data_matches_manifest" in env["checks"] else "")]
        for name, p in peers.items():
            L.append(f"- With Servers `{name}` Environment（env_check_{name}.json，{p['time']}）：Environmental inspection{'passed' if p['pass'] else 'failed'}；"
                     f"Objective、Commodities and prices、Searching fingerprints、data {'all identical' if p['same_env'] else 'different：' + '、'.join(p['diff'])}；"
                     f"smoke-test goals{'identical' if p['same_smoke_goals'] else 'Different or missing goals_' + name + '.json'}")
        L.append("")
    L += ["## Three models（Each model has the same target、Each 1 episodes，temperature 0.7，At most %d steps）" % wcfg.MAX_STEPS, "",
          "| model | Number of bureaux | success rate | Average incentive | purchased the wrong item | timeout | Average number of steps | not executable / steps | Initial answer cannot be implemented / steps | truncation | Lose history | Maximum prompts | Optional Actions per Page | seconds / episodes |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, s in res["models"].items():
        L.append(f"| {m} | {s['n']} | {s['success']:.2f} | {s['reward_mean']:.2f} | {s['bought_wrong']:.2f} | {s['timeout']:.2f} | "
                 f"{s['steps_mean']:.1f} | {s['invalid_frac']:.1%} | {s['first_bad_frac']:.1%} | {s['trunc_frac']:.1%} | "
                 f"{s['drop_frac']:.1%} | {s['max_prompt_tokens']} | {s['actions_per_page_mean']:.1f} | "
                 f"{s.get('sec_per_episode', float('nan')):.1f} |")
    L += ["", "## decision（Criteria for the adoption of registration）", ""]
    L.append(f"- environment：{'passed' if env and env.get('pass') else 'failed'}"
             + (f"；Consistency across servers：{'passed' if peers_ok else 'failed'}" if peers else ""))
    for m, g in res["gates"].items():
        L.append(f"- {m}：" + "、".join(f"{k} {'✓' if v else '✗'}" for k, v in g.items()))
    L += ["", f"**Conclusions：{'passed' if res['pass'] else 'needs inspection'}**", ""]
    P("report").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"[ws report] -> {P('report')}：" + ("passed" if res["pass"] else "needs inspection"))
    return 0 if res["pass"] else 1


# ================================================================ all
def sub(stage, extra=()):
    cmd = [sys.executable, str(Path(__file__).resolve()), stage, "--out-dir", str(STATE["out"])] + list(extra)
    if STATE["mock"]:
        cmd += ["--mock", "--ws-python", STATE["python"], "--ws-dir", STATE["webshop_dir"], "--stubs", STATE["stubs"]]
    if STATE["n_goals"]:
        cmd += ["--n-goals", str(STATE["n_goals"])]
    if STATE["models"]:
        cmd += ["--models", ",".join(STATE["models"])]
    log(f"----- Subprocess：{' '.join(cmd[2:])}")
    rc = subprocess.call(cmd, env=dict(os.environ, PYTHONHASHSEED=acfg.HASH_SEED))
    if rc != 0:
        log(f"----- Subprocess Failed（rc={rc}）：{stage}；after fixing the issue, rerun the same command to resume here")
    return rc


def stage_all():
    free = disk_free_gb(STATE["out"])
    log(f"[ws all] Output folders left on disk {free:.1f} GB")
    if free < 1.0 and not STATE["mock"]:
        log("[ws all] Disk is not left 1 GB，free disk space first（df -h）")
        return 1
    rc = sub("env")
    if rc:
        return rc
    for m in models():
        rc = sub("run", ["--model", m])
        if rc:
            return rc
    return sub("report")


# ================================================================ selftest（Claude Workspace：Synthetic Data + mock model）
def _synthetic_webshop(dst, n_products=40, seed=5):
    """copy vendor Unlock it dst，create a dataset with the same structure as WebShop Small data sets with the same data，Generate search for documents.jsonl。"""
    with tarfile.open(wcfg.VENDOR) as t:
        t.extractall(dst)
    rng = random.Random(seed)
    nouns = ["shoes", "jacket", "lamp", "mug", "pillow", "backpack", "watch", "shirt"]
    mats = ["leather", "cotton", "steel", "wool", "ceramic", "plastic"]
    colors = ["red", "blue", "black", "white", "green"]
    sizes = ["small", "medium", "large", "x-large"]
    products, ins, human = [], {}, {}
    for i in range(n_products):
        asin = "B0" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(8))
        noun, mat = rng.choice(nouns), rng.choice(mats)
        price = round(rng.uniform(5, 90), 2)
        pricing = f"${price}" if rng.random() < 0.7 else f"${price} - ${round(price * 1.5, 2)}"
        opts = {}
        if rng.random() < 0.8:
            opts["Color"] = [{"value": c.title(), "image": None} for c in rng.sample(colors, 3)]
        if rng.random() < 0.6:
            opts["Size"] = [{"value": s.title(), "image": None} for s in rng.sample(sizes, 3)]
        name = f"{mat.title()} {noun.title()} Model {i}"
        products.append({"asin": asin, "name": name, "full_description": f"A {mat} {noun} of fine quality, item {i}.",
                         "small_description": [f"made of {mat}", f"great {noun}"], "pricing": pricing,
                         "customization_options": opts or None, "images": [f"http://img/{asin}.jpg"],
                         "category": "home", "query": noun, "product_category": f"Home › {noun.title()} › {mat.title()}"})
        ins[asin] = {"attributes": [mat, noun], "instruction": f"i want a {mat} {noun}",
                     "instruction_attributes": [mat]}
        want = [o["value"].lower() for vs in (opts or {}).values() for o in vs[:1]]
        human[asin] = [{"instruction": f"i need a {mat} {noun}" + (f" in {' and '.join(want)}" if want else ""),
                        "instruction_attributes": [mat], "instruction_options": want}]
    d = Path(dst) / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / wcfg.PRODUCT_FILE).write_text(json.dumps(products), encoding="utf-8")
    (d / wcfg.ATTR_FILE).write_text(json.dumps(ins), encoding="utf-8")
    (d / wcfg.HUMAN_FILE).write_text(json.dumps(human), encoding="utf-8")
    man = {"outputs": {f: {"sha256": sha256(d / f)} for f in (wcfg.PRODUCT_FILE, wcfg.ATTR_FILE)}}
    (d / wcfg.DATA_MANIFEST).write_text(json.dumps(man), encoding="utf-8")
    # On the server setup_webshop.sh number 7 Same step：Ill be right back ws_convert.py Run the original conversion script with the specified data file
    subprocess.run([sys.executable, str(ROOT / "webshop" / "ws_convert.py"), "--webshop-dir", str(dst),
                    "--product-file", wcfg.PRODUCT_FILE, "--attr-file", wcfg.ATTR_FILE, "--resources", wcfg.RESOURCES_NAME,
                    "--stubs", str(wcfg.STUBS), "--compat", str(wcfg.COMPAT), "--shims", str(wcfg.SHIMS)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def selftest():
    ok = [True]

    def check(c, msg):
        print(("  ✓ " if c else "  ✗ ") + msg, flush=True)
        ok[0] = ok[0] and bool(c)
    tmp = Path(tempfile.mkdtemp(prefix="ws_smoke_"))
    saved = dict(STATE)
    try:
        print("[ws selftest] Parsing and prompts")
        st = {"search": False, "clicks": ["back to search", "< prev", "red", "large", "buy now", "b0abc12345"]}
        pa = ws_agent.parse_action
        check(pa("Thought: x\nAction: click[Buy Now]", st) == {"cmd": "click[buy now]", "kind": "valid"}, "click Case")
        check(pa("Action: click[B0ABC12345]", st)["kind"] == "valid", "Uppercase for commodity number")
        check(pa("Action: buy now", st) == {"cmd": "click[buy now]", "kind": "alias"}, "Click without square brackets")
        check(pa("Action: search[red shoes]", st)["kind"] == "invalid", "When no search box is available search not executable")
        check(pa("Action: search[<r> x]", {"search": True, "clicks": []})["kind"] == "invalid"
              and pa("Action: search[<keywords>]", {"search": True, "clicks": []})["kind"] == "invalid",
              "Special Query & PhotoCamp Placeholder Not Executable")
        check(pa("Action: search[Red Shoes]", {"search": True, "clicks": []}) == {"cmd": "search[red shoes]", "kind": "valid"},
              "search Uniform lowercase")
        check(pa("no action here", st)["kind"] == "no_action", "Nothing Action rows")
        msgs = ws_agent.build_messages("buy red shoes", [{"cmd": "search[red shoes]", "obs": "x" * 1000}],
                                       {"obs": "WebShop\nInstruction:\nbuy red shoes\n[button] Buy Now [button_]",
                                        "search": False, "clicks": ["buy now"]}, 15)
        u = msgs[1]["content"]
        check("Instruction: buy red shoes" in u and "click[buy now]" in u and u.count("buy red shoes") == 1
              and ("x" * 301) not in u, "prompt：The command only appears once、History Page Cut、List executable actions")
        print("[ws selftest] Synthetic WebShop data + Original Code（Dependencies test_stubs，Retrieval BM25 Substitution）")
        wd = tmp / "WebShop"
        _synthetic_webshop(wd)
        os.environ["WS_SEARCH"] = "bm25"
        STATE.update(mock=True, out=tmp / "out", stubs=str(wcfg.STUBS), python=sys.executable, webshop_dir=str(wd),
                     n_goals=12)
        check(stage_check() == 0, "check")
        STATE["n_goals"] = 50                      # Synthetic data only 40 Target
        check(stage_env() == 1 and jload(P("env"))["checks"]["enough_goals"] is False and not P("goals").exists(),
              "When the target is not enough env mark as failed（do not silently reduce the goal count）")
        STATE["n_goals"] = 12
        check(stage_env() == 0, "env：Target list is consistent with search、Three resets are consistent and consistent")
        envr = jload(P("env"))
        check(envr["checks"].get("data_matches_manifest") is True and envr["checks"]["enough_goals"] is True,
              "data and manifest sha256 identical")
        h0 = envr["hello"][0]
        check(h0["product_file"] == wcfg.PRODUCT_FILE and h0["search_fp"]["hits"] > 0 and h0["n_goals"] == 40,
              f"Subprocess uses the specified data file、Weve got results from the fingerprint search（{h0['product_file']}，{h0['search_fp']}）")
        check(envr["walks"]["invalid"] > 0 and envr["walks"]["steps"] > 30, f"random walks include non-executable actions（{envr['walks']}）")
        for m in models():
            check(stage_run(m) == 0, f"run {m}（mock model）")
        eps = jl_read(P("episodes", "q7"))
        check(len(eps) == 12 and all(len(e["steps"]) <= wcfg.MAX_STEPS for e in eps), "Every target 1 episodes、Up to the maximum number of steps")
        check(any(e["done"] for e in eps) and any(not e["done"] for e in eps) or len(eps) > 0, "includes both purchases and timeouts（Or at least run through）")
        lines = P("episodes", "q7").read_text().splitlines()
        P("episodes", "q7").write_text("\n".join(lines[:5]) + "\n")
        check(stage_run("q7") == 0, "Redo complete after interruption")
        lines2 = P("episodes", "q7").read_text().splitlines()
        key = lambda ls: sorted(json.dumps({k: v for k, v in json.loads(x).items()}, sort_keys=True) for x in ls)
        check(key(lines2) == key(lines), "resumed episodes match the originals byte for byte（Same torrent、Same environment）")
        # continuation：Run from the prefix of the middle of the game，The prefix is gradually aligned with the original
        e0 = max(eps, key=lambda e: len(e["steps"]))
        k = min(3, len(e0["steps"]) - 1)
        with make_pool(1) as pl:
            o = pl.open([{"rid": "t|x", "goal": e0["goal"], "prefix": [s["cmd"] for s in e0["steps"][:k]]}])["t|x"]
            pl.close(["t|x"])
        same = [ws_agent.clean_obs(p["state"]["obs"], o["instruction"]) for p in o["prefix"]] == \
            [s["obs"] for s in e0["steps"][:k]]
        check(same, "Restored from Midway Prefix：The same page per step as the original")
        check(stage_report() == 0 or P("report").exists(), "report")
        # The same output folder cannot run two at once；Bad JSON Give a clear statement
        os.environ.pop(LOCK_ENV, None)
        check(take_lock() is None, "Get a lock for the output folder")
        env2 = {k: v for k, v in os.environ.items() if k != LOCK_ENV}
        r2 = subprocess.run([sys.executable, str(Path(__file__).resolve()), "report", "--out-dir", str(STATE["out"]),
                             "--mock"], env=env2, capture_output=True, text=True)
        check(r2.returncode == 2 and "The other one ws_smoke" in r2.stdout, "Another process uses the same folder at the same time → reject（rc 2）")
        bad = STATE["out"] / "bad.json"
        bad.write_bytes(b"")
        try:
            jload(bad)
            check(False, "Empty files should be punished for wrong")
        except RuntimeError as e:
            check("incomplete JSON" in str(e) and "0 Bytes" in str(e), "Empty JSON File gives a clear indication")
        bad.unlink()
        rep = P("report").read_text(encoding="utf-8")
        check("| q7 |" in rep and "| l8 |" in rep and "Replaying Progressively" in rep, "The report contains three models and environmental inspections")
        # Cross-server：The other one env_check / goals unchanged → identical；Change search fingerprint or smoke target → The reports not going through
        base = jload(P("summary"))["pass"]
        shutil.copy(P("env"), STATE["out"] / "env_check_peer.json")
        shutil.copy(P("goals"), STATE["out"] / "goals_peer.json")
        stage_report()
        sm = jload(P("summary"))
        check(sm["peers"]["peer"]["same_env"] and sm["peers"]["peer"]["same_smoke_goals"] and sm["pass"] == base,
              "The other server has the same environment → identical")
        pe = jload(STATE["out"] / "env_check_peer.json")
        pe["hello"][0]["search_fp"]["hash"] = "0" * 16
        jdump(STATE["out"] / "env_check_peer.json", pe)
        stage_report()
        sm = jload(P("summary"))
        check(sm["peers"]["peer"]["diff"] == ["search_fp"] and sm["pass"] is False
              and "Searching fingerprints" in P("report").read_text(encoding="utf-8") and "different：search_fp" in P("report").read_text(encoding="utf-8"),
              "search fingerprints differ → mark as failed")
        pg = jload(STATE["out"] / "goals_peer.json")
        pg["goals"] = pg["goals"][1:]
        jdump(STATE["out"] / "goals_peer.json", pg)
        shutil.copy(P("env"), STATE["out"] / "env_check_peer.json")
        stage_report()
        check(jload(P("summary"))["pass"] is False, "The smoke is different → mark as failed")
        for f in ("env_check_peer.json", "goals_peer.json"):
            (STATE["out"] / f).unlink()
        stage_report()
        # With original WebAgentTextEnv The results of the direct operation are identical word by word（Prove that the seal doesnt change environmental behavior）
        sys.path[:0] = [str(wcfg.STUBS), str(wcfg.COMPAT), str(wcfg.SHIMS), str(wd)]
        cwd = os.getcwd()
        os.chdir(wd)
        try:
            random.seed(wcfg.ENV_SEED)
            from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
            sys.path.insert(0, str(ROOT / "webshop"))
            import ws_data
            fp, _ = ws_data.apply(str(wd), wcfg.PRODUCT_FILE, wcfg.ATTR_FILE)
            env = WebAgentTextEnv(observation_mode=wcfg.OBS_MODE, file_path=fp, num_products=wcfg.NUM_PRODUCTS,
                                  human_goals=1)
            env.reset(session=e0["goal"])
            ok_obs = True
            for s in e0["steps"]:
                if s["valid"]:
                    env.step(s["cmd"])
                instr = env.instruction_text.split("Instruction:", 1)[-1].strip()
                ok_obs = ok_obs and ws_agent.clean_obs(env.observation, instr) == s["obs"]
            check(ok_obs, "With Direct Use Original WebAgentTextEnv Progressively Same")
        finally:
            os.chdir(cwd)
    finally:
        lk = STATE.get("_lock")
        if lk:
            lk.close()
        STATE.clear()
        STATE.update(saved)
        os.environ.pop("WS_SEARCH", None)
        os.environ.pop(LOCK_ENV, None)
        if os.environ.get("WS_KEEP"):
            print(f"[ws selftest] Keep temporary directory {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ws selftest] {'all passed' if ok[0] else 'failures detected'}")
    return 0 if ok[0] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "env", "run", "report", "all", "selftest"])
    ap.add_argument("--model", choices=["q7", "q14", "l8"])
    ap.add_argument("--models", default=None, help="all / report Just these models，Comma Separated（default q7,q14,l8）")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-goals", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--ws-python", default=None)
    ap.add_argument("--ws-dir", default=None)
    ap.add_argument("--stubs", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    STATE.update(mock=a.mock, out=(a.out_dir or STATE["out"]).resolve(), python=a.ws_python, webshop_dir=a.ws_dir,
                 stubs=a.stubs, n_goals=a.n_goals, models=a.models.split(",") if a.models else None)
    if a.cmd == "check":
        return stage_check()
    who = take_lock()
    if who:
        print(f"[ws {a.cmd}] The other one ws_smoke Working on it {STATE['out']}（{who}）；wait for completion，Or use it pgrep -af ws_smoke inspect", flush=True)
        return 2
    log(f"===== ws_smoke {a.cmd}{' --model ' + a.model if a.model else ''}  output {STATE['out']} =====")
    if a.cmd == "env":
        return stage_env()
    if a.cmd == "run":
        if not a.model:
            sys.exit("run Yes --model q7|q14|l8")
        return stage_run(a.model)
    if a.cmd == "report":
        return stage_report()
    if a.cmd == "all":
        return stage_all()
    return 0


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != acfg.HASH_SEED:
        os.environ["PYTHONHASHSEED"] = acfg.HASH_SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)
    sys.exit(main())
