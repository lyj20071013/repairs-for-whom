"""WebShop Environment pool：Main Process（vLLM Environment）Start several ws_worker.py Subprocess（WebShop Environment），Send and receive by line JSON。

Interface with alfw/pool.py Same idea：
    open([{rid, goal, prefix}])  -> {rid: {instruction, init, prefix, state, t}}
    step([{rid, cmd}])           -> {rid: {state, valid, dt}}
    close([rid])
    probe([{goal, prefix}])      -> [{state, hashes, valid, instruction}]（Order and Enter）
    hello() -> Every subprocess {n_goals, goals_hash, ...}；goal([idx]) -> Complete information about the target（For analysis only）
"""
import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from . import wcfg


def java_home(ws_python):
    """WebShop In the environment JDK（conda-forge openjdk）or systems JDK；No returns found None（pyserini Need it）。"""
    if os.environ.get("WS_JAVA_HOME"):
        return os.environ["WS_JAVA_HOME"]
    prefix = Path(ws_python).resolve().parent.parent
    cands = sorted(glob.glob(str(prefix / "lib" / "jvm" / "lib" / "server" / "libjvm.so"))) + \
        sorted(glob.glob(str(prefix / "lib" / "server" / "libjvm.so"))) + \
        sorted(glob.glob("/usr/lib/jvm/*/lib/server/libjvm.so"))
    if not cands:
        return os.environ.get("JAVA_HOME")
    return str(Path(cands[0]).parent.parent.parent)


class WSPool:
    def __init__(self, n_workers=None, log_dir=None, python=None, webshop_dir=None, stubs=None, search=None,
                 seed=None, mode=None, num_products=None):
        self.n = n_workers or wcfg.n_workers()
        self.python = python or wcfg.WS_PYTHON
        self.webshop_dir = str(webshop_dir or wcfg.WEBSHOP_DIR)
        search = search or wcfg.search_backend()
        log_dir = Path(log_dir or (wcfg.ROOT / "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [self.python, str(Path(__file__).resolve().parent / "ws_worker.py"), "--webshop-dir", self.webshop_dir,
               "--mode", mode or wcfg.OBS_MODE, "--num-products", str(num_products or wcfg.NUM_PRODUCTS),
               "--human-goals", str(wcfg.HUMAN_GOALS), "--seed", str(wcfg.ENV_SEED if seed is None else seed),
               "--compat", str(wcfg.COMPAT), "--product-file", wcfg.PRODUCT_FILE, "--attr-file", wcfg.ATTR_FILE]
        if search == "bm25":
            cmd += ["--shims", str(wcfg.SHIMS)]
        if stubs:
            cmd += ["--stubs", str(stubs)]
        env = dict(os.environ, PYTHONHASHSEED=wcfg.HASH_SEED, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="1")
        jh = java_home(self.python)
        if jh:
            env["JAVA_HOME"] = jh
        self.procs, self.logs = [], []
        for i in range(self.n):
            lf = open(log_dir / f"ws_worker_{i}.log", "a", encoding="utf-8")
            lf.write(time.strftime("\n===== %Y-%m-%d %H:%M:%S Start ") + " ".join(cmd) + "\n")
            lf.flush()
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=lf, env=env,
                                 text=True, encoding="utf-8", bufsize=1)
            self.procs.append(p)
            self.logs.append(lf)
        errors = []
        for i, p in enumerate(self.procs):
            r = self._recv(i)
            if not r.get("ok"):
                errors.append(f"WebShop Subprocess {i} Failed to start（log {log_dir / f'ws_worker_{i}.log'}）：\n{r.get('error')}")
        if errors:
            self.shutdown()
            raise RuntimeError("\n".join(errors))
        self.route = {}
        self.log_dir = log_dir
        self.search = search

    # ---------------------------------------------------------- Bottom
    def _recv(self, i):
        line = self.procs[i].stdout.readline()
        if not line:
            return {"ok": False, "error": f"Subprocess exit（exitcode={self.procs[i].poll()}），See Log"}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"unparseable response：{line[:200]}"}

    def _call(self, per_worker):
        for i, (op, payload) in per_worker.items():
            self.procs[i].stdin.write(json.dumps({"op": op, "payload": payload}, ensure_ascii=False) + "\n")
            self.procs[i].stdin.flush()
        res, errors = {}, []
        for i in per_worker:
            r = self._recv(i)
            if r.get("ok"):
                res[i] = r["result"]
            else:
                errors.append(f"WebShop Subprocess {i}：\n{r.get('error')}")
        if errors:
            raise RuntimeError("\n".join(errors))
        return res

    # ---------------------------------------------------------- Operation
    def hello(self):
        res = self._call({i: ("hello", None) for i in range(self.n)})
        return [res[i] for i in range(self.n)]

    def goal(self, idxs):
        return self._call({0: ("goal", [int(i) for i in idxs])})[0]

    def stats(self):
        res = self._call({i: ("stats", None) for i in range(self.n)})
        return [res[i] for i in range(self.n)]

    def open(self, items):
        items = sorted(items, key=lambda it: it["rid"])
        load = {i: sum(1 for w in self.route.values() if w == i) for i in range(self.n)}
        per = {}
        for it in items:                         # Each of the subprocesses with the smallest load of the current load
            w = min(load, key=lambda i: (load[i], i))
            load[w] += 1
            per.setdefault(w, []).append(it)
            self.route[it["rid"]] = w
        out = {}
        for r in self._call({w: ("open", its) for w, its in per.items()}).values():
            out.update(r)
        return out

    def step(self, items):
        per = {}
        for it in items:
            per.setdefault(self.route[it["rid"]], []).append(it)
        out = {}
        for r in self._call({w: ("step", its) for w, its in per.items()}).values():
            out.update(r)
        return out

    def close(self, rids):
        rids = [r for r in rids if r in self.route]
        per = {}
        for r in rids:
            per.setdefault(self.route[r], []).append(r)
        if per:
            self._call({w: ("close", rs) for w, rs in per.items()})
        for r in rids:
            self.route.pop(r, None)

    def probe(self, items, worker=None):
        """worker=None：Distribute to all subprocesses；Give a number：perform all operations in that worker（Validity check）。"""
        if worker is not None:
            return self._call({worker: ("probe", items)})[worker]
        size = max(1, math.ceil(len(items) / self.n))
        per = {w: ("probe", items[j:j + size]) for w, j in enumerate(range(0, len(items), size))}
        res = self._call(per)
        out = []
        for w in sorted(res):
            out += res[w]
        return out

    def shutdown(self):
        for i, p in enumerate(self.procs):
            try:
                if p.poll() is None:
                    p.stdin.write(json.dumps({"op": "exit"}) + "\n")
                    p.stdin.flush()
                    p.stdout.readline()
            except Exception:                     # noqa: BLE001
                pass
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        for lf in self.logs:
            try:
                lf.close()
            except Exception:                     # noqa: BLE001
                pass
        self.procs, self.logs = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()


def make_pool(**kw):
    return WSPool(**kw)


if __name__ == "__main__":                       # Manual check：python -m webshop.ws_pool
    with WSPool(n_workers=1) as pl:
        print(json.dumps(pl.hello(), ensure_ascii=False, indent=1))
        sys.exit(0)
