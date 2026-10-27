"""Environment pool：Main process only 7B Generate，Environment advancing in the sub-process（TextWorld To generate text for all executable actions per step，CPU high overhead）。

Two realizations，Same interface：
    InlinePool(kind)            Same process，For testing and small-scale inspection
    ProcPool(kind, n_workers)   spawn Subprocess（not fork：the main process contains CUDA / vLLM）

Operation：
    open([{rid, gamefile, prefix, want_key}])  -> {rid: {init_obs, prefix_obs, prefix_won, state, key, t}}
    step([{rid, cmd, want_key}])              -> {rid: {state, key, dt}}
    plan([rid])                               -> {rid: {plan, dt}}
    close([rid])
    probe([{gamefile, prefix}])               -> [ {plan, dt_plan, dt_replay, won} ]（Order and Enter）
    stats()                                   -> Number of environmental examples of sub-processes、Loads、Memory
Every one rollout Taking an environmental example；Go back to the free list after the end of the case，Next priority is assigned to the same game（Dont reload）。
"""
import math
import re
import multiprocessing as mp
import os
import time
import traceback

from . import twenv


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _plan_with_timeout(env, timeout):
    """at fork Call the planner in the sub-process，terminate the worker on timeout。Back (Planned or None, Timeout)。

    fast-downward search runs in C Intruder，Python cannot be interrupted by this signal；fork Copy Current Environment（Current status including the planner library），
    Once the sub-process has been planned, write the results back to the pipe。Use only in environmental subprocesses（these processes do not have CUDA、No other threads）。
    """
    import json
    import select
    import signal
    if not timeout or not hasattr(os, "fork"):
        return env.plan(), False
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                  # Subprocess
        code = 0
        try:
            os.close(r)
            data = json.dumps(env.plan()).encode()
            with os.fdopen(w, "wb") as f:
                f.write(data)
        except BaseException:
            code = 1
        finally:
            os._exit(code)
    os.close(w)
    deadline = time.monotonic() + timeout
    chunks, timed_out = [], False
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([r], [], [], left)
            if ready:
                chunk = os.read(r, 1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
    finally:
        os.close(r)
        if timed_out:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _, status = os.waitpid(pid, 0)
    if timed_out:
        return None, True
    if status != 0 or not chunks:
        raise RuntimeError(f"The planning subprocess exits abnormally（status={status}）")
    return json.loads(b"".join(chunks)), False


_SEE_RE = re.compile(r"Looking quickly around you, you see (.+?)\.\s*(?:\n|$)", re.S)


def parse_receptacles(text):
    """All container names listed in the opening description（"a desk 1, a drawer 2, and a shelf 1" -> [desk 1, drawer 2, shelf 1]）。"""
    m = _SEE_RE.search(text or "")
    if not m:
        return []
    out = []
    for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1)):
        x = re.sub(r"^(?:a|an)\s+", "", x.strip())
        if x:
            out.append(x)
    return out


class _Worker:
    def __init__(self, kind, patches=None):
        self.kind = kind
        self.patches = []
        self.tmpdir = twenv.use_data_disk_tmp() if kind == "tw" else None
        if kind == "tw" and patches:
            self.patches = twenv.install_patches(**patches)
        self.free = {}
        self.active = {}
        self.envs = []

    def _acquire(self, gamefile):
        lst = self.free.get(gamefile)
        if lst:
            return lst.pop()
        for lst in self.free.values():
            if lst:
                return lst.pop()
        env = twenv.make_env(self.kind)
        self.envs.append(env)
        return env

    def _release(self, env):
        self.free.setdefault(env.gamefile, []).append(env)

    def op_open(self, items):
        out = {}
        for it in items:
            t0 = time.perf_counter()
            env = self._acquire(it["gamefile"])
            env.load(it["gamefile"])
            st = env.reset()
            init = st["obs"]
            pobs, pwon = [], []
            for c in it.get("prefix", []):
                st = env.step(c)
                pobs.append(st["obs"])
                pwon.append(st["won"])
            self.active[it["rid"]] = env
            out[it["rid"]] = {"init_obs": init, "prefix_obs": pobs, "prefix_won": pwon, "state": st,
                              "key": env.state_key() if it.get("want_key") else None,
                              "t": time.perf_counter() - t0}
        return out

    def op_step(self, items):
        out = {}
        for it in items:
            env = self.active[it["rid"]]
            t0 = time.perf_counter()
            st = env.step(it["cmd"])
            out[it["rid"]] = {"state": st, "key": env.state_key() if it.get("want_key") else None,
                              "dt": time.perf_counter() - t0}
        return out

    def op_plan(self, rids):
        out = {}
        for rid in rids:
            t0 = time.perf_counter()
            p = self.active[rid].plan()
            out[rid] = {"plan": p, "dt": time.perf_counter() - t0}
        return out

    def op_close(self, rids):
        for rid in rids:
            env = self.active.pop(rid, None)
            if env is not None:
                self._release(env)
        return len(rids)

    def op_probe(self, items):
        out = []
        for it in items:
            env = self._acquire(it["gamefile"])
            t0 = time.perf_counter()
            env.load(it["gamefile"])
            st = env.reset()
            for c in it.get("prefix", []):
                st = env.step(c)
            t1 = time.perf_counter()
            adm_before = list(st["admissible"])
            timed_out = False
            if st["won"]:
                p = []
            elif it.get("timeout"):
                p, timed_out = _plan_with_timeout(env, it["timeout"])
            else:
                p = env.plan()
            t2 = time.perf_counter()
            r = {"plan": p, "dt_plan": t2 - t1, "dt_replay": t1 - t0, "won": st["won"], "timeout": timed_out}
            if it.get("want_adm"):
                r["admissible"] = adm_before
            if it.get("verify") and p is not None:   # Implementation as planned：every command is executable before execution，Completion of implementation to achieve objectives
                adm_ok = True
                for c in p:
                    adm_ok = adm_ok and c in st["admissible"]
                    st = env.step(c)
                r["admissible_ok"], r["reaches_goal"] = adm_ok, bool(st["won"])
            self._release(env)
            out.append(r)
        return out

    def op_survey(self, items):
        """[{gamefile}] -> the contents of each container at the start：One by one go to，open closed containers first open，Record raw feedback from the environment。
        Only for giving“informed judge”Provide hidden information；reset for each episode reset，Does not affect other operations。
        Some containers are in the same position as the previous container（adjacent drawers、Ill wait for her），at this point "go to" not executable：Go around the other container and come back。"""
        out = []
        for it in items:
            env = self._acquire(it["gamefile"])
            env.load(it["gamefile"])
            st = env.reset()
            receps = parse_receptacles(st["obs"])
            rows = []
            for r in receps:
                detour = None
                if f"go to {r}" not in st["admissible"]:
                    kind = r.rsplit(" ", 1)[0]
                    detour = next((x for x in receps if x.rsplit(" ", 1)[0] != kind
                                   and f"go to {x}" in st["admissible"]), None)
                    if detour is not None:
                        st = env.step(f"go to {detour}")
                st = env.step(f"go to {r}")
                o1, o2 = st["obs"], None
                if f"{r} is closed" in o1:
                    st = env.step(f"open {r}")
                    o2 = st["obs"]
                rows.append({"name": r, "closed": o2 is not None, "obs": o1, "open_obs": o2, "detour": detour,
                             "ok": o1.startswith(f"You arrive at {r}")})
            self._release(env)
            out.append({"gamefile": it["gamefile"], "receptacles": rows})
        return out

    def op_replay(self, items):
        """[{gamefile, cmds, plan_at}] -> Gradual Fingerprints with plan_at Planning of the Office（Test call planner for no side effects）。"""
        out = []
        for it in items:
            env = self._acquire(it["gamefile"])
            r = twenv.replay(env, it["gamefile"], it["cmds"], plan_at=set(it.get("plan_at", ())))
            self._release(env)
            out.append({"hashes": r["hashes"], "plans": r["plans"]})
        return out

    def op_stats(self, _):
        return {"pid": os.getpid(), "n_envs": len(self.envs), "n_loads": sum(e.n_loads for e in self.envs),
                "tmpdir": self.tmpdir,
                "active": len(self.active), "rss_mb": _rss_mb(), "patches": self.patches,
                "hashseed": os.environ.get("PYTHONHASHSEED")}

    def handle(self, op, payload):
        return getattr(self, "op_" + op)(payload)


def _worker_main(conn, kind, patches):
    try:
        w = _Worker(kind, patches)
        conn.send(("ready", os.getpid()))
    except Exception:
        conn.send(("__error__", traceback.format_exc()))
        return
    while True:
        try:
            op, payload = conn.recv()
        except EOFError:
            return
        if op == "exit":
            conn.send(("ok", None))
            return
        try:
            conn.send(("ok", w.handle(op, payload)))
        except Exception:
            conn.send(("__error__", traceback.format_exc()))


class _PoolBase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()


class InlinePool(_PoolBase):
    def __init__(self, kind, patches=None):
        self.w = _Worker(kind, patches)
        self.n = 1

    def open(self, items):
        return self.w.op_open(items)

    def step(self, items):
        return self.w.op_step(items)

    def plan(self, rids):
        return self.w.op_plan(rids)

    def close(self, rids):
        self.w.op_close(list(rids))

    def probe(self, items):
        return self.w.op_probe(items)

    def replay(self, items):
        return self.w.op_replay(items)

    def survey(self, items):
        return self.w.op_survey(items)

    def stats(self):
        return [self.w.op_stats(None)]

    def shutdown(self):
        pass


class ProcPool(_PoolBase):
    def __init__(self, kind, n_workers, patches=None):
        ctx = mp.get_context("spawn")
        self.n = n_workers
        self.conns, self.procs = [], []
        for _ in range(n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_main, args=(child, kind, patches), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        errors = []
        for i, c in enumerate(self.conns):
            try:
                tag, val = c.recv()
            except EOFError:
                tag, val = "__error__", f"Process exit（exitcode={self.procs[i].exitcode}），Look up stderr"
            if tag != "ready":
                errors.append(f"Subprocess {i} Failed to start：\n{val}")
        if errors:
            self.shutdown()
            raise RuntimeError("\n".join(errors))
        self.route = {}

    def _call(self, per_worker):
        for i, (op, payload) in per_worker.items():
            self.conns[i].send((op, payload))
        res, errors = {}, []
        for i in per_worker:
            try:
                tag, val = self.conns[i].recv()
            except EOFError:
                errors.append(f"Environment subprocess {i} Unusual exit（exitcode={self.procs[i].exitcode}）")
                continue
            if tag == "__error__":
                errors.append(f"Environment subprocess {i}：\n{val}")
            else:
                res[i] = val
        if errors:
            raise RuntimeError("\n".join(errors))
        return res

    def open(self, items):
        items = sorted(items, key=lambda it: (it["gamefile"], it["rid"]))
        size = max(1, math.ceil(len(items) / self.n))
        load = {i: sum(1 for w in self.route.values() if w == i) for i in range(self.n)}
        order = sorted(range(self.n), key=lambda i: load[i])
        per = {}
        for j in range(0, len(items), size):
            w = order[(j // size) % self.n]
            per.setdefault(w, []).extend(items[j:j + size])
        for w, its in per.items():
            for it in its:
                self.route[it["rid"]] = w
        out = {}
        for r in self._call({w: ("open", its) for w, its in per.items()}).values():
            out.update(r)
        return out

    def _by_route(self, keys, getter):
        per = {}
        for k in keys:
            per.setdefault(self.route[getter(k)], []).append(k)
        return per

    def step(self, items):
        per = self._by_route(items, lambda it: it["rid"])
        out = {}
        for r in self._call({w: ("step", its) for w, its in per.items()}).values():
            out.update(r)
        return out

    def plan(self, rids):
        per = self._by_route(rids, lambda r: r)
        out = {}
        for r in self._call({w: ("plan", rs) for w, rs in per.items()}).values():
            out.update(r)
        return out

    def close(self, rids):
        rids = [r for r in rids if r in self.route]
        per = self._by_route(rids, lambda r: r)
        self._call({w: ("close", rs) for w, rs in per.items()})
        for r in rids:
            self.route.pop(r, None)

    def probe(self, items):
        return self._map("probe", items)

    def replay(self, items):
        return self._map("replay", items)

    def survey(self, items):
        return self._map("survey", items)

    def _map(self, op, items):
        groups = {}
        for idx, it in enumerate(items):
            groups.setdefault(it["gamefile"], []).append(idx)
        # The same game is given to the same process as much as possible（Reuse Loaded Environments），But not more than the average share per block，Well put it in a box by the size of it
        # It takes a lot of time to plan（Isolated close 100 seconds），Up each piece 2 items，Avoiding all detection of a slow game on the same sub-process
        cap = max(1, min(2, math.ceil(len(items) / self.n)))
        chunks = [idxs[j:j + cap] for idxs in groups.values() for j in range(0, len(idxs), cap)]
        bins = {i: [] for i in range(self.n)}
        sizes = {i: 0 for i in range(self.n)}
        for idxs in sorted(chunks, key=len, reverse=True):
            w = min(sizes, key=sizes.get)
            bins[w].extend(idxs)
            sizes[w] += len(idxs)
        per = {w: (op, [items[i] for i in idxs]) for w, idxs in bins.items() if idxs}
        res = self._call(per)
        out = [None] * len(items)
        for w, idxs in bins.items():
            for i, r in zip(idxs, res.get(w, [])):
                out[i] = r
        return out

    def stats(self):
        res = self._call({i: ("stats", None) for i in range(self.n)})
        return [res[i] for i in range(self.n)]

    def shutdown(self):
        for c, p in zip(self.conns, self.procs):
            try:
                if p.is_alive():
                    c.send(("exit", None))
                    c.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.conns, self.procs = [], []


def make_pool(kind, n_workers, patches=None):
    if n_workers <= 0:
        return InlinePool(kind, patches)
    return ProcPool(kind, n_workers, patches)
