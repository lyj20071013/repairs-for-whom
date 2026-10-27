"""WebShop Environment subprocess（at WebShop its own Python run in the environment，and vLLM Isolation in the environment）。

Main Process（webshop/ws_pool.py）use subprocess Start it，Send and receive by line JSON：
    Request {"op": ..., "payload": ...}  ->  Response {"ok": true, "result": ...} or {"ok": false, "error": "..."}
Protocol only uses standard input / Output Original File Descriptor；WebShop Change all printing at initialization to standard error（Main process in log file）。

Use only WebShop Original Code（princeton-nlp/WebShop，commit 64fa2a5，Unchanged）Yeah SimServer and WebAgentTextEnv：
- One for all sessions SimServer（Commodities、Retrieval Index、Target list loaded only once）；Every one rollout One WebAgentTextEnv，
  Session number = "<rid>|<Target Number>"，Targets are numbered（Not random）。
- SimServer Use global when initializing random Price ceilings for prices and targets generated for inter-pricing commodities（the original varies between runs）；
  fix this before initialization random.seed(--seed)，So all subprocesses、All running target lists are word by word（hello Go back to Hashi to check it out）。
- Executable Actions = WebAgentTextEnv.get_available_actions()：When there is a search box search[Keywords]，and all clickable entries on the page
  （Button、Commodity number、Options）click[...]。Actions not in the list do not change state、Expend one step（Original environment allowed on any page search，
  This is the same as the list，Not implemented）。
- Success = Points Buy Now the resulting reward is 1.0（WebShop Standards“success rate”）；Keeping trajectories of the reward。
- Commodities / Attribute file by --product-file / --attr-file Assign（10 Ten thousand Configurations，See ws_data.py；Default with original when not giving 1000 Documentation）。
- hello Go back to the fingerprint search（before 10 Query for a targets command、before 50 Hashi the number and fraction of the result），Checks the same search for two servers。
"""
import argparse
import glob
import hashlib
import importlib.metadata as md
import json
import os
import random
import re
import sys
import time
import traceback

BASE_URL = "http://127.0.0.1:3000"
SEARCH_TEMPLATE = "search[<keywords>]"


def _h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16]


def instruction_of(env):
    """Command Text on Page（Get rid of the beginning "Instruction:"）。"""
    return re.sub(r"^\s*Instruction:\s*", "", env.instruction_text or "").strip()


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


class Worker:
    def __init__(self, a):
        self.a = a
        sys.path.insert(0, a.webshop_dir)
        for d in (a.compat, a.shims, a.stubs):
            if d:
                sys.path.insert(0, d)
        random.seed(a.seed)
        t0 = time.time()
        from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv
        from web_agent_site.utils import DEFAULT_FILE_PATH
        file_path = DEFAULT_FILE_PATH
        if a.product_file:
            import ws_data                     # Same table of contents as this document
            file_path, _ = ws_data.apply(os.path.abspath(a.webshop_dir), a.product_file, a.attr_file)
        self.file_path = file_path
        self.WebAgentTextEnv = WebAgentTextEnv
        self.server = SimServer(BASE_URL, file_path, filter_goals=None, limit_goals=-1,
                                num_products=a.num_products, human_goals=bool(a.human_goals), show_attrs=False)
        self.init_s = time.time() - t0
        self.envs = {}

    # ---------------------------------------------------------- state
    def _page(self, env):
        return self.server.get_page_name(env.browser.current_url) or "index"

    def _key(self, env):
        url = env.browser.current_url or ""
        return url.replace(env.session, "*", 1) if env.session else url

    def state(self, env, done=False, reward=0.0):
        avail = env.get_available_actions()
        clicks = [c for c in avail["clickables"] if c != "search"]
        st = {"obs": env.observation, "search": bool(avail["has_search_bar"]) and not done,
              "clicks": [] if done else clicks, "done": bool(done), "reward": float(reward),
              "won": bool(done and reward >= 1.0 - 1e-9), "page": self._page(env), "key": self._key(env)}
        st["hash"] = _h([st["obs"], st["search"], st["clicks"], st["done"], st["reward"], st["key"]])
        return st

    @staticmethod
    def parse(cmd):
        """'search[...]' / 'click[...]' -> (name, parameters)；other -> (None, None)。"""
        m = re.fullmatch(r"\s*(search|click)\s*\[(.*)\]\s*", cmd or "", re.S | re.I)
        if not m:
            return None, None
        return m.group(1).lower(), m.group(2).strip()

    def _do(self, rec, cmd):
        env, st = rec["env"], rec["state"]
        name, arg = self.parse(cmd)
        valid = False
        if not st["done"]:
            # The original version of the trial special query（<r> Randomly sampled goods、<a> <c> <q>）Do not execute query with copying placeholder，Guarantee replayable
            special = bool(arg) and (arg.split()[0].lower() in ("<r>", "<a>", "<c>", "<q>") or arg == "<keywords>")
            if name == "search" and st["search"] and arg and not special:
                valid = True
            elif name == "click" and arg is not None and arg.lower() in st["clicks"]:
                valid = True
        if valid:
            _, reward, done, _ = env.step(f"{name}[{arg}]")
            rec["state"] = self.state(env, done=done, reward=reward)
        else:                                   # Not in executable list：Do not change state，Only one step
            rec["state"] = dict(st)
        return rec["state"], valid

    # ---------------------------------------------------------- Operation
    def search_fingerprint(self, n=10, k=50):
        """before n Query for a targets command，Prior k Numbering and fractions of the result of the article（4 Bit Decimal）Please, Hashi：Are the two servers searched the same way?。"""
        qs = [re.sub(r"\s+", " ", g["instruction_text"].lower()).strip() for g in self.server.goals[:n]]
        res = []
        for q in qs:
            hits = self.server.search_engine.search(q, k=k)
            res.append([[h.docid, round(float(h.score), 4)] for h in hits])
        return {"hash": _h([qs, res]), "queries": len(qs), "hits": sum(len(r) for r in res)}

    def op_hello(self, _):
        s = self.server
        goals = [[g["asin"], g["instruction_text"], g.get("price_upper"), sorted(map(str, g.get("goal_options") or []))]
                 for g in s.goals]
        vers = {}
        for name in ("flask", "beautifulsoup4", "spacy", "en_core_web_sm", "thefuzz", "pyserini", "pyjnius", "torch",
                     "numpy"):
            try:
                vers[name] = md.version(name)
            except md.PackageNotFoundError:
                vers[name] = "missing"
        vers["gym"] = vers["selenium"] = "compat（Minimal replacement）"
        try:
            import pyserini
            jars = glob.glob(os.path.join(os.path.dirname(pyserini.__file__), "resources", "jars", "anserini-*-fatjar.jar"))
            vers["anserini_jar"] = os.path.basename(sorted(jars)[-1]) if jars else "?"
        except Exception:                           # noqa: BLE001
            vers["anserini_jar"] = "?"
        search = type(s.search_engine).__module__ + " " + str(vers.get("pyserini"))
        return {"pid": os.getpid(), "n_goals": len(s.goals), "goals_hash": _h(goals), "n_products": len(s.all_products),
                "products_hash": _h(sorted([p["asin"], s.product_prices.get(p["asin"])] for p in s.all_products)),
                "search_fp": self.search_fingerprint(), "product_file": os.path.basename(self.file_path),
                "search_backend": search, "python": sys.version.split()[0], "versions": vers,
                "java_home": os.environ.get("JAVA_HOME"), "rss_mb": _rss_mb(),
                "init_s": round(self.init_s, 2), "seed": self.a.seed, "mode": self.a.mode,
                "num_products": self.a.num_products, "human_goals": self.a.human_goals}

    def op_goal(self, idxs):
        """Complete information about the target（privileged information：Target commodities and options；For analysis only，No smart body）。"""
        out = {}
        for i in idxs:
            g = self.server.goals[int(i)]
            out[str(i)] = {"asin": g["asin"], "instruction": g["instruction_text"], "price_upper": g.get("price_upper"),
                           "attributes": g.get("attributes"), "goal_options": g.get("goal_options"),
                           "query": g.get("query"), "category": g.get("category")}
        return out

    def _new_env(self, rid, goal):
        env = self.WebAgentTextEnv(observation_mode=self.a.mode, server=self.server, session_prefix=f"{rid}|")
        self.server.user_sessions.pop(env.session, None)          # Construct entries left over from opportunism
        env.reset(session=int(goal))
        return env

    def op_open(self, items):
        out = {}
        for it in items:
            t0 = time.perf_counter()
            env = self._new_env(it["rid"], it["goal"])
            rec = {"env": env, "state": None}
            rec["state"] = self.state(env)
            init = dict(rec["state"])
            pre = []
            for c in it.get("prefix", []):
                st, valid = self._do(rec, c)
                pre.append({"state": st, "valid": valid})
            self.envs[it["rid"]] = rec
            out[it["rid"]] = {"instruction": instruction_of(env), "init": init, "prefix": pre,
                              "state": rec["state"], "t": time.perf_counter() - t0}
        return out

    def op_step(self, items):
        out = {}
        for it in items:
            t0 = time.perf_counter()
            st, valid = self._do(self.envs[it["rid"]], it["cmd"])
            out[it["rid"]] = {"state": st, "valid": valid, "dt": time.perf_counter() - t0}
        return out

    def op_close(self, rids):
        for rid in rids:
            rec = self.envs.pop(rid, None)
            if rec is not None:
                self.server.user_sessions.pop(rec["env"].session, None)
        return len(rids)

    def privileged(self, env, goal):
        """Next steps in known target commodities and options（Prescription reference，Corresponds to ALFWorld Planner；analysis plan.md 6.13 Official experiments）。
        Search Page：Search with the title of the target commodity；Search Results Page：The target is on this page，otherwise back to search；Target commodity page：The first one hasnt been chosen yet
        Target Options，Weve got a pick buy now；Other commodity and subpages：< prev。Back None That means no（for example, already terminated）。"""
        st = self.state(env)
        if st["done"]:
            return None
        g = self.server.goals[int(goal)]
        tasin = str(g["asin"]).lower()
        clicks = st["clicks"]
        if st["search"]:
            title = re.sub(r"\s+", " ", str(self.server.product_item_dict[g["asin"]]["Title"]).lower()).strip()
            return f"search[{title}]" if title else None
        page = st["page"]
        sess = self.server.user_sessions.get(env.session) or {}
        if page == "search_results":
            return f"click[{tasin}]" if tasin in clicks else ("click[back to search]" if "back to search" in clicks else None)
        if page == "item_page":
            if str(sess.get("asin") or "").lower() == tasin:
                chosen = {str(v).strip().lower() for v in (sess.get("options") or {}).values()}
                for o in g.get("goal_options") or []:
                    o = str(o).strip().replace("/", " | ").lower()
                    if o in clicks and o not in chosen:
                        return f"click[{o}]"
                return "click[buy now]" if "buy now" in clicks else None
            return "click[< prev]" if "< prev" in clicks else None
        return "click[< prev]" if "< prev" in clicks else None

    def op_probe(self, items):
        """[{goal, prefix}] -> Replay the prefix status and step by step Hash（Do not keep session）；Take it "priv": true add a privileged action when。"""
        out = []
        for j, it in enumerate(items):
            rid = f"probe{os.getpid()}_{j}_{time.monotonic_ns()}"
            env = self._new_env(rid, it["goal"])
            rec = {"env": env, "state": None}
            rec["state"] = self.state(env)
            hashes = [rec["state"]["hash"]]
            valids = []
            for c in it.get("prefix", []):
                st, valid = self._do(rec, c)
                hashes.append(st["hash"])
                valids.append(valid)
            o = {"state": rec["state"], "hashes": hashes, "valid": valids, "instruction": instruction_of(env)}
            if it.get("priv"):
                o["priv"] = self.privileged(env, it["goal"])
            out.append(o)
            self.server.user_sessions.pop(env.session, None)
        return out

    def op_stats(self, _):
        return {"pid": os.getpid(), "active": len(self.envs), "sessions": len(self.server.user_sessions),
                "rss_mb": _rss_mb(), "search_time": round(self.server.search_time, 3),
                "render_time": round(self.server.render_time, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webshop-dir", required=True)
    ap.add_argument("--mode", default="text_rich")
    ap.add_argument("--num-products", type=int, default=1000)
    ap.add_argument("--product-file", default=None, help="data/ Under Commodity Document Name（10 Ten thousand Configurations）；Default with original when not giving")
    ap.add_argument("--attr-file", default=None, help="data/ Properties File Name")
    ap.add_argument("--human-goals", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shims", default=None, help="Retrieval backend replacement realization（pyserini use if installation fails BM25）")
    ap.add_argument("--compat", default=None, help="gym / selenium Minimum replacement for")
    ap.add_argument("--stubs", default=None, help="For use only Claude offline workspace self-tests")
    a = ap.parse_args()
    if a.num_products <= 0:
        a.num_products = None
    proto_in = sys.stdin
    proto_out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)                         # And then everything thats written into a standard output is standard wrong（log），It wont mix with the deal
    sys.stdout = sys.stderr
    os.chdir(a.webshop_dir)

    def send(obj):
        proto_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto_out.flush()
    try:
        w = Worker(a)
        send({"ok": True, "result": "ready"})
    except Exception:                     # noqa: BLE001
        send({"ok": False, "error": traceback.format_exc()})
        return 1
    for line in proto_in:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            send({"ok": False, "error": "bad json"})
            continue
        if req.get("op") == "exit":
            send({"ok": True, "result": None})
            return 0
        try:
            send({"ok": True, "result": getattr(w, "op_" + req["op"])(req.get("payload"))})
        except Exception:                 # noqa: BLE001
            send({"ok": False, "error": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
