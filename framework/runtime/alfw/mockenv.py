"""Simulation of the home environment for offline testing：and twenv.AlfEnv Same interface，Simulation of text format ALFWorld。

Two categories of mandates：pick_and_place_simple（copy X Put it on Y）and pick_heat_then_place_in_recep（Well heat it up and then well put it on）。
Planner is accurate BFS（Best number of steps）。For checking process only，has no scientific interpretation。
"""
import hashlib
import random
from collections import deque

_RECEPS = ["cabinet 1", "cabinet 2", "cabinet 3", "countertop 1", "countertop 2", "drawer 1",
           "drawer 2", "fridge 1", "microwave 1", "shelf 1", "sinkbasin 1", "diningtable 1"]
_OPENABLE = {"cabinet", "drawer", "fridge", "microwave"}
_OBJECTS = ["apple", "mug", "bowl", "potato", "cup", "tomato", "egg", "bread"]


def _is_openable(r):
    return r.split(" ")[0] in _OPENABLE


class MockGame:
    def __init__(self, gamefile):
        seed = int(gamefile.split("://", 1)[1])
        rng = random.Random(seed * 7919 + 13)
        self.seed = seed
        others = [r for r in _RECEPS if r != "microwave 1"]
        self.receps = sorted(rng.sample(others, 7) + ["microwave 1"])
        objs = rng.sample(_OBJECTS, 4)
        self.objects = [f"{o} 1" for o in objs]
        self.target = self.objects[0]
        place_receps = [r for r in self.receps if r != "microwave 1"]
        self.goal = rng.choice(place_receps)
        self.init_loc = {}
        for o in self.objects:
            self.init_loc[o] = rng.choice([r for r in place_receps if r != self.goal])
        self.task_type = "pick_heat_then_place_in_recep" if seed % 3 == 0 else "pick_and_place_simple"
        name = self.target.split(" ")[0]
        goal_name = self.goal.split(" ")[0]
        self.task = (f"heat some {name} and put it in {goal_name}" if self.heat
                     else f"put some {name} in {goal_name}")

    @property
    def heat(self):
        return self.task_type == "pick_heat_then_place_in_recep"


class _State:
    __slots__ = ("at", "holding", "opened", "loc", "hot")

    def __init__(self, at, holding, opened, loc, hot):
        self.at, self.holding, self.opened, self.loc, self.hot = at, holding, opened, loc, hot

    def key(self):
        return (self.at, self.holding, tuple(sorted(self.opened)), tuple(sorted(self.loc.items())),
                tuple(sorted(self.hot)))

    def copy(self):
        return _State(self.at, self.holding, set(self.opened), dict(self.loc), set(self.hot))


class MockEnv:
    kind = "mock"

    def __init__(self):
        self.game = None
        self.gamefile = None
        self.s = None
        self.n_loads = 0

    def load(self, gamefile):
        if gamefile != self.gamefile:
            self.game = MockGame(gamefile)
            self.gamefile = gamefile
            self.n_loads += 1

    # ---------- Rules ----------
    def _contents(self, s, r):
        return sorted(o for o, where in s.loc.items() if where == r)

    def _see(self, s, r):
        if _is_openable(r) and r not in s.opened:
            return f"The {r} is closed."
        items = self._contents(s, r)
        lst = ", ".join(f"a {o}" for o in items) if items else "nothing"
        if _is_openable(r):
            return f"The {r} is open. In it, you see {lst}."
        return f"On the {r}, you see {lst}."

    def _admissible(self, s):
        g = self.game
        cmds = [f"go to {r}" for r in g.receps if r != s.at]
        cmds += ["inventory", "look"]
        if s.at is not None:
            r = s.at
            cmds.append(f"examine {r}")
            if _is_openable(r):
                cmds.append(f"close {r}" if r in s.opened else f"open {r}")
            accessible = (not _is_openable(r)) or r in s.opened
            if accessible and s.holding is None:
                cmds += [f"take {o} from {r}" for o in self._contents(s, r)]
            if accessible and s.holding is not None and r != "microwave 1":
                cmds.append(f"move {s.holding} to {r}")
            if r == "microwave 1" and s.holding is not None:
                cmds.append(f"heat {s.holding} with microwave 1")
        if s.holding is not None:
            cmds.append(f"examine {s.holding}")
        return sorted(set(cmds))

    def _apply(self, s, cmd):
        """Back (New Status, Feedback)；Return in case of irregularity (None, None)。"""
        if cmd not in self._admissible(s):
            return None, None
        t = s.copy()
        if cmd.startswith("go to "):
            t.at = cmd[6:]
            return t, f"You arrive at {t.at}. {self._see(t, t.at)}"
        if cmd.startswith("open "):
            t.opened.add(s.at)
            return t, f"You open the {s.at}. {self._see(t, s.at)}"
        if cmd.startswith("close "):
            t.opened.discard(s.at)
            return t, f"You close the {s.at}."
        if cmd.startswith("take "):
            o = cmd[5:].split(" from ")[0]
            t.holding = o
            t.loc[o] = "agent"
            return t, f"You pick up the {o} from the {s.at}."
        if cmd.startswith("move "):
            o = s.holding
            t.loc[o] = s.at
            t.holding = None
            return t, f"You move the {o} to the {s.at}."
        if cmd.startswith("heat "):
            t.hot.add(s.holding)
            return t, f"You heat the {s.holding} using the microwave 1."
        if cmd == "inventory":
            return t, (f"You are carrying: a {s.holding}." if s.holding else "You are not carrying anything.")
        if cmd == "look":
            where = s.at or "the middle of a room"
            return t, f"You are facing the {where}. Next to it, you see nothing."
        if cmd.startswith("examine "):
            x = cmd[8:]
            if x == s.holding:
                return t, (f"This is a hot {x}." if x in s.hot else f"There's nothing special about {x}.")
            return t, self._see(s, x)
        return None, None

    def _won(self, s):
        g = self.game
        ok = s.loc.get(g.target) == g.goal
        return ok and (g.target in s.hot if g.heat else True)

    def _pack(self, s, obs):
        return {"obs": obs, "admissible": self._admissible(s), "won": self._won(s)}

    # ---------- Interface ----------
    def reset(self):
        g = self.game
        self.s = _State(None, None, set(), dict(g.init_loc), set())
        seen = ", ".join(f"a {r}" for r in g.receps)
        intro = ("-= Welcome to TextWorld, ALFRED! =-\n\n"
                 f"You are in the middle of a room. Looking quickly around you, you see {seen}.\n\n"
                 f"Your task is to: {g.task}.")
        return self._pack(self.s, intro)

    def step(self, cmd):
        t, fb = self._apply(self.s, (cmd or "").strip())
        if t is None:
            return self._pack(self.s, "Nothing happens.")
        self.s = t
        return self._pack(self.s, fb)

    def plan(self):
        """BFS Shortest command sequence（Best number of steps）。"""
        if self._won(self.s):
            return []
        start = self.s
        q = deque([start])
        parent = {start.key(): None}
        while q:
            s = q.popleft()
            for c in self._admissible(s):
                if c in ("look", "inventory") or c.startswith("examine") or c.startswith("close"):
                    continue
                t, _ = self._apply(s, c)
                k = t.key()
                if k in parent:
                    continue
                parent[k] = (s.key(), c)
                if self._won(t):
                    path = []
                    while parent[k] is not None:
                        pk, pc = parent[k]
                        path.append(pc)
                        k = pk
                    return path[::-1]
                q.append(t)
        return []

    def state_key(self):
        return hashlib.sha1(repr(self.s.key()).encode()).hexdigest()[:16]
