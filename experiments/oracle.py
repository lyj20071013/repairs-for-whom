"""Environmentally accessible oracle：Precision、Quick，Do not rely on models。

One-time construction of each map“Physical status map”，Reverse from successful export BFS，Get every physical state
Minimum number of actions required to succeed dist。therefore

    F(h) = 1  if and only if  h Success，or h Unfinished and dist(h) <= Remaining Step(s)

Move All Call Real GridWorld.step，No more rules；and diagnose.shortest_solution
Consistency by selftest Status-by-State check。

Use：
1. unsolvable within budget（F=0）the branch of the，Any strategy success rate is just as good as it looks 0，no sampling needed；
2. “oracle continuation”Condition：One by one 7B Much stronger controlled evaluators；
3. “Noise oracle”Condition：Yes CPU Running Random Appraiser，also the mock policy for offline tests。
"""
from collections import deque

from env import GridWorld

# and diagnose.SEARCH_ACTIONS identical；look / give_up cannot reduce the distance to success。
SEARCH_ACTIONS = [{"tool": "move", "arg": d} for d in ("up", "down", "left", "right")] + [
    {"tool": "pick"}, {"tool": "open"}]

_SUCCESS, _FAIL = "SUCCESS", "FAIL"
_BIG_BUDGET = 10 ** 9


def physical_key(snap):
    """and diagnose.physical_key identical：Transfers rely only on these fields（Budget is different）。"""
    return (tuple(snap["pos"]), snap["has_key"], snap["has_gem"], snap["door_open"],
            tuple(tuple(row) for row in snap["layout"]))


def _env_at(template, key):
    """By template Bottom，Replace the physical state with key；Zero steps、The budget is huge，Avoid time-consuming interference with the transfer。"""
    env = GridWorld.restore(template)
    pos, has_key, has_gem, door_open, layout = key
    env.pos = tuple(pos)
    env.has_key, env.has_gem, env.door_open = has_key, has_gem, door_open
    env.layout = [list(row) for row in layout]
    env.steps, env.max_steps = 0, _BIG_BUDGET
    env.done, env.success, env.fail_reason = False, False, None
    env.action_log, env.history = [], []
    return env


class DistanceTable:
    """A map（rooted at the initial snapshot）Shortest range of success in all available physical state。"""

    def __init__(self, root_snapshot):
        self.template = root_snapshot
        root = physical_key(root_snapshot)
        self.edges = {}          # key -> [(action_index, next_key | SUCCESS | FAIL)]
        queue, seen = deque([root]), {root}
        while queue:
            key = queue.popleft()
            out = []
            for i, action in enumerate(SEARCH_ACTIONS):
                env = _env_at(root_snapshot, key)
                env.step(action)
                if env.success:
                    out.append((i, _SUCCESS))
                elif env.done:
                    out.append((i, _FAIL))
                else:
                    nxt = physical_key(env.snapshot())
                    out.append((i, nxt))
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            self.edges[key] = out
        # Reverse BFS：Unit boundaries，Expanding by incremental distance is the shortest distance。
        preds = {}
        self.dist = {}
        frontier = []
        for key, out in self.edges.items():
            for _, nxt in out:
                if nxt == _SUCCESS:
                    if key not in self.dist:
                        self.dist[key] = 1
                        frontier.append(key)
                elif nxt != _FAIL:
                    preds.setdefault(nxt, []).append(key)
        queue = deque(frontier)
        while queue:
            key = queue.popleft()
            for p in preds.get(key, ()):
                if p not in self.dist:
                    self.dist[p] = self.dist[key] + 1
                    queue.append(p)

    def distance(self, snap):
        """Minimum number of actions successful；return when success is impossible None。Successfully returned 0。"""
        if snap["done"]:
            return 0 if snap["success"] else None
        key = physical_key(snap)
        if key not in self.edges:
            raise KeyError("snapshot Not in the maps reachable status map")
        return self.dist.get(key)

    def feasible(self, snap):
        """F(h)：Whether successful action sequences exist within the remaining budget。"""
        if snap["done"]:
            return bool(snap["success"])
        d = self.distance(snap)
        return d is not None and d <= snap["max_steps"] - snap["steps"]

    def optimal_actions(self, snap):
        """Ambassador dist Minimum All Actions（Ignore budget）；Returns if no path exists []。"""
        if snap["done"]:
            return []
        key = physical_key(snap)
        best, acts = None, []
        for i, nxt in self.edges[key]:
            if nxt == _SUCCESS:
                d = 1
            elif nxt == _FAIL:
                continue
            else:
                sub = self.dist.get(nxt)
                if sub is None:
                    continue
                d = 1 + sub
            if best is None or d < best:
                best, acts = d, [SEARCH_ACTIONS[i]]
            elif d == best:
                acts.append(SEARCH_ACTIONS[i])
        return acts


_TABLES = {}


def table_for(snap):
    """Cache by Initial Map。All states of the same game can be reached from the beginning。"""
    key = tuple("".join(row) for row in snap["initial_layout"])
    tab = _TABLES.get(key)
    if tab is None:
        root = GridWorld(snap["initial_layout"], max_steps=snap["max_steps"], seed=snap["seed"]).snapshot()
        tab = DistanceTable(root)
        _TABLES[key] = tab
    return tab


def feasible(snap):
    return table_for(snap).feasible(snap)
