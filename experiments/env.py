"""
Text gridworld。Design Constraint：
1. The state can be pure JSON A snapshot/restore，No hidden RNG state
2. Each failed type has a definite end mark，You can split it without loss
3. look() exists but is uninformative（The map is completely visible），used for batch generation ramp Type trajectories

Topography Symbols：. Empty land  # Wall  T A trap  D The door（locked）  k Keys  g Gems  A agent
"""
from copy import deepcopy
from collections import deque

WALL, EMPTY, TRAP, DOOR, KEY, GEM = "#", ".", "T", "D", "k", "g"
START = (0, 0)

# Actionset
ACTIONS = ["move", "open", "pick", "look", "give_up"]
# Observation and transfer share the same chart of directions，Avoiding inconsistencies between description and actual implementation
MOVE_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _in_bounds(pos, n):
    return 0 <= pos[0] < n and 0 <= pos[1] < n


def _dist(a, b):
    """Chelseas distance：Neighborhood（Italic Angle）is 1"""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class GridWorld:
    def __init__(self, layout, max_steps=24, seed=0):
        """layout: 2D Character Matrix（Does not include agent）"""
        self.n = len(layout)
        self.initial_layout = [list(row) for row in layout]
        self.max_steps = max_steps
        self.seed = seed
        self.reset()

    # ---------- Construct ----------
    @classmethod
    def random_layout(cls, n=6, n_traps=2, n_walls=4, seed=0):
        """
        Stabilised bone + Random Details：
        - The doors always there (n-1, n-2)，the gem is at (n-1, n-1)（Behind the door）
        - Key Location Random，But not in the same corner of the door
        - A trap/Wall randomly placed，must not block all paths，must not be placed at the starting position/Key Position
        """
        import random
        if n < 3 or min(n_traps, n_walls) < 0:
            raise ValueError("Need n >= 3 and nonnegative obstacle counts")
        rng = random.Random(seed)
        g = [[EMPTY for _ in range(n)] for _ in range(n)]
        door_pos = (n - 1, n - 2)
        gem_pos = (n - 1, n - 1)
        g[door_pos[0]][door_pos[1]] = DOOR
        g[gem_pos[0]][gem_pos[1]] = GEM
        # the door is the gems only entrance；This fixed wall doesnt count as random n_walls。
        gate_wall = (n - 2, n - 1)
        g[gate_wall[0]][gate_wall[1]] = WALL

        # Can not open message：Not at the beginning、not on the ground/Gems、not directly in front of the door
        reserved = {START, door_pos, gem_pos, gate_wall, (n - 2, n - 2)}
        cells = [(r, c) for r in range(n) for c in range(n) if (r, c) not in reserved]
        if len(cells) < 1 + n_walls + n_traps:
            raise ValueError("Not enough cells for key, random walls and traps")
        rng.shuffle(cells)

        key_pos = cells.pop()
        g[key_pos[0]][key_pos[1]] = KEY

        for _ in range(n_walls):
            if not cells:
                break
            r, c = cells.pop()
            g[r][c] = WALL
        for _ in range(n_traps):
            if not cells:
                break
            r, c = cells.pop()
            g[r][c] = TRAP
        return g

    # ---------- Accomplishable verification ----------
    @staticmethod
    def _bfs_reachable(grid, src, dst, pass_door=True):
        """
        Flood filling。trap cells do not count as traversable paths（agent does not intentionally enter traps）。
        used only to exclude completely blocked maps——otherwise V(h_0) Always 0，Contamination threshold calibration。
        """
        n = len(grid)
        if src == dst:
            return True
        seen = {src}
        queue = deque([src])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if not _in_bounds(nxt, n) or nxt in seen:
                    continue
                cell = grid[nxt[0]][nxt[1]]
                if cell == WALL:
                    continue
                if cell == DOOR and not pass_door:
                    continue
                if cell == TRAP:
                    continue
                if nxt == dst:
                    return True
                seen.add(nxt)
                queue.append(nxt)
        return False

    @classmethod
    def random_layout_checked(cls, n=6, n_traps=2, n_walls=4, seed=0, max_tries=200):
        """
        Layout generation with reachability retry。Must be satisfied：
          1. The starting point is the key（Go around the door）
          2. Keys can reach the door
          3. You can reach the gems behind the door
        generate another if the criteria fail seed Again。
        """
        door_pos = (n - 1, n - 2)
        gem_pos = (n - 1, n - 1)
        for attempt in range(max_tries):
            g = cls.random_layout(n=n, n_traps=n_traps, n_walls=n_walls,
                                  seed=seed * 10007 + attempt)
            key_pos = None
            for r in range(n):
                for c in range(n):
                    if g[r][c] == KEY:
                        key_pos = (r, c)
            if key_pos is None:
                continue
            if not cls._bfs_reachable(g, START, key_pos, pass_door=False):
                continue
            if not cls._bfs_reachable(g, key_pos, (n - 2, n - 2), pass_door=False):
                continue
            if not cls._bfs_reachable(g, door_pos, gem_pos, pass_door=True):
                continue
            return g
        raise RuntimeError(
            f"layout generation failed after {max_tries} tries; copy n_traps/n_walls Scale"
        )

    # ---------- Life cycle ----------
    def reset(self):
        self.layout = deepcopy(self.initial_layout)
        self.pos = START
        self.has_key = False
        self.has_gem = False
        self.door_open = False
        self.steps = 0
        self.done = False
        self.success = False
        self.fail_reason = None
        self.action_log = []
        self.history = []
        return self.observe()

    # ---------- Observation ----------
    def render_map(self):
        rows = []
        for r in range(self.n):
            cells = []
            for c in range(self.n):
                if (r, c) == self.pos:
                    cells.append("A")
                else:
                    cells.append(self.layout[r][c])
            rows.append(f"{r:2d} " + " ".join(cells))
        return "\n".join(rows)

    # —— Textification of the sensor layer ——
    # The purpose is not to plan for the model，instead replace「Read Absolute Coordinates + The heart counts the direction + Check the target」
    # These three 7B Something unstable under the zero-perception budget，with sentences the model can understand。
    # Keep complete maps and absolute coordinates，the information content is unchanged，only the representation changes。
    def _obs_cell(self, r, c):
        """What does a grid mean to a model?（Description Only，Do Not Give Action）。"""
        if not _in_bounds((r, c), self.n):
            return "outside the map (moving there does nothing)"
        cell = self.layout[r][c]
        if cell == WALL:
            return "wall (blocked)"
        if cell == TRAP:
            return "TRAP (stepping on it ends the episode in failure)"
        if cell == DOOR:
            return "locked door (blocks movement; open() can remove it)"
        if cell == KEY:
            return "key (call pick() while standing here)"
        if cell == GEM:
            return "gem (call pick() while standing here)"
        return "empty"

    def _move_outcomes(self):
        """The consequences of each of the four directions。The traps are single-marked dangerous，eliminate wall collisions directly/and trap-entry failures。"""
        lines = []
        for name, delta in MOVE_DELTAS.items():
            nxt = (self.pos[0] + delta[0], self.pos[1] + delta[1])
            lines.append(f"- {name}: {self._obs_cell(*nxt)}")
        return lines

    def _here(self):
        cell = self.layout[self.pos[0]][self.pos[1]]
        if cell == KEY:
            return "key (call pick() while standing here)"
        if cell == GEM:
            return "gem (call pick() while standing here)"
        return "empty"

    def _where(self, target):
        """copy「target position relative to the agent」Write a directional phrase，avoid mental coordinate subtraction。"""
        dr = target[0] - self.pos[0]
        dc = target[1] - self.pos[1]
        if dr == 0 and dc == 0:
            return "under your feet"
        bits = []
        if dr:
            bits.append(f"{abs(dr)} row{'s' if abs(dr) > 1 else ''} "
                        f"{'down' if dr > 0 else 'up'}")
        if dc:
            bits.append(f"{abs(dc)} column{'s' if abs(dc) > 1 else ''} "
                        f"{'right' if dc > 0 else 'left'}")
        return " and ".join(bits)

    def _target_lines(self):
        """List the unfinished targets in order of task，Change distance to square phrases。"""
        key_pos = door_pos = gem_pos = None
        for r in range(self.n):
            for c in range(self.n):
                cell = self.layout[r][c]
                if cell == KEY:
                    key_pos = (r, c)
                elif cell == DOOR and not self.door_open:
                    door_pos = (r, c)
                elif cell == GEM:
                    gem_pos = (r, c)
        lines = []
        if key_pos is not None and not self.has_key:
            lines.append(f"- key at (row {key_pos[0]}, col {key_pos[1]}): {self._where(key_pos)}")
        if door_pos is not None:
            nearby = _dist(self.pos, door_pos) <= 1
            lines.append(
                f"- locked door at (row {door_pos[0]}, col {door_pos[1]}): {self._where(door_pos)}"
                f" — {'within reach of open()' if nearby else 'too far to open()'}"
            )
        if gem_pos is not None and not self.has_gem:
            lines.append(f"- gem at (row {gem_pos[0]}, col {gem_pos[1]}): {self._where(gem_pos)}")
        lines.append(f"- start cell (row 0, col 0): {self._where(START)}")
        return lines

    def observe(self):
        parts = [
            f"Step {self.steps}/{self.max_steps} ({self.max_steps - self.steps} steps left)",
            f"Your position: row {self.pos[0]}, col {self.pos[1]}",
            f"Holding key: {self.has_key}",
            f"Holding gem: {self.has_gem}",
            f"Door open: {self.door_open}",
            "RESULT OF YOUR LAST ACTION: " + (
                self.history[-1]["result"] if self.history else "Episode just started."
            ),
        ]

        # Objective（Relative position）And directional consequences in front of the map：The first thing the model reads is the information that can act directly
        parts.append("Targets relative to you:")
        parts.extend(self._target_lines())
        parts.append("What each move leads to:")
        parts.extend(self._move_outcomes())
        parts.append(f"Under your feet: {self._here()}")

        parts.append("Map (. empty, # wall, T trap, D locked door, k key, g gem, A you):")
        parts.append("   " + " ".join(str(c) for c in range(self.n)))
        parts.append(self.render_map())

        # The absolute coordinates of the walls and traps are left to turn back/Reference for detour
        hazards = []
        for r in range(self.n):
            for c in range(self.n):
                cell = self.layout[r][c]
                if cell in (WALL, TRAP):
                    hazards.append(f"{'wall' if cell == WALL else 'trap'} at (row {r}, col {c})")
        parts.append("Walls and traps on the map: " + ("; ".join(hazards) or "none"))
        parts.append("Goal: pick up the gem, then return to (row 0, col 0).")

        if self.done:
            parts.append("EPISODE OVER — success" if self.success
                         else f"EPISODE OVER — failure: {self.fail_reason}")
        return "\n".join(parts)

    # ---------- A snapshot/restore（pilot foundation） ----------
    def snapshot(self):
        return {
            "n": self.n,
            "layout": [row[:] for row in self.layout],
            "initial_layout": deepcopy(self.initial_layout),
            "max_steps": self.max_steps,
            "seed": self.seed,
            "pos": list(self.pos),
            "has_key": self.has_key,
            "has_gem": self.has_gem,
            "door_open": self.door_open,
            "steps": self.steps,
            "done": self.done,
            "success": self.success,
            "fail_reason": self.fail_reason,
            "action_log": deepcopy(self.action_log),
            "history": deepcopy(self.history),
        }

    @classmethod
    def restore(cls, snap):
        obj = cls.__new__(cls)
        obj.n = snap["n"]
        obj.layout = [row[:] for row in snap["layout"]]
        obj.initial_layout = deepcopy(snap.get("initial_layout", snap["layout"]))
        obj.max_steps = snap["max_steps"]
        obj.seed = snap["seed"]
        obj.pos = tuple(snap["pos"])
        obj.has_key = snap["has_key"]
        obj.has_gem = snap["has_gem"]
        obj.door_open = snap["door_open"]
        obj.steps = snap["steps"]
        obj.done = snap["done"]
        obj.success = snap["success"]
        obj.fail_reason = snap["fail_reason"]
        obj.action_log = deepcopy(snap["action_log"])
        obj.history = deepcopy(snap.get("history", [
            {"action": action, "step": t, "result": "Execution feedback not recorded in legacy snapshot."}
            for t, action in enumerate(obj.action_log)
        ]))
        return obj

    # ---------- Transfer ----------
    def step(self, action):
        """action: dict，A shape like that {"tool": "move", "arg": "up"}"""
        if self.done:
            return self.observe(), True

        self.steps += 1
        action = deepcopy(action)
        self.action_log.append(action)
        before = list(self.pos)
        tool = action.get("tool")
        arg = action.get("arg")
        n = self.n

        if tool == "give_up":
            self.done, self.success, self.fail_reason = True, False, "abandonment"
            result = "Gave up. Episode ended in failure."
            code = "give_up"

        elif tool == "look":
            # Deliberately useless tool：The map is all there is。only consumes a step。
            result, code = "Looked at the already visible map; consumed one step.", "look"

        elif tool == "move":
            delta = MOVE_DELTAS.get(arg)
            if delta is None:
                result, code = f"Invalid direction {arg!r}; use up/down/left/right. Did not move.", "invalid_arg"
            else:
                nxt = (self.pos[0] + delta[0], self.pos[1] + delta[1])
                if not _in_bounds(nxt, n):
                    result, code = f"Blocked: (row {nxt[0]}, col {nxt[1]}) is outside the map. Did not move.", "boundary"
                else:
                    cell = self.layout[nxt[0]][nxt[1]]
                    if cell == WALL:
                        result, code = f"Blocked: wall at (row {nxt[0]}, col {nxt[1]}). Did not move.", "wall"
                    elif cell == DOOR and not self.door_open:
                        # colliding with a locked door：does not terminate the episode，only consumes steps（this is ramp one source of）
                        result, code = f"Blocked: locked door at (row {nxt[0]}, col {nxt[1]}). Did not move.", "locked_door"
                    else:
                        self.pos = nxt
                        result, code = f"Moved {arg} to (row {nxt[0]}, col {nxt[1]}).", "move"
                        if cell == TRAP:
                            # Step on the trap：The gems are permanently destroyed，Irreversible —— cliff Events
                            self.done, self.success, self.fail_reason = True, False, "trap_step"
                            result, code = "Stepped on a trap. Episode ended in failure.", "trap"
                        # Attention：Items do not automatically pick up，You have to make a public call pick()。
                        # Rationale：LLM of function-calling Intuition is what it is pick，
                        # Asked it to be counter-intuitive pick Itll make a lot of things that couldve been right，
                        # all failures concentrate in redundant_steps，cannot fill quotas for all six types。

        elif tool == "open":
            door_pos = None
            for r in range(n):
                for c in range(n):
                    if self.layout[r][c] == DOOR:
                        door_pos = (r, c)
            if door_pos is None:
                result, code = "No locked door remains; open had no effect.", "no_door"
            elif _dist(self.pos, door_pos) > 1:
                result, code = (f"Too far to open door at (row {door_pos[0]}, col {door_pos[1]}). "
                                "Must be within one row AND one column of it. Door is still locked."), "too_far"
            elif not self.has_key:
                # No key to open the door：terminate immediately —— cliff Events
                self.done, self.success, self.fail_reason = True, False, "precondition_violation"
                result, code = "Tried opening the adjacent door without a key. Episode ended in failure.", "missing_key"
            else:
                self.door_open = True
                self.has_key = False
                self.layout[door_pos[0]][door_pos[1]] = EMPTY
                result, code = "Opened the door successfully. The key was consumed.", "open"

        elif tool == "pick":
            # Show Pick：only applies to the object at the current cell
            cur = self.layout[self.pos[0]][self.pos[1]]
            if cur == KEY:
                self.has_key = True
                self.layout[self.pos[0]][self.pos[1]] = EMPTY
                result, code = "Picked up the key. You are holding the key.", "pick_key"
            elif cur == GEM:
                self.has_gem = True
                self.layout[self.pos[0]][self.pos[1]] = EMPTY
                result, code = "Picked up the gem. Carry it back to (row 0, col 0).", "pick_gem"
            else:
                result, code = "Nothing to pick up on this cell; consumed one step.", "nothing_to_pick"
            # Do not call on item pick：Invalid，Just one step

        else:
            result, code = f"Unknown tool {tool!r}; consumed one step with no effect.", "invalid_tool"

        # Conditions for success：Take the stones back to the beginning
        if not self.done and self.has_gem and self.pos == START:
            self.done, self.success, self.fail_reason = True, True, None
            result += " Returned with the gem: success."

        if not self.done and self.steps >= self.max_steps:
            self.done, self.success = True, False
            self.fail_reason = self._timeout_reason()
            result += " Step budget exhausted."

        self.history.append({"action": action, "step": self.steps - 1, "result": result,
                             "event": code, "pos_before": before, "pos_after": list(self.pos)})
        return self.observe(), self.done

    def _timeout_reason(self):
        """Failed to break down after timeout，For ex post diversion of quotas。"""
        if self.has_gem:
            return "timeout_return"
        if not self.door_open:
            return "timeout_before_door"
        return "timeout_before_gem"

    # ---------- Tools schema ----------
    TOOL_SPEC = """Available tools:
- move(direction): "up" decreases row; "down" increases row; "left" decreases column;
  "right" increases column. Moves one cell. Walls, boundaries and locked doors block movement.
- open(): opens the locked door. You must be holding the key and within one row AND
  one column of the door (diagonal adjacency is allowed). The key is consumed on success.
  Opening the door WITHOUT the key ends the episode in failure.
- pick(): picks up the item on your current cell (key or gem). You MUST call this
  while standing on the item — walking onto it does not pick it up automatically.
- look(): reveals your surroundings. The map above is already fully visible, so this is rarely needed.
- give_up(): ends the episode as a failure.

Goal: pick up the gem (g), carry it back to the start cell (row 0, col 0), and you win.
Stepping on a trap (T) destroys the gem and ends the episode in failure."""
