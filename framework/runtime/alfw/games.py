"""Game file collection and sampling。Filter rules with alfworld AlfredTWEnv.collect_game_files identical，But sort results，Guaranteed reappearance。

Read-only json，Do Not Import textworld，So nothing GPU、Unloaded alfworld It can be tested on the machine。
"""
import json
import os
import random
from pathlib import Path

from . import acfg

MOCK_PREFIX = "mock://"


def collect(split, data_root=None, verbose=False):
    """Back (list of absolute game-file paths [Sorted], Statistics)。"""
    if split.startswith("mock"):
        n = int(split.split(":")[1]) if ":" in split else 40
        return [f"{MOCK_PREFIX}{i}" for i in range(n)], {"n": n}
    data_root = Path(data_root or acfg.DATA)
    base = data_root / acfg.SPLIT_DIRS[split]
    if not base.is_dir():
        raise FileNotFoundError(f"not found {base}（Run first alfworld-download，and set ALFWORLD_DATA）")
    allowed = set(acfg.TASK_TYPES.values())
    stats = {"dirs": 0, "movable_or_sliced": 0, "task_type": 0, "no_game": 0, "no_solvable_key": 0,
             "unsolvable": 0, "kept": 0}
    out = []
    for root, _dirs, files in os.walk(base, topdown=False):
        if "traj_data.json" not in files:
            continue
        stats["dirs"] += 1
        if "movable" in root or "Sliced" in root:
            stats["movable_or_sliced"] += 1
            continue
        with open(os.path.join(root, "traj_data.json"), "r") as f:
            traj = json.load(f)
        if traj.get("task_type") not in allowed:
            stats["task_type"] += 1
            continue
        game = os.path.join(root, "game.tw-pddl")
        if not os.path.exists(game):
            stats["no_game"] += 1
            continue
        with open(game, "r") as f:
            gd = json.load(f)
        if "solvable" not in gd:
            stats["no_solvable_key"] += 1
            continue
        if not gd["solvable"]:
            stats["unsolvable"] += 1
            continue
        out.append(game)
        stats["kept"] += 1
    out.sort()
    if verbose:
        print(f"[games] {split}: {stats}")
    return out, stats


def cached_collect(split, data_root=None):
    """With a cache collect；Cache Records Root Directory，Change of directory will be recalculated。"""
    data_root = str(Path(data_root or acfg.DATA))
    if split.startswith("mock"):
        return collect(split)[0]
    acfg.OUT.mkdir(parents=True, exist_ok=True)
    path = acfg.OUT / f"games_{split}.json"
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("data_root") == data_root:
            return d["games"]
    games, stats = collect(split, data_root)
    path.write_text(json.dumps({"data_root": data_root, "stats": stats, "games": games}, indent=1))
    return games


def task_type_of(gamefile):
    if gamefile.startswith(MOCK_PREFIX):
        from .mockenv import MockGame
        return MockGame(gamefile).task_type
    with open(os.path.join(os.path.dirname(gamefile), "traj_data.json")) as f:
        return json.load(f)["task_type"]


def sample(games, n, seed, exclude=()):
    ex = set(exclude)
    pool = [g for g in games if g not in ex]
    if n >= len(pool):
        return list(pool)
    return sorted(random.Random(seed).sample(pool, n))


def short_name(gamefile):
    """An easy-to-read name：_Other Organiser/Test Directory。"""
    if gamefile.startswith(MOCK_PREFIX):
        return gamefile
    p = Path(gamefile)
    return f"{p.parent.parent.name}/{p.parent.name}"
