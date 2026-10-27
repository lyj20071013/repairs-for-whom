"""Strict JSON I/O and content fingerprints. No downloads or environment imports."""
import hashlib
import json
from pathlib import Path


def _unique(pairs):
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError(f"Duplicate JSON key: {k}")
        obj[k] = v
    return obj


def loads(s):
    return json.loads(s, object_pairs_hook=_unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-finite JSON: {x}")))


def read_json(path):
    return loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(loads(line))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{Path(path).name}:{i}: {exc}") from exc
    return rows


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seed_for(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7fffffff


def safe_member(root, relative):
    root = Path(root).resolve()
    dest = (root / relative).resolve()
    if not dest.is_relative_to(root) or dest == root:
        raise ValueError(f"Path outside bundle: {relative}")
    return dest
