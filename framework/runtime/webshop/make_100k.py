"""from WebShop Before full data intercepts 10 10,000 commodities（run once on your local machine；Use only Python Standard library）。For registration see analysis plan.md 6.13 Revision 1。

    python D:\\Paper\\repair_transfer\\webshop\\make_100k.py

Read it webshop_data\\ Yeah（Do not change filename）：
    items_shuffle.json      Total merchandise（WebShop setup.sh -d all The file，approximately 5 GB）
    items_ins_v2.json       Full Properties
    items_human_ins.json    People write orders
Write webshop_data\\ Yeah：
    items_shuffle_100k.json.gz   before 10 Original text of 10,000 commodity elements（Ill take it word for word，Do not reorder）
    items_ins_v2_100k.json.gz    these products in items_ins_v2.json Entries in
    data_100k_manifest.json      Enter and Output sha256、Number of items、Target、and 1000 Comparison of the unwritten version

WebShop Load Original num_products = 100000 select the first entries of the product file 10 Ten thousand elements（The file itself is out of order），
So load and intercept the file = Reload full file before loading 10 Ten thousand。Attribute files by commodity number，The search results for these commodities remain unchanged in the subset。
Full files read only in sequence（At the same time sha256、Number of elements），Memory is only a few hundred MB。
"""
import argparse
import codecs
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

N_DEFAULT = 100000
WS = re.compile(r"[ \t\n\r]*")


class ArrayReader:
    """Take a big one JSON The array reads it by element：Back (Original Element, parsing objects)。At the same time, count the entire file sha256。"""

    def __init__(self, path, chunk=1 << 24):
        self.f = open(path, "rb")
        self.dec = codecs.getincrementaldecoder("utf-8")()
        self.jd = json.JSONDecoder()
        self.sha = hashlib.sha256()
        self.chunk = chunk
        self.buf, self.pos, self.eof, self.bytes = "", 0, False, 0

    def _more(self):
        b = self.f.read(self.chunk)
        self.sha.update(b)
        self.bytes += len(b)
        if not b:
            self.eof = True
            text = self.dec.decode(b"", final=True)
        else:
            text = self.dec.decode(b)
        self.buf = self.buf[self.pos:] + text
        self.pos = 0

    def _skip_ws(self):
        while True:
            self.pos = WS.match(self.buf, self.pos).end()
            if self.pos < len(self.buf) or self.eof:
                return
            self._more()

    def _peek(self):
        self._skip_ws()
        return self.buf[self.pos] if self.pos < len(self.buf) else ""

    def __iter__(self):
        if self._peek() == "\ufeff":
            self.pos += 1
        if self._peek() != "[":
            raise ValueError("The document is not based on [ The beginning JSON Array")
        self.pos += 1
        first = True
        while True:
            c = self._peek()
            if c == "]":
                self.pos += 1
                return
            if not first:
                if c != ",":
                    raise ValueError(f"elements must be separated by commas，Read {c!r}（The file may not be downloaded in full）")
                self.pos += 1
                self._skip_ws()
            first = False
            while True:
                try:
                    obj, end = self.jd.raw_decode(self.buf, self.pos)
                    break
                except json.JSONDecodeError:
                    if self.eof:
                        raise ValueError("File ends in the middle of an element（likely an incomplete download）")
                    self._more()
            raw = self.buf[self.pos:end]
            self.pos = end
            yield raw, obj

    def finish(self):
        """Read the remaining bytes（Only for a complete file sha256），check for extra content after the array。"""
        self._skip_ws()                          # Read to next non-space character or end of file
        tail = self.buf[self.pos:self.pos + 20]
        while not self.eof:
            self.buf, self.pos = "", 0
            self._more()
        self.f.close()
        return tail


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class HashingGzip:
    """Write gzip（mtime Fixed As 0，The same thing gets in the same document），also compute over decompressed content sha256。"""

    def __init__(self, path):
        self.raw = open(path, "wb")
        self.gz = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0, compresslevel=6)
        self.sha = hashlib.sha256()
        self.n = 0

    def write(self, s):
        b = s.encode("utf-8")
        self.sha.update(b)
        self.n += len(b)
        self.gz.write(b)

    def close(self):
        self.gz.close()
        self.raw.close()


def valid_asin(asin):
    """and WebShop load_products Same filter：asin is 'nan' or longer than 10 Skip for a character。"""
    return isinstance(asin, str) and asin != "nan" and len(asin) <= 10


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=here.parent / "webshop_data")
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="number of initial products to retain（default 100000）")
    ap.add_argument("--suffix", default="100k", help="Tags in the output filename（default 100k）")
    a = ap.parse_args()
    d = a.data_dir
    full, attrs_p, human_p = d / "items_shuffle.json", d / "items_ins_v2.json", d / "items_human_ins.json"
    for p in (full, attrs_p, human_p):
        if not p.exists():
            sys.exit(f"not found {p}（from Google Drive Download and put in {d}，Do not change filename）")
    out_prod = d / f"items_shuffle_{a.suffix}.json.gz"
    out_attr = d / f"items_ins_v2_{a.suffix}.json.gz"
    out_man = d / f"data_{a.suffix}_manifest.json"
    t0 = time.time()
    print(f"Read it {full}（{full.stat().st_size / 1e9:.2f} GB），Before intercepting {a.n} Annex -> {out_prod.name}", flush=True)

    # ---------- Commodities：before n Original text of the elements；Keep reading the whole file，Numerical elements、Yes sha256
    rd = ArrayReader(full)
    w = HashingGzip(str(out_prod) + ".tmp")
    w.write("[")
    asins, first1000, total = [], [], 0
    for raw, obj in rd:
        if total < a.n:
            if total:
                w.write(",\n")
            w.write(raw)
            asins.append(obj.get("asin") if isinstance(obj, dict) else None)
            if total < 1000:
                first1000.append(obj)
        total += 1
        if total == a.n:
            w.write("]")
            w.close()
            print(f"  already wrote the first {a.n} Annex（{w.n / 1e6:.0f} MB Uncompressed）；Then read the whole file and check the integrity……", flush=True)
        elif total % 200000 == 0:
            print(f"  Read {total} Annex（{rd.bytes / 1e9:.2f} GB，{time.time() - t0:.0f} seconds）", flush=True)
    tail = rd.finish()
    if total < a.n:
        w.write("]")
        w.close()
    if tail:
        sys.exit(f"extra content after the end of the array：{tail!r}（There may be a problem with the file）")
    Path(str(out_prod) + ".tmp").replace(out_prod)
    full_sha = rd.sha.hexdigest()
    print(f"  Full file total {total} Annex，sha256 {full_sha[:16]}…（{time.time() - t0:.0f} seconds）", flush=True)
    if total < a.n:
        sys.exit(f"Full file only {total} Annex，less than {a.n} Annex（likely an incomplete download）")

    # ---------- Properties：Entries for these commodities
    print(f"Read it {attrs_p.name}（{attrs_p.stat().st_size / 1e6:.0f} MB）……", flush=True)
    attrs_sha = sha256_file(attrs_p)
    attrs = json.loads(attrs_p.read_text(encoding="utf-8"))
    want = {x for x in asins if isinstance(x, str)}
    sub = {k: v for k, v in attrs.items() if k in want}
    wa = HashingGzip(str(out_attr) + ".tmp")
    wa.write(json.dumps(sub, ensure_ascii=False))
    wa.close()
    Path(str(out_attr) + ".tmp").replace(out_attr)
    n_attrs_full = len(attrs)
    del attrs

    # ---------- Target（and WebShop load_products + get_human_goals The same rules）
    human = json.loads(human_p.read_text(encoding="utf-8"))
    seen, n_goals, n_goal_products, n_dup, n_bad = set(), 0, 0, 0, 0
    for x in asins:
        if not valid_asin(x):
            n_bad += 1
            continue
        if x in seen:
            n_dup += 1
            continue
        seen.add(x)
        if x in human:
            k = sum(1 for ins in human[x] if ins.get("instruction_attributes"))
            n_goals += k
            n_goal_products += 1 if k else 0

    # ---------- and 1000 Subtext comparison（informational only，Without prejudice to outcome）
    cmp = {}
    small = d / "items_shuffle_1000.json"
    if small.exists():
        sm = json.loads(small.read_text(encoding="utf-8"))
        cmp["items_shuffle_1000_is_prefix"] = sm == first1000[:len(sm)]
    small_a = d / "items_ins_v2_1000.json"
    if small_a.exists():
        sa = json.loads(small_a.read_text(encoding="utf-8"))
        cmp["items_ins_v2_1000_entries_match"] = all(sub.get(k) == v for k, v in sa.items() if k in sub)
        cmp["items_ins_v2_1000_covered"] = sum(1 for k in sa if k in sub)

    man = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "n": a.n,
        "inputs": {"items_shuffle.json": {"bytes": full.stat().st_size, "sha256": full_sha, "elements": total},
                   "items_ins_v2.json": {"bytes": attrs_p.stat().st_size, "sha256": attrs_sha, "entries": n_attrs_full},
                   "items_human_ins.json": {"bytes": human_p.stat().st_size, "sha256": sha256_file(human_p),
                                            "asins": len(human)}},
        "outputs": {f"items_shuffle_{a.suffix}.json": {"sha256": w.sha.hexdigest(), "bytes": w.n, "elements": len(asins),
                                                    "gz": out_prod.name, "gz_sha256": sha256_file(out_prod)},
                    f"items_ins_v2_{a.suffix}.json": {"sha256": wa.sha.hexdigest(), "bytes": wa.n, "entries": len(sub),
                                                   "gz": out_attr.name, "gz_sha256": sha256_file(out_attr)}},
        "expected": {"unique_valid_products": len(seen), "duplicate_asins": n_dup, "invalid_asins": n_bad,
                     "human_goals": n_goals, "products_with_human_goals": n_goal_products},
        "compare_1000": cmp,
    }
    out_man.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(f"complete（{time.time() - t0:.0f} seconds）：")
    print(f"  {out_prod.name}：{len(asins)} Annex，After Compression {out_prod.stat().st_size / 1e6:.0f} MB")
    print(f"  {out_attr.name}：{len(sub)} Bar Properties，After Compression {out_attr.stat().st_size / 1e6:.1f} MB")
    print(f"  Expected objectives（People write orders）：{n_goals} items，Distributed in {n_goal_products} On commodities"
          f"（Go back to the rear {len(seen)} Commodities；Repeat Numbering {n_dup}、Invalid numbering {n_bad}）")
    if cmp:
        print(f"  and 1000 Subtext comparison：{json.dumps(cmp, ensure_ascii=False)}")
    print(f"  List：{out_man.name}")
    print("Next：use sync.py push Uploading these three files（See README 6.13）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
