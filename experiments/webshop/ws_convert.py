"""Generate search for documents.jsonl：Use specified commodities / Property File Run Original search_engine/convert_product_file_format.py（Original file unchanged）。

at WebShop of Python run in the environment（setup_webshop.sh number 7 Step-by-Step；selftest Use it, too）：
    python webshop/ws_convert.py --webshop-dir /root/autodl-tmp/webshop/WebShop \\
        --product-file items_shuffle_100k.json --attr-file items_ins_v2_100k.json --resources resources_100k
Original Script resources、resources_100、resources_1k、resources_100k Four；after writing, in
search_engine/<resources>.done Write the number of documents for this、sha256 & Peak Memory（setup Use it to determine whether this step is complete or not）。

setup Always add when running it --shims（Pure Python of LuceneSearcher Substitution）：Convert only in the original version load_products，
engine.py Top import of LuceneSearcher is never called here；So you dont have to load for the conversion torch and Java（Cost-saving 1 GB Memory），
Generated documents.jsonl and using the real pyserini identical byte for byte when。
"""
import argparse
import hashlib
import json
import os
import resource
import runpy
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webshop-dir", required=True)
    ap.add_argument("--product-file", required=True)
    ap.add_argument("--attr-file", required=True)
    ap.add_argument("--resources", required=True, help="The one to check，e.g resources_100k")
    ap.add_argument("--compat", default=None)
    ap.add_argument("--shims", default=None)
    ap.add_argument("--stubs", default=None)
    a = ap.parse_args()
    wd = os.path.abspath(a.webshop_dir)
    se = os.path.join(wd, "search_engine")
    sys.path.insert(0, wd)
    for d in (a.compat, a.shims, a.stubs):
        if d:
            sys.path.insert(0, os.path.abspath(d))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.chdir(se)
    marker = os.path.join(se, a.resources + ".done")
    if os.path.exists(marker):
        os.remove(marker)
    for r in ("resources", "resources_100", "resources_1k", "resources_100k"):
        os.makedirs(r, exist_ok=True)
    import ws_data
    fp, atp = ws_data.apply(wd, a.product_file, a.attr_file)
    t0 = time.time()
    runpy.run_path("convert_product_file_format.py", run_name="__main__")
    path = os.path.join(se, a.resources, "documents.jsonl")
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for line in f:
            h.update(line)
            n += 1
    info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "product_file": fp, "attr_file": atp,
            "documents": n, "sha256": h.hexdigest(), "seconds": round(time.time() - t0, 1),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2),
            "pyserini_module": getattr(sys.modules.get("pyserini"), "__file__", None)}
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
