"""Copy only environment/policy runtime code, never credentials, logs or raw catalog data."""
import argparse
import shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from repair_eval.io import sha256,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True)
    args=ap.parse_args()
    src=Path(args.source)
    files=["alfw/"+name+".py" for name in ["__init__","acfg","agent","games","mockenv","pool","rollout","twenv"]]
    files += ["webshop/"+name+".py" for name in ["__init__","wcfg","ws_agent","ws_pool","ws_worker","ws_data","ws_convert","ws_rollout","make_100k"]]
    files += ["webshop/setup_webshop.sh","alfw_setup.sh","webshop/vendor/webshop_64fa2a5_min.tar.gz"]
    for folder in ["webshop/compat","webshop/shims","webshop/test_stubs"]:
        files += [p.relative_to(src).as_posix() for p in (src/folder).rglob("*.py")]
    hashes={}
    for f in files:
        target=ROOT/"runtime"/f
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src/f,target)
        hashes[f]=sha256(target)
    write_json(ROOT/"provenance"/"runtime_sources.json",hashes)
    print(f"Copied {len(hashes)} runtime files")


if __name__=="__main__":
    main()
