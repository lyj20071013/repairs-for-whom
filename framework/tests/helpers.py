from pathlib import Path
from repair_eval.io import write_json, write_jsonl, sha256, fingerprint
from repair_eval.dataset import Dataset


def tiny_bundle(path):
    path=Path(path)
    tr={"traj_id":"t0","max_steps":4,"candidates":[
        {"id":"a","branch_id":"a","original_branch_id":"o","position_index":0},
        {"id":"b","branch_id":"b","original_branch_id":"o","position_index":0}]}
    branches={bid:{"traj_id":"t0","t":1,"prefix":[bid],"replay":{},"deterministic":{"success":None},
                   "metadata":{"planner":False,"random":False,"proposers":[],"flash_rank":None,"oracle_F":0.,"oracle_slack":0.}}
              for bid in ["a","b","o"]}
    task={"id":"t0","environment":"synthetic","deployer":"d","instruction":"Synthetic unit test",
          "history":[{"step":1,"action":"o","observation":"nothing"}],"max_steps":4,
          "candidates":[{"id":"a","step":1,"action":"a"},{"id":"b","step":1,"action":"b"}]}
    records=[]
    for bid,s in [("a",[1.,0.]),("b",[0.,0.]),("o",[0.,0.])]:
        for mode,v in [("sample",s),("greedy",[0.])]:
            records.append({"branch_id":bid,"mode":mode,"success":v,"seeds":list(range(len(v)))})
    write_json(path/"design.json",{"trajectories":[tr],"branches":branches})
    write_jsonl(path/"tasks.jsonl",[task])
    write_jsonl(path/"outcomes/d.jsonl",records)
    m={"schema_version":"repair-eval/1","pool_id":"synthetic","environment":"synthetic","deployer":"d",
       "metrics":["success"],"primary_metric":"success","executors":{"d":{"samples":2}},"max_steps":4,
       "files":{p.relative_to(path).as_posix():sha256(p) for p in path.rglob("*") if p.is_file()}}
    m["bundle_id"]=fingerprint(m)
    write_json(path/"manifest.json",m)
    return Dataset(path)


class FakeBackend:
    synthetic=True
    def __init__(self,config):
        self.config=config
    def run(self,jobs):
        for i,j in enumerate(jobs):
            if self.config.get("partial") and i==3 and len(jobs)>3:
                return
            yield {"job_id":j["job_id"],"success":float(j["branch_id"]=="a")}
    def close(self):
        pass


def fake_backend(config):
    return FakeBackend(config)
