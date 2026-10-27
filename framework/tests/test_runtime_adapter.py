"""Exercise the real runtime adapter orchestration with simulated environments/models.

These tests do not claim that ALFWorld/WebShop or vLLM were installed or run.
"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from repair_eval.backends.paper_runtime import Backend

RUNTIME=Path(__file__).resolve().parents[1]/"runtime"
sys.path.insert(0,str(RUNTIME))


class AlfAdapter(Backend):
    def __init__(self):
        self.config={"max_active":2}
        self.pool=None
    def _open(self,env):
        from alfw import pool,rollout,agent
        self.pool=pool.make_pool("mock",1)
        self.rollout=rollout
        self.backend=agent.MockBackend()
    def _spec(self,j):
        return {"rid":j["job_id"],"gamefile":"mock://1","prefix":j["prefix"],"seed":j["seed"]}


class WebPool:
    def __init__(self):self.states={}
    def state(self,n):
        return {"obs":"[button] Buy Now [button_]" if n<2 else "Done", "clicks":["buy now"],
                "search":False,"page":"item_page","done":n>=2,"reward":float(n>=2),
                "won":n>=2,"key":str(n)}
    def open(self,items):
        out={}
        for s in items:
            self.states[s["rid"]]=len(s["prefix"])
            pre=[{"state":self.state(i+1),"valid":True} for i in range(len(s["prefix"]))]
            out[s["rid"]]={"instruction":"Buy something","init":self.state(0),"prefix":pre,
                           "state":self.state(len(pre))}
        return out
    def step(self,items):
        out={}
        for s in items:
            self.states[s["rid"]]+=1
            out[s["rid"]]={"state":self.state(self.states[s["rid"]]),"valid":True,"dt":0.}
        return out
    def close(self,rids):
        for rid in rids:self.states.pop(rid,None)
    def shutdown(self):pass


class WebModel:
    context=16384
    def render(self,m):return str(m)
    def count(self,p):return len(p)
    def generate(self,prompts,temperature,seeds,max_new_tokens,hints=None):
        return [SimpleNamespace(text="Thought: Buy.\nAction: click[buy now]",finish_reason="stop",n_prompt=10,n_gen=10) for _ in prompts]


class WebAdapter(Backend):
    def __init__(self):self.config={"max_active":2};self.pool=None
    def _open(self,env):
        from webshop import ws_rollout
        self.pool=WebPool();self.backend=WebModel();self.rollout=ws_rollout


def job(env):
    return {"job_id":"j0","branch_id":"b0","trajectory":"t0","environment":env,
            "mode":"sample","prefix":["look" if env=="alfworld" else "click[item]"],
            "replay":{"goal":0},"seed":42,"max_steps":4,"temperature":.7,"expected_prefix_observations":[]}


class RuntimeTests(unittest.TestCase):
    def test_alfworld_adapter_flow(self):
        b=AlfAdapter()
        try:
            rows=list(b.run([job("alfworld")]))
            self.assertEqual(len(rows),1)
            self.assertIn(rows[0]["success"],[0.,1.])
            self.assertLessEqual(rows[0]["steps"],3)
        finally:b.close()
    def test_webshop_adapter_flow(self):
        b=WebAdapter()
        try:
            rows=list(b.run([job("webshop")]))
            self.assertEqual(rows[0]["reward"],1.)
            self.assertEqual(rows[0]["success"],1.)
        finally:b.close()
    def test_replay_mismatch_stops_before_scoring(self):
        b=AlfAdapter();j=job("alfworld");j["expected_prefix_observations"]=["wrong"]
        try:
            with self.assertRaisesRegex(ValueError,"Prefix observations changed"):
                list(b.run([j]))
        finally:b.close()


if __name__=="__main__":unittest.main()
