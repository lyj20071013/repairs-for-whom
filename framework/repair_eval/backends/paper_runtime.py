"""Adapter to the packaged original environment and vLLM rollout implementations.

Requires Linux, separately installed model/environment dependencies, and explicit paths.
No network calls, installation or GPU work occurs until this backend is invoked.
"""
import functools
import os
from pathlib import Path
import sys


class Backend:
    def __init__(self, config):
        if os.environ.get("PYTHONHASHSEED") != "0":
            raise ValueError("Start Python with PYTHONHASHSEED=0 for replay consistency")
        self.config=config
        self.runtime=Path(config["runtime_root"]).resolve()
        self.model_path=Path(config["model_path"]).resolve()
        if not self.model_path.is_dir() or not (self.runtime/"alfw"/"agent.py").is_file():
            raise ValueError("Missing model or packaged runtime")
        if config.get("model_name") != config["expected_model"]:
            raise ValueError("Backend model_name must match the frozen replay plan")
        sys.path.insert(0,str(self.runtime))
        self.pool=None
        self.backend=None

    def _open(self, env):
        from alfw import acfg, pool
        from alfw.agent import VLLMBackend
        workers=int(self.config.get("workers",1))
        if env=="alfworld":
            from alfw import rollout
            self.data=Path(self.config["alfworld_data"]).resolve()
            self.pool=pool.make_pool("tw",workers)
            context=8192
        elif env=="webshop":
            from webshop import ws_rollout as rollout
            from webshop.ws_pool import WSPool
            self.pool=WSPool(n_workers=workers,python=self.config["webshop_python"],
                             webshop_dir=self.config["webshop_dir"], search="pyserini",
                             seed=0,num_products=100000,mode="text_rich",
                             log_dir=self.config.get("worker_logs","outputs/worker_logs"))
            context=16384
        else:
            raise ValueError("This backend supports ALFWorld and WebShop")
        self.rollout=rollout
        # Set the original runtime's generation settings explicitly.
        acfg.MODEL_CONTEXT=context
        acfg.TOP_P=.95
        tp=int(self.config.get("tensor_parallel_size",1))
        if tp>1:
            import vllm
            vllm.LLM=functools.partial(vllm.LLM,tensor_parallel_size=tp)
        self.backend=VLLMBackend(model_dir=str(self.model_path),context=context)
        # Llama requires both EOS and end-of-turn stopping, as in the source runner.
        if self.config.get("llama_stop_tokens",False):
            tok=self.backend.tok
            stop={tok.eos_token_id}
            vocab=tok.get_vocab()
            for token in ["<|eot_id|>","<|end_of_text|>"]:
                if token in vocab:
                    stop.add(vocab[token])
            self.backend._SP=functools.partial(self.backend._SP,stop_token_ids=sorted(x for x in stop if x is not None))

    def _spec(self,j):
        spec={"rid":j["job_id"],"seed":j["seed"],"prefix":j["prefix"]}
        if j["environment"]=="alfworld":
            game=(self.data/j["replay"]["game"]).resolve()
            if not game.is_relative_to(self.data) or not game.is_file():
                raise ValueError("Missing/invalid ALFWorld game file")
            spec["gamefile"]=str(game)
        else:
            spec["goal"]=j["replay"]["goal"]
        return spec

    def _verify(self,j):
        spec=self._spec(j)
        spec["prefix"]=spec["prefix"][:-1]
        opened=self.pool.open([spec])[spec["rid"]]
        try:
            if j["environment"]=="alfworld":
                obs=opened["prefix_obs"]
            else:
                from webshop.ws_agent import clean_obs
                obs=[clean_obs(s["state"]["obs"],opened["instruction"]) for s in opened["prefix"]]
            if obs!=j["expected_prefix_observations"]:
                raise ValueError(f"Prefix observations changed: {j['trajectory']}, branch {j['branch_id']}")
        finally:
            self.pool.close([spec["rid"]])

    def run(self,jobs):
        if not jobs:
            return
        env=jobs[0]["environment"]
        if any(j["environment"]!=env for j in jobs):
            raise ValueError("A replay run must use one environment")
        self._open(env)
        checked=set()
        width=int(self.config.get("max_active",32))
        if width<1:
            raise ValueError("max_active must be positive")
        for mode in ["sample","greedy"]:
            group=[j for j in jobs if j["mode"]==mode]
            for start in range(0,len(group),width):
                chunk=group[start:start+width]
                for j in chunk:
                    key=(j["trajectory"],len(j["prefix"]))
                    if key not in checked:
                        self._verify(j)
                        checked.add(key)
                done=[]
                def on_done(e):
                    st=self.rollout.episode_stats(e)
                    r={"job_id":e["rid"],"success":float(e["won"]),"steps":st["n"],
                       "invalid":st["invalid"]+st["no_action"],"retries":st["retries"]}
                    if env=="webshop":
                        r["reward"]=float(e["reward"])
                    done.append(r)
                self.rollout.run_stream(self.backend,self.pool,[self._spec(j) for j in chunk],
                                        temperature=chunk[0]["temperature"],on_done=on_done,
                                        max_active=width,max_steps=chunk[0]["max_steps"],max_new_tokens=256)
                yield from done

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


def create(config):
    return Backend(config)
