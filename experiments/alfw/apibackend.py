"""ALFWorld：API The model is a repeater（e.g Qwen3.8-flash）。and alfw.agent.VLLMBackend Same interface，Got it alfw.rollout wave-based continuations，
So prompts（build_messages，Include Status Lines）、Unexecutable Reaction、Script context、Gradual Feeds、Action parsing and 7B exactly identical。

- render(messages) Can not open message JSON String（API Get the message straight，do not apply a local chat template）；generate Can not open message。
- Context cap and 7B identical（acfg.MODEL_CONTEXT），The first steps of the same rule are omitted from history for a very long time。token Locally preferred to count Qwen2.5 A wordbox
  （ALFW_TOKENIZER_DIR，default 7B Model Directory），Number of characters without them / 3 Rough estimate。7B At stage B Maximum prompts 5,500 token，It doesnt actually trigger an omission。
- Sample：temperature 0.7、top_p 0.95、Up to one step 256 token、stop and 7B identical；temperature > 0 Gradual Feeds；thinking disabled。
- Every wave of requests sent and sent out（QWEN_EXEC_WORKERS，default 48）；each QWEN_EXEC_HEARTBEAT seconds（default 60）Print cumulative call and estimated cost。
- One call to retry has failed：Throw out an anomaly、The whole operation stopped（Completed Branch Disk，Run again with the same order）。Never fail to use a call as a strategy。
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from . import acfg


class APIBackend:
    name = "api"

    def __init__(self, model, context=None, workers=None, tokenizer_dir=None, heartbeat=True):
        from api_client import ChatClient
        self.client = ChatClient(model)
        self.model = model
        self.context = context or acfg.MODEL_CONTEXT
        self.workers = workers or int(os.environ.get("QWEN_EXEC_WORKERS", "48"))
        self.usage = self.client.usage
        self.tok = None
        tdir = tokenizer_dir or os.environ.get("ALFW_TOKENIZER_DIR") or str(acfg.MODEL_DIR)
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(tdir)
        except Exception:
            self.tok = None
        self._stop = threading.Event()
        self.t0 = time.time()
        if heartbeat:
            threading.Thread(target=self._beat, daemon=True).start()

    def settings(self):
        return dict(self.client.settings(), context=self.context, top_p=acfg.TOP_P, stop=acfg.STOP)

    def _beat(self):
        from api_client import cost_yuan
        while not self._stop.wait(float(os.environ.get("QWEN_EXEC_HEARTBEAT", "60"))):
            u = self.usage
            print(f"    [API {(time.time() - self.t0) / 60:.1f} minutes] Cumulative call {u['calls']}，Cache hit "
                  f"{u['cached_tokens'] / max(1, u['prompt_tokens']):.0%}，Cost estimates ¥{cost_yuan(u):.1f}，Try again {u['retries']}",
                  flush=True)

    def close(self):
        self._stop.set()

    # ---- and VLLMBackend Same interface ----
    def render(self, messages):
        return json.dumps(messages, ensure_ascii=False)

    def count(self, prompt):
        msgs = json.loads(prompt)
        if self.tok is not None:
            return len(self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        return len(prompt) // 3

    def generate(self, prompts, temperature, seeds, max_new_tokens, hints=None):
        def one(p, s):
            text, fr, u = self.client.call(json.loads(p), temperature, acfg.TOP_P, max_new_tokens,
                                           seed=s if temperature > 0 else None, stop=acfg.STOP)
            return SimpleNamespace(text=text, finish_reason=fr, n_prompt=(u or {}).get("prompt_tokens", 0) or 0,
                                   n_gen=(u or {}).get("completion_tokens", 0) or 0)
        if len(prompts) == 1:
            return [one(prompts[0], seeds[0])]
        with ThreadPoolExecutor(min(self.workers, len(prompts))) as ex:
            return list(ex.map(one, prompts, seeds))

    def preflight(self):
        """before execution, send one request at each temperature 0.7（with a seed）And temperature 0 A very short request，Confirm parameters（Ham stop）Accepted。"""
        msgs = [{"role": "user", "content": "Reply with exactly one line: Action: look"}]
        out = [self.client.call(msgs, t, acfg.TOP_P, 16, seed=12345 if t > 0 else None, stop=acfg.STOP)[0]
               for t in (acfg.TEMPERATURE, 0.0)]
        self.client.samples.clear()            # The original output example retains only the output of the repeat itself
        return out
