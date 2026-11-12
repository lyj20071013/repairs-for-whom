"""τ²-bench 的两个可选扩展（2026-10-29 写；第 6 篇）。默认都不启用，不改 τ² 的源码，只在本进程里打补丁。

1) 用户重复终止（repeat-stop）：用户连续 N 条文本消息（不含用户的工具调用）去首尾空白后完全相同，
   就当作用户在第 N 条处结束对话（termination_reason = USER_STOP），之后照常判分。
   - 这是对 τ² 的偏离，用了就要在论文里写明。
   - 与事后截断（repeatstop_posthoc.py）等价：触发前的对话完全一样，只是运行时省下后面的步数。
   - 先导 1 的事后检验（10-29）：telecom 本地 9B 用户的 45 个 max_steps 里只有 14 个是这种重复；
     其余 31 个是用户在不停自己排障、对话长度超过 100 步上限。所以这条规则只是省算力的后备，不能解决 telecom 的问题。

2) 命令行包装：python tau2_ext.py [--repeat-stop N] -- <tau2 的参数…>
   等同于 `tau2 <参数…>`，只是先装上补丁。run_tau2.sh 在 REPEAT_STOP>0 时用它。
"""
import sys

_INSTALLED = {"repeat_stop": 0}


def _user_text(m):
    return m.role == "user" and not getattr(m, "tool_calls", None)


def repeat_trigger(messages, n):
    """返回最后一条消息是否恰好构成"用户连续 n 条相同文本"。只看用户文本消息的序列。"""
    texts = [(m.content or "").strip() for m in messages if _user_text(m)]
    if len(texts) < n or not texts[-1]:
        return False
    return len(set(texts[-n:])) == 1


def install_repeat_stop(n=3):
    """给半双工 Orchestrator.step 打补丁。n <= 0 时什么也不做。可重复调用（只装一次）。"""
    if n <= 0:
        return
    from tau2.data_model.simulation import TerminationReason
    from tau2.orchestrator.orchestrator import Orchestrator, Role

    if _INSTALLED["repeat_stop"]:
        _INSTALLED["repeat_stop"] = n
        return
    orig = Orchestrator.step
    _INSTALLED["repeat_stop"] = n

    def step(self):
        orig(self)
        k = _INSTALLED["repeat_stop"]
        if (k > 0 and not self.done and self.from_role == Role.USER and self.to_role == Role.AGENT
                and self.trajectory and _user_text(self.trajectory[-1])
                and repeat_trigger(self.trajectory, k)):
            self.done = True
            self.termination_reason = TerminationReason.USER_STOP

    Orchestrator.step = step


def main(argv):
    n = 0
    if argv[:1] == ["--repeat-stop"]:
        n = int(argv[1])
        argv = argv[2:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    install_repeat_stop(n)
    print(f"[tau2_ext] 用户重复终止：{'开，N=' + str(n) if n > 0 else '关'}", flush=True)
    from tau2.cli import main as tau2_main
    sys.argv = ["tau2"] + argv
    return tau2_main()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
