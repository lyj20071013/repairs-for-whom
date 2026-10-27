import re


class _Tok:
    def __init__(self, t):
        self.text, self.pos_ = t, "NOUN"


def load(name):
    return lambda text: [_Tok(t) for t in re.findall(r"[A-Za-z]+", text or "")]
