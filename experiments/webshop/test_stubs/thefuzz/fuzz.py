import difflib
import re


def token_set_ratio(a, b):
    sa = " ".join(sorted(set(re.findall(r"\w+", str(a).lower()))))
    sb = " ".join(sorted(set(re.findall(r"\w+", str(b).lower()))))
    return int(round(100 * difflib.SequenceMatcher(None, sa, sb).ratio()))
