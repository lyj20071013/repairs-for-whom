"""LuceneSearcher pure Python Substitution：Read it search_engine/resources*/documents.jsonl（Same document for indexing）。"""
import json
import math
import os
import re
from collections import Counter

# Lucene EnglishAnalyzer Default deactivated words
STOP = set("a an and are as at be but by for if in into is it no not of on or such that the their then there these "
           "they this to was will with".split())
K1, B = 0.9, 0.4


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in STOP]


class _Hit:
    __slots__ = ("docid", "score")

    def __init__(self, docid, score):
        self.docid, self.score = docid, score


class _Doc:
    def __init__(self, raw):
        self._raw = raw

    def raw(self):
        return self._raw


class LuceneSearcher:
    def __init__(self, index_dir):
        d = os.path.normpath(index_dir)
        res = os.path.join(os.path.dirname(d), os.path.basename(d).replace("indexes", "resources"))
        path = os.path.join(res, "documents.jsonl")
        self.ids, self.raws, self.tf, self.len = [], {}, [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                toks = tokenize(doc["contents"])
                self.ids.append(doc["id"])
                self.raws[doc["id"]] = line.strip()
                self.tf.append(Counter(toks))
                self.len.append(len(toks))
        self.N = len(self.ids)
        self.avgdl = sum(self.len) / max(1, self.N)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, q, k=10):
        terms = tokenize(q)
        scores = []
        for i, c in enumerate(self.tf):
            s = 0.0
            for t in terms:
                f = c.get(t)
                if f:
                    s += self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * self.len[i] / self.avgdl))
            if s > 0:
                scores.append((-s, i))
        scores.sort()
        return [_Hit(self.ids[i], -s) for s, i in scores[:k]]

    def doc(self, docid):
        return _Doc(self.raws[docid])
