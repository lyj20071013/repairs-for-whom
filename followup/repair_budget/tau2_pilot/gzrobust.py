"""稳健地读"追加写的 gzip jsonl"（2026-11-01 写）。
续跑把每次运行的新消息追加成一个 gzip 块；进程被杀时会留下写了一半的块，之后的运行再追加新块。
标准的 gzip 顺序读取在坏块处就会报错（EOFError 或 zlib.error: invalid block type），后面的块全读不出来。
做法：找出所有 gzip 块头（1f 8b 08）；从一个块头开始解压到下一个块头，若还没到块尾就往后延到再下一个块头
（中间那个是压缩数据里碰巧出现的假块头），直到块完整、或延长后报错（说明碰到了下一个真块，当前块是写了一半的）。
写了一半的块里解出来的完整行照用。只保留以换行结尾、能解析的 JSON 行。
"""
import json
import re
import zlib

HEADER = re.compile(rb"\x1f\x8b\x08")


def read_records(path):
    """返回 (记录列表, 是否完好)。完好 = 所有块都完整、没有丢弃任何行。"""
    raw = open(path, "rb").read()
    starts = [m.start() for m in HEADER.finditer(raw)]
    n = len(starts)
    recs, ok = [], True
    i = 0
    while i < n:
        s = starts[i]
        best, best_j, complete = None, None, False
        j = i + 1
        while True:
            e = starts[j] if j < n else len(raw)
            d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            try:
                out = d.decompress(raw[s:e])
            except zlib.error:
                break
            best, best_j, complete = out, j, d.eof
            if d.eof or j >= n:
                break
            j += 1
        if best is None:                 # 假块头，或这一段从头就坏
            i += 1
            continue
        lines = best.decode("utf-8", errors="replace").split("\n")
        if not complete:
            ok = False
            lines = lines[:-1]           # 写了一半：最后一行可能不完整
        for l in lines:
            if not l.strip():
                continue
            try:
                recs.append(json.loads(l))
            except json.JSONDecodeError:
                ok = False
        i = best_j if best_j is not None else i + 1
    return recs, ok
