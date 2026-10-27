"""pyserini Minimum replacement for（Achieved only WebShop used by LuceneSearcher.search / doc），Pure Python of BM25。

Only in the original version pyserini use if installation fails（WS_SEARCH=bm25）：hello Retrieval backend for report，The smoke reports and papers are written。
and Lucene The difference：Words are lowercase alphanumeric、Get rid of it Lucene Disable Words in English，No word drying；BM25 Parameter Same Anserini default（k1=0.9，b=0.4）。
"""
__version__ = "shim-bm25"
