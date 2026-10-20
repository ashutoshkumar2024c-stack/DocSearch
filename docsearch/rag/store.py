"""
Stage 4b: vector index + hybrid retrieval.

* Dense search: FAISS ``IndexFlatIP`` (exact cosine) when faiss is installed,
  otherwise the same exact search in NumPy. Results are identical; FAISS is just
  faster on large collections.
* Sparse search: BM25 over the same chunks. OCR'd text contains exact tokens
  (invoice numbers, drug names, clause numbers) that small embedding models
  blur; BM25 catches those.
* Hybrid: Reciprocal Rank Fusion (RRF) of the dense and BM25 rankings.
* Persistence: ``save(dir)`` / ``VectorStore.load(dir)`` - vectors.npy + chunks.jsonl
  + meta.json, so documents are embedded once, not on every app start
  (RAG-Chatbot rebuilt its FAISS index on every run).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*")
_STOP = set("a an the of to in on for and or is are was were be by with as at from that this "
            "it its into than then there their what which who whom how why when where does do "
            "did can could should would will shall may might".split())


def stem(t: str) -> str:
    """Light suffix stripping so 'discharged' matches 'discharge' and 'rates' matches 'rate'."""
    if len(t) <= 4 or not t.isalpha():
        return t
    for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ed", "es", "s", "e"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in (m.group(0).lower() for m in _TOKEN.finditer(text))
            if t not in _STOP and (len(t) > 1 or t.isdigit())]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=np.float32)
        self.avgdl = float(self.lens.mean()) if len(docs) else 0.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.tfs), dtype=np.float32)
        for i, tf in enumerate(self.tfs):
            denom_base = self.k1 * (1 - self.b + self.b * self.lens[i] / max(self.avgdl, 1e-9))
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + denom_base)
            out[i] = s
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    visual_rank: int | None = None


class VectorStore:
    def __init__(self, dim: int, embedder_name: str = ""):
        self.dim, self.embedder_name = dim, embedder_name
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None
        self._bm25: BM25 | None = None

    # ---- building -------------------------------------------------------- #
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Add chunks, skipping ids already present. Returns number added."""
        existing = {c.id for c in self.chunks}
        keep = [i for i, c in enumerate(chunks) if c.id not in existing]
        if not keep:
            return 0
        vectors = np.asarray(vectors, dtype=np.float32)[keep]
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != index dim {self.dim}")
        self.chunks.extend(chunks[i] for i in keep)
        self.vectors = np.vstack([self.vectors, vectors])
        self._faiss = self._bm25 = None           # rebuild lazily
        return len(keep)

    def remove_source(self, source: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.metadata.get("source") != source]
        removed = len(self.chunks) - len(keep)
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, self.dim), np.float32)
        self._faiss = self._bm25 = None
        return removed

    def sources(self) -> list[str]:
        return sorted({c.metadata.get("source", "?") for c in self.chunks})

    def __len__(self):
        return len(self.chunks)

    # ---- searching ------------------------------------------------------- #
    def _dense_scores(self, qvec: np.ndarray) -> np.ndarray:
        try:
            import faiss  # noqa: F401
        except ImportError:
            return self.vectors @ qvec.astype(np.float32)
        if self._faiss is None:
            import faiss
            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(self.vectors)
        scores, ids = self._faiss.search(qvec.reshape(1, -1).astype(np.float32), len(self.chunks))
        out = np.full(len(self.chunks), -np.inf, dtype=np.float32)
        out[ids[0]] = scores[0]
        return out

    def search(self, qvec: np.ndarray, query: str = "", k: int = 4, mode: str = "hybrid",
               rrf_k: int = 60, source_filter: list[str] | None = None,
               visual_ranking: list[str] | None = None) -> list[Hit]:
        """mode: 'dense' | 'bm25' | 'hybrid'.
        visual_ranking: chunk ids ranked by a vision retriever (ColPali), fused in hybrid mode."""
        if not self.chunks:
            return []
        n = len(self.chunks)
        allowed = np.ones(n, dtype=bool)
        if source_filter:
            allowed = np.array([c.metadata.get("source") in source_filter for c in self.chunks])

        dense = self._dense_scores(qvec) if mode in ("dense", "hybrid") else None
        sparse = None
        if mode in ("bm25", "hybrid") and query:
            if self._bm25 is None:
                self._bm25 = BM25([c.text for c in self.chunks])
            sparse = self._bm25.scores(query)

        def ranking(scores):
            order = [i for i in np.argsort(-scores) if allowed[i]]
            return {i: r for r, i in enumerate(order)}

        if mode == "dense" or sparse is None:
            order = self._unique([i for i in np.argsort(-dense) if allowed[i]], k)
            return [Hit(self.chunks[i], float(dense[i]), dense_rank=r) for r, i in enumerate(order)]
        if mode == "bm25":
            order = self._unique([i for i in np.argsort(-sparse) if allowed[i] and sparse[i] > 0], k)
            return [Hit(self.chunks[i], float(sparse[i]), sparse_rank=r) for r, i in enumerate(order)]

        dr, sr = ranking(dense), ranking(sparse)
        fused = {}
        for i in dr:
            fused[i] = 1 / (rrf_k + dr[i] + 1)
            if sparse[i] > 0:                    # only count BM25 when a query term matched
                fused[i] += 1 / (rrf_k + sr[i] + 1)
        vr: dict[int, int] = {}
        if visual_ranking:
            pos = {c.id: i for i, c in enumerate(self.chunks)}
            for r, cid in enumerate(visual_ranking):
                i = pos.get(cid)
                if i is not None and allowed[i] and i not in vr:
                    vr[i] = r
                    fused[i] = fused.get(i, 0.0) + 1 / (rrf_k + r + 1)
        top = self._unique(sorted(fused, key=fused.get, reverse=True), k)
        return [Hit(self.chunks[i], fused[i], dr.get(i), sr.get(i) if sparse[i] > 0 else None,
                    vr.get(i)) for i in top]

    def _unique(self, order, k: int) -> list[int]:
        """First k indices whose passage text is not a repeat (identical tables / slides)."""
        out, seen = [], set()
        for i in order:
            key = re.sub(r"\W+", " ", self.chunks[i].text.split("\n", 1)[-1]).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(i)
            if len(out) == k:
                break
        return out

    # ---- persistence ----------------------------------------------------- #
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps({"id": c.id, "text": c.text, "metadata": c.metadata},
                                    ensure_ascii=False) + "\n")
        (d / "meta.json").write_text(json.dumps({"dim": self.dim, "embedder": self.embedder_name,
                                                 "count": len(self.chunks)}, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        store = cls(meta["dim"], meta.get("embedder", ""))
        with open(d / "chunks.jsonl", encoding="utf-8") as fh:
            store.chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]
        store.vectors = np.load(d / "vectors.npy").astype(np.float32)
        if len(store.chunks) != len(store.vectors):
            raise ValueError(f"corrupt index in {d}: chunk/vector count mismatch")
        return store

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / "meta.json").exists()
