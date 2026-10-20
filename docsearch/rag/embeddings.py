"""
Stage 4a: embeddings. Every embedder returns L2-normalised float32 vectors, so
inner product = cosine similarity.

* ``MiniLMEmbedder``  - sentence-transformers/all-MiniLM-L6-v2 (384-d), the model
  used in RAG-Chatbot and named in the project report. Runs on CPU.
* ``OllamaEmbedder``  - nomic-embed-text through a local Ollama server (from V.A.U.L.T.).
* ``HashingEmbedder`` - no model download, lexical only. For tests and offline demos;
  NOT semantic. Don't report results from it as embedding results.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 loaded with ``transformers`` + ``torch`` directly (mean pooling,
    exactly what sentence-transformers does). It deliberately avoids importing
    sentence-transformers, which pulls in scikit-learn; on locked-down Windows PCs the
    scikit-learn DLLs are often blocked by an Application Control policy."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str | None = None, batch_size: int = 32, max_tokens: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.name = model
        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model).eval()
        if device:
            self._model.to(device)
        self._device = device
        self.dim = int(self._model.config.hidden_size)
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    def embed_documents(self, texts):
        torch, texts, out = self._torch, list(texts), []
        for i in range(0, len(texts), self.batch_size):
            enc = self._tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                            max_length=self.max_tokens, return_tensors="pt")
            if self._device:
                enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(pooled.cpu().numpy())
        return _normalise(np.vstack(out)) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class OllamaEmbedder:
    def __init__(self, model: str | None = None, host: str | None = None):
        import ollama

        self.name = model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self._client = ollama.Client(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.dim = len(self._embed(["dimension probe"])[0])

    def _embed(self, texts):
        resp = self._client.embed(model=self.name, input=list(texts))
        vecs = getattr(resp, "embeddings", None)
        if vecs is None and isinstance(resp, dict):
            vecs = resp.get("embeddings")
        if not vecs:
            raise RuntimeError("Ollama returned no embeddings")
        return vecs

    def embed_documents(self, texts, batch: int = 32):
        out = []
        for i in range(0, len(texts), batch):
            out.extend(self._embed(texts[i:i + batch]))
        return _normalise(np.array(out))

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class HashingEmbedder:
    """Word + character n-gram hashing. Deterministic, instant, no downloads, numpy only
    (no scikit-learn)."""

    def __init__(self, dim: int = 4096):
        self.name, self.dim = f"hashing-{dim}", dim

    @staticmethod
    def _tokens(text: str):
        import re
        words = re.findall(r"(?u)\b\w\w+\b", text.lower())
        for w in words:
            yield w, 2.0
        for a, b in zip(words, words[1:]):
            yield f"{a} {b}", 2.0
        for w in words:
            p = f" {w} "
            for n in (3, 4):
                for i in range(len(p) - n + 1):
                    yield "#" + p[i:i + n], 0.5

    def _vec(self, text: str) -> np.ndarray:
        import zlib
        v = np.zeros(self.dim, dtype=np.float32)
        for tok, w in self._tokens(text):
            v[zlib.crc32(tok.encode("utf-8")) % self.dim] += w
        return np.sqrt(v)                  # sqrt damps very frequent terms

    def embed_documents(self, texts):
        rows = [self._vec(t) for t in texts]
        return _normalise(np.vstack(rows)) if rows else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_embedder(name: str | None = None) -> Embedder:
    name = (name or os.getenv("EMBEDDER", "minilm")).lower()
    if name in ("minilm", "sentence-transformers", "st"):
        # ST_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 for non-English documents
        import sys
        import traceback
        # scikit-learn is optional for everything we use. If its DLLs are blocked (Windows
        # Application Control) or it is broken, make "import sklearn" fail cleanly so that
        # libraries that probe for it just carry on without it.
        saved = {k: sys.modules.get(k, False) for k in ("sklearn",)}
        for k in saved:
            if saved[k] is False:
                sys.modules[k] = None
        try:
            return MiniLMEmbedder(os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        except BaseException as exc:       # blocked DLL, missing package, no network on first run
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            frames = [f for f in traceback.extract_tb(exc.__traceback__)][-4:]
            where = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}" for f in reversed(frames))
            emb = HashingEmbedder()
            emb.note = (f"MiniLM could not be loaded ({type(exc).__name__}: {str(exc)[:160]}) "
                        f"[{where}]. Using the offline keyword-style embedder instead; answers "
                        "still work but semantic matching is weaker.")
            return emb
        finally:
            for k, v in saved.items():
                if v is False and sys.modules.get(k, 1) is None:
                    del sys.modules[k]
    if name in ("ollama", "nomic"):
        return OllamaEmbedder()
    if name in ("hashing", "hash", "offline"):
        return HashingEmbedder()
    raise ValueError(f"Unknown embedder {name!r} (minilm | ollama | hashing)")