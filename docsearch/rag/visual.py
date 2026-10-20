"""
Optional vision-native retrieval (ColPali), applied selectively.

ColPali (ICLR 2025) embeds whole page images with a vision-language model and
matches queries by late interaction. It avoids OCR errors on visual pages, but
it stores ~1,000 vectors per page and needs a GPU, which are the gaps noted in the
literature survey.

This project's middle ground: the router already knows which pages are
visual (route "visual" or "hybrid", or images classified as graphic or photo). Only those
pages are embedded with ColPali; text pages stay on the cheap OCR/text path.
The ColPali ranking is fused with the dense and BM25 rankings by RRF. The
`stats()` method reports how many pages were indexed visually, so the storage
saving versus "ColPali on every page" can be reported.

Enable with VISUAL_RETRIEVER=colpali (needs `pip install colpali-engine torch` and,
in practice, a CUDA GPU). Disabled by default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from PIL import Image

VISUAL_ROUTES = {"visual", "hybrid"}


class VisualRetriever(Protocol):
    name: str

    def add_pages(self, items: list[tuple[str, Image.Image]]) -> int: ...
    def remove_source(self, source: str) -> int: ...
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...
    def save(self, directory: str | Path) -> None: ...
    def stats(self) -> dict: ...


def page_key(source: str, page: int | None) -> str:
    return f"{source}::{page if page is not None else 0}"


def needs_visual_index(page) -> bool:
    """Should this PageText also be indexed by the vision retriever?"""
    if page.route in VISUAL_ROUTES:
        return True
    plan = getattr(page, "plan", None)
    kind = (plan.evidence.get("image", {}) or {}).get("kind") if plan else None
    return kind in ("graphic", "photo")


class ColPaliRetriever:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.name = model_name or os.getenv("COLPALI_MODEL", "vidore/colpali-v1.3")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._torch = torch
        self.model = ColPali.from_pretrained(self.name, torch_dtype=dtype, device_map=self.device).eval()
        self.processor = ColPaliProcessor.from_pretrained(self.name)
        self.keys: list[str] = []
        self.embeddings: list = []          # one (n_patches, dim) tensor per page

    def add_pages(self, items, batch_size: int = 2) -> int:
        torch = self._torch
        new = [(k, img) for k, img in items if k not in self.keys]
        for i in range(0, len(new), batch_size):
            batch = new[i:i + batch_size]
            inputs = self.processor.process_images([img for _, img in batch]).to(self.model.device)
            with torch.no_grad():
                out = self.model(**inputs)
            for (k, _), emb in zip(batch, out):
                self.keys.append(k)
                self.embeddings.append(emb.to("cpu", dtype=torch.float32))
        return len(new)

    def remove_source(self, source: str) -> int:
        keep = [i for i, k in enumerate(self.keys) if not k.startswith(source + "::")]
        removed = len(self.keys) - len(keep)
        self.keys = [self.keys[i] for i in keep]
        self.embeddings = [self.embeddings[i] for i in keep]
        return removed

    def search(self, query: str, k: int = 10):
        if not self.keys:
            return []
        torch = self._torch
        q = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            qe = self.model(**q).to("cpu", dtype=torch.float32)
        scores = self.processor.score_multi_vector(qe, self.embeddings)[0]
        order = scores.argsort(descending=True)[:k].tolist()
        return [(self.keys[i], float(scores[i])) for i in order]

    def save(self, directory):
        d = Path(directory) / "visual"
        d.mkdir(parents=True, exist_ok=True)
        self._torch.save({"keys": self.keys, "embeddings": self.embeddings}, d / "colpali.pt")
        (d / "meta.json").write_text(json.dumps({"model": self.name, "pages": len(self.keys)}))

    def load(self, directory):
        f = Path(directory) / "visual" / "colpali.pt"
        if f.exists():
            data = self._torch.load(f)
            self.keys, self.embeddings = data["keys"], data["embeddings"]
        return self

    def stats(self) -> dict:
        vectors = sum(int(e.shape[0]) for e in self.embeddings)
        return {"retriever": self.name, "pages": len(self.keys), "vectors": vectors,
                "approx_mb": round(vectors * 128 * 4 / 1e6, 1)}


def get_visual_retriever(name: str | None = None, index_dir=None):
    name = (name or os.getenv("VISUAL_RETRIEVER", "none")).lower()
    if name in ("", "none", "off", "false"):
        return None
    if name == "colpali":
        r = ColPaliRetriever()
        return r.load(index_dir) if index_dir else r
    raise ValueError(f"Unknown visual retriever {name!r} (none | colpali)")
