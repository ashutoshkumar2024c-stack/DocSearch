"""
Stage 5a: language models.

* ``MistralLLM`` - Mistral's hosted API (what RAG-Chatbot used). Called over plain
  HTTPS, so there is no ``langchain-mistralai`` version pin to fight.
  RAG-Chatbot used ``mistral-tiny`` (a retired alias) at temperature 0.7; for
  grounded answers we default to ``mistral-small-latest`` at temperature 0.1.
* ``OllamaLLM``  - a local model through Ollama (default ``qwen3:4b`` from
  V.A.U.L.T.). Keeps everything on one machine, with no data leaving it.

Pick with LLM_PROVIDER=mistral|ollama (default: mistral if MISTRAL_API_KEY is set,
otherwise ollama).
"""

from __future__ import annotations

import math
import os
import re
import time
from typing import Protocol


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], temperature: float | None = None,
             max_tokens: int = 800) -> str: ...


def strip_thinking(text: str) -> str:
    text = re.sub(r"(?s)<think>.*?</think>", "", text or "")
    return text.strip()


def clean_key(key: str | None) -> str | None:
    """Keys pasted into .env often carry quotes, spaces or a 'Bearer ' prefix."""
    if not key:
        return key
    key = key.strip().strip("\"'").strip()
    return re.sub(r"^(?i:bearer)\s+", "", key)


def gemini_keys(extra: str | None = None) -> list[str]:
    """Every Gemini key configured: `extra`, GEMINI_API_KEY, GOOGLE_API_KEY and the
    comma-separated GEMINI_API_KEYS. With several keys (ideally from different Google projects,
    each has its own free quota) a 429 on one key just moves on to the next."""
    raw = [extra, os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY")]
    raw += (os.getenv("GEMINI_API_KEYS") or "").split(",")
    out = []
    for k in raw:
        k = clean_key(k)
        if k and k not in out:
            out.append(k)
    return out


class MistralLLM:
    URL = "https://api.mistral.ai/v1/chat/completions"
    _blocked: dict = {}                 # api key -> time until which we skip calling (after a 429)

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.1, timeout: float = 60):
        self.model = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        self.api_key = clean_key(api_key or os.getenv("MISTRAL_API_KEY"))
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set (put it in a .env file)")
        self.name = f"mistral:{self.model}"
        self.temperature, self.timeout = temperature, timeout

    def chat(self, messages, temperature=None, max_tokens=800):
        import requests

        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": self.temperature if temperature is None else temperature}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        left = self._blocked.get(self.api_key, 0) - time.time()
        if left > 0:                    # recently rate-limited: answer offline at once instead of waiting
            raise RuntimeError(f"Mistral is rate-limiting this key; using quoted passages for about "
                               f"{int(left) + 1} more seconds.")
        for attempt in range(2):        # one short retry only: a blocked key should fail fast
            resp = requests.post(self.URL, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code not in (429, 500, 502, 503) or attempt == 1:
                break
            try:
                wait = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                wait = 1.5
            time.sleep(min(wait, 3))
        if resp.status_code == 429:
            self._blocked[self.api_key] = time.time() + 60
        if resp.status_code in (401, 403):
            raise RuntimeError("Mistral rejected the API key (401). Check that MISTRAL_API_KEY is a valid, "
                               "active Mistral key (console.mistral.ai > API keys), with no quotes or spaces.")
        if resp.status_code == 429:
            raise RuntimeError("Mistral rate limit reached (too many requests on this key). "
                               "Wait a minute, try another model (e.g. open-mistral-nemo) or use a key on a higher plan.")
        if resp.status_code != 200:
            raise RuntimeError(f"Mistral API error {resp.status_code}: {resp.text[:300]}")
        return strip_thinking(resp.json()["choices"][0]["message"]["content"])


_GEMINI_DEFAULT = "gemini-flash-latest"          # Google's alias: always the current Flash model
_gemini_models: dict[str, list[str]] = {}


def gemini_candidates(api_key: str) -> list[str]:
    """Models this key can call, best first: newest stable Flash, then newest Flash-Lite. Used when the
    configured model is retired (404) or overloaded (503), so one bad model never breaks the app."""
    if api_key in _gemini_models:
        return _gemini_models[api_key]
    import requests
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                         headers={"x-goog-api-key": api_key}, timeout=15)
        models = r.json().get("models", []) if r.status_code == 200 else []
    except Exception:
        return []
    flash, lite = [], []
    for m in models:
        name = (m.get("name") or "").replace("models/", "")
        if "generateContent" not in (m.get("supportedGenerationMethods") or []) or "flash" not in name:
            continue
        if any(x in name for x in ("image", "tts", "live", "audio", "embed", "thinking", "exp", "preview", "8b")):
            continue
        ver = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        key = (float(ver.group(1)) if ver else 0.0, "latest" not in name)
        (lite if "lite" in name else flash).append((key, name))
    ordered = [n for _, n in sorted(flash, reverse=True)] + [n for _, n in sorted(lite, reverse=True)]
    if ordered:
        _gemini_models[api_key] = ordered
    return ordered


def gemini_pick_model(api_key: str) -> str | None:
    c = gemini_candidates(api_key)
    return c[0] if c else None


def gemini_post(api_key: str, model: str, payload: dict, timeout: float, retries: int = 2,
                wait: float = 1.5, max_models: int = 4):
    """POST generateContent. A busy model (429/500/503) is retried with a growing wait, then the next
    available Flash model is tried; a retired model (404) is skipped at once. Returns (response, model_used)."""
    import requests
    url = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    order, resp, used, i = [model], None, model, 0
    extended = False
    while i < len(order) and i < max_models + 1:
        used = order[i]
        for attempt in range(retries):
            resp = requests.post(url.format(used), headers=headers, json=payload, timeout=timeout)
            if resp.status_code not in (429, 500, 503):
                break
            if attempt < retries - 1:
                time.sleep(wait * (attempt + 1))
        if resp.status_code == 200 or (resp.status_code not in (404, 429, 500, 503)):
            return resp, used
        if not extended:                                  # first failure: learn which other models exist
            others = [m for m in gemini_candidates(api_key) if m != model]
            lite = [m for m in others if "lite" in m] or ["gemini-flash-lite-latest"]
            # free-tier quotas are per model: try a Flash-Lite model early, it usually has the largest quota
            order += lite[:1] + [m for m in others if m not in lite[:1]]
            extended = True
        i += 1
    return resp, used


class GeminiLLM:
    """Google Gemini through its REST API (the GOOGLE_API_KEY from Google AI Studio works)."""

    BASE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    _blocked: dict = {}

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.1, timeout: float = 60):
        self.model = model or os.getenv("GEMINI_MODEL", _GEMINI_DEFAULT)
        self.api_key = clean_key(api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        self.name = f"gemini:{self.model}"
        self.temperature, self.timeout = temperature, timeout

    def chat(self, messages, temperature=None, max_tokens=800):
        import requests

        left = self._blocked.get(self.api_key, 0) - time.time()
        if left > 0:
            raise RuntimeError(f"Gemini is rate-limiting this key; skipping for about {int(left) + 1}s.")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        gen = {"temperature": self.temperature if temperature is None else temperature,
               "maxOutputTokens": max(max_tokens, 1024)}
        if "2.5-flash" in self.model:
            gen["thinkingConfig"] = {"thinkingBudget": 0}        # answer directly; thinking would eat the token budget
        else:
            gen["maxOutputTokens"] = max(max_tokens, 4096)       # newer models may "think" first
        payload = {"contents": contents, "generationConfig": gen}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        resp, used = gemini_post(self.api_key, self.model, payload, self.timeout)
        if resp.status_code == 400 and "thinking" in resp.text.lower() and "thinkingConfig" in gen:
            gen.pop("thinkingConfig", None)
            resp, used = gemini_post(self.api_key, used, payload, self.timeout)
        if used != self.model:
            self.model, self.name = used, f"gemini:{used}"
        if resp.status_code == 429:
            self._blocked[self.api_key] = time.time() + 60
            raise RuntimeError("Gemini rate limit reached.")
        if resp.status_code in (400, 401, 403):
            raise RuntimeError(f"Gemini rejected the request ({resp.status_code}): {resp.text[:200]}")
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
        def text_of(d):
            try:
                return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]).strip()
            except (KeyError, IndexError, TypeError):
                return ""
        data = resp.json()
        text = text_of(data)
        if not text:                                         # newer models can spend the whole budget "thinking"
            gen["maxOutputTokens"] = 8192
            resp2, _ = gemini_post(self.api_key, self.model, payload, self.timeout)
            if resp2.status_code == 200:
                data = resp2.json()
                text = text_of(data)
        if not text:
            raise RuntimeError(f"Gemini returned no answer: {str(data)[:200]}")
        return strip_thinking(text)


class FallbackLLM:
    """Tries each provider in order and moves on when one fails (rate limit, bad key, no network).
    A provider that failed is skipped for a while so later questions are not slowed down.
    The last provider is normally the offline ExtractiveLLM, which cannot fail."""

    def __init__(self, providers: list, cooldown: float = 60.0):
        self.providers, self.cooldown = providers, cooldown
        self.skip_until: dict[int, float] = {}
        self.errors: dict[str, str] = {}
        self.last: str = providers[0].name

    @property
    def name(self) -> str:
        return self.last

    @property
    def model(self) -> str:
        return getattr(self.providers[0], "model", "")

    def chat(self, messages, temperature=None, max_tokens=800):
        for i, p in enumerate(self.providers):
            last = i == len(self.providers) - 1
            if not last and self.skip_until.get(i, 0) > time.time():
                continue
            try:
                out = p.chat(messages, temperature=temperature, max_tokens=max_tokens)
                if not out and not last:
                    raise RuntimeError("empty answer")
                self.last = p.name
                self.errors.pop(p.name, None)
                return out
            except Exception as exc:
                self.errors[p.name] = str(exc)
                self.skip_until[i] = time.time() + self.cooldown
                if last:
                    raise
        raise RuntimeError("no language model available")


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None,
                 temperature: float = 0.1):
        import ollama

        self.model = model or os.getenv("LLM_MODEL", "qwen3:4b")
        self.name = f"ollama:{self.model}"
        self._client = ollama.Client(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.temperature = temperature

    def chat(self, messages, temperature=None, max_tokens=800):
        resp = self._client.chat(
            model=self.model, messages=messages, think=False,
            options={"temperature": self.temperature if temperature is None else temperature,
                     "num_predict": max_tokens},
        )
        msg = getattr(resp, "message", None)
        if msg is None and isinstance(resp, dict):
            msg = resp.get("message")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        return strip_thinking(content)


_FILLER = {"more", "other", "others", "another", "any", "all", "list", "tell", "explain", "give", "show",
           "describe", "about", "please", "mention", "name", "detail", "details", "briefly", "also",
           "doc", "document", "pdf", "file", "module", "slide", "slides", "mean", "meant", "define"}
_LIST_Q = re.compile(r"\b(types?|kinds?|list|categories|category|levels?|steps?|stages?|phases?|examples?|"
                     r"advantages?|disadvantages?|benefits?|principles?|characteristics?|features?|"
                     r"components?|elements?|what are|which are|how many)\b", re.I)
_BULLET = re.compile(r"^\s*([➢▪•◦●■□◆▶►\-*–]|\d+[.)]|[a-z][.)])\s*")
_NUMBER_ABBR = re.compile(r"\b(?:no\.?|nos\.?|num|id|ref)\b\s*[:#.-]", re.I)
_SOURCE = re.compile(r"\[(\d+)\] \((.*?)\)\n(.*?)(?=\n\n\[\d+\] \(|\n\nQuestion:)", re.S)


class ExtractiveLLM:
    """
    No-model fallback: answers by quoting the passages that best match the question,
    with citations. It cannot reason, paraphrase or summarise freely, but it never
    invents anything, and it keeps the app usable with no API key and no Ollama.

    It recognises three kinds of question:
      overview  ("what is this document about?")  -> title and section list from the outline
      list      ("types of cohesion", "what are the steps")  -> all matching table rows / bullets
      factual   -> the best-matching whole sentences or bullet points
    """

    name = "extractive (no LLM)"

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _sources(content: str) -> list[dict]:
        out = []
        for m in _SOURCE.finditer(content):
            n, label, body = int(m.group(1)), m.group(2), m.group(3)
            sec = ""
            sm = re.match(r"^\[([^\]]*)\]\n", body)
            if sm:
                sec, body = sm.group(1), body[sm.end():]
            kind = ("outline" if "document outline" in label or body.startswith("Document overview:")
                    else "table" if ", table" in label else "figure" if "figure" in label else "text")
            out.append({"n": n, "label": label, "section": sec, "body": body.strip(), "kind": kind})
        return out

    @staticmethod
    def _blocks(body: str) -> list[str]:
        """Join wrapped PDF lines back into bullet points / sentences."""
        blocks, cur = [], ""
        for line in body.splitlines():
            ln = line.strip()
            if not ln:
                if cur:
                    blocks.append(cur)
                cur = ""
                continue
            if _BULLET.match(ln) and cur:
                blocks.append(cur)
                cur = ln
            else:
                cur = f"{cur} {ln}".strip() if cur else ln
        if cur:
            blocks.append(cur)
        units = []
        for b in blocks:
            if len(b) > 320:
                units += [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z➢▪•])", b) if x.strip()]
            else:
                units.append(b)
        return units

    @staticmethod
    def _table(body: str):
        rows = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("|")]
        rows = [r for r in rows if not re.fullmatch(r"\|?[\s:\-|]+\|?", r)]
        if len(rows) < 2:
            return None, []
        cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]   # noqa: E731
        return cells(rows[0]), [cells(r) for r in rows[1:]]

    @staticmethod
    def _outline_children(outline: list[dict], q: set, tokenize) -> str | None:
        """For 'what are the golden rules of X' style questions: find the section whose
        title matches and list its sub-headings (e.g. the three golden rules)."""
        listy = {"type", "kind", "list", "rule", "principl", "step", "stage", "phas", "level",
                 "categori", "advantag", "disadvantag", "benefit", "characteristic", "featur",
                 "component", "model", "exampl", "many"}
        key = {t for t in q if t not in listy} or q
        items = []
        for src in outline:
            for ln in src["body"].splitlines():
                m = re.match(r"^(\s*)- (.*?)(?: \(p\. (\d+)\))?$", ln)
                if m:
                    items.append((len(m.group(1)) // 2, m.group(2), m.group(3), src["n"]))
        best, best_cov = None, 0.0
        for i, (lvl, text, page, n) in enumerate(items):
            toks = set(tokenize(text))
            cov = len(key & toks) / len(key)
            if cov > best_cov:
                best, best_cov = i, cov
        if best is None or best_cov < 0.6:
            return None
        lvl0, parent, page0, n0 = items[best]
        kids = []
        prefix = re.split(r"\s+[–—-]\s+", parent)[0]
        for lvl, text, page, n in items[best + 1:]:
            if lvl <= lvl0 and (lvl < lvl0 or re.split(r"\s+[–—-]\s+", text)[0] == prefix):
                break                                   # next sibling section
            if lvl == lvl0 or lvl == lvl0 + 1:
                kids.append((text, page))
            if len(kids) >= 15:
                break
        if len(kids) < 2:
            return None
        return (f"**{parent}**" + (f" (p. {page0})" if page0 else "") + f" [{n0}]:\n\n"
                + "\n".join(f"- {t}" + (f" (p. {p})" if p else "") for t, p in kids))

    @staticmethod
    def _abbreviation_answer(question, sources) -> str | None:
        """'Full form of CIDR?' -> find 'CIDR (Classless Inter-Domain Routing)' or the reverse in the text."""
        if not re.search(r"\b(full[- ]?form|stands?\s+for|abbreviation|acronym|expand|meaning of)\b", question, re.I):
            return None
        m = re.search(r"\b([A-Z][A-Za-z]{1,7})\b(?=\W*$|\s+(?:in|of|used|mean)\b)", question) or re.search(r"\b([A-Z]{2,8})\b", question)
        if not m:
            return None
        acr = m.group(1)
        for s in sources:
            body = re.sub(r"\s+", " ", s["body"])
            f = re.search(rf"\b{re.escape(acr)}\b\s*[\(\[:\-–]\s*([A-Z][\w\- ]{{4,80}}?)\s*[\)\]]", body)
            if f:
                return f"**{acr}** stands for **{f.group(1).strip()}** [{s['n']}]."
            g = re.search(rf"([A-Z][\w\- ]{{4,80}}?)\s*\(\s*{re.escape(acr)}\s*\)", body)
            if g:
                return f"**{acr}** stands for **{g.group(1).strip()}** [{s['n']}]."
        return None

    @staticmethod
    def _structure_answer(sources) -> str | None:
        """List the tables and figures among the passages, with what each shows."""
        figs = [s for s in sources if s["kind"] == "figure"]
        tabs = [s for s in sources if s["kind"] == "table"]
        # a bracketed "[Diagram ...]" line inside text also counts as a figure
        for s in sources:
            if s["kind"] == "text":
                for m in re.finditer(r"^\[(?:Diagram|Figure|Chart|Graph|Image|Illustration|Table)[^\]]{5,300}\]", s["body"], re.M | re.I):
                    figs.append({**s, "body": m.group(0).strip("[]")})
        if not figs and not tabs:
            return "I did not find any tables or figures in the pages I looked at."
        pg = lambda s: (re.search(r"page (\d+)", s["label"]) or [None, None])[1]    # noqa: E731
        out = []
        if figs:
            out.append(f"**Figures / diagrams ({len(figs)}):**")
            for s in figs[:6]:
                txt = re.sub(r"\s+", " ", s["body"]).strip()
                out.append(f"- {txt[:220]}{'…' if len(txt) > 220 else ''}" + (f" (page {pg(s)})" if pg(s) else "") + f" [{s['n']}]")
        if tabs:
            out.append(f"**Tables ({len(tabs)}):**")
            for s in tabs[:6]:
                head = next((ln for ln in s["body"].splitlines() if ln.strip().startswith("|")), "")
                cols = [c.strip() for c in head.strip().strip("|").split("|") if c.strip()]
                out.append(f"- columns: {', '.join(cols[:8])}" + (f" (page {pg(s)})" if pg(s) else "") + f" [{s['n']}]")
        return "\n".join(out)

    @staticmethod
    def _chart_pairs(body: str):
        """(title, [(label, value_text, number)]) from an OCR'd chart's 'Chart values:' line."""
        cv = re.search(r"^Chart values:\s*(.+)$", body, flags=re.M)
        if not cv:
            return None, []
        title = next((ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("[")), "")
        pairs = []
        for pair in cv.group(1).split(";"):
            if ":" in pair:
                label, value = (x.strip() for x in pair.split(":", 1))
                m = re.search(r"-?\d[\d,]*\.?\d*", value)
                if label and m:
                    try:
                        pairs.append((label, value, float(m.group(0).replace(",", ""))))
                    except ValueError:
                        pass
        return title, pairs

    @classmethod
    def _describe_without_headings(cls, sources, doc_name):
        """Overview of a document that has no headings (a chart image, a receipt, a short note)."""
        for s in sources:
            if s["kind"] == "outline":
                continue
            title, pairs = cls._chart_pairs(s["body"])
            name = f"**{doc_name}**" if doc_name else "This document"
            if pairs:
                vals = ", ".join(f"{l} {v}" for l, v, _ in pairs)
                hi = max(pairs, key=lambda x: x[2])
                lo = min(pairs, key=lambda x: x[2])
                return (f"{name} is a chart titled **{title}**. It shows {len(pairs)} values: {vals}. "
                        f"The highest is {hi[0]} ({hi[1]}) and the lowest is {lo[0]} ({lo[1]}) [{s['n']}].")
            blocks = [b for b in cls._blocks(s["body"]) if len(b) >= 12][:4]
            if blocks:
                return f"{name} contains [{s['n']}]:\n\n" + "\n".join(f"- {b}" for b in blocks)
        return None

    @classmethod
    def _chart_answer(cls, question, q, content_sources, tokenize):
        """Answer lookups, extremes, totals and listings straight from chart value pairs."""
        ql = question.lower()
        high = re.search(r"\b(highest|most|largest|biggest|maximum|max|top|greatest|more)\b", ql)
        low = re.search(r"\b(lowest|least|smallest|minimum|min|fewest|lowest|less)\b", ql)
        total = re.search(r"\b(total|sum|altogether|combined)\b", ql)
        avg = re.search(r"\b(average|mean)\b", ql)
        every = re.search(r"\b(all|each|every|list|values|compare|comparison|show|rank|ranking|order)\b", ql)
        for s in content_sources:
            title, pairs = cls._chart_pairs(s["body"])
            if not pairs:
                continue
            label_toks = [(set(tokenize(l)), (l, v, n)) for l, v, n in pairs]
            title_toks = set(tokenize(title))
            hit = [p for t, p in label_toks if t and t <= set(tokenize(question))]
            ref = f"[{s['n']}]"
            unit = ""
            if hit and not (high or low or total or avg):
                l, v, _ = hit[0]
                return f"**{l}: {v}** ({title}) {ref}"
            on_topic = bool(q & title_toks) or (bool(every) and all(t.startswith(("valu", "chart", "graph", "data", "figure", "bar", "number")) for t in q))
            if not (on_topic or hit):
                continue
            if high or low:
                p = max(pairs, key=lambda x: x[2]) if high and not low else min(pairs, key=lambda x: x[2])
                word = "highest" if high and not low else "lowest"
                return f"The {word} value in **{title}** is **{p[0]}: {p[1]}** {ref}."
            if total:
                return f"The total across all {len(pairs)} values in **{title}** is **{sum(n for _, _, n in pairs):g}** {ref}."
            if avg:
                return f"The average across {len(pairs)} values in **{title}** is **{sum(n for _, _, n in pairs) / len(pairs):.1f}** {ref}."
            if every:
                ranked = sorted(pairs, key=lambda x: -x[2])
                return (f"**{title}** {ref}:\n\n| Item | Value |\n|---|---|\n"
                        + "\n".join(f"| {l} | {v} |" for l, v, _ in ranked))
        return None

    # -- main -----------------------------------------------------------------
    def chat(self, messages, temperature=None, max_tokens=800):
        from .answer import NOT_FOUND
        from .store import tokenize

        content = messages[-1]["content"]
        if content.startswith("Rewrite the follow-up"):
            m = re.search(r"Follow-up question: (.*)", content)
            prev = re.findall(r"User: (.*)", content)
            follow = m.group(1).strip() if m else ""
            refers = re.search(r"\b(it|its|that|this|those|these|they|them|their|he|she|his|her|same|"
                               r"above|previous|more|other|another|else)\b", follow, re.I) \
                or not [t for t in tokenize(follow) if t not in _FILLER]
            return f"{follow} {prev[-1]}" if prev and refers else follow

        from ..pipeline import is_overview_question

        question = content.split("Question:")[-1].strip()
        sources = self._sources(content)
        if not sources:
            return NOT_FOUND

        from ..pipeline import is_structure_question
        structure = self._structure_answer(sources) if is_structure_question(question) else None

        # 1. overview questions -> outline
        outline = [s for s in sources if s["kind"] == "outline"]
        if outline and is_overview_question(question):
            o = outline[0]
            title = re.search(r"Title: (.*)", o["body"])
            doc = re.search(r"Document overview: (.*)", o["body"])
            body_lines = [ln for s_ in outline for ln in s_["body"].splitlines()]
            items = [re.sub(r" \(p\. \d+\)$", "", ln[2:]) for ln in body_lines if ln.startswith("- ")]
            items = [it for it in items if "no headings found" not in it.lower()]
            if len(items) < 5:                       # short document: list nested sections too
                items = [re.sub(r" \(p\. \d+\)$", "", ln.strip()[2:]) for ln in body_lines
                         if ln.strip().startswith("- ")]
            # Slide decks title every slide "Topic – N. Subtopic - detail": group by topic.
            groups: dict[str, list[str]] = {}
            for it in items:
                parts = re.split(r"\s+[–—-]\s+", it, maxsplit=1)
                top = parts[0].strip()
                segs = [re.sub(r"^\d+\.\s*", "", x.strip())
                        for x in re.split(r"\s+[–—-]\s+", parts[1])][:1] if len(parts) > 1 else []
                sub = " › ".join(x for x in segs if x)
                lst = groups.setdefault(top, [])
                if sub and sub not in lst:
                    lst.append(sub)
            items = [it for it in items if "no headings found" not in it.lower()]
            if not items:
                described = self._describe_without_headings(sources, doc.group(1) if doc else None)
                if described:
                    return described + (f"\n\n{structure}" if structure else "")
            lines = []
            for top, subs in groups.items():
                if subs:
                    lines.append(f"- **{top}**: " + ", ".join(subs[:14]) + ("…" if len(subs) > 14 else ""))
                elif len(lines) < 24:
                    lines.append(f"- {top}")
            head = f"**{doc.group(1) if doc else 'This document'}**" + (f" ({title.group(1)})" if title else "")
            return (f"{head} covers these topics [{o['n']}]:\n\n" + "\n".join(lines[:24])
                    + ("\n- …" if len(lines) > 24 else "") + (f"\n\n{structure}" if structure else ""))
        if structure:
            return structure

        q = {t for t in tokenize(question) if t not in _FILLER}
        content_sources = [s for s in sources if s["kind"] != "outline"]

        abbr = self._abbreviation_answer(question, content_sources)
        if abbr:
            return abbr
        chart = self._chart_answer(question, q, content_sources, tokenize)
        if chart:
            return chart
        if not q:
            return NOT_FOUND

        # rarity weights over all candidate units
        units = []        # (source, display, match tokens, section tokens, is_row, header)
        for s in content_sources:
            sec_toks = set(tokenize(s["section"]))
            header, rows = self._table(s["body"]) if s["kind"] == "table" else (None, [])
            if header:
                for r in rows:
                    units.append((s, r, set(tokenize(" ".join(header + r))), sec_toks, True, header))
            body_text = s["body"]
            cv = re.search(r"^Chart values:\s*(.+)$", body_text, flags=re.M)
            if cv:
                # OCR'd chart: answer from the "label: value" pairs, not from the noisy axis text
                title = next((ln.strip() for ln in body_text.splitlines() if ln.strip() and not ln.startswith("[")), "")
                for pair in cv.group(1).split(";"):
                    if ":" in pair:
                        label, value = (x.strip() for x in pair.split(":", 1))
                        units.append((s, f"{label}: {value}  ({title})", set(tokenize(f"{label} {value} {title}")),
                                      sec_toks, False, None))
                body_text = title
            for b in self._blocks(re.sub(r"^\|.*$", "", body_text, flags=re.M)):
                if len(b) >= 12 or (len(b) >= 6 and re.search(r"\d", b)):     # keep short "TOTAL 540" lines
                    toks = set(tokenize(b))
                    if _NUMBER_ABBR.search(b):
                        toks |= set(tokenize("number"))                        # "Invoice No: 417" answers "invoice number"
                    units.append((s, b, toks, sec_toks, False, None))
        if not units:
            return NOT_FOUND
        # A word that only appears in the title/outline ("policy" in "Leave Policy") is fine, but a
        # question none of whose words occur in the passages ("capital of France") cannot be answered.
        df = {t: sum(t in (u[2] | u[3]) for u in units) for t in q}
        q = {t for t in q if df[t] > 0}               # judge the passage on words it can contain
        if not q:
            return NOT_FOUND
        idf = {t: math.log(1 + len(units) / (1 + df[t])) for t in q}
        total = sum(idf.values()) or 1.0

        def score(u):
            own = sum(idf[t] for t in q & u[2])
            sec = sum(idf[t] for t in (q & u[3]) - u[2])
            sc = own + 0.6 * sec
            return sc * 0.7 if u[0]["kind"] == "figure" else sc      # prose answers beat diagram label dumps

        scored = sorted(((score(u), i) for i, u in enumerate(units)), reverse=True)
        if scored[0][0] < 0.35 * total:
            return NOT_FOUND

        # 2. list questions -> every matching table row (merged across slides) or bullet
        if _LIST_Q.search(question):
            rows = [(u, sc) for sc, i in scored if (u := units[i])[4] and sc >= 0.35 * total]
            if rows:
                header = rows[0][0][5]
                seen, lines = set(), []
                for u, _ in sorted(rows, key=lambda x: x[0][0]["n"]):
                    if len(u[5]) != len(header):
                        continue
                    key = re.sub(r"\W+", " ", u[1][0]).lower().strip()
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append("| " + " | ".join(u[1]) + f" | [{u[0]['n']}] |")
                if lines:
                    md = "| " + " | ".join(header) + " | Source |\n|" + "---|" * (len(header) + 1)
                    return (f"Found {len(lines)} matching row{'s' if len(lines) > 1 else ''}:\n\n"
                            + md + "\n" + "\n".join(lines[:25]))
            children = self._outline_children(outline, q, tokenize)
            if children:
                return children
            bullets = [units[i] for sc, i in scored if not units[i][4] and sc >= 0.5 * scored[0][0]]
            if len(bullets) >= 2:
                seen, out = set(), []
                for u in bullets[:10]:
                    if u[1] not in seen:
                        seen.add(u[1])
                        out.append(f"- {_BULLET.sub('', u[1])} [{u[0]['n']}]")
                return "Relevant points:\n\n" + "\n".join(out)

        # 3. factual -> best sentences / bullets / table rows
        best, seen = [], set()
        for sc, i in scored[:6]:
            u = units[i]
            if sc < scored[0][0] * 0.75 or len(best) == 3:
                break
            text = ("| " + " | ".join(u[5]) + " |\n|" + "---|" * len(u[5]) + "\n| " + " | ".join(u[1]) + " |"
                    if u[4] else _BULLET.sub("", u[1]))
            if not u[4] and len(text) < 70 and text.rstrip().endswith(("?", ":")):
                nxt = [units[j][1] for j in range(i + 1, min(i + 3, len(units)))
                       if units[j][0] is u[0] and not units[j][4]]
                if nxt:                                  # "Why not Waterfall?" -> include the explanation
                    text = text + "\n\n" + " ".join(_BULLET.sub("", x) for x in nxt[:2])
            if text in seen:
                continue
            seen.add(text)
            best.append(f"{text} [{u[0]['n']}]")
        return "Most relevant passage" + ("s" if len(best) > 1 else "") + ":\n\n" + "\n\n".join(best)


def ollama_reachable(host: str | None = None) -> bool:
    try:
        import requests
        host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return requests.get(host.rstrip("/") + "/api/tags", timeout=1.0).ok
    except Exception:
        return False


def get_llm(provider: str | None = None, **kwargs) -> LLM:
    """mistral | gemini | ollama | extractive | auto.

    auto (default): every provider that is configured, in this order: Mistral (MISTRAL_API_KEY),
    Gemini (GEMINI_API_KEY or GOOGLE_API_KEY), Ollama (if running); the offline extractive mode
    is always the last resort. If a provider fails, the next one answers."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "").lower()
    if provider == "mistral":
        return MistralLLM(**kwargs)
    if provider in ("gemini", "google"):
        return GeminiLLM(**kwargs)
    if provider == "ollama":
        return OllamaLLM(**kwargs)
    if provider in ("extractive", "none", "offline"):
        return ExtractiveLLM()
    if provider and provider != "auto":
        raise ValueError(f"Unknown LLM provider {provider!r} (auto | mistral | gemini | ollama | extractive)")
    chain = []
    if clean_key(os.getenv("MISTRAL_API_KEY")):
        chain.append(MistralLLM())
    for gk in gemini_keys():
        chain.append(GeminiLLM(api_key=gk))
    if ollama_reachable():
        try:
            chain.append(OllamaLLM())
        except Exception:
            pass
    return FallbackLLM(chain + [ExtractiveLLM()]) if chain else ExtractiveLLM()