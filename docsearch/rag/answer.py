"""
Stage 5b: retrieval-augmented generation with verifiable citations.

RAG-Chatbot used LangChain's ``ConversationalRetrievalChain`` (deprecated in
LangChain 0.3+) and returned only the answer text: no sources, so a user could
not check where an answer came from. Here:

1. Follow-up questions are rewritten into standalone questions using the
   chat history (the same idea as ConversationalRetrievalChain's condense step).
2. The top-k chunks are given to the LLM as numbered sources [1]..[k], each
   labelled with file, page and whether it came from OCR.
3. The model must cite [n] after each claim, and must say so when the
   sources don't contain the answer instead of guessing.
4. Citations are parsed back and mapped to file + page; citations to sources
   that don't exist are reported as invalid (a simple hallucination check).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLM
from .store import Hit

NOT_FOUND = "I couldn't find this in the uploaded documents."

SYSTEM_PROMPT = f"""You are a document question-answering assistant.
Answer ONLY from the numbered sources provided. Rules:
- After every sentence that uses a source, cite it like [1] or [2][3].
- Do not use outside knowledge. Do not invent facts, numbers or names.
- If the sources do not contain the answer, reply exactly: "{NOT_FOUND}"
- Some sources were produced by OCR and may contain small recognition errors; read past obvious typos but do not "correct" numbers.
- Tables are given in Markdown: read values from the correct row and column.
- A "document outline" source lists a document's title and sections; use it for questions about what a document covers.
- For "what is this document about" questions, start with one or two sentences saying what the subject is and what the document is for, then list the main topics it covers (from the outline), with citations.
- For "types of / list" questions, include every item found across all sources, not just the first.
- Be concise: a direct answer first, then supporting detail if useful."""

CONDENSE_PROMPT = """Rewrite the follow-up question as a single standalone question that can be understood without the conversation. Keep names, numbers and document references. Return ONLY the rewritten question.

Conversation:
{history}

Follow-up question: {question}
Standalone question:"""


@dataclass
class Citation:
    number: int
    source: str
    page: int | None
    method: str
    snippet: str
    element: str = "text"
    ocr_used: bool = False
    section: str = ""

    def label(self) -> str:
        p = f", p. {self.page}" if self.page is not None else ""
        kind = f", {self.element}" if self.element in ("table", "figure") else ""
        ocr = " (OCR)" if self.ocr_used else ""
        return f"[{self.number}] {self.source}{p}{kind}{ocr}"

    def to_dict(self) -> dict:
        return {"number": self.number, "source": self.source, "page": self.page, "method": self.method,
                "snippet": self.snippet, "element": self.element, "ocr_used": self.ocr_used,
                "section": self.section, "label": self.label()}


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    standalone_question: str | None = None
    invalid_citations: list[int] = field(default_factory=list)
    warning: str = ""                 # set when the language model failed and the offline fallback answered

    @property
    def found(self) -> bool:
        return NOT_FOUND.lower()[:30] not in self.answer.lower()

    def format(self) -> str:
        out = self.answer
        if self.citations:
            out += "\n\nSources:\n" + "\n".join("  " + c.label() for c in self.citations)
        return out


def format_sources(hits: list[Hit]) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        md = h.chunk.metadata
        page = f", page {md['page']}" if md.get("page") is not None else ""
        ocr = ", OCR text" if md.get("ocr_used") else ""
        kind = {"table": ", table", "figure": ", figure/chart",
                "outline": ", document outline"}.get(md.get("element", ""), "")
        blocks.append(f"[{n}] ({md.get('source', '?')}{page}{kind}{ocr})\n{h.chunk.text}")
    return "\n\n".join(blocks)


_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(text: str) -> list[int]:
    nums = []
    for m in _CITE.finditer(text):
        for n in m.group(1).split(","):
            n = int(n)
            if n not in nums:
                nums.append(n)
    return nums


_REFERS = re.compile(r"\b(it|its|that|this|those|these|they|them|their|he|she|his|her|there|same|above|"
                     r"previous|more|other|another|else|also|too)\b", re.I)


def needs_condense(question: str) -> bool:
    """Only follow-ups that lean on earlier turns ("what about its cost?", "more?") need a
    rewrite. Self-contained questions skip the extra model call, which halves the number of
    requests (and so the chance of hitting an API rate limit)."""
    return bool(_REFERS.search(question)) or len(question.split()) < 4


def condense_question(llm: LLM, question: str, history: list[tuple[str, str]]) -> str:
    if not history or not needs_condense(question):
        return question
    convo = "\n".join(f"User: {q}\nAssistant: {a[:400]}" for q, a in history[-3:])
    rewritten = llm.chat([{"role": "user",
                           "content": CONDENSE_PROMPT.format(history=convo, question=question)}],
                         temperature=0.0, max_tokens=120).strip().strip('"')
    return rewritten.splitlines()[0] if rewritten else question


def generate_answer(llm: LLM, question: str, hits: list[Hit],
                    standalone: str | None = None) -> Answer:
    if not hits:
        return Answer(question, NOT_FOUND, [], [], standalone)

    q = standalone or question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {q}"},
    ]
    text = llm.chat(messages)

    cited = parse_citations(text)
    valid = [n for n in cited if 1 <= n <= len(hits)]
    invalid = [n for n in cited if n not in valid]
    citations = []
    for n in sorted(valid):
        md = hits[n - 1].chunk.metadata
        citations.append(Citation(n, md.get("source", "?"), md.get("page"),
                                  md.get("method", "native"), hits[n - 1].chunk.text[:400],
                                  md.get("element", "text"), bool(md.get("ocr_used")),
                                  md.get("section", "")))
    return Answer(question, text, citations, hits, standalone, invalid)
