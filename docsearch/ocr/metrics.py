"""OCR quality metrics: Character Error Rate (CER) and Word Error Rate (WER).

    CER = (substitutions + deletions + insertions) / characters in the reference
    WER = same, counted over words

Lower is better; 0.0 means a perfect transcription.
"""

from __future__ import annotations

import re


def _levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _norm(text: str, lowercase: bool) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def cer(reference: str, hypothesis: str, lowercase: bool = False) -> float:
    ref, hyp = _norm(reference, lowercase), _norm(hypothesis, lowercase)
    return _levenshtein(ref, hyp) / max(len(ref), 1)


def wer(reference: str, hypothesis: str, lowercase: bool = False) -> float:
    ref, hyp = _norm(reference, lowercase).split(), _norm(hypothesis, lowercase).split()
    return _levenshtein(ref, hyp) / max(len(ref), 1)
