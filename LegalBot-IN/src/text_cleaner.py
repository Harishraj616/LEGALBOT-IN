"""Conservative cleaning utilities for extracted BNS PDF text.

The functions in this module repair predictable PDF-extraction artifacts only.
They do not paraphrase, summarize, or otherwise alter the legal meaning of text.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Sequence


PAGE_MARKER_TEMPLATE = "\n\n=== Page {page_number} ===\n\n"
_WHITESPACE_RE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
_WRAPPED_HYPHEN_RE = re.compile(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])")


def _normalise_for_comparison(line: str) -> str:
    """Return a comparison key without changing the output text."""
    return _WHITESPACE_RE.sub(" ", line).strip().casefold()


def detect_repeated_margin_lines(page_texts: Sequence[str]) -> set[str]:
    """Identify likely repeated headers/footers found at page margins.

    Only first and last two non-empty lines are considered. A candidate must
    recur on three pages and at least 20% of document pages.
    """
    if not page_texts:
        return set()
    candidates: list[str] = []
    for page_text in page_texts:
        lines = [line for line in page_text.splitlines() if line.strip()]
        candidates.extend(_normalise_for_comparison(line) for line in lines[:2])
        candidates.extend(_normalise_for_comparison(line) for line in lines[-2:])
    threshold = max(3, (len(page_texts) + 4) // 5)
    counts = Counter(candidate for candidate in candidates if candidate)
    return {line for line, count in counts.items() if count >= threshold}


def clean_page_text(page_text: str, repeated_margin_lines: set[str]) -> str:
    """Clean one page while retaining legal wording, numbering, and headings."""
    text = page_text.replace("\r\n", "\n").replace("\r", "\n")
    # Join only a lower-case continuation after a wrapping hyphen.
    text = _WRAPPED_HYPHEN_RE.sub("", text)
    lines: list[str] = []
    for line in text.split("\n"):
        compact_line = _WHITESPACE_RE.sub(" ", line).strip()
        if _normalise_for_comparison(compact_line) in repeated_margin_lines:
            continue
        lines.append(compact_line)
    return _EXCESS_BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def clean_pages(page_texts: Sequence[str]) -> str:
    """Return cleaned text with explicit page markers retained for traceability."""
    repeated_margin_lines = detect_repeated_margin_lines(page_texts)
    cleaned_pages = [
        PAGE_MARKER_TEMPLATE.format(page_number=index)
        + clean_page_text(page_text, repeated_margin_lines)
        for index, page_text in enumerate(page_texts, start=1)
    ]
    return "".join(cleaned_pages).strip() + "\n"
