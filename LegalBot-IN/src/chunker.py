"""Create section-wise, non-paraphrased chunks from the cleaned BNS text.

Run from the repository root with ``python -m src.chunker``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_TEXT = PROJECT_ROOT / "data" / "processed" / "bns_cleaned_text.txt"
SECTIONS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "bns_sections.json"
REPORT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "bns_chunking_report.json"
ACT_NAME = "Bharatiya Nyaya Sanhita, 2023"

# Section bodies begin with an Arabic section number. Subsections and clauses
# begin with parentheses, so they remain inside their parent section.
SECTION_START_RE = re.compile(r"^(?P<section>[1-9]\d{0,2})\.\s*(?P<line>\S.*)$", re.MULTILINE)
CHAPTER_RE = re.compile(r"^CHAPTER\s+(?P<number>[IVXLCDM]+)\s*$", re.MULTILINE)
PAGE_MARKER_RE = re.compile(r"^=== Page \d+ ===\s*$", re.MULTILINE)
LOGGER = logging.getLogger(__name__)

MAX_CHUNK_CHARACTERS = 6_000
SHORT_SECTION_CHARACTERS = 100


class ChunkingError(RuntimeError):
    """Raised when the cleaned BNS input cannot be parsed into sections."""


def _remove_extraction_markers(text: str) -> str:
    """Remove page markers and their isolated page-number lines only."""
    without_markers = PAGE_MARKER_RE.sub("", text)
    return re.sub(r"(?m)^\d{1,3}\s*$", "", without_markers)


def _substantive_text_start(text: str) -> int:
    """Skip front matter and arrangement-of-sections entries.

    The enacted text is introduced immediately before the substantive Chapter I.
    Falling back to the final occurrence of Chapter I remains deterministic for
    PDFs that omit the enactment formula during extraction.
    """
    enactment = re.search(r"BE it enacted by Parliament", text, flags=re.IGNORECASE)
    if enactment:
        chapter = CHAPTER_RE.search(text, enactment.end())
        if chapter:
            return chapter.start()
    chapters = list(CHAPTER_RE.finditer(text))
    if len(chapters) >= 2:
        return chapters[1].start()
    if chapters:
        return chapters[0].start()
    raise ChunkingError("No chapter heading was found in the cleaned BNS text.")


def _chapter_at(text: str, position: int) -> str:
    """Return the nearest preceding chapter label and its following title line."""
    chapters = list(CHAPTER_RE.finditer(text, 0, position))
    if not chapters:
        return ""
    chapter = chapters[-1]
    after_heading = text[chapter.end() : position]
    title = next((line.strip() for line in after_heading.splitlines() if line.strip()), "")
    # A new section should never be treated as a chapter title.
    if SECTION_START_RE.match(title):
        title = ""
    return f"CHAPTER {chapter.group('number')}{': ' + title if title else ''}"


def _heading_from_first_line(first_line: str) -> str:
    """Extract the printed heading before its legislative dash where available."""
    heading = re.split(r"[—–]{1,2}", first_line, maxsplit=1)[0].strip()
    return heading.rstrip(".").strip()


def _split_long_section(text: str, limit: int = MAX_CHUNK_CHARACTERS) -> list[str]:
    """Split only long sections at blank or line boundaries without rewriting."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        boundary = remaining.rfind("\n\n", 0, limit + 1)
        if boundary < limit // 2:
            boundary = remaining.rfind("\n", 0, limit + 1)
        if boundary < limit // 2:
            boundary = remaining.rfind(" ", 0, limit + 1)
        if boundary <= 0:
            boundary = limit
        parts.append(remaining[:boundary].rstrip())
        remaining = remaining[boundary:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts


def parse_sections(cleaned_text: str) -> list[dict[str, str]]:
    """Parse substantive BNS text into complete legal sections.

    Original wording is preserved verbatim apart from extraction-only page
    markers and isolated page numbers. No subsection or clause is independently
    split from its parent section.
    """
    text = _remove_extraction_markers(cleaned_text)
    text = text[_substantive_text_start(text) :]
    candidate_starts = list(SECTION_START_RE.finditer(text))
    if not candidate_starts:
        raise ChunkingError("No numbered BNS sections were detected in the cleaned text.")

    # The BNS sections are a consecutive enacted series. Footnotes can also
    # begin at line start with e.g. ``1.``; accepting only the next expected
    # number keeps those footnotes in the preceding section body.
    starts: list[re.Match[str]] = []
    expected_section = 1
    for candidate in candidate_starts:
        if int(candidate.group("section")) == expected_section:
            starts.append(candidate)
            expected_section += 1
    if not starts:
        raise ChunkingError("The consecutive BNS section series could not be detected.")

    sections: list[dict[str, str]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        section_text = text[match.start() : end].strip()
        # Chapter labels between adjacent sections belong to the following
        # section's metadata, not to the preceding section's legal wording.
        next_chapter = CHAPTER_RE.search(section_text, 1)
        if next_chapter:
            section_text = section_text[: next_chapter.start()].rstrip()
        first_line = match.group("line")
        sections.append(
            {
                "act": ACT_NAME,
                "section": match.group("section"),
                "heading": _heading_from_first_line(first_line),
                "chapter": _chapter_at(text, match.start()),
                "text": section_text,
            }
        )
    return sections


def create_chunks(sections: list[dict[str, str]]) -> list[dict[str, str]]:
    """Create one chunk per section unless an unusually long section needs splits."""
    chunks: list[dict[str, str]] = []
    for section in sections:
        for part in _split_long_section(section["text"]):
            chunks.append({**section, "text": part})
    return chunks


def build_report(sections: list[dict[str, str]], chunks: list[dict[str, str]]) -> dict[str, Any]:
    """Return transparent chunking diagnostics for academic review."""
    short = [item["section"] for item in sections if len(item["text"]) < SHORT_SECTION_CHARACTERS]
    long = [item["section"] for item in sections if len(item["text"]) > MAX_CHUNK_CHARACTERS]
    return {
        "total_chunks": len(chunks),
        "sections_detected": len(sections),
        "sections_with_missing_headings": [item["section"] for item in sections if not item["heading"]],
        "sections_with_unusually_short_text": short,
        "sections_with_unusually_long_text": long,
        "first_5_sample_chunks": chunks[:5],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_chunking(input_path: Path = INPUT_TEXT) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Read the cleaned BNS text and write section chunks and a report."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Cleaned BNS text not found: {input_path}. Run python -m src.pdf_extractor first."
        )
    sections = parse_sections(input_path.read_text(encoding="utf-8"))
    chunks = create_chunks(sections)
    report = build_report(sections, chunks)
    _write_json(SECTIONS_OUTPUT, chunks)
    _write_json(REPORT_OUTPUT, report)
    return chunks, report


def main() -> int:
    """Command-line entry point for BNS legal section-wise chunking."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        chunks, report = run_chunking()
    except (FileNotFoundError, ChunkingError) as error:
        LOGGER.error("BNS chunking failed: %s", error)
        return 1
    LOGGER.info("BNS chunking completed")
    LOGGER.info("Sections detected: %d", report["sections_detected"])
    LOGGER.info("Chunks created: %d", len(chunks))
    LOGGER.info("Output: %s", SECTIONS_OUTPUT)
    LOGGER.info("Report: %s", REPORT_OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
