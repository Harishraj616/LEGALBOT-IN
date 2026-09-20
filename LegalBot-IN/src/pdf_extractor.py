"""Extract and conservatively clean the BNS, 2023 PDF using PyMuPDF.

Run from the repository root with: ``python -m src.pdf_extractor``.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Sequence

import fitz

from .text_cleaner import PAGE_MARKER_TEMPLATE, clean_pages


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PDF = PROJECT_ROOT / "data" / "raw" / "BNS_2023.pdf"
RAW_TEXT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "bns_raw_text.txt"
CLEANED_TEXT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "bns_cleaned_text.txt"
LOGGER = logging.getLogger(__name__)


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot be safely extracted."""


def extract_pages(pdf_path: Path) -> list[str]:
    """Load a PDF and return one text item per page, in document order."""
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"BNS PDF not found: {pdf_path}. "
            "Place the file at data/raw/BNS_2023.pdf and run the command again."
        )
    try:
        with fitz.open(pdf_path) as document:
            if document.page_count == 0:
                raise PDFExtractionError(f"The PDF contains no pages: {pdf_path}")
            page_texts = [page.get_text("text") for page in document]
    except fitz.FileDataError as error:
        raise PDFExtractionError(f"Unable to read PDF {pdf_path}: {error}") from error
    except RuntimeError as error:
        raise PDFExtractionError(f"Unable to extract PDF {pdf_path}: {error}") from error
    if not any(text.strip() for text in page_texts):
        raise PDFExtractionError(
            "No extractable text was found in the PDF. It may be image-only or corrupt."
        )
    return page_texts


def format_raw_pages(page_texts: Sequence[str]) -> str:
    """Add stable, one-based page markers to unmodified extracted text."""
    return "".join(
        PAGE_MARKER_TEMPLATE.format(page_number=page_number) + page_text
        for page_number, page_text in enumerate(page_texts, start=1)
    ).strip() + "\n"


def write_text(output_path: Path, content: str) -> None:
    """Write UTF-8 output, creating the processed-data directory if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def run_extraction(pdf_path: Path = INPUT_PDF) -> tuple[Path, Path]:
    """Extract BNS page text, save raw and cleaned files, and report progress."""
    page_texts = extract_pages(pdf_path)
    write_text(RAW_TEXT_OUTPUT, format_raw_pages(page_texts))
    write_text(CLEANED_TEXT_OUTPUT, clean_pages(page_texts))
    LOGGER.info("Pages processed: %d", len(page_texts))
    LOGGER.info("Total extracted characters: %d", sum(len(text) for text in page_texts))
    LOGGER.info("Raw text output: %s", RAW_TEXT_OUTPUT)
    LOGGER.info("Cleaned text output: %s", CLEANED_TEXT_OUTPUT)
    return RAW_TEXT_OUTPUT, CLEANED_TEXT_OUTPUT


def main() -> int:
    """Command-line entry point for BNS text ingestion."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_extraction()
    except (FileNotFoundError, PDFExtractionError) as error:
        LOGGER.error("Extraction failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
