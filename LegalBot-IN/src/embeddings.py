"""Build the persistent ChromaDB collection for BNS legal-section chunks.

Run from the repository root with ``python -m src.embeddings``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Sentence-Transformers uses PyTorch here. Prevent Transformers from loading
# an unrelated TensorFlow installation during import on local Windows setups.
os.environ.setdefault("USE_TF", "0")

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECTIONS_INPUT = PROJECT_ROOT / "data" / "processed" / "bns_sections.json"
CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "bns_legal_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ACT_NAME = "Bharatiya Nyaya Sanhita, 2023"
BATCH_SIZE = 32
LOGGER = logging.getLogger(__name__)


class EmbeddingDatabaseError(RuntimeError):
    """Raised when BNS vector-database construction cannot be completed."""


def load_chunks(input_path: Path = SECTIONS_INPUT) -> list[dict[str, str]]:
    """Load and validate the section chunks without altering their legal text."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"BNS section chunks not found: {input_path}. Run python -m src.chunker first."
        )
    try:
        chunks = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EmbeddingDatabaseError(f"Invalid section JSON: {error}") from error
    if not isinstance(chunks, list) or not chunks:
        raise EmbeddingDatabaseError("The BNS section JSON must contain a non-empty chunk list.")
    required_fields = {"act", "section", "heading", "chapter", "text"}
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not required_fields <= chunk.keys():
            raise EmbeddingDatabaseError(f"Chunk {index} does not contain all required fields.")
        if not chunk["text"].strip():
            raise EmbeddingDatabaseError(f"Chunk {index} has no legal text to embed.")
    return chunks


def stable_chunk_ids(chunks: list[dict[str, str]]) -> list[str]:
    """Return deterministic IDs, using a suffix only for split sections."""
    totals: dict[str, int] = {}
    for chunk in chunks:
        totals[chunk["section"]] = totals.get(chunk["section"], 0) + 1

    occurrences: dict[str, int] = {}
    ids: list[str] = []
    for chunk in chunks:
        section = chunk["section"]
        occurrences[section] = occurrences.get(section, 0) + 1
        base = f"bns_section_{section}"
        ids.append(base if totals[section] == 1 else f"{base}_chunk_{occurrences[section]}")
    if len(ids) != len(set(ids)):
        raise EmbeddingDatabaseError("Stable chunk ID generation produced duplicates.")
    return ids


def _metadata(chunk: dict[str, str]) -> dict[str, str]:
    """Select only the required legal metadata for ChromaDB."""
    return {
        "act": chunk["act"] or ACT_NAME,
        "section": chunk["section"],
        "heading": chunk["heading"],
        "chapter": chunk["chapter"],
    }


def create_vector_database() -> Any:
    """Create/rebuild the local BNS ChromaDB collection and store all chunks.

    Each original chunk text is stored verbatim as the Chroma document. The
    collection is deliberately rebuilt so the database precisely matches the
    verified `bns_sections.json` input on each explicit build invocation.
    """
    chunks = load_chunks()
    documents = [chunk["text"] for chunk in chunks]
    ids = stable_chunk_ids(chunks)
    metadatas = [_metadata(chunk) for chunk in chunks]

    LOGGER.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    LOGGER.info("Embedding model loaded")
    LOGGER.info("Number of chunks: %d", len(chunks))
    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    LOGGER.info("Number of embeddings generated: %d", len(embeddings))

    CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))
    try:
        client.delete_collection(name=COLLECTION_NAME)
        LOGGER.info("Existing ChromaDB collection removed for rebuild")
    except Exception as error:
        # Chroma raises when no previous collection exists. Any other issue is
        # surfaced by get_or_create_collection or add below.
        LOGGER.debug("No existing collection removed: %s", error)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    LOGGER.info("ChromaDB collection created: %s", COLLECTION_NAME)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )
    LOGGER.info("Number of documents stored: %d", collection.count())
    return collection, model, len(chunks)


def validate_collection(collection: Any, model: SentenceTransformer, expected_count: int) -> None:
    """Verify collection size and print the top three semantic retrieval hits."""
    count = collection.count()
    if count != expected_count:
        raise EmbeddingDatabaseError(
            f"Collection count mismatch: expected {expected_count}, found {count}."
        )
    LOGGER.info("Collection count verified: %d", count)
    query = "What is the punishment for murder?"
    result = collection.query(
        query_embeddings=[model.encode(query, convert_to_numpy=True).tolist()],
        n_results=min(3, count),
        include=["documents", "metadatas", "distances"],
    )
    LOGGER.info("Sample retrieval query: %s", query)
    for rank, (document, metadata, distance) in enumerate(
        zip(result["documents"][0], result["metadatas"][0], result["distances"][0]), start=1
    ):
        preview = " ".join(document.split())[:180]
        LOGGER.info(
            "%d. Section %s — %s | distance: %.4f | %s%s",
            rank,
            metadata["section"],
            metadata["heading"],
            distance,
            preview,
            "..." if len(preview) == 180 else "",
        )


def main() -> int:
    """Command-line entry point for BNS embeddings and ChromaDB storage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        collection, model, chunk_count = create_vector_database()
        validate_collection(collection, model, chunk_count)
    except (FileNotFoundError, EmbeddingDatabaseError) as error:
        LOGGER.error("Vector database build failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
