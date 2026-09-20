# LEGALBOT-IN

An M.Tech academic project that will implement a basic Retrieval-Augmented Generation (RAG) workflow for legal assistance using the **Bharatiya Nyaya Sanhita (BNS), 2023**.

This Phase 1 project adapts the baseline workflow described in *LegalBot-EC: An LLM-Based Chatbot for Legal Assistance in Ecuadorian Law* to Indian criminal law.

## Phase 1 scope

The planned pipeline is:

```text
BNS 2023 PDF
  -> PyMuPDF text extraction
  -> text cleaning
  -> section-wise legal chunking
  -> sentence-transformer embeddings
  -> ChromaDB vector database
  -> semantic retrieval
  -> LLaMA 3.1 8B via Ollama
  -> generated answer
```

Phase 2 capabilities are explicitly out of scope for now: dynamic updates, amendment detection, version management, FAISS, BM25, cross-encoder reranking, automated source monitoring, citation verification, and hallucination detection.

## Project layout

```text
LEGALBOT-IN/
├── data/
│   ├── raw/                 # Source BNS PDF files
│   └── processed/           # Extracted, cleaned, and chunked data
├── embeddings/              # Persisted embedding artifacts, if required
├── chroma_db/               # Local ChromaDB persistence directory
├── src/
│   ├── pdf_extractor.py     # PDF-to-text extraction interface
│   ├── text_cleaner.py      # Legal-text cleaning interface
│   ├── chunker.py           # Section-wise chunking interface
│   ├── embeddings.py        # Embedding generation interface
│   ├── retriever.py         # ChromaDB semantic retrieval interface
│   └── rag_pipeline.py      # Retrieval and Ollama generation orchestration
├── evaluation/              # Evaluation datasets and scripts
├── app/                     # Future user interface layer
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.11 or newer
- The dependencies in `requirements.txt` (PyMuPDF is required for this extraction stage)

Ollama and LLaMA are not needed for the current ingestion stage; they belong to a later Phase 1 step.

## Setup

Create and activate a virtual environment, then install the Phase 1 dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the authoritative BNS, 2023 PDF at `data/raw/BNS_2023.pdf`. Do not place confidential user material in the repository.

## Extract the BNS text

From the repository root, run:

```powershell
python -m src.pdf_extractor
```

The command writes `data/processed/bns_raw_text.txt` (page-by-page PyMuPDF extraction) and `data/processed/bns_cleaned_text.txt` (conservative cleanup). It prints page count, total extracted characters, and both output paths. If the PDF is missing, it exits with a clear error and does not download a replacement.

## Create BNS legal-section chunks

After extraction, run from the repository root:

```powershell
python -m src.chunker
```

This reads `data/processed/bns_cleaned_text.txt` and writes section-wise JSON chunks to `data/processed/bns_sections.json`, plus diagnostics and sample chunks to `data/processed/bns_chunking_report.json`. Legal wording, subsections, and clauses remain in their parent section; only unusually long sections are split at existing text boundaries.

## Build embeddings and the local vector database

Install project dependencies:

```powershell
pip install -r requirements.txt
```

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`, loaded through the `sentence-transformers` Python library. Run the database build from the repository root:

```powershell
python -m src.embeddings
```

The command reads `data/processed/bns_sections.json`, embeds each original legal-text chunk without modifying it, and rebuilds the persistent ChromaDB collection `bns_legal_documents` in `chroma_db/`. Each record stores the original legal text as its document, alongside `act`, `section`, `heading`, and `chapter` metadata. It also validates the collection count and prints the top three results for a sample semantic query.

## Module responsibilities

| Module | Intended responsibility |
| --- | --- |
| `pdf_extractor.py` | Extract page-level text and basic source metadata with PyMuPDF. |
| `text_cleaner.py` | Normalize extraction artifacts while preserving legal meaning. |
| `chunker.py` | Produce BNS section-oriented chunks with metadata. |
| `embeddings.py` | Create sentence-transformer embeddings for chunks and queries. |
| `retriever.py` | Store chunks in and retrieve them semantically from ChromaDB. |
| `rag_pipeline.py` | Combine retrieved context with a user query and call Ollama. |

## Academic design notes

- Keep source text, cleaned text, chunks, and retrieval metadata reproducible and separate.
- Preserve BNS section identifiers, headings, source page numbers, and document title in chunk metadata.
- Treat responses as informational assistance, not legal advice.
- Record experimental configuration and evaluation inputs in `evaluation/` for reproducibility.

## Current status

The BNS PDF ingestion, cleaning, section-wise chunking, embeddings, and ChromaDB storage stages are implemented. Retrieval, Ollama, and RAG generation are not yet implemented.
