"""
Unit & Integration Tests for Member 1: Document Engineering & Ingestion Pipeline.
Satisfies DS205.3 Automated Testing (CI/CD) and OOP Verification Rubric.
"""

import os
import json
import pytest
from src.interfaces.contracts import BaseChunker, DocumentChunk
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.clause_chunker import ClauseAwareChunker


PDF_DIR = "data/raw_pdfs"
PROCESSED_JSON = "data/processed_chunks.json"
CHROMA_DIR = "data/chroma_db"


def test_base_chunker_inheritance():
    """Verify ClauseAwareChunker implements the BaseChunker abstract contract."""
    chunker = ClauseAwareChunker()
    assert isinstance(chunker, BaseChunker)
    assert hasattr(chunker, "parse_and_chunk")


def test_pdf_loader_metadata_all_docs():
    """Verify PDFLoader resolves accurate metadata for all 5 university policy PDFs."""
    expected_doc_ids = {
        "01": "POL-ACAD-EX-01",
        "02": "POL-ACAD-AT-02",
        "03": "POL-ACAD-CW-03",
        "04": "POL-MED-WF-04",
        "05": "HBK-ACAD-PR-05"
    }

    assert os.path.isdir(PDF_DIR), f"Directory {PDF_DIR} missing"
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    assert len(pdf_files) == 5, f"Expected 5 PDFs, found {len(pdf_files)}"

    for f in pdf_files:
        prefix = f[:2]
        if prefix in expected_doc_ids:
            loader = PDFLoader(os.path.join(PDF_DIR, f))
            assert loader.metadata["doc_id"] == expected_doc_ids[prefix]
            assert "effective_year" in loader.metadata
            assert loader.metadata["effective_year"] in [2023, 2025]


def test_pdf_loader_table_extraction():
    """Verify that tabular matrices are parsed into Markdown format."""
    exam_pdf = os.path.join(PDF_DIR, "01_Examination_and_Medical_Consideration_Regulations.pdf")
    loader = PDFLoader(exam_pdf)
    pages = loader.load_pages()

    assert len(pages) >= 4
    # Check that at least one page has tables
    tables_found = [p for p in pages if len(p["tables"]) > 0]
    assert len(tables_found) > 0

    first_table = tables_found[0]["tables"][0]["markdown"]
    assert "|" in first_table
    assert "---" in first_table


def test_clause_chunker_structure_and_header_injection():
    """Verify Clause-Aware Chunking, Context Header Injection, and DocumentChunk typing."""
    chunker = ClauseAwareChunker()
    chunks = chunker.parse_and_chunk(PDF_DIR)

    assert len(chunks) > 50, f"Expected rich chunks, got {len(chunks)}"

    # Check first chunk structure
    first_chunk = chunks[0]
    assert isinstance(first_chunk, DocumentChunk)
    assert first_chunk.chunk_id
    assert first_chunk.doc_id
    assert first_chunk.policy_version
    assert first_chunk.clause_id

    # Verify Context Header Injection
    assert first_chunk.text.startswith("[Doc:")
    assert " | Version:" in first_chunk.text
    assert " | Page:" in first_chunk.text


def test_metadata_enrichment():
    """Verify that forms, timeline constraints, and legacy tags are enriched."""
    chunker = ClauseAwareChunker()
    chunks = chunker.parse_and_chunk(PDF_DIR)

    # Find chunks with administrative forms
    chunks_with_forms = [c for c in chunks if len(c.metadata.get("mentioned_forms", [])) > 0]
    assert len(chunks_with_forms) > 0, "No chunks with detected forms found"

    # Verify legacy policy flag for 2023 handbook
    legacy_chunks = [c for c in chunks if c.effective_year == 2023]
    assert len(legacy_chunks) > 0
    assert any(c.metadata.get("is_legacy") is True for c in legacy_chunks)


def test_processed_chunks_json_contract():
    """Verify data/processed_chunks.json is valid and matches Member 2 data contract."""
    assert os.path.isfile(PROCESSED_JSON), f"File {PROCESSED_JSON} not found"

    with open(PROCESSED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) > 50

    sample = data[0]
    required_keys = [
        "chunk_id", "doc_id", "doc_title", "policy_version",
        "effective_year", "page_number", "section_title",
        "clause_id", "chunk_type", "text", "metadata"
    ]
    for k in required_keys:
        assert k in sample, f"Missing required key '{k}' in JSON chunks"


def test_persistent_chroma_store_retrieval():
    """Verify Persistent ChromaDB index exists and supports cosine similarity queries."""
    if not os.path.isdir(CHROMA_DIR):
        pytest.skip(f"ChromaDB directory {CHROMA_DIR} is gitignored in CI; skipping local vector store check.")

    from src.ingestion.vector_store_setup import PersistentChromaStore

    store = PersistentChromaStore(persist_dir=CHROMA_DIR)
    assert store.collection.count() > 50

    results = store.query("inpatient hospitalization 10 days deferred exam", top_k=3)
    assert len(results["ids"][0]) == 3
    assert len(results["documents"][0]) == 3
    assert len(results["metadatas"][0]) == 3
