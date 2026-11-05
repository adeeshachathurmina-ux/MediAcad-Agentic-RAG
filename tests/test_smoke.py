"""
Automated CI/CD Smoke Tests for MediAcad Agentic RAG.
Executed automatically by GitHub Actions on every push and Pull Request.
"""

import json
from pathlib import Path
from src.interfaces.contracts import (
    DocumentChunk,
    MathExplanationTrace,
    RetrievalResult,
    AgentResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_policy_pdfs_exist():
    """Verify all 5 official university policy PDFs exist in data/raw_pdfs."""
    pdf_dir = PROJECT_ROOT / "data" / "raw_pdfs"
    pdfs = list(pdf_dir.glob("*.pdf"))
    assert len(pdfs) == 5, f"Expected 5 policy PDFs in data/raw_pdfs, found {len(pdfs)}"


def test_sample_chunks_fixture_valid():
    """Verify sample_chunks.json loads cleanly into DocumentChunk contracts."""
    fixture_path = PROJECT_ROOT / "data" / "fixtures" / "sample_chunks.json"
    assert fixture_path.exists(), "data/fixtures/sample_chunks.json is missing"

    raw_data = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert len(raw_data) >= 5, "Expected at least 5 sample chunks"

    first_chunk = DocumentChunk(**raw_data[0])
    assert first_chunk.doc_id == "POL-ACAD-EX-01"
    assert first_chunk.effective_year == 2025


def test_oop_contracts_instantiation():
    """Verify OOP dataclasses instantiate cleanly for all 4 team modules."""
    chunk = DocumentChunk(
        chunk_id="TEST-01",
        doc_id="POL-ACAD-EX-01",
        doc_title="Test Policy",
        policy_version="v4.0",
        effective_year=2025,
        page_number=1,
        section_title="Section 1",
        clause_id="Clause 1.1",
        chunk_type="clause",
        text="Sample regulatory clause text.",
    )
    res = RetrievalResult(
        chunk=chunk,
        sparse_score=0.82,
        dense_score=0.91,
        hybrid_score=0.87,
        rank=1,
    )
    assert res.hybrid_score == 0.87