"""
Hybrid Explainable Retriever Unit Test Suite.
MediAcad Agentic RAG - Member 2 (Hybrid Retrieval & Mathematical Explainability).

Verifies:
  - BaseRetriever OOP interface and Strategy Pattern conformance
  - Convex fusion alpha parameter tuning (Sparse-only, Dense-only, Hybrid)
  - Mathematical Explainability Trace (MathExplanationTrace) accuracy
  - Dependency Injection with custom embedders
  - High-precision retrieval across the 3 core university policy scenarios
"""

import json
import math
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interfaces.contracts import (
    BaseRetriever,
    DocumentChunk,
    MathExplanationTrace,
    RetrievalResult,
)
from src.retrieval.hybrid_retriever import HybridExplainableRetriever


@pytest.fixture
def sample_chunks():
    """Loads 5 official university policy chunks from fixtures."""
    fixture_path = PROJECT_ROOT / "data" / "fixtures" / "sample_chunks.json"
    raw_data = json.loads(fixture_path.read_text(encoding="utf-8"))
    return [DocumentChunk(**item) for item in raw_data]


def dummy_dense_embedder(texts):
    """Predictable mock embedder for fast, isolated mathematical testing."""
    # Returns 16-dimensional deterministic vectors
    vectors = []
    for text in texts:
        vec = np.zeros(16, dtype=np.float64)
        for ch in text.lower():
            vec[ord(ch) % 16] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        vectors.append(vec)
    return np.array(vectors)


class TestHybridRetrieverArchitecture:
    """Verifies OOP and Strategy Pattern adherence."""

    def test_implements_base_retriever_interface(self, sample_chunks):
        retriever = HybridExplainableRetriever(
            chunks=sample_chunks,
            dense_embedder=dummy_dense_embedder,
        )
        assert isinstance(retriever, BaseRetriever)

    def test_empty_retriever_safety(self):
        retriever = HybridExplainableRetriever()
        results = retriever.retrieve("any query")
        assert results == []

    def test_invalid_alpha_raises_value_error(self, sample_chunks):
        with pytest.raises(ValueError, match="alpha must be in"):
            HybridExplainableRetriever(chunks=sample_chunks, alpha=1.5)
        with pytest.raises(ValueError, match="alpha must be between"):
            r = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.5)
            r.set_alpha(-0.1)


class TestScoreFusionAndStrategyPattern:
    """Verifies alpha weighting convex combination S_hybrid = a*S_dense + (1-a)*S_sparse."""

    def test_pure_sparse_mode_alpha_zero(self, sample_chunks):
        retriever = HybridExplainableRetriever(
            chunks=sample_chunks,
            alpha=0.0,
            dense_embedder=dummy_dense_embedder,
        )
        results = retriever.retrieve("Form EC-01 hospitalization 10-day", top_k=3)
        assert len(results) > 0
        for r in results:
            # Hybrid score strictly equals sparse score
            assert math.isclose(r.hybrid_score, r.sparse_score, abs_tol=1e-6)

    def test_pure_dense_mode_alpha_one(self, sample_chunks):
        retriever = HybridExplainableRetriever(
            chunks=sample_chunks,
            alpha=1.0,
            dense_embedder=dummy_dense_embedder,
        )
        results = retriever.retrieve("Form EC-01 hospitalization 10-day", top_k=3)
        assert len(results) > 0
        for r in results:
            # Hybrid score strictly equals dense score
            assert math.isclose(r.hybrid_score, r.dense_score, abs_tol=1e-6)

    def test_balanced_hybrid_mode(self, sample_chunks):
        alpha = 0.60
        retriever = HybridExplainableRetriever(
            chunks=sample_chunks,
            alpha=alpha,
            dense_embedder=dummy_dense_embedder,
        )
        results = retriever.retrieve("Form EC-01 hospitalization 10-day", top_k=3)
        for r in results:
            expected = alpha * r.dense_score + (1.0 - alpha) * r.sparse_score
            assert math.isclose(r.hybrid_score, expected, abs_tol=1e-6)


class TestMathematicalExplainabilityTrace:
    """Verifies that every retrieval result provides full mathematical proof."""

    def test_math_trace_fields_and_integrity(self, sample_chunks):
        retriever = HybridExplainableRetriever(
            chunks=sample_chunks,
            alpha=0.60,
            dense_embedder=dummy_dense_embedder,
        )
        results = retriever.retrieve("missed examination Form EC-01 10-day", top_k=2)
        top = results[0]

        trace = top.math_trace
        assert isinstance(trace, MathExplanationTrace)

        # Check vector magnitudes
        assert trace.query_vector_norm > 0.0
        assert trace.doc_vector_norm > 0.0

        # Check inner dot product & cosine similarity equivalence
        expected_cosine = trace.dot_product / (trace.query_vector_norm * trace.doc_vector_norm)
        assert math.isclose(trace.cosine_similarity, expected_cosine, abs_tol=1e-6)

        # Check BM25 score bounds
        assert 0.0 <= trace.bm25_normalized_score <= 1.0

        # Check matched query terms
        assert len(trace.matched_query_terms) > 0
        for term in trace.matched_query_terms:
            assert term in trace.term_frequencies
            assert term in trace.idf_values


class TestRealPolicyRetrievalScenarios:
    """Verifies top-1 retrieval accuracy across the 3 core university policy scenarios."""

    def test_scenario_1_examination_hospitalization(self, sample_chunks):
        """Query: Form EC-01 for missed examination with 10-day hospitalization."""
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.60)
        results = retriever.retrieve("missed final examination Form EC-01 hospitalization 10-day", top_k=3)

        assert len(results) > 0
        top = results[0]
        assert top.rank == 1
        assert top.chunk.doc_id == "POL-ACAD-EX-01"
        assert top.chunk.clause_id == "Clause 3.2"

    def test_scenario_2_attendance_condonation(self, sample_chunks):
        """Query: 72% attendance condonation with Form ATT-02A."""
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.60)
        results = retriever.retrieve("attendance condonation 72% Form ATT-02A 80%", top_k=3)

        assert len(results) > 0
        top = results[0]
        assert top.rank == 1
        assert top.chunk.doc_id == "POL-ACAD-AT-02"

    def test_scenario_3_coursework_extension(self, sample_chunks):
        """Query: Coursework late submission penalty 5% extension Tier 3 14 days."""
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.60)
        results = retriever.retrieve("coursework extension Form EXT-03C late penalty 5% 14 days", top_k=3)

        assert len(results) > 0
        top = results[0]
        assert top.rank == 1
        assert top.chunk.doc_id == "POL-ACAD-CW-03"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
