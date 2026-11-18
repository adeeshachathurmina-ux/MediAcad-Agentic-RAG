"""
Mathematical Retrieval Integration & Strategy Pattern Test Suite.
MediAcad Agentic RAG - Member 2 (Hybrid Retrieval & Mathematical Explainability).

Satisfies Step 6 of Member 2 Plan:
  1. Strategy Pattern verification (SparseOnly, DenseOnly, Hybrid).
  2. Mathematical identity bounds (Cauchy-Schwarz, Collinearity, Orthogonality).
  3. Score fusion invariants (Hybrid score strictly bounded by dense and sparse inputs).
  4. End-to-end evaluation across all 5 university policy modules in sample_chunks.json.
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
from src.retrieval.math_mechanics import (
    vector_magnitude,
    dot_product,
    l2_normalize,
    cosine_similarity,
    min_max_normalize,
    hybrid_fuse_score,
)
from src.retrieval.sparse_engine import TFIDFAndBM25Engine
from src.retrieval.hybrid_retriever import HybridExplainableRetriever


@pytest.fixture
def sample_chunks():
    """Loads 5 official university policy chunks from fixtures."""
    fixture_path = PROJECT_ROOT / "data" / "fixtures" / "sample_chunks.json"
    raw_data = json.loads(fixture_path.read_text(encoding="utf-8"))
    return [DocumentChunk(**item) for item in raw_data]


class TestMathematicalIdentitiesAndBounds:
    """Rigorous verification of linear algebra theorems and score bounds."""

    def test_cauchy_schwarz_inequality(self):
        """Theorem: |u . v| <= ||u||_2 * ||v||_2 for all real vectors."""
        np.random.seed(42)
        for _ in range(20):
            dim = np.random.randint(5, 50)
            u = np.random.randn(dim)
            v = np.random.randn(dim)

            inner_prod = abs(dot_product(u, v))
            norm_product = vector_magnitude(u) * vector_magnitude(v)

            assert inner_prod <= norm_product + 1e-9, (
                f"Cauchy-Schwarz violated: |u.v|={inner_prod} > ||u||*||v||={norm_product}"
            )

    def test_collinearity_and_cosine_unit(self):
        """Theorem: cos(v, c*v) == 1.0 for any scalar c > 0."""
        v = np.array([3.14, -2.71, 1.414, 0.577])
        scaled_v = 4.2 * v
        assert math.isclose(cosine_similarity(v, scaled_v), 1.0, abs_tol=1e-9)

    def test_orthogonality_and_cosine_zero(self):
        """Theorem: cos(u, v) == 0.0 when u and v are orthogonal."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert math.isclose(cosine_similarity(u, v), 0.0, abs_tol=1e-9)

    def test_convex_combination_invariant(self):
        """Theorem: min(s_dense, s_sparse) <= S_hybrid <= max(s_dense, s_sparse) for alpha in [0, 1]."""
        for alpha in [0.0, 0.25, 0.50, 0.60, 0.75, 1.0]:
            for s_d in [0.1, 0.4, 0.9]:
                for s_s in [0.2, 0.5, 0.8]:
                    s_h = hybrid_fuse_score(s_d, s_s, alpha=alpha)
                    lower = min(s_d, s_s)
                    upper = max(s_d, s_s)
                    assert lower - 1e-9 <= s_h <= upper + 1e-9


class TestStrategyPatternRetrieval:
    """Tests runtime strategy switching between SparseOnly, DenseOnly, and Hybrid."""

    def test_sparse_only_strategy(self, sample_chunks):
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.0)
        results = retriever.retrieve("Form EC-01 10-day hospitalization", top_k=5)

        assert len(results) > 0
        for r in results:
            assert math.isclose(r.hybrid_score, r.sparse_score, abs_tol=1e-6)
            assert r.math_trace.alpha_weight == 0.0

    def test_dense_only_strategy(self, sample_chunks):
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=1.0)
        results = retriever.retrieve("Form EC-01 10-day hospitalization", top_k=5)

        assert len(results) > 0
        for r in results:
            assert math.isclose(r.hybrid_score, r.dense_score, abs_tol=1e-6)
            assert r.math_trace.alpha_weight == 1.0

    def test_dynamic_strategy_alpha_switch(self, sample_chunks):
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.60)
        res_hybrid = retriever.retrieve("missed examination", top_k=1)[0]

        # Switch to pure BM25
        retriever.set_alpha(0.0)
        res_sparse = retriever.retrieve("missed examination", top_k=1)[0]
        assert math.isclose(res_sparse.hybrid_score, res_sparse.sparse_score, abs_tol=1e-6)

        # Switch to pure dense
        retriever.set_alpha(1.0)
        res_dense = retriever.retrieve("missed examination", top_k=1)[0]
        assert math.isclose(res_dense.hybrid_score, res_dense.dense_score, abs_tol=1e-6)


class TestFivePolicyModuleRetrievalCoverage:
    """Verifies retrieval across all 5 official policy domains."""

    def test_module_1_examination_regulations(self, sample_chunks):
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.60)
        res = retriever.retrieve("missed final examination Form EC-01 hospitalization 10-day", top_k=1)
        assert res[0].chunk.doc_id == "POL-ACAD-EX-01"

    def test_module_2_attendance_policy(self, sample_chunks):
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.60)
        res = retriever.retrieve("attendance condonation 72% Form ATT-02A 80%", top_k=1)
        assert res[0].chunk.doc_id == "POL-ACAD-AT-02"

    def test_module_3_coursework_regulations(self, sample_chunks):
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.60)
        res = retriever.retrieve("coursework extension Form EXT-03C late penalty 5% 14 days", top_k=1)
        assert res[0].chunk.doc_id == "POL-ACAD-CW-03"

    def test_module_4_medical_guidelines(self, sample_chunks):
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.60)
        res = retriever.retrieve("SLMC medical certificate Discharge Summary University Medical Officer", top_k=2)
        doc_ids = [r.chunk.doc_id for r in res]
        assert "POL-MED-WF-04" in doc_ids or "POL-ACAD-EX-01" in doc_ids

    def test_module_5_student_handbook_legacy(self, sample_chunks):
        retriever = HybridExplainableRetriever(chunks=sample_chunks, alpha=0.60)
        res = retriever.retrieve("student handbook 2023 legacy Form EX-MC01 7 days", top_k=2)
        doc_ids = [r.chunk.doc_id for r in res]
        assert "HBK-ACAD-PR-05" in doc_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

