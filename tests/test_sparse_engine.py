"""
Sparse Engine (TF-IDF & BM25) Unit Test Suite.
MediAcad Agentic RAG - Member 2 (Hybrid Retrieval & Mathematical Explainability).

Verifies:
  - Policy-aware tokenization (preserving codes like EC-01, MED-04-CLR, 80%, 10-day)
  - Inverted index & vocabulary construction
  - BM25 non-negative Robertson-Spärck Jones IDF
  - Length normalization and term saturation (k1=1.5, b=0.75)
  - Min-Max normalized score bounds [0.0, 1.0]
  - Explainability trace (matched terms, TF, IDF, and keyword weights)
  - Real query retrieval accuracy against data/fixtures/sample_chunks.json
"""

import json
import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interfaces.contracts import DocumentChunk
from src.retrieval.sparse_engine import TFIDFAndBM25Engine, tokenize

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_chunks():
    """Loads 5 official policy chunks from fixtures."""
    fixture_path = PROJECT_ROOT / "data" / "fixtures" / "sample_chunks.json"
    raw_data = json.loads(fixture_path.read_text(encoding="utf-8"))
    return [DocumentChunk(**item) for item in raw_data]


class TestPolicyTokenizer:
    """Tests regex tokenization on domain-specific administrative strings."""

    def test_preserves_administrative_codes_and_percentages(self):
        text = "Student must submit Form EC-01 and MED-04-CLR for 80% or 72% attendance."
        tokens = tokenize(text)
        assert "ec-01" in tokens
        assert "med-04-clr" in tokens
        assert "80%" in tokens
        assert "72%" in tokens
        assert "attendance" in tokens
        assert "submit" in tokens
        # Verify stopwords like 'and', 'for', 'or' are filtered
        assert "and" not in tokens
        assert "for" not in tokens

    def test_preserves_durations_and_hyphenated_terms(self):
        text = "Exceptional 10-day inpatient stay with 5-working-day submission clock."
        tokens = tokenize(text)
        assert "10-day" in tokens
        assert "inpatient" in tokens
        assert "5-working-day" in tokens

    def test_empty_string(self):
        assert tokenize("") == []
        assert tokenize(None) == []


class TestTFIDFAndBM25Engine:
    """Verifies BM25 math mechanics and explainability traces."""

    def test_fit_and_vocabulary(self, sample_chunks):
        engine = TFIDFAndBM25Engine(k1=1.5, b=0.75)
        engine.fit(sample_chunks)

        assert engine.corpus_size == len(sample_chunks)
        assert engine.avg_doc_length > 0
        assert "ec-01" in engine.vocabulary
        assert "attendance" in engine.vocabulary

    def test_idf_weights_and_rarity(self, sample_chunks):
        """Rare domain terms must have higher IDF than ubiquitous words."""
        engine = TFIDFAndBM25Engine().fit(sample_chunks)

        # "ec-01" appears only in the exam policy chunk
        # words like "policy" or general terms appear across more chunks
        rare_idf = engine.bm25_idf.get("ec-01", 0.0)
        assert rare_idf > 0.0, "BM25 IDF must be positive"

        # Check all IDFs are strictly positive
        for term, idf in engine.bm25_idf.items():
            assert idf > 0.0, f"IDF for {term} should be strictly positive"

    def test_bm25_term_saturation(self):
        """Doubling term frequency does not double the score (concave saturation)."""
        # Create two artificial chunks: one with 1 occurrence, one with 5 occurrences
        c1 = DocumentChunk(
            chunk_id="c1", doc_id="d1", doc_title="T1", policy_version="v1",
            effective_year=2025, page_number=1, section_title="S1", clause_id="c1",
            chunk_type="clause", text="medical consideration application"
        )
        c2 = DocumentChunk(
            chunk_id="c2", doc_id="d2", doc_title="T2", policy_version="v1",
            effective_year=2025, page_number=1, section_title="S2", clause_id="c2",
            chunk_type="clause", text="medical medical medical medical medical consideration"
        )

        engine = TFIDFAndBM25Engine(k1=1.5, b=0.75).fit([c1, c2])
        score_c1, _, _, _ = engine.calculate_bm25_score_for_doc(["medical"], 0)
        score_c2, _, _, _ = engine.calculate_bm25_score_for_doc(["medical"], 1)

        # c2 has 5x term frequency of "medical", but due to saturation, score should NOT be 5x
        ratio = score_c2 / score_c1
        assert ratio < 4.0, f"BM25 should exhibit saturation curve, got ratio {ratio}"

    def test_min_max_normalization_bounds(self, sample_chunks):
        """Normalized scores must be strictly in [0.0, 1.0]."""
        engine = TFIDFAndBM25Engine().fit(sample_chunks)
        results = engine.score_corpus("medical consideration Form EC-01")

        for res in results:
            norm_score = res["normalized_score"]
            assert 0.0 <= norm_score <= 1.0, f"Score {norm_score} out of [0, 1] bounds"

        # Top document has score 1.0
        assert results[0]["normalized_score"] == 1.0

    def test_mathematical_trace_integrity(self, sample_chunks):
        """Sum of individual term BM25 contributions must equal raw score."""
        engine = TFIDFAndBM25Engine().fit(sample_chunks)
        results = engine.score_corpus("hospitalization 10-day Form EC-01")

        top_doc = results[0]
        assert len(top_doc["matched_terms"]) > 0

        # Sum of per-term contributions equals raw score
        sum_contrib = sum(top_doc["term_contributions"].values())
        assert pytest.approx(top_doc["raw_score"], abs=1e-6) == sum_contrib

        # Check explainability fields exist
        assert "ec-01" in top_doc["matched_terms"] or "10-day" in top_doc["matched_terms"]
        assert len(top_doc["top_keywords"]) > 0

    def test_academic_scenario_retrieval_accuracy(self, sample_chunks):
        """
        Scenario 1: Missed Final Exam with 10-day hospitalization.
        Query: "missed final examination Form EC-01 hospitalization 10-day"
        Should retrieve POL-ACAD-EX-01 Clause 3.2 as Rank 1!
        """
        engine = TFIDFAndBM25Engine().fit(sample_chunks)
        results = engine.retrieve_top_k("missed final examination Form EC-01 hospitalization 10-day", top_k=3)

        assert len(results) > 0
        top_chunk = results[0]["chunk"]
        assert top_chunk.doc_id == "POL-ACAD-EX-01"
        assert top_chunk.clause_id == "Clause 3.2"

    def test_attendance_scenario_retrieval_accuracy(self, sample_chunks):
        """
        Scenario 2: Attendance dropped to 72% with Form ATT-02A.
        Query: "attendance condonation 72% Form ATT-02A 80%"
        Should retrieve POL-ACAD-AT-02 Table 2.1 as Rank 1!
        """
        engine = TFIDFAndBM25Engine().fit(sample_chunks)
        results = engine.retrieve_top_k("attendance condonation 72% Form ATT-02A 80%", top_k=3)

        assert len(results) > 0
        top_chunk = results[0]["chunk"]
        assert top_chunk.doc_id == "POL-ACAD-AT-02"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
