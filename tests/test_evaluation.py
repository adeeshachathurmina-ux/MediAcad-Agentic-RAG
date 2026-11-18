"""
Automated Unit Tests for Member 4's Ground-Truth Dataset & RAG Triad Evaluation Engine.
"""

from src.evaluation.metrics import RAGTriadEvaluator, lexical_cosine_similarity
from src.evaluation.run_benchmark import run_full_benchmark


def test_ground_truth_dataset_has_32_cases():
    """Verify the ground-truth benchmark has >= 30 cases across all 5 categories."""
    evaluator = RAGTriadEvaluator()
    assert len(evaluator.ground_truth_cases) >= 30
    categories = {c["category"] for c in evaluator.ground_truth_cases}
    assert "Single-Policy Clause" in categories
    assert "Complex Table Lookup" in categories
    assert "Multi-Document Synthesis" in categories
    assert "Policy Conflict Resolution" in categories
    assert "Out-of-Scope & Guardrail" in categories


def test_lexical_cosine_similarity_math():
    """Verify L2-normalized cosine similarity bounds [0.0, 1.0]."""
    sim_identical = lexical_cosine_similarity(
        "Submit Form EC-01 within 5 working days",
        "Submit Form EC-01 within 5 working days",
    )
    sim_disjoint = lexical_cosine_similarity(
        "Form EC-01 examination deferral",
        "laboratory safety goggles chemistry",
    )
    assert sim_identical == 1.0
    assert 0.0 <= sim_disjoint < 0.3


def test_full_rag_triad_benchmark_execution():
    """Verify full 32-case benchmark executes and achieves >80% triad score."""
    report = run_full_benchmark()
    assert report.total_cases == 32
    assert 0.80 <= report.mean_context_relevance <= 1.0
    assert 0.85 <= report.mean_groundedness <= 1.0
    assert 0.75 <= report.mean_answer_relevance <= 1.0
    assert report.mean_overall_score >= 0.80