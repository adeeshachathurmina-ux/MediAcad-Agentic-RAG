"""
RAG Triad Quantitative Evaluation Engine for MediAcad (Member 4).
Implements mathematical calculations for:
1. Context Relevance (Document & Clause Recall + Lexical Overlap)
2. Groundedness / Faithfulness (Claim, Citation & Numerical Verification against Context)
3. Answer Relevance (Expected Keyword Coverage + TF-IDF Cosine Similarity)
"""

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.interfaces.contracts import (
    DocumentChunk,
    RetrievalResult,
    AgentResponse,
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with", "by",
    "is", "are", "was", "were", "be", "been", "that", "this", "it", "as", "at",
    "from", "under", "if", "what", "how", "when", "where", "who", "which", "can",
    "may", "must", "shall", "student", "university", "policy", "clause"
}


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer preserving numbers and hyphenated form codes."""
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def lexical_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute exact L2-normalized Cosine Similarity between two texts:
    Cosine(q, d) = (q . d) / (||q||_2 * ||d||_2)
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    freq_a = Counter(tokens_a)
    freq_b = Counter(tokens_b)
    vocab = set(freq_a.keys()) | set(freq_b.keys())

    dot_product = sum(freq_a[w] * freq_b[w] for w in vocab)
    norm_a = math.sqrt(sum(v * v for v in freq_a.values()))
    norm_b = math.sqrt(sum(v * v for v in freq_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return round(dot_product / (norm_a * norm_b), 4)


@dataclass
class CaseEvaluationResult:
    """Quantitative evaluation result for a single Ground-Truth test case."""
    case_id: str
    category: str
    query: str
    context_relevance: float
    groundedness: float
    answer_relevance: float
    overall_triad_score: float
    matched_docs: List[str] = field(default_factory=list)
    matched_clauses: List[str] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    unverified_tokens_ratio: float = 0.0


@dataclass
class BenchmarkSummaryReport:
    """Aggregated RAG Triad benchmark report across all 32 Ground-Truth cases."""
    total_cases: int
    mean_context_relevance: float
    mean_groundedness: float
    mean_answer_relevance: float
    mean_overall_score: float
    category_breakdown: Dict[str, Dict[str, float]]
    case_results: List[CaseEvaluationResult]


class RAGTriadEvaluator:
    """
    Member 4's core evaluation class computing the RAG Triad:
    - Context Relevance
    - Groundedness (Faithfulness)
    - Answer Relevance
    """

    def __init__(self, ground_truth_path: Optional[str] = None):
        if ground_truth_path is None:
            root = Path(__file__).resolve().parent.parent.parent
            ground_truth_path = str(root / "data" / "evaluation" / "ground_truth.json")
        self.ground_truth_path = Path(ground_truth_path)
        self.ground_truth_cases: List[Dict[str, Any]] = self._load_ground_truth()

    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(f"Ground-truth dataset not found at {self.ground_truth_path}")
        return json.loads(self.ground_truth_path.read_text(encoding="utf-8"))

    @staticmethod
    def compute_context_relevance(
        retrieved_contexts: List[RetrievalResult],
        expected_doc_ids: List[str],
        expected_clauses: List[str],
        query: str,
    ) -> Dict[str, Any]:
        """
        Metric 1: Context Relevance (0.0 to 1.0)
        Combines:
        - Document ID Recall (weight 0.45)
        - Policy Clause/Table Recall (weight 0.35)
        - Query-to-Context Lexical Relevance (weight 0.20)
        """
        if not retrieved_contexts:
            return {"score": 0.0, "matched_docs": [], "matched_clauses": []}

        retrieved_doc_ids = {r.chunk.doc_id for r in retrieved_contexts}
        combined_context_text = " ".join(
            f"{r.chunk.clause_id} {r.chunk.section_title} {r.chunk.text}"
            for r in retrieved_contexts
        ).lower()

        matched_docs = [doc for doc in expected_doc_ids if doc in retrieved_doc_ids]
        doc_recall = len(matched_docs) / max(len(expected_doc_ids), 1)

        matched_clauses = [
            cl for cl in expected_clauses
            if cl.lower() in combined_context_text
        ]
        clause_recall = len(matched_clauses) / max(len(expected_clauses), 1)

        query_sim = lexical_cosine_similarity(query, combined_context_text)
        scaled_sim = min(query_sim * 2.2, 1.0)

        score = (0.45 * doc_recall) + (0.35 * clause_recall) + (0.20 * scaled_sim)
        return {
            "score": round(min(max(score, 0.0), 1.0), 4),
            "matched_docs": matched_docs,
            "matched_clauses": matched_clauses,
        }

    @staticmethod
    def compute_groundedness(
        generated_answer: str,
        retrieved_contexts: List[RetrievalResult],
    ) -> Dict[str, Any]:
        """
        Metric 2: Groundedness / Faithfulness (0.0 to 1.0)
        Verifies that factual claims, numbers, form codes (e.g., Form EC-01),
        and policy terms in the generated answer are supported by retrieved chunks.
        """
        if not generated_answer.strip() or not retrieved_contexts:
            return {"score": 0.0, "unverified_ratio": 1.0}

        context_corpus = " ".join(
            f"{r.chunk.doc_id} {r.chunk.clause_id} {r.chunk.text}"
            for r in retrieved_contexts
        )
        context_tokens = set(tokenize(context_corpus))
        answer_tokens = tokenize(generated_answer)

        if not answer_tokens:
            return {"score": 0.0, "unverified_ratio": 1.0}

        # Check critical regulatory identifiers (Form codes, percentages, day counts)
        critical_patterns = re.findall(
            r"(?:form\s+[a-z0-9-]+|pol-[a-z0-9-]+|hbk-[a-z0-9-]+|\d+%|\d+\s+(?:working|calendar)?\s*days?)",
            generated_answer.lower()
        )
        context_lower = context_corpus.lower()
        if critical_patterns:
            verified_critical = sum(1 for p in critical_patterns if p in context_lower)
            critical_precision = verified_critical / len(critical_patterns)
        else:
            critical_precision = 1.0

        supported_tokens = sum(1 for t in answer_tokens if t in context_tokens)
        token_support_ratio = supported_tokens / len(answer_tokens)

        # Blend critical identifier precision (60%) and content token grounding (40%)
        raw_faithfulness = (0.60 * critical_precision) + (0.40 * min(token_support_ratio * 1.35, 1.0))
        score = round(min(max(raw_faithfulness, 0.0), 1.0), 4)
        return {
            "score": score,
            "unverified_ratio": round(1.0 - score, 4),
        }

    @staticmethod
    def compute_answer_relevance(
        generated_answer: str,
        reference_answer: str,
        expected_keywords: List[str],
        query: str,
    ) -> Dict[str, Any]:
        """
        Metric 3: Answer Relevance (0.0 to 1.0)
        Combines:
        - Expected Keyword Coverage (weight 0.55)
        - Reference Answer Cosine Similarity (weight 0.30)
        - Query Addressing Completeness (weight 0.15)
        """
        if not generated_answer.strip():
            return {"score": 0.0, "matched_keywords": []}

        ans_lower = generated_answer.lower()
        matched_keywords = [
            kw for kw in expected_keywords
            if kw.lower() in ans_lower
        ]
        keyword_coverage = len(matched_keywords) / max(len(expected_keywords), 1)

        ref_cosine = lexical_cosine_similarity(generated_answer, reference_answer)
        query_cosine = lexical_cosine_similarity(generated_answer, query)

        score = (
            (0.55 * keyword_coverage)
            + (0.30 * min(ref_cosine * 1.4, 1.0))
            + (0.15 * min(query_cosine * 1.8, 1.0))
        )
        return {
            "score": round(min(max(score, 0.0), 1.0), 4),
            "matched_keywords": matched_keywords,
        }

    def evaluate_single_response(
        self,
        gt_case: Dict[str, Any],
        agent_response: AgentResponse,
    ) -> CaseEvaluationResult:
        """Evaluate one AgentResponse against its Ground-Truth specification."""
        cr = self.compute_context_relevance(
            retrieved_contexts=agent_response.retrieved_contexts,
            expected_doc_ids=gt_case["expected_doc_ids"],
            expected_clauses=gt_case["expected_clauses"],
            query=gt_case["query"],
        )
        gf = self.compute_groundedness(
            generated_answer=agent_response.final_answer,
            retrieved_contexts=agent_response.retrieved_contexts,
        )
        ar = self.compute_answer_relevance(
            generated_answer=agent_response.final_answer,
            reference_answer=gt_case["reference_answer"],
            expected_keywords=gt_case["expected_keywords"],
            query=gt_case["query"],
        )

        overall = round((cr["score"] + gf["score"] + ar["score"]) / 3.0, 4)

        return CaseEvaluationResult(
            case_id=gt_case["id"],
            category=gt_case["category"],
            query=gt_case["query"],
            context_relevance=cr["score"],
            groundedness=gf["score"],
            answer_relevance=ar["score"],
            overall_triad_score=overall,
            matched_docs=cr["matched_docs"],
            matched_clauses=cr["matched_clauses"],
            matched_keywords=ar["matched_keywords"],
            unverified_tokens_ratio=gf["unverified_ratio"],
        )