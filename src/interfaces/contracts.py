"""
Shared OOP Interfaces and Data Contracts for MediAcad Agentic RAG.
Satisfies DS205.3 High-Distinction OOP Architecture Requirements:
- Abstract Base Classes (ABC) & @abstractmethod
- Strongly-typed Dataclasses for zero-blocking parallel development
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ==========================================
# 1. DATA CONTRACTS (SHARED ACROSS 4 MEMBERS)
# ==========================================

@dataclass
class DocumentChunk:
    """Produced by Member 1 (Ingestion) and consumed by Member 2 (Retrieval)."""
    chunk_id: str
    doc_id: str
    doc_title: str
    policy_version: str          # e.g., "v4.0 (2025)" or "v1.8 (2023 Legacy)"
    effective_year: int          # e.g., 2025 or 2023
    page_number: int
    section_title: str
    clause_id: str               # e.g., "Clause 3.2" or "Table 2.1"
    chunk_type: str              # "clause", "table", or "callout"
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MathExplanationTrace:
    """Produced by Member 2 to prove mathematical mechanics in Viva & UI."""
    matched_query_terms: List[str]
    term_frequencies: Dict[str, float]
    idf_values: Dict[str, float]
    bm25_raw_score: float
    bm25_normalized_score: float
    query_vector_norm: float
    doc_vector_norm: float
    dot_product: float
    cosine_similarity: float
    alpha_weight: float
    hybrid_score: float


@dataclass
class RetrievalResult:
    """Produced by Member 2 and consumed by Member 3 (Agent) & Member 4 (UI/Eval)."""
    chunk: DocumentChunk
    sparse_score: float
    dense_score: float
    hybrid_score: float
    rank: int
    math_trace: Optional[MathExplanationTrace] = None


@dataclass
class AgentStepLog:
    """Single step in the Agentic Loop (Observation -> Reasoning -> Action -> Persistence)."""
    step_number: int
    phase: str                   # "OBSERVATION", "REASONING", "ACTION", "VERIFICATION", "PERSISTENCE"
    title: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyConflictAlert:
    """Detected when a 2023 Handbook rule conflicts with a 2025 Senate Policy."""
    conflict_id: str
    topic: str
    legacy_doc_id: str
    legacy_clause: str
    legacy_rule_summary: str
    authoritative_doc_id: str
    authoritative_clause: str
    authoritative_rule_summary: str
    resolution_explanation: str


@dataclass
class AgentResponse:
    """Final output from Member 3 (Agent) passed to Member 4 (UI & Evaluation)."""
    query: str
    detected_issues: List[str]
    sub_queries: List[str]
    final_answer: str
    action_checklist: List[str]
    required_documents: List[str]
    retrieved_contexts: List[RetrievalResult]
    reasoning_trace: List[AgentStepLog]
    conflict_alerts: List[PolicyConflictAlert] = field(default_factory=list)
    missing_info_warnings: List[str] = field(default_factory=list)
    confidence_status: str = "VERIFIED"


# ==========================================
# 2. ABSTRACT BASE CLASSES (OOP INTERFACES)
# ==========================================

class BaseChunker(ABC):
    """Member 1 implements this interface for structure-aware PDF chunking."""

    @abstractmethod
    def parse_and_chunk(self, pdf_directory: str) -> List[DocumentChunk]:
        pass


class BaseRetriever(ABC):
    """Member 2 implements this interface (Strategy Pattern for Sparse/Dense/Hybrid)."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        pass


class BaseAgent(ABC):
    """Member 3 implements this interface for the Explainable Agentic RAG workflow."""

    @abstractmethod
    def run(self, user_query: str, session_id: str = "default") -> AgentResponse:
        pass