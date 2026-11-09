"""
Retrieval Package for MediAcad Agentic RAG.
Member 2: Hybrid Retrieval & Mathematical Explainability.
"""

from src.retrieval.math_mechanics import (
    vector_magnitude,
    dot_product,
    l2_normalize,
    cosine_similarity,
    batch_cosine_similarity,
    min_max_normalize,
    hybrid_fuse_score,
)
from src.retrieval.sparse_engine import (
    TFIDFAndBM25Engine,
    tokenize,
)
from src.retrieval.hybrid_retriever import (
    HybridExplainableRetriever,
)

__all__ = [
    "vector_magnitude",
    "dot_product",
    "l2_normalize",
    "cosine_similarity",
    "batch_cosine_similarity",
    "min_max_normalize",
    "hybrid_fuse_score",
    "TFIDFAndBM25Engine",
    "tokenize",
    "HybridExplainableRetriever",
]

