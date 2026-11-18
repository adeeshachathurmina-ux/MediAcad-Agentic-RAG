"""
Base Retriever Interface for MediAcad Agentic RAG.
Member 2 - Hybrid Retrieval & Mathematical Explainability.

Defines the Abstract Base Class (BaseRetriever) adhering to the Strategy Pattern
for Sparse (BM25/TF-IDF), Dense (ChromaDB/SentenceTransformers), and Hybrid retrieval.
"""

from abc import ABC, abstractmethod
from typing import List

# Import shared contracts to preserve single source of truth
from src.interfaces.contracts import (
    BaseRetriever,
    RetrievalResult,
    MathExplanationTrace,
    DocumentChunk,
)

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "MathExplanationTrace",
    "DocumentChunk",
]

