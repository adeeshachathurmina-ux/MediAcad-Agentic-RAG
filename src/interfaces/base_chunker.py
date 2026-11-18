"""
Abstract Base Class and Dataclass for Document Chunking in MediAcad.
Member 1: Ingestion Pipeline Interface.
Satisfies DS205.3 OOP Architecture Requirements:
- Abstract Base Classes (ABC) & @abstractmethod
- Strongly-typed Dataclass (DocumentChunk)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.interfaces.contracts import DocumentChunk, BaseChunker

__all__ = ["BaseChunker", "DocumentChunk"]
