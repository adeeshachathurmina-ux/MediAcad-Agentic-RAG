"""
Base Agent Interface for MediAcad Agentic RAG.
Member 3 - Agentic Architecture, OOP & LLM Integration.

Defines the Abstract Base Class (BaseAgent) and re-exports core contracts
for the Explainable Agentic Loop (Observation -> Reasoning -> Action -> Persistence).
"""

from abc import ABC, abstractmethod
from typing import List, Optional

# Import shared contracts to preserve single source of truth
from src.interfaces.contracts import (
    BaseAgent as _BaseAgentContract,
    AgentResponse,
    AgentStepLog,
    PolicyConflictAlert,
    RetrievalResult,
    DocumentChunk,
)


class BaseAgent(_BaseAgentContract):
    """
    Abstract Base Class for MediAcad Case Resolution Agent.
    Implements Template/Strategy Pattern for 6-step Agentic Loop.
    """

    @abstractmethod
    def run(self, user_query: str, session_id: str = "default") -> AgentResponse:
        """
        Executes the full 6-step Agentic Loop:
        1. Observation (Identify medical & academic issues)
        2. Reasoning (Decompose into module sub-queries)
        3. Action (Retrieve relevant regulatory chunks)
        4. Persistence (Record state & mathematical traces)
        5. Verification (Conflict detection & missing evidence guardrails)
        6. Synthesis (Grounded multi-step case resolution with citations)
        """
        pass

    def run_case(self, student_query: str) -> AgentResponse:
        """
        Convenience alias for run() adhering to integration specification.
        Allows seamless plug-and-play call from UI (Member 4) or main pipeline.
        """
        return self.run(user_query=student_query, session_id="case_session")


__all__ = [
    "BaseAgent",
    "AgentResponse",
    "AgentStepLog",
    "PolicyConflictAlert",
    "RetrievalResult",
    "DocumentChunk",
]
