"""
MediAcad Agent Module.
Member 3 - Agentic Architecture, OOP & LLM Integration.

Exports the core agent components:
- BaseAgent (OOP Interface)
- MediAcadAgent (6-Step Agentic Loop Implementation)
- MockRetriever (Zero-blocking Day 1 mock)
- AgentState & PersistentSessionLogger (Persistence & Auditability)
- QueryPlanner & PlannedSubQuery (Observation & Reasoning)
- EvidenceVerifier (Verification, Conflict Detection & Guardrails)
"""

from src.interfaces.base_agent import BaseAgent
from src.agent.agent_state import AgentState, PersistentSessionLogger
from src.agent.query_planner import QueryPlanner, PlannedSubQuery, ObservationResult
from src.agent.evidence_verifier import EvidenceVerifier
from src.agent.mediacad_agent import MediAcadAgent, MockRetriever

__all__ = [
    "BaseAgent",
    "MediAcadAgent",
    "MockRetriever",
    "AgentState",
    "PersistentSessionLogger",
    "QueryPlanner",
    "PlannedSubQuery",
    "ObservationResult",
    "EvidenceVerifier",
]
