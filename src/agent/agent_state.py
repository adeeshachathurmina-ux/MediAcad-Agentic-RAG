"""
Agent State & Persistence Schema for MediAcad Agentic RAG.
Member 3 - Agentic Architecture, OOP & LLM Integration.

Fulfills Step 1 & Step 4 of the Agentic Loop:
- Dataclass schema managing full state across Observation, Reasoning, Action, Verification, and Persistence.
- PersistentSessionLogger for saving white-box JSON execution traces.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.interfaces.contracts import (
    AgentResponse,
    AgentStepLog,
    PolicyConflictAlert,
    RetrievalResult,
)


@dataclass
class AgentState:
    """
    Central state container for MediAcad Agent execution lifecycle.
    Maintains transparency, auditability, and replay capability.
    """
    session_id: str
    original_query: str
    identified_issues: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    retrieved_chunks: List[RetrievalResult] = field(default_factory=list)
    math_traces: List[Dict[str, Any]] = field(default_factory=list)
    conflict_alerts: List[PolicyConflictAlert] = field(default_factory=list)
    missing_evidence_warnings: List[str] = field(default_factory=list)
    action_checklist: List[str] = field(default_factory=list)
    required_documents: List[str] = field(default_factory=list)
    final_answer: str = ""
    reasoning_steps: List[AgentStepLog] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    confidence_status: str = "INITIALIZED"

    def add_step(
        self,
        phase: str,
        title: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AgentStepLog:
        """Appends an auditable step log to the reasoning trace."""
        step = AgentStepLog(
            step_number=len(self.reasoning_steps) + 1,
            phase=phase,
            title=title,
            description=description,
            details=details or {},
        )
        self.reasoning_steps.append(step)
        return step

    def to_dict(self) -> Dict[str, Any]:
        """Converts state into a JSON-serializable dictionary."""
        data = asdict(self)
        return data

    def to_agent_response(self) -> AgentResponse:
        """Converts internal state to the standardized shared AgentResponse contract."""
        return AgentResponse(
            query=self.original_query,
            detected_issues=list(self.identified_issues),
            sub_queries=list(self.sub_queries),
            final_answer=self.final_answer,
            action_checklist=list(self.action_checklist),
            required_documents=list(self.required_documents),
            retrieved_contexts=list(self.retrieved_chunks),
            reasoning_trace=list(self.reasoning_steps),
            conflict_alerts=list(self.conflict_alerts),
            missing_info_warnings=list(self.missing_evidence_warnings),
            confidence_status=self.confidence_status,
        )


class PersistentSessionLogger:
    """
    Session persistence manager storing state logs on disk as JSON.
    Enables White-Box mathematical explainability & compliance auditing.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            self.storage_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sessions"
        else:
            self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: AgentState) -> Path:
        """Serializes AgentState into a structured JSON file."""
        file_path = self.storage_dir / f"session_{state.session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
        return file_path

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Loads a session JSON file by session_id."""
        file_path = self.storage_dir / f"session_{session_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_sessions(self) -> List[str]:
        """Returns a list of all recorded session IDs."""
        return [
            p.stem.replace("session_", "")
            for p in self.storage_dir.glob("session_*.json")
        ]
