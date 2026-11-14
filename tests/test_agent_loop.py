"""
Comprehensive Test Suite for MediAcad Agentic RAG Architecture.
Member 3 - Agentic Architecture, OOP & LLM Integration.

Tests:
1. BaseAgent ABC interface & inheritance compliance.
2. AgentState lifecycle, step logging & session persistence.
3. QueryPlanner observation, entity extraction & sub-query decomposition.
4. EvidenceVerifier confidence scoring, missing-evidence guardrails & 2023/2025 conflict detection.
5. MockRetriever zero-blocking Day 1 data contract compliance.
6. MediAcadAgent 6-Step Agentic Loop end-to-end execution.
7. Constructor Dependency Injection.
"""

import json
from pathlib import Path
import pytest

from src.interfaces.base_agent import BaseAgent
from src.interfaces.contracts import (
    AgentResponse,
    DocumentChunk,
    RetrievalResult,
    PolicyConflictAlert,
)
from src.agent.agent_state import AgentState, PersistentSessionLogger
from src.agent.query_planner import QueryPlanner, ObservationResult
from src.agent.evidence_verifier import EvidenceVerifier
from src.agent.mediacad_agent import MediAcadAgent, MockRetriever


# ===================================================================
# 1. Interface & Inheritance Tests
# ===================================================================

def test_base_agent_interface_inheritance():
    """Verify MediAcadAgent inherits from BaseAgent and fulfills OOP contracts."""
    assert issubclass(MediAcadAgent, BaseAgent)
    agent = MediAcadAgent()
    assert hasattr(agent, "run")
    assert hasattr(agent, "run_case")


# ===================================================================
# 2. AgentState & Persistence Tests
# ===================================================================

def test_agent_state_step_logging_and_conversion():
    """Verify AgentState records execution phases and converts to AgentResponse."""
    state = AgentState(session_id="test_sess_01", original_query="Missed final exam due to fever")
    state.add_step(
        phase="OBSERVATION",
        title="Detected Fever",
        description="Identified 1 medical issue",
        details={"condition": "fever"},
    )
    assert len(state.reasoning_steps) == 1
    assert state.reasoning_steps[0].step_number == 1
    assert state.reasoning_steps[0].phase == "OBSERVATION"

    # Test conversion to response
    state.final_answer = "Apply for Deferred Exam via Form EC-01"
    response = state.to_agent_response()
    assert isinstance(response, AgentResponse)
    assert response.query == "Missed final exam due to fever"
    assert response.final_answer == "Apply for Deferred Exam via Form EC-01"


def test_persistent_session_logger(tmp_path):
    """Verify session logger serializes state to disk and loads back cleanly."""
    logger = PersistentSessionLogger(storage_dir=tmp_path)
    state = AgentState(session_id="unit_test_99", original_query="Test query")
    state.identified_issues = ["Exam Issue", "Attendance Issue"]
    saved_path = logger.save(state)

    assert saved_path.exists()
    loaded_data = logger.load("unit_test_99")
    assert loaded_data is not None
    assert loaded_data["session_id"] == "unit_test_99"
    assert loaded_data["identified_issues"] == ["Exam Issue", "Attendance Issue"]
    assert "unit_test_99" in logger.list_sessions()


# ===================================================================
# 3. QueryPlanner Observation & Reasoning Tests
# ===================================================================

def test_query_planner_observation_and_decomposition():
    """Verify natural language entity extraction and multi-query decomposition."""
    planner = QueryPlanner()
    query = (
        "මම ඩෙංගු හැදිලා දවස් 10ක් රෝහල්ගත වුණා (Hospitalized for 10 days). "
        "Final exam එක මගහැරුණා, Attendance එක 72% දක්වා අඩුවුණා, Assignment එක ප්‍රමාද වුණා."
    )
    obs = planner.observe(query)

    assert isinstance(obs, ObservationResult)
    assert obs.hospitalization_days == 10
    assert obs.attendance_percentage == 72.0
    assert obs.medical_condition in ["Dengue", "ඩෙංගු"]
    assert len(obs.detected_issues) >= 3

    # Check generated sub-queries
    sub_queries = obs.sub_queries
    assert len(sub_queries) == 5
    modules = [sq.module for sq in sub_queries]
    assert "EXAMINATION" in modules
    assert "ATTENDANCE" in modules
    assert "COURSEWORK" in modules
    assert "WELFARE" in modules
    assert "PROGRESSION" in modules


# ===================================================================
# 4. EvidenceVerifier & Conflict Detection Tests
# ===================================================================

def test_conflict_detection_legacy_vs_2025():
    """Verify detection of policy conflicts between 2023 Legacy Handbook and 2025 Senate Policies."""
    verifier = EvidenceVerifier()

    # Create dummy chunks representing a legacy policy conflict
    legacy_chunk = DocumentChunk(
        chunk_id="HBK-2023-TEST",
        doc_id="HBK-ACAD-PR-05",
        doc_title="Academic Progression Regulations",
        policy_version="v1.8 (2023 Legacy Edition)",
        effective_year=2023,
        page_number=4,
        section_title="Legacy Rules",
        clause_id="Clause 5.1",
        chunk_type="clause",
        text="Clause 5.1 states missed exam medical certificates may be submitted within 7 calendar days. "
             "Clause 5.2 states attendance condonation is allowed down to 60%. "
             "Clause 5.3 states late coursework penalty is 2% per day up to 10 days. "
             "Clause 5.4 states lecturers may grant 21-day extensions.",
    )
    res = RetrievalResult(
        chunk=legacy_chunk,
        sparse_score=0.8,
        dense_score=0.8,
        hybrid_score=0.8,
        rank=1,
    )

    conflicts = verifier.detect_conflicts([res])
    assert len(conflicts) >= 3

    conflict_ids = [c.conflict_id for c in conflicts]
    assert "CONF-EXAM-01" in conflict_ids  # 7 calendar days vs 5 working days
    assert "CONF-ATT-02" in conflict_ids   # 60% vs 65% attendance floor
    assert "CONF-CW-03" in conflict_ids    # 2% vs 5% daily penalty
    assert "CONF-EXT-04" in conflict_ids   # 21-day lecturer vs 14-day EC Panel


def test_evidence_verifier_missing_evidence_guardrails():
    """Verify verifier raises warnings and generates re-retrieval queries when evidence is insufficient."""
    verifier = EvidenceVerifier()
    # Empty retrieval input
    chunks, warnings, re_queries = verifier.verify_evidence([], ["Exam Issue"])
    assert len(warnings) > 0
    assert "CRITICAL" in warnings[0]
    assert len(re_queries) > 0


# ===================================================================
# 5. MockRetriever Zero-Blocking Tests
# ===================================================================

def test_mock_retriever_standalone_operation():
    """Verify MockRetriever loads sample chunks cleanly without external services."""
    retriever = MockRetriever()
    assert len(retriever.chunks) >= 5

    results = retriever.retrieve("missed final examination hospitalization 10 days", top_k=3)
    assert len(results) > 0
    assert results[0].math_trace is not None
    assert results[0].hybrid_score > 0.0


# ===================================================================
# 6. End-to-End 6-Step Agentic Loop Execution Tests
# ===================================================================

def test_mediacad_agent_end_to_end_execution(tmp_path):
    """
    Executes the real 10-day hospitalization scenario through all 6 agentic phases:
    Observation -> Reasoning -> Action -> Persistence -> Verification -> Grounded Synthesis.
    """
    logger = PersistentSessionLogger(storage_dir=tmp_path)
    agent = MediAcadAgent(session_logger=logger)

    student_query = (
        "හිතන්න මම ඩෙංගු හැදිලා දවස් 10ක් රෝහල්ගත වෙලා හිටියා. Final Exam එක මගහැරුණා. "
        "Attendance එක 72% දක්වා අඩුවුණා. Assignment එක භාරදෙන්න බැරි වුණා. මම මොකද කරන්නේ?"
    )

    response: AgentResponse = agent.run(student_query, session_id="case_10day_dengue")

    # 1. Verify contract type
    assert isinstance(response, AgentResponse)

    # 2. Verify all 6 phases recorded in reasoning trace
    phases = [step.phase for step in response.reasoning_trace]
    assert "OBSERVATION" in phases
    assert "REASONING" in phases
    assert "ACTION" in phases
    assert "PERSISTENCE" in phases
    assert "VERIFICATION" in phases
    assert "GROUNDED_SYNTHESIS" in phases

    # 3. Verify Detected Issues
    assert len(response.detected_issues) >= 3
    assert len(response.sub_queries) == 5

    # 4. Verify Retrieved Contexts
    assert len(response.retrieved_contexts) > 0

    # 5. Verify Policy Conflict Alerts
    assert len(response.conflict_alerts) > 0
    for alert in response.conflict_alerts:
        assert isinstance(alert, PolicyConflictAlert)
        assert alert.authoritative_doc_id != ""

    # 6. Verify Ordered Action Checklist & Official Forms
    assert len(response.action_checklist) == 6
    assert any("Form EC-01" in item or "EC-01" in item for item in response.action_checklist)
    assert any("Form ATT-02A" in item or "ATT-02" in item for item in response.action_checklist)
    assert any("Form EXT-03C" in item or "EXT-03" in item for item in response.action_checklist)
    assert any("Form MED-04-CLR" in item or "MED-04" in item for item in response.action_checklist)
    assert any("Form PRG-05-CASE" in item or "PRG-05" in item for item in response.action_checklist)

    # 7. Verify Required Documents
    assert len(response.required_documents) >= 5

    # 8. Verify Grounded Citations in Final Answer
    assert "POL-ACAD-EX-01" in response.final_answer
    assert "POL-ACAD-AT-02" in response.final_answer
    assert "POL-ACAD-CW-03" in response.final_answer
    assert "POL-MED-WF-04" in response.final_answer
    assert "HBK-ACAD-PR-05" in response.final_answer
    assert response.confidence_status == "VERIFIED_GROUNDED"

    # 9. Verify Session Persistence to disk
    persisted_file = tmp_path / "session_case_10day_dengue.json"
    assert persisted_file.exists()
    content = json.loads(persisted_file.read_text(encoding="utf-8"))
    assert content["session_id"] == "case_10day_dengue"


def test_dependency_injection():
    """Verify that any custom retriever conforming to BaseRetriever can be injected."""
    class CustomRetriever(MockRetriever):
        def retrieve(self, query: str, top_k: int = 5):
            results = super().retrieve(query, top_k)
            for r in results:
                r.sparse_score = 0.999
            return results

    custom_retriever = CustomRetriever()
    agent = MediAcadAgent(retriever=custom_retriever)
    response = agent.run("Simple attendance query 72%", session_id="di_test")
    assert isinstance(response, AgentResponse)
    assert response.retrieved_contexts[0].sparse_score == 0.999
