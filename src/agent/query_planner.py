"""
Query Planner and Observation Engine for MediAcad Agentic RAG.
Member 3 - Agentic Architecture, OOP & LLM Integration.

Fulfills Step 1 (Observation) & Step 2 (Reasoning) of the Agentic Loop:
- Analyzes student input for medical event & academic issues (Exam, Attendance, Coursework, Progression).
- Decomposes complex multi-issue academic emergencies into targeted, module-specific sub-queries.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PlannedSubQuery:
    """Represents a decomposed sub-query mapped to a specific regulatory module."""
    module: str                  # "EXAMINATION", "ATTENDANCE", "COURSEWORK", "WELFARE", "PROGRESSION"
    sub_query: str
    target_doc_id: str
    rationale: str
    priority: int                # 1 (Highest) to 5


@dataclass
class ObservationResult:
    """Structured output of Step 1 (Observation) and Step 2 (Reasoning)."""
    original_query: str
    detected_issues: List[str]
    hospitalization_days: Optional[int] = None
    attendance_percentage: Optional[float] = None
    medical_condition: Optional[str] = None
    sub_queries: List[PlannedSubQuery] = field(default_factory=list)

    def get_query_strings(self) -> List[str]:
        """Convenience method returning plain text sub-queries for retrievers."""
        return [sq.sub_query for sq in self.sub_queries]


class QueryPlanner:
    """
    Observation & Reasoning Engine.
    Decomposes single unstructured student queries into structured multi-document queries.
    """

    # Keyword mappings for academic administrative domains
    EXAM_KEYWORDS = [
        "exam", "examination", "final exam", "deferred", "df grade",
        "missed exam", "i-med", "ex-mc01", "ec-01", "විභාග", "විභාගය"
    ]
    ATTENDANCE_KEYWORDS = [
        "attendance", "attendance percentage", "condonation", "absence",
        "att-mc02", "att-02", "72%", "80%", "65%", "පැමිණීම", "පැමිණීමේ"
    ]
    COURSEWORK_KEYWORDS = [
        "assignment", "coursework", "project", "deadline", "extension",
        "late penalty", "cw-ext03", "ext-03", "පැවරුම්", "පැවරුම"
    ]
    MEDICAL_KEYWORDS = [
        "hospital", "hospitalized", "hospitalization", "medical", "doctor",
        "dengue", "fever", "discharge ticket", "slmc", "umc", "med-04",
        "රෝහල්", "රෝහල්ගත", "වෛද්‍ය", "සහතික"
    ]
    PROGRESSION_KEYWORDS = [
        "progression", "semester", "prerequisite", "action plan",
        "next steps", "handbook", "prg-05", "ඉදිරි අධ්‍යයන", "පියවර"
    ]

    def observe(self, student_query: str) -> ObservationResult:
        """
        Step 1 (Observation): Extracts case facts, numbers, and affected academic domains.
        Step 2 (Reasoning): Formulates targeted retrieval queries across official policies.
        """
        query_lower = student_query.lower()

        # 1. Extract Hospitalization Days
        days_match = re.search(r"(\d+)\s*(?:day|days|දින|දවස්)", query_lower)
        hospitalization_days = int(days_match.group(1)) if days_match else None

        # 2. Extract Attendance Percentage
        att_match = re.search(r"(\d+(?:\.\d+)?)\s*%", query_lower)
        attendance_pct = float(att_match.group(1)) if att_match else None

        # 3. Detect Medical Condition
        condition = None
        for med_term in ["dengue", "fever", "accident", "surgery", "infection", "ඩෙංගු", "උණ"]:
            if med_term in query_lower:
                condition = med_term.capitalize()
                break
        if not condition and ("hospital" in query_lower or "රෝහල්" in query_lower):
            condition = "Hospitalized Inpatient Medical Event"

        # 4. Identify Affected Academic Domains
        detected_issues = []

        is_exam = any(k in query_lower for k in self.EXAM_KEYWORDS) or hospitalization_days is not None
        if is_exam:
            detected_issues.append("Missed Final Examination / Deferred Examination Eligibility")

        is_att = any(k in query_lower for k in self.ATTENDANCE_KEYWORDS) or attendance_pct is not None
        if is_att:
            pct_str = f" ({attendance_pct}%)" if attendance_pct else ""
            detected_issues.append(f"Attendance Shortfall Below 80% Requirement{pct_str}")

        is_cw = any(k in query_lower for k in self.COURSEWORK_KEYWORDS) or hospitalization_days is not None
        if is_cw:
            detected_issues.append("Missed Coursework/Assignment Deadline Extension Request")

        # Always verify medical documentation & progression for comprehensive guidance
        detected_issues.append("Official Medical Certification & Hospital Discharge Verification")
        detected_issues.append("Integrated Ordered Action Plan & Pre-requisite Progression")

        # 5. Formulate Multi-Step Sub-Queries (Step 2: Reasoning)
        sub_queries: List[PlannedSubQuery] = []

        # Medical Documentation Verification (Highest priority - prerequisite for all appeals)
        sub_queries.append(
            PlannedSubQuery(
                module="WELFARE",
                sub_query="hospitalization discharge ticket SLMC medical certificate UMC clearance verification Form MED-04-CLR counter 204",
                target_doc_id="POL-MED-WF-04",
                rationale="Identify mandatory hospital discharge ticket, SLMC doctor validation, and UMC Chief Medical Officer clearance procedure.",
                priority=1,
            )
        )

        # Examination Sub-Query
        if is_exam:
            days_str = f"{hospitalization_days} days hospitalization" if hospitalization_days else "hospitalization"
            sub_queries.append(
                PlannedSubQuery(
                    module="EXAMINATION",
                    sub_query=f"missed final examination {days_str} deferred exam I-Med Form EC-01 deadline 5 working days discharge",
                    target_doc_id="POL-ACAD-EX-01",
                    rationale="Retrieve authoritative regulations for missed final exams, Form EC-01 timeline, and hospitalization discharge rules.",
                    priority=2,
                )
            )

        # Attendance Sub-Query
        if is_att:
            att_val_str = f"{attendance_pct}%" if attendance_pct else "72% attendance"
            sub_queries.append(
                PlannedSubQuery(
                    module="ATTENDANCE",
                    sub_query=f"attendance condonation {att_val_str} medical absence Form ATT-02A Form ATT-02B 65 percent dean waiver",
                    target_doc_id="POL-ACAD-AT-02",
                    rationale="Check condonation threshold (70-79.9% HOD approval) vs exceptional hospitalization waiver (65-69.9% Dean approval).",
                    priority=3,
                )
            )

        # Coursework Extension Sub-Query
        if is_cw:
            sub_queries.append(
                PlannedSubQuery(
                    module="COURSEWORK",
                    sub_query="coursework assignment extension Tier 3 hospitalization Form EXT-03C late penalty waiver 5 percent per day",
                    target_doc_id="POL-ACAD-CW-03",
                    rationale="Determine retrospective coursework extension limits (up to 14 days) and late submission penalty waiver.",
                    priority=4,
                )
            )

        # Progression & Action Plan Sub-Query
        sub_queries.append(
            PlannedSubQuery(
                module="PROGRESSION",
                sub_query="10-day hospitalization consolidated action plan Form PRG-05-CASE provisional progression pre-requisites",
                target_doc_id="HBK-ACAD-PR-05",
                rationale="Extract official ordered 6-step roadmap and check for superseded 2023 rules vs 2025 updates.",
                priority=5,
            )
        )

        return ObservationResult(
            original_query=student_query,
            detected_issues=detected_issues,
            hospitalization_days=hospitalization_days,
            attendance_percentage=attendance_pct,
            medical_condition=condition,
            sub_queries=sub_queries,
        )
