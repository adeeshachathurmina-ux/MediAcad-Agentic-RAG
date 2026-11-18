"""
Evidence Verifier, Missing Evidence Guardrails & Conflict Detection Engine.
Member 3 - Agentic Architecture, OOP & LLM Integration.

Fulfills Step 3 (Evidence Verification), Step 4 (Policy Conflict Detection),
and Step 5 (Missing Evidence Guardrails) of the MediAcad Agentic Loop.
"""

from typing import List, Dict, Any, Tuple, Optional
from src.interfaces.contracts import (
    RetrievalResult,
    PolicyConflictAlert,
    DocumentChunk,
)


class EvidenceVerifier:
    """
    Verifies retrieved evidence quality, checks for missing regulatory data,
    and identifies conflicts between 2023 Legacy Handbook and 2025 Senate Policies.
    """

    CONFIDENCE_THRESHOLD: float = 0.40

    KNOWN_CONFLICT_RULES = [
        {
            "conflict_id": "CONF-EXAM-01",
            "topic": "Missed Exam Medical Submission Deadline",
            "trigger_keywords": ["7 calendar days", "7 days", "calendar days", "submission clock", "clause 5.1"],
            "legacy_doc_id": "HBK-ACAD-PR-05 (v1.8 / 2023)",
            "legacy_clause": "Clause 5.1",
            "legacy_rule_summary": "Legacy 2023 handbook allowed medical submissions within 7 calendar days without distinguishing hospitalization discharge dates.",
            "authoritative_doc_id": "POL-ACAD-EX-01 (v4.0 / 2025)",
            "authoritative_clause": "Clause 3.2",
            "authoritative_rule_summary": "Mandatory 2025 Senate Revision enforces strict 5 working days from discharge date for hospitalized students submitting Form EC-01.",
            "resolution_explanation": "POL-ACAD-EX-01 (v4.0 / 2025) strictly supersedes all legacy timelines. The student has 5 working days from official hospital discharge.",
        },
        {
            "conflict_id": "CONF-ATT-02",
            "topic": "Attendance Condonation Minimum Threshold",
            "trigger_keywords": ["60%", "60 percent", "60.0%", "attendance condonation", "clause 5.2"],
            "legacy_doc_id": "HBK-ACAD-PR-05 (v1.8 / 2023)",
            "legacy_clause": "Clause 5.2",
            "legacy_rule_summary": "Legacy 2023 handbook suggested attendance condonation could be granted down to a 60% minimum.",
            "authoritative_doc_id": "POL-ACAD-AT-02 (v3.2 / 2025)",
            "authoritative_clause": "Table 2.1 & Clause 3.1",
            "authoritative_rule_summary": "Mandatory 2025 Senate Revision sets Standard Condonation at 70.0%–79.9% (HOD approval) and Exceptional Hospitalization Waiver at 65.0%–69.9% (Dean approval). Under 65.0% is a strict hard floor barred from exams.",
            "resolution_explanation": "POL-ACAD-AT-02 (2025) is authoritative. For 72% attendance, the student falls in the 70.0%–79.9% HOD Condonation Zone via Form ATT-02A.",
        },
        {
            "conflict_id": "CONF-CW-03",
            "topic": "Late Coursework Submission Penalty Rate",
            "trigger_keywords": ["2%", "2 percent", "10 days", "late coursework penalty", "clause 5.3"],
            "legacy_doc_id": "HBK-ACAD-PR-05 (v1.8 / 2023)",
            "legacy_clause": "Clause 5.3",
            "legacy_rule_summary": "Legacy rules applied a 2% daily deduction up to 10 days for unapproved late submissions.",
            "authoritative_doc_id": "POL-ACAD-CW-03 (v3.0 / 2025)",
            "authoritative_clause": "Clause 4.1",
            "authoritative_rule_summary": "Mandatory 2025 Senate Revision imposes 5% per working day penalty up to 5 working days, capped at 0% thereafter unless retrospective medical waiver is granted.",
            "resolution_explanation": "POL-ACAD-CW-03 (2025) applies. Approved retrospective Tier-3 application (Form EXT-03C) completely waives the 5% daily late penalty.",
        },
        {
            "conflict_id": "CONF-EXT-04",
            "topic": "Coursework Extension Authority & Maximum Duration",
            "trigger_keywords": ["21-day", "21 days", "lecturers may grant", "clause 5.4"],
            "legacy_doc_id": "HBK-ACAD-PR-05 (v1.8 / 2023)",
            "legacy_clause": "Clause 5.4",
            "legacy_rule_summary": "Legacy 2023 guidelines informally allowed module lecturers to grant up to 21 calendar days extension.",
            "authoritative_doc_id": "POL-ACAD-CW-03 (v3.0 / 2025)",
            "authoritative_clause": "Table 2.1",
            "authoritative_rule_summary": "Mandatory 2025 Three-Tier structure limits Module Leaders to 1-3 working days. Inpatient hospitalization requires Faculty EC Panel approval for 8-14 calendar days via Form EXT-03C.",
            "resolution_explanation": "Informal lecturer permissions exceeding 3 days are invalid under 2025 Senate Policy. Student must file Form EXT-03C for a 10-14 day extension.",
        },
        {
            "conflict_id": "CONF-MED-05",
            "topic": "Medical Documentation and Ayurvedic Practitioner Acceptance",
            "trigger_keywords": ["private ayurvedic", "ayurvedic notes", "accepted directly", "clause 5.5"],
            "legacy_doc_id": "HBK-ACAD-PR-05 (v1.8 / 2023)",
            "legacy_clause": "Clause 5.5",
            "legacy_rule_summary": "Legacy 2023 guidelines accepted notes from private Ayurvedic physicians without central university validation.",
            "authoritative_doc_id": "POL-MED-WF-04 (v3.5 / 2025)",
            "authoritative_clause": "Clause 3.2 & Table 4.1",
            "authoritative_rule_summary": "Mandatory 2025 Revision strictly requires SLMC registered practitioners or Government Ayurvedic Hospitals with mandatory Chief Medical Officer (UMC) clearance (Form MED-04-CLR). Private Ayurvedic notes are rejected.",
            "resolution_explanation": "POL-MED-WF-04 (2025) requires institutional verification at UMC Counter 204. Unverified private medical slips will be rejected.",
        },
    ]

    def verify_evidence(
        self,
        retrieved_results: List[RetrievalResult],
        identified_issues: List[str],
    ) -> Tuple[List[RetrievalResult], List[str], List[str]]:
        """
        Evaluates evidence sufficiency:
        Returns:
            - validated_chunks: Results passing the confidence threshold
            - missing_evidence_warnings: Warnings for uncovered areas or low confidence
            - re_retrieval_queries: Suggested refined queries if critical evidence is lacking
        """
        validated_chunks: List[RetrievalResult] = []
        missing_evidence_warnings: List[str] = []
        re_retrieval_queries: List[str] = []

        if not retrieved_results:
            missing_evidence_warnings.append(
                "CRITICAL: Zero policy chunks retrieved. Automated fallback to sample regulations initiated."
            )
            return validated_chunks, missing_evidence_warnings, ["emergency university policy regulations"]

        # Filter by score confidence
        for result in retrieved_results:
            if result.hybrid_score >= self.CONFIDENCE_THRESHOLD:
                validated_chunks.append(result)

        if not validated_chunks:
            # If all were below threshold, retain the best available but flag warning
            validated_chunks = sorted(retrieved_results, key=lambda x: x.hybrid_score, reverse=True)[:3]
            missing_evidence_warnings.append(
                f"Low retrieval confidence: Retrieved evidence scored below {self.CONFIDENCE_THRESHOLD:.2f}. "
                "Answers are bounded strictly by available policy citations."
            )

        # Check coverage of key academic topics
        all_text = " ".join([r.chunk.text.lower() for r in validated_chunks])

        if "exam" in " ".join(identified_issues).lower():
            if not any(k in all_text for k in ["ec-01", "deferred", "discharge", "examination"]):
                missing_evidence_warnings.append(
                    "Missing specific regulatory evidence for missed examination procedure."
                )
                re_retrieval_queries.append("POL-ACAD-EX-01 Form EC-01 deferred exam missed examination discharge")

        if "attendance" in " ".join(identified_issues).lower():
            if not any(k in all_text for k in ["att-02", "condonation", "waiver", "attendance"]):
                missing_evidence_warnings.append(
                    "Missing specific regulatory evidence for attendance condonation thresholds."
                )
                re_retrieval_queries.append("POL-ACAD-AT-02 Form ATT-02 attendance medical condonation threshold")

        if "coursework" in " ".join(identified_issues).lower():
            if not any(k in all_text for k in ["ext-03", "tier 3", "coursework", "penalty"]):
                missing_evidence_warnings.append(
                    "Missing specific regulatory evidence for coursework extension limits and penalty waivers."
                )
                re_retrieval_queries.append("POL-ACAD-CW-03 Form EXT-03 coursework extension late penalty waiver")

        return validated_chunks, missing_evidence_warnings, re_retrieval_queries

    def detect_conflicts(
        self,
        retrieved_results: List[RetrievalResult],
    ) -> List[PolicyConflictAlert]:
        """
        Step 4: Automated Policy Conflict Detector.
        Scans retrieved text across legacy (2023) and updated (2025) policy versions.
        Flags known and contextual contradictions.
        """
        detected_conflicts: List[PolicyConflictAlert] = []

        combined_text = " ".join([r.chunk.text.lower() for r in retrieved_results])
        doc_ids = {r.chunk.doc_id for r in retrieved_results}
        versions = {r.chunk.policy_version for r in retrieved_results}

        # Check if legacy content or mentions of superseded rules exist in the retrieved corpus
        has_legacy_corpus = (
            any("2023" in v for v in versions)
            or any("legacy" in v.lower() for v in versions)
            or "superseded by 2025" in combined_text
            or "legacy 2023" in combined_text
            or any(r.chunk.effective_year == 2023 for r in retrieved_results)
        )

        for rule in self.KNOWN_CONFLICT_RULES:
            # Check if any trigger keywords appear in the retrieved text
            matched_keywords = [
                kw for kw in rule["trigger_keywords"]
                if kw.lower() in combined_text
            ]
            if matched_keywords or has_legacy_corpus:
                # If legacy handbook is present or specific conflict triggers match
                if any(kw.lower() in combined_text for kw in rule["trigger_keywords"]):
                    alert = PolicyConflictAlert(
                        conflict_id=rule["conflict_id"],
                        topic=rule["topic"],
                        legacy_doc_id=rule["legacy_doc_id"],
                        legacy_clause=rule["legacy_clause"],
                        legacy_rule_summary=rule["legacy_rule_summary"],
                        authoritative_doc_id=rule["authoritative_doc_id"],
                        authoritative_clause=rule["authoritative_clause"],
                        authoritative_rule_summary=rule["authoritative_rule_summary"],
                        resolution_explanation=rule["resolution_explanation"],
                    )
                    detected_conflicts.append(alert)

        return detected_conflicts
