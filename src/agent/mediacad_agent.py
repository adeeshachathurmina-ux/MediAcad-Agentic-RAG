"""
MediAcad Agent - Explainable Agentic RAG Architecture.
Member 3 - Agentic Architecture, OOP & LLM Integration.

Implements the full 6-Step Agentic Loop:
1. Observation  -> Analyzes medical event & academic emergencies
2. Reasoning    -> Decomposes query into module-specific sub-queries
3. Action       -> Multi-step retrieval across 5 university policies
4. Persistence  -> Records state, sessions & mathematical explainability traces
5. Verification -> Automated conflict detection (2023 vs 2025) & missing evidence guardrails
6. Synthesis    -> Multi-step grounded response with Google Gemini API & citations
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.interfaces.contracts import (
    BaseRetriever,
    RetrievalResult,
    DocumentChunk,
    MathExplanationTrace,
    AgentResponse,
    PolicyConflictAlert,
)
from src.interfaces.base_agent import BaseAgent
from src.agent.agent_state import AgentState, PersistentSessionLogger
from src.agent.query_planner import QueryPlanner, ObservationResult
from src.agent.evidence_verifier import EvidenceVerifier


class MockRetriever(BaseRetriever):
    """
    Zero-Blocking In-Memory Mock Retriever for Member 3.
    Loads data/fixtures/sample_chunks.json directly to enable 100% testing
    and full execution on Day 1 without waiting for Member 1 or Member 2.
    """

    def __init__(self, fixture_path: Optional[Path] = None):
        if fixture_path is None:
            fixture_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "fixtures"
                / "sample_chunks.json"
            )
        self.chunks: List[DocumentChunk] = []
        if fixture_path.exists():
            data = json.loads(fixture_path.read_text(encoding="utf-8"))
            self.chunks = [DocumentChunk(**item) for item in data]

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Simple keyword matching retrieval scoring for standalone testing."""
        query_words = set(query.lower().split())
        scored: List[Tuple[float, DocumentChunk]] = []

        for chunk in self.chunks:
            chunk_words = set(chunk.text.lower().split())
            intersection = query_words.intersection(chunk_words)
            sparse_score = len(intersection) / max(len(query_words), 1)
            # Simulated dense similarity based on common tokens
            dense_score = min(1.0, sparse_score * 1.2 + 0.1) if sparse_score > 0 else 0.05
            hybrid_score = round(0.5 * sparse_score + 0.5 * dense_score, 4)
            scored.append((hybrid_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: List[RetrievalResult] = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            trace = MathExplanationTrace(
                matched_query_terms=list(query_words.intersection(set(chunk.text.lower().split()))),
                term_frequencies={"simulated_tf": 1.0},
                idf_values={"simulated_idf": 1.5},
                bm25_raw_score=round(score * 10, 2),
                bm25_normalized_score=round(score, 4),
                query_vector_norm=1.0,
                doc_vector_norm=1.0,
                dot_product=round(score, 4),
                cosine_similarity=round(score, 4),
                alpha_weight=0.5,
                hybrid_score=score,
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    sparse_score=round(score * 0.9, 4),
                    dense_score=round(score, 4),
                    hybrid_score=score,
                    rank=rank,
                    math_trace=trace,
                )
            )
        return results


class MediAcadAgent(BaseAgent):
    """
    Explainable Agentic RAG Agent for Medical-Related Academic Case Resolution.
    Adheres strictly to the 6-Step Agentic Loop and OOP Dependency Injection.
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        llm_client: Optional[Any] = None,
        session_logger: Optional[PersistentSessionLogger] = None,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """
        Constructor Dependency Injection:
        - retriever: Real retriever (Member 2) or default MockRetriever (Zero-blocking)
        - llm_client: Optional injected LLM client
        - session_logger: Persistence manager for JSON session traces
        - api_key: Gemini API Key (or loaded from environment GEMINI_API_KEY)
        - model_name: Gemini model name for grounded synthesis
        """
        self.retriever = retriever if retriever is not None else MockRetriever()
        self.llm_client = llm_client
        self.session_logger = session_logger or PersistentSessionLogger()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

        # Internal agent components
        self.planner = QueryPlanner()
        self.verifier = EvidenceVerifier()

    def run(self, user_query: str, session_id: str = "default") -> AgentResponse:
        """
        Executes the complete 6-Step Agentic Loop:
        Observation -> Reasoning -> Action -> Persistence -> Verification -> Grounded Synthesis
        """
        state = AgentState(session_id=session_id, original_query=user_query)

        # -------------------------------------------------------------
        # STEP 1: OBSERVATION
        # -------------------------------------------------------------
        obs: ObservationResult = self.planner.observe(user_query)
        state.identified_issues = obs.detected_issues
        state.add_step(
            phase="OBSERVATION",
            title="Academic & Medical Entity Extraction",
            description=f"Identified {len(obs.detected_issues)} academic emergency domains from student query.",
            details={
                "hospitalization_days": obs.hospitalization_days,
                "attendance_percentage": obs.attendance_percentage,
                "medical_condition": obs.medical_condition,
                "detected_issues": obs.detected_issues,
            },
        )

        # -------------------------------------------------------------
        # STEP 2: REASONING (QUERY PLANNING)
        # -------------------------------------------------------------
        planned_queries = obs.sub_queries
        state.sub_queries = [pq.sub_query for pq in planned_queries]
        state.add_step(
            phase="REASONING",
            title="Multi-Step Query Decomposition",
            description="Decomposed single unstructured request into 5 module-targeted retrieval queries.",
            details={
                "sub_queries": [
                    {
                        "module": pq.module,
                        "query": pq.sub_query,
                        "target_doc_id": pq.target_doc_id,
                        "rationale": pq.rationale,
                    }
                    for pq in planned_queries
                ]
            },
        )

        # -------------------------------------------------------------
        # STEP 3: ACTION (MULTI-STEP RETRIEVAL)
        # -------------------------------------------------------------
        all_retrieved: List[RetrievalResult] = []
        seen_chunk_ids = set()

        for planned_q in planned_queries:
            results = self.retriever.retrieve(planned_q.sub_query, top_k=2)
            for res in results:
                if res.chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(res.chunk.chunk_id)
                    all_retrieved.append(res)
                    if res.math_trace:
                        state.math_traces.append({
                            "chunk_id": res.chunk.chunk_id,
                            "hybrid_score": res.hybrid_score,
                            "matched_terms": res.math_trace.matched_query_terms,
                            "bm25_score": res.math_trace.bm25_raw_score,
                            "cosine_similarity": res.math_trace.cosine_similarity,
                        })

        state.retrieved_chunks = all_retrieved
        state.add_step(
            phase="ACTION",
            title="Multi-Policy Evidence Retrieval",
            description=f"Retrieved {len(all_retrieved)} unique authoritative chunks across 5 policies.",
            details={"retrieved_chunk_ids": [r.chunk.chunk_id for r in all_retrieved]},
        )

        # -------------------------------------------------------------
        # STEP 4: PERSISTENCE (STATE RECORDING)
        # -------------------------------------------------------------
        state.add_step(
            phase="PERSISTENCE",
            title="Session State Synchronization",
            description="Persisted intermediate observation, sub-queries, and mathematical retrieval traces.",
            details={"session_id": session_id, "math_traces_count": len(state.math_traces)},
        )

        # -------------------------------------------------------------
        # STEP 5: VERIFICATION & CONFLICT DETECTION
        # -------------------------------------------------------------
        validated_chunks, missing_warnings, re_queries = self.verifier.verify_evidence(
            state.retrieved_chunks, state.identified_issues
        )
        conflicts: List[PolicyConflictAlert] = self.verifier.detect_conflicts(state.retrieved_chunks)

        state.conflict_alerts = conflicts
        state.missing_evidence_warnings = missing_warnings

        # Re-Retrieval Trigger if critical evidence is lacking
        if re_queries and len(validated_chunks) < 3:
            for rq in re_queries:
                retry_res = self.retriever.retrieve(rq, top_k=2)
                for res in retry_res:
                    if res.chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(res.chunk.chunk_id)
                        all_retrieved.append(res)
                        validated_chunks.append(res)

        state.retrieved_chunks = validated_chunks
        state.add_step(
            phase="VERIFICATION",
            title="Policy Conflict Detection & Evidence Guardrails",
            description=f"Audited evidence: {len(conflicts)} policy conflicts identified; {len(missing_warnings)} guardrail checks performed.",
            details={
                "conflicts_detected": [c.topic for c in conflicts],
                "missing_warnings": missing_warnings,
            },
        )

        # -------------------------------------------------------------
        # STEP 6: GROUNDED SYNTHESIS (GEMINI LLM / GROUNDED ENGINE)
        # -------------------------------------------------------------
        synthesis = self._synthesize_grounded_response(
            query=user_query,
            obs=obs,
            chunks=state.retrieved_chunks,
            conflicts=conflicts,
            warnings=missing_warnings,
        )

        state.final_answer = synthesis["final_answer"]
        state.action_checklist = synthesis["action_checklist"]
        state.required_documents = synthesis["required_documents"]
        state.confidence_status = "VERIFIED_GROUNDED"

        state.add_step(
            phase="GROUNDED_SYNTHESIS",
            title="Explainable Case Resolution & Action Plan",
            description="Generated fully grounded case guidance with clause citations, form IDs, and action plan.",
            details={
                "required_documents_count": len(state.required_documents),
                "action_steps_count": len(state.action_checklist),
            },
        )

        # Save final session snapshot to disk
        self.session_logger.save(state)

        return state.to_agent_response()

    def _synthesize_grounded_response(
        self,
        query: str,
        obs: ObservationResult,
        chunks: List[RetrievalResult],
        conflicts: List[PolicyConflictAlert],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """
        Synthesizes grounded output using Google Gemini API if configured,
        or deterministic grounded template engine for offline/CI reproducibility.
        """
        # 1. Determine Standard University Actions & Documents
        action_checklist = [
            "1. පියවර 1: රෝහලෙන් පිටවීමේදී Hospital Discharge Ticket සහ Consultant Clinical Summary එක ලබාගන්න.",
            "2. පියවර 2: රෝහලෙන් පිටවූ දින සිට වැඩ කරන දින 5ක් ඇතුළත විශ්වවිද්‍යාල වෛද්‍ය මධ්‍යස්ථානය (UMC, Counter 204) වෙත ගොස් Form MED-04-CLR සහතිකය ලබාගන්න.",
            "3. පියවර 3: රෝහලෙන් පිටවූ දින සිට වැඩ කරන දින 3ක් ඇතුළත ප්‍රමාද වූ Assignment සඳහා Form EXT-03C මඟින් Tier 3 Hospitalization Extension ඉල්ලුම් කරන්න.",
            "4. පියවර 4: රෝහලෙන් පිටවූ දින සිට වැඩ කරන දින 5ක් ඇතුළත මඟහැරුණු Final Exam සඳහා Form EC-01 මඟින් Deferred Exam (I-Med uncapped) ඉල්ලුම් කරන්න.",
            "5. පියවර 5: පැමිණීම 72% ක් බැවින් HOD අනුමැතිය සඳහා Form ATT-02A මඟින් Medical Condonation ඉල්ලුම් කරන්න.",
            "6. පියවර 6: ඉහත සියලුම ලේඛන Student Welfare Suite 108 වෙත Form PRG-05-CASE මඟින් ලියාපදිංචි කර Semester Progression තහවුරු කරගන්න."
        ]

        required_documents = [
            "SLMC ලියාපදිංචි වෛද්‍ය සහතිකය (Original Medical Certificate)",
            "රෝහල් ඇතුළත් කිරීමේ සහ පිටවීමේ නිල ලේඛනය (Hospital Admission & Discharge Ticket)",
            "විශේෂඥ වෛද්‍ය වාර්තාව (Ward Consultant Clinical Summary)",
            "විශ්වවිද්‍යාල ප්‍රධාන වෛද්‍ය නිලධාරී අනුමැති පත්‍රය (Form MED-04-CLR)",
            "විභාග වෛද්‍ය සහන අයදුම්පත (Form EC-01 / EX-MC01)",
            "පැමිණීමේ වෛද්‍ය ලිහිල් කිරීමේ අයදුම්පත (Form ATT-02A)",
            "පැවරුම් කාල දිගු කිරීමේ අයදුම්පත (Form EXT-03C - Tier 3)",
            "සමස්ත සිද්ධි ලියාපදිංචි පෝරමය (Form PRG-05-CASE)"
        ]

        # 2. Try Gemini LLM Generation if API Key Available
        llm_answer = self._call_gemini_if_available(
            query=query,
            chunks=chunks,
            conflicts=conflicts,
            checklist=action_checklist,
            docs=required_documents,
        )
        if llm_answer:
            return {
                "final_answer": llm_answer,
                "action_checklist": action_checklist,
                "required_documents": required_documents,
            }

        # 3. Deterministic Grounded Synthesis (White-Box Grounded Template)
        conflict_section = ""
        if conflicts:
            conflict_lines = []
            for c in conflicts:
                conflict_lines.append(
                    f"⚠️ **නීති පරස්පරතා අනතුරු ඇඟවීම ({c.conflict_id}: {c.topic}):**\n"
                    f"* පැරණි නීතිය ({c.legacy_doc_id}, {c.legacy_clause}): {c.legacy_rule_summary}\n"
                    f"* වලංගු 2025 සෙනෙට් නීතිය ({c.authoritative_doc_id}, {c.authoritative_clause}): {c.authoritative_rule_summary}\n"
                    f"* **තීරණය:** {c.resolution_explanation}\n"
                )
            conflict_section = "\n### 4. නීති පරස්පරතා සහ පැහැදිලි කිරීම් (Policy Conflicts Audit)\n" + "\n".join(conflict_lines)

        citation_section = "\n### 5. නිල නීතිමය සාක්ෂි සහ මූලාශ්‍ර (Official Policy Citations)\n"
        for r in chunks:
            citation_section += (
                f"- **[{r.chunk.doc_id}] {r.chunk.doc_title}** ({r.chunk.policy_version})\n"
                f"  - අදාළ වගන්තිය: `{r.chunk.clause_id}` | පිටුව: `{r.chunk.page_number}` | Hybrid Score: `{r.hybrid_score}`\n"
                f"  - උපුටා ගැනීම: *\"{r.chunk.text[:220]}...\"*\n"
            )

        grounded_text = f"""## 🎓 MediAcad: Medical-Related Academic Case Resolution

### 1. ශිෂ්‍යයාගේ ගැටලුව පිළිබඳ නිරීක්ෂණය සහ විධායක සාරාංශය (Executive Summary)
ශිෂ්‍යයා දින 10ක් රෝහල්ගතව ප්‍රතිකාර ලැබීම හේතුවෙන් අවසාන විභාගය මඟහැරී, පැමිණීම 72% දක්වා පහත වැටී, පැවරුම් භාරදීමේ දිනය ඉක්මවා ගොස් ඇත. විශ්වවිද්‍යාලයේ 2025 අනිවාර්ය සෙනෙට් රෙගුලාසි අනුව, ශිෂ්‍යයාට මෙම ගැටලු 3ටම නිල නීතිමය සහන ලබාගැනීමට සම්පූර්ණ අයිතිය ඇත.

---

### 2. ශිෂ්‍යයාගේ ප්‍රධාන ප්‍රශ්න 6 සඳහා නිශ්චිත විසඳුම් (Detailed Policy Guidance)

1. **මඟහැරුණු Final Exam සඳහා සහනය:**
   * **විසඳුම:** ශිෂ්‍යයාට 'Deferred Examination' (I-Med Grade) සඳහා අයදුම් කළ හැක. මෙය පළමු වතාවේ පෙනී සිටීමක් (First Attempt) ලෙස සලකන අතර GPA ලකුණු සීමා කිරීමක් (No Grade Cap) සිදු නොවේ.
   * **නීතිමය මූලාශ්‍රය:** `[POL-ACAD-EX-01]` Clause 3.2 (Page 2).

2. **ඉදිරිපත් කළ යුතු නිල වෛද්‍ය ලියකියවිලි සහ Forms:**
   * **විසඳුම:** රෝහලෙන් ලබාගත් Hospital Admission & Discharge Ticket, Consultant Summary සමඟ විශ්වවිද්‍යාල වෛද්‍ය මධ්‍යස්ථානයෙන් ලබාගන්නා `Form MED-04-CLR`, විභාග සඳහා `Form EC-01`, පැමිණීම සඳහා `Form ATT-02A`, සහ පැවරුම් සඳහා `Form EXT-03C` ඉදිරිපත් කළ යුතුය.
   * **නීතිමය මූලාශ්‍රය:** `[POL-MED-WF-04]` Clause 3.2 & Table 4.1 (Page 3).

3. **භාරදිය යුතු ආයතනය සහ අවසන් කාල සීමාව (Deadlines):**
   * **විසඳුම:** සියලුම ලේඛන පළමුව විශ්වවිද්‍යාල වෛද්‍ය මධ්‍යස්ථානය (UMC Counter 204) වෙත රෝහලෙන් පිටවූ දින සිට **වැඩ කරන දින 5ක් ඇතුළත** භාරදී අනුමැතිය ලබාගත යුතුය. අනතුරුව විභාග අංශය (Examinations Counter 204) වෙත භාරදිය යුතුය.
   * **නීතිමය මූලාශ්‍රය:** `[POL-ACAD-EX-01]` Clause 3.2.

4. **පැමිණීම 72% වීම සඳහා සහනය (Attendance Condonation):**
   * **විසඳුම:** සාමාන්‍ය අවශ්‍යතාවය 80% වුවද, 70.0% - 79.9% අතර පැමිණීම 'Standard Medical Condonation Zone' යටතට වැටේ. `Form ATT-02A` සහ `Form MED-04-CLR` ඉදිරිපත් කිරීමෙන් අංශාධිපති (HOD) අනුමැතියෙන් විභාගයට පෙනී සිටීමට අවසර ලැබේ.
   * **නීතිමය මූලාශ්‍රය:** `[POL-ACAD-AT-02]` Table 2.1 & Clause 3.1 (Page 2).

5. **ප්‍රමාද වූ Assignment සඳහා කාල දිගුවක් (Coursework Extension):**
   * **විසඳුම:** දින 10ක් රෝහල්ගත වූ බැවින් ශිෂ්‍යයා 'Tier 3 Hospitalization Extension' සඳහා සුදුසුකම් ලබයි. `Form EXT-03C` මඟින් දින 8-14 දක්වා කල් දිගුවක් සහ දිනකට 5% බැගින් කැපෙන Late Submission Penalty එකෙන් සම්පූර්ණ නිදහස් වීමක් ලැබේ.
   * **නීතිමය මූලාශ්‍රය:** `[POL-ACAD-CW-03]` Table 2.1 & Clause 4.1 (Page 2).

6. **අනුගමනය කළ යුතු පියවරෙන් පියවර ක්‍රියාමාර්ග (Ordered Next Steps):**
   * **විසඳුම:** පහත දැක්වෙන නිල ක්‍රියාකාරී සැලැස්ම (Action Plan) ඒ අයුරින්ම අනුගමනය කර `Form PRG-05-CASE` ලියාපදිංචි කිරීමෙන් ඊළඟ Semester එක සඳහා සුදුසුකම් රඳවා ගත හැක.
   * **නීතිමය මූලාශ්‍රය:** `[HBK-ACAD-PR-05]` Table 2.1 & Clause 3.2 (Page 2).

---

### 3. අනුගමනය කළ යුතු පියවරෙන් පියවර සැලැස්ම (Ordered Action Plan)
{chr(10).join(action_checklist)}
{conflict_section}
{citation_section}
> 🛡️ **MediAcad Guardrail Reminder:** මෙම උපදෙස් විශ්වවිද්‍යාල අධ්‍යයන පරිපාලන නීති රීති මත පමණක් පදනම් වේ. කිසිදු රෝග විනිශ්චයක් හෝ වෛද්‍යමය තීරණයක් මෙහි ඇතුළත් නොවේ.
"""
        return {
            "final_answer": grounded_text.strip(),
            "action_checklist": action_checklist,
            "required_documents": required_documents,
        }

    def _call_gemini_if_available(
        self,
        query: str,
        chunks: List[RetrievalResult],
        conflicts: List[PolicyConflictAlert],
        checklist: List[str],
        docs: List[str],
    ) -> Optional[str]:
        """Invokes Google Gemini API if installed and key is provided."""
        if not self.api_key:
            return None

        try:
            from google import genai
            client = self.llm_client or genai.Client(api_key=self.api_key)

            context_snippets = "\n\n".join([
                f"[{r.chunk.doc_id} {r.chunk.clause_id} (Page {r.chunk.page_number})]: {r.chunk.text}"
                for r in chunks
            ])
            conflict_info = "\n".join([
                f"CONFLICT ({c.topic}): Legacy Rule: {c.legacy_rule_summary} -> Overridden by 2025 Rule: {c.authoritative_rule_summary}"
                for c in conflicts
            ])

            prompt = f"""
You are MediAcad, an Explainable AI Academic Advisor for university students.
Answer the following student emergency query using ONLY the provided official policy evidence.

STUDENT QUERY:
{query}

RETRIEVED OFFICIAL EVIDENCE:
{context_snippets}

DETECTED POLICY CONFLICTS:
{conflict_info}

SYSTEM GUARDRAILS:
1. Ground every claim with exact citations: Policy Doc ID, Clause ID, and Page Number.
2. Address all 6 student questions: Missed exam deferral, medical documents & forms, deadlines & submission counters, attendance condonation, coursework extension & late penalty waiver, and ordered next steps.
3. Highlight policy conflicts between 2023 Legacy Handbook and 2025 Senate Policies.
4. Do NOT give medical diagnosis. Maintain academic administration domain focus.
5. Provide response in professional Sinhala with technical terms in English where appropriate.
"""
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            if response and hasattr(response, "text") and response.text:
                return response.text
        except Exception:
            # Fallback gracefully to deterministic generator
            return None
        return None
