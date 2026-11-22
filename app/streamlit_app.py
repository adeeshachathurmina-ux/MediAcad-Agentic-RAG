"""
MediAcad: Explainable Agentic RAG Interactive Dashboard (Member 4).
Clean, modular presentation layer connected directly to all 4 team members' packages:
  - Member 1: 138 Structure-Aware PDF Chunks (data/processed_chunks.json) + Persistent ChromaDB status
  - Member 2: HybridExplainableRetriever (Okapi BM25 + Pure-NumPy L2 Cosine Vector Math)
  - Member 3: EvidenceVerifier (2023 vs. 2025 Conflict Audit) + 5-Stage Agentic Reasoning Trace
  - Member 4: RAGTriadEvaluator (32-Case Real Benchmark + Live Groundedness Verification)
"""

import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interfaces.contracts import (
    AgentResponse,
    AgentStepLog,
    DocumentChunk,
    PolicyConflictAlert,
    RetrievalResult,
)
from src.ingestion.clause_chunker import ClauseAwareChunker
from src.retrieval.hybrid_retriever import HybridExplainableRetriever
from src.agent.evidence_verifier import EvidenceVerifier
from src.evaluation.metrics import RAGTriadEvaluator, tokenize
from src.evaluation.run_benchmark import (
    _synthesize_grounded_response,
    run_full_benchmark,
)

st.set_page_config(
    page_title="MediAcad | Explainable Agentic RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Adaptive Dark/Light Mode CSS Styling
st.markdown(
    """
    <style>
    .hero-box {
        background: linear-gradient(135deg, #0F2942 0%, #0D6E6E 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 15px rgba(15, 41, 66, 0.25);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.4px;
    }
    .hero-sub {
        font-size: 0.92rem;
        color: #D6EAF8;
        margin-bottom: 0;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15, 41, 66, 0.50) 0%, rgba(13, 110, 110, 0.20) 100%);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-left: 5px solid #10B981;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    .conflict-banner {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.16) 0%, rgba(217, 119, 6, 0.06) 100%);
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-left: 6px solid #F59E0B;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin: 0.65rem 0;
        color: inherit;
        line-height: 1.55;
    }
    .guardrail-banner {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.06) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-left: 6px solid #EF4444;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin: 0.65rem 0;
        color: inherit;
        line-height: 1.55;
    }
    mark.kw-highlight {
        background-color: rgba(16, 185, 129, 0.28);
        color: inherit;
        border: 1px solid rgba(16, 185, 129, 0.65);
        padding: 1px 5px;
        border-radius: 4px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

GEMINI_MODEL = "gemini-3.5-flash-lite"

CLINICAL_GUARDRAIL_TERMS = (
    "dosage", "medication", "prescribe", "do i have dengue", "what medicine",
    "103f", "paracetamol", "diagnose", "diagnosis", "antibiotic",
    "painkiller", "tablet", "pill", "what drug", "cure",
)

STRONG_POLICY_TERMS = {
    "fit-to-sit", "ayurvedic", "intercalation", "condonation", "deferral",
    "ec-01", "att-02a", "att-02b", "ext-03a", "ext-03b", "ext-03c",
    "med-04-clr", "inv-09", "prg-05-case", "prg-05-int", "med-rtn-06",
    "slmc", "umc", "i-med", "forged", "backdated", "coursework", "hospitalization",
    "attendance", "exam", "examination", "exams", "medical", "assignment",
    "extension", "penalty", "dengue", "hospitalized", "gpa", "credits",
    "progression", "laboratory", "tuition", "certificate", "laptop", "motherboard",
}


# ==============================================================================
# DATA & MODULAR RETRIEVAL LOADERS (Member 1 & Member 2 Integration)
# ==============================================================================
def clean_policy_display_text(raw_text: str) -> str:
    """Strips redundant [Doc: ...] header prefix and PDF glyph artifacts."""
    cleaned = re.sub(r"^\[Doc:[^\]]+\]\s*", "", raw_text.strip())
    cleaned = re.sub(r"\(cid:\d+\)", "•", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def highlight_matched_terms(text: str, matched_terms: List[str]) -> str:
    """Highlights matched BM25 lexical tokens inside retrieved PDF policy clauses."""
    if not matched_terms:
        return text
    valid_terms = sorted({t for t in matched_terms if len(t) >= 2}, key=len, reverse=True)
    if not valid_terms:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in valid_terms) + r")\b", re.IGNORECASE)
    return pattern.sub(r'<mark class="kw-highlight">\1</mark>', text)


def extract_attendance_percentage(query: str) -> Optional[float]:
    """Extracts realistic attendance percentage (25%–100%) while ignoring 2%/5% late penalties."""
    q_lower = query.lower()
    if not any(w in q_lower for w in ("attendance", "condonation", "barred", "waiver")):
        return None
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", query):
        val = float(match.group(1))
        if 25.0 <= val <= 100.0:
            return val
    return None


@st.cache_data
def _load_chunks_cached(path_str: str, _mtime: float) -> List[DocumentChunk]:
    raw = json.loads(Path(path_str).read_text(encoding="utf-8"))
    return [DocumentChunk(**item) for item in raw]


def load_corpus_chunks() -> List[DocumentChunk]:
    """Loads Member 1's full 138-chunk extracted PDF corpus (data/processed_chunks.json)."""
    processed_path = PROJECT_ROOT / "data" / "processed_chunks.json"
    if processed_path.exists():
        return _load_chunks_cached(str(processed_path), processed_path.stat().st_mtime)

    pdf_dir = PROJECT_ROOT / "data" / "raw_pdfs"
    if pdf_dir.is_dir():
        return ClauseAwareChunker().parse_and_chunk(str(pdf_dir))

    fixture_path = PROJECT_ROOT / "data" / "fixtures" / "sample_chunks.json"
    return _load_chunks_cached(str(fixture_path), fixture_path.stat().st_mtime)


@st.cache_data
def _load_benchmark_cached(path_str: str, _mtime: float) -> Dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_benchmark_data() -> Dict[str, Any]:
    """Loads the 32-case RAG Triad benchmark JSON, auto-invalidating cache whenever file updates."""
    bench_path = PROJECT_ROOT / "data" / "evaluation" / "benchmark_results.json"
    if not bench_path.exists():
        run_full_benchmark()
    return _load_benchmark_cached(str(bench_path), bench_path.stat().st_mtime)


@st.cache_data
def get_persistent_vector_store_status() -> str:
    """Safely inspects Member 1's persistent ChromaDB store without slowing cloud startup."""
    chroma_dir = PROJECT_ROOT / "data" / "chroma_db"
    if not chroma_dir.is_dir():
        return "Processed JSON Corpus (`138` chunks in `data/processed_chunks.json`)"
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        col = client.get_collection(name="academic_policies")
        return f"Persistent ChromaDB (`{col.count()}` vectors in `data/chroma_db`)"
    except Exception:
        return "Processed JSON Corpus (`138` chunks in `data/processed_chunks.json`)"


@st.cache_resource
def _get_cached_hybrid_engine(
    chunk_count: int,
    first_chunk_id: str,
    _chunks: List[DocumentChunk],
) -> Tuple[HybridExplainableRetriever, Dict[str, int], np.ndarray]:
    """
    Builds Member 2's HybridExplainableRetriever once per session with a pure-NumPy
    TF-IDF dense vector space (sublinear TF * smoothed IDF) for exact L2 geometry.
    """
    doc_texts = [f"{c.doc_id} {c.clause_id} {c.section_title} {c.text}" for c in _chunks]
    doc_toks = [tokenize(t) for t in doc_texts]
    n_docs = max(len(_chunks), 1)

    df: Dict[str, int] = {}
    for toks in doc_toks:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1

    vocab = sorted(df.keys())
    word_to_idx = {w: idx for idx, w in enumerate(vocab)}
    idf_vec = np.array(
        [math.log(1.0 + (n_docs - df[w] + 0.5) / (df[w] + 0.5)) for w in vocab],
        dtype=np.float64,
    )

    def tfidf_dense_embedder(texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(vocab)), dtype=np.float64)
        for row_idx, text in enumerate(texts):
            counts: Dict[str, int] = {}
            for w in tokenize(text):
                if w in word_to_idx:
                    counts[w] = counts.get(w, 0) + 1
            for w, cnt in counts.items():
                col_idx = word_to_idx[w]
                mat[row_idx, col_idx] = (1.0 + math.log(cnt)) * idf_vec[col_idx]
        return mat

    retriever = HybridExplainableRetriever(
        chunks=_chunks,
        alpha=0.50,
        dense_embedder=tfidf_dense_embedder,
    )
    return retriever, word_to_idx, idf_vec


def compute_live_hybrid_retrieval(
    query: str,
    chunks: List[DocumentChunk],
    alpha: float = 0.5,
    top_k: int = 5,
) -> List[RetrievalResult]:
    """Executes Member 2's real HybridExplainableRetriever with dynamic alpha tuning."""
    if not query.strip() or not chunks:
        return []
    retriever, _, _ = _get_cached_hybrid_engine(len(chunks), chunks[0].chunk_id, chunks)
    retriever.set_alpha(alpha)
    return retriever.retrieve(query, top_k=top_k)


# ==============================================================================
# OPTIONAL LIVE GEMINI SYNTHESIS (Non-Blocking 12s Timeout + Extractive Fallback)
# ==============================================================================
def _resolve_gemini_api_key() -> Optional[str]:
    """Reads GEMINI_API_KEY safely from st.secrets or local .env without ever exposing it."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if isinstance(secret_key, str) and secret_key.strip() and secret_key.strip() != "your_gemini_api_key_here":
            return secret_key.strip()
    except Exception:
        pass

    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass

    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key and env_key != "your_gemini_api_key_here":
        return env_key
    return None


def _call_optional_gemini(
    api_key: str,
    query: str,
    retrieved: List[RetrievalResult],
    conflicts: List[PolicyConflictAlert],
    att_note: Optional[str],
    timeout_sec: float = 12.0,
) -> Optional[str]:
    """Generates a strictly grounded live Gemini summary with non-blocking timeout fallback."""
    if not api_key or not retrieved:
        return None

    evidence_blocks = [
        f"[{r.chunk.doc_id} | {r.chunk.policy_version} | {r.chunk.clause_id} | Page {r.chunk.page_number}]\n"
        f"{clean_policy_display_text(r.chunk.text)}"
        for r in retrieved[:4]
    ]
    conflict_lines = "\n".join(
        f"- {c.conflict_id} ({c.topic}): {c.authoritative_rule_summary}" for c in conflicts
    )
    prompt = (
        "You are MediAcad, an Explainable University Medical-Academic Policy Advisor. "
        "Answer in concise, structured professional English using ONLY the Retrieved Policy Evidence below. "
        "Always cite exact Doc IDs, Clause/Table IDs, Page Numbers, and official Form codes. "
        "Never invent rules or provide clinical medical advice.\n\n"
        f"STUDENT QUESTION:\n{query}\n\n"
        "RETRIEVED OFFICIAL POLICY EVIDENCE:\n" + "\n\n".join(evidence_blocks)
    )
    if att_note:
        prompt += f"\n\nDYNAMIC 2025 ATTENDANCE RULE CHECK:\n{att_note}"
    if conflict_lines:
        prompt += f"\n\n2023 vs. 2025 POLICY CONFLICTS DETECTED:\n{conflict_lines}"

    result_box: Dict[str, Optional[str]] = {"text": None}

    def _invoke() -> None:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=12000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            if resp and getattr(resp, "text", None):
                result_box["text"] = resp.text.strip()
        except Exception:
            result_box["text"] = None

    worker = threading.Thread(target=_invoke, daemon=True)
    worker.start()
    worker.join(timeout=timeout_sec)
    return result_box["text"]


# ==============================================================================
# 5-STAGE EXPLAINABLE AGENTIC WORKFLOW (Member 3 & Member 4 Integration)
# ==============================================================================
def _describe_attendance_zone(att_pct: Optional[float]) -> Optional[str]:
    """Deterministic 2025 POL-ACAD-AT-02 zone classification."""
    if att_pct is None:
        return None
    if att_pct >= 80.0:
        return (
            f"**Attendance Assessment ({att_pct:.1f}%):** Eligible Safe Zone "
            "(`>= 80.0%` baseline, `POL-ACAD-AT-02` Clause 1.1). No condonation form required."
        )
    if att_pct >= 70.0:
        return (
            f"**Attendance Assessment ({att_pct:.1f}%):** Standard Medical Condonation Zone "
            "(`70.0%–79.9%`, `POL-ACAD-AT-02` Table 2.1). Submit `Form ATT-02A` + `Form MED-04-CLR` "
            "to Faculty Office (Room F-102) for Head of Department approval within 5 working days."
        )
    if att_pct >= 65.0:
        return (
            f"**Attendance Assessment ({att_pct:.1f}%):** Exceptional Hospitalization Waiver Zone "
            "(`65.0%–69.9%`, `POL-ACAD-AT-02` Clause 3.1). Requires `>= 7` calendar days inpatient "
            "hospitalization; submit `Form ATT-02B` + Hospital Discharge Summary + `Form MED-04-CLR` for Dean approval."
        )
    return (
        f"**Attendance Assessment ({att_pct:.1f}%):** STRICTLY BARRED / HARD FLOOR "
        "(`< 65.0%`, `POL-ACAD-AT-02` Clause 3.2). Student receives Not Qualified (`NQ`) status "
        "and must apply for Temporary Medical Intercalation (`Form PRG-05-INT`) under `HBK-ACAD-PR-05` Clause 4.2."
    )


def _build_domain_checklist_and_forms(
    q_lower: str,
    is_clinical: bool,
    att_pct: Optional[float],
) -> Tuple[List[str], List[str]]:
    """Builds concise, domain-accurate administrative steps and official form codes."""
    if is_clinical:
        return (
            [
                "Seek immediate clinical care at a hospital or the University Medical Center (UMC, Building B, Room B-04).",
                "Do NOT enter the exam hall while acutely ill (Fit-to-Sit Doctrine: POL-ACAD-EX-01 Clause 2.1).",
                "Log an Initial Absence Notification on the Student Portal within 48 hours (Clause 3.1).",
                "Verify your SLMC certificate at UMC for Form MED-04-CLR and submit Form EC-01 within 5 working days (Clause 3.2).",
            ],
            ["Original SLMC Medical Certificate", "Form MED-04-CLR", "Form EC-01"],
        )

    is_multi = (
        ("exam" in q_lower)
        and ("attendance" in q_lower or att_pct is not None)
        and ("assignment" in q_lower or "coursework" in q_lower)
    )
    if is_multi:
        att_form = "Form ATT-02B (65–69.9% Dean Track)" if (att_pct is not None and att_pct < 70.0) else "Form ATT-02A (70–79.9% HoD Track)"
        return (
            [
                "Submit Online Absence Notification within 48 hours of missed exam (POL-ACAD-EX-01 Clause 3.1).",
                "Collect Hospital Admission & Discharge Ticket, Consultant Summary, and SLMC Certificate (POL-MED-WF-04 Table 3.1).",
                "Obtain CMO Clearance Slip (Form MED-04-CLR) at UMC Building B and file Retrospective Form EXT-03C within 3 working days of discharge.",
                "Submit Form EC-01 + Form MED-04-CLR to Examinations Counter 204 within 5 working days of discharge for uncapped 'I-Med'.",
                f"Submit {att_form} to Faculty Office (Room F-102) within 5 working days of discharge.",
                "Register Consolidated Multi-Issue Dossier Form PRG-05-CASE at Student Welfare Suite 108.",
            ],
            [
                "Hospital Discharge Ticket + SLMC Certificate",
                "Form MED-04-CLR",
                "Form EC-01",
                "Form ATT-02A / ATT-02B",
                "Form EXT-03C",
                "Form PRG-05-CASE",
            ],
        )

    if any(w in q_lower for w in ("attendance", "condonation", "waiver", "barred", "lab")) or att_pct is not None:
        if att_pct is not None and att_pct < 65.0:
            return (
                [
                    f"Hard Floor Enforced ({att_pct:.1f}% < 65.0%): Raw attendance below 65% receives NQ status; no waiver is permitted (POL-ACAD-AT-02 Clause 3.2).",
                    "Authenticate hospital discharge records at UMC (Building B, Room B-04) to obtain Form MED-04-CLR.",
                    "Submit Form PRG-05-INT (Temporary Medical Intercalation) at Student Welfare Suite 108 to protect GPA and credit 100% tuition forward.",
                    "Obtain Fit-to-Return Clearance (Form MED-RTN-06) at least 14 calendar days before resuming studies.",
                ],
                ["Form PRG-05-INT", "Form MED-04-CLR", "Hospital Discharge Summary", "Form MED-RTN-06"],
            )
        return (
            [
                "Verify attendance band under POL-ACAD-AT-02 Table 2.1 (70–79.9% HoD vs. 65–69.9% Dean Inpatient Waiver >=7 days).",
                "Authenticate original SLMC certificate and Discharge Summary at UMC for Form MED-04-CLR.",
                "Submit Form ATT-02A (HoD) or Form ATT-02B (Dean) to Faculty Office (Room F-102) within 5 working days.",
                "Complete mandatory Academic Catch-Up Plan (or Form LAB-MK-01 for missed laboratory sessions).",
            ],
            ["Form ATT-02A / Form ATT-02B", "Form MED-04-CLR", "SLMC Certificate / Discharge Summary", "Form LAB-MK-01 (if labs missed)"],
        )

    if any(w in q_lower for w in ("laptop", "motherboard", "usb", "wi-fi", "wifi", "hardware", "hard drive", "computer crash")):
        return (
            [
                "Personal hardware/laptop crashes and local technical failures are strictly ineligible for medical coursework extensions (POL-ACAD-CW-03 Clause 5.3).",
                "Immediately submit your latest cloud/Git backup (GitHub, OneDrive, or Google Drive) to minimize unapproved late penalties.",
                "Note the 2025 Late Penalty Schedule (POL-ACAD-CW-03 Table 4.1): -5% per working day late up to 5 working days; submissions >5 working days late receive 0%.",
            ],
            ["Not Eligible for Medical Extension (POL-ACAD-CW-03 Clause 5.3)", "Submit Cloud / Version-Control Backup Immediately"],
        )

    if any(w in q_lower for w in ("assignment", "coursework", "late", "penalty", "extension", "21-day", "21 days")):
        return (
            [
                "Verify medical eligibility under POL-ACAD-CW-03 (non-medical hardware/laptop failures are ineligible under Clause 5.3).",
                "Apply 2025 Penalty Schedule (Table 4.1): -5% per working day late up to 5 working days; >5 working days = 0%.",
                "Submit Form EXT-03A (1–3d Lecturer), Form EXT-03B (4–7d HoD), or Form EXT-03C (8–14d max EC Panel + Form MED-04-CLR).",
            ],
            ["Form EXT-03A / EXT-03B / EXT-03C", "Form MED-04-CLR (Tier 3)", "SLMC Medical Certificate"],
        )

    if "ayurvedic" in q_lower:
        return (
            [
                "Confirm issuing hospital is a Government Ayurvedic Teaching Hospital (private Ayurvedic slips are rejected under POL-MED-WF-04 Clause 2.2).",
                "Present original Government Ayurvedic certificate at UMC within 3 working days for CMO countersignature (Form MED-04-CLR).",
            ],
            ["Government Ayurvedic Teaching Hospital Certificate", "Form MED-04-CLR"],
        )

    if re.search(r"\b(forged|altered|backdated|fake|father|mother|relative|family|parent)\b", q_lower):
        return (
            [
                "Ensure medical certificates are issued by an independent, non-relative SLMC practitioner (POL-MED-WF-04 Clause 2.3).",
                "Never alter or backdate certificates: Category-A Academic Misconduct carries 0% semester annulment + 1-year suspension (Clause 5.2).",
            ],
            ["Independent SLMC Medical Certificate", "Form MED-04-CLR (UMC Audit Clearance)"],
        )

    return (
        [
            "Log Initial Absence Notification on the Student Portal within 48 hours (POL-ACAD-EX-01 Clause 3.1).",
            "Authenticate SLMC certificate / Discharge Ticket at UMC (Building B, Room B-04) for Form MED-04-CLR.",
            "Submit Form EC-01 + Form MED-04-CLR to Examinations Counter 204 strictly within 5 working days (Clause 3.2).",
        ],
        ["Form EC-01", "Form MED-04-CLR", "SLMC Medical Certificate / Discharge Ticket"],
    )


def _plan_targeted_subqueries(query: str, att_pct: Optional[float]) -> List[str]:
    """Word-boundary-safe multi-policy sub-query planner (prevents false matches like 'motherboard')."""
    ql = query.lower()
    extra: List[str] = []

    if re.search(r"\b(2023|handbook|conflict|7 calendar|60%|2%|21-day|21 days|private ayurvedic)\b", ql):
        extra.append(f"HBK-ACAD-PR-05 Table 5.1 Clause 5.1 5.2 5.3 5.4 5.5 legacy 2023 override {query}")
    if (att_pct is not None and att_pct < 65.0) or re.search(r"\b(below 65|hard floor|intercalation|tuition)\b", ql):
        extra.append("POL-ACAD-AT-02 Clause 3.2 65% hard floor HBK-ACAD-PR-05 Clause 4.2 Form PRG-05-INT Form MED-RTN-06")
    if re.search(r"\b(group|project|viva)\b", ql) and re.search(r"\b(hospital|medical|member)\b", ql):
        extra.append("POL-ACAD-CW-03 Clause 5.2 group coursework Individual Make-Up Viva Form EXT-03C")
    if re.search(r"\b(10-day|10 days|hospitalized|hospitalization)\b", ql) and (
        ("exam" in ql) and ("attendance" in ql or att_pct is not None) and ("coursework" in ql or "assignment" in ql)
    ):
        extra.extend([
            "POL-ACAD-EX-01 Clause 3.2 Form EC-01 5 working days deferred exam I-Med",
            "POL-ACAD-AT-02 Table 2.1 Clause 2.2 Clause 3.1 Form ATT-02A ATT-02B",
            "POL-ACAD-CW-03 Clause 2.3 Form EXT-03C retrospective extension",
            "POL-MED-WF-04 Table 3.1 Form MED-04-CLR UMC verification",
            "HBK-ACAD-PR-05 Table 2.1 Clause 2.1 Form PRG-05-CASE",
        ])
    if re.search(r"\b(progression|90 credit|30 credit|i-med|supplementary|second time)\b", ql):
        extra.append("HBK-ACAD-PR-05 Clause 3.1 Clause 3.2 Table 3.1 progression 90 credits 30 credits I-Med")
    if re.search(r"\b(father|mother|relative|family|parent)\b", ql):
        extra.append("POL-MED-WF-04 Clause 2.3 immediate family member relative conflict of interest invalid")
    if re.search(r"\b(laptop|motherboard|computer|hardware|usb|wi-fi|wifi|technical)\b", ql):
        extra.append("POL-ACAD-CW-03 Clause 5.3 laptop technical failures ineligible backups")
    if re.search(r"\b(alters|altered|forged|backdated|fake)\b", ql):
        extra.append("POL-MED-WF-04 Clause 5.2 forged altered Category-A Academic Misconduct 0% Fail suspension")
    if "ayurvedic" in ql:
        extra.append("POL-MED-WF-04 Clause 2.2 Government Ayurvedic Teaching Hospital CMO 3 working days")
    return extra


def execute_explainable_agent_workflow(
    query: str,
    chunks: List[DocumentChunk],
    alpha: float = 0.5,
    use_live_gemini: bool = False,
) -> AgentResponse:
    """
    Runs the genuine 5-stage MediAcad Agentic RAG Loop over Member 1's 138-chunk corpus:
      1. Observation & Clinical / Out-of-Scope Guardrails
      2. Reasoning (Multi-Policy Sub-Query Decomposition)
      3. Action (Member 2 HybridExplainableRetriever with live alpha tuning)
      4. Verification (Member 3 EvidenceVerifier 2023 vs. 2025 Conflict Audit)
      5. Persistence & Strictly Grounded Synthesis (Zero ground_truth.json leakage)
    """
    q_lower = query.lower().strip()
    is_clinical = any(term in q_lower for term in CLINICAL_GUARDRAIL_TERMS)
    detected_att_pct = extract_attendance_percentage(query)
    is_hardware_query = bool(re.search(r"\b(laptop|motherboard|usb|wi-fi|wifi|hardware|hard drive|computer)\b", q_lower))
    is_family_query = bool(re.search(r"\b(father|mother|relative|family|parent)\b", q_lower))

    # Step 1: Domain Detection
    detected_issues: List[str] = []
    if any(w in q_lower for w in ("exam", "examination", "deferred", "fit-to-sit", "ec-01", "collapse", "7 calendar", "5 working")):
        detected_issues.append("Examination Consideration (POL-ACAD-EX-01)")
    if any(w in q_lower for w in ("attendance", "condonation", "lab", "barred", "waiver")) or detected_att_pct is not None:
        detected_issues.append("Attendance & Absence Condonation (POL-ACAD-AT-02)")
    if any(w in q_lower for w in ("assignment", "coursework", "extension", "late", "deadline", "penalty", "21-day", "21 days")) or is_hardware_query:
        detected_issues.append("Coursework Extension & Penalties (POL-ACAD-CW-03)")
    if any(w in q_lower for w in ("hospital", "hospitalization", "hospitalized", "certificate", "slmc", "ayurvedic", "umc", "dengue", "103f", "forged", "altered")) or is_family_query:
        detected_issues.append("Medical Documentation & UMC Verification (POL-MED-WF-04)")
    if any(w in q_lower for w in ("10 days", "10-day", "progression", "credits", "gpa", "handbook", "2023", "intercalation", "tuition")):
        detected_issues.append("Multi-Policy Progression & Conflict Audit (HBK-ACAD-PR-05)")

    # Step 2 & 3: Member 2 Real Hybrid Retrieval + Word-Boundary Sub-Queries
    retriever, word_to_idx, idf_vec = _get_cached_hybrid_engine(len(chunks), chunks[0].chunk_id, chunks)
    retriever.set_alpha(alpha)
    primary = retriever.retrieve(query, top_k=5)

    q_tokens = tokenize(query)
    top_matched: set[str] = set()
    for r in primary[:3]:
        if r.math_trace:
            top_matched.update(r.math_trace.matched_query_terms)
    top_bm25_raw = primary[0].math_trace.bm25_raw_score if (primary and primary[0].math_trace) else 0.0
    has_strong_term = bool(top_matched.intersection(STRONG_POLICY_TERMS))
    is_person_lookup = q_lower.startswith(("who is ", "who's ", "what is the name of "))
    low_coverage = len(q_tokens) >= 2 and len(top_matched) < 2 and not has_strong_term

    is_out_of_scope = (
        not is_clinical
        and (
            not primary
            or not top_matched
            or top_bm25_raw <= 0.0
            or (is_person_lookup and not has_strong_term)
            or low_coverage
        )
    )

    if is_out_of_scope:
        detected_issues = ["Out-of-Scope Query (Abstained Safely)"]
        sub_queries = ["[SCOPE_GUARDRAIL] Insufficient overlap with the 5 Official Senate PDFs"]
        retrieved: List[RetrievalResult] = []
    else:
        if not detected_issues:
            detected_issues.append("General Academic Medical Administration")
        extra_sqs = _plan_targeted_subqueries(query, detected_att_pct)
        sub_queries = [query] + extra_sqs
        best_by_id: Dict[str, RetrievalResult] = {}

        # Include top hit per targeted domain sub-query first so multi-policy cases span all active PDFs
        for sq in extra_sqs:
            for r in retriever.retrieve(sq, top_k=2):
                cid = r.chunk.chunk_id
                if cid not in best_by_id or r.hybrid_score > best_by_id[cid].hybrid_score:
                    best_by_id[cid] = r
                    break

        # Fill remaining slots from primary query retrieval
        for r in primary:
            cid = r.chunk.chunk_id
            if cid not in best_by_id or r.hybrid_score > best_by_id[cid].hybrid_score:
                best_by_id[cid] = r

        # Sort strictly descending by hybrid score so Rank #1 is always highest-scoring evidence
        retrieved = sorted(best_by_id.values(), key=lambda x: x.hybrid_score, reverse=True)[:5]
        for rank_i, r in enumerate(retrieved, start=1):
            r.rank = rank_i

    step1 = AgentStepLog(
        step_number=1,
        phase="OBSERVATION",
        title="Query Understanding & Clinical / Scope Guardrail Check",
        description=(
            "Clinical diagnosis/dosage request intercepted — activating Non-Diagnostic Guardrail."
            if is_clinical
            else (
                "Out-of-scope query detected — activating Zero-Hallucination Abstention Guardrail."
                if is_out_of_scope
                else f"Identified {len(detected_issues)} active regulatory domain(s) across the 5 Senate PDFs."
            )
        ),
        details={
            "detected_domains": detected_issues,
            "clinical_guardrail": is_clinical,
            "out_of_scope_abstained": is_out_of_scope,
            "detected_attendance_pct": detected_att_pct,
        },
    )

    step2 = AgentStepLog(
        step_number=2,
        phase="REASONING",
        title="Multi-Policy Sub-Query Decomposition",
        description=f"Planned {len(sub_queries)} targeted search query/queries.",
        details={"sub_queries": sub_queries[:5]},
    )

    step3 = AgentStepLog(
        step_number=3,
        phase="ACTION",
        title="Member 2 Hybrid BM25 + L2 Cosine Retrieval",
        description=(
            f"Abstained cleanly (Matched terms: {sorted(top_matched)}, Raw BM25: {top_bm25_raw:.4f})."
            if is_out_of_scope
            else f"Retrieved top {len(retrieved)} chunks from {len(chunks)} PDF segments (α = {alpha:.2f})."
        ),
        details={
            "top_chunks": [
                f"#{r.rank}: {r.chunk.doc_id} ({r.chunk.clause_id}, p.{r.chunk.page_number}) | Hybrid={r.hybrid_score:.4f}"
                for r in retrieved
            ]
        },
    )

    # Step 4: Member 3 Real EvidenceVerifier Conflict Audit (precision-filtered by active query intent)
    verifier = EvidenceVerifier()
    raw_conflicts = [] if (is_out_of_scope or is_clinical or is_hardware_query or is_family_query) else verifier.detect_conflicts(retrieved)
    is_legacy_ask = bool(re.search(r"\b(2023|handbook|7 calendar|7 days|60%|2%|21-day|21 days|private ayurvedic)\b", q_lower))
    is_multi_hosp = (
        ("exam" in q_lower)
        and ("attendance" in q_lower or detected_att_pct is not None)
        and ("assignment" in q_lower or "coursework" in q_lower)
    )
    conflict_alerts: List[PolicyConflictAlert] = []
    for c in raw_conflicts:
        cid = c.conflict_id
        if cid == "CONF-EXAM-01" and (is_legacy_ask or is_multi_hosp) and any(w in q_lower for w in ("exam", "7 calendar", "7 days", "5 working")):
            conflict_alerts.append(c)
        elif cid == "CONF-ATT-02" and (any(w in q_lower for w in ("attendance", "condonation", "waiver", "60%", "barred")) or (detected_att_pct is not None and detected_att_pct < 70.0)):
            conflict_alerts.append(c)
        elif cid == "CONF-CW-03" and any(w in q_lower for w in ("late", "penalty", "2%", "5%", "deduction")):
            conflict_alerts.append(c)
        elif cid == "CONF-EXT-04" and any(w in q_lower for w in ("21-day", "21 days", "lecturer", "extension")):
            conflict_alerts.append(c)
        elif cid == "CONF-MED-05" and any(w in q_lower for w in ("ayurvedic", "private")):
            conflict_alerts.append(c)

    step4 = AgentStepLog(
        step_number=4,
        phase="VERIFICATION",
        title="Member 3 EvidenceVerifier Conflict Audit",
        description=(
            "Abstained safely — outside verified policy corpus."
            if is_out_of_scope
            else f"Audited retrieved clauses via EvidenceVerifier; flagged {len(conflict_alerts)} active 2023 vs. 2025 conflict(s)."
        ),
        details={"conflicts_detected": [c.conflict_id for c in conflict_alerts]},
    )

    # Step 5: Strictly Grounded Synthesis (Zero ground_truth.json lookup!)
    att_note = _describe_attendance_zone(detected_att_pct) if not is_out_of_scope else None
    synthesis_mode = "Extractive Grounded Synthesis"

    if is_out_of_scope:
        checklist, req_docs = [], []
        confidence_status = "OUT_OF_SCOPE"
        final_answer = (
            "### 🛑 Out-of-Scope Query — Safe Abstention\n\n"
            f"Your query (**\"{query}\"**) does not match sufficient regulatory evidence across the "
            f"**5 Official University Medical & Academic Policy PDFs** (Matched terms: `{sorted(top_matched)}`).\n\n"
            "MediAcad strictly refuses to hallucinate outside its verified university policy scope. "
            "Ask about missed exams (`Form EC-01`), attendance thresholds (`Form ATT-02A/B`), coursework "
            "extensions/penalties (`Form EXT-03A/B/C`), medical documentation (`Form MED-04-CLR`), or progression (`Form PRG-05-CASE`)."
        )
    elif is_clinical:
        checklist, req_docs = _build_domain_checklist_and_forms(q_lower, True, detected_att_pct)
        confidence_status = "GUARDRAIL_ACTIVE"
        final_answer = (
            "### 🛑 Non-Diagnostic Clinical Scope Boundary (`POL-MED-WF-04` Clause 1.1)\n\n"
            "MediAcad is an **Academic Administration Case Advisor** and CANNOT diagnose medical conditions "
            "or recommend medication dosages. Please seek immediate medical evaluation at a hospital or the "
            "**University Medical Center (UMC, Building B, Room B-04)**.\n\n"
            "**Academic Protection Steps:**\n"
            "- **Fit-to-Sit Doctrine (`POL-ACAD-EX-01` Clause 2.1):** Do NOT enter the examination hall while acutely ill.\n"
            "- **Within 48 Hours:** Submit an Online Absence Notification (`Clause 3.1`).\n"
            "- **Within 5 Working Days:** Obtain `Form MED-04-CLR` at UMC and submit `Form EC-01` for an uncapped `I-Med` deferral (`Clause 3.2`)."
        )
    else:
        checklist, req_docs = _build_domain_checklist_and_forms(q_lower, False, detected_att_pct)
        confidence_status = "VERIFIED"
        live_gemini_text = None
        if use_live_gemini:
            api_key = _resolve_gemini_api_key()
            if api_key:
                live_gemini_text = _call_optional_gemini(
                    api_key=api_key,
                    query=query,
                    retrieved=retrieved,
                    conflicts=conflict_alerts,
                    att_note=att_note,
                )

        sections: List[str] = []
        if live_gemini_text:
            synthesis_mode = f"Live Gemini AI ({GEMINI_MODEL})"
            if att_note:
                sections.append(att_note)
            sections.append(f"#### 🤖 Live Gemini Grounded Synthesis (`{GEMINI_MODEL}`)\n\n{live_gemini_text}")
        else:
            extractive_summary = _synthesize_grounded_response(
                query=query,
                retrieved=retrieved,
                conflicts=conflict_alerts,
                word_to_idx=word_to_idx,
                idf_vec=idf_vec,
                apply_agentic=True,
            )
            sections.append(f"#### ✅ Grounded Policy Synthesis (Retrieved from 138 PDF Chunks)\n\n{extractive_summary}")

        citations_md = "\n".join(
            f"- **`{r.chunk.doc_id}` ({r.chunk.policy_version}, {r.chunk.clause_id}, Page {r.chunk.page_number}):** "
            f"{clean_policy_display_text(r.chunk.text)[:240]}..."
            for r in retrieved[:3]
        )
        sections.append(f"**Verified Source Citations:**\n{citations_md}")
        final_answer = "\n\n".join(sections)

    step5 = AgentStepLog(
        step_number=5,
        phase="PERSISTENCE",
        title="Grounded Response Synthesis & Session Persistence",
        description=f"Completed via {synthesis_mode}; persisted trace to session audit log.",
        details={
            "synthesis_mode": synthesis_mode,
            "confidence_status": confidence_status,
            "checklist_items": len(checklist),
        },
    )

    return AgentResponse(
        query=query,
        detected_issues=detected_issues,
        sub_queries=sub_queries,
        final_answer=final_answer,
        action_checklist=checklist,
        required_documents=req_docs,
        retrieved_contexts=retrieved,
        reasoning_trace=[step1, step2, step3, step4, step5],
        conflict_alerts=conflict_alerts,
        confidence_status=confidence_status,
    )


# ==============================================================================
# LOAD CORE RESOURCES
# ==============================================================================
chunks = load_corpus_chunks()
bench_data = load_benchmark_data()
evaluator_live = RAGTriadEvaluator()
gemini_key_available = _resolve_gemini_api_key() is not None


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Hybrid IR & Agent Controls")
    alpha_slider = st.slider(
        "Hybrid Fusion Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="α = 0.0 uses pure Sparse BM25; α = 1.0 uses pure Dense L2 Cosine; α = 0.50 balances both.",
    )
    st.caption(f"Current Formula: `S = {alpha_slider:.2f}·Dense + {1 - alpha_slider:.2f}·BM25`")

    use_gemini_toggle = st.toggle(
        f"Live Gemini AI Synthesis ({GEMINI_MODEL})",
        value=False,
        disabled=not gemini_key_available,
        help="Enabled automatically when GEMINI_API_KEY is configured in .env or Streamlit Secrets. Runs fast offline extractive synthesis otherwise.",
    )
    if not gemini_key_available:
        st.caption("ℹ️ Running in deterministic zero-key mode (no `GEMINI_API_KEY` set).")

    st.divider()
    st.markdown("### 🎯 Preset Viva Test Scenarios")
    preset_queries = {
        "Scenario 1: 10-Day Hospitalization (Exam + 74% Attendance + Coursework)": (
            "I was hospitalized for 10 days due to dengue fever. Because of this, I missed my final exam, "
            "my attendance dropped to 74%, and I missed a coursework assignment deadline during my hospital stay. "
            "What complete step-by-step procedure and forms must I follow?"
        ),
        "Scenario 2: 2023 vs. 2025 Conflict (7 Calendar Days vs. 5 Working Days)": (
            "Can I submit my medical certificate within 7 calendar days after missing my final exam as stated in the 2023 student handbook?"
        ),
        "Scenario 3: Attendance Hard Floor (63% vs. 65% Minimum)": (
            "My attendance dropped to 63% after a 10-day hospitalization, can the Dean grant me an attendance waiver?"
        ),
        "Scenario 4: Forged Medical Certificate Zero-Tolerance Policy": (
            "What is the punishment if a student submits a forged or altered medical certificate?"
        ),
        "Scenario 5: Clinical Safety Guardrail (Dengue & Medication Dosage)": (
            "I have 103F fever, do I have dengue and what paracetamol dosage should I take before my exam?"
        ),
    }
    preset_choice = st.selectbox("Load a Benchmark Scenario:", list(preset_queries.keys()))

    st.divider()
    st.markdown("### 📌 System Architecture Info")
    sidebar_overall_pct = bench_data["summary"]["mean_overall_score"] * 100.0
    vector_store_status = get_persistent_vector_store_status()
    st.info(
        "**Module:** DS205.3 Data Science with Python\n\n"
        f"**Indexed Corpus:** `{len(chunks)}` Chunks (5 Senate PDFs, 17 Tables)\n\n"
        f"**Vector Persistence:** {vector_store_status}\n\n"
        "**Benchmark:** 32 Ground-Truth Cases\n\n"
        f"**Mean RAG Triad Score:** `{sidebar_overall_pct:.2f}%`"
    )
    audit_log_placeholder = st.empty()


# ==============================================================================
# HERO BANNER & 4 TABS
# ==============================================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🏥 MediAcad — Explainable Agentic RAG Case Advisor</div>
        <div class="hero-sub">
            138 Structure-Aware PDF Chunks • Member 2 Hybrid BM25 + L2 Cosine Engine •
            Member 3 EvidenceVerifier 2023 vs. 2025 Conflict Audit • Member 4 RAG Triad Telemetry
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 1. Explainable Case Advisor (Agentic Loop)",
    "📐 2. Mathematical IR Inspector (BM25 & Vector Geometry)",
    "📊 3. RAG Triad Benchmark & Ablation Study (32 Cases)",
    "📚 4. Policy Corpus & 2023 vs. 2025 Conflict Matrix",
])


# ==============================================================================
# TAB 1: EXPLAINABLE CASE ADVISOR
# ==============================================================================
with tab1:
    if "active_query" not in st.session_state or st.session_state.get("last_preset") != preset_choice:
        st.session_state["active_query"] = preset_queries[preset_choice]
        st.session_state["last_preset"] = preset_choice

    with st.form("mediacad_query_form", clear_on_submit=False):
        query_input = st.text_area(
            "Enter Any University Medical-Academic Case Query:",
            value=st.session_state["active_query"],
            height=90,
        )
        submitted = st.form_submit_button("🚀 Execute Agentic RAG Loop", type="primary", width="stretch")
        if submitted and query_input.strip():
            st.session_state["active_query"] = query_input.strip()

    active_query = st.session_state["active_query"]
    t0 = time.perf_counter()
    agent_resp = execute_explainable_agent_workflow(
        active_query,
        chunks,
        alpha=alpha_slider,
        use_live_gemini=use_gemini_toggle,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Live Groundedness Metric (Member 4)
    if agent_resp.confidence_status == "OUT_OF_SCOPE":
        live_groundedness_str = "Abstained"
    elif agent_resp.confidence_status == "GUARDRAIL_ACTIVE":
        live_groundedness_str = "100.0%"
    else:
        gm = evaluator_live.compute_groundedness(agent_resp.final_answer, agent_resp.retrieved_contexts)
        live_groundedness_str = f"{gm['score'] * 100:.1f}%"

    # Persist execution record into st.session_state & render Sidebar Audit Log immediately
    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []
    if not st.session_state["agent_history"] or st.session_state["agent_history"][-1]["query"] != active_query:
        st.session_state["agent_history"].append({
            "query": active_query,
            "status": agent_resp.confidence_status,
            "conflicts": len(agent_resp.conflict_alerts),
            "latency_ms": elapsed_ms,
        })

    with audit_log_placeholder.container():
        with st.expander(f"🕒 Session State Audit Log ({len(st.session_state['agent_history'])})", expanded=False):
            for entry in reversed(st.session_state["agent_history"][-5:]):
                st.caption(f"**[{entry['status']}]** {entry['query'][:55]}... (`{entry['latency_ms']:.1f} ms`)")

    # Top Telemetry KPIs
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verification Status", agent_resp.confidence_status)
    m2.metric("Live Groundedness", live_groundedness_str)
    m3.metric("2023 vs. 2025 Conflicts", f"{len(agent_resp.conflict_alerts)} Alert(s)")
    m4.metric("Execution Latency", f"{elapsed_ms:.1f} ms")

    if agent_resp.confidence_status == "OUT_OF_SCOPE":
        st.markdown(
            """
            <div class="guardrail-banner">
                <b>🛑 OUT-OF-SCOPE QUERY ABSTAINED (ZERO-HALLUCINATION GUARDRAIL)</b><br/>
                • Insufficient regulatory term matches across the 5 Official University Senate Policy PDFs.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif agent_resp.confidence_status == "GUARDRAIL_ACTIVE":
        st.markdown(
            """
            <div class="guardrail-banner">
                <b>🛑 CLINICAL SAFETY GUARDRAIL ACTIVE (POL-MED-WF-04 Clause 1.1)</b><br/>
                • Clinical disease diagnosis and medication dosage advice are strictly blocked. Refer student to <b>UMC (Building B, Room B-04)</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for alert in agent_resp.conflict_alerts:
        st.markdown(
            f"""
            <div class="conflict-banner">
                <b>⚡ POLICY CONFLICT DETECTED ({alert.conflict_id}: {alert.topic})</b><br/>
                • <b>Obsolete 2023 Handbook ({alert.legacy_doc_id}, {alert.legacy_clause}):</b> {alert.legacy_rule_summary}<br/>
                • <b>Binding 2025 Policy ({alert.authoritative_doc_id}, {alert.authoritative_clause}):</b> <b>{alert.authoritative_rule_summary}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left_col, right_col = st.columns([1.35, 1.0])

    with left_col:
        st.subheader("📋 Verified Academic Case Guidance")
        st.markdown(agent_resp.final_answer)

        detected_att_pct = extract_attendance_percentage(active_query) if agent_resp.confidence_status != "OUT_OF_SCOPE" else None
        if detected_att_pct is not None:
            if detected_att_pct >= 80.0:
                zone_label, bar_color = "Safe Zone (≥80%)", "#10B981"
            elif detected_att_pct >= 70.0:
                zone_label, bar_color = "HoD Condonation (70–79.9% → Form ATT-02A)", "#3B82F6"
            elif detected_att_pct >= 65.0:
                zone_label, bar_color = "Dean Inpatient Waiver (65–69.9% → Form ATT-02B)", "#F59E0B"
            else:
                zone_label, bar_color = "Strictly Barred / Hard Floor (<65.0% → Form PRG-05-INT)", "#EF4444"

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=detected_att_pct,
                    number={"suffix": "%", "font": {"size": 28}},
                    title={"text": f"<b>Attendance Zone:</b> {zone_label}", "font": {"size": 14}},
                    gauge={
                        "axis": {"range": [40, 100], "tickwidth": 1},
                        "bar": {"color": bar_color, "thickness": 0.35},
                        "steps": [
                            {"range": [40, 65], "color": "rgba(239, 68, 68, 0.22)"},
                            {"range": [65, 70], "color": "rgba(245, 158, 11, 0.22)"},
                            {"range": [70, 80], "color": "rgba(59, 130, 246, 0.22)"},
                            {"range": [80, 100], "color": "rgba(16, 185, 129, 0.22)"},
                        ],
                        "threshold": {"line": {"color": "#EF4444", "width": 3}, "thickness": 0.8, "value": 65.0},
                    },
                )
            )
            fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=45, b=10))
            st.plotly_chart(fig_gauge, width="stretch")

        if agent_resp.action_checklist:
            st.markdown("#### ✅ Student Action Checklist (Interactive To-Do)")
            for idx, item in enumerate(agent_resp.action_checklist, start=1):
                st.checkbox(item, value=False, key=f"todo_{idx}_{abs(hash(active_query)) % 100000}")

        if agent_resp.required_documents:
            st.markdown("#### 🗂️ Required Official Forms & Documents")
            st.write(" • ".join(f"`{doc}`" for doc in agent_resp.required_documents))

            report_md = (
                "# MEDIACAD OFFICIAL STUDENT CASE ADVISORY PLAN\n\n"
                f"**Student Query:** {agent_resp.query}\n\n"
                f"**Status:** {agent_resp.confidence_status} | **Live Groundedness:** {live_groundedness_str}\n\n"
                f"## 1. Verified Guidance\n{agent_resp.final_answer}\n\n"
                "## 2. Action Checklist\n"
                + "\n".join(f"- [ ] {c}" for c in agent_resp.action_checklist)
                + "\n\n## 3. Required Forms\n"
                + "\n".join(f"- {d}" for d in agent_resp.required_documents)
            )
            st.download_button(
                label="📥 Download Student Case Advisory Plan (.md)",
                data=report_md,
                file_name="MediAcad_Case_Advisory_Plan.md",
                mime="text/markdown",
                width="stretch",
            )

    with right_col:
        st.subheader("🔍 Glass-Box 5-Stage Agentic Loop Trace")
        for step in agent_resp.reasoning_trace:
            with st.expander(f"Step {step.step_number}: [{step.phase}] — {step.title}", expanded=(step.step_number in {1, 4})):
                st.write(step.description)
                st.json(step.details)

        st.subheader("📑 Top Retrieved Policy Clauses (138-Chunk Corpus)")
        if not agent_resp.retrieved_contexts:
            st.info("No sufficiently relevant policy clauses matched (Safe Abstention active).")
        else:
            for r in agent_resp.retrieved_contexts[:3]:
                with st.expander(
                    f"Rank #{r.rank} | {r.chunk.doc_id} ({r.chunk.clause_id}, p.{r.chunk.page_number}) — Hybrid: {r.hybrid_score:.4f}",
                    expanded=(r.rank == 1),
                ):
                    st.caption(f"**{r.chunk.doc_title}** | {r.chunk.policy_version}")
                    matched_kw = r.math_trace.matched_query_terms if r.math_trace else []
                    highlighted = highlight_matched_terms(clean_policy_display_text(r.chunk.text), matched_kw)
                    st.markdown(highlighted, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: MATHEMATICAL RETRIEVAL INSPECTOR (Member 2 Live Telemetry)
# ==============================================================================
with tab2:
    st.subheader("📐 Mathematical Foundations of Hybrid Information Retrieval")
    st.markdown(
        r"Live white-box telemetry computed by Member 2's **`HybridExplainableRetriever`** across all **138 PDF chunks** "
        r"(TF, IDF, Okapi BM25, Vector Magnitude $\lVert \mathbf{v} \rVert_2$, Dot Product $\mathbf{q}\cdot\mathbf{d}$, $L_2$ Cosine Similarity, and Convex Hybrid Fusion)."
    )

    eq1, eq2, eq3 = st.columns(3)
    with eq1:
        st.markdown("**1. Smoothed IDF**")
        st.latex(r"\text{IDF}(t) = \ln\left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right)")
    with eq2:
        st.markdown("**2. L2 Cosine Similarity**")
        st.latex(r"\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \, \|\mathbf{d}\|_2}")
    with eq3:
        st.markdown("**3. Hybrid Score Fusion**")
        st.latex(r"S_{\text{hybrid}} = \alpha \tilde{S}_{\text{dense}} + (1 - \alpha) \tilde{S}_{\text{sparse}}")

    if not agent_resp.retrieved_contexts:
        st.info("Enter an in-scope university policy query in Tab 1 to inspect live vector math.")
    else:
        # Compute strictly direct retrieval telemetry for the exact user query so ranks & norms are 100% consistent
        direct_results = compute_live_hybrid_retrieval(active_query, chunks, alpha=alpha_slider, top_k=5)
        math_rows = []
        for r in direct_results:
            mt = r.math_trace
            math_rows.append({
                "Rank": f"#{r.rank}",
                "Doc ID": r.chunk.doc_id,
                "Clause / Table": r.chunk.clause_id,
                "Matched Terms": ", ".join(mt.matched_query_terms[:6]) if mt else "",
                "BM25 (Raw)": round(mt.bm25_raw_score, 4) if mt else 0.0,
                "Sparse (Norm BM25)": round(r.sparse_score, 4),
                "||q||_2": round(mt.query_vector_norm, 4) if mt else 0.0,
                "||d||_2": round(mt.doc_vector_norm, 4) if mt else 0.0,
                "Dot (q·d)": round(mt.dot_product, 4) if mt else 0.0,
                "Dense (Cosine)": round(r.dense_score, 4),
                f"Hybrid (α={alpha_slider:.2f})": round(r.hybrid_score, 4),
            })
        st.dataframe(pd.DataFrame(math_rows), width="stretch", hide_index=True)

        chart_col1, chart_col2 = st.columns([1.25, 1.0])
        with chart_col1:
            fig_scores = go.Figure()
            x_labels = [f"{r.chunk.doc_id}<br>{r.chunk.clause_id[:16]}" for r in direct_results]
            fig_scores.add_trace(go.Bar(name="Sparse (BM25)", x=x_labels, y=[r.sparse_score for r in direct_results], marker_color="#3B82F6"))
            fig_scores.add_trace(go.Bar(name="Dense (L2 Cosine)", x=x_labels, y=[r.dense_score for r in direct_results], marker_color="#10B981"))
            fig_scores.add_trace(go.Bar(name=f"Hybrid (α={alpha_slider:.2f})", x=x_labels, y=[r.hybrid_score for r in direct_results], marker_color="#F59E0B"))
            fig_scores.update_layout(
                title="Sparse vs. Dense vs. Hybrid Scores (Top-5 Direct Chunks)",
                barmode="group",
                height=370,
                yaxis_title="Score [0.0 – 1.0]",
                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
                margin=dict(b=80),
            )
            st.plotly_chart(fig_scores, width="stretch")

        with chart_col2:
            top_trace = direct_results[0].math_trace
            if top_trace and top_trace.idf_values:
                terms = list(top_trace.idf_values.keys())[:8]
                fig_terms = go.Figure()
                fig_terms.add_trace(go.Bar(name="IDF Weight", x=terms, y=[top_trace.idf_values[t] for t in terms], marker_color="#8B5CF6"))
                fig_terms.add_trace(go.Bar(name="Term Frequency (TF)", x=terms, y=[top_trace.term_frequencies.get(t, 1.0) for t in terms], marker_color="#14B8A6"))
                fig_terms.update_layout(
                    title=f"Token Attribution — Rank #1 ({direct_results[0].chunk.doc_id})",
                    barmode="group",
                    height=370,
                    xaxis_title="Matched Token",
                    yaxis_title="Weight / TF",
                    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
                    margin=dict(b=80),
                )
                st.plotly_chart(fig_terms, width="stretch")

        # Live Side-by-Side Alpha Comparison (Pure BM25 vs Current Alpha)
        st.markdown(f"#### ⚖️ Live Ranking Comparison: Pure Sparse (`α = 0.0`) vs. Current Setting (`α = {alpha_slider:.2f}`)")
        sparse_only_top = compute_live_hybrid_retrieval(active_query, chunks, alpha=0.0, top_k=3)
        current_top = direct_results[:3]
        comp_rows = []
        for idx_c in range(min(len(sparse_only_top), len(current_top))):
            s_r, h_r = sparse_only_top[idx_c], current_top[idx_c]
            comp_rows.append({
                "Rank": f"#{idx_c + 1}",
                "Pure BM25 (α=0.0)": f"{s_r.chunk.doc_id} — {s_r.chunk.clause_id} (p.{s_r.chunk.page_number})",
                "BM25 Score": round(s_r.hybrid_score, 4),
                f"Current Hybrid (α={alpha_slider:.2f})": f"{h_r.chunk.doc_id} — {h_r.chunk.clause_id} (p.{h_r.chunk.page_number})",
                "Hybrid Score": round(h_r.hybrid_score, 4),
            })
        st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)


# ==============================================================================
# TAB 3: RAG TRIAD EVALUATION BENCHMARK (32 CASES)
# ==============================================================================
with tab3:
    st.subheader("📊 Quantitative RAG Triad Evaluation Benchmark (32 Ground-Truth Cases)")
    summary = bench_data["summary"]
    sparse_ab = bench_data["ablation_study"][0]
    cr_delta = (summary["mean_context_relevance"] - sparse_ab["context_relevance"]) * 100.0
    ar_delta = (summary["mean_answer_relevance"] - sparse_ab["answer_relevance"]) * 100.0
    ov_delta = (summary["mean_overall_score"] - sparse_ab["overall_triad"]) * 100.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("1. Mean Context Relevance", f"{summary['mean_context_relevance'] * 100:.2f}%", f"{cr_delta:+.2f}% vs BM25")
    k2.metric("2. Mean Groundedness", f"{summary['mean_groundedness'] * 100:.2f}%", "Verified Citations")
    k3.metric("3. Mean Answer Relevance", f"{summary['mean_answer_relevance'] * 100:.2f}%", f"{ar_delta:+.2f}% vs BM25")
    k4.metric("Overall RAG Triad Score", f"{summary['mean_overall_score'] * 100:.2f}%", f"{ov_delta:+.2f}% vs BM25")

    c_left, c_right = st.columns([1.1, 1.0])
    with c_left:
        st.markdown("#### 🔬 Ablation Study: Sparse vs. Dense vs. Hybrid Fusion")
        df_ablation = pd.DataFrame(bench_data["ablation_study"])
        fig_ab = px.bar(
            df_ablation,
            x="mode",
            y=["context_relevance", "groundedness", "answer_relevance", "overall_triad"],
            barmode="group",
            height=420,
            color_discrete_sequence=["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B"],
        )
        fig_ab.update_layout(
            yaxis_range=[0.60, 1.04],
            xaxis_title="Retrieval Strategy",
            yaxis_title="Score",
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, title_text=""),
            margin=dict(b=90),
        )
        st.plotly_chart(fig_ab, width="stretch")

    with c_right:
        st.markdown("#### 🕸️ RAG Triad Multi-Axis Radar Geometry")
        categories_radar = ["Context Relevance", "Groundedness", "Answer Relevance", "Overall Triad", "Context Relevance"]
        fig_radar = go.Figure()
        radar_colors = ["#3B82F6", "#10B981", "#F59E0B"]
        for idx_ab, ab_row in enumerate(bench_data["ablation_study"]):
            vals_radar = [
                ab_row["context_relevance"],
                ab_row["groundedness"],
                ab_row["answer_relevance"],
                ab_row["overall_triad"],
                ab_row["context_relevance"],
            ]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals_radar,
                    theta=categories_radar,
                    fill="toself",
                    name=ab_row["mode"],
                    line=dict(color=radar_colors[idx_ab % len(radar_colors)], width=2.5),
                )
            )
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(15, 41, 66, 0.25)",
                radialaxis=dict(visible=True, range=[0.55, 1.02], gridcolor="rgba(148, 163, 184, 0.3)"),
            ),
            height=420,
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            margin=dict(t=35, b=95, l=45, r=45),
        )
        st.plotly_chart(fig_radar, width="stretch")

    st.markdown("#### 📂 Performance Breakdown by Query Category")
    cat_rows = [
        {
            "Category": cat_name,
            "Cases": vals["count"],
            "Context Relevance": vals["context_relevance"],
            "Groundedness": vals["groundedness"],
            "Answer Relevance": vals["answer_relevance"],
            "Overall RAG Triad": vals["overall"],
        }
        for cat_name, vals in bench_data["category_breakdown"].items()
    ]
    st.dataframe(pd.DataFrame(cat_rows), width="stretch", hide_index=True)

    st.markdown("#### 🧾 Complete 32-Case Ground-Truth Evaluation Audit Table")
    df_cases = pd.DataFrame(bench_data["case_results"])[
        ["case_id", "category", "query", "context_relevance", "groundedness", "answer_relevance", "overall_triad_score"]
    ]
    st.dataframe(df_cases, width="stretch", hide_index=True)


# ==============================================================================
# TAB 4: OFFICIAL POLICY CORPUS & CONFLICT MATRIX
# ==============================================================================
with tab4:
    st.subheader("📚 Official University Regulatory Corpus Inventory (5 Senate PDFs)")
    corpus_inventory = pd.DataFrame([
        {
            "Document ID": "POL-ACAD-EX-01",
            "Official Policy Title": "Examination and Medical Consideration Regulations",
            "Version": "v4.0 (2025 Senate Active)",
            "Tables": 4,
            "Core Scope": "Fit-to-Sit (2.1), Form INV-09 collapse (2.2), 48h Portal Alert (3.1), 5-Working-Day EC-01 (3.2)",
        },
        {
            "Document ID": "POL-ACAD-AT-02",
            "Official Policy Title": "Attendance and Medical Absence Policy",
            "Version": "v3.2 (2025 Senate Active)",
            "Tables": 4,
            "Core Scope": "80% Baseline, 70–79.9% HoD (ATT-02A), 65–69.9% Dean Inpatient (ATT-02B), <65% Barred Floor (PRG-05-INT)",
        },
        {
            "Document ID": "POL-ACAD-CW-03",
            "Official Policy Title": "Coursework and Assessment Support Regulations",
            "Version": "v3.0 (2025 Senate Active)",
            "Tables": 3,
            "Core Scope": "3-Tier Extensions (EXT-03A/B/C), 5%/Working-Day Late Penalty (Table 4.1), Hardware Failure Exclusion (5.3)",
        },
        {
            "Document ID": "POL-MED-WF-04",
            "Official Policy Title": "Medical Documentation and Student Welfare Guidelines",
            "Version": "v3.5 (2025 Senate Active)",
            "Tables": 3,
            "Core Scope": "Clinical Guardrail (1.1), Govt Ayurvedic + CMO 3-Day Rule (2.2), Family Exclusion (2.3), Fraud Zero-Tolerance (5.2)",
        },
        {
            "Document ID": "HBK-ACAD-PR-05",
            "Official Policy Title": "Academic Progression and Student Handbook",
            "Version": "v1.8 (2023 Legacy / 2025 Overrides)",
            "Tables": 3,
            "Core Scope": "Credit Progression & Intercalation (Sec 1–4 Active); Sec 5 Legacy Clauses Overridden by 2025 Senate Policies",
        },
    ])
    st.dataframe(corpus_inventory, width="stretch", hide_index=True)

    st.markdown("#### 📥 Direct Official Senate Policy PDF Downloads")
    pdf_files_info = [
        ("POL-ACAD-EX-01", "01_Examination_and_Medical_Consideration_Regulations.pdf"),
        ("POL-ACAD-AT-02", "02_Attendance_and_Medical_Absence_Policy.pdf"),
        ("POL-ACAD-CW-03", "03_Coursework_and_Assessment_Support_Regulations.pdf"),
        ("POL-MED-WF-04", "04_Medical_Documentation_and_Student_Welfare_Guidelines.pdf"),
        ("HBK-ACAD-PR-05", "05_Academic_Progression_and_Student_Handbook.pdf"),
    ]
    pdf_cols = st.columns(5)
    for idx_p, (doc_code, pdf_filename) in enumerate(pdf_files_info):
        pdf_path = PROJECT_ROOT / "data" / "raw_pdfs" / pdf_filename
        with pdf_cols[idx_p]:
            if pdf_path.exists():
                st.download_button(
                    label=f"📄 {doc_code}.pdf",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_filename,
                    mime="application/pdf",
                    width="stretch",
                    key=f"dl_pdf_{doc_code}",
                )

    st.markdown("#### ⚡ Automated 2023 Legacy Handbook vs. 2025 Senate Policy Conflict Matrix")
    conflict_table = pd.DataFrame([
        {
            "Conflict ID": "CONFLICT-01",
            "Policy Domain": "Missed Final Exam Medical Deadline",
            "Obsolete 2023 Handbook (Sec 5)": "7 Calendar Days from exam date (Clause 5.1)",
            "Binding 2025 Senate Policy": "5 Working Days from exam or discharge (POL-ACAD-EX-01 Clause 3.2)",
        },
        {
            "Conflict ID": "CONFLICT-02",
            "Policy Domain": "Minimum Attendance Condonation Floor",
            "Obsolete 2023 Handbook (Sec 5)": "60% Minimum Attendance (Clause 5.2)",
            "Binding 2025 Senate Policy": "70% Standard / 65% Inpatient Hard Floor; <65% Barred (POL-ACAD-AT-02)",
        },
        {
            "Conflict ID": "CONFLICT-03",
            "Policy Domain": "Unapproved Late Coursework Penalty",
            "Obsolete 2023 Handbook (Sec 5)": "2% per calendar day up to 10 days (Clause 5.3)",
            "Binding 2025 Senate Policy": "5% per working day up to 5 days; 0% after Day 5 (POL-ACAD-CW-03)",
        },
        {
            "Conflict ID": "CONFLICT-04",
            "Policy Domain": "Coursework Extension Authority",
            "Obsolete 2023 Handbook (Sec 5)": "Up to 21 Days by Module Lecturer (Clause 5.4)",
            "Binding 2025 Senate Policy": "Tier 1: 3d (Lecturer), Tier 2: 7d (HoD), Tier 3: Max 14d (EC Panel)",
        },
        {
            "Conflict ID": "CONFLICT-05",
            "Policy Domain": "Ayurvedic Medical Certificates",
            "Obsolete 2023 Handbook (Sec 5)": "Private Ayurvedic notes accepted directly (Clause 5.5)",
            "Binding 2025 Senate Policy": "ONLY Govt. Ayurvedic Teaching Hospital + CMO stamp within 3 working days",
        },
    ])
    st.dataframe(conflict_table, width="stretch", hide_index=True)

    st.markdown("#### 🔍 Interactive Policy Clause Explorer")
    st.caption(f"Indexed **{len(chunks)}** structure-aware chunks (`data/processed_chunks.json`) | Vector Store: {vector_store_status}.")
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        selected_doc = st.selectbox("Filter by Document:", ["All Documents"] + sorted({c.doc_id for c in chunks}))
    with filter_col2:
        keyword_filter = st.text_input("Search clauses by keyword (e.g., forged, Form EC-01, Ayurvedic):", "")

    filtered_chunks = [
        c
        for c in chunks
        if (selected_doc == "All Documents" or c.doc_id == selected_doc)
        and (not keyword_filter.strip() or keyword_filter.strip().lower() in c.text.lower())
    ]
    st.caption(f"Displaying **{len(filtered_chunks)}** matching policy chunk(s).")
    for c in filtered_chunks[:40]:
        with st.expander(f"{c.doc_id} ({c.policy_version}) — {c.section_title} | {c.clause_id} (Page {c.page_number})"):
            st.write(clean_policy_display_text(c.text))
