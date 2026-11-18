# MediAcad: Explainable Agentic RAG
## Member 03 — Technical Report Contribution & Viva Voce Guide
**Role:** Agentic Architecture, OOP & LLM Integration (25% Equal Contribution)  
**Package:** `src/agent/` & `src/interfaces/base_agent.py`  
**Git Branch:** `member3-agent`  

---

## 📑 SECTION III: Agentic Architecture & Multi-Step Reasoning (~600 Words)

### 1. Architectural Philosophy & The 6-Step Agentic Loop
Standard RAG systems rely on naive single-shot semantic vector search. When a university student presents a complex medical emergency—such as being hospitalized for 10 days with Dengue fever—a single similarity query retrieves fragmented chunks and fails to reconcile contradictory regulatory policies. 

To solve this, MediAcad Member 03 engineered a stateful, explainable **6-Step Agentic Loop** operating through **Observation $\rightarrow$ Reasoning $\rightarrow$ Action $\rightarrow$ Persistence $\rightarrow$ Verification $\rightarrow$ Grounded Synthesis**:

1. **Phase 1: Observation (`QueryPlanner.observe`)**:
   The agent ingests the unstructured student narrative and parses critical quantitative entities (e.g., $10\text{ days}$ hospitalization, $72\%$ attendance) and medical events. It detects the three concurrent academic crises: missed final examinations, attendance shortfall below the 80% threshold, and overdue continuous assessment coursework.

2. **Phase 2: Reasoning & Multi-Step Decomposition (`QueryPlanner`)**:
   Instead of dispatching one noisy query, the planner decomposes the crisis into five domain-targeted sub-queries mapped directly to the university's regulatory framework:
   * **Welfare Sub-Query:** UMC Chief Medical Officer clearance procedure (`POL-MED-WF-04`).
   * **Examination Sub-Query:** Missed final exam, Form EC-01, and hospitalization discharge extensions (`POL-ACAD-EX-01`).
   * **Attendance Sub-Query:** Condonation bands ($70.0\%\text{--}79.9\%$ vs. $65.0\%\text{--}69.9\%$ exceptional Dean waiver) (`POL-ACAD-AT-02`).
   * **Coursework Sub-Query:** Tier-3 hospitalization extensions and late penalty waivers (`POL-ACAD-CW-03`).
   * **Progression Sub-Query:** Consolidated 6-step roadmap and prerequisite credit tracking (`HBK-ACAD-PR-05`).

3. **Phase 3: Action & Multi-Document Retrieval**:
   The agent queries the underlying retriever (sparse BM25 + dense semantic vector embeddings) for each sub-query, gathering evidence chunks while capturing granular mathematical explainability traces (BM25 scores, term frequencies, vector magnitudes, dot products, and cosine similarities).

4. **Phase 4: Persistence (`AgentState` & `PersistentSessionLogger`)**:
   Every intermediate state transition, identified issue, and retrieval trace is serialized to disk as structured JSON (`data/sessions/session_<id>.json`), guaranteeing complete white-box replayability and compliance auditing.

5. **Phase 5: Verification & Conflict Detection (`EvidenceVerifier`)**:
   The verification engine inspects retrieved contexts against a confidence threshold ($\tau = 0.40$). If evidence is lacking, it dynamically generates refined queries for re-retrieval. Simultaneously, the **Automated Policy Conflict Detector** audits discrepancies between the 2023 Legacy Student Handbook and the 2025 Mandatory Senate Revisions (resolving submission clocks, attendance floors, and penalty rates).

6. **Phase 6: Grounded Synthesis & Guardrails (`MediAcadAgent`)**:
   Leveraging Google Gemini LLM API (with deterministic fallback for zero-downtime offline execution), the agent synthesizes a multi-step grounded resolution with exact clause citations (`[POL-ACAD-EX-01] Clause 3.2`), official form codes (`Form EC-01`, `Form ATT-02A`, `Form EXT-03C`, `Form MED-04-CLR`, `Form PRG-05-CASE`), and a step-by-step action roadmap.

---

## 🛡️ SECTION III (Cont.): Automated Policy Conflict Detection & Guardrails

A major innovation of Member 03's architecture is the **Automated Policy Conflict Detector**. When policies evolve over time, universities frequently have legacy documents in circulation that contradict modern Senate revisions:

| Conflict ID | Topic | Legacy 2023 Handbook | Authoritative 2025 Senate Policy | Resolution & Precedence |
|---|---|---|---|---|
| **CONF-EXAM-01** | Missed Exam Application Deadline | 7 calendar days | **5 working days from discharge** (`POL-ACAD-EX-01`, Clause 3.2) | 2025 Policy takes absolute precedence; countdown starts upon discharge. |
| **CONF-ATT-02** | Attendance Condonation Floor | Condonable down to 60% | **70%–79.9% HOD / 65% Hard Floor** (`POL-ACAD-AT-02`, Table 2.1) | 72% attendance falls into 70%–79.9% HOD Condonation band via Form ATT-02A. |
| **CONF-CW-03** | Late Coursework Penalty | 2% daily deduction up to 10 days | **5% per working day up to 5 days** (`POL-ACAD-CW-03`, Clause 4.1) | Retrospective Tier-3 Form EXT-03C waives the 5% daily penalty entirely. |
| **CONF-EXT-04** | Coursework Extension Limits | Lecturers grant up to 21 days | **Tier-1: 3 days, Tier-2: 7 days, Tier-3: 14 days** (`POL-ACAD-CW-03`, Table 2.1) | Informal lecturer promises invalid; Faculty EC Panel grants up to 14 days. |
| **CONF-MED-05** | Medical Certificate Authority | Private Ayurvedic notes accepted | **SLMC / Govt Ayurvedic + UMC CMO Countersignature** (`POL-MED-WF-04`) | Central UMC clearance slip (`Form MED-04-CLR`) mandatory. |

---

## 🔍 SECTION V: Personal Reflection & Engineering Lessons (~300 Words)

Leading the design and implementation of Member 03's Agentic Architecture presented both algorithmic and software engineering challenges:

1. **Object-Oriented Design & Zero-Blocking Strategy:**
   Developing in parallel across four team members typically leads to severe blocking. By defining explicit Abstract Base Classes (`BaseAgent`, `BaseRetriever`) and strongly-typed dataclasses in Day 1 shared contracts, I was able to build and test the entire 6-Step Agentic Loop using a local `MockRetriever` before Member 1 and Member 2 finalized their chunkers and vector indices.
2. **Dependency Injection:**
   Decoupling the retrieval engine and LLM synthesizer via constructor Dependency Injection (`MediAcadAgent(retriever=..., llm_client=...)`) proved invaluable. This design pattern allowed unit testing with deterministic mocks in CI while supporting seamless drop-in integration of Member 2's hybrid BM25 + dense retriever.
3. **Guardrails Against LLM Hallucinations:**
   Ensuring that generative AI provides strictly grounded advice required strict system prompts and programmatic verification guardrails. Implementing explicit missing evidence warnings prevents the agent from fabricating administrative procedures not present in official university regulations.

---

## 🎤 Viva Voce Preparation & Key Questions

### Question 1: Why did you use Dependency Injection in your Agent?
**Answer:**  
*"We used Dependency Injection in `MediAcadAgent.__init__(self, retriever=..., ...)` to satisfy the Dependency Inversion Principle (SOLID). This decoupled the agent's reasoning logic from the underlying retrieval implementation. It enabled Zero-Blocking development: I could test the complete Agentic Loop on Day 1 using `MockRetriever`, and when Member 2 completed the Hybrid Vector Engine, we simply injected `HybridExplainableRetriever` without altering a single line of agent code."*

### Question 2: How does the Agent detect and resolve policy conflicts?
**Answer:**  
*"The `EvidenceVerifier` scans retrieved chunks for legacy metadata and triggers (such as 2023 edition clauses). When older rules (e.g., 7 calendar days or 60% attendance) are retrieved alongside updated 2025 Senate revisions (5 working days or 65% hard floor), the conflict engine flags a `PolicyConflictAlert` and programmatically enforces the 2025 Senate revision as authoritative."*

### Question 3: How do you prevent hallucinations when policies have missing information?
**Answer:**  
*"We enforce a two-tier guardrail: First, `EvidenceVerifier` checks retrieval confidence against a 0.40 threshold and verifies keyword coverage across required domains. Second, the system prompt strictly instructs the LLM never to extrapolate or fabricate non-existent university policies, outputting an explicit missing information warning and referring the student to the Academic Registrar."*

---

## 🎬 5-Minute Video Presentation Script (Timestamp 2:30 — 3:45)

* **2:30 – 2:45:** *"ආයුබෝවන්, මම Member 3. මම MediAcad පද්ධතියේ Agentic Architecture, 6-Step Loop සහ Policy Conflict Detection කොටස මෙහෙයවනු ලැබුවා."*
* **2:45 – 3:05:** *"ශිෂ්‍යයෙක් තමන්ගේ ගැටලුව ඉදිරිපත් කළාම, අපේ QueryPlanner එක Observation සහ Reasoning පියවර හරහා ගැටලුව තනි ප්‍රශ්නයක් ලෙස නොව, Policies 5ටම අදාළව Sub-queries 5කට decompose කරනවා."*
* **3:05 – 3:25:** *"ඊළඟට Action පියවරෙන් chunks retrieve කරගත් පසු, අපේ EvidenceVerifier එක මඟින් 2023 Legacy Handbook එක සහ 2025 Senate Policies අතර තියෙන පරස්පරතා (උදාහරණයක් ලෙස විභාග අයදුම්පත් දින 7ක්ද වැඩ කරන දින 5ක්ද යන්න) ස්වයංක්‍රීයව හඳුනාගෙන 2025 නව නීතිය නිවැරදිව තෝරාගන්නවා."*
* **3:25 – 3:45:** *"අවසානයේදී Google Gemini LLM එක හරහා Form EC-01, Form ATT-02A, Form EXT-03C, Form MED-04-CLR සහ Form PRG-05-CASE සහිත Ordered Action Plan එකක්, නිල Policy Clauses සහ පිටු අංක සහිත Grounded Citations එක්ක ශිෂ්‍යයාට ලබා දෙනවා."*
