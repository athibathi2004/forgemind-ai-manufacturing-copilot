# ForgeMind AI — Architectural Thinking & Engineering Decisions

## 1. Understand the User

### Shop-Floor Persona: Precision Manufacturing Supervisor
* **Operational Context**: Kaveri Precision Components Pvt. Ltd. (Hosur Plant) is a high-volume precision turning and machining facility operating multi-axis CNC machines (TX-200 line) and strict quality control processes.
* **Working Environment**: Fast-paced shop-floor environment. Supervisors and operators move between turning bays (CNC-04 to CNC-09), inspection rooms (CMM, digital calipers, pin gauges), and quarantine stores (Cage Q1, Stores Bay 3).
* **Physical Constraints**: Operators wear shop-floor PPE, use mobile phones or wall-mounted touch tablets, and frequently encounter loud ambient noise, vibrations, and time-critical alarms.
* **Core Risk Profile**: An incorrect answer or hallucinated parameter (e.g., incorrect chuck torque, wrong hydraulic pressure, missing LOTO isolation step, or using superseded inspection rules) can cause catastrophic machine damage, defective part shipment, or severe personnel injury.

---

## 2. User Journey

### A. Morning Shift Machine Alarm (AL-104)
1. **Trigger**: Turning centre CNC-06 halts with alarm `AL-104`.
2. **Supervisor Action**: Supervisor opens ForgeMind AI on a smartphone connected to the plant Wi-Fi (`http://192.168.1.x:8000`).
3. **Query**: *"The machine is showing alarm AL-104. What does it mean and what should the operator do?"*
4. **Copilot Response**:
   - Status: `ANSWERED`
   - Directive: Way lube oil level is low. Top up with **ISO VG 68** oil to the MAX mark, then press RESET. Machine can continue after reset.
   - Evidence Citation: `MNT-MAN-CNC-02 Rev 02, Page 2, Section 6`.
5. **Outcome**: Operator immediately fetches ISO VG 68 oil instead of guessing, tops up the reservoir, resets the alarm, and resumes production within 4 minutes.

### B. Shift Handover LOTO Lock Removal Query
1. **Trigger**: An employee from Shift A left their red lock on machine CNC-05. Shift B needs to service the chuck.
2. **Supervisor Action**: Supervisor asks: *"Can a Shift Supervisor independently remove an absent employee's lock?"*
3. **Copilot Response**:
   - Status: `ANSWERED`
   - Directive: **No.** Under `SAF-SOP-003 Rev 05 Section 6`, a Shift Supervisor cannot independently remove an absent employee's lock. The supervisor must first make every reasonable effort to contact the owner by phone. If unreachable, **both the Production Manager and the Safety Officer must authorize removal**, confirm zero energy by physical inspection, record on form `SF-12`, and notify the owner before their next shift.
   - Safety Co-retrieval: Links both the general rule (Section 5) and the safety-critical exception (Section 6).
4. **Outcome**: Avoids unauthorized lock cutting, preventing severe electrical or hydraulic energization hazards.

### C. Incoming Inspection Lot Sampling (Revision Awareness)
1. **Trigger**: Stores receives a lot of 200 machined parts from a sub-contractor.
2. **Inspector Action**: Inspector queries: *"What is the sampling requirement for a lot of 200 pieces?"*
3. **Copilot Response**:
   - Status: `ANSWERED`
   - Directive: Under current `QC-SOP-014 Rev 04`, for lot size 91–500 pieces, inspect **13 samples**. Accept if minor defects are 0; reject if minor defects are 1 or more. Critical or major defects reject the entire lot.
   - Revision Governance: Successfully ignores superseded Rev 03 (which had an obsolete rule of 5 pieces for all lots) unless explicitly queried for historical audit.
4. **Outcome**: 13 pieces are inspected, ensuring compliance with current QMS standards.

---

## 3. The AI Challenge

1. **Hallucination in High-Stakes Environments**: General-purpose LLMs tend to extrapolate plausible-sounding mechanical torque values, tolerances, and procedures. In manufacturing, extrapolating 80 Nm instead of 110 Nm or suggesting hydraulic pressure without zero-energy verification can be fatal.
2. **Revision Governance & Document Clashes**: Plants frequently maintain older archived documents. If a vector search retrieves chunks from both Rev 03 and Rev 04, standard RAG combines them into an inconsistent answer (e.g. "inspect 5 samples or 13 samples").
3. **Safety Rule and Exception Separation**: Procedural manuals often state a general prohibition in one section and an emergency exception in another (e.g., LOTO Section 5: "Nobody may remove another person's lock" vs Section 6: "Removal of a Lock When Owner Is Absent"). Standard chunking often separates them, leading to incomplete advice.
4. **Exact Identifier Fragility**: Vector embeddings alone struggle with exact alarm alphanumeric strings (`AL-104` vs `AL-106`). If `AL-106` is queried, pure semantic search retrieves `AL-104` or `AL-101` as the nearest vector and might invent a meaning.
5. **Local Air-Gapped Manufacturing Deployment**: Proprietary plant drawings, machine manuals, and quality defect rates cannot be transmitted to external public clouds without risking IP leakage or violating plant compliance.

---

## 4. Improvement Added and Rationale: Evidence-First, Revision-Aware Grounding

### Technical Rationale
To solve the governance, safety, and hallucination challenges identified above, ForgeMind AI implements:

1. **Pre-Ranking Revision Filter**:
   - The Document Registry inspects revision numbers and explicit `Supersedes Rev XX` clauses.
   - Superseded chunks (e.g., `QC-SOP-014 Rev 03`) are partitioned out of normal operational queries.
   - If a historical query explicitly requests Rev 03, the retriever admits those chunks with an explicit `[SUPERSEDED]` provenance tag.

2. **Deterministic Alphanumeric Identifier Safeguard**:
   - Regular expressions extract machine codes (`CNC-02`), alarm codes (`AL-104`, `AL-106`), and part numbers (`BH-220`).
   - If an identifier like `AL-106` is queried and does not exist anywhere in the corpus, ForgeMind short-circuits to `NOT_DOCUMENTED` rather than letting semantic similarity hallucinate an answer.

3. **Rule-and-Exception Co-Retrieval**:
   - Chunks covering general rules (`SAF-SOP-003 Section 5`) are explicitly bound to their corresponding safety exceptions (`SAF-SOP-003 Section 6`).
   - Cross-document references (e.g. `MNT-MAN-CNC-02 Section 7` referencing `SAF-SOP-003`) automatically expand the retrieval bundle.

4. **Programmatic Post-Generation Evidence Verification**:
   - Rather than trusting generated text, every cited chunk ID and quoted passage is verified against source text using normalized substring matching.
   - Ground-truth excerpts are rendered directly from the validated chunk, giving supervisors traceable evidence.

5. **Local Offline Stack**:
   - Dense embeddings: `BAAI/bge-small-en-v1.5` run completely locally on CPU/GPU via `sentence-transformers`.
   - BM25 Okapi for exact keyword and identifier matching.
   - Local LLM: Ollama (`llama3.2:1b` / `qwen3:8b`) with optional enterprise Groq acceleration.

---

## 5. Final Reflection

### What Worked Well
1. **Hybrid Retrieval with Exact Identifier Boosting**: Combining BM25Okapi for exact alphanumeric codes (`AL-104`, `BH-220`, `CNC-04`, `SF-12`) with local dense semantic embeddings (`BAAI/bge-small-en-v1.5`) via Reciprocal Rank Fusion ($k=60$) eliminated semantic drift while maintaining natural language flexibility.
2. **Deterministic Revision Governance**: Parsing document revision history and `Supersedes Rev XX` clauses at ingestion prevents superseded procedures (`QC-SOP-014 Rev 03`) from contaminating current operational responses, while preserving historical audit access with clear `[SUPERSEDED]` provenance badges.
3. **Safety Rule-and-Exception Co-Retrieval**: Binding general safety prohibitions (`SAF-SOP-003 Section 5`) with emergency owner-absent exceptions (`Section 6`) ensured that critical safety protocols (dual authorization from Production Manager and Safety Officer, SF-12 form logging, physical inspection) are never omitted.
4. **Deterministic Dimensional Extraction**: Bypassing free-form LLM generation for critical dimensional tables guaranteed 100% fidelity on asymmetric tolerances (BH-220 bore: `52.000 +0.030 / -0.000 mm`, measured via CMM).

### What Was Harder Than Expected
1. **Small Language Model (SLM) Priors**: Local 1B–3B models (`llama3.2:1b`) have strong semantic priors that aggressively rewrite asymmetric engineering tolerances (e.g., converting `+0.030 / -0.000` into `+/- 0.030`) and summarize away multi-step emergency safety exceptions. Implementing deterministic extraction safeguards for safety-critical tables proved essential.
2. **Context Window & JSON Generation Limits**: Small models running on CPU frequently truncated JSON responses when `num_predict` was set too low (256 tokens). Tuning token predictions to 512 and implementing JSON repair logic resolved malformed response payloads.
3. **Cross-Page Document Continuity**: Real plant SOPs split crucial procedures across page breaks (e.g., LOTO Section 6 begins at the bottom of Page 1 and concludes on Page 2). Implementing section-aware chunking with multi-page spans (`Pages 1–2`) was critical for complete evidence citation.

### What We Would Build Next With More Time
1. **OCR Pipeline for Scanned Legacy Documentation**: Integrate Tesseract or PaddleOCR to extract text and dimensional drawings from legacy scanned TIFF/PDF files without native text layers.
2. **Live QMS & ERP Webhook Synchronization**: Connect ForgeMind AI to plant Document Management Systems (SAP DMS, MasterControl, Veeva Vault) to automatically trigger re-indexing upon Engineering Change Order (ECO) approvals.
3. **Multimodal Engineering Drawing Analysis**: Incorporate vision-language models to directly inspect 2D technical drawings, GD&T symbols (geometric dimensioning and tolerancing), and wiring schematics.
4. **Shop-Floor Hands-Free Voice Assistant**: Deploy a noise-robust offline speech-to-text engine (such as Whisper Small) tailored for shop-floor operators wearing heavy PPE and work gloves in high-noise environments.

