# ForgeMind AI — Manufacturing Intelligence Copilot
### Plant-Floor Grounded Knowledge Retrieval & Revision Governance System
**Deployment**: Kaveri Precision Components Pvt. Ltd. (Hosur Plant)  
**Target Machines & Areas**: TX-200 CNC Turning Centres (CNC-04 to CNC-09), Incoming Inspection (Cage Q1, Stores Bay 3), LOTO Safety Operations.

---

## 1. Project Overview

**ForgeMind AI** is an enterprise-grade manufacturing intelligence assistant designed for shop-floor supervisors, machine operators, and quality inspectors. Built for high-stakes precision manufacturing environments, it provides immediate, evidence-grounded, and revision-aware operational answers from authorized plant documentation (PDF standard operating procedures, maintenance manuals, and engineering drawings).

### The Manufacturing Reality & Architectural Guardrails
In a precision manufacturing facility, hallucinations are safety risks:
* Recommending the wrong chuck mounting bolt torque can fracture machine fixtures.
* Omitting the owner-absent exception during Lockout/Tagout can result in unauthorized lock removal or electrical hazards.
* Mixing superseded sampling plans (e.g. QC-SOP-014 Rev 03 with Rev 04) leads to defective lot acceptance and customer escalations.
* Confusing machine numbers (e.g. assuming the manual for CNC-04 through CNC-09 covers CNC-02 because the document number contains "CNC-02") causes incorrect maintenance procedures.

ForgeMind AI eliminates these risks through **deterministic identifier safeguards**, **pre-ranking revision governance**, **rule-and-exception co-retrieval**, and **programmatic post-generation evidence verification**.

---

## 2. Technical Architecture

```
                    +------------------------------------+
                    |  Shop-Floor Client (Mobile / Web)  |
                    +-----------------+------------------+
                                      | HTTP REST / JSON
                                      v
                    +------------------------------------+
                    |        FastAPI Host (0.0.0.0)      |
                    +-----------------+------------------+
                                      |
                +---------------------+---------------------+
                |                                           |
                v                                           v
     +---------------------+                     +---------------------+
     |   Document Registry |                     |  Hybrid Retriever   |
     | - Revision Governance|                     | - Dense: BGE-Small  |
     | - Rev 04 (Current)  |                     | - Sparse: BM25Okapi |
     | - Rev 03 (Superseded)|                    | - RRF Fusion (k=60) |
     +---------------------+                     | - Identifier Boost  |
                                                 +----------+----------+
                                                            |
                                                            v
                                                 +---------------------+
                                                 |  Reference Expansion|
                                                 | - LOTO Sec 5 & 6    |
                                                 | - Hose -> SAF-SOP   |
                                                 +----------+----------+
                                                            |
                                                            v
+------------------------------------+           +---------------------+
|        LLM Provider Layer          |<----------| Compact Structured  |
| - GroqService (Cloud Acceleration) |           | Prompting Context   |
| - OllamaService (Local Llama 3.2)  |           +---------------------+
+-----------------+------------------+
                  | Validated JSON Output
                  v
+------------------------------------+
|     Response Validator Engine      |
| - Citation Chunk Verification      |
| - Numerical / Unit Consistency     |
| - Ground-Truth Excerpt Extraction  |
| - 6 State Machine Resolution       |
+-----------------+------------------+
                  |
                  v
       Verified Grounded Directive
```

---

## 3. Tech Stack

* **Backend Framework**: Python 3.11, FastAPI, Uvicorn, Pydantic v2.
* **Document Processing**: PyMuPDF (`fitz`), pdfplumber (table headers, tolerances, and cross-page section binding).
* **Embeddings**: `BAAI/bge-small-en-v1.5` running 100% locally on CPU/GPU via `sentence-transformers` (vector dimension 384, offline cache).
* **Keyword Retrieval**: `rank-bm25` (BM25Okapi) with identifier-preserving tokenization.
* **Hybrid Fusion**: Reciprocal Rank Fusion ($k=60$) with exact identifier boosting ($+0.05$).
* **LLM Engine**: Provider-independent abstraction:
  - **Local Default**: Ollama (`llama3.2:1b` / `qwen3:8b`) with JSON mode and deterministic generation (temperature = 0).
  - **Cloud Alternative**: Groq API (`llama-3.3-70b-versatile`) when `GROQ_API_KEY` is present.
* **Frontend**: Mobile-first responsive HTML5, CSS3 (Enterprise dark navy palette with glassmorphism), and Vanilla JavaScript (zero external CDN or runtime framework overhead).

---

## 4. Key Enterprise Features

1. **Pre-Ranking Revision Governance**:
   - Explicitly catalogs document revisions and effective dates.
   - Automatically marks `QC-SOP-014 Rev 03` as superseded by `Rev 04` based on document analysis and explicit QMS revision notes.
   - Partitions superseded content out of standard operational searches.
   - Permits historical queries (e.g. *"What did superseded Rev 03 specify?"*) with a prominent `[SUPERSEDED]` warning badge.
2. **Deterministic Alphanumeric Identifier Safeguards**:
   - Parses machine IDs (`CNC-04`, `CNC-02`), alarm codes (`AL-104`, `AL-106`), part codes (`BH-220`), and forms (`SF-12`, `QF-07A`).
   - If an alarm code (such as `AL-106`) is queried that is absent from the corpus, the system returns `NOT_DOCUMENTED` without attempting semantic extrapolation.
3. **Safety Rule-and-Exception Co-Retrieval**:
   - Binds `SAF-SOP-003 Section 5` (General Lock Removal) with `Section 6` (Owner-Absent Exception) into a single retrievable unit, preventing safety breaches.
4. **Cross-Document Reference Expansion**:
   - When a procedure references an external standard (e.g. TX-200 hydraulic hose replacement referencing `SAF-SOP-003`), the system bundles the referenced safety isolation steps.
5. **Six Distinct Response States**:
   - `ANSWERED`: Supported by verified citations.
   - `PARTIAL`: Incomplete coverage (e.g. replacement hose procedure is documented, but the numeric pressure rating of the replacement hose is not).
   - `NOT_DOCUMENTED`: The queried procedure or alarm does not exist in the documents.
   - `CONFLICT`: Governance mismatch or conflicting instructions across versions.
   - `NEEDS_CLARIFICATION`: Query ambiguity.
   - `ERROR`: System or provider fault.
6. **Wi-Fi & LAN Host Detection**:
   - Automatically discovers the host workstation's local IP address and displays a connection pill (`http://192.168.x.x:8000`) for seamless mobile/tablet access across the shop floor.

---

## 5. Project Structure

```
forgemind-ai/
├── data/
│   ├── documents/                  # Authorized QMS source PDFs
│   │   ├── MNT-MAN-CNC-02_TX-200_Preventive_Maintenance.pdf
│   │   ├── QC-SOP-014_Rev03_Incoming_Material_Inspection.pdf
│   │   ├── QC-SOP-014_Rev04_Incoming_Material_Inspection.pdf
│   │   ├── SAF-SOP-003_Lockout_Tagout_Procedure.pdf
│   │   └── Sample_Questions.pdf
│   └── eval_results.json           # Persisted benchmark evaluation results
├── forgemind/
│   ├── __init__.py
│   ├── config.py                   # Environment and path settings
│   ├── core/
│   │   ├── __init__.py
│   │   └── models.py               # Pydantic schemas (Metadata, Chunk, Citation, AskResponse)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── registry.py             # QMS Document Registry & Revision Governance
│   │   └── chunker.py              # Section-aware & table-aware chunking with identifier indexing
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py             # Local BAAI/bge-small-en-v1.5 embedding service
│   │   └── hybrid_retriever.py     # BM25 + Dense + RRF + Exact Identifier Boosting
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract LLM interface
│   │   ├── groq_service.py         # Groq API provider client
│   │   ├── ollama_service.py       # Ollama local HTTP provider client
│   │   └── factory.py              # Dynamic provider selection factory
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── validator.py            # Post-generation citation and numeric validation
│   │   └── rag_engine.py           # RAG coordinator & structured output synthesis
│   └── api/
│       ├── __init__.py
│       └── routes.py               # FastAPI endpoints (/health, /api/ask, /api/documents, etc.)
├── static/
│   ├── index.html                  # Responsive enterprise UI
│   ├── style.css                   # Enterprise navy styling & typography
│   └── app.js                      # UI logic and reactive state management
├── tests/
│   └── test_copilot.py             # Pytest test suite for governance and retrieval
├── .env.example                    # Template environment variables
├── .gitignore                      # Git exclusion rules
├── eval.py                         # 13-question benchmark evaluation runner
├── main.py                         # FastAPI application entry point with LAN banner
├── README.md                       # Comprehensive project documentation
└── THINKING.md                     # Engineering decisions and user journey analysis
```

---

## 6. Installation & Environment Configuration

### Prerequisites
* Windows, macOS, or Linux.
* Python 3.11 (tested on CPython 3.11.17).
* [Ollama](https://ollama.com/) (for local offline LLM inference) with model `llama3.2:1b` installed:
  ```bash
  ollama pull llama3.2:1b
  ```

### Step 1: Clone and Create Virtual Environment
```bash
git clone <repo-url> forgemind-ai
cd forgemind-ai

# Create virtual environment with Python 3.11
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
# Or using uv:
uv pip install pymupdf pdfplumber pydantic fastapi uvicorn sentence-transformers rank-bm25 httpx python-dotenv groq pytest
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` as desired:
```ini
LLM_PROVIDER=auto
GROQ_API_KEY=your_groq_api_key_here    # Optional: Leave empty to run 100% locally via Ollama
GROQ_MODEL=llama-3.3-70b-versatile
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:1b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
HOST=0.0.0.0
PORT=8000
```

---

## 7. Running the Application

Start the server using `main.py`:
```bash
python main.py
```

Console output will display the active host addresses:
```
======================================================================
               FORGEMIND AI — MANUFACTURING COPILOT                   
            Kaveri Precision Components Pvt. Ltd. (Hosur)             
======================================================================
  * Local URL:       http://localhost:8000
  * LAN/Wi-Fi URL:   http://192.168.1.15:8000
  * Host binding:    0.0.0.0:8000
  * LLM Provider:    auto (ollama: llama3.2:1b)
  * Embeddings:      BAAI/bge-small-en-v1.5 (Local BGE)
======================================================================
  Connect any phone, tablet, or terminal on the same Wi-Fi to:
  --> http://192.168.1.15:8000
======================================================================
```

---

## 8. Mobile & LAN Deployment Instructions

To access ForgeMind AI from an Android phone, iPhone, iPad, or shop-floor tablet:
1. Ensure the mobile device is connected to the same local Wi-Fi router or hotspot as the host machine.
2. Note the LAN IP shown on the console banner or the application header pill (e.g. `http://192.168.1.15:8000`).
3. Open the mobile browser (Chrome, Safari, Firefox) and navigate to `http://<HOST-LAN-IP>:8000`.
4. **Windows Firewall Note**: If the page does not load, allow Python/Uvicorn through Windows Defender Firewall, or run PowerShell as Administrator:
   ```powershell
   New-NetFirewallRule -DisplayName "ForgeMind AI" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
   ```

---

## 9. API Documentation

### `GET /health`
Returns system operational health.
```json
{
  "status": "healthy",
  "service": "ForgeMind AI Copilot",
  "plant": "Kaveri Precision Components Pvt. Ltd. - Hosur Plant"
}
```

### `GET /api/documents`
Returns all registered plant QMS documents with derived revision governance status.
```json
[
  {
    "document_number": "MNT-MAN-CNC-02",
    "revision": "02",
    "effective_date": "20-Jan-2026",
    "superseded_revision": null,
    "filename": "MNT-MAN-CNC-02_TX-200_Preventive_Maintenance.pdf",
    "page_count": 2,
    "document_status": "current",
    "title": "Preventive Maintenance Manual - TX-200 CNC Turning Centre",
    "extraction_status": "indexed"
  },
  {
    "document_number": "QC-SOP-014",
    "revision": "04",
    "effective_date": "01-Jul-2026",
    "superseded_revision": "03",
    "filename": "QC-SOP-014_Rev04_Incoming_Material_Inspection.pdf",
    "page_count": 2,
    "document_status": "current",
    "title": "Incoming Material Inspection - Standard Operating Procedure",
    "extraction_status": "indexed"
  },
  {
    "document_number": "QC-SOP-014",
    "revision": "03",
    "effective_date": "15-Mar-2023",
    "superseded_revision": null,
    "filename": "QC-SOP-014_Rev03_Incoming_Material_Inspection.pdf",
    "page_count": 1,
    "document_status": "superseded",
    "title": "Incoming Material Inspection - Standard Operating Procedure",
    "extraction_status": "indexed"
  }
]
```

### `POST /api/ask`
Executes hybrid retrieval, structured LLM generation, citation verification, and status synthesis.
* **Request Body**:
  ```json
  {
    "question": "What is the outer diameter tolerance for the BH-220 bearing housing?",
    "allow_superseded": false
  }
  ```
* **Response Body**:
  ```json
  {
    "status": "answered",
    "answer": "The outer diameter tolerance for the BH-220 bearing housing is 90.000 +/- 0.050 mm, measured using a digital caliper as a Major characteristic.",
    "citations": [
      {
        "chunk_id": "QC-SOP-014_R04_S06_P02",
        "quote": "Outer diameter: Nominal = 90.000 mm | Tolerance = +/- 0.050 (or 90.000 +/- 0.050 mm) | Method = Digital caliper | Class = Major",
        "document_number": "QC-SOP-014",
        "revision": "04",
        "page_number": 2,
        "section_title": "Inspection Dimensions - BH-220 Bearing Housing",
        "verified": true,
        "source_passage": "Section 6: Inspection Dimensions - BH-220 Bearing Housing (Machined, from sub-contractor)..."
      }
    ],
    "missing_info": [],
    "conflicts": [],
    "query_metadata": {
      "has_verified_citation": true,
      "latency_ms": 354.2,
      "provider": "ollama",
      "model": "llama3.2:1b",
      "query_identifiers": ["BH-220"]
    }
  }
  ```

---

## 10. Automated Testing & Evaluation

### Run Unit Tests
```bash
pytest -o pythonpath=. tests/test_copilot.py
```

### Run Full 13-Question Benchmark
```bash
python eval.py
```

---

## 11. Benchmark Questions & Actual Copilot Responses

### BiztelAI Sample Questions (1–5)
1. **Coolant Concentration**:
   - **Answer**: The recommended coolant concentration for the TX-200 CNC turning centre is **6 - 8 %**, checked using a refractometer reading multiplied by a 1.0 factor.
   - **Provenance**: `MNT-MAN-CNC-02 Rev 02, Page 1, Section 3`.
2. **Alarm AL-104**:
   - **Answer**: Alarm AL-104 indicates **way lube oil level low**. The operator should top up with **ISO VG 68** oil to the MAX mark, then press **RESET**. The machine can continue after reset.
   - **Provenance**: `MNT-MAN-CNC-02 Rev 02, Page 2, Section 6`.
3. **Lockout Lock Removal**:
   - **Answer**: Under the general rule, each authorised person removes only their own personal lock; nobody may remove another person's lock. However, under the mandatory owner-absent exception, if the lock owner has left the plant, the Shift Supervisor must first make reasonable phone attempts. If unreachable, **both the Production Manager and Safety Officer must authorize removal** after physical inspection, documented on form `SF-12`, and the employee notified on their next shift.
   - **Provenance**: `SAF-SOP-003 Rev 05, Page 1–2, Sections 5 & 6`.
4. **BH-220 Outer Diameter Tolerance**:
   - **Answer**: The outer diameter nominal dimension is **90.000 mm** with a tolerance of **+/- 0.050 mm**, measured with a digital caliper (Major classification).
   - **Provenance**: `QC-SOP-014 Rev 04, Page 2, Section 6`.
5. **Spindle Runout Frequency**:
   - **Answer**: Spindle runout must be checked **Monthly** by a **Maintenance technician** (tolerance: 0.005 mm max. at chuck face, using dial indicator).
   - **Provenance**: `MNT-MAN-CNC-02 Rev 02, Page 1, Section 4`.

### Additional Domain Verification Questions (6–13)
6. **Lot Sampling (200 pieces)**:
   - **Answer**: For a lot of 200 pieces (lot size range 91–500 pieces), the sampling requirement is **13 samples**. The lot is accepted if minor defects are 0, and rejected if minor defects are 1 or more. Critical or major defects reject the entire lot.
   - **Provenance**: `QC-SOP-014 Rev 04, Page 1, Section 4`.
7. **Missing MTC**:
   - **Answer**: The supplier's Material Test Certificate (MTC) is mandatory. If missing, the received lot is placed on **HOLD** and must not be inspected or accepted until the MTC is received.
   - **Provenance**: `QC-SOP-014 Rev 04, Page 2, Section 5`.
8. **Hydraulic Hose Replacement**:
   - **Answer**: 1) Isolate machine per SAF-SOP-003 and release stored hydraulic pressure; 2) Confirm panel pressure gauge reads zero before loosening fittings; 3) Place drip tray and loosen fittings slowly; 4) Fit new hose of the same rating; 5) Remove locks per SAF-SOP-003, start hydraulic power unit and verify pressure returns to 3.5–4.0 MPa without leaks.
   - **Provenance**: `MNT-MAN-CNC-02 Rev 02, Page 2, Section 7` (expanded with `SAF-SOP-003`).
9. **Hydraulic Hose Pressure Rating**:
   - **Answer**: The procedure specifies fitting *"a new hose of the same rating"*, but the **specific numerical pressure rating of the replacement hose itself is not documented in the manual**. The normal system operating pressure is 3.5–4.0 MPa (35–40 bar).
   - **Status**: `PARTIAL` / `NOT_DOCUMENTED`.
10. **Alarm AL-106**:
    - **Answer**: Alarm code `AL-106` is **not documented in the available maintenance manual** (MNT-MAN-CNC-02 documents AL-101, AL-104, AL-210, AL-305, AL-412). The system strictly avoids inferring instructions from similar alarms.
    - **Status**: `NOT_DOCUMENTED`.
11. **Superseded Rev 03 Sampling**:
    - **Answer**: Superseded Revision 03 (15-Mar-2023) specified drawing **5 samples at random from every lot, irrespective of lot size**. Note: This procedure is superseded by Revision 04 and is shown for historical reference only.
    - **Provenance**: `QC-SOP-014 Rev 03, Page 1, Section 4`.
12. **Shift Supervisor Independent Lock Removal**:
    - **Answer**: **No.** A Shift Supervisor cannot independently remove an absent employee's lock. Dual authorization from both the **Production Manager AND the Safety Officer** is mandatory.
    - **Provenance**: `SAF-SOP-003 Rev 05, Page 1–2, Section 6`.
13. **TX-200 Scope for CNC-02**:
    - **Answer**: **No.** Section 1 explicitly specifies that the manual covers machines **CNC-04 to CNC-09** on the shaft machining line. Despite the document number containing "CNC-02", it does not cover machine CNC-02.
    - **Provenance**: `MNT-MAN-CNC-02 Rev 02, Page 1, Section 1`.

---

## 12. Known Limitations & Production Roadmap

1. **Non-Searchable Scanned Images**: The current ingestion engine relies on text layer extraction via PyMuPDF. If future legacy QMS documents are scanned bitmaps without OCR text layers, an integrated OCR pipeline (Tesseract or PaddleOCR) will be required.
2. **Local CPU Generation Speed**: Running local 8B models on CPU without AVX-512 takes 20–35s per query. In production, local deployment should utilize an entry-level GPU (e.g. RTX 3060 / 4060) or Groq cloud acceleration for sub-second response times.
3. **Live QMS Integration**: The prototype uses a static document directory. In enterprise deployment, ForgeMind can connect via webhooks to Document Management Systems (MasterControl, SAP DMS, or Veeva Vault) for real-time document change order (ECO) syncing.

---

## 13. Safety Disclaimer

> **REFERENCE ASSISTANT ONLY**: Verify operational instructions against the latest approved QMS documentation. This application does not authorize machine operation or maintenance. All shop-floor personnel must adhere to plant EHS guidelines and authorized work permits.
