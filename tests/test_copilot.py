import pytest
from forgemind.ingestion.registry import DocumentRegistry
from forgemind.ingestion.chunker import DocumentChunker, extract_identifiers, normalize_identifier
from forgemind.retrieval.hybrid_retriever import HybridRetriever
from forgemind.pipeline.rag_engine import RAGEngine
from forgemind.core.models import AskRequest, ResponseStatus, DocumentStatus

@pytest.fixture(scope="module")
def registry():
    return DocumentRegistry()

@pytest.fixture(scope="module")
def chunker(registry):
    return DocumentChunker(registry)

@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever()

@pytest.fixture(scope="module")
def engine(retriever):
    return RAGEngine(retriever=retriever)

def test_registry_governance(registry):
    docs = { (d.document_number, d.revision): d for d in registry.get_documents() }
    assert ("MNT-MAN-CNC-02", "02") in docs
    assert docs[("MNT-MAN-CNC-02", "02")].document_status == DocumentStatus.CURRENT

    assert ("SAF-SOP-003", "05") in docs
    assert docs[("SAF-SOP-003", "05")].document_status == DocumentStatus.CURRENT

    assert ("QC-SOP-014", "04") in docs
    assert docs[("QC-SOP-014", "04")].document_status == DocumentStatus.CURRENT
    assert docs[("QC-SOP-014", "04")].superseded_revision == "03"

    assert ("QC-SOP-014", "03") in docs
    assert docs[("QC-SOP-014", "03")].document_status == DocumentStatus.SUPERSEDED

def test_identifier_extraction():
    text = "Machine CNC 04 showed alarm AL 104, check BH 220 using form SF 12 in Cage Q1"
    ids = extract_identifiers(text)
    assert "AL-104" in ids
    assert "CNC-04" in ids
    assert "BH-220" in ids
    assert "SF-12" in ids
    assert "CAGE Q1" in ids

def test_retriever_undocumented_identifier(retriever):
    res = retriever.retrieve("What is the meaning of alarm AL-106?")
    assert "AL-106" in res.unrecognized_identifiers

def test_retriever_superseded_filtering(retriever):
    # Operational query should not return Rev 03
    res_normal = retriever.retrieve("What is the incoming inspection sampling requirement?", allow_superseded=False)
    for c in res_normal.chunks:
        assert c.document_status != DocumentStatus.SUPERSEDED

    # Historical query should retrieve Rev 03
    res_hist = retriever.retrieve("What did the superseded Revision 03 specify about sampling?", allow_superseded=True)
    rev03_found = any(c.revision == "03" for c in res_hist.chunks)
    assert rev03_found, "Expected Revision 03 chunk in historical query."

def test_loto_rule_and_exception_coretrieval(retriever):
    res = retriever.retrieve("Who is allowed to remove a lockout lock from a machine?")
    cids = [c.chunk_id for c in res.chunks]
    assert "SAF-SOP-003_R05_S05_S06_P01_P02" in cids

# ==============================================================================
# FASTAPI HTTP & INTEGRATION TESTS
# ==============================================================================
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ForgeMind AI" in data["service"]

def test_api_documents(client):
    response = client.get("/api/documents")
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) >= 4
    doc_numbers = [d["document_number"] for d in docs]
    assert "MNT-MAN-CNC-02" in doc_numbers
    assert "SAF-SOP-003" in doc_numbers
    assert "QC-SOP-014" in doc_numbers

    # Verify Rev 04 is current and Rev 03 is superseded
    qc_docs = {d["revision"]: d for d in docs if d["document_number"] == "QC-SOP-014"}
    assert qc_docs["04"]["document_status"] == "current"
    assert qc_docs["03"]["document_status"] == "superseded"

def test_api_system_status(client):
    response = client.get("/api/system/status")
    assert response.status_code == 200
    status = response.json()
    assert status["status"] == "ready"
    assert status["documents_count"] >= 4
    assert status["chunks_count"] >= 20
    assert "bge-small-en" in status["embedding_model"].lower()

def test_api_sample_questions(client):
    response = client.get("/api/sample-questions")
    assert response.status_code == 200
    data = response.json()
    assert len(data["biztel_questions"]) == 5
    assert any("coolant concentration" in q.lower() for q in data["biztel_questions"])
    assert any("AL-104" in q for q in data["biztel_questions"])

def test_api_frontend_static_serving(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ForgeMind AI" in response.text

def test_api_ask_undocumented_alarm_safeguard(client):
    # Tests that undocumented identifier AL-106 is safely handled deterministically
    payload = {
        "question": "What is the procedure for alarm AL-106 on the TX-200?",
        "allow_superseded": False
    }
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_documented"
    assert "AL-106" in data["answer"]
    assert len(data["citations"]) == 0
    assert "AL-106" in data["missing_info"][0]

def test_adversarial_machine_scope_cnc02(retriever):
    # Verify MNT-MAN-CNC-02 chunk 1 clearly excludes CNC-02
    res = retriever.retrieve("Does the manual apply to machine CNC-02?")
    top_chunk = res.chunks[0]
    assert "CNC-04 to CNC-09" in top_chunk.content
    assert "Does NOT cover CNC-02" in top_chunk.content

def test_adversarial_bh220_asymmetric_tolerance(retriever):
    # Verify BH-220 inspection chunk retains exact asymmetric tolerances
    res = retriever.retrieve("What is the bore diameter tolerance for BH-220?")
    cids = [c.chunk_id for c in res.chunks]
    assert "QC-SOP-014_R04_S06_P02" in cids
    chunk = next(c for c in res.chunks if c.chunk_id == "QC-SOP-014_R04_S06_P02")
    assert "52.000" in chunk.content
    assert "+0.030 / -0.000" in chunk.content
    assert "CMM" in chunk.content
    assert "Critical" in chunk.content

def test_adversarial_hydraulic_hose_cross_ref(retriever):
    # Verify hose replacement co-retrieves SAF-SOP-003 LOTO
    res = retriever.retrieve("What is the procedure for replacing a hydraulic hose on TX-200?")
    cids = [c.chunk_id for c in res.chunks]
    assert "MNT-MAN-CNC-02_R02_S07_P02" in cids
    # SAF-SOP-003 Lockout/Tagout should be co-retrieved
    assert any("SAF-SOP-003" in cid for cid in cids)

# ==============================================================================
# SUBMISSION REGRESSION TESTS FOR FIX 1, FIX 2, AND FIX 3
# ==============================================================================

def test_regression_fix1_loto_completeness(engine):
    req = AskRequest(question="Who is allowed to remove a lockout lock?", allow_superseded=False)
    resp = engine.ask(req)
    assert resp.status == ResponseStatus.ANSWERED
    ans = resp.answer.lower()
    
    # Check general rule & owner-absent exception conditions
    assert "only their own" in ans or "removes only their own" in ans
    assert "shift supervisor" in ans
    assert "phone" in ans or "contact" in ans
    assert "production manager" in ans
    assert "safety officer" in ans
    assert "physically inspected" in ans or "physical" in ans
    assert "sf-12" in ans
    assert "next shift" in ans
    
    # Check provenance
    assert len(resp.citations) >= 1
    cit = resp.citations[0]
    assert cit.document_number == "SAF-SOP-003"
    assert cit.revision == "05"
    assert cit.page_number == 1
    assert cit.page_span in ("1–2", "1-2")
    assert cit.verified is True

def test_regression_fix1_loto_supervisor_independent(engine):
    req = AskRequest(question="Can a Shift Supervisor independently remove an absent employee's lock?", allow_superseded=False)
    resp = engine.ask(req)
    assert resp.status == ResponseStatus.ANSWERED
    ans = resp.answer.lower()
    assert "no" in ans
    assert "cannot independently remove" in ans or "cannot" in ans
    assert "production manager" in ans
    assert "safety officer" in ans
    assert "sf-12" in ans

def test_regression_fix2_spindle_runout_completeness(engine):
    req = AskRequest(question="How often should spindle runout be checked, and who does it?", allow_superseded=False)
    resp = engine.ask(req)
    assert resp.status == ResponseStatus.ANSWERED
    ans = resp.answer.lower()
    
    # Verify frequency and role
    assert "monthly" in ans
    assert "maintenance technician" in ans
    
    # Check provenance
    assert len(resp.citations) >= 1
    cit = resp.citations[0]
    assert cit.document_number == "MNT-MAN-CNC-02"
    assert cit.revision == "02"
    assert cit.page_number == 1
    assert cit.verified is True

def test_regression_fix3_bh220_exact_asymmetric_tolerance(engine):
    req = AskRequest(question="What is the BH-220 bore diameter tolerance?", allow_superseded=False)
    resp = engine.ask(req)
    assert resp.status == ResponseStatus.ANSWERED
    ans = resp.answer
    
    # Verify exact asymmetric tolerance and that it is NEVER written as +/- 0.030
    assert "52.000" in ans
    assert "+0.030 / -0.000" in ans
    assert "+/- 0.030" not in ans
    assert "±0.030" not in ans
    assert "CMM" in ans
    assert "Critical" in ans
    
    # Check provenance
    assert len(resp.citations) >= 1
    cit = resp.citations[0]
    assert cit.document_number == "QC-SOP-014"
    assert cit.revision == "04"
    assert cit.page_number == 2
    assert cit.verified is True

def test_regression_fix3_bh220_outer_diameter_tolerance(engine):
    req = AskRequest(question="What is the BH-220 outer diameter tolerance?", allow_superseded=False)
    resp = engine.ask(req)
    assert resp.status == ResponseStatus.ANSWERED
    ans = resp.answer
    
    # Verify exact nominal and tolerance
    assert "90.000" in ans
    assert "+/- 0.050" in ans
    assert "Digital caliper" in ans or "digital caliper" in ans.lower()
    assert "Major" in ans or "major" in ans.lower()
    
    # Check provenance
    assert len(resp.citations) >= 1
    cit = resp.citations[0]
    assert cit.document_number == "QC-SOP-014"
    assert cit.revision == "04"
    assert cit.page_number == 2
    assert cit.verified is True



