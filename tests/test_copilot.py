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
