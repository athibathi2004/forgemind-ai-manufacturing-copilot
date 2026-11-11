import socket
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from forgemind.core.models import (
    AskRequest,
    AskResponse,
    DocumentMetadata,
    SystemStatus,
    ResponseStatus,
)
from forgemind.config import settings
from forgemind.ingestion.registry import DocumentRegistry
from forgemind.pipeline.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances initialized at app startup
_registry: Optional[DocumentRegistry] = None
_rag_engine: Optional[RAGEngine] = None

def get_registry() -> DocumentRegistry:
    global _registry
    if _registry is None:
        _registry = DocumentRegistry()
    return _registry

def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine

def get_host_lan_ip() -> str:
    """Detects the machine's primary local area network (LAN) IP address without hanging on outbound sockets."""
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip.strip():
            return ip
    except Exception:
        pass
    return "127.0.0.1"

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ForgeMind AI Copilot",
        "plant": settings.plant_name
    }

@router.get("/api/documents", response_model=List[DocumentMetadata])
def get_documents(registry: DocumentRegistry = Depends(get_registry)):
    return registry.get_documents()

@router.get("/api/system/status", response_model=SystemStatus)
def get_system_status(
    registry: DocumentRegistry = Depends(get_registry),
    engine: RAGEngine = Depends(get_rag_engine)
):
    docs = registry.get_documents()
    lan_ip = get_host_lan_ip()
    
    # Active provider name
    try:
        from forgemind.llm.factory import LLMFactory
        service = LLMFactory.get_service()
        active_provider = f"{service.get_provider_name()} ({service.get_model_name()})"
    except Exception:
        active_provider = "unconfigured"

    return SystemStatus(
        app_name=settings.app_name,
        plant_name=settings.plant_name,
        status="ready",
        documents_count=len(docs),
        chunks_count=len(engine.retriever.chunks),
        active_provider=active_provider,
        embedding_model=settings.embedding_model,
        lan_ip=lan_ip,
        documents=docs
    )

@router.get("/api/sample-questions")
def get_sample_questions():
    return {
        "biztel_questions": [
            "What is the recommended coolant concentration for the TX-200 CNC turning centre?",
            "The machine is showing alarm AL-104. What does it mean and what should the operator do?",
            "Who is allowed to remove a lockout lock from a machine?",
            "What is the outer diameter tolerance for the BH-220 bearing housing?",
            "How often should spindle runout be checked, and who does it?"
        ],
        "domain_verification_questions": [
            "What is the sampling requirement for a lot of 200 pieces?",
            "What happens if the incoming material does not have an MTC?",
            "What is the procedure for replacing a hydraulic hose?",
            "What is the pressure rating of the replacement hydraulic hose?",
            "What is the meaning of alarm AL-106?",
            "What did the superseded Revision 03 specify about sampling?",
            "Can a Shift Supervisor independently remove an absent employee's lock?",
            "Does the TX-200 manual cover CNC-02?"
        ]
    }

@router.post("/api/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    engine: RAGEngine = Depends(get_rag_engine)
):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        response = engine.ask(request)
        return response
    except Exception as e:
        logger.exception(f"Error handling /api/ask for query: '{request.question}'")
        raise HTTPException(status_code=500, detail=str(e))
