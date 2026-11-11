from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ResponseStatus(str, Enum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    NOT_DOCUMENTED = "not_documented"
    CONFLICT = "conflict"
    NEEDS_CLARIFICATION = "needs_clarification"
    ERROR = "error"

class DocumentStatus(str, Enum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    REFERENCE = "reference"

class DocumentMetadata(BaseModel):
    document_number: str
    revision: str
    effective_date: str
    superseded_revision: Optional[str] = None
    filename: str
    page_count: int
    document_status: DocumentStatus
    title: str
    extraction_status: str = "indexed"

class Chunk(BaseModel):
    chunk_id: str
    document_number: str
    revision: str
    effective_date: str
    document_status: DocumentStatus
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    page_number: int
    page_span: List[int] = Field(default_factory=list)
    content: str
    source_filename: str
    is_table: bool = False
    identifiers: List[str] = Field(default_factory=list)

class Citation(BaseModel):
    chunk_id: str
    quote: str
    document_number: Optional[str] = None
    revision: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    verified: bool = False
    source_passage: Optional[str] = None

class LLMOutputSchema(BaseModel):
    status: ResponseStatus
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)

class AskRequest(BaseModel):
    question: str
    allow_superseded: bool = False
    provider: Optional[str] = None

class AskResponse(BaseModel):
    status: ResponseStatus
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    query_metadata: Dict[str, Any] = Field(default_factory=dict)

class SystemStatus(BaseModel):
    app_name: str
    plant_name: str
    status: str
    documents_count: int
    chunks_count: int
    active_provider: str
    embedding_model: str
    lan_ip: str
    documents: List[DocumentMetadata]

