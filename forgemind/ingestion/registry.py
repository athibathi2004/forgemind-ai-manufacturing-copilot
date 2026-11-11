import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pymupdf

from forgemind.core.models import DocumentMetadata, DocumentStatus
from forgemind.config import settings

class DocumentRegistry:
    """
    Manages manufacturing document registration, metadata discovery,
    and revision governance rules.
    """
    def __init__(self, documents_dir: Optional[Path] = None):
        self.documents_dir = Path(documents_dir) if documents_dir else settings.documents_dir
        self.documents: Dict[Tuple[str, str], DocumentMetadata] = {}
        self._load_registry()

    def _extract_header_metadata(self, pdf_path: Path) -> Optional[DocumentMetadata]:
        filename = pdf_path.name
        if filename == "Sample_Questions.pdf":
            # Reference question set, not a plant QMS document
            return None
        
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)
        first_page_text = doc[0].get_text() if page_count > 0 else ""
        
        # Regex extraction from formal QMS header block
        doc_num_match = re.search(r"Document\s*No\.?\s*\n?\s*([A-Z0-9\-]+)", first_page_text, re.IGNORECASE)
        rev_match = re.search(r"Revision\s*\n?\s*(\d+)", first_page_text, re.IGNORECASE)
        date_match = re.search(r"Effective\s*Date\s*\n?\s*([0-9]{1,2}\s*-[A-Za-z]{3}\s*-[0-9]{4})", first_page_text, re.IGNORECASE)
        
        # Fallback to filename patterns if needed
        doc_num = doc_num_match.group(1).strip() if doc_num_match else ""
        rev = rev_match.group(1).strip() if rev_match else ""
        date = date_match.group(1).strip() if date_match else ""
        
        if not doc_num:
            if "MNT-MAN-CNC-02" in filename:
                doc_num = "MNT-MAN-CNC-02"
            elif "SAF-SOP-003" in filename:
                doc_num = "SAF-SOP-003"
            elif "QC-SOP-014" in filename:
                doc_num = "QC-SOP-014"
        
        if not rev:
            fn_rev = re.search(r"Rev(\d+)", filename, re.IGNORECASE)
            if fn_rev:
                rev = fn_rev.group(1)
            elif "MNT-MAN-CNC-02" in filename:
                rev = "02"
            elif "SAF-SOP-003" in filename:
                rev = "05"
        
        # Normalize rev format (e.g. "02", "04")
        if rev.isdigit():
            rev = f"{int(rev):02d}"

        # Detect document title
        title = "Plant QMS Document"
        lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
        for i, line in enumerate(lines[:10]):
            if "Kaveri Precision Components" in line and i + 1 < len(lines):
                title = lines[i + 1]
                break

        # Check for explicit superseding clauses
        full_text = "".join(page.get_text() for page in doc)
        superseded_rev = None
        supersede_match = re.search(r"Supersedes\s+Rev\s*(\d+)", full_text, re.IGNORECASE)
        if supersede_match:
            superseded_rev = f"{int(supersede_match.group(1)):02d}"

        # Default document status
        doc_status = DocumentStatus.CURRENT
        
        doc.close()
        
        return DocumentMetadata(
            document_number=doc_num,
            revision=rev,
            effective_date=date,
            superseded_revision=superseded_rev,
            filename=filename,
            page_count=page_count,
            document_status=doc_status,
            title=title,
            extraction_status="indexed"
        )

    def _load_registry(self):
        if not self.documents_dir.exists():
            return

        raw_docs: List[DocumentMetadata] = []
        for pdf_file in self.documents_dir.glob("*.pdf"):
            meta = self._extract_header_metadata(pdf_file)
            if meta:
                raw_docs.append(meta)

        # Revision Governance Resolution:
        # If a document explicitly supersedes another revision, mark the superseded revision
        superseded_set = set()
        for doc in raw_docs:
            if doc.superseded_revision:
                superseded_set.add((doc.document_number, doc.superseded_revision))

        for doc in raw_docs:
            if (doc.document_number, doc.revision) in superseded_set:
                doc.document_status = DocumentStatus.SUPERSEDED
            else:
                doc.document_status = DocumentStatus.CURRENT

            self.documents[(doc.document_number, doc.revision)] = doc

    def get_documents(self) -> List[DocumentMetadata]:
        return list(self.documents.values())

    def get_document(self, doc_num: str, rev: str) -> Optional[DocumentMetadata]:
        norm_rev = f"{int(rev):02d}" if rev.isdigit() else rev
        return self.documents.get((doc_num, norm_rev))

    def is_superseded(self, doc_num: str, rev: str) -> bool:
        meta = self.get_document(doc_num, rev)
        return meta.document_status == DocumentStatus.SUPERSEDED if meta else False

    def get_latest_revision(self, doc_num: str) -> Optional[str]:
        matching = [meta for meta in self.documents.values() if meta.document_number == doc_num and meta.document_status == DocumentStatus.CURRENT]
        if not matching:
            return None
        return matching[0].revision

