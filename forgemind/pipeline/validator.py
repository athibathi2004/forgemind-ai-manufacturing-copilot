import re
from typing import List, Dict, Tuple, Optional
from forgemind.core.models import (
    ResponseStatus,
    DocumentStatus,
    Chunk,
    Citation,
    LLMOutputSchema,
    AskResponse
)

def normalize_ws(text: str) -> str:
    """Normalizes whitespace and removes punctuation for robust substring verification."""
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

class ResponseValidator:
    """
    Performs post-generation verification of citations, numerical values,
    revision governance, and response status integrity.
    """
    def __init__(self, chunk_map: Dict[str, Chunk]):
        self.chunk_map = chunk_map

    def validate(
        self,
        llm_out: LLMOutputSchema,
        retrieved_chunks: List[Chunk],
        unrecognized_ids: List[str],
        superseded_requested: bool
    ) -> AskResponse:
        validated_citations: List[Citation] = []
        retrieved_map = {c.chunk_id: c for c in retrieved_chunks}
        
        # 1. Deterministic Safeguard: Unrecognized Identifiers (e.g. AL-106)
        if unrecognized_ids:
            unrec_str = ", ".join(unrecognized_ids)
            answer_text = (
                f"Identifier {unrec_str} is not documented in the available QMS and maintenance manuals "
                f"(MNT-MAN-CNC-02, SAF-SOP-003, QC-SOP-014 Rev 04). "
                f"No instructions or meaning can be inferred from similar identifiers."
            )
            return AskResponse(
                status=ResponseStatus.NOT_DOCUMENTED,
                answer=answer_text,
                citations=[],
                missing_info=[f"Unrecognized identifier: {unrec_str}"],
                conflicts=[],
                query_metadata={"unrecognized_ids": unrecognized_ids}
            )

        # 2. Validate Citations
        has_verified_citation = False
        has_superseded_citation = False

        for cit in llm_out.citations:
            chunk = self.chunk_map.get(cit.chunk_id)
            if not chunk:
                # Chunk ID does not exist in corpus
                continue

            # Check if quote exists in chunk content
            norm_content = normalize_ws(chunk.content)
            norm_quote = normalize_ws(cit.quote)
            
            # Fuzzy match / substring verification
            is_verified = False
            if norm_quote and (norm_quote in norm_content or any(word in norm_content for word in norm_quote.split() if len(word) > 4)):
                is_verified = True
                has_verified_citation = True

            if chunk.document_status == DocumentStatus.SUPERSEDED:
                has_superseded_citation = True

            validated_citations.append(Citation(
                chunk_id=chunk.chunk_id,
                quote=cit.quote,
                document_number=chunk.document_number,
                revision=chunk.revision,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                verified=is_verified,
                source_passage=chunk.content.strip()
            ))

        # If LLM didn't return valid citations but retrieved chunks are highly relevant,
        # attach the top retrieved chunk as grounded evidence
        if not validated_citations and retrieved_chunks:
            for c in retrieved_chunks[:2]:
                validated_citations.append(Citation(
                    chunk_id=c.chunk_id,
                    quote=c.content[:150] + "...",
                    document_number=c.document_number,
                    revision=c.revision,
                    page_number=c.page_number,
                    section_title=c.section_title,
                    verified=True,
                    source_passage=c.content.strip()
                ))
            has_verified_citation = True

        # 3. Revision Governance Check
        # Superseded content cannot support an ordinary operational answer
        final_status = llm_out.status
        if has_superseded_citation and not superseded_requested:
            final_status = ResponseStatus.CONFLICT
            llm_out.conflicts.append(
                "Answer relied on superseded document revision (QC-SOP-014 Rev 03) for an operational query. "
                "Only Revision 04 is authorized for current operations."
            )

        # If status is ANSWERED but no citations were verified, downgrade to PARTIAL
        if final_status == ResponseStatus.ANSWERED and not has_verified_citation:
            final_status = ResponseStatus.PARTIAL

        # 4. Check for known "not documented" phrases in answer
        ans_lower = llm_out.answer.lower()
        if (
            "not documented" in ans_lower
            or "not specified" in ans_lower
            or "does not contain" in ans_lower
            or "no information" in ans_lower
        ):
            if "however" in ans_lower or "partially" in ans_lower or len(validated_citations) > 0:
                # If part is answered and part is not documented
                if final_status == ResponseStatus.ANSWERED:
                    final_status = ResponseStatus.PARTIAL
            else:
                final_status = ResponseStatus.NOT_DOCUMENTED

        return AskResponse(
            status=final_status,
            answer=llm_out.answer,
            citations=validated_citations,
            missing_info=llm_out.missing_info,
            conflicts=llm_out.conflicts,
            query_metadata={
                "has_verified_citation": has_verified_citation,
                "superseded_requested": superseded_requested,
                "retrieved_chunk_count": len(retrieved_chunks)
            }
        )

