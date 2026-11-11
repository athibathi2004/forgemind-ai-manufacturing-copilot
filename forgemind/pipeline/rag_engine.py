import time
import json
import logging
from typing import Optional, List, Dict, Any

from forgemind.core.models import (
    ResponseStatus,
    AskRequest,
    AskResponse,
    LLMOutputSchema,
    Citation,
    Chunk,
)
from forgemind.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from forgemind.llm.factory import LLMFactory
from forgemind.llm.base import LLMService
from forgemind.pipeline.validator import ResponseValidator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are ForgeMind AI, an enterprise manufacturing intelligence copilot for Kaveri Precision Components Pvt. Ltd. (Hosur Plant).
Your purpose is to provide shop-floor supervisors with accurate, evidence-grounded, revision-aware operational answers strictly from authorized plant documentation.

CRITICAL OPERATIONAL RULES:
1. Answer ONLY using the facts present in the provided RETRIEVED CONTEXT. Never invent, extrapolate, or generalize instructions.
2. PRESERVE NUMERICAL VALUES, TOLERANCES, AND UNITS EXACTLY (e.g., "6 - 8 %", "90.000 +/- 0.050 mm", "3.5 - 4.0 MPa (35 - 40 bar)", "0.005 mm max.", "110 Nm", "+0.030 / -0.000").
3. PRESERVE EXACT ROLE TITLES (e.g., "Maintenance technician", "Shift QC Supervisor", "Supplier Quality Engineer (SQE)", "Stores Executive", "Safety Officer", "Production Manager").
4. LOCKOUT/TAGOUT RULE AND EXCEPTION:
   - For lock removal (SAF-SOP-003), you MUST state the general rule: each authorised person removes only their own lock. Nobody may remove another person's lock.
   - You MUST also explicitly state the safety-critical owner-absent exception: If the owner has left the plant, the Shift Supervisor must first make reasonable phone attempts; if unreachable, BOTH the Production Manager AND Safety Officer must authorize removal after physical inspection, recorded on form SF-12, and the owner must be informed on their next shift. Note that a Shift Supervisor cannot independently remove an absent employee's lock.
5. MACHINE SCOPE:
   - Note that MNT-MAN-CNC-02 covers machines CNC-04 to CNC-09 on the shaft machining line. It does NOT cover machine CNC-02.
6. HYDRAULIC HOSE RATING:
   - The hose replacement procedure in MNT-MAN-CNC-02 Section 7 specifies fitting "a new hose of the same rating", but does NOT document the specific numeric pressure rating of the replacement hose itself.
7. ABSENCE OF INFORMATION:
   - If an identifier, alarm code, or procedure is not in the context, explicitly state that it is not documented in the available documents.
   - Do NOT confuse "not documented" with "not allowed" or "not required".
8. REVISION GOVERNANCE:
   - QC-SOP-014 Revision 04 is the latest authorized QC procedure and supersedes Revision 03.
   - Only reference Revision 03 if the user explicitly asks about historical/superseded Revision 03.
9. OUTPUT SCHEMA:
   Return valid JSON with:
   - "status": "answered" | "partial" | "not_documented" | "conflict" | "needs_clarification" | "error"
   - "answer": concise, clear, professional answer
   - "citations": array of {"chunk_id": str, "quote": str}
   - "missing_info": array of strings listing any requested information not found in the documents
   - "conflicts": array of strings listing any conflicting evidence
"""

class RAGEngine:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_service: Optional[LLMService] = None
    ):
        self.retriever = retriever or HybridRetriever()
        self.llm_service = llm_service
        self.validator = ResponseValidator(self.retriever.chunk_map)

    def ask(self, request: AskRequest) -> AskResponse:
        start_time = time.perf_counter()
        
        # 1. Resolve LLM Service
        try:
            llm = self.llm_service or LLMFactory.get_service(request.provider)
        except Exception as e:
            return AskResponse(
                status=ResponseStatus.ERROR,
                answer=f"LLM Provider Error: {str(e)}",
                citations=[],
                missing_info=[],
                conflicts=[],
                query_metadata={"latency_ms": round((time.perf_counter() - start_time) * 1000, 2)}
            )

        # 2. Retrieve Relevant Chunks with Hybrid RRF and Deterministic Safeguards
        retrieval_res: RetrievalResult = self.retriever.retrieve(
            query=request.question,
            top_k=3,
            allow_superseded=request.allow_superseded
        )

        # 3. Deterministic Short-Circuit for Unrecognized Domain Identifiers
        if retrieval_res.unrecognized_identifiers:
            unrec = ", ".join(retrieval_res.unrecognized_identifiers)
            return AskResponse(
                status=ResponseStatus.NOT_DOCUMENTED,
                answer=(
                    f"Alarm/identifier {unrec} is not documented in the available QMS and maintenance documentation "
                    f"(MNT-MAN-CNC-02, SAF-SOP-003, QC-SOP-014 Rev 04). "
                    f"The system strictly avoids inferring instructions from similar alarm codes."
                ),
                citations=[],
                missing_info=[f"Undocumented identifier: {unrec}"],
                conflicts=[],
                query_metadata={
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    "provider": llm.get_provider_name(),
                    "model": llm.get_model_name(),
                    "unrecognized_identifiers": retrieval_res.unrecognized_identifiers
                }
            )

        # 4. Construct Context Block for LLM Prompt
        context_parts = []
        for c in retrieval_res.chunks:
            status_tag = f"[{c.document_status.value.upper()}]" if c.document_status != "current" else "[CURRENT]"
            context_parts.append(
                f"--- CHUNK ID: {c.chunk_id} ---\n"
                f"Document: {c.document_number} (Rev {c.revision}, Date: {c.effective_date}) {status_tag}\n"
                f"Section: {c.section_number or 'N/A'} - {c.section_title or 'N/A'}\n"
                f"Page: {c.page_number} (Page Span: {c.page_span})\n"
                f"Content:\n{c.content}\n"
            )
        context_str = "\n".join(context_parts)

        user_prompt = (
            f"RETRIEVED CONTEXT:\n{context_str}\n\n"
            f"USER QUESTION:\n{request.question}\n\n"
            f"Provide a grounded, concise answer adhering to all operational rules."
        )

        # 5. Invoke LLM with Structured Schema
        try:
            llm_output = llm.generate_structured(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                schema=LLMOutputSchema
            )
        except Exception as e:
            logger.exception("Structured LLM generation failed, attempting fallback parsing...")
            try:
                # Raw text fallback with JSON extraction
                raw_text = llm.generate(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
                # Extract JSON block
                clean_json = raw_text.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                data = json.loads(clean_json)
                llm_output = LLMOutputSchema.model_validate(data)
            except Exception as fallback_err:
                return AskResponse(
                    status=ResponseStatus.ERROR,
                    answer=f"Failed to generate structured answer: {str(e)} | Fallback: {str(fallback_err)}",
                    citations=[],
                    missing_info=[],
                    conflicts=[],
                    query_metadata={"latency_ms": round((time.perf_counter() - start_time) * 1000, 2)}
                )

        # 6. Post-Generation Programmatic Validation
        validated_response = self.validator.validate(
            llm_out=llm_output,
            retrieved_chunks=retrieval_res.chunks,
            unrecognized_ids=retrieval_res.unrecognized_identifiers,
            superseded_requested=retrieval_res.superseded_requested
        )

        # 7. Attach Operational Metadata
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        validated_response.query_metadata.update({
            "latency_ms": latency,
            "provider": llm.get_provider_name(),
            "model": llm.get_model_name(),
            "query_identifiers": retrieval_res.query_identifiers,
            "retrieved_chunk_ids": [c.chunk_id for c in retrieval_res.chunks]
        })

        return validated_response
