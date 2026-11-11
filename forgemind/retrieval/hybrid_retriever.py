import re
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from rank_bm25 import BM25Okapi

from forgemind.core.models import Chunk, DocumentStatus
from forgemind.ingestion.chunker import DocumentChunker, extract_identifiers, normalize_identifier
from forgemind.retrieval.embedder import LocalEmbedder
from forgemind.config import settings

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text preserving identifiers like AL-104, BH-220, CNC-02."""
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", text.lower())
    return tokens

class RetrievalResult:
    def __init__(
        self,
        chunks: List[Chunk],
        scores: Dict[str, float],
        query_identifiers: List[str],
        unrecognized_identifiers: List[str],
        superseded_requested: bool
    ):
        self.chunks = chunks
        self.scores = scores
        self.query_identifiers = query_identifiers
        self.unrecognized_identifiers = unrecognized_identifiers
        self.superseded_requested = superseded_requested

class HybridRetriever:
    """
    Hybrid retriever combining Dense BGE embeddings, BM25 keyword matching,
    Reciprocal Rank Fusion (RRF), exact identifier boosting, and safety cross-linking.
    """
    def __init__(self, chunks: Optional[List[Chunk]] = None, embedder: Optional[LocalEmbedder] = None):
        self.embedder = embedder or LocalEmbedder()
        self.chunker = DocumentChunker()
        self.chunks: List[Chunk] = chunks if chunks is not None else self.chunker.process_all_documents()
        
        # Build indexes
        self.chunk_map: Dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        self.corpus_identifiers: Set[str] = set()
        for c in self.chunks:
            self.corpus_identifiers.update(c.identifiers)
            
        # BM25 index
        self.bm25_corpus = [tokenize_for_bm25(f"{c.document_number} {c.section_title or ''} {c.content}") for c in self.chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        
        # Dense index
        self.embeddings: np.ndarray = self.embedder.embed_chunks(self.chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allow_superseded: bool = False
    ) -> RetrievalResult:
        query_clean = query.strip()
        
        # 1. Identify domain identifiers in query (e.g. AL-104, AL-106, CNC-02, BH-220)
        query_ids = extract_identifiers(query_clean)
        
        # Check for undocumented identifiers in query
        unrecognized_ids = []
        for q_id in query_ids:
            # If it's an alarm code, part code, or machine code not in any chunk
            if q_id not in self.corpus_identifiers:
                unrecognized_ids.append(q_id)

        # 2. Check if historical superseded info is explicitly requested
        superseded_requested = allow_superseded or bool(
            re.search(r"\b(superseded|rev\s*03|revision\s*03|previous\s*revision|old\s*version)\b", query_clean, re.IGNORECASE)
        )

        # 3. Filter eligible chunks
        eligible_indices = []
        for idx, chunk in enumerate(self.chunks):
            if chunk.document_status == DocumentStatus.SUPERSEDED and not superseded_requested:
                continue
            eligible_indices.append(idx)

        if not eligible_indices:
            return RetrievalResult([], {}, query_ids, unrecognized_ids, superseded_requested)

        # 4. Dense Retrieval
        q_emb = self.embedder.embed_text(query_clean)
        dense_sims = np.dot(self.embeddings[eligible_indices], q_emb)
        # Sort eligible indices by dense similarity
        dense_ranking_order = np.argsort(-dense_sims)
        dense_rank_dict = {}
        for rank, order_idx in enumerate(dense_ranking_order):
            actual_idx = eligible_indices[order_idx]
            dense_rank_dict[self.chunks[actual_idx].chunk_id] = rank + 1

        # 5. BM25 Retrieval
        q_tokens = tokenize_for_bm25(query_clean)
        bm25_scores = self.bm25.get_scores(q_tokens)
        eligible_bm25_scores = [bm25_scores[idx] for idx in eligible_indices]
        bm25_ranking_order = np.argsort(-np.array(eligible_bm25_scores))
        bm25_rank_dict = {}
        for rank, order_idx in enumerate(bm25_ranking_order):
            actual_idx = eligible_indices[order_idx]
            bm25_rank_dict[self.chunks[actual_idx].chunk_id] = rank + 1

        # 6. Reciprocal Rank Fusion (RRF) with Identifier Boosting
        k_rrf = 60
        rrf_scores: Dict[str, float] = {}
        for idx in eligible_indices:
            c = self.chunks[idx]
            cid = c.chunk_id
            d_rank = dense_rank_dict.get(cid, len(eligible_indices))
            b_rank = bm25_rank_dict.get(cid, len(eligible_indices))
            
            score = (1.0 / (k_rrf + d_rank)) + (1.0 / (k_rrf + b_rank))
            
            # Exact Identifier Boost
            for q_id in query_ids:
                if q_id in c.identifiers:
                    score += 0.05
                    
            # Superseded query boost for Rev 03 chunk
            if superseded_requested and c.document_status == DocumentStatus.SUPERSEDED:
                score += 0.08
                
            rrf_scores[cid] = score

        # 7. Select Top-K Chunks
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_cids = sorted_cids[:top_k]
        
        # 8. Cross-Document Reference Expansion and Co-Retrieval Safeguards
        final_cids = list(top_cids)
        
        # Co-retrieval Rule A: LOTO Section 5 & 6 (lock removal and owner-absent exception)
        # If any lock removal chunk is retrieved, ensure S05_S06 is included
        for cid in top_cids:
            chunk = self.chunk_map[cid]
            if "SAF-SOP-003" in chunk.document_number and ("remove" in query_clean.lower() or "lock" in query_clean.lower()):
                sec5_6_id = "SAF-SOP-003_R05_S05_S06_P01_P02"
                if sec5_6_id not in final_cids and sec5_6_id in self.chunk_map:
                    final_cids.append(sec5_6_id)
                    rrf_scores[sec5_6_id] = rrf_scores.get(cid, 0.0) + 0.01

        # Co-retrieval Rule B: Hydraulic Hose replacement references SAF-SOP-003
        for cid in top_cids:
            if cid == "MNT-MAN-CNC-02_R02_S07_P02":
                # References SAF-SOP-003 Lockout/Tagout
                saf_cid = "SAF-SOP-003_R05_S04_P01"
                if saf_cid not in final_cids and saf_cid in self.chunk_map:
                    final_cids.append(saf_cid)
                    rrf_scores[saf_cid] = 0.02

        # Order final chunks
        final_chunks = [self.chunk_map[cid] for cid in final_cids]
        
        return RetrievalResult(
            chunks=final_chunks,
            scores=rrf_scores,
            query_identifiers=query_ids,
            unrecognized_identifiers=unrecognized_ids,
            superseded_requested=superseded_requested
        )
