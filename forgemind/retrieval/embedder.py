import os
from typing import List, Optional
import numpy as np

# Force offline mode so huggingface hub never attempts network checks or hangs on retries
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

from forgemind.config import settings
from forgemind.core.models import Chunk

class LocalEmbedder:
    """
    Local embedding generator using BAAI/bge-small-en-v1.5 via sentence-transformers.
    Runs completely offline locally on CPU/GPU.
    """
    _instance = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            # Fallback if local_files_only isn't supported by the wrapper version
            self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        texts = [f"{c.section_title or ''} {c.content}" for c in chunks]
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

