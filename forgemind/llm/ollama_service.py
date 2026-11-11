import json
import logging
from typing import Type, TypeVar, Optional
import httpx
from pydantic import BaseModel

from forgemind.llm.base import LLMService
from forgemind.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

class OllamaService(LLMService):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = 90.0

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=1.0)
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                # If configured model is in list or any model exists
                return any(self.model in m or m.startswith(self.model.split(":")[0]) for m in models)
        except Exception:
            return False
        return False

    def get_provider_name(self) -> str:
        return "ollama"

    def get_model_name(self) -> str:
        return self.model

    def generate(self, prompt: str, system_prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256, "num_ctx": 2048}
        }
        try:
            r = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def generate_structured(self, prompt: str, system_prompt: str, schema: Type[T]) -> T:
        compact_schema = """Respond with strictly valid JSON in this exact structure:
{
  "status": "answered" | "partial" | "not_documented" | "conflict" | "needs_clarification" | "error",
  "answer": "string",
  "citations": [{"chunk_id": "string", "quote": "string"}],
  "missing_info": [],
  "conflicts": []
}"""
        system_with_schema = f"{system_prompt}\n\n{compact_schema}"
        payload = {
            "model": self.model,
            "system": system_with_schema,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 256, "num_ctx": 2048}
        }
        try:
            r = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            r.raise_for_status()
            raw_output = r.json().get("response", "{}")
            data = json.loads(raw_output)
            return schema.model_validate(data)
        except Exception as e:
            logger.warning(f"Ollama structured generation failed: {e}")
            raise
