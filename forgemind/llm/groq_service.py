import json
import logging
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
from groq import Groq

from forgemind.llm.base import LLMService
from forgemind.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

class GroqService(LLMService):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Groq client initialization failed: {e}")

    def is_available(self) -> bool:
        return bool(self.client and self.api_key and len(self.api_key.strip()) > 5)

    def get_provider_name(self) -> str:
        return "groq"

    def get_model_name(self) -> str:
        return self.model

    def generate(self, prompt: str, system_prompt: str) -> str:
        if not self.is_available():
            raise RuntimeError("Groq API key not configured or client unavailable.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content or ""

    def generate_structured(self, prompt: str, system_prompt: str, schema: Type[T]) -> T:
        if not self.is_available():
            raise RuntimeError("Groq API key not configured or client unavailable.")

        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_with_schema = (
            f"{system_prompt}\n\n"
            f"You MUST output strictly valid JSON conforming to this JSON schema:\n{schema_json}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_with_schema},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return schema.model_validate(data)
