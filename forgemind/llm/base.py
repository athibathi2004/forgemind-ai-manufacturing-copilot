from abc import ABC, abstractmethod
from typing import Type, TypeVar, Optional, Dict, Any
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMService(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generates raw text given a prompt and system instruction."""
        pass

    @abstractmethod
    def generate_structured(self, prompt: str, system_prompt: str, schema: Type[T]) -> T:
        """Generates structured output validated against a Pydantic schema."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the service is configured and reachable."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Returns provider identifier: 'groq' or 'ollama'."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Returns the specific model name in use."""
        pass

