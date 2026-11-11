import logging
from typing import Optional
from forgemind.llm.base import LLMService
from forgemind.llm.groq_service import GroqService
from forgemind.llm.ollama_service import OllamaService
from forgemind.config import settings

logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def get_service(requested_provider: Optional[str] = None) -> LLMService:
        provider = (requested_provider or settings.llm_provider).lower()
        
        if provider == "groq":
            groq = GroqService()
            if not groq.is_available():
                raise RuntimeError(
                    "Groq provider was selected but GROQ_API_KEY is missing or invalid. "
                    "Please configure GROQ_API_KEY in .env."
                )
            return groq
            
        elif provider == "ollama":
            ollama = OllamaService()
            if not ollama.is_available():
                raise RuntimeError(
                    f"Ollama provider was selected but Ollama is not running or model '{settings.ollama_model}' "
                    f"is not installed at {settings.ollama_base_url}. Run 'ollama pull {settings.ollama_model}'."
                )
            return ollama
            
        elif provider == "auto":
            # Auto-detection priority: Groq if key exists, else Ollama
            groq = GroqService()
            if groq.is_available():
                logger.info("Auto-selected Groq LLM service.")
                return groq
                
            ollama = OllamaService()
            if ollama.is_available():
                logger.info(f"Auto-selected Ollama LLM service ({ollama.get_model_name()}).")
                return ollama
                
            raise RuntimeError(
                "No LLM provider available! Please either provide a GROQ_API_KEY in .env "
                f"or start Ollama with model '{settings.ollama_model}'."
            )
        else:
            raise ValueError(f"Unknown LLM provider: '{provider}'. Supported: auto, groq, ollama.")

