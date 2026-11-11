import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load active .env if present
load_dotenv(override=False)

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
INDEX_DIR = BASE_DIR / "data" / "indices"

class Settings(BaseModel):
    app_name: str = "ForgeMind AI"
    app_subtitle: str = "Manufacturing Intelligence Copilot"
    plant_name: str = "Kaveri Precision Components Pvt. Ltd. - Hosur Plant"
    
    # Provider options: "auto", "groq", "ollama"
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto").lower()
    
    # Groq settings
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Ollama settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    
    # Embedding settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    # Network / Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Paths
    base_dir: Path = BASE_DIR
    documents_dir: Path = DOCUMENTS_DIR
    index_dir: Path = INDEX_DIR

settings = Settings()
