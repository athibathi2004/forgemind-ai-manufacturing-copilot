import os
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from forgemind.config import settings
from forgemind.api.routes import router as api_router, get_host_lan_ip

app = FastAPI(
    title=settings.app_name,
    description="Enterprise Manufacturing Intelligence Copilot with revision governance and verifiable evidence",
    version="1.0.0"
)

# Enable CORS for local network and mobile devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(api_router)

# Mount static folder
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
def serve_frontend():
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ForgeMind AI API is active. Static UI files not found."}

def print_banner(host: str, port: int, lan_ip: str):
    banner = f"""
======================================================================
               FORGEMIND AI — MANUFACTURING COPILOT                   
            Kaveri Precision Components Pvt. Ltd. (Hosur)             
======================================================================
  * Local URL:       http://localhost:{port}
  * LAN/Wi-Fi URL:   http://{lan_ip}:{port}
  * Host binding:    {host}:{port}
  * LLM Provider:    {settings.llm_provider}
  * Embeddings:      {settings.embedding_model} (Local BGE)
======================================================================
  Connect any phone, tablet, or terminal on the same Wi-Fi to:
  --> http://{lan_ip}:{port}
======================================================================
"""
    print(banner)

if __name__ == "__main__":
    lan_ip = get_host_lan_ip()
    print_banner(settings.host, settings.port, lan_ip)
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
