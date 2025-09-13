#!/usr/bin/env python3
"""Support Agent RAG System API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.endpoint import router

# Create FastAPI app
app = FastAPI(
    title="Support Agent RAG System",
    description="AI-powered knowledge assistant for support tickets",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)

@app.get("/")
async def root():
    """API information."""
    return {
        "service": "Support Agent RAG System",
        "version": "1.0.0",
        "endpoint": "POST /resolve-ticket"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
