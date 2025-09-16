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
rag_service = None
@app.on_event("startup")
async def startup_event():
    """Initialize services and check for document updates."""
    print("🚀 Support Agent RAG System starting...")
    print("📚 Initializing RAG services...")
    
    try:
        from src.services.rag_service import RAGService
        
        # Create RAG service (this will be reused by endpoints)
        global rag_service
        rag_service = RAGService()
        
        print("Checking for policy updates...")
        updated = await rag_service.ensure_documents_current_on_startup()
        
        if updated:
            print(" Policy check completed")
        else:
            print("⚠️ Policy update had issues, but system is operational")
            
    except Exception as e:
        print(f"⚠️ Startup checks failed: {e}")
        print("🔄 System starting anyway with existing data...")
    
    print("🔗 API available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
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
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
