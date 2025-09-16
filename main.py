#!/usr/bin/env python3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
from src.api.upload_files import router as upload_router
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.endpoint import router

app = FastAPI(
    title="Support Agent RAG System",
    description="AI-powered knowledge assistant for support tickets",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(upload_router, prefix="/api/upload", tags=["File Upload"])
rag_service = None
@app.on_event("startup")
async def startup_event():
    print(" Support Agent RAG System starting...")
    
    try:
        from src.services.rag_service import RAGService
        # Create RAG service
        global rag_service
        rag_service = RAGService()
        
        print("Checking for policy updates...")
        updated = await rag_service.ensure_documents_current_on_startup()
        
        if updated:
            print(" Policy check completed")
        else:
            print("Policy update had issues, but system is operational")
            
    except Exception as e:
        print("System starting anyway with existing data...")
    
    print("API available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
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
