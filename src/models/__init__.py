#!/usr/bin/env python3
"""Core data models for the Knowledge Assistant RAG system."""

# Document models
from .document import Document, DocumentChunk

# Ticket models  
from .ticket import SupportTicket

# Response models
from .response import MCPResponse

# RAG pipeline models
from .rag import RetrievedDocument, RAGContext

# Common utility models
from .common import (
    HealthStatus, ErrorResponse, SystemInfo, 
    PaginationInfo, ValidationError, ValidationErrors, 
    APIResponse
)

# Version info
__version__ = "1.0.0"

# Export all models for easy importing
__all__ = [
    # Document models
    "Document",
    "DocumentChunk",
    
    # Ticket models
    "SupportTicket", 
    
    # Response models
    "MCPResponse",
    
    # RAG models
    "RetrievedDocument",
    "RAGContext",
    
    # Common models
    "HealthStatus",
    "ErrorResponse",
    "SystemInfo",
    "PaginationInfo", 
    "ValidationError",
    "ValidationErrors",
    "APIResponse",
]

# Model groups for organized importing
DOCUMENT_MODELS = [Document, DocumentChunk]
TICKET_MODELS = [SupportTicket]
RESPONSE_MODELS = [MCPResponse]
RAG_MODELS = [RetrievedDocument, RAGContext] 
COMMON_MODELS = [HealthStatus, ErrorResponse, SystemInfo, PaginationInfo, ValidationError, ValidationErrors, APIResponse]

# All models list
ALL_MODELS = DOCUMENT_MODELS + TICKET_MODELS + RESPONSE_MODELS + RAG_MODELS + COMMON_MODELS

# Model validation helpers
def validate_all_models():
    """Test that all models can be instantiated."""
    try:
        # Test document models
        doc = Document(content="Test document content", document_type="guide")
        chunk = DocumentChunk(
            parent_document_id=doc.id,
            content="Test chunk",
            chunk_index=0,
            start_char=0,
            end_char=10,
            document_type="guide"
        )
        
        # Test ticket model
        ticket = SupportTicket(ticket_text="Test support ticket content")
        
        # Test response models
        response = MCPResponse(
            answer="Test response with sufficient content for validation",
            action_required="no_action"
        )
        
        # Test RAG models
        retrieved = RetrievedDocument(
            document_id="test_id",
            content="Test retrieved content",
            similarity_score=0.8,
            document_type="test"
        )
        
        rag_context = RAGContext(
            query="Test query",
            retrieved_documents=[retrieved],
            total_documents_searched=100
        )
        
        # Test common models
        health = HealthStatus(status="healthy")
        error = ErrorResponse(error_code="TEST", message="Test error message")
        pagination = PaginationInfo(page=1, page_size=10, total_items=25)
        api_response = APIResponse.success_response(data={"test": "data"})
        system_info = SystemInfo()
        
        print("✅ All models validated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing all models...")
    validate_all_models()
