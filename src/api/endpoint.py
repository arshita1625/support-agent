#!/usr/bin/env python3
"""API endpoints for Support Agent RAG System."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.ticket import SupportTicket
from services.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG service
rag_service = RAGService()

# Request/Response models
class TicketRequest(BaseModel):
    ticket_text: str = Field(..., min_length=5)

class TicketResponse(BaseModel):
    answer: str
    references: List[str]
    action_required: str

# Create router
router = APIRouter()

@router.post("/resolve-ticket", response_model=TicketResponse)
async def resolve_ticket(request: TicketRequest):
    
    try:
        import main
        if main.rag_service is None:
            from services.rag_service import RAGService
            main.rag_service = RAGService()
            try:
                await main.rag_service.ensure_documents_current_on_startup()
            except Exception as e:
                logger.warning(f"Document update check failed during endpoint init: {e}")

        # Create support ticket
        ticket = SupportTicket(
            ticket_text=request.ticket_text,
            priority="medium"
        )
        
        # Process through RAG pipeline
        mcp_response = await rag_service.process_support_ticket(ticket)
        
        # Return MCP-compliant response
        return TicketResponse(
            answer=mcp_response.answer,
            references=mcp_response.references,
            action_required=mcp_response.action_required
        )
        
    except Exception as e:
        logger.error(f"Failed to process ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
        return "ok"
