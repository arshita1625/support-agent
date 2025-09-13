#!/usr/bin/env python3
"""RAG service for orchestrating query processing, retrieval, and response generation."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.ticket import SupportTicket
from models.response import MCPResponse
from models.rag import RetrievedDocument, RAGContext
from models.common import HealthStatus, ErrorResponse
from services.document_processor import DocumentProcessorService
from services.llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """Main RAG service for processing support tickets and generating responses."""
    
    def __init__(
        self,
        document_processor: Optional[DocumentProcessorService] = None,
        llm_service: Optional[LLMService] = None,
        default_retrieval_limit: int = 5,
        min_similarity_threshold: float = 0.3,
        high_confidence_threshold: float = 0.8
    ):
        """Initialize RAG service.
        
        Args:
            document_processor: Document processing service
            llm_service: LLM service for response generation
            default_retrieval_limit: Default number of documents to retrieve
            min_similarity_threshold: Minimum similarity for relevant documents
            high_confidence_threshold: Threshold for high confidence responses
        """
        
        # Initialize services
        self.document_processor = document_processor or DocumentProcessorService()
        self.llm_service = llm_service or LLMService()
        
        # Configuration
        self.default_retrieval_limit = default_retrieval_limit
        self.min_similarity_threshold = min_similarity_threshold
        self.high_confidence_threshold = high_confidence_threshold
        
        logger.info("🧠 RAG service initialized")
    
    async def check_health(self) -> HealthStatus:
        """Check health of all RAG service components."""
        
        health = HealthStatus(status="healthy", version="1.0.0")
        
        # Ensure additional_context exists
        if not hasattr(health, 'additional_context'):
            health.additional_context = {}
        
        # Check document processor health
        try:
            processor_health = await self.document_processor.check_health()
            health.add_service_status("document_processor", processor_health.is_all_services_healthy())
            
            if hasattr(processor_health, 'additional_context') and processor_health.additional_context:
                health.additional_context.update({
                    f"processor_{k}": v for k, v in processor_health.additional_context.items()
                })
                
        except Exception as e:
            health.add_service_status("document_processor", False)
            logger.error(f"Document processor health check failed: {e}")
        
        # Check LLM service health
        try:
            llm_health = await self.llm_service.check_health()
            health.add_service_status("llm_service", llm_health.is_all_services_healthy())
            
            if hasattr(llm_health, 'additional_context') and llm_health.additional_context:
                health.additional_context.update({
                    f"llm_{k}": v for k, v in llm_health.additional_context.items()
                })
                
        except Exception as e:
            health.add_service_status("llm_service", False)
            logger.error(f"LLM service health check failed: {e}")
        
        # Add RAG-specific context
        health.additional_context.update({
            "retrieval_limit": self.default_retrieval_limit,
            "similarity_threshold": self.min_similarity_threshold,
            "confidence_threshold": self.high_confidence_threshold
        })
        
        # Update overall status
        try:
            health.update_overall_status()
        except AttributeError:
            # Fallback if method doesn't exist
            unhealthy_services = [name for name, status in health.services.items() if not status]
            if not unhealthy_services:
                health.status = "healthy"
            elif len(unhealthy_services) < len(health.services) / 2:
                health.status = "degraded"
            else:
                health.status = "unhealthy"
        
        return health
    
    async def process_support_ticket(
        self,
        ticket: SupportTicket,
        retrieval_limit: Optional[int] = None,
        custom_instructions: Optional[str] = None
    ) -> MCPResponse:
        """Process a support ticket and generate a complete response.
        
        Args:
            ticket: SupportTicket object with customer query
            retrieval_limit: Number of documents to retrieve (uses default if None)
            custom_instructions: Additional instructions for LLM
            
        Returns:
            MCPResponse with generated answer and metadata
        """
        
        start_time = time.time()
        logger.info(f"🎫 Processing support ticket: {ticket.ticket_id}")
        
        try:
            # 1. Validate ticket
            if not ticket.ticket_text or not ticket.ticket_text.strip():
                raise ValueError("Ticket text cannot be empty")
            
            # 2. Retrieve relevant documents
            logger.info("🔍 Retrieving relevant documents...")
            retrieval_limit = retrieval_limit or self.default_retrieval_limit
            
            rag_context = await self.document_processor.search_documents(
                query=ticket.ticket_text,
                limit=retrieval_limit,
                score_threshold=self.min_similarity_threshold
            )
            
            logger.info(f"📊 Retrieved {len(rag_context.retrieved_documents)} documents")
            logger.info(f"📈 Average similarity: {rag_context.average_similarity:.3f}")
            
            # 3. Assess retrieval quality
            retrieval_quality = self._assess_retrieval_quality(rag_context)
            logger.info(f"📋 Retrieval quality: {retrieval_quality['quality_level']}")
            
            # 4. Generate response using LLM
            logger.info("🤖 Generating LLM response...")
            
            response = await self.llm_service.generate_support_response(
                ticket=ticket,
                rag_context=rag_context,
                custom_instructions=custom_instructions
            )
            
            # 5. Enhance response with RAG metadata
            self._enhance_response_with_rag_data(response, rag_context, retrieval_quality, ticket)
            
            # 6. Log processing results
            processing_time = time.time() - start_time
            logger.info(f"✅ Ticket processed in {processing_time:.2f}s")
            logger.info(f"🎯 Response action: {response.action_required}")
            logger.info(f"📊 Confidence: {response.confidence_score:.3f}")
            
            return response
            
        except Exception as e:
            # Generate error response
            logger.error(f"❌ Failed to process ticket {ticket.ticket_id}: {e}")
            
            error_response = MCPResponse(
                answer=f"I apologize, but I encountered an error while processing your request: {str(e)}. Please contact our support team directly for immediate assistance.",
                action_required="escalate_to_technical",
                confidence_score=0.0,
                priority_level="high"
            )
            
            error_response.escalation_reason = f"Processing error: {str(e)}"
            
            return error_response
    
    def _assess_retrieval_quality(self, rag_context: RAGContext) -> Dict[str, Any]:
        """Assess the quality of document retrieval for response confidence."""
        
        if not rag_context.retrieved_documents:
            return {
                "quality_level": "poor",
                "confidence_modifier": 0.0,
                "reasons": ["No relevant documents found"]
            }
        
        reasons = []
        quality_score = 0.0
        
        # Check average similarity
        avg_sim = rag_context.average_similarity
        if avg_sim >= 0.8:
            quality_score += 0.4
            reasons.append(f"High average similarity ({avg_sim:.3f})")
        elif avg_sim >= 0.6:
            quality_score += 0.2
            reasons.append(f"Good average similarity ({avg_sim:.3f})")
        else:
            reasons.append(f"Low average similarity ({avg_sim:.3f})")
        
        # Check maximum similarity
        max_sim = rag_context.max_similarity
        if max_sim >= 0.9:
            quality_score += 0.3
            reasons.append(f"Excellent top match ({max_sim:.3f})")
        elif max_sim >= 0.7:
            quality_score += 0.2
            reasons.append(f"Good top match ({max_sim:.3f})")
        else:
            reasons.append(f"Weak top match ({max_sim:.3f})")
        
        # Check document diversity
        doc_types = set(doc.document_type for doc in rag_context.retrieved_documents)
        if len(doc_types) > 1:
            quality_score += 0.1
            reasons.append(f"Good source diversity ({len(doc_types)} types)")
        
        # Check number of high-confidence documents
        high_conf_docs = [doc for doc in rag_context.retrieved_documents if doc.similarity_score >= self.high_confidence_threshold]
        if len(high_conf_docs) >= 2:
            quality_score += 0.2
            reasons.append(f"Multiple high-confidence sources ({len(high_conf_docs)} docs)")
        elif len(high_conf_docs) >= 1:
            quality_score += 0.1
            reasons.append(f"One high-confidence source")
        else:
            reasons.append("No high-confidence sources")
        
        # Determine quality level
        if quality_score >= 0.8:
            quality_level = "excellent"
        elif quality_score >= 0.6:
            quality_level = "good"
        elif quality_score >= 0.3:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return {
            "quality_level": quality_level,
            "confidence_modifier": min(quality_score, 1.0),
            "reasons": reasons,
            "metrics": {
                "average_similarity": avg_sim,
                "max_similarity": max_sim,
                "document_count": len(rag_context.retrieved_documents),
                "high_confidence_count": len(high_conf_docs),
                "source_diversity": len(doc_types)
            }
        }
    
    def _enhance_response_with_rag_data(
        self,
        response: MCPResponse,
        rag_context: RAGContext,
        retrieval_quality: Dict[str, Any],
        ticket: SupportTicket
    ):
        """Enhance response with RAG-specific metadata and adjustments."""
        
        # Adjust confidence based on retrieval quality
        if response.confidence_score is not None:
            quality_modifier = retrieval_quality["confidence_modifier"]
            adjusted_confidence = response.confidence_score * quality_modifier
            response.confidence_score = max(adjusted_confidence, 0.1)  # Minimum confidence
        
        # Add retrieval metadata to additional context
        response.additional_context.update({
            "retrieval_stats": {
                "documents_retrieved": len(rag_context.retrieved_documents),
                "average_similarity": rag_context.average_similarity,
                "max_similarity": rag_context.max_similarity,
                "retrieval_time_ms": rag_context.retrieval_time_ms,
                "query_embedding_time_ms": rag_context.query_embedding_time_ms
            },
            "quality_assessment": retrieval_quality,
            "ticket_metadata": {
            "ticket_keywords": ticket.extract_keywords(),  
            "priority": ticket.priority,                  
            "urgency_indicator": "high" if ticket.priority in ["high", "urgent"] else "normal"  
        }
        })
        
        # Determine if escalation is needed based on retrieval quality
        if retrieval_quality["quality_level"] == "poor" and response.action_required == "no_action":
            response.action_required = "escalate_to_technical"
            response.escalation_reason = "Low confidence due to poor document retrieval quality"
        
        # Add relevant policy information
        policy_docs = [doc for doc in rag_context.retrieved_documents if doc.document_type == "policy"]
        if policy_docs:
            response.relevant_policies = [doc.get_preview(100) for doc in policy_docs[:3]]
    
    async def process_batch_tickets(
        self,
        tickets: List[SupportTicket],
        max_concurrent: int = 3
    ) -> List[MCPResponse]:
        """Process multiple support tickets concurrently.
        
        Args:
            tickets: List of support tickets to process
            max_concurrent: Maximum number of concurrent processing tasks
            
        Returns:
            List of MCPResponse objects in the same order
        """
        
        logger.info(f"📦 Processing batch of {len(tickets)} tickets (max concurrent: {max_concurrent})")
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_ticket(ticket: SupportTicket) -> MCPResponse:
            async with semaphore:
                return await self.process_support_ticket(ticket)
        
        # Process all tickets concurrently
        start_time = time.time()
        responses = await asyncio.gather(
            *[process_single_ticket(ticket) for ticket in tickets],
            return_exceptions=True
        )
        
        # Handle any exceptions
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = MCPResponse(
                    answer=f"Failed to process ticket: {str(response)}",
                    action_required="escalate_to_technical",
                    confidence_score=0.0
                )
                final_responses.append(error_response)
                logger.error(f"Batch processing error for ticket {i}: {response}")
            else:
                final_responses.append(response)
        
        processing_time = time.time() - start_time
        logger.info(f"📦 Batch processing completed in {processing_time:.2f}s")
        
        return final_responses
    
    async def get_rag_analytics(self) -> Dict[str, Any]:
        """Get analytics about RAG service performance and usage."""
        
        try:
            # Get document processor status
            processor_status = self.document_processor.get_processing_status()
            
            # Get LLM service stats (if available)
            llm_stats = {}
            if hasattr(self.llm_service, 'get_usage_stats'):
                llm_stats = await self.llm_service.get_usage_stats()
            
            analytics = {
                "service_status": "operational",
                "document_index": {
                    "total_documents": processor_status.get("indexed_documents", 0),
                    "document_types": processor_status.get("document_types", {}),
                    "last_updated": processor_status.get("last_updated", "unknown")
                },
                "configuration": {
                    "retrieval_limit": self.default_retrieval_limit,
                    "similarity_threshold": self.min_similarity_threshold,
                    "confidence_threshold": self.high_confidence_threshold,
                    "embedding_model": processor_status.get("embedding_model", "unknown")
                },
                "llm_stats": llm_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get RAG analytics: {e}")
            return {
                "service_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_rag_pipeline(self, test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Test the complete RAG pipeline with sample queries.
        
        Args:
            test_queries: List of test queries (uses defaults if None)
            
        Returns:
            Test results and performance metrics
        """
        
        if test_queries is None:
            test_queries = [
                "My domain was suspended without notice",
                "How do I update WHOIS information?",
                "Payment failed but my card works elsewhere",
                "I need to transfer my domain to another registrar",
                "My website is loading slowly"
            ]
        
        logger.info(f"🧪 Testing RAG pipeline with {len(test_queries)} queries")
        
        test_results = {
            "total_queries": len(test_queries),
            "successful_responses": 0,
            "failed_responses": 0,
            "average_processing_time_ms": 0,
            "average_confidence": 0,
            "query_results": []
        }
        
        total_processing_time = 0
        total_confidence = 0
        
        for i, query in enumerate(test_queries):
            logger.info(f"Testing query {i+1}/{len(test_queries)}: '{query[:50]}...'")
            
            try:
                # Create test ticket
                test_ticket = SupportTicket(
                    ticket_text=query,
                    priority="medium"
                )
                
                # Process ticket
                start_time = time.time()
                response = await self.process_support_ticket(test_ticket)
                processing_time = (time.time() - start_time) * 1000
                
                result = {
                    "query": query,
                    "success": True,
                    "processing_time_ms": round(processing_time, 2),
                    "confidence": response.confidence_score or 0.0,
                    "action_required": response.action_required,
                    "answer_length": len(response.answer),
                    "references_count": len(response.references),
                    "answer_preview": response.answer[:100] + "..." if len(response.answer) > 100 else response.answer
                }
                
                test_results["successful_responses"] += 1
                total_processing_time += processing_time
                total_confidence += (response.confidence_score or 0.0)
                
                logger.info(f"✅ Success - Action: {response.action_required}, Confidence: {response.confidence_score:.3f}")
                
            except Exception as e:
                result = {
                    "query": query,
                    "success": False,
                    "error": str(e),
                    "processing_time_ms": 0,
                    "confidence": 0.0
                }
                
                test_results["failed_responses"] += 1
                logger.error(f"❌ Failed: {e}")
            
            test_results["query_results"].append(result)
        
        # Calculate averages
        if test_results["successful_responses"] > 0:
            test_results["average_processing_time_ms"] = round(
                total_processing_time / test_results["successful_responses"], 2
            )
            test_results["average_confidence"] = round(
                total_confidence / test_results["successful_responses"], 3
            )
        
        success_rate = (test_results["successful_responses"] / test_results["total_queries"]) * 100
        logger.info(f"🎉 RAG pipeline tests completed: {success_rate:.1f}% success rate")
        
        return test_results


# Test the RAG service if run directly
if __name__ == "__main__":
    print("🧪 Testing RAG Service...")
    print("=" * 50)
    
    # Check dependencies
    import os
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    async def run_tests():
        # Initialize RAG service
        print("\n🧠 Initializing RAG Service")
        rag_service = RAGService()
        
        # Health check
        print("\n🏥 Testing Health Check")
        health = await rag_service.check_health()
        print(f"   Overall status: {health.status}")
        print(f"   Services: {health.services}")
        
        if not health.is_all_services_healthy():
            print("\n❌ Some services are unhealthy!")
            unhealthy = health.get_unhealthy_services()
            print(f"   Unhealthy services: {unhealthy}")
            return
        
        # Test single ticket processing
        print("\n🎫 Testing Single Ticket Processing")
        test_ticket = SupportTicket(
            ticket_text="My domain was suspended and I didn't receive any notification. How can I reactivate it?",
            priority="high",
            customer_id="customer_123"
        )
        
        try:
            response = await rag_service.process_support_ticket(test_ticket)
            print(f"   ✅ Response generated:")
            print(f"      Action: {response.action_required}")
            print(f"      Confidence: {response.confidence_score:.3f}")
            print(f"      Answer length: {len(response.answer)} chars")
            print(f"      References: {len(response.references)}")
            print(f"      Answer preview: {response.answer[:150]}...")
        except Exception as e:
            print(f"   ❌ Single ticket processing failed: {e}")
            return
        
        # Test pipeline with multiple queries
        print("\n🧪 Testing Complete RAG Pipeline")
        try:
            pipeline_results = await rag_service.test_rag_pipeline()
            print(f"   ✅ Pipeline test results:")
            print(f"      Success rate: {(pipeline_results['successful_responses']/pipeline_results['total_queries']*100):.1f}%")
            print(f"      Average processing time: {pipeline_results['average_processing_time_ms']:.1f}ms")
            print(f"      Average confidence: {pipeline_results['average_confidence']:.3f}")
        except Exception as e:
            print(f"   ❌ Pipeline test failed: {e}")
        
        # Get analytics
        print("\n📊 Testing Analytics")
        try:
            analytics = await rag_service.get_rag_analytics()
            print(f"   ✅ Analytics retrieved:")
            print(f"      Service status: {analytics.get('service_status', 'unknown')}")
            print(f"      Indexed documents: {analytics.get('document_index', {}).get('total_documents', 0)}")
            print(f"      Embedding model: {analytics.get('configuration', {}).get('embedding_model', 'unknown')}")
        except Exception as e:
            print(f"   ❌ Analytics failed: {e}")
        
        print("\n🎉 RAG Service testing completed!")
        print("\n💡 Next steps:")
        print("   1. Your RAG system is now fully functional")
        print("   2. Create API endpoints to expose the RAG service")
        print("   3. Build a frontend interface for testing")
    
    # Run async tests
    asyncio.run(run_tests())
