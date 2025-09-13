#!/usr/bin/env python3
"""LLM service for generating intelligent support responses using OpenAI."""
import os
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
try:
    import openai
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI client not installed. Install with: pip install openai")
    raise

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.ticket import SupportTicket
from models.response import MCPResponse
from models.rag import RAGContext
from models.common import HealthStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    """Service for generating intelligent support responses using OpenAI's LLM."""
    
    def __init__(
        self,
        api_key: Optional[str] = API_KEY,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.3,
        timeout: float = 30.0
    ):
        """Initialize LLM service.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: OpenAI model to use
            max_tokens: Maximum tokens in response
            temperature: Creativity level (0.0-1.0, lower = more focused)
            timeout: Request timeout in seconds
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Model specifications
        self.model_specs = {
            "gpt-3.5-turbo": {
                "context_window": 4096,
                "cost_per_1k_input_tokens": 0.0010,
                "cost_per_1k_output_tokens": 0.0020
            },
            "gpt-3.5-turbo-16k": {
                "context_window": 16384,
                "cost_per_1k_input_tokens": 0.0030,
                "cost_per_1k_output_tokens": 0.0040
            },
            "gpt-4": {
                "context_window": 8192,
                "cost_per_1k_input_tokens": 0.0300,
                "cost_per_1k_output_tokens": 0.0600
            },
            "gpt-4-32k": {
                "context_window": 32768,
                "cost_per_1k_input_tokens": 0.0600,
                "cost_per_1k_output_tokens": 0.1200
            }
        }
        
        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "last_request_time": None
        }
        
        logger.info(f"🤖 LLM service initialized with model: {model}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            **self.model_specs.get(self.model, {})
        }
    
    async def check_health(self) -> HealthStatus:
        """Check LLM service health and connectivity."""
        
        health = HealthStatus(status="healthy", version="1.0.0")
        
        # Ensure additional_context exists
        if not hasattr(health, 'additional_context'):
            health.additional_context = {}
        
        try:
            # Test API connectivity with a simple completion
            start_time = time.time()
            
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with 'OK' if you can process this."}],
                max_tokens=10,
                temperature=0
            )
            
            response_time = time.time() - start_time
            
            if test_response.choices and test_response.choices[0].message.content:
                health.add_service_status("openai_api", True)
                health.add_service_status("model_access", True)
                
                # Add performance metrics
                health.additional_context.update({
                    "model": self.model,
                    "test_response_time_ms": round(response_time * 1000, 2),
                    "context_window": self.model_specs.get(self.model, {}).get("context_window", "unknown"),
                    "usage_stats": self.usage_stats.copy()
                })
            else:
                health.add_service_status("openai_api", True)
                health.add_service_status("model_access", False)
                
        except Exception as e:
            health.add_service_status("openai_api", False)
            health.add_service_status("model_access", False)
            logger.error(f"OpenAI API health check failed: {e}")
            health.additional_context["health_check_error"] = str(e)
        
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
    
    def _build_system_prompt(self, ticket: SupportTicket) -> str:
        """Build system prompt based on ticket characteristics."""
        
        base_prompt = """You are an expert customer support agent for a domain registration and web hosting company. Your role is to provide helpful, accurate, and professional responses to customer inquiries.

Key guidelines:
1. Be helpful, empathetic, and professional
2. Provide specific, actionable guidance
3. Reference company policies when relevant
4. Suggest appropriate next actions
5. Be concise but comprehensive
6. If uncertain, recommend contacting the appropriate team

You have access to company documentation and should base your responses on the provided context."""
        
        # Add ticket-specific context
        keywords = ticket.extract_keywords()
        
        if 'domain' in keywords or 'suspension' in keywords:
            base_prompt += "\n\nThis inquiry is domain-related. Focus on domain management, suspension issues, WHOIS compliance, and reactivation procedures."
        
        if 'billing' in keywords:
            base_prompt += "\n\nThis inquiry is billing-related. Focus on payment issues, refunds, account problems, and billing procedures."
        
        if 'technical' in keywords:
            base_prompt += "\n\nThis inquiry is technical. Focus on DNS, email, SSL, server issues, and technical troubleshooting."
        
        # Adjust tone based on priority
        if ticket.priority in ["high", "urgent"]:
            base_prompt += "\n\nThis is a high-priority inquiry. Be extra attentive and suggest immediate action steps."
        
        return base_prompt
    
    def _build_user_prompt(self, ticket: SupportTicket, rag_context: RAGContext) -> str:
        """Build user prompt with ticket and context information."""
        
        prompt_parts = [
            "Please help with this customer support inquiry:",
            f"\nCustomer Query: {ticket.ticket_text}",
            f"\nPriority: {ticket.priority.title()}"
        ]
        
        # Add ticket metadata if available
        if ticket.customer_id:
            prompt_parts.append(f"Customer Id: {ticket.customer_id}")
        
        keywords = ticket.extract_keywords()
        if keywords:
            prompt_parts.append(f"Detected Topics: {', '.join(keywords)}")
        
        # Add context from retrieved documents
        if rag_context.retrieved_documents:
            prompt_parts.append(f"\nRelevant Documentation (Average Relevance: {rag_context.average_similarity:.3f}):")
            
            for i, doc in enumerate(rag_context.retrieved_documents[:3], 1):
                prompt_parts.append(
                    f"\n[Document {i} - {doc.document_type.title()} - Relevance: {doc.similarity_score:.3f}]"
                )
                prompt_parts.append(doc.content)
        else:
            prompt_parts.append("\nNo specific documentation was found for this query.")
        
        # Add response requirements
        prompt_parts.append(
            "\nPlease provide a helpful response that includes:"
            "\n1. A clear, empathetic answer to the customer's question"
            "\n2. Specific steps they should take (if applicable)"
            "\n3. Any relevant policies or procedures"
            "\n4. Estimated timeframes when possible"
            "\n5. Contact information for further assistance if needed"
        )
        
        return "\n".join(prompt_parts)
    
    def _determine_action_required(self, response_text: str, ticket: SupportTicket, rag_context: RAGContext) -> str:
        """Determine the required action based on response content and context."""
        
        response_lower = response_text.lower()
        keywords = ticket.extract_keywords()
        
        # High priority escalation indicators
        escalation_indicators = [
            "legal", "fraud", "compliance", "law enforcement", "court order",
            "urgent", "emergency", "critical", "immediate attention"
        ]
        
        if any(indicator in response_lower for indicator in escalation_indicators):
            return "escalate_to_management"
        
        # Domain suspension related
        if any(term in response_lower for term in ["suspension", "suspended", "reactivate", "whois"]):
            if any(term in response_lower for term in ["abuse", "policy violation", "malware"]):
                return "escalate_to_abuse_team"
            elif "whois" in response_lower or "contact information" in response_lower:
                return "update_whois"
            else:
                return "escalate_to_abuse_team"
        
        # Billing related
        if any(term in response_lower for term in ["billing", "payment", "refund", "charge"]):
            return "contact_billing"
        
        # Technical issues
        if any(term in keywords for term in ["technical", "dns", "email", "ssl"]):
            if any(term in response_lower for term in ["complex", "investigate", "server-level"]):
                return "escalate_to_technical"
            else:
                return "no_action"
        
        # Identity verification needed
        if any(term in response_lower for term in ["verify", "identification", "confirm identity"]):
            return "verify_identity"
        
        # Documentation needed
        if any(term in response_lower for term in ["documentation", "documents", "provide proof"]):
            return "provide_documentation"
        
        # Email check needed
        if any(term in response_lower for term in ["check email", "spam folder", "notification"]):
            return "check_email"
        
        # Processing time mentioned
        if any(term in response_lower for term in ["processing", "under review", "takes time"]):
            return "wait_for_processing"
        
        # Follow-up needed
        if any(term in response_lower for term in ["follow up", "contact you", "check back"]):
            return "create_follow_up"
        
        # Customer contact needed
        if any(term in response_lower for term in ["call you", "contact customer", "reach out"]):
            return "contact_customer"
        
        # Low confidence in retrieval suggests escalation
        if rag_context.average_similarity < 0.4:
            return "escalate_to_technical"
        
        # Default to no action for informational responses
        return "no_action"
    
    async def generate_support_response(
        self,
        ticket: SupportTicket,
        rag_context: RAGContext,
        custom_instructions: Optional[str] = None
    ) -> MCPResponse:
        """Generate intelligent support response using LLM.
        
        Args:
            ticket: Support ticket with customer query
            rag_context: Context from document retrieval
            custom_instructions: Additional instructions for the LLM
            
        Returns:
            MCPResponse with generated answer and metadata
        """
        
        start_time = time.time()
        self.usage_stats["total_requests"] += 1
        
        try:
            logger.info(f"🤖 Generating response for ticket: {ticket.ticket_id}")
            
            # Build prompts
            system_prompt = self._build_system_prompt(ticket)
            user_prompt = self._build_user_prompt(ticket, rag_context)
            
            # Add custom instructions if provided
            if custom_instructions:
                system_prompt += f"\n\nAdditional Instructions: {custom_instructions}"
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Estimate token count (rough estimation)
            estimated_tokens = sum(len(msg["content"].split()) * 1.3 for msg in messages)
            context_limit = self.model_specs.get(self.model, {}).get("context_window", 4096)
            
            if estimated_tokens > context_limit * 0.8:  # Leave room for response
                logger.warning(f"Prompt may be too long ({estimated_tokens} estimated tokens)")
                # Truncate context if needed
                user_prompt = user_prompt[:int(context_limit * 0.6 * 4)]  # Rough char to token conversion
                messages[1]["content"] = user_prompt
            
            # Generate response
            logger.debug("Calling OpenAI API...")
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            # Extract response
            response_text = completion.choices[0].message.content.strip()
            
            # Update usage stats
            if completion.usage:
                self.usage_stats["total_input_tokens"] += completion.usage.prompt_tokens
                self.usage_stats["total_output_tokens"] += completion.usage.completion_tokens
                
                # Calculate cost estimate
                model_spec = self.model_specs.get(self.model, {})
                input_cost = (completion.usage.prompt_tokens / 1000) * model_spec.get("cost_per_1k_input_tokens", 0)
                output_cost = (completion.usage.completion_tokens / 1000) * model_spec.get("cost_per_1k_output_tokens", 0)
                self.usage_stats["total_cost"] += input_cost + output_cost
            
            self.usage_stats["successful_requests"] += 1
            
            # Determine required action
            action_required = self._determine_action_required(response_text, ticket, rag_context)
            
            # Calculate confidence based on context quality and response characteristics
            base_confidence = min(rag_context.average_similarity + 0.2, 1.0)
            
            # Adjust confidence based on response characteristics
            if len(response_text) < 50:
                base_confidence *= 0.7  # Very short responses are less confident
            elif "I don't know" in response_text or "uncertain" in response_text.lower():
                base_confidence *= 0.5  # Explicit uncertainty
            elif action_required in ["escalate_to_technical", "escalate_to_management"]:
                base_confidence *= 0.6  # Escalations indicate uncertainty
            
            confidence_score = max(base_confidence, 0.1)  # Minimum confidence
            
            # Create MCP response
            response = MCPResponse(
                answer=response_text,
                action_required=action_required,
                confidence_score=confidence_score,
                priority_level=ticket.priority if ticket.priority in ["high", "urgent"] else None
            )
            
            # Add references from context (will trigger automatic reference setting)
            # The response model will automatically set references based on content
            
            # Add LLM-specific metadata
            response.additional_context["llm_metadata"] = {
                "model": self.model,
                "temperature": self.temperature,
                "input_tokens": completion.usage.prompt_tokens if completion.usage else None,
                "output_tokens": completion.usage.completion_tokens if completion.usage else None,
                "generation_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Update average response time
            response_time = time.time() - start_time
            if self.usage_stats["successful_requests"] == 1:
                self.usage_stats["average_response_time"] = response_time
            else:
                # Running average
                n = self.usage_stats["successful_requests"]
                current_avg = self.usage_stats["average_response_time"]
                self.usage_stats["average_response_time"] = (current_avg * (n-1) + response_time) / n
            
            self.usage_stats["last_request_time"] = datetime.now().isoformat()
            
            logger.info(f"✅ Response generated in {response_time:.2f}s")
            logger.info(f"🎯 Action determined: {action_required}")
            logger.info(f"📊 Confidence: {confidence_score:.3f}")
            
            return response
            
        except Exception as e:
            self.usage_stats["failed_requests"] += 1
            logger.error(f"❌ Failed to generate response: {e}")
            
            # Return error response
            error_response = MCPResponse(
                answer=f"I apologize, but I encountered an error while processing your request. Please contact our support team directly for immediate assistance. Error details have been logged for review.",
                action_required="escalate_to_technical",
                confidence_score=0.0,
                priority_level="high"
            )
            
            error_response.escalation_reason = f"LLM generation error: {str(e)}"
            
            return error_response
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM service usage statistics."""
        
        stats = self.usage_stats.copy()
        
        # Add calculated metrics
        if stats["total_requests"] > 0:
            stats["success_rate"] = (stats["successful_requests"] / stats["total_requests"]) * 100
            stats["average_tokens_per_request"] = (
                (stats["total_input_tokens"] + stats["total_output_tokens"]) / stats["successful_requests"]
                if stats["successful_requests"] > 0 else 0
            )
            stats["average_cost_per_request"] = (
                stats["total_cost"] / stats["successful_requests"]
                if stats["successful_requests"] > 0 else 0
            )
        else:
            stats["success_rate"] = 0
            stats["average_tokens_per_request"] = 0
            stats["average_cost_per_request"] = 0
        
        stats["model_info"] = self.get_model_info()
        stats["timestamp"] = datetime.now().isoformat()
        
        return stats
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "last_request_time": None
        }
        logger.info("📊 Usage statistics reset")


# Test the LLM service if run directly
# Test the LLM service if run directly
if __name__ == "__main__":
    print("🧪 Testing LLM Service...")
    print("=" * 50)
    
    # Check if API key is available
    import os
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    async def run_tests():
        # Initialize service
        print("\n🤖 Initializing LLM Service")
        llm_service = LLMService(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=800
        )
        
        print(f"Model info: {llm_service.get_model_info()}")
        
        # Health check
        print("\n🏥 Testing Health Check")
        health = await llm_service.check_health()
        print(f"   Status: {health.status}")
        print(f"   Services: {health.services}")
        
        if not health.services.get("openai_api", False):
            print("\n❌ OpenAI API is not accessible!")
            return
        
        # Test response generation
        print("\n💬 Testing Response Generation")
        
        # Create test ticket with correct parameters
        from models.ticket import SupportTicket
        from models.rag import RAGContext, RetrievedDocument
        
        test_ticket = SupportTicket(
            ticket_text="My domain was suspended and I didn't receive any notification email. How can I reactivate it?",
            priority="high",
            customer_id="customer_123"  # ✅ Correct parameter name
        )
        
        print(f"   Test ticket created: {test_ticket.ticket_id}")
        print(f"   Keywords: {test_ticket.extract_keywords()}")
        
        # Create mock RAG context
        mock_doc = RetrievedDocument(
            document_id="test_doc_1",
            content="Domain suspension occurs when WHOIS information is incomplete or inaccurate according to ICANN requirements. To reactivate a suspended domain, customers must update their WHOIS contact details through the control panel, verify their email address, and contact our abuse team for manual review. The reactivation process typically takes 24-48 hours once all information is updated.",
            similarity_score=0.87,
            document_type="policy"
        )
        
        mock_context = RAGContext(
            query=test_ticket.ticket_text,
            retrieved_documents=[mock_doc],
            total_documents_searched=100
        )
        
        print(f"   Mock context created with {len(mock_context.retrieved_documents)} documents")
        print(f"   Average similarity: {mock_context.average_similarity:.3f}")
        
        try:
            response = await llm_service.generate_support_response(test_ticket, mock_context)
            
            print(f"   ✅ Response generated:")
            print(f"      Action: {response.action_required}")
            print(f"      Confidence: {response.confidence_score:.3f}")
            print(f"      Answer length: {len(response.answer)} chars")
            print(f"      References: {len(response.references)}")
            print(f"      Answer preview: {response.answer[:200]}...")
            
            # Show LLM metadata if available
            if "llm_metadata" in response.additional_context:
                llm_meta = response.additional_context["llm_metadata"]
                print(f"      Generation time: {llm_meta.get('generation_time_ms', 0):.1f}ms")
                print(f"      Input tokens: {llm_meta.get('input_tokens', 'unknown')}")
                print(f"      Output tokens: {llm_meta.get('output_tokens', 'unknown')}")
            
        except Exception as e:
            print(f"   ❌ Response generation failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test usage statistics
        print("\n📊 Testing Usage Statistics")
        stats = await llm_service.get_usage_stats()
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Total cost: ${stats['total_cost']:.4f}")
        print(f"   Average response time: {stats['average_response_time']:.2f}s")
        print(f"   Average tokens per request: {stats['average_tokens_per_request']:.0f}")
        
        # Test multiple response types
        print("\n🔄 Testing Different Ticket Types")
        test_cases = [
            ("Billing issue", "My payment failed but my card works fine elsewhere", "medium"),
            ("Technical issue", "My website is loading very slowly and emails are bouncing", "high"),
            ("General inquiry", "How do I transfer my domain to another registrar?", "low")
        ]
        
        for case_name, ticket_text, priority in test_cases:
            try:
                test_ticket = SupportTicket(ticket_text=ticket_text, priority=priority)
                mock_context = RAGContext(
                    query=ticket_text,
                    retrieved_documents=[mock_doc],  # Reuse same mock doc
                    total_documents_searched=50
                )
                
                response = await llm_service.generate_support_response(test_ticket, mock_context)
                print(f"   {case_name}: {response.action_required} (confidence: {response.confidence_score:.3f})")
                
            except Exception as e:
                print(f"   {case_name}: Failed - {e}")
        
        print("\n🎉 LLM Service is working correctly!")
        print("\n💡 Next steps:")
        print("   1. The LLM service can now generate intelligent responses")
        print("   2. Test the complete RAG service integration")
        print("   3. Create API endpoints for the system")
    
    # Run async tests
    asyncio.run(run_tests())

