#!/usr/bin/env python3
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
    print("OpenAI client not installed. Install with: pip install openai")
    raise

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.ticket import SupportTicket
from models.response import MCPResponse
from models.rag import RAGContext


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    
    def __init__(
        self,
        api_key: Optional[str] = API_KEY,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.3,
        timeout: float = 30.0
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.client = OpenAI(api_key=api_key) 
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
        
        logger.info(f"LLM service initialized with model: {model}")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            **self.model_specs.get(self.model, {})
        }
    
    def _build_system_prompt(self, ticket: SupportTicket) -> str:
        
        base_prompt = """You are an expert customer support agent for a domain registration and web hosting company. Your role is to provide helpful, accurate, and professional responses to customer inquiries. Provide SHORT, DIRECT responses.
RULES:
- Keep responses under 3 sentences when possible
- Give the most important information first
- Be specific and actionable
- Don't repeat the customer's problem back to them
SCOPE: Only answer questions related to:
- Domain management (registration, transfers, suspension, WHOIS)
- Website hosting and technical issues
- Email hosting and configuration
- Billing and payments
- DNS and SSL certificates
- Company policies and procedures

OUT-OF-SCOPE HANDLING:
If the question is NOT related to domain/hosting services (e.g., job applications, general tech advice, personal matters), respond with:
"I apologize, but I can only assist with domain and hosting related questions. Please contact our customer support team directly for other inquiries."
Key guidelines:
1. Be helpful, empathetic, and professional
2. Provide specific, actionable guidance
3. Reference company policies when relevant
4. Suggest appropriate next actions
5. Be concise but comprehensive
6. If uncertain, recommend contacting the appropriate team

You have access to company documentation and should base your responses on the provided context."""
        
        keywords = ticket.extract_keywords()
        
        if 'domain' in keywords or 'suspension' in keywords:
            base_prompt += "\n\nThis inquiry is domain-related. Focus on domain management, suspension issues, WHOIS compliance, and reactivation procedures."
        
        if 'billing' in keywords:
            base_prompt += "\n\nThis inquiry is billing-related. Focus on payment issues, refunds, account problems, and billing procedures."
        
        if 'technical' in keywords:
            base_prompt += "\n\nThis inquiry is technical. Focus on DNS, email, SSL, server issues, and technical troubleshooting."
    
        if ticket.priority in ["high", "urgent"]:
            base_prompt += "\n\nThis is a high-priority inquiry. Be extra attentive and suggest immediate action steps."
        
        return base_prompt
    
    def _build_user_prompt(self, ticket: SupportTicket, rag_context: RAGContext) -> str:
        
        prompt_parts = [
            "Please help with this customer support inquiry:",
            f"\nCustomer Query: {ticket.ticket_text}",
            f"\nPriority: {ticket.priority.title()}"
        ]
        
        if ticket.customer_id:
            prompt_parts.append(f"Customer Id: {ticket.customer_id}")
        
        keywords = ticket.extract_keywords()
        if keywords:
            prompt_parts.append(f"Detected Topics: {', '.join(keywords)}")
        
        if rag_context.retrieved_documents:
            prompt_parts.append(f"\nRelevant Documentation (Average Relevance: {rag_context.average_similarity:.3f}):")
            
            for i, doc in enumerate(rag_context.retrieved_documents[:3], 1):
                prompt_parts.append(
                    f"\n[Document {i} - {doc.document_type.title()} - Relevance: {doc.similarity_score:.3f}]"
                )
                prompt_parts.append(doc.content)
        else:
            prompt_parts.append("\nNo specific documentation was found for this query.")
        
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
        response_lower = response_text.lower()
        keywords = ticket.extract_keywords()
        escalation_indicators = [
            "legal", "fraud", "compliance", "law enforcement", "court order",
            "urgent", "emergency", "critical", "immediate attention"
        ]
        
        if any(indicator in response_lower for indicator in escalation_indicators):
            return "escalate_to_management"
    
        if any(term in response_lower for term in ["suspension", "suspended", "reactivate", "whois"]):
            if any(term in response_lower for term in ["abuse", "policy violation", "malware"]):
                return "escalate_to_abuse_team"
            elif "whois" in response_lower or "contact information" in response_lower:
                return "update_whois"
            else:
                return "escalate_to_abuse_team"
        
        if any(term in response_lower for term in ["billing", "payment", "refund", "charge"]):
            return "contact_billing"
        
        if any(term in keywords for term in ["technical", "dns", "email", "ssl"]):
            if any(term in response_lower for term in ["complex", "investigate", "server-level"]):
                return "escalate_to_technical"
            else:
                return "no_action"
       
        if any(term in response_lower for term in ["verify", "identification", "confirm identity"]):
            return "verify_identity"
      
        if any(term in response_lower for term in ["documentation", "documents", "provide proof"]):
            return "provide_documentation"
    
        if any(term in response_lower for term in ["check email", "spam folder", "notification"]):
            return "check_email"
     
        if any(term in response_lower for term in ["processing", "under review", "takes time"]):
            return "wait_for_processing"
    
        if any(term in response_lower for term in ["follow up", "contact you", "check back"]):
            return "create_follow_up"
       
        if any(term in response_lower for term in ["call you", "contact customer", "reach out"]):
            return "contact_customer"
        
        if rag_context.average_similarity < 0.4:
            return "escalate_to_technical"
        
        return "no_action"
    
    async def generate_support_response(
        self,
        ticket: SupportTicket,
        rag_context: RAGContext,
        custom_instructions: Optional[str] = None
    ) -> MCPResponse:
        
        start_time = time.time()
        self.usage_stats["total_requests"] += 1
        
        try:
            logger.info(f"Generating response for ticket: {ticket.ticket_id}")
            system_prompt = self._build_system_prompt(ticket)
            user_prompt = self._build_user_prompt(ticket, rag_context)
           
            if custom_instructions:
                system_prompt += f"\n\nAdditional Instructions: {custom_instructions}"
    
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        
            estimated_tokens = sum(len(msg["content"].split()) * 1.3 for msg in messages)
            context_limit = self.model_specs.get(self.model, {}).get("context_window", 4096)
        
            if estimated_tokens > context_limit * 0.8:  
                logger.warning(f"Prompt may be too long ({estimated_tokens} estimated tokens)")
                user_prompt = user_prompt[:int(context_limit * 0.6 * 4)]  
                messages[1]["content"] = user_prompt
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            response_text = completion.choices[0].message.content.strip()
        
            if completion.usage:
                self.usage_stats["total_input_tokens"] += completion.usage.prompt_tokens
                self.usage_stats["total_output_tokens"] += completion.usage.completion_tokens
                

                
                model_spec = self.model_specs.get(self.model, {})
                input_cost = (completion.usage.prompt_tokens / 1000) * model_spec.get("cost_per_1k_input_tokens", 0)
                output_cost = (completion.usage.completion_tokens / 1000) * model_spec.get("cost_per_1k_output_tokens", 0)
                self.usage_stats["total_cost"] += input_cost + output_cost
            
            self.usage_stats["successful_requests"] += 1
        
            action_required = self._determine_action_required(response_text, ticket, rag_context)
        
            base_confidence = min(rag_context.average_similarity + 0.2, 1.0)
            

            
            if len(response_text) < 50:

                base_confidence *= 0.7  
            elif "I don't know" in response_text or "uncertain" in response_text.lower():

                base_confidence *= 0.5  
            elif action_required in ["escalate_to_technical", "escalate_to_management"]:

                base_confidence *= 0.6  
            

            confidence_score = max(base_confidence, 0.1)  
            
            response = MCPResponse(
                answer=response_text,
                action_required=action_required,
                confidence_score=confidence_score,
                priority_level=ticket.priority if ticket.priority in ["high", "urgent"] else None
            )
            
            response.additional_context["llm_metadata"] = {
                "model": self.model,
                "temperature": self.temperature,
                "input_tokens": completion.usage.prompt_tokens if completion.usage else None,
                "output_tokens": completion.usage.completion_tokens if completion.usage else None,
                "generation_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            

            
            response_time = time.time() - start_time
            if self.usage_stats["successful_requests"] == 1:
                self.usage_stats["average_response_time"] = response_time
            else:

                
                n = self.usage_stats["successful_requests"]
                current_avg = self.usage_stats["average_response_time"]
                self.usage_stats["average_response_time"] = (current_avg * (n-1) + response_time) / n
            
            self.usage_stats["last_request_time"] = datetime.now().isoformat()
            
            logger.info(f" Response generated in {response_time:.2f}s")
            logger.info(f" Action determined: {action_required}")
            
            return response
            
        except Exception as e:
            self.usage_stats["failed_requests"] += 1
            logger.error(f" Failed to generate response: {e}")
            
            error_response = MCPResponse(
                answer=f"I apologize, but I encountered an error while processing your request. Please contact our support team directly for immediate assistance. Error details have been logged for review.",
                action_required="escalate_to_technical",
                confidence_score=0.0,
                priority_level="high"
            )
            
            error_response.escalation_reason = f"LLM generation error: {str(e)}"
            
            return error_response
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        
        stats = self.usage_stats.copy()
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

