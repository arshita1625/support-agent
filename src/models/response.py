#!/usr/bin/env python3
"""Response models using Python dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import json
import re
@dataclass
class MCPResponse:
    """Model Context Protocol compliant response for support tickets."""
    
    # Required fields
    answer: str
    action_required: Literal[
        "no_action",
        "escalate_to_abuse_team", 
        "escalate_to_billing",
        "escalate_to_technical",
        "escalate_to_management",
        "update_whois",
        "contact_billing",
        "verify_identity",
        "provide_documentation",
        "check_email",
        "wait_for_processing",
        "contact_customer",
        "create_follow_up"
    ]
    
    # Optional fields with defaults
    references: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    estimated_resolution_time: Optional[str] = None
    priority_level: Optional[Literal["low", "medium", "high", "urgent"]] = None
    follow_up_required: bool = False
    customer_notification_required: bool = True
    
    # Enhanced fields for better support
    next_steps: List[str] = field(default_factory=list)
    relevant_policies: List[str] = field(default_factory=list)
    contact_information: Dict[str, str] = field(default_factory=dict)
    escalation_reason: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and enhance response data after initialization."""
        self._validate_answer()
        self._validate_action_required()
        self._validate_references()
        self._validate_confidence_score()
        self._validate_priority_level()
        self._extract_additional_info()
        
        # Auto-set references if none provided
        if not self.references:
            self._set_references_from_content()
    
    def _validate_answer(self):
        """Validate answer content and quality."""
        if not isinstance(self.answer, str):
            raise ValueError('answer must be a string')
        
        cleaned_answer = self.answer.strip()
        if len(cleaned_answer) < 20:
            raise ValueError('Answer must be at least 20 characters long')
        
        word_count = len(cleaned_answer.split())
        if word_count < 10:
            raise ValueError('Answer must contain at least 10 words')
        
        # Check for placeholder text
        placeholders = [
            '[placeholder]', 'TODO', 'FIXME', 'XXX', 'TBD', 
            '{insert}', '{placeholder}', 'lorem ipsum', 'sample text'
        ]
        answer_lower = cleaned_answer.lower()
        for placeholder in placeholders:
            if placeholder.lower() in answer_lower:
                raise ValueError(f'Answer contains placeholder text: {placeholder}')
        
        self.answer = cleaned_answer
    
    def _validate_action_required(self):
        """Validate the action_required field."""
        valid_actions = [
            "no_action", "escalate_to_abuse_team", "escalate_to_billing",
            "escalate_to_technical", "escalate_to_management", "update_whois",
            "contact_billing", "verify_identity", "provide_documentation",
            "check_email", "wait_for_processing", "contact_customer", "create_follow_up"
        ]
        
        if self.action_required not in valid_actions:
            raise ValueError(f'action_required must be one of: {valid_actions}')
    
    def _validate_references(self):
        """Validate references list."""
        if not isinstance(self.references, list):
            raise ValueError('references must be a list')
        
        cleaned_references = []
        for ref in self.references:
            if not isinstance(ref, str):
                raise ValueError('All references must be strings')
            
            cleaned_ref = ref.strip()
            if len(cleaned_ref) < 5:
                raise ValueError('References must be at least 5 characters long')
            
            cleaned_references.append(cleaned_ref)
        
        self.references = cleaned_references
    
    def _validate_confidence_score(self):
        """Validate confidence score if provided."""
        if self.confidence_score is not None:
            if not isinstance(self.confidence_score, (int, float)):
                raise ValueError('confidence_score must be a number')
            
            if not (0.0 <= self.confidence_score <= 1.0):
                raise ValueError('confidence_score must be between 0.0 and 1.0')
    
    def _validate_priority_level(self):
        """Validate priority level if provided."""
        if self.priority_level is not None:
            valid_priorities = ["low", "medium", "high", "urgent"]
            if self.priority_level not in valid_priorities:
                raise ValueError(f'priority_level must be one of: {valid_priorities}')
    
    def _extract_additional_info(self):
        """Extract additional information from the answer content."""
        answer_lower = self.answer.lower()
        
        # Extract time estimates if not explicitly provided
        if not self.estimated_resolution_time:
            time_patterns = [
                r'(\d+)-(\d+)\s+(hours?|days?|weeks?)',
                r'(\d+)\s+(hours?|days?|weeks?|minutes?)',
                r'(immediately|right away|asap)',
                r'(\d+)\s+(business\s+days?)'
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, answer_lower)
                if match:
                    if isinstance(match.groups(), tuple) and len(match.groups()) > 1:
                        self.estimated_resolution_time = ' '.join(match.groups())
                    else:
                        self.estimated_resolution_time = match.group(0)
                    break
        
        # Extract contact information
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, self.answer)
        
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, self.answer)
        
        if emails:
            self.contact_information['emails'] = emails
        if phones:
            self.contact_information['phones'] = [''.join(phone) for phone in phones]
        
        # Determine if follow-up is needed based on content
        follow_up_indicators = [
            'follow up', 'check back', 'contact you', 'let you know',
            'update you', 'monitor', 'review in', 'schedule'
        ]
        
        if any(indicator in answer_lower for indicator in follow_up_indicators):
            self.follow_up_required = True
    
    def _set_references_from_content(self):
        """Set references based on answer content using your actual documents."""
        
        answer_lower = self.answer.lower()
        references = []
        
        # Domain suspension related
        if any(term in answer_lower for term in ['domain suspend', 'suspended', 'suspension', 'whois']):
            references.extend([
                "Domain Management Policy v2.3 - Section 4.1: Suspension Triggers",
                "Domain Management Policy v2.3 - Section 4.2: Suspension Process",
                "Domain Management Policy v2.3 - Section 4.3: Reactivation Requirements"
            ])
        
        # WHOIS specific issues
        if any(term in answer_lower for term in ['whois', 'contact information', 'registrant', 'verify email']):
            references.extend([
                "Domain Management Policy v2.3 - Section 4.1.A: WHOIS Compliance Issues",
                "FAQ: Why was my domain suspended without notice?"
            ])
        
        # Policy violations and malware
        if any(term in answer_lower for term in ['malware', 'phishing', 'policy violation', 'abuse', 'content']):
            references.extend([
                "Domain Management Policy v2.3 - Section 4.1.B: Terms of Service Violations",
                "Acceptable Use Policy - Section 6.1: Malicious Software",
                "FAQ: My domain was suspended for malware but my site is clean now"
            ])
        
        # Billing and payment issues
        if any(term in answer_lower for term in ['payment', 'billing', 'charge', 'invoice', 'fee', 'refund']):
            references.extend([
                "Billing and Payment Terms v1.8 - Section 2.1: Grace Period Policy",
                "Billing and Payment Terms v1.8 - Section 2.2: Acceptable Payment Methods",
                "FAQ: My payment failed but my card works everywhere else"
            ])
        
        # Chargeback specific
        if any(term in answer_lower for term in ['chargeback', 'dispute', 'bank dispute']):
            references.extend([
                "Billing and Payment Terms v1.8 - Section 2.4: Chargeback Protection",
                "FAQ: Why did my account get suspended for a chargeback?"
            ])
        
        # Refund related
        if any(term in answer_lower for term in ['refund', 'money back', 'unused time', 'cancel']):
            references.extend([
                "Billing and Payment Terms v1.8 - Section 2.3: Refund Policy",
                "FAQ: Can I get a refund for unused time on my hosting plan?"
            ])
        
        # Technical issues
        if any(term in answer_lower for term in ['dns', 'nameserver', 'propagation', 'technical']):
            references.extend([
                "Escalation Matrix and Response Procedures - DNS Propagation Problems",
                "Escalation Matrix and Response Procedures - Escalation Criteria"
            ])
        
        # Email issues
        if any(term in answer_lower for term in ['email', 'mail', 'mx record', 'smtp', 'delivery']):
            references.extend([
                "Escalation Matrix and Response Procedures - Email Delivery Issues",
                "FAQ: I can't receive emails but can send them fine"
            ])
        
        # SSL certificate issues
        if any(term in answer_lower for term in ['ssl', 'certificate', 'https', 'security certificate']):
            references.extend([
                "Escalation Matrix and Response Procedures - SSL Certificate Problems",
                "FAQ: My SSL certificate shows as invalid"
            ])
        
        # Performance and website issues
        if any(term in answer_lower for term in ['slow', 'performance', 'loading', 'website', 'optimize']):
            references.extend([
                "FAQ: My website is loading slowly. What can I do?",
                "Escalation Matrix and Response Procedures - Response Time Commitments"
            ])
        
        # Domain transfer
        if any(term in answer_lower for term in ['transfer', 'move domain', 'registrar']):
            references.extend([
                "FAQ: Can I transfer my domain while it's suspended?"
            ])
        
        # Appeals process
        if any(term in answer_lower for term in ['appeal', 'dispute', 'contest', 'disagree', 'error']):
            references.extend([
                "Domain Management Policy v2.3 - Section 4.4: Appeals Process"
            ])
        
        # Reactivation process
        if any(term in answer_lower for term in ['reactivate', 'restore', 'unsuspend', '24-48 hours']):
            references.extend([
                "Domain Management Policy v2.3 - Section 4.3: Reactivation Requirements",
                "FAQ: How long does domain reactivation take?"
            ])
        
        # Security and abuse policies
        if any(term in answer_lower for term in ['spam', 'malicious', 'prohibited', 'acceptable use']):
            references.extend([
                "Acceptable Use Policy - Section 6.2: Spam and Unsolicited Communications",
                "Acceptable Use Policy - Section 6.3: Content Restrictions"
            ])
        
        # Remove duplicates and limit to 5 references
        unique_references = list(dict.fromkeys(references))[:5]
        
        # If no specific references found, use general ones
        if not unique_references:
            if any(term in answer_lower for term in ['domain', 'website', 'site']):
                unique_references = ["Domain Management Policy v2.3 - General Provisions"]
            elif any(term in answer_lower for term in ['payment', 'billing']):
                unique_references = ["Billing and Payment Terms v1.8 - General Terms"]
            elif any(term in answer_lower for term in ['technical', 'support']):
                unique_references = ["Escalation Matrix and Response Procedures - General Support"]
            else:
                unique_references = ["General Support Documentation"]
        
        self.references = unique_references
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResponse':
        """Create response from dictionary."""
        return cls(**data)
    
    def add_reference(self, reference: str) -> None:
        """Add a reference to the response."""
        if not reference or not reference.strip():
            raise ValueError('Reference cannot be empty')
        
        cleaned_ref = reference.strip()
        if len(cleaned_ref) < 5:
            raise ValueError('Reference must be at least 5 characters long')
        
        if cleaned_ref not in self.references:
            self.references.append(cleaned_ref)
    
    def add_next_step(self, step: str) -> None:
        """Add a next step to the response."""
        if not step or not step.strip():
            raise ValueError('Step cannot be empty')
        
        cleaned_step = step.strip()
        if cleaned_step not in self.next_steps:
            self.next_steps.append(cleaned_step)
    
    def set_confidence(self, confidence: float) -> None:
        """Set confidence score with validation."""
        if not isinstance(confidence, (int, float)):
            raise ValueError('Confidence must be a number')
        
        if not (0.0 <= confidence <= 1.0):
            raise ValueError('Confidence must be between 0.0 and 1.0')
        
        self.confidence_score = float(confidence)
    
    def requires_escalation(self) -> bool:
        """Check if response requires escalation."""
        escalation_actions = [
            "escalate_to_abuse_team",
            "escalate_to_billing", 
            "escalate_to_technical",
            "escalate_to_management"
        ]
        return self.action_required in escalation_actions
    
    def is_high_priority(self) -> bool:
        """Check if response is high priority."""
        return (
            self.priority_level in ["high", "urgent"] or
            self.requires_escalation() or
            (self.confidence_score is not None and self.confidence_score < 0.3)
        )
    
    def get_action_description(self) -> str:
        """Get human-readable description of the required action."""
        action_descriptions = {
            "no_action": "No further action required",
            "escalate_to_abuse_team": "Escalate to abuse team for policy violation review",
            "escalate_to_billing": "Escalate to billing department for payment issues",
            "escalate_to_technical": "Escalate to technical team for complex issues",
            "escalate_to_management": "Escalate to management for high-priority cases",
            "update_whois": "Customer needs to update WHOIS information",
            "contact_billing": "Customer should contact billing department",
            "verify_identity": "Customer identity verification required",
            "provide_documentation": "Customer must provide supporting documentation",
            "check_email": "Customer should check email for notifications",
            "wait_for_processing": "Wait for system processing to complete",
            "contact_customer": "Agent should contact customer directly",
            "create_follow_up": "Create follow-up task for future action"
        }
        
        return action_descriptions.get(self.action_required, "Unknown action")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the response for quick review."""
        return {
            "action_required": self.action_required,
            "action_description": self.get_action_description(),
            "priority": self.priority_level or "not specified",
            "confidence": self.confidence_score or "not specified",
            "estimated_time": self.estimated_resolution_time or "not specified",
            "requires_escalation": self.requires_escalation(),
            "follow_up_required": self.follow_up_required,
            "reference_count": len(self.references),
            "next_steps_count": len(self.next_steps)
        }

@dataclass
class HealthStatus:
    """System health check response."""
    
    status: Literal["healthy", "degraded", "unhealthy"]
    services: Dict[str, bool] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate health status."""
        valid_statuses = ["healthy", "degraded", "unhealthy"]
        if self.status not in valid_statuses:
            raise ValueError(f'status must be one of: {valid_statuses}')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        health_dict = asdict(self)
        health_dict['timestamp'] = self.timestamp.isoformat()
        return health_dict
    
    def add_service_status(self, service_name: str, is_healthy: bool) -> None:
        """Add service health status."""
        self.services[service_name] = is_healthy
    
    def is_all_services_healthy(self) -> bool:
        """Check if all services are healthy."""
        return all(self.services.values()) if self.services else True

@dataclass
class ErrorResponse:
    """Standardized error response."""
    
    error_code: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate error response."""
        if not self.error_code or not self.error_code.strip():
            raise ValueError('error_code cannot be empty')
        
        if not self.message or not self.message.strip():
            raise ValueError('message cannot be empty')
        
        self.error_code = self.error_code.strip()
        self.message = self.message.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        error_dict = asdict(self)
        error_dict['timestamp'] = self.timestamp.isoformat()
        return error_dict

# Test the models if run directly
if __name__ == "__main__":
    print("🧪 Testing Response Models...")
    
    # Test MCP Response
    response = MCPResponse(
        answer="Your domain was suspended due to incomplete WHOIS information. To reactivate it, please update your WHOIS contact details through your control panel, verify your email address, and contact our abuse team for manual review. The process typically takes 24-48 hours.",
        references=[
            "Policy: Domain Suspension Guidelines, Section 4.2",
            "Procedure: WHOIS Update Requirements, Section 2.1"
        ],
        action_required="escalate_to_abuse_team",
        confidence_score=0.89,
        estimated_resolution_time="24-48 hours",
        priority_level="medium"
    )
    
    print(f"✅ MCP Response created successfully")
    print(f"   Action: {response.action_required}")
    print(f"   Confidence: {response.confidence_score}")
    print(f"   References: {len(response.references)}")
    print(f"   Requires escalation: {response.requires_escalation()}")
    
    # Test Health Status
    health = HealthStatus(
        status="healthy",
        services={
            "qdrant": True,
            "openai": True,
            "embedding_model": True
        }
    )
    print(f"✅ Health Status: {health.status}")
    print(f"   All services healthy: {health.is_all_services_healthy()}")
    
    # Test Error Response
    error = ErrorResponse(
        error_code="VALIDATION_ERROR",
        message="Invalid ticket content format",
        details={"field": "ticket_text", "constraint": "min_length"}
    )
    print(f"✅ Error Response: {error.error_code}")
    
    # Test serialization
    response_dict = response.to_dict() 
    response_restored = MCPResponse.from_dict(response_dict)
    print(f"✅ Serialization works - Restored confidence: {response_restored.confidence_score}")
    
    print("🎉 All response models working correctly!")
