#!/usr/bin/env python3
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import json
import re
@dataclass
class MCPResponse:
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
    references: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    estimated_resolution_time: Optional[str] = None
    priority_level: Optional[Literal["low", "medium", "high", "urgent"]] = None
    follow_up_required: bool = False
    customer_notification_required: bool = True
    next_steps: List[str] = field(default_factory=list)
    relevant_policies: List[str] = field(default_factory=list)
    contact_information: Dict[str, str] = field(default_factory=dict)
    escalation_reason: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._validate_answer()
        self._validate_action_required()
        self._validate_references()
        self._validate_confidence_score()
        self._validate_priority_level()
        self._extract_additional_info()
        if not self.references:
            self._set_references_from_content()
    
    def _validate_answer(self):
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
        valid_actions = [
            "no_action", "escalate_to_abuse_team", "escalate_to_billing",
            "escalate_to_technical", "escalate_to_management", "update_whois",
            "contact_billing", "verify_identity", "provide_documentation",
            "check_email", "wait_for_processing", "contact_customer", "create_follow_up"
        ]
        
        if self.action_required not in valid_actions:
            raise ValueError(f'action_required must be one of: {valid_actions}')
    def _validate_references(self):
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
        if self.confidence_score is not None:
            if not isinstance(self.confidence_score, (int, float)):
                raise ValueError('confidence_score must be a number')
            
            if not (0.0 <= self.confidence_score <= 1.0):
                raise ValueError('confidence_score must be between 0.0 and 1.0')
    
    def _validate_priority_level(self):
        if self.priority_level is not None:
            valid_priorities = ["low", "medium", "high", "urgent"]
            if self.priority_level not in valid_priorities:
                raise ValueError(f'priority_level must be one of: {valid_priorities}')
    
    def _extract_additional_info(self):
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
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, self.answer)
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, self.answer)
        
        if emails:
            self.contact_information['emails'] = emails
        if phones:
            self.contact_information['phones'] = [''.join(phone) for phone in phones]
    
        follow_up_indicators = [
            'follow up', 'check back', 'contact you', 'let you know',
            'update you', 'monitor', 'review in', 'schedule'
        ]
        
        if any(indicator in answer_lower for indicator in follow_up_indicators):
            self.follow_up_required = True
    
    def _set_references_from_content(self):
        answer_lower = self.answer.lower()
        references = []
        if any(term in answer_lower for term in ['domain suspend', 'suspended', 'suspension', 'whois']):
            references.extend([
                "Domain Management Policy v2.3 - Section 4.1: Suspension Triggers",
                "Domain Management Policy v2.3 - Section 4.2: Suspension Process",
                "Domain Management Policy v2.3 - Section 4.3: Reactivation Requirements"
            ])
        if any(term in answer_lower for term in ['whois', 'contact information', 'registrant', 'verify email']):
            references.extend([
                "Domain Management Policy v2.3 - Section 4.1.A: WHOIS Compliance Issues",
                "FAQ: Why was my domain suspended without notice?"
            ])
        
        if any(term in answer_lower for term in ['malware', 'phishing', 'policy violation', 'abuse', 'content']):
            references.extend([
                "Domain Management Policy v2.3 - Section 4.1.B: Terms of Service Violations",
                "Acceptable Use Policy - Section 6.1: Malicious Software",
                "Website Security Policy- Section 7.2: Malware Detection",
                "Website Security Policy- Section 7.3: DDoS Protection",
                "FAQ: My domain was suspended for malware but my site is clean now"
            ])
        if any(term in answer_lower for term in ['payment', 'billing', 'charge', 'invoice', 'fee']):
            references.extend([
                "Billing and Payment Terms v1.8 - Section 2.1: Grace Period Policy",
                "Billing and Payment Terms v1.8 - Section 2.2: Acceptable Payment Methods",
                "FAQ: My payment failed but my card works everywhere else"
            ])
        
        if any(term in answer_lower for term in ['chargeback', 'dispute', 'bank dispute']):
            references.extend([
                "Billing and Payment Terms v1.8 - Section 2.4: Chargeback Protection",
                "FAQ: Why did my account get suspended for a chargeback?"
            ])
    
        if any(term in answer_lower for term in ['refund', 'money back', 'unused time', 'cancel']):
            references.extend([
                "Billing and Payment Terms v1.8 - Section 2.3: Refund Policy",
                "FAQ: Can I get a refund for unused time on my hosting plan?"
            ])
        
        if any(term in answer_lower for term in ['dns', 'nameserver', 'propagation', 'technical']):
            references.extend([
                "Escalation Matrix and Response Procedures - DNS Propagation Problems",
                "Escalation Matrix and Response Procedures - Escalation Criteria"
            ])
        
        if any(term in answer_lower for term in ['email', 'mail', 'mx record', 'smtp', 'delivery']):
            references.extend([
                "Escalation Matrix and Response Procedures - Email Delivery Issues",
                "Email Hosting Policy- Section 8",
                "FAQ: I can't receive emails but can send them fine"
            ])
        
        if any(term in answer_lower for term in ['ssl', 'certificate', 'https', 'security certificate']):
            references.extend([
                "Escalation Matrix and Response Procedures - SSL Certificate Problems",
                "Website Security Policy- Section 7.1: SSL Certificate Requirements ",
                "FAQ: My SSL certificate shows as invalid"
            ])
        
        if any(term in answer_lower for term in ['slow', 'performance', 'loading', 'website', 'optimize']):
            references.extend([
                "FAQ: My website is loading slowly. What can I do?",
                "Escalation Matrix and Response Procedures - Response Time Commitments"
            ])
        
        if any(term in answer_lower for term in ['transfer', 'move domain', 'registrar']):
            references.extend([
                "FAQ: Can I transfer my domain while it's suspended?"
            ])
        
        if any(term in answer_lower for term in ['appeal', 'dispute', 'contest', 'disagree', 'error']):
            references.extend([
                "Domain Management Policy v2.3 - Section 4.4: Appeals Process"
            ])
        
        if any(term in answer_lower for term in ['reactivate', 'restore', 'unsuspend', '24-48 hours']):
            references.extend([
                "Domain Management Policy v2.3 - Section 4.3: Reactivation Requirements",
                "Backup Schedule v2.3 - Section 11",
                "Recovery Procedures - Section 12",
                "FAQ: How long does domain reactivation take?"
            ])
        
        if any(term in answer_lower for term in ['spam', 'malicious', 'prohibited', 'acceptable use']):
            references.extend([
                "Acceptable Use Policy - Section 6.1: Malicious Software",
                "Acceptable Use Policy - Section 6.2: Spam and Unsolicited Communications",
                "Acceptable Use Policy - Section 6.3: Content Restrictions"
            ])
        
        unique_references = list(dict.fromkeys(references))[:5]

        if not unique_references:
            if any(term in answer_lower for term in ['domain', 'website', 'site']):
                unique_references = ["Domain Management Policy v2.3 - General Provisions"]
            elif any(term in answer_lower for term in ['payment', 'billing']):
                unique_references = ["Billing and Payment Terms v1.8 - General Terms"]
            elif any(term in answer_lower for term in ['technical', 'support']):
                unique_references = ["Escalation Matrix and Response Procedures - General Support"]
            elif any(term in answer_lower for term in ['memory', 'storage']):
                unique_references = ["Storage Quotas- Section 9"]
            else:
                unique_references = ["General Support Documentation"]
        
        self.references = unique_references
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResponse':
        return cls(**data)
    
    def add_reference(self, reference: str) -> None:
        if not reference or not reference.strip():
            raise ValueError('Reference cannot be empty')
        
        cleaned_ref = reference.strip()
        if len(cleaned_ref) < 5:
            raise ValueError('Reference must be at least 5 characters long')
        
        if cleaned_ref not in self.references:
            self.references.append(cleaned_ref)
    
    def add_next_step(self, step: str) -> None:
        if not step or not step.strip():
            raise ValueError('Step cannot be empty')
        
        cleaned_step = step.strip()
        if cleaned_step not in self.next_steps:
            self.next_steps.append(cleaned_step)
    
    def set_confidence(self, confidence: float) -> None:
        if not isinstance(confidence, (int, float)):
            raise ValueError('Confidence must be a number')
        
        if not (0.0 <= confidence <= 1.0):
            raise ValueError('Confidence must be between 0.0 and 1.0')
        
        self.confidence_score = float(confidence)
    
    def requires_escalation(self) -> bool:
        escalation_actions = [
            "escalate_to_abuse_team",
            "escalate_to_billing", 
            "escalate_to_technical",
            "escalate_to_management"
        ]
        return self.action_required in escalation_actions
    
    def is_high_priority(self) -> bool:
        return (
            self.priority_level in ["high", "urgent"] or
            self.requires_escalation() or
            (self.confidence_score is not None and self.confidence_score < 0.3)
        )
    
    def get_action_description(self) -> str:
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
class ErrorResponse:
    
    error_code: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.error_code or not self.error_code.strip():
            raise ValueError('error_code cannot be empty')
        
        if not self.message or not self.message.strip():
            raise ValueError('message cannot be empty')
        
        self.error_code = self.error_code.strip()
        self.message = self.message.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        error_dict = asdict(self)
        error_dict['timestamp'] = self.timestamp.isoformat()
        return error_dict
