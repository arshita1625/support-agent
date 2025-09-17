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
    retrieved_chunks: List[Dict] = field(default_factory=list)  # New field for chunk metadata
    
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
            # Simple string searches instead of complex regex
            if 'immediately' in answer_lower or 'right away' in answer_lower:
                self.estimated_resolution_time = 'immediately'
            elif 'hours' in answer_lower:
                # Find number before 'hours'
                words = answer_lower.split()
                for i, word in enumerate(words):
                    if 'hour' in word and i > 0:
                        prev_word = words[i-1]
                        if prev_word.isdigit():
                            self.estimated_resolution_time = f"{prev_word} hours"
                            break
            elif 'days' in answer_lower:
                # Find number before 'days'  
                words = answer_lower.split()
                for i, word in enumerate(words):
                    if 'day' in word and i > 0:
                        prev_word = words[i-1]
                        if prev_word.isdigit():
                            self.estimated_resolution_time = f"{prev_word} days"
                            break
        
        # Simple email detection - no regex
        words = self.answer.split()
        emails = []
        for word in words:
            if '@' in word and '.' in word:
                emails.append(word.strip('.,;:'))
        
        if emails:
            self.contact_information['emails'] = emails
        
        # Simple follow-up detection
        follow_up_indicators = [
            'follow up', 'check back', 'contact you', 'let you know',
            'update you', 'monitor', 'review in', 'schedule'
        ]
        
        if any(indicator in answer_lower for indicator in follow_up_indicators):
            self.follow_up_required = True

    def _set_references_from_content(self):

        try:
            # First try to get references from retrieved chunks with heading metadata
            references = self._extract_dynamic_references()
            
            # If no dynamic references found, fall back to keyword-based matching
            if not references:
                references = self._extract_fallback_references()
            
            # Limit to top 5 most relevant references
            self.references = references[:5]
            
        except Exception as e:
            print(f"Error extracting references: {e}")
            self.references = ["General Support Documentation"]

    def _extract_dynamic_references(self):
      
        references = []
        
        # Get the chunks that were retrieved for this query
        if hasattr(self, 'retrieved_chunks') and self.retrieved_chunks:
            for chunk in self.retrieved_chunks:
                reference = self._extract_clean_reference(chunk)
                if reference and reference not in references:
                    references.append(reference)
        
        return references

    def _extract_clean_reference(self, chunk):
        try:
            metadata = chunk.get('metadata', {})
            content = chunk.get('content', '')
            document_type = chunk.get('document_type', '')
            
            # Handle FAQ chunks first
            if document_type == 'faq' or content.startswith('Q:'):
                if content.startswith('Q:'):
                    lines = content.split('\n')
                    question = lines.replace('Q:', '').strip()
                    return f"FAQ: {question}"
                else:
                    return "FAQ Response"
            
            # For policy/procedure chunks, get clean section title
            section_title = metadata.get('section_title', '')
            subsection_title = metadata.get('subsection_title', '')
            
            # Prefer subsection if it exists, otherwise use section
            if subsection_title:
                return subsection_title
            elif section_title:
                return section_title
            
            # Extract from content if metadata is missing
            title_from_content = self._extract_title_from_content(content)
            if title_from_content:
                return title_from_content
            
            # Fallback to document title
            doc_title = metadata.get('parent_title', 'Support Document')
            return doc_title
            
        except Exception as e:
            print(f"Error creating reference from chunk: {e}")
            return None

    def _extract_title_from_content(self, content):
       
        if not content:
            return None
            
        lines = content.split('\n')
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            
            # Look for ## section headers (level 2 headings)
            if line.startswith('## '):
                title = line[3:].strip()  # Remove "## "
                return self._clean_section_title(title)
            
            # Look for ### subsection headers (level 3 headings)  
            if line.startswith('### '):
                title = line[4:].strip()  # Remove "### "
                return self._clean_section_title(title)
            
            # Look for numbered sections (4.1 Something) - NO REGEX
            if '. ' in line and line.split('.')[0].replace(' ', '').isdigit():
                parts = line.split('. ', 1)
                if len(parts) > 1:
                    return self._clean_section_title(parts[1])
        
        return None

    def _clean_section_title(self, title):
        if not title:
            return None
            
        title = str(title).strip()
        
        # Remove leading numbers and dots using string operations only
        while title and (title[0].isdigit() or title[0] in '. '):
            # Remove leading digits
            while title and title[0].isdigit():
                title = title[1:]
            # Remove leading dots and spaces
            while title and title[0] in '. ':
                title = title[1:]
            title = title.strip()
        
        # Remove common document prefixes
        prefixes_to_remove = [
            'Document: ',
            'SECTION ',
            'Section ',
            'DOMAIN SUSPENSION GUIDELINES',
            'PAYMENT FAILURES AND ACCOUNT SUSPENSION',
            'PROHIBITED ACTIVITIES',
            'WEBSITE SECURITY POLICY',
            'EMAIL HOSTING POLICY'
        ]
        
        for prefix in prefixes_to_remove:
            if title.upper().startswith(prefix.upper()):
                title = title[len(prefix):].strip()
        
        # Clean up extra whitespace - no regex
        while '  ' in title:  # Replace double spaces with single
            title = title.replace('  ', ' ')
        title = title.strip()
        
        # Capitalize properly if not empty
        if title:
            title = title.title()
        
        # Truncate if too long
        if len(title) > 60:
            title = title[:57] + "..."
        
        return title if title else None

    def _extract_fallback_references(self):
        answer_lower = self.answer.lower()
        references = []
        
        # Domain suspension
        if any(term in answer_lower for term in ['domain suspend', 'suspended', 'suspension']):
            references.append("Suspension Triggers")
            references.append("Suspension Process")
            references.append("Reactivation Requirements")
        
        # WHOIS issues
        if any(term in answer_lower for term in ['whois', 'contact information', 'registrant']):
            references.append("WHOIS Compliance Issues")
            references.append("Domain Contact Verification")
        
        # Billing issues
        if any(term in answer_lower for term in ['payment', 'billing', 'charge']):
            references.append("Grace Period Policy")
            references.append("Payment Methods")
        
        # Technical issues
        if any(term in answer_lower for term in ['dns', 'nameserver', 'propagation']):
            references.append("DNS Configuration Problems")
            references.append("Technical Support Procedures")
        
        # Email issues
        if any(term in answer_lower for term in ['email', 'mail', 'mx record']):
            references.append("Email Delivery Issues")
            references.append("MX Record Configuration")
        
        # SSL issues
        if any(term in answer_lower for term in ['ssl', 'certificate', 'https']):
            references.append("SSL Certificate Problems")
            references.append("Certificate Installation")
        
        # Security issues
        if any(term in answer_lower for term in ['malware', 'security', 'abuse']):
            references.append("Malware Detection")
            references.append("Security Policy Violations")
        
        # Refunds
        if any(term in answer_lower for term in ['refund', 'money back']):
            references.append("Refund Policy")
        
        # Default if none found
        if not references:
            references = ["General Support Guidelines"]
        
        return references
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Remove retrieved_chunks from the output as it's internal
        result.pop('retrieved_chunks', None)
        return result
    
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
