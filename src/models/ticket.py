#!/usr/bin/env python3
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from datetime import datetime
from uuid import uuid4
import re
import json
from typing import List
@dataclass
class SupportTicket:
    ticket_text: str
    ticket_id: str = field(default_factory=lambda: str(uuid4()))
    priority: Literal["low", "medium", "high"] = "medium"
    category: Optional[str] = None
    customer_id: Optional[str] = None
    submitted_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not isinstance(self.ticket_text, str):
            raise ValueError('ticket_text must be a string')
        cleaned_text = ' '.join(self.ticket_text.split())
        
        if len(cleaned_text) < 10:
            raise ValueError('Ticket text must be at least 10 characters')
        
        if len(cleaned_text) > 2000:
            raise ValueError('Ticket text must be 2000 characters or less')
        if len(cleaned_text.split()) < 3:
            raise ValueError('Ticket must contain at least 3 words')
        spam_patterns = [
            r'^[!@#$%^&*()_+\-=\[\]{}|;\':".,<>?/~`]+$',  # only special characters
            r'^(.)\1{20,}',  # same character repeated 20+ times
        ]
        for pattern in spam_patterns:
            if re.search(pattern, cleaned_text):
                raise ValueError('Ticket content appears to be spam')
        
        self.ticket_text = cleaned_text

        if self.priority not in ["low", "medium", "high"]:
            raise ValueError(f'priority must be one of: low, medium, high')
    
    def to_dict(self) -> dict:
        ticket_dict = asdict(self)
        ticket_dict['submitted_at'] = self.submitted_at.isoformat()
        return ticket_dict
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SupportTicket':
        if isinstance(data.get('submitted_at'), str):
            data['submitted_at'] = datetime.fromisoformat(data['submitted_at'])
        return cls(**data)
    
    def categorize(self, category: str) -> None:
        self.category = category
    
    def set_priority(self, priority: Literal["low", "medium", "high"]) -> None:
        if priority not in ["low", "medium", "high"]:
            raise ValueError(f'priority must be one of: low, medium, high')
        self.priority = priority
    
    def get_word_count(self) -> int:
        return len(self.ticket_text.split())
    
    def extract_keywords(self) -> List[str]:
        text_lower = self.ticket_text.lower()
        keywords = []
        keyword_categories = {
            'domain': ['domain', 'website', 'site', 'url', 'www', 'web address'],
            'email': ['email', 'mail', 'smtp', 'pop', 'imap', 'inbox', 'mailbox', 'e-mail'],
            'billing': ['billing', 'payment', 'charge', 'refund', 'invoice', 'cost', 'price', 'fee', 'money'],
            'suspension': ['suspend', 'suspended', 'suspension', 'disabled', 'blocked', 'deactivated'],
            'technical': ['error', 'not working', 'broken', 'issue', 'problem', 'bug', 'glitch', 'malfunction'],
            'whois': ['whois', 'contact', 'registrant', 'owner', 'admin', 'contact information'],
            'dns': ['dns', 'nameserver', 'name server', 'records', 'a record', 'cname', 'mx record'],
            'hosting': ['hosting', 'server', 'upload', 'ftp', 'files', 'host', 'hosting account'],
            'ssl': ['ssl', 'certificate', 'https', 'secure', 'security', 'encryption'],
            'transfer': ['transfer', 'move', 'migrate', 'change registrar', 'transferring'],
            'support': ['support', 'help', 'assistance', 'customer service', 'customer support'],
            'contact': ['contact', 'talk to', 'speak with', 'reach', 'get in touch', 'communicate'],
            'inquiry': ['question', 'ask', 'inquiry', 'wondering', 'information', 'clarification'],
            'account': ['account', 'login', 'password', 'username', 'profile', 'dashboard'],
            'general': ['how to', 'where can', 'what is', 'when will', 'why is', 'who can'],
            'urgency': ['urgent', 'emergency', 'asap', 'immediately', 'critical', 'right now'],
            'access': ['access', 'login', 'log in', 'sign in', 'cant access', "can't access"],
            'setup': ['setup', 'set up', 'configure', 'install', 'installation', 'configuration'],
            'troubleshooting': ['fix', 'solve', 'resolve', 'repair', 'troubleshoot', 'debug']
        }
        
        for category, terms in keyword_categories.items():
            if any(term in text_lower for term in terms):
                keywords.append(category)
        
        return keywords
    def categorize_automatically(self) -> str:
        keywords = self.extract_keywords()
        category_mapping = {
            'domain_issues': ['domain', 'suspension', 'whois', 'transfer', 'dns'],
            'email_issues': ['email'],
            'billing_issues': ['billing'],
            'technical_issues': ['technical', 'hosting', 'ssl', 'setup', 'troubleshooting'],
            'support_request': ['support', 'contact', 'inquiry'],  
            'account_issues': ['account', 'access'],  
            'general_inquiry': ['general']  
        }
        best_category = 'general_inquiry'
        max_matches = 0
        for category, category_keywords in category_mapping.items():
            matches = len(set(keywords) & set(category_keywords))
            if matches > max_matches:
                max_matches = matches
                best_category = category
        
        return best_category