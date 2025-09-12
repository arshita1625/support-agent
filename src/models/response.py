#!/usr/bin/env python3
"""Response models using Python dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import json

@dataclass
class MCPResponse:
    """Model Context Protocol compliant response."""
    
    # Required fields
    answer: str
    action_required: Literal[
        "no_action",
        "escalate_to_abuse_team", 
        "escalate_to_billing",
        "escalate_to_technical",
        "update_whois",
        "contact_billing",
        "verify_identity",
        "provide_documentation"
    ]
    
    # Optional fields with defaults
    references: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    estimated_resolution_time: Optional[str] = None
    priority_level: Optional[Literal["low", "medium", "high", "urgent"]] = None
    
    def __post_init__(self):
        """Validate response data after initialization."""
        # Validate answer
        if not isinstance(self.answer, str):
            raise ValueError('answer must be a string')
        
        cleaned_answer = self.answer.strip()
        if len(cleaned_answer) < 10:
            raise ValueError('Answer must be at least 10 characters')
        
        if len(cleaned_answer.split()) < 10:
            raise ValueError('Answer must contain at least 10 words')
        
        # Check for placeholder text
        placeholders = ['[placeholder]', 'TODO', 'FIXME', 'XXX', 'TBD']
        answer_lower = cleaned_answer.lower()
        for placeholder in placeholders:
            if placeholder.lower() in answer_lower:
                raise ValueError(f'Answer contains placeholder text: {placeholder}')
        
        self.answer = cleaned_answer
        
        # Validate action_required
        valid_actions = [
            "no_action", "escalate_to_abuse_team", "escalate_to_billing",
            "escalate_to_technical", "update_whois", "contact_billing",
            "verify_identity", "provide_documentation"
        ]
        if self.action_required not in valid_actions:
            raise ValueError(f'action_required must be one of: {valid_actions}')
        
        # Validate references
        if not isinstance(self.references, list):
            raise ValueError('references must be a list')
        
        for ref in self.references:
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError('References cannot be empty strings')
            if len(ref.strip()) < 5:
                raise ValueError('References must be at least 5 characters')
        
        # Clean references
        self.references = [ref.strip() for ref in self.references]
        
        # Validate confidence score
        if self.confidence_score is not None:
            if not isinstance(self.confidence_score, (int, float)):
                raise ValueError('confidence_score must be a number')
            if not (0.0 <= self.confidence_score <= 1.0):
                raise ValueError('confidence_score must be between 0.0 and 1.0')
        
        # Validate priority level
        if self.priority_level is not None:
            valid_priorities = ["low", "medium", "high", "urgent"]
            if self.priority_level not in valid_priorities:
                raise ValueError(f'priority_level must be one of: {valid_priorities}')
    
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
        self.references.append(reference.strip())
    
    def set_confidence(self, confidence: float) -> None:
        """Set confidence score."""
        if not (0.0 <= confidence <= 1.0):
            raise ValueError('Confidence must be between 0.0 and 1.0')
        self.confidence_score = confidence
    
    def requires_escalation(self) -> bool:
        """Check if response requires escalation."""
        escalation_actions = [
            "escalate_to_abuse_team",
            "escalate_to_billing", 
            "escalate_to_technical"
        ]
        return self.action_required in escalation_actions

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
