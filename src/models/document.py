#!/usr/bin/env python3
"""Document and chunk models using Python dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4
import json

@dataclass
class Document:
    """Base document model for knowledge base content."""
    
    # Required fields
    content: str
    document_type: str
    
    # Optional fields with defaults
    id: str = field(default_factory=lambda: str(uuid4()))
    title: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate document data after initialization."""
        # Validate document type
        allowed_types = ['policy', 'faq', 'procedure', 'guide', 'knowledge_article']
        if self.document_type not in allowed_types:
            raise ValueError(f'document_type must be one of: {allowed_types}')
        
        # Validate and clean content
        if not isinstance(self.content, str):
            raise ValueError('content must be a string')
        
        cleaned_content = ' '.join(self.content.split())
        if len(cleaned_content) < 10:
            raise ValueError('Content must be at least 10 meaningful characters')
        
        self.content = cleaned_content
        
        # Validate title length if provided
        if self.title and len(self.title) > 200:
            raise ValueError('Title must be 200 characters or less')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for JSON serialization."""
        doc_dict = asdict(self)
        # Convert datetime objects to ISO strings
        doc_dict['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            doc_dict['updated_at'] = self.updated_at.isoformat()
        else:
            doc_dict['updated_at'] = None
        return doc_dict
    
    def to_json(self) -> str:
        """Convert document to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        # Convert ISO strings back to datetime objects
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at') and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)
    
    def update_content(self, new_content: str) -> None:
        """Update document content and set updated timestamp."""
        self.content = new_content
        self.updated_at = datetime.now()
        # Re-run validation
        self.__post_init__()

@dataclass
class DocumentChunk:
    """Chunked document for vector storage and retrieval."""
    
    # Required fields
    parent_document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    document_type: str
    
    # Optional fields with defaults
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate chunk data after initialization."""
        # Validate content
        if not isinstance(self.content, str) or len(self.content.strip()) < 1:
            raise ValueError('Chunk content cannot be empty')
        
        if len(self.content) > 2000:
            raise ValueError('Chunk content must be 2000 characters or less')
        
        # Validate character positions
        if not isinstance(self.start_char, int) or self.start_char < 0:
            raise ValueError('start_char must be a non-negative integer')
        
        if not isinstance(self.end_char, int) or self.end_char <= self.start_char:
            raise ValueError('end_char must be greater than start_char')
        
        # Validate chunk index
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError('chunk_index must be a non-negative integer')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert chunk to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create chunk from dictionary."""
        return cls(**data)

# Test the models if run directly
if __name__ == "__main__":
    print("🧪 Testing Document Models...")
    
    # Test Document
    doc = Document(
        content="This is a comprehensive domain suspension policy document that outlines the procedures and requirements for domain management.",
        title="Domain Suspension Policy",
        document_type="policy",
        source="policies/domain_management.pdf",
        metadata={
            "section": "4.2",
            "priority": "high",
            "tags": ["domain", "suspension", "policy"]
        }
    )
    print(f"✅ Document: {doc.title} (ID: {doc.id[:8]}...)")
    print(f"   Type: {doc.document_type}")
    print(f"   Content length: {len(doc.content)} characters")
    
    # Test DocumentChunk
    chunk = DocumentChunk(
        parent_document_id=doc.id,
        content="Domain suspension occurs when WHOIS information is incomplete or inaccurate according to ICANN requirements.",
        chunk_index=0,
        start_char=0,
        end_char=100,
        document_type=doc.document_type,
        metadata={"section": "4.2", "focus": "whois_compliance"}
    )
    print(f"✅ DocumentChunk: {chunk.chunk_id[:8]}... (index {chunk.chunk_index})")
    
    # Test serialization
    doc_dict = doc.to_dict()
    chunk_dict = chunk.to_dict()
    print(f"✅ Serialization works - Document dict has {len(doc_dict)} fields")
    
    # Test deserialization
    doc_restored = Document.from_dict(doc_dict)
    chunk_restored = DocumentChunk.from_dict(chunk_dict)
    print(f"✅ Deserialization works - Restored: {doc_restored.title}")
    
    print("🎉 All document models working correctly!")
