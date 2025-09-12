#!/usr/bin/env python3
"""RAG pipeline models using Python dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

@dataclass
class RetrievedDocument:
    """Document retrieved from vector search with similarity score."""
    
    # Required fields
    document_id: str
    content: str
    similarity_score: float
    document_type: str
    
    # Optional fields
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate retrieved document data."""
        # Validate document_id
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError('document_id cannot be empty')
        
        # Validate content
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError('content cannot be empty')
        
        # Validate similarity score
        if not isinstance(self.similarity_score, (int, float)):
            raise ValueError('similarity_score must be a number')
        
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError('similarity_score must be between 0.0 and 1.0')
        
        # Validate document type
        if not isinstance(self.document_type, str) or not self.document_type.strip():
            raise ValueError('document_type cannot be empty')
        
        # Clean fields
        self.document_id = self.document_id.strip()
        self.content = self.content.strip()
        self.document_type = self.document_type.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievedDocument':
        """Create from dictionary."""
        return cls(**data)
    
    def get_preview(self, max_length: int = 100) -> str:
        """Get content preview."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."

@dataclass
class RAGContext:
    """Context package for LLM generation."""
    
    # Required fields
    query: str
    retrieved_documents: List[RetrievedDocument]
    total_documents_searched: int
    
    # Optional fields
    context_summary: Optional[str] = None
    retrieval_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate RAG context data."""
        # Validate query
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError('query cannot be empty')
        
        # Validate retrieved documents
        if not isinstance(self.retrieved_documents, list):
            raise ValueError('retrieved_documents must be a list')
        
        for i, doc in enumerate(self.retrieved_documents):
            if not isinstance(doc, RetrievedDocument):
                raise ValueError(f'retrieved_documents[{i}] must be a RetrievedDocument instance')
        
        # Validate total documents searched
        if not isinstance(self.total_documents_searched, int) or self.total_documents_searched < 0:
            raise ValueError('total_documents_searched must be a non-negative integer')
        
        if self.total_documents_searched < len(self.retrieved_documents):
            raise ValueError('total_documents_searched cannot be less than number of retrieved documents')
        
        # Clean query
        self.query = self.query.strip()
    
    @property
    def context_text(self) -> str:
        """Combine all retrieved document content into single context."""
        if not self.retrieved_documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(self.retrieved_documents):
            context_parts.append(f"[Document {i+1} - {doc.document_type}]: {doc.content}")
        
        return "\n\n".join(context_parts)
    
    @property
    def average_similarity(self) -> float:
        """Calculate average similarity score of retrieved documents."""
        if not self.retrieved_documents:
            return 0.0
        
        total_score = sum(doc.similarity_score for doc in self.retrieved_documents)
        return total_score / len(self.retrieved_documents)
    
    @property
    def max_similarity(self) -> float:
        """Get highest similarity score."""
        if not self.retrieved_documents:
            return 0.0
        return max(doc.similarity_score for doc in self.retrieved_documents)
    
    @property
    def min_similarity(self) -> float:
        """Get lowest similarity score."""
        if not self.retrieved_documents:
            return 0.0
        return min(doc.similarity_score for doc in self.retrieved_documents)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        context_dict = asdict(self)
        context_dict['retrieval_timestamp'] = self.retrieval_timestamp.isoformat()
        return context_dict
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGContext':
        """Create from dictionary."""
        # Convert retrieved documents from dicts to objects
        if 'retrieved_documents' in data:
            data['retrieved_documents'] = [
                RetrievedDocument.from_dict(doc_data) 
                for doc_data in data['retrieved_documents']
            ]
        
        # Convert timestamp
        if isinstance(data.get('retrieval_timestamp'), str):
            data['retrieval_timestamp'] = datetime.fromisoformat(data['retrieval_timestamp'])
        
        return cls(**data)
    
    def get_document_types(self) -> List[str]:
        """Get unique document types in context."""
        return list(set(doc.document_type for doc in self.retrieved_documents))
    
    def filter_by_document_type(self, document_type: str) -> List[RetrievedDocument]:
        """Filter documents by type."""
        return [doc for doc in self.retrieved_documents if doc.document_type == document_type]
    
    def get_top_documents(self, n: int = 3) -> List[RetrievedDocument]:
        """Get top N documents by similarity score."""
        sorted_docs = sorted(self.retrieved_documents, 
                           key=lambda doc: doc.similarity_score, 
                           reverse=True)
        return sorted_docs[:n]
    
    def add_context_summary(self, summary: str) -> None:
        """Add context summary."""
        if not summary or not summary.strip():
            raise ValueError('Summary cannot be empty')
        self.context_summary = summary.strip()

# Test the models if run directly
if __name__ == "__main__":
    print("🧪 Testing RAG Pipeline Models...")
    
    # Test RetrievedDocument
    doc1 = RetrievedDocument(
        document_id="doc_001",
        content="Domain suspension occurs when WHOIS information is incomplete or inaccurate according to ICANN requirements and company policy.",
        similarity_score=0.92,
        document_type="policy",
        metadata={"section": "4.2", "priority": "high"}
    )
    
    doc2 = RetrievedDocument(
        document_id="doc_002", 
        content="To reactivate a suspended domain, customers must update WHOIS information and contact the abuse team for manual review.",
        similarity_score=0.87,
        document_type="procedure",
        metadata={"section": "reactivation", "estimated_time": "24-48 hours"}
    )
    
    print(f"✅ Retrieved Documents created")
    print(f"   Doc 1 similarity: {doc1.similarity_score}")
    print(f"   Doc 2 preview: {doc2.get_preview(50)}")
    
    # Test RAGContext
    context = RAGContext(
        query="My domain was suspended without notice, how do I fix it?",
        retrieved_documents=[doc1, doc2],
        total_documents_searched=1000,
        context_summary="Found relevant policy and procedure documents about domain suspension and reactivation"
    )
    
    print(f"✅ RAG Context created")
    print(f"   Query: {context.query[:50]}...")
    print(f"   Retrieved docs: {len(context.retrieved_documents)}")
    print(f"   Average similarity: {context.average_similarity:.3f}")
    print(f"   Document types: {context.get_document_types()}")
    print(f"   Context length: {len(context.context_text)} characters")
    
    # Test top documents
    top_docs = context.get_top_documents(1)
    print(f"✅ Top document: {top_docs[0].document_id} (score: {top_docs[0].similarity_score})")
    
    # Test serialization
    context_dict = context.to_dict()
    context_restored = RAGContext.from_dict(context_dict)
    print(f"✅ Serialization works - Restored query: {context_restored.query[:30]}...")
    
    print("🎉 All RAG pipeline models working correctly!")
