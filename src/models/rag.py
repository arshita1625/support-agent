#!/usr/bin/env python3
"""RAG pipeline models using Python dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import statistics

@dataclass
class RetrievedDocument:
    """Document retrieved from vector search with similarity score."""
    
    # Required fields from vector search
    document_id: str
    content: str
    similarity_score: float
    document_type: str
    
    # Optional enrichment fields
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: Optional[int] = None
    parent_document_id: Optional[str] = None
    
    # Analysis fields (computed)
    content_length: int = field(init=False)
    word_count: int = field(init=False)
    
    def __post_init__(self):
        """Validate and compute derived fields after initialization."""
        self._validate_fields()
        self._compute_derived_fields()
    
    def _validate_fields(self):
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
    
    def _compute_derived_fields(self):
        """Compute derived fields from content."""
        self.content_length = len(self.content)
        self.word_count = len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievedDocument':
        """Create from dictionary."""
        # Extract only the fields needed for __init__
        init_data = {
            'document_id': data['document_id'],
            'content': data['content'],
            'similarity_score': data['similarity_score'],
            'document_type': data['document_type'],
            'metadata': data.get('metadata', {}),
            'chunk_index': data.get('chunk_index'),
            'parent_document_id': data.get('parent_document_id')
        }
        
        return cls(**init_data)
    
    def get_preview(self, max_length: int = 100) -> str:
        """Get content preview for display."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def get_relevance_category(self) -> str:
        """Categorize relevance based on similarity score."""
        if self.similarity_score >= 0.9:
            return "very_high"
        elif self.similarity_score >= 0.8:
            return "high"
        elif self.similarity_score >= 0.7:
            return "medium"
        elif self.similarity_score >= 0.5:
            return "low"
        else:
            return "very_low"
    
    def is_highly_relevant(self) -> bool:
        """Check if document is highly relevant (score >= 0.8)."""
        return self.similarity_score >= 0.8
    
    def get_content_focus(self) -> str:
        """Extract focus/topic from metadata or analyze content."""
        # Try to get focus from metadata first
        if self.metadata and 'chunk_focus' in self.metadata:
            return self.metadata['chunk_focus']
        
        # Fallback to content analysis
        content_lower = self.content.lower()
        
        focus_patterns = {
            "domain_suspension": ["domain suspend", "suspended", "suspension"],
            "whois_compliance": ["whois", "contact information", "registrant"],
            "billing_issues": ["billing", "payment", "invoice", "charge"],
            "technical_support": ["dns", "email", "ssl", "server", "technical"],
            "policy_violations": ["policy", "terms", "violation", "abuse"],
            "reactivation": ["reactivate", "restore", "unsuspend"],
            "general": []  # Default
        }
        
        for focus, keywords in focus_patterns.items():
            if focus == "general":
                continue
            if any(keyword in content_lower for keyword in keywords):
                return focus
        
        return "general"

@dataclass
class RAGContext:
    """Context package for LLM generation containing retrieved documents and metadata."""
    
    # Required fields
    query: str
    retrieved_documents: List[RetrievedDocument]
    total_documents_searched: int
    
    # Optional context fields
    context_summary: Optional[str] = None
    retrieval_timestamp: datetime = field(default_factory=datetime.now)
    query_embedding_time_ms: Optional[float] = None
    retrieval_time_ms: Optional[float] = None
    
    # Computed analysis fields
    context_stats: Dict[str, Any] = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        """Validate and analyze context after initialization."""
        self._validate_context()
        self._analyze_retrieved_documents()
    
    def _validate_context(self):
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
    
    def _analyze_retrieved_documents(self):
        """Analyze retrieved documents and compute context statistics."""
        if not self.retrieved_documents:
            self.context_stats = {
                "total_retrieved": 0,
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
                "total_content_length": 0,
                "total_word_count": 0,
                "document_types": [],
                "relevance_distribution": {},
                "content_focuses": [],
                "highly_relevant_count": 0,
                "coverage_score": 0.0
            }
            return
        
        # Basic statistics
        similarities = [doc.similarity_score for doc in self.retrieved_documents]
        content_lengths = [doc.content_length for doc in self.retrieved_documents]
        word_counts = [doc.word_count for doc in self.retrieved_documents]
        
        # Document type analysis
        doc_types = [doc.document_type for doc in self.retrieved_documents]
        unique_doc_types = list(set(doc_types))
        
        # Relevance analysis
        relevance_categories = [doc.get_relevance_category() for doc in self.retrieved_documents]
        relevance_distribution = {}
        for category in relevance_categories:
            relevance_distribution[category] = relevance_distribution.get(category, 0) + 1
        
        # Content focus analysis
        content_focuses = [doc.get_content_focus() for doc in self.retrieved_documents]
        unique_focuses = list(set(content_focuses))
        
        # High relevance count
        highly_relevant_count = sum(1 for doc in self.retrieved_documents if doc.is_highly_relevant())
        
        # Coverage score (how well we covered the query)
        coverage_score = self._calculate_coverage_score()
        
        self.context_stats = {
            "total_retrieved": len(self.retrieved_documents),
            "avg_similarity": statistics.mean(similarities),
            "max_similarity": max(similarities),
            "min_similarity": min(similarities),
            "similarity_std": statistics.stdev(similarities) if len(similarities) > 1 else 0.0,
            "total_content_length": sum(content_lengths),
            "total_word_count": sum(word_counts),
            "avg_content_length": statistics.mean(content_lengths),
            "document_types": unique_doc_types,
            "document_type_counts": {dt: doc_types.count(dt) for dt in unique_doc_types},
            "relevance_distribution": relevance_distribution,
            "content_focuses": unique_focuses,
            "focus_distribution": {focus: content_focuses.count(focus) for focus in unique_focuses},
            "highly_relevant_count": highly_relevant_count,
            "coverage_score": coverage_score,
            "retrieval_efficiency": len(self.retrieved_documents) / max(self.total_documents_searched, 1)
        }
    
    def _calculate_coverage_score(self) -> float:
        """Calculate how well the retrieved documents cover the query."""
        if not self.retrieved_documents:
            return 0.0
        
        # Simple coverage based on similarity scores and diversity
        avg_similarity = statistics.mean([doc.similarity_score for doc in self.retrieved_documents])
        
        # Bonus for diversity in document types
        unique_types = len(set(doc.document_type for doc in self.retrieved_documents))
        diversity_bonus = min(unique_types * 0.1, 0.3)  # Max 30% bonus
        
        # Bonus for multiple high-relevance documents
        high_rel_count = sum(1 for doc in self.retrieved_documents if doc.similarity_score >= 0.8)
        high_rel_bonus = min(high_rel_count * 0.05, 0.2)  # Max 20% bonus
        
        coverage = min(avg_similarity + diversity_bonus + high_rel_bonus, 1.0)
        return coverage
    
    @property
    def context_text(self) -> str:
        """Combine all retrieved document content into single context for LLM."""
        if not self.retrieved_documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(self.retrieved_documents):
            # Create structured context with metadata
            doc_header = f"[Document {i+1} - {doc.document_type.title()} - Relevance: {doc.similarity_score:.2f}]"
            content_focus = doc.get_content_focus()
            if content_focus != "general":
                doc_header += f" [Focus: {content_focus.replace('_', ' ').title()}]"
            
            context_parts.append(f"{doc_header}:\n{doc.content}")
        
        return "\n\n".join(context_parts)
    
    @property
    def context_text_simple(self) -> str:
        """Simple context without headers for LLM that prefers clean input."""
        if not self.retrieved_documents:
            return ""
        
        return "\n\n".join([doc.content for doc in self.retrieved_documents])
    
    @property
    def average_similarity(self) -> float:
        """Calculate average similarity score of retrieved documents."""
        if not self.retrieved_documents:
            return 0.0
        return self.context_stats.get("avg_similarity", 0.0)
    
    @property
    def max_similarity(self) -> float:
        """Get highest similarity score."""
        if not self.retrieved_documents:
            return 0.0
        return self.context_stats.get("max_similarity", 0.0)
    
    @property
    def min_similarity(self) -> float:
        """Get lowest similarity score."""
        if not self.retrieved_documents:
            return 0.0
        return self.context_stats.get("min_similarity", 0.0)
    
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
        retrieved_documents = [
            RetrievedDocument.from_dict(doc_data) 
            for doc_data in data['retrieved_documents']
        ]
        
        # Convert timestamp
        retrieval_timestamp = data['retrieval_timestamp']
        if isinstance(retrieval_timestamp, str):
            retrieval_timestamp = datetime.fromisoformat(retrieval_timestamp)
        
        # Extract only the fields needed for __init__
        init_data = {
            'query': data['query'],
            'retrieved_documents': retrieved_documents,
            'total_documents_searched': data['total_documents_searched'],
            'context_summary': data.get('context_summary'),
            'retrieval_timestamp': retrieval_timestamp,
            'query_embedding_time_ms': data.get('query_embedding_time_ms'),
            'retrieval_time_ms': data.get('retrieval_time_ms')
        }
        
        return cls(**init_data)
    def get_document_types(self) -> List[str]:
        """Get unique document types in context."""
        return self.context_stats.get("document_types", [])
    
    def filter_by_document_type(self, document_type: str) -> List[RetrievedDocument]:
        """Filter documents by type."""
        return [doc for doc in self.retrieved_documents if doc.document_type == document_type]
    
    def filter_by_relevance(self, min_similarity: float = 0.8) -> List[RetrievedDocument]:
        """Filter documents by minimum similarity threshold."""
        return [doc for doc in self.retrieved_documents if doc.similarity_score >= min_similarity]
    
    def get_top_documents(self, n: int = 3) -> List[RetrievedDocument]:
        """Get top N documents by similarity score."""
        sorted_docs = sorted(
            self.retrieved_documents, 
            key=lambda doc: doc.similarity_score, 
            reverse=True
        )
        return sorted_docs[:n]
    
    def add_context_summary(self, summary: str) -> None:
        """Add context summary."""
        if not summary or not summary.strip():
            raise ValueError('Summary cannot be empty')
        self.context_summary = summary.strip()
    
    def is_high_quality_context(self) -> bool:
        """Determine if this is high-quality context for LLM generation."""
        if not self.retrieved_documents:
            return False
        
        criteria = [
            self.average_similarity >= 0.7,  # Good average relevance
            self.max_similarity >= 0.8,     # At least one highly relevant doc
            len(self.retrieved_documents) >= 2,  # Multiple sources
            self.context_stats.get("coverage_score", 0) >= 0.6  # Good coverage
        ]
        
        return sum(criteria) >= 3  # Meet at least 3/4 criteria
    
    def get_context_quality_report(self) -> Dict[str, Any]:
        """Generate quality assessment report for the context."""
        return {
            "overall_quality": "high" if self.is_high_quality_context() else "medium" if self.average_similarity >= 0.5 else "low",
            "total_documents": len(self.retrieved_documents),
            "highly_relevant_documents": self.context_stats.get("highly_relevant_count", 0),
            "average_similarity": round(self.average_similarity, 3),
            "coverage_score": round(self.context_stats.get("coverage_score", 0), 3),
            "document_diversity": len(self.get_document_types()),
            "content_focuses": self.context_stats.get("content_focuses", []),
            "total_context_words": self.context_stats.get("total_word_count", 0),
            "retrieval_efficiency": round(self.context_stats.get("retrieval_efficiency", 0), 3),
            "recommendations": self._generate_quality_recommendations()
        }
    
    def _generate_quality_recommendations(self) -> List[str]:
        """Generate recommendations for improving context quality."""
        recommendations = []
        
        if self.average_similarity < 0.5:
            recommendations.append("Consider refining the search query or expanding the knowledge base")
        
        if len(self.retrieved_documents) < 2:
            recommendations.append("Retrieve more documents to provide comprehensive context")
        
        if self.context_stats.get("highly_relevant_count", 0) == 0:
            recommendations.append("No highly relevant documents found - query may need reformulation")
        
        if len(self.get_document_types()) == 1:
            recommendations.append("Consider retrieving documents from diverse sources/types")
        
        if self.context_stats.get("total_word_count", 0) > 2000:
            recommendations.append("Context may be too long - consider filtering to most relevant documents")
        
        if not recommendations:
            recommendations.append("Context quality is good for LLM generation")
        
        return recommendations

# Test the models if run directly
if __name__ == "__main__":
    print("🧪 Testing RAG Pipeline Models...")
    
    # Test 1: RetrievedDocument
    print("\n📄 Test 1: RetrievedDocument Creation")
    doc1 = RetrievedDocument(
        document_id="chunk_001_policy",
        content="Domain suspension occurs when WHOIS information is incomplete or inaccurate according to ICANN requirements and company policy. This helps ensure compliance with registration standards.",
        similarity_score=0.92,
        document_type="policy",
        metadata={
            "chunk_focus": "domain_suspension",
            "section": "4.1",
            "priority": "high"
        }
    )
    
    print(f"   ✅ Document created: {doc1.document_id}")
    print(f"   Content length: {doc1.content_length} chars, {doc1.word_count} words")
    print(f"   Relevance: {doc1.get_relevance_category()}")
    print(f"   Focus: {doc1.get_content_focus()}")
    print(f"   Highly relevant: {doc1.is_highly_relevant()}")
    print(f"   Preview: {doc1.get_preview(60)}")
    
    # Test 2: More RetrievedDocuments
    print("\n📄 Test 2: Additional Documents")
    doc2 = RetrievedDocument(
        document_id="chunk_002_faq",
        content="To reactivate a suspended domain, customers must update WHOIS information and contact the abuse team for manual review. The process typically takes 24-48 hours.",
        similarity_score=0.87,
        document_type="faq",
        metadata={"chunk_focus": "reactivation"}
    )
    
    doc3 = RetrievedDocument(
        document_id="chunk_003_procedure",
        content="WHOIS compliance requires complete registrant contact information including valid email address and phone number. Post office boxes are prohibited for certain TLDs.",
        similarity_score=0.75,
        document_type="procedure",
        metadata={"chunk_focus": "whois_compliance"}
    )
    
    print(f"   ✅ FAQ document: {doc2.similarity_score} similarity")
    print(f"   ✅ Procedure document: {doc3.similarity_score} similarity")
    
    # Test 3: RAGContext
    print("\n🔄 Test 3: RAGContext Creation")
    context = RAGContext(
        query="My domain was suspended without notice, how do I reactivate it?",
        retrieved_documents=[doc1, doc2, doc3],
        total_documents_searched=1000,
        query_embedding_time_ms=45.2,
        retrieval_time_ms=120.8
    )
    
    print(f"   ✅ RAG context created")
    print(f"   Query: {context.query[:50]}...")
    print(f"   Retrieved docs: {len(context.retrieved_documents)}")
    print(f"   Average similarity: {context.average_similarity:.3f}")
    print(f"   Document types: {context.get_document_types()}")
    
    # Test 4: Context Analysis
    print("\n📊 Test 4: Context Analysis")
    print(f"   Context stats:")
    for key, value in context.context_stats.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.3f}")
        elif isinstance(value, (list, dict)) and len(str(value)) > 50:
            print(f"     {key}: {type(value).__name__} with {len(value)} items")
        else:
            print(f"     {key}: {value}")
    
    # Test 5: Context Quality
    print("\n🎯 Test 5: Context Quality Assessment")
    quality_report = context.get_context_quality_report()
    print(f"   Overall quality: {quality_report['overall_quality']}")
    print(f"   Coverage score: {quality_report['coverage_score']}")
    print(f"   Recommendations:")
    for rec in quality_report['recommendations']:
        print(f"     • {rec}")
    
    # Test 6: Context Text Generation
    print("\n📝 Test 6: Context Text Generation")
    context_text = context.context_text
    print(f"   Structured context length: {len(context_text)} characters")
    print(f"   First 200 chars: {context_text[:200]}...")
    
    simple_context = context.context_text_simple
    print(f"   Simple context length: {len(simple_context)} characters")
    
    # Test 7: Filtering and Analysis
    print("\n🔍 Test 7: Document Filtering")
    top_docs = context.get_top_documents(2)
    print(f"   Top 2 documents: {[doc.document_id for doc in top_docs]}")
    
    high_rel_docs = context.filter_by_relevance(0.8)
    print(f"   High relevance docs (≥0.8): {len(high_rel_docs)}")
    
    policy_docs = context.filter_by_document_type("policy")
    print(f"   Policy documents: {len(policy_docs)}")
    
    # Test 8: Serialization
    print("\n💾 Test 8: Serialization")
    context_dict = context.to_dict()
    context_restored = RAGContext.from_dict(context_dict)
    print(f"   Serialization successful: {context_restored.query[:30]}...")
    print(f"   Restored doc count: {len(context_restored.retrieved_documents)}")
    
    print(f"\n🎉 All RAG pipeline models working correctly!")
