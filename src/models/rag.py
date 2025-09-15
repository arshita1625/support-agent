#!/usr/bin/env python3
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import statistics

@dataclass
class RetrievedDocument:
    document_id: str
    content: str
    similarity_score: float
    document_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: Optional[int] = None
    parent_document_id: Optional[str] = None
    content_length: int = field(init=False)
    word_count: int = field(init=False)
    
    def __post_init__(self):
        self._validate_fields()
        self._compute_derived_fields()
    
    def _validate_fields(self):
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError('document_id cannot be empty')
        
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError('content cannot be empty')
    
        if not isinstance(self.similarity_score, (int, float)):
            raise ValueError('similarity_score must be a number')
        
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError('similarity_score must be between 0.0 and 1.0')
        
        if not isinstance(self.document_type, str) or not self.document_type.strip():
            raise ValueError('document_type cannot be empty')
        self.document_id = self.document_id.strip()
        self.content = self.content.strip()
        self.document_type = self.document_type.strip()
    
    def _compute_derived_fields(self):
        self.content_length = len(self.content)
        self.word_count = len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievedDocument':
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
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def get_relevance_category(self) -> str:
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
        return self.similarity_score >= 0.8
    
    def get_content_focus(self) -> str:
        if self.metadata and 'chunk_focus' in self.metadata:
            return self.metadata['chunk_focus']
        
        content_lower = self.content.lower()
        
        focus_patterns = {
            "domain_suspension": ["domain suspend", "suspended", "suspension"],
            "whois_compliance": ["whois", "contact information", "registrant"],
            "billing_issues": ["billing", "payment", "invoice", "charge"],
            "technical_support": ["dns", "email", "ssl", "server", "technical"],
            "policy_violations": ["policy", "terms", "violation", "abuse"],
            "reactivation": ["reactivate", "restore", "unsuspend"],
            "general": [] 
        }
        
        for focus, keywords in focus_patterns.items():
            if focus == "general":
                continue
            if any(keyword in content_lower for keyword in keywords):
                return focus
        
        return "general"

@dataclass
class RAGContext:
    query: str
    retrieved_documents: List[RetrievedDocument]
    total_documents_searched: int
    context_summary: Optional[str] = None
    retrieval_timestamp: datetime = field(default_factory=datetime.now)
    query_embedding_time_ms: Optional[float] = None
    retrieval_time_ms: Optional[float] = None
    context_stats: Dict[str, Any] = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        self._validate_context()
        self._analyze_retrieved_documents()
    
    def _validate_context(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError('query cannot be empty')
        
        if not isinstance(self.retrieved_documents, list):
            raise ValueError('retrieved_documents must be a list')
        
        for i, doc in enumerate(self.retrieved_documents):
            if not isinstance(doc, RetrievedDocument):
                raise ValueError(f'retrieved_documents[{i}] must be a RetrievedDocument instance')
        if not isinstance(self.total_documents_searched, int) or self.total_documents_searched < 0:
            raise ValueError('total_documents_searched must be a non-negative integer')
        
        if self.total_documents_searched < len(self.retrieved_documents):
            raise ValueError('total_documents_searched cannot be less than number of retrieved documents')
       
        self.query = self.query.strip()
    
    def _analyze_retrieved_documents(self):
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
 
        similarities = [doc.similarity_score for doc in self.retrieved_documents]
        content_lengths = [doc.content_length for doc in self.retrieved_documents]
        word_counts = [doc.word_count for doc in self.retrieved_documents]
        doc_types = [doc.document_type for doc in self.retrieved_documents]
        unique_doc_types = list(set(doc_types))
        relevance_categories = [doc.get_relevance_category() for doc in self.retrieved_documents]
        relevance_distribution = {}
        for category in relevance_categories:
            relevance_distribution[category] = relevance_distribution.get(category, 0) + 1
        content_focuses = [doc.get_content_focus() for doc in self.retrieved_documents]
        unique_focuses = list(set(content_focuses))
        highly_relevant_count = sum(1 for doc in self.retrieved_documents if doc.is_highly_relevant())
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
        if not self.retrieved_documents:
            return 0.0

        avg_similarity = statistics.mean([doc.similarity_score for doc in self.retrieved_documents])
        unique_types = len(set(doc.document_type for doc in self.retrieved_documents))
        diversity_bonus = min(unique_types * 0.1, 0.3)  
        high_rel_count = sum(1 for doc in self.retrieved_documents if doc.similarity_score >= 0.8)
        high_rel_bonus = min(high_rel_count * 0.05, 0.2)  
        coverage = min(avg_similarity + diversity_bonus + high_rel_bonus, 1.0)
        return coverage
    
    @property
    def context_text(self) -> str:
        if not self.retrieved_documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(self.retrieved_documents):
            doc_header = f"[Document {i+1} - {doc.document_type.title()} - Relevance: {doc.similarity_score:.2f}]"
            content_focus = doc.get_content_focus()
            if content_focus != "general":
                doc_header += f" [Focus: {content_focus.replace('_', ' ').title()}]"
            
            context_parts.append(f"{doc_header}:\n{doc.content}")
        
        return "\n\n".join(context_parts)
    
    @property
    def context_text_simple(self) -> str:
        if not self.retrieved_documents:
            return ""
        
        return "\n\n".join([doc.content for doc in self.retrieved_documents])
    
    @property
    def average_similarity(self) -> float:
        if not self.retrieved_documents:
            return 0.0
        return self.context_stats.get("avg_similarity", 0.0)
    
    @property
    def max_similarity(self) -> float:
        if not self.retrieved_documents:
            return 0.0
        return self.context_stats.get("max_similarity", 0.0)
    
    @property
    def min_similarity(self) -> float:
        if not self.retrieved_documents:
            return 0.0
        return self.context_stats.get("min_similarity", 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        context_dict = asdict(self)
        context_dict['retrieval_timestamp'] = self.retrieval_timestamp.isoformat()
        return context_dict
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGContext':

        retrieved_documents = [
            RetrievedDocument.from_dict(doc_data) 
            for doc_data in data['retrieved_documents']
        ]
        
        retrieval_timestamp = data['retrieval_timestamp']
        if isinstance(retrieval_timestamp, str):
            retrieval_timestamp = datetime.fromisoformat(retrieval_timestamp)
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
        return self.context_stats.get("document_types", [])
    
    def filter_by_document_type(self, document_type: str) -> List[RetrievedDocument]:
        return [doc for doc in self.retrieved_documents if doc.document_type == document_type]
    
    def filter_by_relevance(self, min_similarity: float = 0.8) -> List[RetrievedDocument]:
        return [doc for doc in self.retrieved_documents if doc.similarity_score >= min_similarity]
    
    def get_top_documents(self, n: int = 3) -> List[RetrievedDocument]:
        sorted_docs = sorted(
            self.retrieved_documents, 
            key=lambda doc: doc.similarity_score, 
            reverse=True
        )
        return sorted_docs[:n]
    
    def add_context_summary(self, summary: str) -> None:
        if not summary or not summary.strip():
            raise ValueError('Summary cannot be empty')
        self.context_summary = summary.strip()
    
    def is_high_quality_context(self) -> bool:
        if not self.retrieved_documents:
            return False
        
        criteria = [
            self.average_similarity >= 0.7,  
            self.max_similarity >= 0.8,     
            len(self.retrieved_documents) >= 2,  
            self.context_stats.get("coverage_score", 0) >= 0.6  
        ]
        
        return sum(criteria) >= 3 
    
    def get_context_quality_report(self) -> Dict[str, Any]:
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