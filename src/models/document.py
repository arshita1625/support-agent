#!/usr/bin/env python3

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4
import json

@dataclass
class Document:
    content: str
    document_type: str
    id: str = field(default_factory=lambda: str(uuid4()))
    title: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        allowed_types = ['policy', 'faq', 'procedure', 'guide', 'knowledge_article']
        if self.document_type not in allowed_types:
            raise ValueError(f'document_type must be one of: {allowed_types}')
        if not isinstance(self.content, str):
            raise ValueError('content must be a string')
        cleaned_content = ' '.join(self.content.split())
        if len(cleaned_content) < 10:
            raise ValueError('Content must be at least 10 meaningful characters')
        
        self.content = cleaned_content
        
        if self.title and len(self.title) > 200:
            raise ValueError('Title must be 200 characters or less')
    
    def to_dict(self) -> Dict[str, Any]: #For JSON serialization 
        doc_dict = asdict(self)
        doc_dict['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            doc_dict['updated_at'] = self.updated_at.isoformat()
        else:
            doc_dict['updated_at'] = None
        return doc_dict
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at') and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)
    
    def update_content(self, new_content: str) -> None:
        """Update document content and set updated timestamp."""
        self.content = new_content
        self.updated_at = datetime.now()
        self.__post_init__()

@dataclass
class DocumentChunk:
    parent_document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    document_type: str
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.content, str) or len(self.content.strip()) < 1:
            raise ValueError('Chunk content cannot be empty')
        
        if len(self.content) > 2000:
            raise ValueError('Chunk content must be 2000 characters or less')
        
        if not isinstance(self.start_char, int) or self.start_char < 0:
            raise ValueError('start_char must be a non-negative integer')
        
        if not isinstance(self.end_char, int) or self.end_char <= self.start_char:
            raise ValueError('end_char must be greater than start_char')
        
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError('chunk_index must be a non-negative integer')
    
    def to_dict(self) -> Dict[str, Any]:   #chunk to dict
        return asdict(self)
    
    def to_json(self) -> str:    #chunk to json string-> seralization
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk': #creating chunk from dict
        return cls(**data)
