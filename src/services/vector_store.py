#!/usr/bin/env python3
"""Vector store service for managing document embeddings in Qdrant."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import ResponseHandlingException
except ImportError:
    print("Qdrant client not installed. Install with: pip install qdrant-client")
    raise

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.document import Document, DocumentChunk
from models.rag import RetrievedDocument, RAGContext
from models.common import ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreService:
    
    def __init__(
        self,
        host: str = "localhost",
        # host: str = "qdrant",
        port: int = 6333,
        collection_name: str = "support_documents",
        vector_size: int = 1536, 
        distance_metric: str = "Cosine"
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, recreate: bool = False) -> bool:
        
        try:
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            if collection_exists:
                if recreate:
                    self.client.delete_collection(collection_name=self.collection_name)
                else:
                    return True
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            logger.info(f"Collection created successfully: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> bool:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if not chunks:
            logger.warning("⚠️ No chunks provided to add")
            return True
        
        try:
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Validate embedding dimension
                if len(embedding) != self.vector_size:
                    raise ValueError(
                        f"Embedding dimension {len(embedding)} doesn't match "
                        f"expected size {self.vector_size}"
                    )
                
                # Create point with metadata
                point = models.PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
                    vector=embedding,
                    payload={
                        # Core document information
                        "chunk_id": chunk.chunk_id,
                        "parent_document_id": chunk.parent_document_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "document_type": chunk.document_type,
                        
                        # Metadata for filtering and analysis
                        "metadata": chunk.metadata,
                        
                        # Computed fields for search optimization
                        "content_length": len(chunk.content),
                        "word_count": len(chunk.content.split()),
                        
                        # Indexing timestamp
                        "indexed_at": datetime.now().isoformat(),
                        
                        # Extract key fields from metadata for easy filtering
                        "chunk_focus": chunk.metadata.get("chunk_focus", "general"),
                        "parent_title": chunk.metadata.get("parent_title", ""),
                        "section_index": chunk.metadata.get("section_index", 0),
                        "priority": chunk.metadata.get("priority", "medium"),
                        "tags": chunk.metadata.get("tags", [])
                    }
                )
                
                points.append(point)
            
            # Insert points in batches for better performance
            batch_size = 100
            total_batches = (len(points) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(points))
                batch_points = points[start_idx:end_idx]
                
                logger.info(f" Uploading batch {batch_num + 1}/{total_batches} ({len(batch_points)} points)")
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
            
            logger.info(f"Successfully added {len(chunks)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        document_type: Optional[str] = None,
        chunk_focus: Optional[str] = None
    ) -> List[RetrievedDocument]:
        
        if len(query_embedding) != self.vector_size:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} doesn't match "
                f"expected size {self.vector_size}"
            )
        
        try:
            # Build search filter
            search_filter = None
            filter_conditions = []
            
            if document_type:
                filter_conditions.append(
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value=document_type)
                    )
                )
            
            if chunk_focus:
                filter_conditions.append(
                    models.FieldCondition(
                        key="chunk_focus", 
                        match=models.MatchValue(value=chunk_focus)
                    )
                )
            
            if filter_conditions:
                search_filter = models.Filter(
                    must=filter_conditions
                )
            
            # Perform vector search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            logger.info(f"Found {len(search_results)} similar documents")
            retrieved_docs = []
            
            for result in search_results:
                payload = result.payload
                
                retrieved_doc = RetrievedDocument(
                    document_id=payload["chunk_id"],
                    content=payload["content"],
                    similarity_score=result.score,
                    document_type=payload["document_type"],
                    metadata=payload.get("metadata", {}),
                    chunk_index=payload.get("chunk_index"),
                    parent_document_id=payload.get("parent_document_id")
                )
                
                retrieved_docs.append(retrieved_doc)
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_collection_stats(self, collection_name: str = "support_documents") -> Dict[str, Any]:
        """Get collection stats using direct HTTP calls."""
        import requests
        
        try:
            response = requests.get(f"http://{self.host}:{self.port}/collections/{collection_name}")
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("result", {})
                
                return {
                    "collection_name": collection_name,
                    "total_points": result.get("points_count", 0),
                    "status": result.get("status", "unknown"),
                    "config": result.get("config", {})
                }
            else:
                return {
                    "collection_name": collection_name,
                    "total_points": 0,
                    "status": "error",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "collection_name": collection_name,
                "total_points": 0,
                "status": "error",
                "error": str(e)
            }


    def delete_documents(self, chunk_ids: List[str]) -> bool:
        if not chunk_ids:
            logger.warning("No chunk IDs provided for deletion")
            return True
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_id",
                                match=models.MatchAny(any=chunk_ids)
                            )
                        ]
                    )
                )
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[])  # Empty filter matches all
                )
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False


# # Test the vector store service if run directly
# if __name__ == "__main__":
#     print("🧪 Testing Vector Store Service...")
#     print("=" * 50)
    
#     # Initialize service
#     print("\n🔌 Initializing Vector Store Service")
#     vector_store = VectorStoreService()
    
#     # Test collection creation
#     print("\n🏗️ Testing Collection Creation")
#     success = vector_store.create_collection(recreate=True)
#     print(f"   Collection created: {success}")
    
#     if success:
#         # Test collection stats
#         print("\n📊 Testing Collection Stats")
#         stats = vector_store.get_collection_stats()
#         print(f"   Stats: {stats}")
        
#         print("\n🎉 Vector Store Service is working correctly!")
#         print("\n💡 Next steps:")
#         print("   1. Implement embedding service to generate vectors")
#         print("   2. Load your processed documents into the vector store") 
#         print("   3. Test semantic search functionality")
#     else:
#         print("\n❌ Collection creation failed!")
