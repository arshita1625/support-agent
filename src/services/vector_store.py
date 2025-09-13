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
    print("❌ Qdrant client not installed. Install with: pip install qdrant-client")
    raise

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.document import Document, DocumentChunk
from models.rag import RetrievedDocument, RAGContext
from models.common import HealthStatus, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreService:
    """Service for managing document embeddings in Qdrant vector database."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "support_documents",
        vector_size: int = 1536,  # OpenAI text-embedding-ada-002 size
        distance_metric: str = "Cosine"
    ):
        """Initialize vector store service.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port  
            collection_name: Name of the document collection
            vector_size: Dimension of embedding vectors
            distance_metric: Distance metric for similarity (Cosine, Dot, Euclid)
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port)
        
        logger.info(f"🔌 Vector store initialized: {host}:{port}")
    
    async def check_health(self) -> HealthStatus:
        """Check vector store health and connectivity."""
        
        health = HealthStatus(status="healthy", version="1.0.0")
        
        try:
            # Check basic connectivity
            collections = self.client.get_collections()
            health.add_service_status("qdrant_connection", True)
            
            # Check if our collection exists
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            health.add_service_status("collection_exists", collection_exists)
            
            # If collection exists, check its health
            if collection_exists:
                try:
                    collection_info = self.client.get_collection(self.collection_name)
                    health.add_service_status("collection_healthy", True)
                    
                    # Add collection statistics
                    health.additional_context = {
                        "collection_name": self.collection_name,
                        "points_count": collection_info.points_count,
                        "vector_size": collection_info.config.params.vectors.size,
                        "distance": collection_info.config.params.vectors.distance.name
                    }
                    
                except Exception as e:
                    health.add_service_status("collection_healthy", False)
                    logger.error(f"Collection health check failed: {e}")
            
        except Exception as e:
            health.add_service_status("qdrant_connection", False)
            logger.error(f"Qdrant connection failed: {e}")
        
        # Update overall status
        health.update_overall_status()
        
        return health
    
    # Replace the create_collection method with this simpler version:

    def create_collection(self, recreate: bool = False) -> bool:
        """Create collection for storing document vectors."""
        
        try:
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if collection_exists:
                if recreate:
                    logger.info(f"🗑️ Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(collection_name=self.collection_name)
                else:
                    logger.info(f"✅ Collection already exists: {self.collection_name}")
                    return True
            
            logger.info(f"🏗️ Creating collection: {self.collection_name}")
            
            # Simple collection creation - works with all Qdrant versions
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            logger.info(f"✅ Collection created successfully: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection: {e}")
            return False

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> bool:
        """Add document chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of embedding vectors (same order as chunks)
            
        Returns:
            bool: True if documents added successfully
        """
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if not chunks:
            logger.warning("⚠️ No chunks provided to add")
            return True
        
        try:
            logger.info(f"📝 Adding {len(chunks)} documents to vector store...")
            
            # Prepare points for insertion
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
                
                logger.info(f"📤 Uploading batch {batch_num + 1}/{total_batches} ({len(batch_points)} points)")
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
            
            logger.info(f"✅ Successfully added {len(chunks)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False
    
    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        document_type: Optional[str] = None,
        chunk_focus: Optional[str] = None
    ) -> List[RetrievedDocument]:
        """Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            document_type: Filter by document type (optional)
            chunk_focus: Filter by chunk focus (optional)
            
        Returns:
            List of RetrievedDocument objects
        """
        
        if len(query_embedding) != self.vector_size:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} doesn't match "
                f"expected size {self.vector_size}"
            )
        
        try:
            logger.info(f"🔍 Searching for similar documents (limit: {limit})")
            
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
            
            logger.info(f"📊 Found {len(search_results)} similar documents")
            
            # Convert to RetrievedDocument objects
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
            logger.error(f"❌ Search failed: {e}")
            return []
    
   # In services/vector_store.py, replace the get_collection_stats method with this simpler version:

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            stats = {
                "collection_name": self.collection_name,
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "status": collection_info.status.name,
                "optimizer_status": collection_info.optimizer_status,
                "indexed_vectors_count": collection_info.indexed_vectors_count
            }
            
            # Skip the complex document type distribution for now
            # This was causing the connection issue
            stats["document_type_distribution"] = {}
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "total_points": 0
            }

    def delete_documents(self, chunk_ids: List[str]) -> bool:
        """Delete documents by their chunk IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            bool: True if deletion successful
        """
        
        if not chunk_ids:
            logger.warning("⚠️ No chunk IDs provided for deletion")
            return True
        
        try:
            logger.info(f"🗑️ Deleting {len(chunk_ids)} documents...")
            
            # Delete by chunk_id filter
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
            
            logger.info(f"✅ Successfully deleted documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection.
        
        Returns:
            bool: True if clearing successful
        """
        
        try:
            logger.info(f"🧹 Clearing all documents from collection: {self.collection_name}")
            
            # Delete all points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[])  # Empty filter matches all
                )
            )
            
            logger.info(f"✅ Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            return False


# Test the vector store service if run directly
if __name__ == "__main__":
    print("🧪 Testing Vector Store Service...")
    print("=" * 50)
    
    # Initialize service
    print("\n🔌 Initializing Vector Store Service")
    vector_store = VectorStoreService()
    
    # Test health check
    print("\n🏥 Testing Health Check")
    health = asyncio.run(vector_store.check_health())
    print(f"   Status: {health.status}")
    print(f"   Services: {health.services}")
    
    if not health.services.get("qdrant_connection", False):
        print("\n❌ Qdrant is not running!")
        print("Please start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        exit(1)
    
    # Test collection creation
    print("\n🏗️ Testing Collection Creation")
    success = vector_store.create_collection(recreate=True)
    print(f"   Collection created: {success}")
    
    if success:
        # Test collection stats
        print("\n📊 Testing Collection Stats")
        stats = vector_store.get_collection_stats()
        print(f"   Stats: {stats}")
        
        print("\n🎉 Vector Store Service is working correctly!")
        print("\n💡 Next steps:")
        print("   1. Implement embedding service to generate vectors")
        print("   2. Load your processed documents into the vector store") 
        print("   3. Test semantic search functionality")
    else:
        print("\n❌ Collection creation failed!")
