#!/usr/bin/env python3
"""Document processor service for loading documents into vector store."""
import os
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.document import Document, DocumentChunk
from models.rag import RetrievedDocument, RAGContext
from models.common import ErrorResponse
from services.vector_store import VectorStoreService
from services.embedding import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessorService:
    """Service for processing and indexing documents with embeddings."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        processed_docs_path: str = "data/processed"
    ):
        """Initialize document processor service.
        
        Args:
            vector_store: Vector store service instance
            embedding_service: Embedding service instance  
            processed_docs_path: Path to processed documents directory
        """
        
        self.processed_docs_path = Path(processed_docs_path)
        
        # Initialize services if not provided
        self.vector_store = vector_store or VectorStoreService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.last_update_check = None
        self._documents_current = False

        # Ensure vector store vector size matches embedding dimensions
        embedding_dims = self.embedding_service.get_embedding_dimensions()
        if self.vector_store.vector_size != embedding_dims:
            logger.warning(
                f"Vector size mismatch: vector store expects {self.vector_store.vector_size}, "
                f"embedding service provides {embedding_dims}. Updating vector store."
            )
            self.vector_store.vector_size = embedding_dims
        
        logger.info("📋 Document processor service initialized")
    
    def get_source_files_info(self) -> dict:
        """Get modification times of all source document files."""
        source_files = {}
        documents_dir = Path("data/documents")
        
        if documents_dir.exists():
            for file_path in documents_dir.glob("*.md"):
                source_files[str(file_path)] = {
                    'path': str(file_path),
                    'modified': os.path.getmtime(file_path),
                    'size': file_path.stat().st_size
                }
        
        return source_files
    
    def should_update_documents(self) -> tuple[bool, str]:
        """Check if documents need updating based on file changes."""
        
        # Get current source file info
        current_sources = self.get_source_files_info()
        
        if not current_sources:
            return False, "No source documents found"
        
        # Check if processed files exist
        processed_chunks = Path("data/processed/chunks.json")
        processed_docs = Path("data/processed/documents.json")
        
        if not (processed_chunks.exists() and processed_docs.exists()):
            return True, "No processed documents found"
        
        # Get processed files modification time
        processed_time = min(
            os.path.getmtime(processed_chunks),
            os.path.getmtime(processed_docs)
        )
        
        # Check if any source file is newer than processed files
        for file_info in current_sources.values():
            if file_info['modified'] > processed_time:
                file_path = file_info['path']
                return True, f"Source file {file_path} is newer than processed documents"
        
        # Check vector store has correct number of documents
        try:
            with open(processed_chunks, 'r') as f:
                chunks = json.load(f)
            expected_count = len(chunks)
            
            from services.vector_store import VectorStoreService
            vs = VectorStoreService()
            stats = vs.get_collection_stats()
            actual_count = stats.get("total_points", 0)
            
            if actual_count != expected_count:
                return True, f"Vector store has {actual_count} documents, expected {expected_count}"
                
        except Exception as e:
            return True, f"Cannot verify vector store: {e}"
        
        return False, "Documents are up to date"
    
    async def auto_update_documents_if_needed(self) -> dict:
        """Automatically update documents if source files have changed."""
        
        needs_update, reason = self.should_update_documents()
        
        if not needs_update:
            logger.info(f"✅ {reason}")
            return {"success": True, "updated": False, "reason": reason}
        
        logger.info(f"🔄 Auto-updating documents: {reason}")
        
        try:
            # Step 1: Clear old processed files
            processed_dir = Path("data/processed")
            for file in processed_dir.glob("*.json"):
                file.unlink(missing_ok=True)
            logger.info("🧹 Cleared old processed files")
            
            # Step 2: Run document processing
            import subprocess
            import sys
            
            result = subprocess.run(
                [sys.executable, "scripts/load_documents.py"],
                capture_output=True,
                text=True,
                cwd="."
            )
            
            if result.returncode != 0:
                error_msg = f"Document processing failed: {result.stderr}"
                logger.error(f" {error_msg}")
                return {"success": False, "error": error_msg}
            
            logger.info("✅ Document processing completed")
            
            # Step 3: Re-index in vector store
            logger.info("🔄 Re-indexing vector store...")
            results = await self.process_and_index_documents(recreate_collection=True)
            
            if results.get('success'):
                chunks_indexed = results.get('chunks_indexed', 0)
                logger.info(f"✅ Successfully indexed {chunks_indexed} chunks")
                
                return {
                    "success": True,
                    "updated": True,
                    "reason": reason,
                    "chunks_indexed": chunks_indexed,
                    "cost": results.get('embedding_cost_estimate', 0)
                }
            else:
                error_msg = f"Vector indexing failed: {results.get('errors', [])}"
                logger.error(f"❌ {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Auto-update failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}

    def load_processed_documents(self) -> Tuple[List[Document], List[DocumentChunk]]:
        """Load documents and chunks from processed files.
        
        Returns:
            Tuple of (documents, chunks)
        """
        
        docs_file = self.processed_docs_path / "documents.json"
        chunks_file = self.processed_docs_path / "chunks.json"
        
        logger.info(f"📂 Loading processed documents from {self.processed_docs_path}")
        
        # Check if files exist
        if not docs_file.exists():
            raise FileNotFoundError(f"Documents file not found: {docs_file}")
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        # Load documents
        try:
            with open(docs_file, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
            
            documents = [Document.from_dict(doc_data) for doc_data in docs_data]
            logger.info(f"✅ Loaded {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise
        
        # Load chunks
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            chunks = [DocumentChunk.from_dict(chunk_data) for chunk_data in chunks_data]
            logger.info(f"✅ Loaded {len(chunks)} document chunks")
            
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            raise
        
        return documents, chunks
    
    async def process_and_index_documents(
        self,
        recreate_collection: bool = False,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process documents and index them in the vector store.
        
        Args:
            recreate_collection: Whether to recreate the collection
            batch_size: Batch size for processing (uses service default if None)
            
        Returns:
            Processing results and statistics
        """
        
        start_time = time.time()
        logger.info("🚀 Starting document processing and indexing...")
        
        results = {
            "success": False,
            "documents_loaded": 0,
            "chunks_processed": 0,
            "chunks_indexed": 0,
            "processing_time_seconds": 0,
            "embedding_cost_estimate": 0.0,
            "errors": []
        }
        
        try:
            # 1. Load processed documents
            logger.info("📂 Loading processed documents...")
            documents, chunks = self.load_processed_documents()
            
            results["documents_loaded"] = len(documents)
            results["chunks_processed"] = len(chunks)
            
            if not chunks:
                logger.warning("⚠️ No chunks to process")
                return results
            
            # 2. Setup vector store collection
            logger.info("🏗️ Setting up vector store collection...")
            collection_success = self.vector_store.create_collection(recreate=recreate_collection)
            
            if not collection_success:
                raise RuntimeError("Failed to create vector store collection")
            
            # 3. Generate embeddings for chunks
            logger.info("🤖 Generating embeddings for document chunks...")
            
            # Set batch size if provided
            if batch_size:
                original_batch_size = self.embedding_service.batch_size
                self.embedding_service.batch_size = batch_size
            
            try:
                # Calculate cost estimate before processing
                chunk_texts = [chunk.content for chunk in chunks]
                cost_estimate = self.embedding_service.calculate_cost_estimate(chunk_texts)
                results["embedding_cost_estimate"] = cost_estimate["estimated_cost_usd"]
                
                logger.info(f"💰 Estimated embedding cost: ${cost_estimate['estimated_cost_usd']:.4f}")
                logger.info(f"📊 Processing {cost_estimate['text_count']} texts in {cost_estimate['batch_count']} batches")
                
                # Generate embeddings
                embedding_start = time.time()
                valid_chunks, embeddings = await self.embedding_service.embed_document_chunks(chunks)
                embedding_time = time.time() - embedding_start
                
                logger.info(f"✅ Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")
                
            finally:
                # Restore original batch size
                if batch_size:
                    self.embedding_service.batch_size = original_batch_size
            
            # 4. Index chunks with embeddings in vector store
            logger.info("💾 Indexing chunks in vector store...")
            
            indexing_start = time.time()
            indexing_success = self.vector_store.add_documents(valid_chunks, embeddings)
            indexing_time = time.time() - indexing_start
            
            if not indexing_success:
                raise RuntimeError("Failed to index documents in vector store")
            
            results["chunks_indexed"] = len(valid_chunks)
            logger.info(f"✅ Indexed {len(valid_chunks)} chunks in {indexing_time:.2f}s")
            
            # 5. Verify indexing
            logger.info("🔍 Verifying indexing...")
            collection_stats = self.vector_store.get_collection_stats()
            
            if collection_stats.get("total_points", 0) > 0:
                logger.info(f"✅ Verification successful: {collection_stats['total_points']} points in collection")
                results["success"] = True
            else:
                logger.error("❌ Verification failed: no points found in collection")
                results["errors"].append("Indexing verification failed")
            
            # Add collection stats to results
            results["collection_stats"] = collection_stats
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            results["errors"].append(str(e))
            raise
        
        finally:
            results["processing_time_seconds"] = time.time() - start_time
        
        processing_time = results["processing_time_seconds"]
        logger.info(f"🎉 Document processing completed in {processing_time:.2f}s")
        
        return results
    
    async def search_documents(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
        document_type: Optional[str] = None
    ) -> RAGContext:
        """Search for documents using semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            document_type: Filter by document type
            
        Returns:
            RAGContext with retrieved documents
        """
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"🔍 Searching for: '{query[:50]}...'")
        
        search_start = time.time()
        
        try:
            # 1. Generate query embedding
            embedding_start = time.time()
            query_embedding = await self.embedding_service.generate_embedding(query)
            embedding_time = time.time() - embedding_start
            
            logger.debug(f"Generated query embedding in {embedding_time*1000:.1f}ms")
            
            # 2. Search vector store
            retrieval_start = time.time()
            retrieved_docs = self.vector_store.search(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                document_type=document_type
            )
            retrieval_time = time.time() - retrieval_start
            
            logger.info(f"📊 Found {len(retrieved_docs)} relevant documents")
            
            # 3. Get collection stats for context
            collection_stats = self.vector_store.get_collection_stats()
            total_documents = collection_stats.get("total_points", 0)
            
            # 4. Create RAG context
            context = RAGContext(
                query=query.strip(),
                retrieved_documents=retrieved_docs,
                total_documents_searched=total_documents,
                query_embedding_time_ms=embedding_time * 1000,
                retrieval_time_ms=retrieval_time * 1000
            )
            
            search_time = time.time() - search_start
            logger.info(f"✅ Search completed in {search_time*1000:.1f}ms")
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            raise
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status and statistics."""
        
        try:
            collection_stats = self.vector_store.get_collection_stats()
            
            status = {
                "vector_store_healthy": True,
                "collection_name": collection_stats.get("collection_name", "unknown"),
                "indexed_documents": collection_stats.get("total_points", 0),
                "collection_status": collection_stats.get("status", "unknown"),
                "vector_dimensions": collection_stats.get("vector_size", 0),
                "distance_metric": collection_stats.get("distance_metric", "unknown"),
                "document_types": collection_stats.get("document_type_distribution", {}),
                "embedding_model": self.embedding_service.model,
                "last_updated": datetime.now().isoformat()
            }
            
            # Add processed documents info
            try:
                chunks_file = self.processed_docs_path / "chunks.json"
                if chunks_file.exists():
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        chunks_data = json.load(f)
                    status["available_chunks"] = len(chunks_data)
                    status["indexed_percentage"] = (
                        (status["indexed_documents"] / len(chunks_data)) * 100 
                        if chunks_data else 0
                    )
                else:
                    status["available_chunks"] = 0
                    status["indexed_percentage"] = 0
            except:
                status["available_chunks"] = "unknown"
                status["indexed_percentage"] = "unknown"
            
            return status
            
        except Exception as e:
            return {
                "vector_store_healthy": False,
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    async def test_end_to_end_search(self, test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Test end-to-end search functionality with sample queries.
        
        Args:
            test_queries: List of test queries (uses defaults if None)
            
        Returns:
            Test results and performance metrics
        """
        
        if test_queries is None:
            test_queries = [
                "domain suspension",
                "WHOIS information update",
                "billing payment issue",
                "reactivate suspended domain",
                "technical DNS problem"
            ]
        
        logger.info(f"🧪 Running end-to-end search tests with {len(test_queries)} queries")
        
        test_results = {
            "total_queries": len(test_queries),
            "successful_searches": 0,
            "failed_searches": 0,
            "average_search_time_ms": 0,
            "average_results_per_query": 0,
            "query_results": []
        }
        
        total_search_time = 0
        total_results = 0
        
        for i, query in enumerate(test_queries):
            logger.info(f"Testing query {i+1}/{len(test_queries)}: '{query}'")
            
            try:
                start_time = time.time()
                context = await self.search_documents(query, limit=3)
                search_time = (time.time() - start_time) * 1000
                
                result = {
                    "query": query,
                    "success": True,
                    "search_time_ms": round(search_time, 2),
                    "results_found": len(context.retrieved_documents),
                    "average_similarity": round(context.average_similarity, 3),
                    "max_similarity": round(context.max_similarity, 3),
                    "top_result_preview": (
                        context.retrieved_documents[0].get_preview(100) 
                        if context.retrieved_documents else None
                    )
                }
                
                test_results["successful_searches"] += 1
                total_search_time += search_time
                total_results += len(context.retrieved_documents)
                
                logger.info(f"✅ Found {len(context.retrieved_documents)} results (avg similarity: {context.average_similarity:.3f})")
                
            except Exception as e:
                result = {
                    "query": query,
                    "success": False,
                    "error": str(e),
                    "search_time_ms": 0,
                    "results_found": 0
                }
                
                test_results["failed_searches"] += 1
                logger.error(f"❌ Search failed: {e}")
            
            test_results["query_results"].append(result)
        
        # Calculate averages
        if test_results["successful_searches"] > 0:
            test_results["average_search_time_ms"] = round(
                total_search_time / test_results["successful_searches"], 2
            )
            test_results["average_results_per_query"] = round(
                total_results / test_results["successful_searches"], 1
            )
        
        logger.info(f"🎉 Search tests completed: {test_results['successful_searches']}/{test_results['total_queries']} successful")
        
        return test_results


# Test the document processor service if run directly
if __name__ == "__main__":
    print("🧪 Testing Document Processor Service...")
    print("=" * 60)
    
    # Check dependencies
    import os
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    async def run_tests():
        # Initialize service
        print("\n📋 Initializing Document Processor Service")
        processor = DocumentProcessorService()
        # Check if we have processed documents to work with
        try:
            documents, chunks = processor.load_processed_documents()
            print(f"\n📂 Found {len(documents)} documents with {len(chunks)} chunks")
        except FileNotFoundError as e:
            print(f"\n❌ {e}")
            print("\n💡 Run 'python scripts/load_documents.py' first to process documents")
            return
        
        # Test document processing and indexing
        print("\n🚀 Testing Document Processing and Indexing")
        try:
            results = await processor.process_and_index_documents(recreate_collection=True)
            
            if results["success"]:
                print(f"   ✅ Successfully processed {results['chunks_processed']} chunks")
                print(f"   ✅ Indexed {results['chunks_indexed']} chunks")
                print(f"   ⏱️ Processing time: {results['processing_time_seconds']:.2f}s")
                print(f"   💰 Estimated cost: ${results['embedding_cost_estimate']:.4f}")
            else:
                print(f"   ❌ Processing failed: {results.get('errors', [])}")
                return
                
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            return
        
        # Test search functionality
        print("\n🔍 Testing Search Functionality")
        test_queries = [
            "domain suspension",
            "WHOIS information",
            "billing issue"
        ]
        
        for query in test_queries:
            try:
                context = await processor.search_documents(query, limit=2)
                print(f"   Query: '{query}'")
                print(f"     Results: {len(context.retrieved_documents)}")
                if context.retrieved_documents:
                    top_result = context.retrieved_documents[0]
                    print(f"     Top match: {top_result.similarity_score:.3f} - {top_result.get_preview(60)}")
                print()
            except Exception as e:
                print(f"   ❌ Search failed for '{query}': {e}")
        
        # Test end-to-end functionality
        print("\n🧪 Running End-to-End Tests")
        try:
            test_results = await processor.test_end_to_end_search()
            print(f"   Success rate: {test_results['successful_searches']}/{test_results['total_queries']}")
            print(f"   Average search time: {test_results['average_search_time_ms']:.1f}ms")
            print(f"   Average results per query: {test_results['average_results_per_query']}")
        except Exception as e:
            print(f"   ❌ End-to-end test failed: {e}")
        
        # Show processing status
        print("\n📊 Processing Status")
        status = processor.get_processing_status()
        print(f"   Collection: {status.get('collection_name', 'unknown')}")
        print(f"   Indexed documents: {status.get('indexed_documents', 0)}")
        print(f"   Available chunks: {status.get('available_chunks', 0)}")
        print(f"   Index completion: {status.get('indexed_percentage', 0):.1f}%")
        print(f"   Embedding model: {status.get('embedding_model', 'unknown')}")
        
        print("\n🎉 All Document Processor tests completed!")
        print("\n💡 Next steps:")
        print("   1. Your documents are now indexed and searchable")
        print("   2. Implement the main RAG service for query processing")
        print("   3. Create API endpoints for the RAG system")
    
    # Run async tests
    asyncio.run(run_tests())
