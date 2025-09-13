#!/usr/bin/env python3
"""Embedding service for converting text to vector embeddings."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
try:
    import openai
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI client not installed. Install with: pip install openai")
    raise

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.document import Document, DocumentChunk
from models.common import HealthStatus, ErrorResponse
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings using OpenAI's API."""
    
    def __init__(
        self,
        api_key: Optional[str] = API_KEY,
        model: str = "text-embedding-ada-002",
        max_tokens: int = 8191,
        batch_size: int = 100,
        rate_limit_rpm: int = 3000  # Requests per minute
    ):
        """Initialize embedding service.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: Embedding model to use
            max_tokens: Maximum tokens per text chunk
            batch_size: Number of texts to process in parallel
            rate_limit_rpm: Rate limit in requests per minute
        """
        self.model = model
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.rate_limit_rpm = rate_limit_rpm
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Rate limiting
        self.min_request_interval = 60.0 / rate_limit_rpm  # Seconds between requests
        self.last_request_time = 0.0
        
        # Model specifications
        self.model_specs = {
            "text-embedding-ada-002": {
                "dimensions": 1536,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.0001
            },
            "text-embedding-3-small": {
                "dimensions": 1536,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.00002
            },
            "text-embedding-3-large": {
                "dimensions": 3072,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.00013
            }
        }
        
        logger.info(f"🤖 Embedding service initialized with model: {model}")
    
    def get_embedding_dimensions(self) -> int:
        """Get the vector dimensions for the current model."""
        return self.model_specs.get(self.model, {}).get("dimensions", 1536)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "model": self.model,
            "dimensions": self.get_embedding_dimensions(),
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "rate_limit_rpm": self.rate_limit_rpm,
            **self.model_specs.get(self.model, {})
        }
    
    async def check_health(self) -> HealthStatus:
        """Check embedding service health and connectivity."""
        
        health = HealthStatus(status="healthy", version="1.0.0")
        
        try:
            # Test API connectivity with a simple embedding request
            start_time = time.time()
            test_embedding = await self.generate_embeddings(["test"])
            response_time = time.time() - start_time
            
            if test_embedding and len(test_embedding) == 1:
                health.add_service_status("openai_api", True)
                health.add_service_status("embedding_generation", True)
                
                # Add performance metrics
                health.additional_context = {
                    "model": self.model,
                    "dimensions": self.get_embedding_dimensions(),
                    "test_response_time_ms": round(response_time * 1000, 2),
                    "rate_limit_rpm": self.rate_limit_rpm
                }
            else:
                health.add_service_status("openai_api", True)
                health.add_service_status("embedding_generation", False)
                
        except Exception as e:
            health.add_service_status("openai_api", False)
            health.add_service_status("embedding_generation", False)
            logger.error(f"OpenAI API health check failed: {e}")
        
        # Update overall status
        health.update_overall_status()
        
        return health
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Clean and normalize text
        cleaned = text.strip()
        
        if not cleaned:
            raise ValueError("Input text cannot be empty")
        
        # Replace newlines with spaces for better embedding quality
        cleaned = " ".join(cleaned.split())
        
        # Truncate if too long (rough token estimation: 1 token ≈ 4 characters)
        estimated_tokens = len(cleaned) // 4
        if estimated_tokens > self.max_tokens:
            # Truncate to approximate token limit
            max_chars = self.max_tokens * 4
            cleaned = cleaned[:max_chars]
            logger.warning(f"Text truncated from {len(text)} to {len(cleaned)} characters")
        
        return cleaned
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floating point numbers representing the embedding
        """
        
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Enforce rate limiting
            self._enforce_rate_limit()
            
            # Generate embedding
            logger.debug(f"Generating embedding for text: {processed_text[:100]}...")
            
            response = self.client.embeddings.create(
                input=[processed_text],
                model=self.model
            )
            
            embedding = response.data[0].embedding
            
            # Validate embedding
            expected_dim = self.get_embedding_dimensions()
            if len(embedding) != expected_dim:
                raise ValueError(
                    f"Unexpected embedding dimension: got {len(embedding)}, "
                    f"expected {expected_dim}"
                )
            
            logger.debug(f"Generated {len(embedding)}-dimensional embedding")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embeddings in the same order as input texts
        """
        
        if not texts:
            return []
        
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All inputs must be strings")
        
        # Filter out empty texts and preprocess
        processed_data = []
        for i, text in enumerate(texts):
            try:
                if text and text.strip():
                    processed_text = self._preprocess_text(text)
                    processed_data.append((i, processed_text))
                else:
                    logger.warning(f"Skipping empty text at index {i}")
            except ValueError as e:
                logger.warning(f"Skipping invalid text at index {i}: {e}")
        
        if not processed_data:
            logger.warning("No valid texts to process")
            return []
        
        # Process in batches for better performance
        all_embeddings = {}  # Dictionary to maintain order
        total_batches = (len(processed_data) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {len(processed_data)} texts in {total_batches} batches")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(processed_data))
            batch_data = processed_data[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_data)} items)")
            
            try:
                # Enforce rate limiting
                self._enforce_rate_limit()
                
                # Prepare batch
                batch_texts = [item[1] for item in batch_data]
                
                # Generate embeddings for batch
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model
                )
                
                # Store embeddings with original indices
                for (original_idx, _), embedding_data in zip(batch_data, response.data):
                    all_embeddings[original_idx] = embedding_data.embedding
                
                logger.debug(f"Generated {len(response.data)} embeddings in batch")
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num + 1}: {e}")
                # Continue with next batch rather than failing completely
                continue
        
        # Reconstruct embeddings list in original order
        embeddings = []
        for i in range(len(texts)):
            if i in all_embeddings:
                embeddings.append(all_embeddings[i])
            else:
                # Placeholder for failed/empty texts
                embeddings.append([0.0] * self.get_embedding_dimensions())
                logger.warning(f"Using zero embedding for text at index {i}")
        
        logger.info(f"Successfully generated {len([e for e in embeddings if any(e)])} embeddings")
        
        return embeddings
    
    async def embed_document_chunks(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], List[List[float]]]:
        """Generate embeddings for document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Tuple of (chunks, embeddings) in the same order
        """
        
        if not chunks:
            return [], []
        
        logger.info(f"Generating embeddings for {len(chunks)} document chunks...")
        
        # Extract text from chunks
        chunk_texts = []
        valid_chunks = []
        
        for chunk in chunks:
            if chunk.content and chunk.content.strip():
                chunk_texts.append(chunk.content)
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Skipping chunk {chunk.chunk_id} with empty content")
        
        if not chunk_texts:
            logger.warning("No valid chunks to process")
            return [], []
        
        # Generate embeddings
        start_time = time.time()
        embeddings = await self.generate_embeddings(chunk_texts)
        processing_time = time.time() - start_time
        
        logger.info(f"Generated {len(embeddings)} embeddings in {processing_time:.2f}s")
        
        # Calculate cost estimate
        total_tokens = sum(len(text.split()) for text in chunk_texts)
        model_cost = self.model_specs.get(self.model, {}).get("cost_per_1k_tokens", 0.0001)
        estimated_cost = (total_tokens / 1000) * model_cost
        
        logger.info(f"Estimated cost: ${estimated_cost:.4f} ({total_tokens} tokens)")
        
        return valid_chunks, embeddings
    
    def calculate_cost_estimate(self, texts: List[str]) -> Dict[str, Any]:
        """Calculate cost estimate for embedding generation.
        
        Args:
            texts: List of texts to estimate cost for
            
        Returns:
            Dictionary with cost breakdown
        """
        
        # Rough token estimation (1 token ≈ 4 characters for English)
        total_chars = sum(len(text) for text in texts if text)
        estimated_tokens = total_chars // 4
        
        model_cost = self.model_specs.get(self.model, {}).get("cost_per_1k_tokens", 0.0001)
        estimated_cost = (estimated_tokens / 1000) * model_cost
        
        return {
            "model": self.model,
            "text_count": len(texts),
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "cost_per_1k_tokens": model_cost,
            "estimated_cost_usd": round(estimated_cost, 4),
            "batch_count": (len(texts) + self.batch_size - 1) // self.batch_size
        }


# Test the embedding service if run directly
if __name__ == "__main__":
    print("🧪 Testing Embedding Service...")
    print("=" * 50)
    
    # Check if API key is available
    import os
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    # Initialize service
    print("\n🤖 Initializing Embedding Service")
    embedding_service = EmbeddingService(
        model="text-embedding-ada-002",
        batch_size=3  # Small batch for testing
    )
    
    print(f"Model info: {embedding_service.get_model_info()}")
    
    # Test health check
    print("\n🏥 Testing Health Check")
    health = asyncio.run(embedding_service.check_health())
    print(f"   Status: {health.status}")
    print(f"   Services: {health.services}")
    
    if not health.services.get("openai_api", False):
        print("\n❌ OpenAI API is not accessible!")
        print("Please check your API key and internet connection.")
        exit(1)
    
    # Test single embedding
    print("\n🔤 Testing Single Embedding")
    test_text = "Domain suspension occurs when WHOIS information is incomplete or inaccurate."
    
    try:
        embedding = asyncio.run(embedding_service.generate_embedding(test_text))
        print(f"   Generated {len(embedding)}-dimensional embedding")
        print(f"   First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"   ❌ Single embedding failed: {e}")
        exit(1)
    
    # Test batch embeddings
    print("\n📦 Testing Batch Embeddings")
    test_texts = [
        "Your domain was suspended due to incomplete WHOIS information.",
        "To reactivate your domain, please update your contact details.",
        "Domain transfers typically take 5-7 business days to complete."
    ]
    
    try:
        embeddings = asyncio.run(embedding_service.generate_embeddings(test_texts))
        print(f"   Generated {len(embeddings)} embeddings")
        print(f"   All embeddings have {len(embeddings[0])} dimensions: {all(len(e) == len(embeddings[0]) for e in embeddings)}")
    except Exception as e:
        print(f"   ❌ Batch embedding failed: {e}")
        exit(1)
    
    # Test cost estimation
    print("\n💰 Testing Cost Estimation")
    cost_estimate = embedding_service.calculate_cost_estimate(test_texts)
    print(f"   Estimated cost: ${cost_estimate['estimated_cost_usd']}")
    print(f"   Tokens: {cost_estimate['estimated_tokens']}")
    print(f"   Batches: {cost_estimate['batch_count']}")
    
    print("\n🎉 Embedding Service is working correctly!")
    print("\n💡 Next steps:")
    print("   1. Use this service to embed your processed document chunks")
    print("   2. Store the embeddings in your vector store") 
    print("   3. Test end-to-end semantic search")
