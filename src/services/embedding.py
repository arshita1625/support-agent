#!/usr/bin/env python3
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
try:
    import openai
    from openai import OpenAI
except ImportError:
    print("OpenAI client not installed. Install with: pip install openai")
    raise

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.document import Document, DocumentChunk
from models.common import ErrorResponse
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    
    def __init__(
        self,
        api_key: Optional[str] = API_KEY,
        model: str = "text-embedding-ada-002",
        max_tokens: int = 8191,
        batch_size: int = 100,
        rate_limit_rpm: int = 3000  
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.rate_limit_rpm = rate_limit_rpm
        
        
        self.client = OpenAI(api_key=api_key)
        
        
        self.min_request_interval = 60.0 / rate_limit_rpm  
        self.last_request_time = 0.0
        
        
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
    
    def get_embedding_dimensions(self) -> int:
        
        return self.model_specs.get(self.model, {}).get("dimensions", 1536)
    
    def get_model_info(self) -> Dict[str, Any]:
        
        return {
            "model": self.model,
            "dimensions": self.get_embedding_dimensions(),
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "rate_limit_rpm": self.rate_limit_rpm,
            **self.model_specs.get(self.model, {})
        }

    
    def _enforce_rate_limit(self):
        
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _preprocess_text(self, text: str) -> str:
        
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        
        cleaned = text.strip()
        
        if not cleaned:
            raise ValueError("Input text cannot be empty")
        
        
        cleaned = " ".join(cleaned.split())
        
        
        estimated_tokens = len(cleaned) // 4
        if estimated_tokens > self.max_tokens:
            
            max_chars = self.max_tokens * 4
            cleaned = cleaned[:max_chars]
        
        return cleaned
    
    async def generate_embedding(self, text: str) -> List[float]:
        
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        try:
            
            processed_text = self._preprocess_text(text)
            
            
            self._enforce_rate_limit()
            
            
            logger.debug(f"Generating embedding for text: {processed_text[:100]}...")
            
            response = self.client.embeddings.create(
                input=[processed_text],
                model=self.model
            )
            
            embedding = response.data[0].embedding
            
            
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
        
        if not texts:
            return []
        
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All inputs must be strings")
        
        
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
        
        
        all_embeddings = {}  
        total_batches = (len(processed_data) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {len(processed_data)} texts in {total_batches} batches")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(processed_data))
            batch_data = processed_data[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_data)} items)")
            
            try:
                
                self._enforce_rate_limit()   
                batch_texts = [item[1] for item in batch_data]
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model
                )
                
                for (original_idx, _), embedding_data in zip(batch_data, response.data):
                    all_embeddings[original_idx] = embedding_data.embedding
                
                logger.debug(f"Generated {len(response.data)} embeddings in batch")
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num + 1}: {e}")
                
                continue
        
        embeddings = []
        for i in range(len(texts)):
            if i in all_embeddings:
                embeddings.append(all_embeddings[i])
            else:
                
                embeddings.append([0.0] * self.get_embedding_dimensions())
                logger.warning(f"Using zero embedding for text at index {i}")
        
        logger.info(f"Successfully generated {len([e for e in embeddings if any(e)])} embeddings")
        
        return embeddings
    
    async def embed_document_chunks(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], List[List[float]]]:
        
        if not chunks:
            return [], []
        
        logger.info(f"Generating embeddings for {len(chunks)} document chunks...")
        
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
        
        start_time = time.time()
        embeddings = await self.generate_embeddings(chunk_texts)
        processing_time = time.time() - start_time
        
        logger.info(f"Generated {len(embeddings)} embeddings in {processing_time:.2f}s")
        
        return valid_chunks, embeddings
