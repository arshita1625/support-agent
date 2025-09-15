#!/usr/bin/env python3
import sys
from pathlib import Path
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from services.document_processor import DocumentProcessorService

class TestDocumentProcessor:
    
    @pytest.fixture
    def mock_embedding_service(self):
        mock = AsyncMock()
        mock.generate_embedding.return_value = [0.1] * 1536
        mock.generate_embeddings.return_value = [[0.1] * 1536, [0.2] * 1536]
        return mock
    
    @pytest.fixture
    def mock_vector_store(self):
        mock = MagicMock()
        mock.create_collection.return_value = True
        mock.get_collection_stats.return_value = {"total_points": 5}
        mock.add_documents.return_value = True
        return mock
    
    @pytest.fixture
    def processor(self):
        mock_embedding = MagicMock()  
        mock_vector = MagicMock()
        with patch('services.document_processor.EmbeddingService', return_value=mock_embedding), \
            patch('services.document_processor.VectorStoreService', return_value=mock_vector):
            return DocumentProcessorService()

    def test_get_source_files_info_basic(self, processor):
        with patch('pathlib.Path') as mock_path:
            
            mock_path.return_value.glob.return_value = []
            
            source_info = processor.get_source_files_info()
            
            assert isinstance(source_info, dict)
    
    def test_should_update_documents_basic(self, processor):
    
        with patch.object(processor, 'get_source_files_info', return_value={}):
            result = processor.should_update_documents()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], str)
    
    @pytest.mark.asyncio
    async def test_auto_update_documents_if_needed_exists(self, processor):
        with patch.object(processor, 'should_update_documents', return_value=(False, "Up to date")):
            result = await processor.auto_update_documents_if_needed()
            
            assert isinstance(result, dict)
            assert "success" in result
            assert "updated" in result
    
    def test_processor_initialization(self, processor):
        assert processor is not None
        assert hasattr(processor, 'get_source_files_info')
        assert hasattr(processor, 'should_update_documents')

class TestBasicFunctionality:

    def test_processor_can_be_created(self):
        with patch('services.document_processor.EmbeddingService'), \
             patch('services.document_processor.VectorStoreService'):
            processor = DocumentProcessorService()
            assert processor is not None
    
    def test_processor_has_required_methods(self):
    
        with patch('services.document_processor.EmbeddingService'), \
             patch('services.document_processor.VectorStoreService'):
            processor = DocumentProcessorService()
            
            
            required_methods = [
                'get_source_files_info',
                'should_update_documents', 
                'auto_update_documents_if_needed'
            ]
            
            for method_name in required_methods:
                assert hasattr(processor, method_name), f"Missing method: {method_name}"

class TestMethodSignatures:
    @pytest.fixture
    def processor(self):
        with patch('services.document_processor.EmbeddingService'), \
             patch('services.document_processor.VectorStoreService'):
            return DocumentProcessorService()
    
    def test_get_source_files_info_signature(self, processor):
        import inspect
        
        
        assert callable(getattr(processor, 'get_source_files_info', None))
        
        
        sig = inspect.signature(processor.get_source_files_info)
        
        
        params = [p for name, p in sig.parameters.items() if name != 'self']
        assert len(params) == 0, f"Expected 0 params, got {len(params)}: {[p.name for p in params]}"
    
    @pytest.mark.asyncio
    async def test_auto_update_method_signature(self, processor):
        import inspect
        
        method = getattr(processor, 'auto_update_documents_if_needed', None)
        assert method is not None, "auto_update_documents_if_needed method not found"
        assert inspect.iscoroutinefunction(method), "auto_update_documents_if_needed should be async"
