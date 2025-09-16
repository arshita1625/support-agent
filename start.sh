#!/bin/bash

echo "🚀 Starting RAG Support Agent..."

# Wait for Qdrant to be fully ready
echo "⏳ Waiting for Qdrant to be ready..."
until curl -s http://qdrant:6333/collections > /dev/null; do
  echo "Waiting for Qdrant..."
  sleep 2
done

echo "✅ Qdrant is ready!"

# Check if documents are already loaded
echo "📊 Checking if documents are already loaded..."
COLLECTION_EXISTS=$(python -c "
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(host='qdrant', port=6333)
    collections = client.get_collections().collections
    if any(col.name == 'support_documents' for col in collections):
        count = client.count('support_documents')
        print(count.count)
    else:
        print(0)
except Exception as e:
    print(0)
" 2>/dev/null)

if [ "$COLLECTION_EXISTS" -eq "0" ]; then
    echo "📄 No documents found. Loading documents into Qdrant..."
    
    # Load documents from your data directory
    python -c "
import os
import sys
sys.path.append('/app')

try:
    # Import your document loading function
    from services.document_processor import DocumentProcessor
    from services.vector_store import VectorStore
    
    # Initialize services
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Load documents from data directory
    data_path = '/app/data/documents'
    if os.path.exists(data_path):
        print(f'📂 Loading documents from {data_path}')
        # Add your specific document loading logic here
        # This depends on your DocumentProcessor implementation
        doc_processor.load_and_process_documents(data_path)
        print('✅ Documents loaded successfully!')
    else:
        print('❌ Data directory not found at /app//documents')
        
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('Make sure your document processor is properly configured')
except Exception as e:
    print(f'❌ Error loading documents: {e}')
"

    # Verify documents were loaded
    echo "🔍 Verifying document load..."
    python -c "
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(host='qdrant', port=6333)
    count = client.count('support_documents')
    print(f'✅ Successfully loaded {count.count} documents')
except Exception as e:
    print(f'❌ Verification failed: {e}')
"
else
    echo "✅ Found $COLLECTION_EXISTS documents already loaded"
fi

echo "🎯 Starting RAG Support Agent API..."

# Start the FastAPI application
exec python main.py
