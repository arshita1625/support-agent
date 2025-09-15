#!/bin/bash

echo "🚀 Starting RAG Support Agent..."

# Activate environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start Qdrant (remove if exists, then start fresh)
docker rm -f qdrant-server 2>/dev/null || true
docker run -d --name qdrant-server -p 6333:6333 qdrant/qdrant

# Wait for Qdrant to start
sleep 5

# Run the application
python main.py

