🚀  Support Agent RAG System - Complete API Documentation

Prerequisites
Python 3.8+

Docker & Docker Compose

OpenAI API Key

Installation & Setup

# Clone and setup
git clone <repository-url>
cd support-assistant

# Environment configuration
cp .env.example .env
# Edit .env with your OpenAI API key

# Start everything with one command
docker compose up --build
Your application will be available at:

Main API: http://localhost:8000

Interactive API Tester (Swagger): http://localhost:8000/docs

API Documentation: http://localhost:8000/redoc

🏗️ System Architecture
graph TB
    A[Client Upload] --> B[FastAPI Upload Endpoint]
    B --> C[File Validation & Storage]
    C --> D[Document Processing Pipeline]
    D --> E[Chunking & Metadata Extraction]
    E --> F[OpenAI Embedding Generation]
    F --> G[Qdrant Vector Database]
    G --> H[RAG Service]
    H --> I[Support Ticket Resolution]
    
    J[User Query] --> H
    H --> K[Vector Search]
    K --> L[Context Retrieval]
    L --> M[OpenAI Response Generation]
    M --> N[Structured Response]

Core Components

Upload API (src/api/upload_endpoint.py) - File management endpoints

Document Processor (scripts/load_documents.py) - Text processing and chunking

Vector Store (src/services/vector_store.py) - Qdrant integration

Embedding Service (src/services/embedding_service.py) - OpenAI embeddings

RAG Service (src/services/rag_service.py) - Query processing pipeline

Ticket Resolver (src/api/endpoints.py) - Support ticket endpoints

📋 API Endpoints
🔄 Interactive API Testing
Visit http://localhost:8000/docs for a complete interactive API tester where you can:

Upload files directly through the browser

Test all endpoints with real data

View detailed request/response schemas

Execute API calls without writing code

📤 File Upload API
Upload Document
Endpoint: POST /upload

Description: Upload a markdown document and automatically process it for RAG queries.

Request
bash
curl -X POST "http://localhost:8000/upload" \
     -F "file=@policy_document.md" \
     -F "process_immediately=true"
Python Example
python
import requests

with open('policy_document.md', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f},
        data={'process_immediately': True}
    )
print(response.json())
Sample Input File (domain_policy.md)

## 1. Domain Suspension Policy

### 1.1 Suspension Triggers
Domain suspension occurs automatically under the following conditions:

- Incomplete registrant contact information
- Invalid email address in WHOIS record
- Terms of service violations
- Payment issues beyond 30-day grace period

### 1.2 Reactivation Process
To reactivate a suspended domain:

1. Update all required WHOIS fields through control panel
2. Verify email address by clicking confirmation link
3. Submit government-issued ID for identity verification
4. Wait 24-48 hours for automated verification

## FAQ Section

Q: Why was my domain suspended without notice?
A: Suspension notices are sent to the email address listed in your domain's WHOIS record. If this email is outdated or invalid, you won't receive notifications.
Actions: [update_whois, check_spam_folder]
Escalation: no_action
Response

{
  "message": "File uploaded successfully",
  "filename": "domain_policy.md",
  "file_path": "data/documents/domain_policy.md",
  "file_size": 1024,
  "processing_triggered": true,
  "rag_updated": true
}
Response Fields
Field	Type	Description
message	string	Success/error message
filename	string	Original filename
file_path	string	Relative path where file was saved
file_size	integer	File size in bytes
processing_triggered	boolean	Whether document processing started
rag_updated	boolean	Whether vector database was updated
📁 File Management API
List All Files
Endpoint: GET /files

Description: Retrieve a list of all uploaded markdown files with metadata.

Request
bash
curl -X GET "http://localhost:8000/files" \
     -H "accept: application/json"
Response
json
{
  "message": "Files retrieved successfully",
  "total_files": 3,
  "files": [
    {
      "filename": "domain_policy.md",
      "size": 2048,
      "modified": 1726454400.123
    },
    {
      "filename": "billing_policy.md", 
      "size": 1536,
      "modified": 1726368000.456
    },
    {
      "filename": "faq_collection.md",
      "size": 3072,
      "modified": 1726281600.789
    }
  ]
}
Delete File
Endpoint: DELETE /files/{filename}

Description: Delete a specific markdown file from the system.

Request
bash
curl -X DELETE "http://localhost:8000/files/domain_policy.md" \
     -H "accept: application/json"
Response
json
{
  "message": "File 'domain_policy.md' deleted successfully",
  "filename": "domain_policy.md"
}
Manual Processing Trigger
Endpoint: POST /process

Description: Manually trigger document processing for all files in the documents directory.

Request
bash
curl -X POST "http://localhost:8000/process" \
     -H "accept: application/json"
Response
json
{
  "message": "Document processing completed successfully",
  "success": true,
  "rag_updated": true
}
🎯 Support Ticket Resolution API
Resolve Support Ticket
Endpoint: POST /resolve-ticket

Description: Process a support ticket and generate an intelligent response using RAG.

Request
bash
curl -X POST "http://localhost:8000/resolve-ticket" \
     -H "Content-Type: application/json" \
     -d '{
       "ticket_text": "My domain was suspended yesterday and I need to reactivate it urgently. What steps should I take?"
     }'
Python Example

ticket_data = {
    "ticket_text": "My domain was suspended yesterday and I need to reactivate it urgently. What steps should I take?"
}

response = requests.post(
    'http://localhost:8000/resolve-ticket',
    json=ticket_data
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Action Required: {result['action_required']}")
Sample Input Variations
Domain Suspension Query:


{
  "ticket_text": "Help! My website is showing a suspension page. I haven't received any emails about this. How do I fix it?"
}

🔍 System Health & Monitoring
Health Check
Endpoint: GET /health

Description: Check system health and service status.

Request
bash
curl -X GET "http://localhost:8000/health"
Response
json
{
  "status": "healthy",
  "timestamp": "2025-09-16T05:42:00Z",
  "services": {
    "vector_store": "connected",
    "embedding_service": "ready", 
    "document_processor": "ready"
  },
  "documents": {
    "total_files": 5,
    "total_chunks": 127,
    "last_update": "2025-09-16T05:30:15Z"
  }
}
🐳 Docker Deployment
Complete System Startup
bash
# Start all services (Qdrant + Application)
docker compose up --build

# Start in background
docker compose up --build -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
Docker Compose Configuration
version: '3.8'

# 1. Check system health
curl -X GET "http://localhost:8000/health"

# 2. Upload a policy document
curl -X POST "http://localhost:8000/upload" \
     -F "file=@domain_policy.md"

# 3. Verify file was uploaded
curl -X GET "http://localhost:8000/files"

# 4. Restart the server and test ticket resolution
curl -X POST "http://localhost:8000/resolve-ticket" \
     -H "Content-Type: application/json" \
     -d '{
       "ticket_text": "My domain is suspended, how do I reactivate it?"
     }'

# 5. Clean up - delete file
curl -X DELETE "http://localhost:8000/files/domain_policy.md"
Advanced Testing with Multiple Documents
bash
