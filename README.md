# 🧠 Support Agent RAG System

A **Knowledge Assistant** that helps support teams respond to customer tickets efficiently using a **Retrieval-Augmented Generation (RAG)** pipeline powered by **OpenAI's LLM** and following **Model Context Protocol (MCP)** standards.

## Overview

This system analyzes customer support queries and returns structured, relevant, and helpful responses. It combines document retrieval with language model generation to provide support agents with:

- **AI-generated responses** based on company documentation
- **Relevant policy references** for accurate information
- **Recommended actions** for proper ticket routing

# Smart Auto-Update System

The RAG system automatically detects policy document changes and updates the knowledge base without manual intervention.

## How It Works

### 🔍 **Automatic Detection**
- Compares file modification timestamps between source and processed documents
- Validates vector store consistency on startup
- Triggers updates only when changes are detected

## Sample Usage

**Input:**
{
"ticket_text": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
}

**Output (MCP-compliant):**
{
"answer": "I understand your concern about the domain suspension. Based on our documentation, domain suspensions typically occur due to incomplete WHOIS information or policy violations. To reactivate your domain, please update your WHOIS contact details through the control panel and contact our abuse team for manual review. The reactivation process usually takes 24-48 hours once all information is updated.",
"references": [
"Domain Management Policy v2.3 - Section 4.2: Suspension Process",
"Domain Management Policy v2.3 - Section 4.3: Reactivation Requirements"
],
"action_required": "escalate_to_abuse_team"
}

## Architecture

### RAG Pipeline Components

1. **Document Processing** - Loads and chunks company policies/FAQs
2. **Vector Store** - Qdrant database for semantic document search  
3. **Embedding Service** - OpenAI text-embedding-ada-002 for vector generation
4. **LLM Service** - GPT-3.5-turbo for intelligent response generation
5. **RAG Orchestrator** - Coordinates retrieval and generation workflow

### System Flow


## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **OpenAI API Key** 
- **Qdrant Vector Database**

### 1. Installation

Clone the repository
git clone [your-repo-url]
cd support-agent-rag

Install dependencies
pip install -r requirements.txt

### 2. Create .env file

### 3. Start Vector Database
docker run -d --name qdrant-server -p 6333:6333 qdrant/qdrant

### 4. Initialize Document Index(optional if you dont see processed folder )

python services/document_processor.py or python3 services/document_processor.py


### 5. Start the API Server

python main.py or python3 main.py

curl -X 'POST' \
  'http://localhost:8000/resolve-ticket' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "ticket_text": "How can i talk to customer service?"
}'


### Interactive API Documentation
Visit [**http://localhost:8000/docs**](http://localhost:8000/docs) for Swagger UI with interactive testing.


## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings/LLM | Yes |
| `QDRANT_HOST` | Qdrant server host | No (default: localhost) |
| `QDRANT_PORT` | Qdrant server port | No (default: 6333) |

### Model Configuration

- **Embedding Model:** `text-embedding-ada-002` (1536 dimensions)
- **LLM Model:** `gpt-3.5-turbo` (optimized for support responses)
- **Vector Distance:** Cosine similarity
- **Chunk Size:** 500 characters with 50 character overlap

## API Endpoints

### Main Endpoint

**POST /resolve-ticket**
- **Purpose:** Process support ticket and return AI-generated response
- **Input:** `{ "ticket_text": "customer query..." }`
- **Output:** MCP-compliant JSON with answer, references, and action

### Support Endpoints

**GET /health**
- **Purpose:** Check system health and service status
- **Output:** Service availability and system metrics

**GET /**
- **Purpose:** API information and usage instructions
- **Output:** Endpoint documentation and examples

## Key Features

### RAG Pipeline
- **Semantic Search:** Finds relevant documentation using vector similarity
- **Context Assembly:** Combines multiple document chunks for comprehensive context
- **Intelligent Generation:** Uses LLM to create helpful, accurate responses

### Response Quality
- **Professional Tone:** Empathetic and helpful customer service language
- **Structured Answers:** Clear steps and actionable guidance
- **Source References:** Citations from company documentation
- **Smart Routing:** Recommends appropriate team escalation

### Production Ready
- **Error Handling:** Graceful degradation and informative error messages
- **Performance Monitoring:** Request timing and usage statistics
- **Scalable Architecture:** Async processing and connection pooling
- **API Documentation:** Auto-generated OpenAPI/Swagger docs

## Performance

### Response Times
- **Average:** 2-4 seconds per ticket
- **Document Retrieval:** 200-500ms  
- **LLM Generation:** 1-3 seconds

### Cost Estimates (GPT-3.5-turbo)
- **Per Ticket:** ~$0.001-0.003
- **1000 Tickets/month:** ~$1-3

### Scalability
- **Concurrent Requests:** 50+ (with proper async handling)
- **Document Capacity:** 10,000+ chunks
- **Storage:** ~100MB for typical document set

## Technical Decisions

### Why GPT-3.5-turbo?
- **Cost-effective** for high-volume support tickets
- **Fast responses** suitable for customer service
- **Excellent instruction following** for structured tasks

### Why Qdrant?
- **High-performance** vector similarity search
- **Scalable** for production workloads  
- **Easy deployment** with Docker

### Why RAG over Fine-tuning?
- **Dynamic knowledge** updates without retraining
- **Transparent citations** for support agent confidence
- **Cost-effective** for evolving documentation

## Requirements Fulfilled

**RAG Pipeline** - Document embedding and retrieval with vector database  
**LLM Integration** - GPT-3.5-turbo with contextual prompt engineering  
**MCP Compliance** - Structured JSON output with defined schema  
**API Endpoint** - Single `POST /resolve-ticket` endpoint as specified  
**Code Quality** - Modular, documented, and maintainable architecture  

## 👤 Author

**Arshita**  
*AI Engineer Candidate*  
*aarshita@uwaterloo.ca*
*[https://www.linkedin.com/in/arshita01625/]*
*[https://portfolio-i242.onrender.com/]*
---

*Built for Tucows AI Engineer Interview - Demonstrates production-ready RAG system design and implementation.*

