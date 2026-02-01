# RAG Ticket Support System

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent ticket support automation. This system leverages embeddings, semantic search, and large language models to provide accurate, context-aware responses to customer support tickets.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

The RAG Ticket Support System automates customer support by:

- **Retrieving** relevant support documentation based on customer queries
- **Augmenting** LLM prompts with context from knowledge base documents
- **Generating** accurate, well-informed responses to support tickets

### Key Features

✅ **Semantic Search** - Retrieves relevant documentation using embeddings and vector similarity  
✅ **Context-Aware Responses** - LLM augmented with support documentation context  
✅ **Multi-Document Support** - Processes multiple support guides and policies  
✅ **REST API** - FastAPI-based HTTP endpoint for easy integration  
✅ **Containerized** - Docker and Docker Compose for seamless deployment  
✅ **Comprehensive Logging** - Full audit trail of requests and operations  
✅ **CORS Enabled** - Configurable cross-origin resource sharing  
✅ **Health Checks** - System status monitoring endpoints

---

---

## System Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Client / Frontend                         │
│               (HTTP REST API Consumer)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Server                             │
│              (API Gateway & Orchestrator)                    │
└────────────────┬────────────────────────┬────────────────────┘
                 │                        │
         ┌───────▼─────────┐      ┌──────▼──────────┐
         │  Text Processor │      │  ServiceContainer│
         │  - FileLoader   │      │  - LLM Service   │
         │  - TextChunker  │      │  - Embeddings    │
         └────────┬────────┘      │  - Vector Store  │
                  │               │  - RAG Agent     │
                  ▼               └──────┬───────────┘
    ┌────────────────────────┐           │
    │ Knowledge Base (Data)  │           │
    │ - account_recovery     │           │
    │ - domain_suspension    │           │
    │ - technical_support    │           │
    └────────────────────────┘           │
                                         │
                  ┌──────────────────────┘
                  │
     ┌────────────▼────────────┐
     │  Embedding Service      │
     │  (Sentence Transformers)│
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │   Vector Store (FAISS)  │
     │   - Dense Vector Index  │
     │   - Similarity Search   │
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │    RAG Agent            │
     │  - Query Embedding      │
     │  - Document Retrieval   │
     │  - Relevancy Check      │
     │  - Prompt Building      │
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │   LLM Service           │
     │   (Google Generative AI)│
     │   - API Communication   │
     │   - Response Generation │
     └────────────────────────┘
```

### Data Flow

1. **User Query** → FastAPI endpoint receives support ticket
2. **Text Processing** → Query is embedded into vector representation
3. **Retrieval** → FAISS vector store retrieves top-k most similar documents
4. **Relevancy Check** → System verifies retrieved documents meet relevancy threshold
5. **Prompt Building** → Context-aware prompt constructed with retrieved documents
6. **LLM Generation** → Google Generative AI generates structured response
7. **Response** → Formatted ticket response returned to client

---

## Tech Stack

| Component           | Technology               |
| ------------------- | ------------------------ |
| **Web Framework**   | FastAPI                  |
| **LLM Provider**    | Google Generative AI     |
| **Embeddings**      | Sentence Transformers    |
| **Vector Database** | FAISS                    |
| **Deep Learning**   | PyTorch                  |
| **NLP Pipeline**    | Transformers             |
| **Chunking**        | LangChain Text Splitters |
| **Server**          | Uvicorn                  |
| **Math/Arrays**     | NumPy                    |

## Setup Instructions

### Option 1: Local Development Setup

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-interview
```

#### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Set Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
ENV=development
```

#### 5. Verify Knowledge Base Files

Ensure support documents exist in `src/data/`:

- `account_recovery.txt`
- `domain_suspension_policy.txt`
- `technical_support_guide.txt`

#### 6. Run the Application

```bash
cd src/api
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

**API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger UI

---

### Option 2: Docker Setup

#### 1. Build Docker Image

```bash
docker build -t rag-support-system .
```

#### 2. Set Environment Variables

Update `docker-compose.yml` or create `.env.docker`:

```env
GOOGLE_API_KEY=your_google_api_key_here
ENV=production
```

#### 3. Run with Docker Compose

```bash
docker-compose up -d
```

Access the API at `http://localhost:8000`

#### 4. View Logs

```bash
docker-compose logs -f api
```

#### 5. Stop Services

```bash
docker-compose down
```

---

## Usage

### Quick Start: Interactive Swagger UI

The easiest way to test the API is through the interactive Swagger UI:

1. **Start the server** (if not already running):
   ```bash
   cd src/api
   uvicorn app:app --reload --port 8000
   ```

2. **Open Swagger UI**:
   - Navigate to: `http://localhost:8000/docs`
   - All endpoints with request/response examples are documented
   - Click "Try it out" on any endpoint to test directly in the browser

---

### API Endpoints

#### Health Check

**Using Swagger UI:**
- Open `http://localhost:8000/docs`
- Find the `GET /health` endpoint
- Click "Try it out" and then "Execute"

**Using cURL:**

```bash
curl -X GET "http://localhost:8000/health" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "status": "ok",
  "services": {
    "llm_initialized": true,
    "embedding_initialized": true,
    "vector_store_initialized": true,
    "vector_store_size": 150,
    "rag_initialized": true
  }
}
```

#### Process Support Ticket

**Using Swagger UI:**
1. Navigate to `http://localhost:8000/docs`
2. Expand the `POST /resolve-ticket` endpoint
3. Click "Try it out"
4. Enter the following in the request body:
   ```json
   {
     "query": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
   }
   ```
5. Click "Execute" and view the response

**Using cURL:**

```
curl -X POST "http://localhost:8000/resolve-ticket" \
  -H "Content-Type: application/json" \
  -d "{
    "query": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
  }"

```

**Response:**
```json
{
  "answer": "Your domain may have been suspended due to a violation of policy or missing WHOIS information. Please update your WHOIS details and contact support.",
  "references": ["Domain Suspension Guidelines"],
  "action_required": "escalate_to_abuse_team"
}
```

## Project Structure

```
ai-interview/
├── README.md                          # This file
├── Dockerfile                         # Container image definition
├── docker-compose.yml                 # Multi-container orchestration
├── requirements.txt                   # Python dependencies
├── .env                               # Environment configuration
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                     # FastAPI application & endpoints
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── config.py                  # Configuration & constants
│   │   ├── logger_config.py           # Logging setup
│   │   ├── EmbeddingService.py        # Text embedding logic
│   │   ├── LLMService.py              # LLM API integration
│   │   ├── VectorStore.py             # FAISS vector database
│   │   ├── RAGService.py              # RAG orchestration
│   │   ├── PromptBuilder.py           # Prompt engineering
│   │   └── TextProcessor.py           # Document loading & chunking
│   │
│   ├── data/
│   │   ├── account_recovery.txt       # Knowledge base: account recovery
│   │   ├── domain_suspension_policy.txt # Knowledge base: policies
│   │   └── technical_support_guide.txt # Knowledge base: tech support
│   │
│   └── tests/
│       ├── test_embedding_service.py
│       ├── test_llm_service.py
│       ├── test_prompt_builder.py
│       ├── test_rag_service.py
│       ├── test_text_processor.py
│       └── test_vector_store.py
```

---

## Future Improvements

- [ ] **Response Confidence Scoring** - Provide confidence metrics for generated responses
- [ ] **Caching Layer** - Integrate cache for frequent responses
- [ ] **Feedback Loop** - User ratings to improve response quality
- [ ] **Dashboard** - Analytics and monitoring interface
- [ ] **Advanced Chunking** - Semantic chunking, Document chunking, Parent-Child Chunking, etc
- [ ] **Hybrid Search** - Combine semantic + keyword-based search + reranking
- [ ] **Performance Optimization** - Model quantization and optimization

## Running Tests

Execute the test suite:

```bash
# Run all tests
pytest src/tests/

# Run specific test file
pytest src/tests/test_rag_service.py
```

---

## Troubleshooting

### Issue: `GOOGLE_API_KEY not set` Warning

**Solution:** Set your Google API key in `.env` file:

```env
GOOGLE_API_KEY=your_key_here
```

### Issue: `No documents found in data directory`

**Solution:** Ensure text files exist in `src/data/`:

```bash
ls -la src/data/
# Should show: account_recovery.txt, domain_suspension_policy.txt, technical_support_guide.txt
```

### Issue: Port 8000 Already in Use

**Solution:**

```bash
# Use different port
uvicorn app:app --port 8001
```

### Issue: CORS Errors

**Solution:** Update `ALLOWED_ORIGINS` in `.env`:

```env
ALLOWED_ORIGINS=http://localhost:3000,http://your-domain.com
```

## Author

Ibraheem Aloran

AI Engineer

---
