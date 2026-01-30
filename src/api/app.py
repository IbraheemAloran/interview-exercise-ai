import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from TextProcessor import FileLoader, TextChunker
import config
from LLMService import LLMService
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from EmbeddingService import EmbeddingService
from PromptBuilder import PromptBuilder
from VectorStore import VectorStore
from RAGService import RAGAgent
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Initializing services")

llm = LLMService(api_key=config.LLMServiceConfig.api_key)
logger.info("LLM Initialized")

embed_engine = EmbeddingService()
logger.info("Embedding Model Initialized")

vector_store = VectorStore()
logger.info("Vector Store Initialized")

loader = FileLoader(config.FileLoaderConfig.path)
docs = loader.load_files()
logger.info("Loading Files")

chunker = TextChunker(docs, chunk_size=config.ChunkerConfig.chunk_size, chunk_overlap=config.ChunkerConfig.chunk_overlap)
chunks = chunker.split_docs()
logger.info("Chunking Files")

embeds, metas = embed_engine.embed_documents(chunks)
vector_store.add(embeds, metas)
logger.info("Populating Vector Store")

rag = RAGAgent(llm, vector_store, embed_engine, config.TicketResponse)


@app.post("/resolve-ticket", response_model=config.TicketResponse)
async def resolve_ticket(request: config.TicketRequest):
    try:
        # Validate input
        if not request.query or not request.query.strip():
            logger.warning("Resolve ticket called with empty query")
            return config.TicketResponse(
                answer="Unable to process empty query. Please provide a valid support question.",
                references=[],
                action_required="user_input_required"
            )
        
        # Log the incoming request
        logger.info(f"Processing support ticket: {request.query[:100]}...")
        
        # Call RAG pipeline with the user's query
        response = await rag.answer_query(request.query)
        
        # Log successful resolution
        logger.info(f"Ticket resolved successfully: {response}")
        
        return response
        
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error in resolve_ticket: {str(e)}")
        raise HTTPException(status_code=500, detail="Validation error occurred while processing your request.")
        
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Unexpected error in resolve_ticket: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your request.")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # Listen on 0.0.0.0 to allow access from other machines/tabs
    uvicorn.run(app, host="0.0.0.0", port=8000)
