import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from TextProcessor import FileLoader, TextChunker
import config
from LLMService import LLMService
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
from EmbeddingService import EmbeddingService
from VectorStore import VectorStore
from RAGService import RAGAgent
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
ENV = os.getenv("ENV", "development")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set; LLM service may fail at runtime")


class ServiceContainer:
    #Manages service lifecycle and dependencies
    def __init__(self):
        self.llm = None
        self.embed_engine = None
        self.vector_store = None
        self.rag = None
        self.initialized = False

    def initialize(self):
        #Initialize all services with error handling
        try:
            logger.info("Starting service initialization...")
            
            # Initialize LLM service
            self.llm = LLMService(api_key=GOOGLE_API_KEY or "")
            logger.info("LLM Service initialized")
            
            # Initialize embedding engine
            self.embed_engine = EmbeddingService()
            logger.info("Embedding Service initialized")
            
            # Initialize vector store
            self.vector_store = VectorStore()
            logger.info("Vector Store initialized")
            
            # Load and process documents
            loader = FileLoader(config.FileLoaderConfig.path)
            docs = loader.load_files()
            if not docs:
                raise RuntimeError("No documents found in data directory")
            logger.info(f"Loaded {len(docs)} documents")
            
            # Chunk documents
            chunker = TextChunker(docs, 
                                 chunk_size=config.ChunkerConfig.chunk_size,
                                 chunk_overlap=config.ChunkerConfig.chunk_overlap)
            chunks = chunker.split_docs()
            if not chunks:
                raise RuntimeError("Failed to chunk documents")
            logger.info(f"Created {len(chunks)} chunks")
            
            # Embed and populate vector store
            embeds, metas = self.embed_engine.embed_documents(chunks)
            self.vector_store.add(embeds, metas)
            logger.info(f"Populated vector store with {self.vector_store.index.ntotal} vectors")
            
            # Initialize RAG agent
            self.rag = RAGAgent(self.llm, self.vector_store, self.embed_engine, config.TicketResponse)
            logger.info("RAG Agent initialized")
            
            self.initialized = True
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.exception(f"Service initialization failed: {str(e)}")
            self.initialized = False
            raise

    def get_status(self) -> dict:
        #Get health status of all services
        return {
            "llm_initialized": self.llm is not None,
            "embedding_initialized": self.embed_engine is not None,
            "vector_store_initialized": self.vector_store is not None,
            "vector_store_size": self.vector_store.index.ntotal if self.vector_store else 0,
            "rag_initialized": self.rag is not None,
            "overall_initialized": self.initialized,
        }


app = FastAPI(
    title="RAG Support Ticket API",
    description="RAG support ticket resolution",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


services = ServiceContainer()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        services.initialize()
        if not services.initialized:
            raise RuntimeError("Services failed to initialize")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


def get_services() -> ServiceContainer:
    #Dependency injection for services
    if not services.initialized:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized. Check startup logs."
        )
    return services


@app.post("/resolve-ticket", response_model=config.TicketResponse)
async def resolve_ticket(request: config.TicketRequest, svc: ServiceContainer = Depends(get_services)):
    try:
        # Validate input
        if not request.query or not request.query.strip():
            logger.warning("Resolve ticket called with empty query")
            return config.TicketResponse(
                answer="Unable to process empty query. Please provide a valid support question.",
                references=[],
                action_required="follow_up_required"
            )
        
        # Log the incoming request
        logger.info(f"Processing support ticket: {request.query[:100]}...")
        
        # Call RAG pipeline with the user's query
        response = await svc.rag.answer_query(request.query)
        
        # Log successful resolution
        logger.info(f"Ticket resolved successfully")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in resolve_ticket: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid request data.")
        
    except Exception as e:
        logger.exception(f"Unexpected error in resolve_ticket: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your request.")


@app.get("/health")
async def health_check(svc: ServiceContainer = Depends(get_services)):
    """Enhanced health check with service status"""
    status = svc.get_status()
    return {
        "status": "healthy" if svc.initialized else "degraded",
        "services": status,
        "environment": ENV,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
