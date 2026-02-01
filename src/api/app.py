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


from ServiceContainer import ServiceContainer


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
