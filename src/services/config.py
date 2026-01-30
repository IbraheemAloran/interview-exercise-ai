from pydantic import BaseModel, ValidationError
from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ChunkerConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 100

@dataclass
class FileLoaderConfig:
    path: str = ROOT / "data"

@dataclass
class VectorStoreConfig:
    embedding_dim: int = 384
    top_k: int=5
    path: str=ROOT / "faiss_store"

@dataclass
class EmbeddingServiceConfig:
    model_name: str="all-MiniLM-L6-v2"

@dataclass
class LLMServiceConfig:
    api_key: str = ""
    model: str = "gemini-3-flash-preview"


class TicketResponse(BaseModel):
    answer: str
    references: List[str]
    action_required: str


class TicketRequest(BaseModel):
    query: str
    context: List[dict]


