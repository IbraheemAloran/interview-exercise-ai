
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from logger_config import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str="all-MiniLM-L6-v2", embedding_dim: int=384):
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = embedding_dim
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
            logger.info(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise e

    def embed_documents(self, chunks: list[dict]) -> tuple[np.ndarray, list[dict]]:
        #Embed a list of text chunks and return their embeddings along with metadata.
        texts = [chunk['text'] for chunk in chunks]
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            if embeddings.shape[1] != self.embedding_dim:
                logging.warning(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.embedding_dim}")



            # # Store metadata aligned with FAISS vectors
            self.metadata.extend([chunk['metadata'] for chunk in chunks])

            return embeddings, chunks
        except Exception as e:
            logging.error(f"Failed to embed documents: {e}")
            raise e


    def embed_query(self, query:str) -> np.ndarray:
      #Embed a single query string and return its embedding.
      if not query:
            raise ValueError("No texts provided for embedding")

      try:
          return self.model.encode(query, convert_to_numpy=True, show_progress_bar=True)
      except Exception as e:
          logging.error(f"Embedding failed: {e}")
          raise
