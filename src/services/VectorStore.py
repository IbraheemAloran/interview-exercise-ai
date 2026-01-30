import faiss
import pickle
import os
from logger_config import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []

        logger.info(f"Initialized FAISS index with dim={embedding_dim}")

    def add(self, embeddings, metadatas):
        #add embeddings to the index
        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadata length mismatch")

        try:
            self.index.add(embeddings)
            self.metadata.extend(metadatas)
            logger.info(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Failed adding vectors: {e}")
            raise

    def search(self, query_embedding, top_k=5):
        #search for similar vectors in the index
        try:
            distances, indices = self.index.search(query_embedding, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata):
                    results.append({
                        "score": float(dist),
                        "metadata": self.metadata[idx]
                    })

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []


    def save(self, path="faiss_store"):
        #save faiss index to disk
        try:
            os.makedirs(path, exist_ok=True)

            faiss.write_index(self.index, f"{path}/index.faiss")

            with open(f"{path}/metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)

            logger.info(f"FAISS store saved to {path}")

        except Exception as e:
            logger.error(f"Failed saving FAISS store: {e}")
            raise


    def load(self, path="faiss_store"):
        #load faiss index from disk
        try:
            self.index = faiss.read_index(f"{path}/index.faiss")

            with open(f"{path}/metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)

            logger.info(
                f"Loaded FAISS store from {path}. Total vectors: {self.index.ntotal}"
            )

        except Exception as e:
            logger.error(f"Failed loading FAISS store: {e}")
            raise