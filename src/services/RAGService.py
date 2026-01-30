from typing import List, Optional
from PromptBuilder import PromptBuilder
import config
from logger_config import get_logger


class RAGAgent:
    #RAG Agent class to carry out user queries and return responses

    def __init__(
        self,
        llm_service,
        vector_store,
        embedding_service,
        output_schema: config.TicketResponse,
    ):
        self.llm = llm_service
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.prompter = PromptBuilder()
        self.logger = get_logger(__name__)

        self.logger.info("RAG Agent initialized")


    def embed_query(self, query: str) -> List[float]:
        try:
            self.logger.info("Embedding user query")
            embedding = self.embedding_service.embed_query(query)
            return embedding
        except Exception as e:
            self.logger.exception("Failed to embed query", exc_info=True)


    def retrieve_documents(
        self, embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        #retrieve docs from vector store
        try:
            self.logger.info("Retrieving documents from vector store")
            docs = self.vector_store.search(embedding, top_k=top_k)
            return docs
        except Exception as e:
            self.logger.exception("Vector store retrieval failed", exc_info=True)



    async def answer_query(self, query: str, top_k: int = 5) -> config.TicketResponse:
        #RAG Pipeline
        try:
            embedding = self.embed_query([query])
            docs = self.retrieve_documents(embedding, top_k=top_k)
            prompt = self.prompter.build_prompt(query, docs)
            response = await self.llm.generate(prompt, config.TicketResponse)
            return response
        except Exception as e:
            self.logger.exception("Unexpected RAG error", exc_info=True)