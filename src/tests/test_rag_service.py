import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from RAGService import RAGAgent


class DummyEmbeddingService:
    def embed_query(self, q):
        return [0.1, 0.2, 0.3]


class DummyVectorStore:
    def __init__(self, docs=None):
        self.docs = docs or []

    def search(self, embedding, top_k=5):
        return self.docs[:top_k]


class DummyPromptBuilder:
    @staticmethod
    def build_prompt(query, docs):
        return f"Prompt for {query}"


@pytest.fixture
def dummy_docs():
    return [
        {'score': 0.95, 'metadata': {'filename': 'a.txt', 'content': 'highly relevant'}},
        {'score': 0.85, 'metadata': {'filename': 'b.txt', 'content': 'relevant'}},
        {'score': 0.5, 'metadata': {'filename': 'c.txt', 'content': 'somewhat relevant'}},
    ]


@pytest.fixture
def rag_agent(dummy_docs):
    emb = DummyEmbeddingService()
    vs = DummyVectorStore(docs=dummy_docs)
    return RAGAgent(llm_service=None, vector_store=vs, embedding_service=emb, output_schema=None)


def test_embed_query(rag_agent):
    result = rag_agent.embed_query('test query')
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == 0.1


def test_embed_query_with_different_input(rag_agent):
    result = rag_agent.embed_query('another query')
    assert result == [0.1, 0.2, 0.3]


def test_retrieve_documents(rag_agent, dummy_docs):
    embedding = [0.1, 0.2, 0.3]
    docs = rag_agent.retrieve_documents(embedding, top_k=2)
    assert len(docs) <= 2
    assert docs == dummy_docs[:2]


def test_retrieve_documents_with_different_top_k(rag_agent, dummy_docs):
    embedding = [0.1, 0.2, 0.3]
    docs = rag_agent.retrieve_documents(embedding, top_k=1)
    assert len(docs) == 1
    assert docs[0]['score'] == 0.95


def test_check_relevancy_high_threshold_passes(rag_agent, dummy_docs):
    assert rag_agent.check_relevancy(dummy_docs, threshold=0.7) is True


def test_check_relevancy_very_high_threshold_fails(rag_agent, dummy_docs):
    assert rag_agent.check_relevancy(dummy_docs, threshold=0.95) is False


def test_check_relevancy_empty_docs(rag_agent):
    assert rag_agent.check_relevancy([], threshold=0.1) is False


def test_check_relevancy_single_doc(rag_agent):
    docs = [{'score': 0.8, 'metadata': {'filename': 'test.txt'}}]
    assert rag_agent.check_relevancy(docs, threshold=0.5) is True
    assert rag_agent.check_relevancy(docs, threshold=0.9) is False
