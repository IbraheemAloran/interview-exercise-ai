import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from VectorStore import VectorStore


class SimpleIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype=float)
        self.ntotal = 0

    def add(self, arr):
        arr = np.asarray(arr)
        self.vectors = np.vstack([self.vectors, arr])
        self.ntotal += len(arr)

    def search(self, query_embedding, top_k):
        if self.ntotal == 0:
            return np.empty((1, 0)), np.empty((1, 0), dtype=int)
        q = np.asarray(query_embedding).reshape(-1)
        dists = np.sum((self.vectors - q) ** 2, axis=1)
        order = np.argsort(dists)[:top_k]
        return dists[np.newaxis, order], order[np.newaxis, :]


@pytest.fixture
def vector_store(monkeypatch):
    monkeypatch.setattr('VectorStore.faiss', type('f', (), {'IndexFlatL2': SimpleIndex}))
    return VectorStore(embedding_dim=3)


def test_vector_store_initialization(vector_store):
    assert vector_store.embedding_dim == 3
    assert vector_store.metadata == []


def test_add_single_vector(vector_store):
    vectors = np.array([[1.0, 2.0, 3.0]], dtype=float)
    metas = [{'id': 1}]
    vector_store.add(vectors, metas)
    assert len(vector_store.metadata) == 1
    assert vector_store.metadata[0]['id'] == 1


def test_add_multiple_vectors(vector_store):
    vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    metas = [{'id': 1}, {'id': 2}]
    vector_store.add(vectors, metas)
    assert len(vector_store.metadata) == 2
    assert vector_store.index.ntotal == 2


def test_add_mismatched_vectors_and_metadata(vector_store):
    vectors = np.array([[1.0, 2.0, 3.0]], dtype=float)
    metas = [{'id': 1}, {'id': 2}]
    with pytest.raises(ValueError):
        vector_store.add(vectors, metas)


def test_search_basic(vector_store):
    vectors = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    metas = [{'id': 1, 'doc': 'a'}, {'id': 2, 'doc': 'b'}]
    vector_store.add(vectors, metas)
    
    q = np.array([[0.1, 0.1, 0.1]], dtype=float)
    results = vector_store.search(q, top_k=2)
    
    assert isinstance(results, list)
    assert len(results) <= 2
    if results:
        assert 'score' in results[0]
        assert 'metadata' in results[0]


def test_search_returns_closest_vector(vector_store):
    vectors = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=float)
    metas = [{'id': 1}, {'id': 2}]
    vector_store.add(vectors, metas)
    
    q = np.array([[0.1, 0.1, 0.1]], dtype=float)
    results = vector_store.search(q, top_k=1)
    
    assert len(results) == 1
    assert results[0]['metadata']['id'] == 1


def test_search_empty_store(vector_store):
    q = np.array([[1.0, 2.0, 3.0]], dtype=float)
    results = vector_store.search(q, top_k=5)
    assert results == []


def test_search_top_k_limit(vector_store):
    vectors = np.array([[float(i), float(i), float(i)] for i in range(5)], dtype=float)
    metas = [{'id': i} for i in range(5)]
    vector_store.add(vectors, metas)
    
    q = np.array([[0.0, 0.0, 0.0]], dtype=float)
    results = vector_store.search(q, top_k=2)
    assert len(results) <= 2


def test_save_and_load(vector_store, tmp_path):
    vectors = np.array([[1.0, 2.0, 3.0]], dtype=float)
    metas = [{'id': 1, 'content': 'test'}]
    vector_store.add(vectors, metas)
    
    path = str(tmp_path / 'store')
    vector_store.save(path)
    
    new_store = VectorStore(embedding_dim=3)
    new_store.load(path)
    
    assert len(new_store.metadata) == 1
    assert new_store.metadata[0]['id'] == 1


def test_save_creates_directory(vector_store, tmp_path):
    path = str(tmp_path / 'new_store')
    vectors = np.array([[1.0, 2.0, 3.0]], dtype=float)
    vector_store.add(vectors, [{'id': 1}])
    vector_store.save(path)
    
    assert Path(path).exists()


def test_load_nonexistent_path(vector_store):
    with pytest.raises(Exception):
        vector_store.load('/nonexistent/path/store')
