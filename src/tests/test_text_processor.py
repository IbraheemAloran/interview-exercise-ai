import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from TextProcessor import FileLoader, TextChunker


class DummySplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        mid = max(1, len(text) // 2)
        return [text[:mid], text[mid:]]


@pytest.fixture
def dummy_documents():
    return [('Document One', 'This is the first document with some content.'),
            ('Document Two', 'This is the second document with more content.')]


@pytest.fixture
def text_chunker(monkeypatch, dummy_documents):
    monkeypatch.setattr('TextProcessor.RecursiveCharacterTextSplitter', DummySplitter)
    return TextChunker(documents=dummy_documents, chunk_size=100, chunk_overlap=20)


def test_text_chunker_initialization(text_chunker):
    assert text_chunker.chunk_size == 100
    assert text_chunker.chunk_overlap == 20
    assert len(text_chunker.documents) == 2


def test_text_chunker_split_single_document(text_chunker):
    chunks = text_chunker.split_into_chunks('Test Doc', 'hello world')
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all('text' in c and 'metadata' in c for c in chunks)


def test_text_chunker_split_single_document_metadata(text_chunker):
    chunks = text_chunker.split_into_chunks('TestFile', 'some text content')
    for chunk in chunks:
        assert chunk['metadata']['filename'] == 'TestFile'


def test_text_chunker_split_docs(text_chunker):
    all_chunks = text_chunker.split_docs()
    assert isinstance(all_chunks, list)
    assert len(all_chunks) > 0
    assert all('text' in c and 'metadata' in c for c in all_chunks)


def test_text_chunker_split_docs_metadata(text_chunker, dummy_documents):
    all_chunks = text_chunker.split_docs()
    filenames = set(c['metadata']['filename'] for c in all_chunks)
    assert 'Document One' in filenames
    assert 'Document Two' in filenames


def test_text_chunker_initialization_with_invalid_documents():
    with pytest.raises(ValueError):
        TextChunker(documents=None)


def test_text_chunker_initialization_with_empty_documents():
    with pytest.raises(ValueError):
        TextChunker(documents=[])


def test_file_loader_initialization():
    loader = FileLoader(directory='test_dir')
    assert loader.directory == 'test_dir'


def test_file_loader_load_files_nonexistent_directory():
    loader = FileLoader(directory='nonexistent_dir_xyz')
    files = loader.load_files()
    assert files == []
