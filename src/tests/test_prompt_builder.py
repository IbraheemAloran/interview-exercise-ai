import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))
from PromptBuilder import PromptBuilder


def test_build_prompt_basic():
    docs = [
        {
            'metadata': {
                'text': 'This is a test document',
                'metadata': {'filename': 'doc1.txt'}
            }
        }
    ]
    prompt = PromptBuilder.build_prompt('How to test?', docs)
    assert isinstance(prompt, str)
    assert 'How to test?' in prompt
    assert 'doc1.txt' in prompt
    assert 'OUTPUT SCHEMA' in prompt


def test_build_prompt_with_multiple_docs():
    docs = [
        {'metadata': {'text': 'First doc', 'metadata': {'filename': 'doc1.txt'}}},
        {'metadata': {'text': 'Second doc', 'metadata': {'filename': 'doc2.txt'}}},
    ]
    prompt = PromptBuilder.build_prompt('Test query', docs)
    assert 'doc1.txt' in prompt
    assert 'doc2.txt' in prompt


def test_format_context_documents_empty():
    assert PromptBuilder._format_context_documents([]) == '[No documents provided]'


def test_format_context_documents_with_docs():
    docs = [
        {'metadata': {'text': 'content', 'metadata': {'filename': 'test.txt'}}}
    ]
    result = PromptBuilder._format_context_documents(docs)
    assert '[Document' in result
    assert 'test.txt' in result
    assert 'content' in result


def test_format_actions():
    result = PromptBuilder._format_actions()
    assert 'none' in result
    assert 'escalate_to_abuse_team' in result


def test_format_json_schema():
    schema = PromptBuilder._format_json_schema()
    assert isinstance(schema, str)
    assert 'answer' in schema
    assert 'references' in schema
    assert 'action_required' in schema


def test_dict_to_json():
    data = {'key': 'value', 'number': 42}
    json_str = PromptBuilder._dict_to_json(data)
    assert isinstance(json_str, str)
    assert 'key' in json_str
    assert 'value' in json_str


def test_build_prompt_invalid_query():
    with pytest.raises(ValueError):
        PromptBuilder.build_prompt('', [{'metadata': {'text': 'doc', 'metadata': {'filename': 'f.txt'}}}])


def test_build_prompt_invalid_context():
    with pytest.raises(ValueError):
        PromptBuilder.build_prompt('query', [])
