import asyncio
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from LLMService import LLMService, LLMServiceError


class FakeModels:
    def __init__(self, response_text, empty=False):
        self._response_text = response_text
        self._empty = empty

    def generate_content(self, **kwargs):
        class R:
            def __init__(self, t, empty=False):
                self.text = '' if empty else t
        return R(self._response_text, self._empty)


class FakeClient:
    def __init__(self, response_text, empty=False):
        self.models = FakeModels(response_text, empty)


class FakeResponseModel:
    @classmethod
    def model_json_schema(cls):
        return {}

    @classmethod
    def model_validate_json(cls, text):
        class Parsed:
            def model_dump(self):
                return {"answer": "success", "references": ["doc1"], "action_required": "none"}
        return Parsed()


@pytest.fixture
def llm_service(monkeypatch):
    def make_client(api_key):
        return FakeClient('{"test": true}')
    monkeypatch.setattr('LLMService.genai', type('g', (), {'Client': make_client}))
    return LLMService(api_key='test_key')


def test_llm_service_initialization(monkeypatch):
    def make_client(api_key):
        return FakeClient('{}')
    monkeypatch.setattr('LLMService.genai', type('g', (), {'Client': make_client}))
    svc = LLMService(api_key='test_key', model='gemini-pro')
    assert svc.model == 'gemini-pro'


@pytest.mark.asyncio
async def test_generate_success(llm_service):
    result = await llm_service.generate("hello", FakeResponseModel, temperature=0.5)
    assert isinstance(result, dict)
    assert result['answer'] == 'success'
    assert 'references' in result


@pytest.mark.asyncio
async def test_generate_with_different_temperature(llm_service):
    result = await llm_service.generate("test query", FakeResponseModel, temperature=0.8)
    assert isinstance(result, dict)
    assert result['action_required'] == 'none'


@pytest.mark.asyncio
async def test_generate_empty_response_raises(monkeypatch):
    def make_client(api_key):
        return FakeClient('', empty=True)
    monkeypatch.setattr('LLMService.genai', type('g', (), {'Client': make_client}))
    svc = LLMService(api_key='test_key')
    with pytest.raises(LLMServiceError):
        await svc.generate("query", FakeResponseModel)
