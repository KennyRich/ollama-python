import pytest
import responses
from requests.exceptions import HTTPError
from ollama_python.endpoints.embedding import EmbeddingAPI
from ollama_python.models.embedding import Embedding
from tests.utils.utils import mock_api_response


@pytest.fixture
def embedding_api() -> EmbeddingAPI:
    return EmbeddingAPI(
        model="test-embedding-model", base_url="http://test-servers/api"
    )


@responses.activate
def test_get_embedding_success(embedding_api):
    result = {
        "embedding": [1, 2, 3, 5, 6, 7],
    }
    mock_api_response("/embedding", result)
    embedding = embedding_api.get_embedding(prompt="test prompt")

    assert isinstance(embedding, Embedding)
    assert embedding.embedding == result["embedding"]


@responses.activate
def test_get_embedding_success_with_options(embedding_api):
    result = {
        "embedding": [1, 2, 3, 5, 6, 7],
    }
    mock_api_response("/embedding", result)
    embedding = embedding_api.get_embedding(
        prompt="test prompt", options={"temperature": 0.5}
    )

    assert isinstance(embedding, Embedding)
    assert embedding.embedding == result["embedding"]


def test_get_embedding_failure_with_invalid_options(embedding_api):
    with pytest.raises(ValueError):
        embedding_api.get_embedding(
            prompt="test prompt", options={"invalid": 0.5, "temperature": 0.5}
        )


@responses.activate
def test_get_embedding_failure(embedding_api):
    with pytest.raises(HTTPError):
        mock_api_response("/embedding", status=400)
        embedding_api.get_embedding(prompt="test prompt")
