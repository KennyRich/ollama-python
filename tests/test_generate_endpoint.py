import pytest
import responses
import json
from ollama_python.endpoints.generate import GenerateAPI
from ollama_python.models.generate import (
    Completion,
    StreamCompletion,
    ChatCompletion,
    Message,
    StreamChatCompletion,
)


GENERATE_ENDPOINT = "/generate"
GENERATE_CHAT_ENDPOINT = "/chat"


def mock_api_response(
    endpoint: str,
    response_body: dict | list[dict],
    status: int = 200,
    stream: bool = False,
):
    """Mock the API responses for the given endpoint"""
    endpoint = f"http://test-servers/api{endpoint}"
    if stream:
        body = "\n".join(json.dumps(item) for item in response_body).encode()
        responses.add(responses.POST, endpoint, body=body, status=status, stream=True)
    else:
        responses.add(responses.POST, endpoint, json=response_body, status=status)


@pytest.fixture
def generate_api():
    return GenerateAPI(model="test-model", base_url="http://test-servers/api")


@responses.activate
def test_generate_completion_success_without_streaming(generate_api):
    result = {
        "model": "test-model",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "response": "This is a sample response",
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 10706818083,
        "load_duration": 6338219291,
        "prompt_eval_count": 26,
        "prompt_eval_duration": 130079000,
        "eval_count": 259,
        "eval_duration": 4232710000,
    }
    mock_api_response(GENERATE_ENDPOINT, result)
    completion = generate_api.generate(
        prompt="test prompt", images=["random_images"], options={"temperature": 0.5}
    )

    assert isinstance(completion, Completion)
    assert completion.response == result["response"]
    assert completion.done
    assert completion.model == result["model"]


@responses.activate
def test_generate_completion_success_with_streaming(generate_api):
    result = [
        {
            "model": "test-model",
            "created_at": "2023-08-04T08:52:19.385406455-07:00",
            "response": "The",
            "done": False,
        },
        {
            "model": "test-model",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "response": "Sun is blue",
            "done": True,
            "context": [1, 2, 3],
            "total_duration": 10706818083,
            "load_duration": 6338219291,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 130079000,
            "eval_count": 259,
            "eval_duration": 4232710000,
        },
    ]
    mock_api_response(GENERATE_ENDPOINT, result, stream=True)
    results = list(
        generate_api.generate(prompt="test prompt", format="json", stream=True)
    )
    assert all(isinstance(completion, StreamCompletion) for completion in results)
    assert all(completion.model == "test-model" for completion in results)
    assert any(completion.done for completion in results)


def test_generate_completion_failure(generate_api):
    with pytest.raises(ValueError):
        generate_api.generate(prompt="test prompt", format="csv")


def test_generate_completions_with_invalid_options(generate_api):
    with pytest.raises(ValueError):
        generate_api.generate(prompt="test prompt", options={"invalid_option": "test"})


@responses.activate
def test_generate_chat_completion_success_without_streaming(generate_api):
    result = {
        "model": "test-model",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 10706818083,
        "load_duration": 6338219291,
        "prompt_eval_count": 26,
        "prompt_eval_duration": 130079000,
        "eval_count": 259,
        "eval_duration": 4232710000,
    }
    mock_api_response(GENERATE_CHAT_ENDPOINT, result)
    completion = generate_api.generate_chat_completion(
        messages=[{"role": "user", "content": "Hello"}]
    )

    assert isinstance(completion, ChatCompletion)
    assert completion.messages == [Message(**msg) for msg in result["messages"]]
    assert completion.done
    assert completion.model == result["model"]


@responses.activate
def test_generate_chat_completion_success_with_streaming(generate_api):
    result = [
        {
            "model": "test-model",
            "created_at": "2023-08-04T08:52:19.385406455-07:00",
            "messages": [{"role": "user", "content": "Hello"}],
            "done": False,
        },
        {
            "model": "test-model",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "done": True,
            "context": [1, 2, 3],
            "total_duration": 10706818083,
            "load_duration": 6338219291,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 130079000,
            "eval_count": 259,
            "eval_duration": 4232710000,
        },
    ]
    mock_api_response(GENERATE_CHAT_ENDPOINT, result, stream=True)
    results = list(
        generate_api.generate_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            format="json",
            options={"seed": 42},
            stream=True,
        )
    )

    assert all(isinstance(completion, StreamChatCompletion) for completion in results)
    assert all(completion.model == "test-model" for completion in results)
    assert any(completion.done for completion in results)


def test_generate_chat_completion_invalid_format(generate_api):
    with pytest.raises(ValueError):
        generate_api.generate_chat_completion(
            messages=[{"role": "user", "content": "Hello"}], format="csv"
        )


def test_generate_chat_completion_invalid_options(generate_api):
    with pytest.raises(ValueError):
        generate_api.generate_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            options={"invalid_option": "test"},
        )


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Hello", "invalid_field": "test"}],
        [{"role": "bot", "content": "Hello"}],
    ],
)
def test_generate_chat_completion_invalid_messages(messages):
    with pytest.raises(ValueError):
        generate_api = GenerateAPI(
            model="test-model", base_url="http://test-servers/api"
        )
        generate_api.generate_chat_completion(messages=messages)
