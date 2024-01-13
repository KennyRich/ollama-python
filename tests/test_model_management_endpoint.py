import pytest
import responses
from ollama_python.endpoints.model_management import ModelManagementAPI
from ollama_python.models.model_management import (
    ResponsePayload,
    ModelTagList,
    ModelInformation,
)
from tests.utils.utils import mock_api_response
from requests.exceptions import HTTPError


@pytest.fixture
def model_management_api() -> ModelManagementAPI:
    return ModelManagementAPI(base_url="http://test-servers/api/")


@responses.activate
def test_create_management_api_success_without_streaming(model_management_api):
    result = {"status": "success"}
    mock_api_response("/create", result)
    response = model_management_api.create(
        name="test-model", model_file="sample model file"
    )

    assert isinstance(response, ResponsePayload)
    assert response.status == "success"


@responses.activate
def test_create_management_api_success_with_streaming(model_management_api):
    result = [{"status": "creating system layer"}, {"status": "success"}]
    mock_api_response("/create", result, stream=True)
    response = list(
        model_management_api.create(
            name="test-model", model_file="sample model file", stream=True
        )
    )

    assert len(response) == 2
    assert response[0].status == "creating system layer"
    assert response[1].status == "success"


@responses.activate
def test_check_blob_exists(model_management_api):
    mock_api_response("/blob/test-digest", status=200, request_type=responses.HEAD)
    response = model_management_api.check_blob_exists(digest="test-digest")

    assert response == 200


@responses.activate
def test_check_blob_exists_failure(model_management_api):
    with pytest.raises(HTTPError):
        mock_api_response("/blob/test-digest", status=400, request_type=responses.HEAD)
        model_management_api.check_blob_exists(digest="test-digest")


@responses.activate
def test_create_blob(model_management_api):
    mock_api_response("/blob/test-digest", status=200)
    response = model_management_api.create_blob(digest="test-digest")

    assert response == 200


@responses.activate
def test_create_blob_failure(model_management_api):
    with pytest.raises(HTTPError):
        mock_api_response("/blob/test-digest", status=400)
        model_management_api.create_blob(digest="test-digest")


@responses.activate
def test_list_tags(model_management_api):
    result = {
        "models": [
            {
                "name": "test-model",
                "digest": "test-digest",
                "size": 100,
                "modified_at": "2023-08-04T19:22:45.499127Z",
                "details": {
                    "format": "test-format",
                    "family": "test-family",
                    "families": ["test-family"],
                    "parameter_size": "test-parameter-size",
                    "quantization_level": "test-quantization-level",
                },
            }
        ]
    }
    mock_api_response("/tags", result, request_type=responses.GET)
    response = model_management_api.list_tags()

    assert isinstance(response, ModelTagList)
    assert len(response.models) == 1
    assert response.models[0].name == "test-model"
    assert response.models[0].digest == "test-digest"
    assert response.models[0].size == 100
    assert response.models[0].modified_at == "2023-08-04T19:22:45.499127Z"
    assert response.models[0].details.format == "test-format"
    assert response.models[0].details.family == "test-family"
    assert response.models[0].details.families == ["test-family"]
    assert response.models[0].details.parameter_size == "test-parameter-size"
    assert response.models[0].details.quantization_level == "test-quantization-level"


@responses.activate
def test_list_tags_failure(model_management_api):
    with pytest.raises(HTTPError):
        mock_api_response("/tags", status=400, request_type=responses.GET)
        model_management_api.list_tags()


@responses.activate
def test_show(model_management_api):
    result = {
        "modelfile": "test-modelfile",
        "parameters": "test-parameters",
        "template": "test-template",
        "details": {
            "format": "test-format",
            "family": "test-family",
            "families": ["test-family"],
            "parameter_size": "test-parameter-size",
            "quantization_level": "test-quantization-level",
        },
    }
    mock_api_response("/show", result)
    response = model_management_api.show(name="test-model")

    assert isinstance(response, ModelInformation)
    assert response.modelfile == "test-modelfile"
    assert response.parameters == "test-parameters"
    assert response.template == "test-template"
    assert response.details.format == "test-format"
    assert response.details.family == "test-family"
    assert response.details.families == ["test-family"]
    assert response.details.parameter_size == "test-parameter-size"
    assert response.details.quantization_level == "test-quantization-level"


@responses.activate
def test_show_failure(model_management_api):
    with pytest.raises(HTTPError):
        mock_api_response("/show", status=400)
        model_management_api.show(name="test-model")


@responses.activate
def test_copy(model_management_api):
    mock_api_response("/copy", status=200)
    response = model_management_api.copy(
        source="test-source", destination="test-destination"
    )

    assert response == 200


@responses.activate
def test_copy_failure(model_management_api):
    with pytest.raises(HTTPError):
        mock_api_response("/copy", status=400)
        model_management_api.copy(source="test-source", destination="test-destination")


@responses.activate
def test_delete(model_management_api):
    mock_api_response("/delete", status=200)
    response = model_management_api.delete(name="test-model")

    assert response == 200


@responses.activate
def test_delete_failure(model_management_api):
    with pytest.raises(HTTPError):
        mock_api_response("/delete", status=400)
        model_management_api.delete(name="test-model")


@responses.activate
def test_pull_success_without_streaming(model_management_api):
    result = {"status": "success"}
    mock_api_response("/pull", result)
    response = model_management_api.pull(name="test-model", insecure=True)

    assert isinstance(response, ResponsePayload)
    assert response.status == "success"


@responses.activate
def test_pull_success_with_streaming(model_management_api):
    result = [{"status": "creating system layer"}, {"status": "success"}]
    mock_api_response("/pull", result, stream=True)
    response = list(
        model_management_api.pull(name="test-model", insecure=True, stream=True)
    )

    assert len(response) == 2
    assert response[0].status == "creating system layer"
    assert response[1].status == "success"


@responses.activate
def test_pull_failure(model_management_api):
    with pytest.raises(HTTPError):
        mock_api_response("/pull", status=400)
        model_management_api.pull(name="test-model", insecure=True)


@responses.activate
def test_push_success_without_streaming(model_management_api):
    result = {"status": "success"}
    mock_api_response("/push", result)
    response = model_management_api.push(name="test-model", insecure=True)

    assert isinstance(response, ResponsePayload)
    assert response.status == "success"


@responses.activate
def test_push_success_with_streaming(model_management_api):
    result = [{"status": "creating system layer"}, {"status": "success"}]
    mock_api_response("/push", result, stream=True)
    response = list(
        model_management_api.push(name="test-model", insecure=True, stream=True)
    )

    assert len(response) == 2
    assert response[0].status == "creating system layer"
    assert response[1].status == "success"


@responses.activate
def test_push_failure(model_management_api):
    with pytest.raises(HTTPError):
        mock_api_response("/push", status=400)
        model_management_api.push(name="test-model", insecure=True)
