from typing import Optional, Generator, Union
import requests
from ollama_python.endpoints.base import BaseAPI
from ollama_python.models.model_management import (
    ResponsePayload,
    ModelTagList,
    ModelInformation,
)


class ModelManagementAPI(BaseAPI):
    """
    A client for the model management endpoints
    """

    def create(
        self,
        name: str,
        model_file: Optional[str] = None,
        stream: bool = False,
        path: Optional[str] = None,
    ) -> Union[ResponsePayload, Generator]:
        """
        Create a model
        :param name: The name of the model
        :param model_file: The path to the model file
        :param stream: If false the response will be returned as a single response object, rather than a stream of objects
        :param path: The path to the model file
        :return:
        """
        parameters = {
            "name": name,
            "model_file": model_file,
            "stream": stream,
            "path": path,
        }

        if stream:
            return self._stream(
                parameters=parameters, endpoint="create", return_type=ResponsePayload
            )

        return self._post(
            parameters=parameters, endpoint="create", return_type=ResponsePayload
        )

    def check_blob_exists(self, digest: str) -> int:
        """
        Check if a blob exists
        :param digest: The digest of the blob to check
        :return: The status code of the request
        """
        response = requests.head(f"{self.base_url}/blob/{digest}")
        response.raise_for_status()

        return response.status_code

    def create_blob(self, digest: str) -> int:
        """
        Create a blob
        :param digest: The digest of the blob to create
        :return: The status code of the request
        """
        endpoint = f"blob/{digest}"

        return self._post(endpoint=endpoint, parameters=None)

    def list_local_models(self) -> ModelTagList:
        """
        List all tags
        :return: A list of local models
        """
        return self._get(endpoint="tags", return_type=ModelTagList)

    def show(self, name: str) -> ModelInformation:
        """
        Show a model
        :param name: The name of the model to show
        :return: The status code of the request
        """

        return self._post(
            endpoint="show", parameters={"name": name}, return_type=ModelInformation
        )

    def copy(self, source: str, destination: str) -> int:
        """
        Copy a model
        :param source: The source model to copy
        :param destination: The destination model to copy to
        :return: The status code of the request
        """

        return self._post(
            endpoint="copy", parameters={"source": source, "destination": destination}
        )

    def delete(self, name: str) -> int:
        """
        Delete a model
        :param name: The name of the model to delete
        :return: The status code of the request
        """

        return self._post(endpoint="delete", parameters={"name": name})

    def pull(
        self, name: str, insecure: Optional[bool] = None, stream: bool = False
    ) -> Union[ResponsePayload, Generator]:
        """
        Download a model from the ollama library. Cancelled pulls are resumed from where they left off,
        and multiple calls will share the same download progress.
        :param name:  The name of the model to pull
        :param insecure: Allow insecure connections to the library. Only use this if you are pulling from
                        your own library during development.
        :param stream: if false the response will be returned as a single response object, rather than a stream of objects
        :return: ResponsePayload if stream is false, otherwise a generator that yields the response
        """
        parameters = {
            "name": name,
            "stream": stream,
        }
        if parameters:
            parameters["insecure"] = insecure
        if stream:
            return self._stream(
                endpoint="pull", parameters=parameters, return_type=ResponsePayload
            )
        return self._post(
            endpoint="pull", parameters=parameters, return_type=ResponsePayload
        )

    def push(
        self, name: str, insecure: Optional[bool] = None, stream: bool = False
    ) -> Union[ResponsePayload, Generator]:
        """
        Upload a model to the ollama library. Requires registering for ollama.ai and adding a public key first.
        :param name:  The name of the model to push
        :param insecure: Allow insecure connections to the library. Only use this if you are pushing to
                        your own library during development.
        :param stream: if false the response will be returned as a single response object, rather than a stream of objects
        :return: ResponsePayload if stream is false, otherwise a generator that yields the response
        """
        parameters = {
            "name": name,
            "stream": stream,
        }
        if parameters:
            parameters["insecure"] = insecure
        if stream:
            return self._stream(
                endpoint="push", parameters=parameters, return_type=ResponsePayload
            )
        return self._post(
            endpoint="push", parameters=parameters, return_type=ResponsePayload
        )
