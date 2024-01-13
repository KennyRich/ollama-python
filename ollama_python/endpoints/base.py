"""Base API for all endpoints"""
import json
import requests
from typing import Callable, Generator, Optional


class BaseAPI:
    def __init__(self, base_url: str = "http://localhost:11434/api"):
        """
        Initialize the base API endpoint
        :param base_url: The base URL of the API
        """
        self.base_url = self._format_base_url(base_url=base_url)

    def _format_base_url(self, base_url: str) -> str:
        """
        Format the base URL
        :param base_url: The base URL to format
        :return: The formatted base URL
        """
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        return base_url

    def _stream(
        self, endpoint: str, parameters: dict, return_type: Optional[Callable] = None
    ) -> Generator:
        """
        Stream the response from the given endpoint
        :param endpoint: The endpoint to stream from
        :param parameters: The parameters to send
        :return: A generator that yields the response
        """
        with requests.post(
            f"{self.base_url}/{endpoint}", json=parameters, stream=True
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    resp = json.loads(line)
                    yield return_type(**resp) if return_type else resp

    def _post(
        self,
        endpoint: str,
        parameters: Optional[dict] = None,
        return_type: Optional[Callable] = None,
    ):
        """
        Send a POST request to the given endpoint
        :param endpoint:
        :param parameters:
        :param return_type:
        :return:
        """
        response = requests.post(f"{self.base_url}/{endpoint}", json=parameters)
        response.raise_for_status()
        return return_type(**response.json()) if return_type else response.status_code

    def _get(self, endpoint: str, return_type: Optional[Callable] = None):
        """
        Send a GET request to the given endpoint
        :param endpoint:
        :param return_type:
        :return:
        """
        response = requests.get(f"{self.base_url}/{endpoint}")
        response.raise_for_status()
        return return_type(**response.json()) if return_type else response.status_code
