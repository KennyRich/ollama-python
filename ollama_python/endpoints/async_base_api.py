from ollama_python.endpoints.base import BaseAPI
import aiohttp
import json
from typing import Optional, Callable, AsyncGenerator


class AsyncBaseApi(BaseAPI):
    def __init__(self, base_url: str = "http://localhost:11434/api"):
        super().__init__(base_url)
        self.session = aiohttp.ClientSession()

    async def _stream(
        self, endpoint: str, parameters: dict, return_type: Optional[Callable] = None
    ) -> AsyncGenerator:
        """
        Stream the response from the given endpoint
        :param endpoint: The endpoint to stream from
        :param parameters: The parameters to send
        :return: A generator that yields the response
        """
        async with self.session.post(
            f"{self.base_url}/{endpoint}",
            json=parameters,
            stream=True,
            raise_for_status=True,
        ) as session:
            async for line in session.iter_lines():
                if line:
                    resp = json.loads(line)
                    yield return_type(**resp) if return_type else resp

    async def _post(
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
        async with self.session.post(
            f"{self.base_url}/{endpoint}", json=parameters, raise_for_status=True
        ) as session:
            data = await session.json()

            return return_type(**data) if return_type else session.status

    async def _get(self, endpoint: str, return_type: Optional[Callable] = None):
        """
        Send a GET request to the given endpoint
        :param endpoint:
        :param return_type:
        :return:
        """
        async with self.session.get(
            f"{self.base_url}/{endpoint}", raise_for_status=True
        ) as session:
            data = await session.json()
            return return_type(**data) if return_type else session.status
