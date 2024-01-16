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
        with self.session.post(
            f"{self.base_url}/{endpoint}", json=parameters, stream=True
        ) as session:
            session.raise_for_status()
            for line in session.iter_lines():
                if line:
                    resp = json.loads(line)
                    yield return_type(**resp) if return_type else resp

    async def _post():
        pass

    async def _get():
        pass
