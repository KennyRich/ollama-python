from typing import Optional
from ollama_python.endpoints.base import BaseAPI
from ollama_python.models.generate import Options
from ollama_python.models.embedding import Embedding


class EmbeddingAPI(BaseAPI):
    def __init__(self, model: str, base_url: str = "http://localhost:11434/api"):
        """
        Initialize the embedding API
        :param base_url: The base URL of the API
        """
        super().__init__(base_url=base_url)
        self.model = model

    def get_embedding(self, prompt: str, options: Optional[dict] = None) -> Embedding:
        """
        Get the embedding for the given prompt
        :param prompt: The prompt to get the embedding for
        :param options: Additional model parameters listed in the documentation for the Modelfile such as temperature
        :return: The embedding
        """
        parameters = {"prompt": prompt, "model": self.model}

        if options:
            validated_options = Options(
                **options
            )  # Basically to validate the types by Pydantic
            options_dict = validated_options.model_dump(exclude_none=True)
            parameters["options"] = options_dict

        return self._post(
            parameters=parameters, endpoint="embedding", return_type=Embedding
        )
