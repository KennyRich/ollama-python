from ollama_python.models.generate import (
    Completion,
    Options,
    StreamCompletion,
    ChatCompletion,
    Message,
    StreamChatCompletion,
)
from ollama_python.endpoints.base import BaseAPI
from typing import BinaryIO, Optional, Generator, Union


class GenerateAPI(BaseAPI):
    def __init__(self, model: str, base_url: str = "http://localhost:11434/api"):
        """
        Initialize the Generate API endpoint

        :param model: The model to use for generating completions
        :param base_url: The base URL of the API
        """
        super().__init__(base_url=base_url)
        self.model = model

    def generate(
        self,
        prompt: str,
        images: Optional[list[Union[str, BinaryIO]]] = None,
        options: Optional[dict] = None,
        system: Optional[str] = None,
        stream: bool = False,
        format: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[list[int]] = None,
        raw: bool = False,
    ) -> Union[Completion, Generator]:
        """
        Generate a completion using the given prompt

        :param prompt: The prompt to use for generating the completion
        :param images : A list of base64-encoded images (for multimodal models such as llava)
        :param options: Additional model parameters listed in the documentation for the Modelfile such as temperature
        :param system:  System message to (overrides what is defined in the Modelfile)
        :param stream: If false the response will be returned as a single response object, rather than a stream of objects
        :param format: The format of the response, currently only support "json"
        :param template: the prompt template to use (overrides what is defined in the Modelfile)
        :param context: The context parameter returned from a previous request to /generate, this can be used to keep a short conversational memory
        :param raw: If true no formatting will be applied to the prompt. You may choose to use the raw parameter if you are specifying a full templated prompt in your request to the API.
        :return: The completion
        """
        if format != "json" and format is not None:
            raise ValueError("Only JSON format is supported")

        parameters = {
            "prompt": prompt,
            "model": self.model,
            "raw": raw,
            "stream": stream,
            "system": system,
            "context": context,
            "template": template,
        }

        if options:
            validated_options = Options(
                **options
            )  # Basically to validate the types by Pydantic
            options_dict = validated_options.model_dump(exclude_none=True)
            parameters["options"] = options_dict

        if images:
            parameters["images"] = images

        if format:
            parameters["format"] = format

        if stream:
            return self._stream(
                parameters=parameters, endpoint="generate", return_type=StreamCompletion
            )

        return self._post(
            parameters=parameters, endpoint="generate", return_type=Completion
        )

    def generate_chat_completion(
        self,
        messages: list[dict],
        format: Optional[str] = None,
        options: Optional[dict] = None,
        template: Optional[str] = None,
        stream: bool = False,
    ) -> Union[ChatCompletion, Generator]:
        """
        Generate a completion using the given prompt
        :param messages: The list of messages e.g [{"role": "user", "content": "Hello"}]
        :param options: Additional model parameters listed in the documentation for the Modelfile such as temperature
        :param stream: If false the response will be returned as a single response object, rather than a stream of objects
        :param format: The format of the response, currently only support "json"
        :param template: the prompt template to use (overrides what is defined in the Modelfile)
        """
        if format != "json" and format is not None:
            raise ValueError("Only JSON format is supported")

        # validating the message input
        [Message(**message) for message in messages]

        parameters = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "template": template,
        }

        if options:
            validated_options = Options(**options)
            options_dict = validated_options.model_dump(exclude_none=True)
            parameters["options"] = options_dict

        if format:
            parameters["format"] = format

        if stream:
            return self._stream(
                parameters=parameters, endpoint="chat", return_type=StreamChatCompletion
            )

        return self._post(
            parameters=parameters, endpoint="chat", return_type=ChatCompletion
        )
