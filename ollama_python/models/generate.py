"""Models for the OlLAMA generate endpoint"""
from pydantic import BaseModel, Field, ConfigDict, Extra
from typing import Optional, Literal


class Message(BaseModel):
    """A message prompt sent to the OLLAMA generate chat endpoint"""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message"
    )
    content: str = Field(..., description="The content of the message")
    images: Optional[bytes | str] = Field(None, description="A base64-encoded image")

    model_config = ConfigDict(extra=Extra.forbid)


class BaseCompletion(BaseModel):
    """A base completion returned by the OLLAMA generate endpoint"""

    model: str = Field(..., description="The model used to generate the response")
    created_at: str = Field(..., description="The time the request was made")
    done: bool = Field(..., description="Whether the response is complete")
    context: list[int] = Field(
        ...,
        description="An encoding of the conversation used in this response, this can be sent in the next request to keep a conversational memory",
    )
    total_duration: int = Field(..., description="Time spent generating the response")
    load_duration: int = Field(
        ..., description="Time spent in nanoseconds loading the model"
    )
    prompt_eval_count: Optional[int] = Field(
        None, description="Number of tokens in the prompt"
    )
    prompt_eval_duration: int = Field(
        ..., description="Time spent in nanoseconds evaluating the prompt"
    )
    eval_count: int = Field(..., description="Number of tokens in the response")
    eval_duration: int = Field(
        ..., description="Time in nanoseconds spent generating the response"
    )


class Completion(BaseCompletion):
    """A completion returned by the OlLAMA generate Completion endpoint"""

    response: str = Field(..., description="The generated response")


class ChatCompletion(BaseCompletion):
    """A completion returned by the OlLAMA generate Chat endpoint"""

    messages: list[Message] = Field(..., description="The generated messages")


class StreamCompletion(Completion):
    """A completion returned by the OlLAMA generate Completion endpoint when streaming the response"""

    context: Optional[list[int]] = Field(
        None,
        description="An encoding of the conversation used in this response, this can be sent in the next request to keep a conversational memory",
    )
    total_duration: Optional[int] = Field(
        None, description="Time spent generating the response"
    )
    load_duration: Optional[int] = Field(
        None, description="Time spent in nanoseconds loading the model"
    )
    prompt_eval_count: Optional[int] = Field(
        None, description="Number of tokens in the prompt"
    )
    prompt_eval_duration: Optional[int] = Field(
        None, description="Time spent in nanoseconds evaluating the prompt"
    )
    eval_count: Optional[int] = Field(
        None, description="Number of tokens in the response"
    )
    eval_duration: Optional[int] = Field(
        None, description="Time in nanoseconds spent generating the response"
    )


class StreamChatCompletion(ChatCompletion):
    """A completion result returned by the OlLAMA generate Chat endpoint when streaming the response"""

    messages: Optional[list[Message]] = None

    context: Optional[list[int]] = Field(
        None,
        description="An encoding of the conversation used in this response, this can be sent in the next request to keep a conversational memory",
    )
    total_duration: Optional[int] = Field(
        None, description="Time spent generating the response"
    )
    load_duration: Optional[int] = Field(
        None, description="Time spent in nanoseconds loading the model"
    )
    prompt_eval_count: Optional[int] = Field(
        None, description="Number of tokens in the prompt"
    )
    prompt_eval_duration: Optional[int] = Field(
        None, description="Time spent in nanoseconds evaluating the prompt"
    )
    eval_count: Optional[int] = Field(
        None, description="Number of tokens in the response"
    )
    eval_duration: Optional[int] = Field(
        None, description="Time in nanoseconds spent generating the response"
    )


class Options(BaseModel):
    """Valid options for the OlLAMA generate endpoint"""

    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: Optional[bool] = None
    stop: Optional[list[str]] = None
    numa: Optional[bool] = None
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gqa: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: Optional[bool] = None
    f16_kv: Optional[bool] = None
    vocab_only: Optional[bool] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    embedding_only: Optional[bool] = None
    rope_frequency_base: Optional[float] = None
    rope_frequency_scale: Optional[float] = None
    num_thread: Optional[int] = None

    model_config = ConfigDict(extra=Extra.forbid)
