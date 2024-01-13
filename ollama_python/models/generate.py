"""Models for the OlLAMA generate endpoint"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, Union


class Message(BaseModel):
    """A message prompt sent to the OLLAMA generate chat endpoint"""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message"
    )
    content: str = Field(..., description="The content of the message")
    images: Optional[list[Union[bytes, str]]] = Field(
        None, description="A list of base64-encoded images"
    )

    model_config = ConfigDict(extra="forbid")


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

    message: list[Message] = Field(..., description="The generated messages")


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

    message: Optional[list[Message]] = None

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
    seed: Optional[int] = Field(
        None,
        description="Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)",
    )
    num_predict: Optional[int] = Field(
        None,
        description="Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)",
    )
    top_k: Optional[int] = Field(
        None,
        description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
    )
    top_p: Optional[float] = Field(
        None,
        description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
    )
    tfs_z: Optional[float] = Field(
        None,
        description="Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)",
    )
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = Field(
        None,
        description="Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)",
    )
    temperature: Optional[float] = Field(
        None,
        description="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)",
    )
    repeat_penalty: Optional[float] = Field(
        None,
        description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)",
    )
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = Field(
        None,
        description="Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)",
    )
    mirostat_tau: Optional[float] = Field(
        None,
        description="Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)",
    )
    mirostat_eta: Optional[float] = Field(
        None,
        description="Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)",
    )
    penalize_newline: Optional[bool] = None
    stop: Optional[list[str]] = Field(
        None,
        description="Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate stop parameters in a modelfile.",
    )
    numa: Optional[bool] = None
    num_ctx: Optional[int] = Field(
        None,
        description="Sets the size of the context window used to generate the next token. (Default: 2048)",
    )
    num_batch: Optional[int] = None
    num_gqa: Optional[int] = Field(
        None,
        description="The number of GQA groups in the transformer layer. Required for some models, for example it is 8 for llama2:70b",
    )
    num_gpu: Optional[int] = Field(
        None,
        description="The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable.",
    )
    main_gpu: Optional[int] = None
    low_vram: Optional[bool] = None
    f16_kv: Optional[bool] = None
    vocab_only: Optional[bool] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    embedding_only: Optional[bool] = None
    rope_frequency_base: Optional[float] = None
    rope_frequency_scale: Optional[float] = None
    num_thread: Optional[int] = Field(
        None,
        description="Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores).",
    )

    model_config = ConfigDict(extra="forbid")
