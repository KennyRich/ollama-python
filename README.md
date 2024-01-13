# Ollama Python Library
The ollama python library provides the easiest way to integrate yout python project with [Ollama](https://github.com/KennyRich/ollama-python)


## Getting Started
This requires a python version of 3.9 or higher

```shell
pip install ollama
```

The python package splits the functionality into three core endpoints
1. Model Management Endpoints: This includes the ability to create, delete, pull, push and list models amongst others
2. Generate Endpoint: This includes the generate and chat endpoints in Ollama
3. Embedding Endpoint: This includes the ability to generate embeddings for a given text

Pydantic is used to verify user input and Responses from the server are parsed into pydantic models

## Example Usage
### Generate Endpoint
#### Completions (Generate)
##### Without Streaming
```python
from ollama_python.endpoints import GenerateAPI

api = GenerateAPI(base_url="http://localhost:8000", model="mistral")
result = api.generate(prompt="Hello World", options=dict(num_tokens=10), format="json")
```

##### With Streaming
```python
from ollama_python.endpoints import GenerateAPI

api = GenerateAPI(base_url="http://localhost:8000", model="mistral")
for res in api.generate(prompt="Hello World", options=dict(num_tokens=10), format="json", stream=True):
    print(res.response)
```

#### Chat Completions
##### Without Streaming
```python
from ollama_python.endpoints import GenerateAPI

api = GenerateAPI(base_url="http://localhost:8000", model="mistral")
messages = [{'role': 'user', 'content': 'Why is the sky blue?'}]

result = api.generate_chat_completion(messages=messages, options=dict(num_tokens=10), format="json")
```

##### With Streaming
```python
from ollama_python.endpoints import GenerateAPI

api = GenerateAPI(base_url="http://localhost:8000", model="mistral")
messages = [{'role': 'user', 'content': 'Why is the sky blue?'}]

for res in api.generate_chat_completion(messages=messages, options=dict(num_tokens=10), format="json", stream=True):
    print(res.message)
```

###### Chat request with images
```python
from ollama_python.endpoints import GenerateAPI

api = GenerateAPI(base_url="http://localhost:8000", model="llava")
messages = [{'role': 'user', 'content': 'What is in this image', 'image': 'iVBORw0KGgoAAAANSUhEUgAAAG0AAABmCAYAAADBPx+VAAAACXBIWXMAAAsTAAALEwEAmp'}]

result = api.generate_chat_completion(messages=messages, options=dict(num_tokens=10), format="json")
print(result.message)
```

### Model Management Endpoints
####  Create a model
##### Without Streaming
```python
from ollama_python.endpoints import ModelManagementAPI

api = ModelManagementAPI(base_url="http://localhost:8000")
result = api.create(name="test_model", model_file="random model_file")
```
##### With Streaming
```python
from ollama_python.endpoints import ModelManagementAPI

api = ModelManagementAPI(base_url="http://localhost:8000")
for res in api.create(name="test_model", model_file="random model_file", stream=True):
    print(res.status)
```

### Check if a blob exists
```python
from ollama_python.endpoints import ModelManagementAPI

api = ModelManagementAPI(base_url="http://localhost:8000")
result = api.check_blob_exists(digest="sha256:29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2")
```

### Create a blob
```python
from ollama_python.endpoints import ModelManagementAPI

api = ModelManagementAPI(base_url="http://localhost:8000")
result = api.create_blob(digest="sha256:29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2")
```

### List local models
```python
from ollama_python.endpoints import ModelManagementAPI

api = ModelManagementAPI(base_url="http://localhost:8000")
result = api.list_local_models()

print(result.models)
```

### Show model information
```python
from ollama_python.endpoints import ModelManagementAPI

api = ModelManagementAPI(base_url="http://localhost:8000")
result = api.show(name="mistral")

print(result.details)
```

### Copy a model
```python
from ollama_python.endpoints import ModelManagementAPI

api = ModelManagementAPI(base_url="http://localhost:8000")
result = api.copy(source="mistral", destination="mistral_copy")
```

### Delete a model
```python
from ollama_python.endpoints import ModelManagementAPI

api = ModelManagementAPI(base_url="http://localhost:8000")
api.delete(name="mistral_copy")
```



### Valid Options

| Parameter      | Description                                                                                                                                                                                                                                             | Value Type | Example Usage        |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------------------- |
| mirostat       | Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)                                                                                                                                         | int        | mirostat 0           |
| mirostat_eta   | Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)                        | float      | mirostat_eta 0.1     |
| mirostat_tau   | Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)                                                                                                         | float      | mirostat_tau 5.0     |
| num_ctx        | Sets the size of the context window used to generate the next token. (Default: 2048)                                                                                                                                                                    | int        | num_ctx 4096         |
| num_gqa        | The number of GQA groups in the transformer layer. Required for some models, for example it is 8 for llama2:70b                                                                                                                                         | int        | num_gqa 1            |
| num_gpu        | The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable.                                                                                                                                            | int        | num_gpu 50           |
| num_thread     | Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores). | int        | num_thread 8         |
| repeat_last_n  | Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)                                                                                                                                           | int        | repeat_last_n 64     |
| repeat_penalty | Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)                                                                     | float      | repeat_penalty 1.1   |
| temperature    | The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)                                                                                                                                     | float      | temperature 0.7      |
| seed           | Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)                                                                                       | int        | seed 42              |
| stop           | Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate `stop` parameters in a modelfile.                                      | string     | stop "AI assistant:" |
| tfs_z          | Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)                                               | float      | tfs_z 1              |
| num_predict    | Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)                                                                                                                                   | int        | num_predict 42       |
| top_k          | Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)                                                                        | int        | top_k 40             |
| top_p          | Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)                                                                 | float      | top_p 0.9            |


## Todo
Add support for Asynchronous version of the library

## To Contribute
1. Clone the repo
2. Run `poetry install`
3. Run `pre-commit install`

Then you're ready to contribute to the repo
