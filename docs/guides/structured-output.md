# Structured Output

vllm-mlx supports structured output (constrained decoding) for cases where the model must follow a specific output format.

Under the hood, this uses [Outlines](https://github.com/dottxt-ai/outlines) for token-level enforcement, so invalid tokens are blocked during generation rather than cleaned up afterward.

## Installation

Structured output requires the optional `outlines` dependency:

```bash
pip install 'vllm-mlx[structured]'
```

Currently this feature requires `outlines==1.0.4`.

## OpenAI-Compatible API

Use `response_format` with `type: "json_schema"` to force output that matches a JSON schema.

### Curl Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Return a person record for Alice, age 30"}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
          },
          "required": ["name", "age"]
        }
      }
    }
  }'
```

Example response content:

```json
{"name":"Alice","age":30}
```

### Another Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "List 3 colors"}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "colors",
        "schema": {
          "type": "object",
          "properties": {
            "colors": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["colors"]
        }
      }
    }
  }'
```

## Python API

For direct engine usage, pass `GuidedDecodingParams` with `guided_decoding=`.

### Simple Engine

```python
from vllm_mlx.engine import SimpleEngine
from vllm_mlx.guided_decoding import GuidedDecodingParams

engine = SimpleEngine("mlx-community/Llama-3.2-3B-Instruct-4bit")
await engine.start()

schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country": {"type": "string"}
    },
    "required": ["city", "country"]
}

output = await engine.generate(
    prompt="Return Tokyo as a city record",
    max_tokens=100,
    guided_decoding=GuidedDecodingParams(
        kind="json_schema",
        json_schema=schema,
    ),
)

print(output.text)

await engine.stop()
```

### Batched Engine

```python
from vllm_mlx.engine import BatchedEngine
from vllm_mlx.guided_decoding import GuidedDecodingParams

engine = BatchedEngine("mlx-community/Llama-3.2-3B-Instruct-4bit")
await engine.start()

schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "confidence"]
}

output = await engine.generate(
    prompt="Answer whether Paris is in France",
    max_tokens=100,
    guided_decoding=GuidedDecodingParams(
        kind="json_schema",
        json_schema=schema,
    ),
)

print(output.text)

await engine.stop()
```

## Thinking Models

For reasoning models that emit `<think>...</think>` tags, enable thinking support in guided decoding:

```python
from vllm_mlx.guided_decoding import GuidedDecodingParams

guided = GuidedDecodingParams(
    kind="json_schema",
    json_schema={
        "type": "object",
        "properties": {
            "result": {"type": "string"}
        },
        "required": ["result"]
    },
    enable_thinking=True,
)
```

This is useful for models like Qwen3.5 that may think before producing the final structured result.

## Supported Constraint Types

| Type | Description |
|------|-------------|
| `json_schema` | Generate JSON matching a JSON Schema |
| `regex` | Generate text matching a regular expression |

### Regex Example

```python
from vllm_mlx.guided_decoding import GuidedDecodingParams

guided = GuidedDecodingParams(
    kind="regex",
    regex=r"\d{4}"
)
```

## Limitations

- Requires `outlines==1.0.4`
- Guided decoding processors are stateful, so each request needs its own processor instance
- OpenAI-compatible API support uses `response_format` with `json_schema`
- Thinking-model support must be enabled explicitly with `enable_thinking=True`

## Related

- [OpenAI-Compatible Server](server.md)
- [Python API](python-api.md)
- [Reasoning Models](reasoning.md)