# fm-pydantic-ai

Pydantic AI on Apple's on-device Foundation Model (macOS 27), via Apple's Python SDK.

```bash
uv run python weather_agent.py
```

`apple_fm_model.py` is a Pydantic AI `Model` with tools, structured output, streaming, token usage
and image input. Apple runs tools inside `session.respond()`; the model registers a proxy tool per
Pydantic AI tool that suspends the generation until Pydantic AI sends the tool result back.

`vendor/apple_fm_sdk-0.2.1` is the PyPI sdist with two local patches: no Xcode check in `build_backend.py`, so the
Command Line Tools can build it, and optional-property detection in `generation_property.py` that works on Python 3.14.

MIT licensed, except `vendor/apple_fm_sdk-0.2.1` (Apache 2.0, Apple).

```bash
uv run --group dev ruff check . && uv run --group dev ruff format --check . && uv run --group dev mypy && uv run --group dev pytest
```
