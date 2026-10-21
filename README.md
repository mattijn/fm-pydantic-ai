# fm-pydantic-ai

Pydantic AI on Apple's on-device Foundation Model (macOS 27), via Apple's Python SDK.

```bash
uv run python weather_agent.py
```

`apple_fm_model.py` is a Pydantic AI `Model` with tools, structured output, streaming, token usage
and image input. Apple runs tools inside `session.respond()`; the model registers a proxy tool per
Pydantic AI tool that suspends the generation until Pydantic AI sends the tool result back.

`vendor/apple_fm_sdk-0.2.1` is the PyPI sdist without the Xcode check in
`build_backend.py`, so the Command Line Tools can build it.
