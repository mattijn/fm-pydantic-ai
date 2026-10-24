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

`weather_mcp.py` serves the weather agent as a local MCP server. When a question needs your location, the on-device
model calls a `get_user_location` tool that asks you via MCP elicitation. On the classic protocol the question goes over
the open connection while the generation waits; on the stateless protocol (2026-07-28) the call ends with
`input_required`, the message history travels in the sealed request state, and the retry regenerates the turn while the
bridge replays your answer. `.mcp.json` registers the server for Claude Code.

`evals/run.py` runs every scenario ten times with Pydantic Evals and grades each trial with deterministic checks on
its trajectory: tool calls and their arguments, questions to the user, rounds and output. The outcomes, transcripts and
OpenTelemetry spans of every trial go to `evals/results.json`.

```bash
uv run --group evals python -m evals.run --repeat 10
```

MIT licensed, except `vendor/apple_fm_sdk-0.2.1` (Apache 2.0, Apple).

```bash
uv run --group dev ruff check . && uv run --group dev ruff format --check . && uv run --group dev mypy && uv run --group dev pytest
```
