# fm-pydantic-ai

[Pydantic AI](https://ai.pydantic.dev) on Apple's on-device Foundation Model. See the [documentation](https://mattijn.github.io/fm-pydantic-ai/).

```bash
uv sync
uv run python weather_agent.py
```

`vendor/apple_fm_sdk-0.2.1` is Apple's SDK from PyPI, patched to build with the Command Line Tools and to run on
Python 3.14.

```bash
uv run --group dev ruff check . && uv run --group dev ruff format --check . && uv run --group dev mypy && uv run --group dev pytest
```

MIT licensed, except `vendor/apple_fm_sdk-0.2.1` (Apache 2.0, Apple).
