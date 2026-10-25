# Installation

Requires macOS 27 with Apple Intelligence, the Xcode Command Line Tools and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mattijn/fm-pydantic-ai
cd fm-pydantic-ai
uv sync
```

`uv sync` builds the vendored Apple Foundation Models SDK. To check:

```bash
uv run python weather_agent.py
```
