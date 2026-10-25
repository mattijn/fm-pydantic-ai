# Usage

## Tools

```python
import apple_fm_sdk as fm
from pydantic_ai import Agent

from apple_fm_model import AppleFoundationModel

agent = Agent(AppleFoundationModel(fm.SystemLanguageModel()))


@agent.tool_plain
def get_temperature(city: str) -> float:
    """Get the current temperature in Celsius."""
    return 3.0


print(agent.run_sync("How cold is it in Kampen?").output)
```

Apple runs tools inside its own generation. The model pauses that generation until Pydantic AI sends the result back. Every tool needs a docstring.

## MCP server

`weather_mcp.py` serves the weather agent over MCP, and `.mcp.json` registers it for Claude Code. Start a session in the repository and approve the `weather` server.

When a question needs your location, the model asks for it through MCP elicitation: over the open connection on the classic protocol, or with `input_required` and a replayed turn on the stateless protocol.
