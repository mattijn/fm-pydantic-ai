import asyncio
import os
import sys
from typing import Literal

import pytest
from mcp import StdioServerParameters, types
from mcp.client import Client, ClientRequestContext
from mcp.server.mcpserver import MCPServer

from weather_mcp import QUESTION, server

Mode = Literal["legacy", "auto"]
STDIO = StdioServerParameters(command=sys.executable, args=["weather_mcp.py"], env=dict(os.environ))


def ask(
    question: str, action: Literal["accept", "decline"], mode: Mode, target: MCPServer | StdioServerParameters = server
) -> tuple[list[str], types.CallToolResult]:
    asked: list[str] = []

    async def reply(context: ClientRequestContext, params: types.ElicitRequestParams) -> types.ElicitResult:
        asked.append(params.message)
        return types.ElicitResult(action=action, content={"city": "Kampen"} if action == "accept" else None)

    async def call() -> types.CallToolResult:
        async with Client(target, mode=mode, elicitation_callback=reply) as client:
            return await client.call_tool("ask_weather", {"question": question})

    return asked, asyncio.run(call())


def text(result: types.CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, types.TextContent))


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_agent_asks_for_the_location(mode: Mode) -> None:
    asked, result = ask("What's the weather where I am?", "accept", mode)
    assert asked == [QUESTION]
    assert "Kampen" in text(result)


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_agent_does_not_ask_when_the_city_is_given(mode: Mode) -> None:
    asked, result = ask("What's the weather in Paris?", "accept", mode)
    assert asked == []
    assert not result.is_error


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_declined_question_reaches_the_agent(mode: Mode) -> None:
    asked, result = ask("What's the weather where I am?", "decline", mode)
    assert asked == [QUESTION]
    assert not result.is_error
    assert "Kampen" not in text(result)


def test_stateless_round_trip_over_stdio() -> None:
    asked, result = ask("What's the weather where I am?", "accept", "auto", STDIO)
    assert asked == [QUESTION]
    assert "Kampen" in text(result)
