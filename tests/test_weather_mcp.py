import asyncio
from typing import Literal

from mcp import types
from mcp.client import Client, ClientRequestContext

from weather_mcp import server


def ask(question: str, action: Literal["accept", "decline"]) -> tuple[list[str], types.CallToolResult]:
    asked: list[str] = []

    async def reply(context: ClientRequestContext, params: types.ElicitRequestParams) -> types.ElicitResult:
        asked.append(params.message)
        return types.ElicitResult(action=action, content={"city": "Kampen"} if action == "accept" else None)

    async def call() -> types.CallToolResult:
        async with Client(server, mode="legacy", elicitation_callback=reply) as client:
            return await client.call_tool("ask_weather", {"question": question})

    return asked, asyncio.run(call())


def text(result: types.CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, types.TextContent))


def test_agent_asks_for_the_location_via_elicitation() -> None:
    asked, result = ask("What's the weather where I am?", "accept")
    assert asked == ["In which city are you?"]
    assert "Kampen" in text(result)


def test_agent_does_not_ask_when_the_city_is_given() -> None:
    asked, result = ask("What's the weather in Paris?", "accept")
    assert asked == []
    assert not result.is_error


def test_declined_elicitation_reaches_the_agent() -> None:
    asked, result = ask("What's the weather where I am?", "decline")
    assert asked
    assert not result.is_error
    assert "Kampen" not in text(result)
