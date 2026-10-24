import asyncio

import apple_fm_sdk as fm
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from apple_fm_model import (
    AppleFoundationModel,
    SchemaBuilder,
    split_history,
    transcript_dict,
    user_items,
    with_apple_object_keys,
)
from weather_agent import weather_agent


class Address(BaseModel):
    street: str
    city: str


class Guest(BaseModel):
    name: str
    age: int | None
    address: Address


class AddGuests(BaseModel):
    guests: list[Guest]


class Node(BaseModel):
    children: list["Node"]


class Plan(BaseModel):
    city: str
    needs_tent: bool


def tool_def(args: type[BaseModel], description: str | None = "A tool.") -> ToolDefinition:
    return ToolDefinition(name="tool", description=description, parameters_json_schema=args.model_json_schema())


def tool_history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart("Weather in Kampen?")]),
        ModelResponse(parts=[ToolCallPart("get_weather", '{"city": "Kampen"}', "call_1")]),
        ModelRequest(parts=[ToolReturnPart("get_weather", "3°C", "call_1")]),
    ]


@pytest.fixture(scope="module")
def model() -> AppleFoundationModel:
    return AppleFoundationModel(fm.SystemLanguageModel())


def test_apple_object_keys_are_added_to_nested_objects() -> None:
    schema = with_apple_object_keys(AddGuests.model_json_schema(), "AddGuests")
    assert schema["x-order"] == ["guests"]
    assert schema["$defs"]["Guest"] == {**schema["$defs"]["Guest"], "title": "Guest", "additionalProperties": False}
    assert schema["$defs"]["Guest"]["x-order"] == ["name", "age", "address"]


def test_nested_tool_arguments() -> None:
    schema = SchemaBuilder(tool_def(AddGuests)).build().to_dict()
    assert set(schema["$defs"]) == {"Guest", "Address"}
    assert schema["$defs"]["Guest"]["properties"]["address"] == {"$ref": "#/$defs/Address"}


def test_recursive_tool_arguments_are_rejected() -> None:
    with pytest.raises(NotImplementedError, match="Recursive"):
        SchemaBuilder(tool_def(Node)).build()


def test_tools_need_a_description(model: AppleFoundationModel) -> None:
    def undocumented(city: str) -> str:
        return city

    with pytest.raises(ValueError, match="needs a description"):
        Agent(model, tools=[undocumented]).run_sync("Hi")


def test_history_ending_in_tool_results_replays_them() -> None:
    history, prompt, replay = split_history(tool_history())
    assert (history, prompt) == ([], ["Weather in Kampen?"])
    assert replay.take("get_weather", '{"city": "Utrecht"}') is None
    assert replay.take("get_weather", '{"city":"Kampen"}') == "3°C"
    assert replay.take("get_weather", '{"city": "Kampen"}') is None


def test_history_without_user_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="no user prompt"):
        split_history(tool_history()[1:])


def test_unsupported_prompt_content_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="Only text and images"):
        user_items(UserPromptPart(["Listen:", BinaryContent(b"", media_type="audio/wav")]))


def test_history_converts_to_an_apple_transcript() -> None:
    data = asyncio.run(transcript_dict("Be brief.", tool_history(), with_images=False))
    assert [e["role"] for e in data["transcript"]["entries"]] == ["instructions", "user", "response", "tool"]
    asyncio.run(fm.Transcript.from_dict(data))


def test_weather_agent_uses_its_tools() -> None:
    result = weather_agent.run_sync("What's the weather in Paris, in Celsius?")
    called = {p.tool_name for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)}
    assert {"get_lat_long", "get_weather"} <= called


def test_structured_output(model: AppleFoundationModel) -> None:
    result = Agent(model, output_type=Plan).run_sync("Party in Kampen in January. Plan it.")
    assert result.output.city == "Kampen"


def test_streaming(model: AppleFoundationModel) -> None:
    async def stream() -> list[str]:
        async with Agent(model).run_stream("Write three sentences about Kampen.") as result:
            return [delta async for delta in result.stream_text(delta=True)]

    assert len(asyncio.run(stream())) > 1
