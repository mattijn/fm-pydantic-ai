"""Pydantic AI `Model` for Apple's on-device Foundation Model, built on `apple_fm_sdk`.

Apple runs tools inside `session.respond()`, while Pydantic AI expects tool calls to be returned.
Each Pydantic AI tool is therefore registered as a proxy `fm.Tool` that suspends the generation and
hands its call to `request()`; the `ToolReturnPart`s of the next request resume it.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import itertools
import json
import mimetypes
import subprocess
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from operator import attrgetter
from pathlib import Path
from types import GenericAlias
from typing import Any

import apple_fm_sdk as fm
from apple_fm_sdk.generation_property import Property
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse, download_item
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RequestUsage

__all__ = ["AppleFoundationModel", "AppleStreamedResponse"]

PARALLEL_TOOL_CALL_WINDOW = 0.3  # seconds to wait for sibling tool calls after the first one

JSON_TYPES: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}

PromptItem = str | ImageUrl | BinaryContent


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class ToolCall:
    name: str
    arguments: str
    result: concurrent.futures.Future[str]
    tool_call_id: str = field(default_factory=new_call_id)


@dataclass
class Snapshot:
    text: str  # Apple streams the full text so far, not deltas


@dataclass
class Done:
    generation: asyncio.Task[str]  # `.result()` returns the output or raises the generation's error


Event = ToolCall | Snapshot | Done


@dataclass
class Turn:
    """One generation, suspended inside Apple's framework while its tool calls run."""

    session: fm.LanguageModelSession
    events: asyncio.Queue[Event]
    input_tokens: int  # context so far: transcript, prompt, tools, earlier outputs and tool results
    tools: list[fm.Tool]  # Apple keeps only the native tools; their Python callbacks must outlive the generation
    pending: dict[str, ToolCall] = field(default_factory=dict)
    stop: threading.Event = field(default_factory=threading.Event)
    task: asyncio.Task[str] | None = None

    async def __aiter__(self) -> AsyncIterator[Event]:
        for _ in itertools.count():
            yield await self.events.get()

    def cancel(self) -> None:
        self.stop.set()
        for call in self.pending.values():
            call.result.set_exception(RuntimeError("Generation cancelled"))
        if self.task:
            self.task.cancel()


@dataclass
class Replay:
    """Recorded tool results, served when a regenerated turn makes the same tool call again."""

    results: list[tuple[str, Any, str]]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, name: str, arguments: str) -> str | None:
        key = (name, json.loads(arguments))
        with self.lock:
            if match := next((r for r in self.results if r[:2] == key), None):
                self.results.remove(match)
                return match[2]
        return None


class AppleFoundationModel(Model):
    """Apple's on-device Foundation Model as a Pydantic AI model."""

    def __init__(self, fm_model: fm.SystemLanguageModel):
        available, reason = fm_model.is_available()
        if not available:
            raise RuntimeError(f"Apple Foundation Model is not available: {reason}")
        self.fm_model = fm_model
        super().__init__(
            profile=ModelProfile(
                supports_tools=True,
                supports_json_schema_output=True,
                supports_json_object_output=False,
                default_structured_output_mode="native",
                context_window=fm_model.context_size,
            ),
        )
        self.suspended: dict[str, Turn] = {}  # tool_call_id -> turn awaiting that call's result

    @property
    def model_name(self) -> str:
        return "system"

    @property
    def system(self) -> str:
        return "apple"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        turn = await self.open_turn(messages, model_settings, model_request_parameters, stream=False)
        # a turn started by request_stream() also emits snapshots
        event = await anext(e async for e in turn if not isinstance(e, Snapshot))

        if isinstance(event, Done):
            output = event.generation.result()
            parts: list[TextPart | ToolCallPart] = [TextPart(output)]
            usage = await self.usage_for(turn, output)
        else:
            calls = await self.suspend(turn, event)
            parts = [ToolCallPart(c.name, c.arguments, c.tool_call_id) for c in calls]
            usage = await self.usage_for(turn, calls_text(calls))
        return ModelResponse(
            parts=parts, usage=usage, model_name=self.model_name, provider_name=self.system, timestamp=utc_now()
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        turn = await self.open_turn(messages, model_settings, model_request_parameters, stream=True)
        yield AppleStreamedResponse(model_request_parameters=model_request_parameters, apple_model=self, turn=turn)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        model_settings, params = self.prepare_request(model_settings, model_request_parameters)
        instructions = self.instructions_for(messages, params)
        transcript = await fm.Transcript.from_dict(await transcript_dict(instructions, messages, with_images=False))
        tools = [
            {"name": t.name, "description": t.description, "parameters": t.parameters_json_schema}
            for t in [*params.function_tools, *params.output_tools]
        ]
        return RequestUsage(
            input_tokens=await self.token_count(transcript) + await self.token_count(tools and json.dumps(tools))
        )

    async def aclose(self) -> None:
        """Cancel generations still waiting for tool results, e.g. after an output tool ended the run."""
        for turn in {id(t): t for t in self.suspended.values()}.values():
            turn.cancel()
        self.suspended.clear()

    def abandon(self, tool_call_ids: list[str]) -> None:
        """Cancel the generations waiting for these tool calls; a request with their results regenerates the turn."""
        for turn in {id(t): t for t in map(self.suspended.__getitem__, tool_call_ids)}.values():
            for call_id in turn.pending:
                del self.suspended[call_id]
            turn.cancel()

    def instructions_for(self, messages: list[ModelMessage], params: ModelRequestParameters) -> str | None:
        parts = self._get_instruction_parts(messages, params)
        return InstructionPart.join(parts) if parts else None

    async def open_turn(
        self,
        messages: list[ModelMessage],
        settings: ModelSettings | None,
        params: ModelRequestParameters,
        *,
        stream: bool,
    ) -> Turn:
        return await self.resume(messages) or await self.start(messages, settings, params, stream=stream)

    async def resume(self, messages: list[ModelMessage]) -> Turn | None:
        """Hand tool results to the suspended generation that asked for them, if it's still alive."""
        returns = [
            p
            for p in messages[-1].parts
            if isinstance(p, (ToolReturnPart, RetryPromptPart)) and p.tool_call_id in self.suspended
        ]
        if not returns:
            return None
        turn = self.suspended[returns[0].tool_call_id]
        for part in returns:
            output = tool_output(part)
            turn.input_tokens += await self.token_count(output)
            del self.suspended[part.tool_call_id]
            turn.pending.pop(part.tool_call_id).result.set_result(output)
        return turn

    async def start(
        self,
        messages: list[ModelMessage],
        settings: ModelSettings | None,
        params: ModelRequestParameters,
        *,
        stream: bool,
    ) -> Turn:
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[Event] = asyncio.Queue()
        instructions = self.instructions_for(messages, params)
        history, prompt_items, replay = split_history(messages)
        tools = [proxy_tool(t, loop, events, replay) for t in [*params.function_tools, *params.output_tools]]

        image_dir = tempfile.TemporaryDirectory(prefix="apple-fm-")  # prompt images must outlive the generation
        prompt = await prompt_components(prompt_items, Path(image_dir.name))
        transcript = await fm.Transcript.from_dict(await transcript_dict(instructions, history, with_images=True))
        session = fm.LanguageModelSession.from_transcript(transcript, model=self.fm_model, tools=tools)

        # Apple can't count image tokens, so only text is counted
        text_transcript = await fm.Transcript.from_dict(await transcript_dict(instructions, history, with_images=False))
        prompt_text = "\n".join(c for c in prompt if isinstance(c, str))
        tokens = [
            await self.token_count(text_transcript),
            await self.token_count(prompt_text),
            await self.token_count(tools),
        ]
        turn = Turn(session, events, sum(tokens), tools)

        options = generation_options(settings)
        if params.output_mode == "native" and params.output_object:
            # Apple only streams plain text, so structured output arrives in one piece
            schema = params.output_object.json_schema
            coro: Coroutine[Any, Any, Any] = session.respond(
                prompt, json_schema=with_apple_object_keys(schema, schema["title"]), options=options
            )
        elif stream:
            # The SDK's stream iterator blocks its event loop while waiting, which would also block the proxies'
            # hand-off to this loop, so it gets a thread of its own.
            emit = partial(emit_snapshot, loop, events)
            coro = asyncio.to_thread(stream_in_thread, session, prompt, options, emit, turn.stop)
        else:
            coro = session.respond(prompt, options=options)
        turn.task = asyncio.create_task(as_text(coro))
        turn.task.add_done_callback(partial(generation_done, events, image_dir))
        return turn

    async def suspend(self, turn: Turn, first: ToolCall) -> list[ToolCall]:
        await asyncio.sleep(PARALLEL_TOOL_CALL_WINDOW)
        siblings = [turn.events.get_nowait() for _ in range(turn.events.qsize())]
        calls = [first, *(e for e in siblings if isinstance(e, ToolCall))]
        if len(calls) != len(siblings) + 1:
            raise RuntimeError(f"Unexpected events while collecting tool calls: {siblings}")
        for call in calls:
            turn.pending[call.tool_call_id] = call
            self.suspended[call.tool_call_id] = turn
        return calls

    async def usage_for(self, turn: Turn, output: str) -> RequestUsage:
        output_tokens = await self.token_count(output)
        usage = RequestUsage(input_tokens=turn.input_tokens, output_tokens=output_tokens)
        turn.input_tokens += output_tokens
        return usage

    async def token_count(self, value: Any) -> int:
        """Token count by Apple's tokenizer; 0 for empty values."""
        if not value:
            return 0
        return await self.fm_model.token_count(value)


def no_provider_url(response: AppleStreamedResponse) -> None:
    return None


@dataclass
class AppleStreamedResponse(StreamedResponse):
    apple_model: AppleFoundationModel
    turn: Turn
    started_at: datetime = field(default_factory=utc_now)

    model_name = property(attrgetter("apple_model.model_name"))
    provider_name = property(attrgetter("apple_model.system"))
    provider_url = property(no_provider_url)
    timestamp = property(attrgetter("started_at"))

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        sent = ""
        async for event in self.turn:
            if not isinstance(event, Snapshot):
                break
            delta, sent = event.text.removeprefix(sent), event.text
            if delta:
                for e in self._parts_manager.handle_text_delta(vendor_part_id="content", content=delta):
                    yield e

        if isinstance(event, Done):
            output = event.generation.result()
            if not sent:  # structured output isn't streamed
                sent = output
                for e in self._parts_manager.handle_text_delta(vendor_part_id="content", content=sent):
                    yield e
            self._usage = await self.apple_model.usage_for(self.turn, sent)
        elif isinstance(event, ToolCall):
            calls = await self.apple_model.suspend(self.turn, event)
            for c in calls:
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=c.tool_call_id, tool_name=c.name, args=c.arguments, tool_call_id=c.tool_call_id
                )
            self._usage = await self.apple_model.usage_for(self.turn, calls_text(calls))
        self._usage_received = True

    async def close_stream(self) -> None:
        self.turn.cancel()


async def as_text(generation: Coroutine[Any, Any, Any]) -> str:
    result = await generation
    return result.to_json() if isinstance(result, fm.GeneratedContent) else str(result)


def generation_done(
    events: asyncio.Queue[Event], image_dir: tempfile.TemporaryDirectory[str], task: asyncio.Task[str]
) -> None:
    image_dir.cleanup()
    events.put_nowait(Done(task))


def emit_snapshot(loop: asyncio.AbstractEventLoop, events: asyncio.Queue[Event], text: str) -> None:
    loop.call_soon_threadsafe(events.put_nowait, Snapshot(text))


def stream_in_thread(
    session: fm.LanguageModelSession,
    prompt: list[str | fm.Attachment],
    options: fm.GenerationOptions | None,
    emit: Callable[[str], None],
    stop: threading.Event,
) -> str:
    async def consume() -> str:
        text = ""
        async for text in session.stream_response(prompt, options=options):
            if stop.is_set():
                break
            emit(text)
        return text

    return asyncio.run(consume())


def proxy_tool(
    tool: ToolDefinition, loop: asyncio.AbstractEventLoop, events: asyncio.Queue[Event], replay: Replay
) -> fm.Tool:
    """An `fm.Tool` that hands its call to `request()` and waits for Pydantic AI to run it."""
    if tool.description is None:
        raise ValueError(f"Tool {tool.name!r} needs a description")

    async def call(self: fm.Tool, args: fm.GeneratedContent) -> str:
        arguments = args.to_json()
        if (recorded := replay.take(tool.name, arguments)) is not None:
            return recorded
        # runs on a thread owned by the SDK, so hand over to `loop` thread-safely
        result: concurrent.futures.Future[str] = concurrent.futures.Future()
        loop.call_soon_threadsafe(events.put_nowait, ToolCall(tool.name, arguments, result))
        return await asyncio.wrap_future(result)

    attrs = {
        "name": tool.name,
        "description": tool.description,
        "arguments_schema": SchemaBuilder(tool).build(),
        "call": call,
    }
    proxy: fm.Tool = type(f"Proxy_{tool.name}", (fm.Tool,), attrs)()
    return proxy


def split_history(messages: list[ModelMessage]) -> tuple[list[ModelMessage], list[PromptItem], Replay]:
    """Split messages into transcript history, the prompt items to respond to, and tool results to replay.

    Normally the last message holds the new user prompt. If it holds tool results that no live generation
    awaits (after a restart, or a deferred tool approved later), the turn is regenerated from its user prompt
    and repeated tool calls are answered from the recorded results.
    """
    if retries := [p for p in messages[-1].parts if isinstance(p, RetryPromptPart) and p.tool_name is None]:
        return messages[:-1], [p.model_response() for p in retries], Replay([])  # e.g. output validation failed

    for i in reversed(range(len(messages))):
        message = messages[i]
        if isinstance(message, ModelRequest) and (
            prompts := [p for p in message.parts if isinstance(p, UserPromptPart)]
        ):
            return (
                messages[:i],
                [item for p in prompts for item in user_items(p)],
                recorded_tool_results(messages[i:]),
            )
    raise ValueError("Message history has no user prompt")


def recorded_tool_results(messages: list[ModelMessage]) -> Replay:
    parts = [p for m in messages for p in m.parts]
    calls = {p.tool_call_id: p for p in parts if isinstance(p, ToolCallPart)}
    return Replay(
        [
            (calls[p.tool_call_id].tool_name, json.loads(calls[p.tool_call_id].args_as_json_str()), tool_output(p))
            for p in parts
            if isinstance(p, ToolReturnPart)
        ]
    )


async def transcript_dict(
    instructions: str | None, history: list[ModelMessage], *, with_images: bool
) -> dict[str, Any]:
    """Message history in Apple's `FoundationModels.Transcript` JSON format."""

    async def contents(items: list[PromptItem]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                out.append({"type": "text", "id": uid(), "text": item})
            elif with_images:  # the transcript only decodes JPEG data URIs
                jpeg = await asyncio.to_thread(to_jpeg, *await image_bytes(item))
                data = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
                out.append({"type": "attachment", "id": uid(), "attachment": {"type": "image", "data": data}})
        return out

    entries: list[dict[str, Any]] = (
        [{"role": "instructions", "id": uid(), "contents": await contents([instructions])}] if instructions else []
    )
    for message in history:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                match part:
                    case UserPromptPart():
                        entries.append(
                            {"role": "user", "id": uid(), "options": {}, "contents": await contents(user_items(part))}
                        )
                    case ToolReturnPart() | RetryPromptPart() if part.tool_name:
                        entries.append(
                            {
                                "role": "tool",
                                "id": part.tool_call_id,
                                "toolCallID": part.tool_call_id,
                                "toolName": part.tool_name,
                                "contents": await contents([tool_output(part)]),
                            }
                        )
            continue
        if calls := [p for p in message.parts if isinstance(p, ToolCallPart)]:
            tool_calls = [{"id": c.tool_call_id, "name": c.tool_name, "arguments": c.args_as_json_str()} for c in calls]
            entries.append({"role": "response", "id": uid(), "toolCalls": tool_calls})
        if text := "".join(p.content for p in message.parts if isinstance(p, TextPart)):
            entries.append({"role": "response", "id": uid(), "assets": [], "contents": await contents([text])})
    return {"type": "FoundationModels.Transcript", "version": "1.1", "transcript": {"entries": entries}}


def user_items(part: UserPromptPart) -> list[PromptItem]:
    """The text and images of a user prompt."""
    items = [part.content] if isinstance(part.content, str) else list(part.content)
    supported = [i for i in items if isinstance(i, (str, ImageUrl)) or (isinstance(i, BinaryContent) and i.is_image)]
    if len(supported) != len(items):
        raise NotImplementedError(f"Only text and images are supported, got {[type(i).__name__ for i in items]}")
    return supported


async def prompt_components(items: list[PromptItem], image_dir: Path) -> list[str | fm.Attachment]:
    """Prompt items as Apple prompt components: text stays text, images become files on disk."""
    components: list[str | fm.Attachment] = []
    for item in items:
        if isinstance(item, str):
            components.append(item)
            continue
        data, media_type = await image_bytes(item)
        path = image_dir / f"{uid()}{file_extension(media_type)}"
        path.write_bytes(data)
        components.append(fm.ImageAttachment(path))
    return components


async def image_bytes(item: ImageUrl | BinaryContent) -> tuple[bytes, str]:
    if isinstance(item, BinaryContent):
        return item.data, item.media_type
    downloaded = await download_item(item, data_format="bytes")
    return downloaded["data"], downloaded["data_type"]


def to_jpeg(data: bytes, media_type: str) -> bytes:
    if media_type == "image/jpeg":
        return data
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp, f"in{file_extension(media_type)}"), Path(tmp, "out.jpg")
        src.write_bytes(data)
        subprocess.run(["sips", "-s", "format", "jpeg", src, "--out", dst], check=True, capture_output=True)
        return dst.read_bytes()


def file_extension(media_type: str) -> str:
    if (extension := mimetypes.guess_extension(media_type)) is None:
        raise ValueError(f"Unknown media type: {media_type}")
    return extension


def uid() -> str:
    return str(uuid.uuid4())


def tool_output(part: ToolReturnPart | RetryPromptPart) -> str:
    return part.model_response_str() if isinstance(part, ToolReturnPart) else part.model_response()


def calls_text(calls: list[ToolCall]) -> str:
    return "\n".join(f"{c.name}({c.arguments})" for c in calls)


def generation_options(settings: ModelSettings | None) -> fm.GenerationOptions | None:
    if settings is None:
        return None
    temperature, max_tokens = settings.get("temperature"), settings.get("max_tokens")
    if temperature is None and max_tokens is None:
        return None
    return fm.GenerationOptions(temperature=temperature, maximum_response_tokens=max_tokens)


def with_apple_object_keys(schema: Any, title: str) -> Any:
    """Add the `title`, `x-order` and `additionalProperties` keys Apple's decoder requires on object schemas."""
    if isinstance(schema, list):
        return [with_apple_object_keys(s, title) for s in schema]
    if not isinstance(schema, dict):
        return schema
    schema = {k: with_apple_object_keys(v, k) for k, v in schema.items()}
    if "properties" in schema:
        schema = {"title": title, "x-order": list(schema["properties"]), "additionalProperties": False, **schema}
    return schema


class SchemaBuilder:
    """Translates a tool's JSON schema into an Apple `GenerationSchema`.

    Handles primitives, enums, arrays, optional fields (incl. `anyOf [X, null]`) and nested objects
    (incl. `$ref`/`$defs`). Apple requires every nested object schema on the root schema, so they're
    collected in `nested`.
    """

    def __init__(self, tool: ToolDefinition):
        self.tool = tool
        self.defs: dict[str, Any] = tool.parameters_json_schema.get("$defs", {})
        self.nested: dict[str, fm.GenerationSchema] = {}
        self.building: set[str] = set()

    def build(self) -> fm.GenerationSchema:
        properties = self.properties(self.resolve(self.tool.parameters_json_schema), self.tool.name)
        return fm.GenerationSchema(
            type(f"{self.tool.name}_args", (), {}), self.tool.description, properties, list(self.nested.values())
        )

    def properties(self, schema: dict[str, Any], path: str) -> list[Property]:
        required = set(schema.get("required", []))
        properties = []
        for name, prop in schema.get("properties", {}).items():
            py_type, guides = self.type_of(prop, f"{path}_{name}")
            description = prop.get("description") or self.resolve(prop).get("description")
            properties.append(Property(name, py_type if name in required else py_type | None, description, guides))
        return properties

    def resolve(self, prop: dict[str, Any]) -> dict[str, Any]:
        return self.resolve(self.defs[prop["$ref"].rsplit("/", 1)[-1]]) if "$ref" in prop else prop

    def type_of(self, prop: dict[str, Any], path: str) -> tuple[Any, list[fm.GenerationGuide]]:
        name = prop["$ref"].rsplit("/", 1)[-1] if "$ref" in prop else path
        prop = self.resolve(prop)
        if len(variants := [v for v in prop.get("anyOf", []) if v.get("type") != "null"]) == 1:
            return self.type_of(variants[0], path)
        if "enum" in prop:
            return str, [fm.GenerationGuide.anyOf([str(v) for v in prop["enum"]])]
        if prop.get("type") == "array":
            return GenericAlias(list, (self.type_of(prop["items"], f"{path}_item")[0],)), []
        if prop.get("type") == "object" and "properties" in prop:
            return self.object_type(prop, name), []
        if prop.get("type") not in JSON_TYPES:
            raise NotImplementedError(f"Unsupported schema in tool {self.tool.name!r}: {prop}")
        return JSON_TYPES[prop["type"]], []

    def object_type(self, prop: dict[str, Any], name: str) -> type:
        if name in self.building:
            raise NotImplementedError(f"Recursive schema {name!r} in tool {self.tool.name!r}")
        if name not in self.nested:
            self.building.add(name)
            properties = self.properties(prop, name)  # registers child objects first
            self.building.discard(name)
            self.nested[name] = fm.GenerationSchema(
                type(name, (), {}), prop.get("description"), properties, list(self.nested.values())
            )
        return self.nested[name].type_class
