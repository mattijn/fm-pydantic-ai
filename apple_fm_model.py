"""Pydantic AI `Model` for Apple's on-device Foundation Model, built on `apple_fm_sdk`.

Apple runs tools inside `session.respond()`, while Pydantic AI expects tool calls to be returned.
Each Pydantic AI tool is therefore registered as a proxy `fm.Tool` that suspends the generation and
hands its call to `request()`; the `ToolReturnPart`s of the next request resume it.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import json
import mimetypes
import subprocess
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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

PARALLEL_TOOL_CALL_WINDOW = 0.3  # seconds to wait for sibling tool calls after the first one

_JSON_TYPES: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}


@dataclass
class _ToolCall:
    name: str
    arguments: str
    result: concurrent.futures.Future[str]
    tool_call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")


@dataclass
class _Snapshot:
    text: str  # Apple streams the full text so far, not deltas


@dataclass
class _Done:
    output: str | None = None
    error: BaseException | None = None


_Event = _ToolCall | _Snapshot | _Done


@dataclass
class _Turn:
    """One generation, suspended inside Apple's framework while its tool calls run."""

    session: fm.LanguageModelSession
    events: asyncio.Queue[_Event]
    pending: dict[str, _ToolCall] = field(default_factory=dict)
    stop: threading.Event = field(default_factory=threading.Event)
    task: asyncio.Task[None] | None = None
    input_tokens: int = 0  # context so far: transcript, prompt, tools, earlier outputs and tool results

    def cancel(self) -> None:
        self.stop.set()
        for call in self.pending.values():
            if not call.result.done():
                call.result.set_result("Cancelled.")
        if self.task:
            self.task.cancel()


@dataclass
class _Replay:
    """Recorded tool results, served when a regenerated turn calls the same tool again."""

    results: list[tuple[str, Any, str]]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, name: str, arguments: str) -> str | None:
        args = _parse_json(arguments)
        with self.lock:
            if match := next((r for r in self.results if r[:2] == (name, args)), None) or next(
                (r for r in self.results if r[0] == name), None
            ):
                self.results.remove(match)
                return match[2]
        return None


class AppleFoundationModel(Model):
    """Apple's on-device Foundation Model as a Pydantic AI model."""

    def __init__(
        self,
        *,
        use_case: fm.SystemLanguageModelUseCase = fm.SystemLanguageModelUseCase.GENERAL,
        guardrails: fm.SystemLanguageModelGuardrails = fm.SystemLanguageModelGuardrails.DEFAULT,
        settings: ModelSettings | None = None,
    ):
        self._fm = fm.SystemLanguageModel(use_case=use_case, guardrails=guardrails)
        available, reason = self._fm.is_available()
        if not available:
            raise RuntimeError(f"Apple Foundation Model is not available: {reason}")
        super().__init__(
            settings=settings,
            profile=ModelProfile(
                supports_tools=True,
                supports_json_schema_output=True,
                supports_json_object_output=False,
                default_structured_output_mode="native",
                context_window=self._fm.context_size,
            ),
        )
        self._suspended: dict[str, _Turn] = {}  # tool_call_id -> turn awaiting that call's result

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
        turn = await self._turn(messages, model_settings, model_request_parameters, stream=False)
        while isinstance(event := await turn.events.get(), _Snapshot):  # from a turn started by request_stream()
            pass

        if isinstance(event, _Done):
            if event.error:
                raise event.error
            parts = [TextPart(event.output or "")]
            usage = await self._usage(turn, event.output)
        else:
            calls = await self._suspend(turn, event)
            parts = [ToolCallPart(c.name, c.arguments, c.tool_call_id) for c in calls]
            usage = await self._usage(turn, _calls_text(calls))
        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=self.model_name,
            provider_name=self.system,
            timestamp=datetime.now(timezone.utc),
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
        turn = await self._turn(messages, model_settings, model_request_parameters, stream=True)
        yield AppleStreamedResponse(model_request_parameters=model_request_parameters, _model=self, _turn=turn)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        model_settings, params = self.prepare_request(model_settings, model_request_parameters)
        instructions = InstructionPart.join(self._get_instruction_parts(messages, params) or [])
        transcript = await fm.Transcript.from_dict(await _transcript_dict(instructions, messages))
        tools = [
            {"name": t.name, "description": t.description, "parameters": t.parameters_json_schema}
            for t in [*params.function_tools, *params.output_tools]
        ]
        return RequestUsage(input_tokens=await self._count(transcript) + await self._count(tools and json.dumps(tools)))

    async def aclose(self) -> None:
        """Cancel generations still waiting for tool results, e.g. after an output tool ended the run."""
        for turn in {id(t): t for t in self._suspended.values()}.values():
            turn.cancel()
        self._suspended.clear()

    async def _turn(
        self, messages: list[ModelMessage], settings: ModelSettings | None, params: ModelRequestParameters, *, stream: bool
    ) -> _Turn:
        return await self._resume(messages) or await self._start(messages, settings, params, stream=stream)

    async def _resume(self, messages: list[ModelMessage]) -> _Turn | None:
        """Hand tool results to the suspended generation that asked for them, if it's still alive."""
        returns = [
            p
            for p in messages[-1].parts
            if isinstance(p, (ToolReturnPart, RetryPromptPart)) and p.tool_call_id in self._suspended
        ]
        if not returns:
            return None
        turn = self._suspended[returns[0].tool_call_id]
        for part in returns:
            output = _tool_output(part)
            turn.input_tokens += await self._count(output)
            del self._suspended[part.tool_call_id]
            turn.pending.pop(part.tool_call_id).result.set_result(output)
        return turn

    async def _start(
        self, messages: list[ModelMessage], settings: ModelSettings | None, params: ModelRequestParameters, *, stream: bool
    ) -> _Turn:
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[_Event] = asyncio.Queue()
        instructions = InstructionPart.join(self._get_instruction_parts(messages, params) or [])
        history, prompt_items, replay = _split_history(messages)
        tools = [_proxy_tool(t, loop, events, replay) for t in [*params.function_tools, *params.output_tools]]

        image_dir = tempfile.TemporaryDirectory(prefix="apple-fm-")  # prompt images must outlive the generation
        prompt = await _prompt_components(prompt_items, Path(image_dir.name))
        transcript = await fm.Transcript.from_dict(await _transcript_dict(instructions, history))
        session = fm.LanguageModelSession.from_transcript(transcript, model=self._fm, tools=tools)
        turn = _Turn(session, events)
        # Apple can't count image tokens, so only the prompt's text is counted
        prompt_text = "\n".join(c for c in prompt if isinstance(c, str))
        turn.input_tokens = sum([await self._count(transcript), await self._count(prompt_text), await self._count(tools)])

        options = _generation_options(settings)
        if params.output_mode == "native" and params.output_object:
            # Apple only streams plain text, so structured output arrives in one piece
            schema = _with_apple_object_keys(params.output_object.json_schema)
            coro = session.respond(prompt, json_schema=schema, options=options)
        elif not stream:
            coro = session.respond(prompt, options=options)
        else:
            # The SDK's stream iterator blocks its event loop while waiting, which would also block the proxies'
            # hand-off to this loop, so it gets a thread of its own.
            args = (session, prompt, options, loop, events, turn.stop, image_dir.cleanup)
            threading.Thread(target=_stream_in_thread, args=args, daemon=True).start()
            return turn
        turn.task = asyncio.create_task(_respond(coro, events, image_dir.cleanup))
        return turn

    async def _suspend(self, turn: _Turn, first: _ToolCall) -> list[_ToolCall]:
        calls = [first]
        with contextlib.suppress(TimeoutError):
            while isinstance(event := await asyncio.wait_for(turn.events.get(), PARALLEL_TOOL_CALL_WINDOW), _ToolCall):
                calls.append(event)
            turn.events.put_nowait(event)  # not a tool call: leave it for the next segment
        for call in calls:
            turn.pending[call.tool_call_id] = call
            self._suspended[call.tool_call_id] = turn
        return calls

    async def _usage(self, turn: _Turn, output: str | None) -> RequestUsage:
        output_tokens = await self._count(output)
        usage = RequestUsage(input_tokens=turn.input_tokens, output_tokens=output_tokens)
        turn.input_tokens += output_tokens
        return usage

    async def _count(self, value: Any) -> int:
        """Token count by Apple's tokenizer; 0 for empty or uncountable values."""
        if not value:
            return 0
        try:
            return await self._fm.token_count(value)
        except fm.FoundationModelsError:
            return 0


@dataclass
class AppleStreamedResponse(StreamedResponse):
    _model: AppleFoundationModel
    _turn: _Turn
    _timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    model_name = property(lambda self: self._model.model_name)
    provider_name = property(lambda self: self._model.system)
    provider_url = property(lambda self: None)
    timestamp = property(lambda self: self._timestamp)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        sent = ""
        while isinstance(event := await self._turn.events.get(), _Snapshot):
            delta, sent = event.text.removeprefix(sent), event.text
            if delta:
                for e in self._parts_manager.handle_text_delta(vendor_part_id="content", content=delta):
                    yield e

        if isinstance(event, _Done):
            if event.error:
                raise event.error
            if event.output and not sent:  # structured output isn't streamed
                sent = event.output
                for e in self._parts_manager.handle_text_delta(vendor_part_id="content", content=sent):
                    yield e
            self._usage = await self._model._usage(self._turn, sent)
        else:
            calls = await self._model._suspend(self._turn, event)
            for c in calls:
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=c.tool_call_id, tool_name=c.name, args=c.arguments, tool_call_id=c.tool_call_id
                )
            self._usage = await self._model._usage(self._turn, _calls_text(calls))
        self._usage_received = True

    async def close_stream(self) -> None:
        self._turn.cancel()


async def _respond(coro: Any, events: asyncio.Queue[_Event], cleanup: Callable[[], None]) -> None:
    try:
        result = await coro
        events.put_nowait(_Done(output=result.to_json() if isinstance(result, fm.GeneratedContent) else str(result)))
    except Exception as e:
        events.put_nowait(_Done(error=e))
    finally:
        cleanup()


def _stream_in_thread(
    session: fm.LanguageModelSession,
    prompt: list[Any],
    options: Optional[fm.GenerationOptions],
    loop: asyncio.AbstractEventLoop,
    events: asyncio.Queue[_Event],
    stop: threading.Event,
    cleanup: Callable[[], None],
) -> None:
    def emit(event: _Event) -> None:
        with contextlib.suppress(RuntimeError):  # the caller's loop is gone
            loop.call_soon_threadsafe(events.put_nowait, event)

    async def consume() -> str:
        text = ""
        async for text in session.stream_response(prompt, options=options):
            if stop.is_set():
                break
            emit(_Snapshot(text))
        return text

    try:
        emit(_Done(output=asyncio.run(consume())))
    except Exception as e:
        emit(_Done(error=e))
    finally:
        cleanup()


def _proxy_tool(
    tool: ToolDefinition, loop: asyncio.AbstractEventLoop, events: asyncio.Queue[_Event], replay: _Replay
) -> fm.Tool:
    """An `fm.Tool` that hands its call to `request()` and waits for Pydantic AI to run it."""
    arguments_schema = _SchemaBuilder(tool).build()

    async def call(self: fm.Tool, args: fm.GeneratedContent) -> str:
        arguments = args.to_json()
        if (recorded := replay.take(tool.name, arguments)) is not None:
            return recorded
        # runs on a thread owned by the SDK, so hand over to `loop` thread-safely
        result: concurrent.futures.Future[str] = concurrent.futures.Future()
        loop.call_soon_threadsafe(events.put_nowait, _ToolCall(tool.name, arguments, result))
        return await asyncio.wrap_future(result)

    attrs = {
        "name": tool.name,
        "description": tool.description or tool.name,
        "arguments_schema": property(lambda self: arguments_schema),
        "call": call,
    }
    return type(f"Proxy_{tool.name}", (fm.Tool,), attrs)()


def _split_history(messages: list[ModelMessage]) -> tuple[list[ModelMessage], list[Any], _Replay]:
    """Split messages into transcript history, the prompt items to respond to, and tool results to replay.

    Normally the last message holds the new user prompt. If it holds tool results that no live generation
    awaits (after a restart, or a deferred tool approved later), the turn is regenerated from its user prompt
    and repeated tool calls are answered from the recorded results.
    """
    if retries := [p for p in messages[-1].parts if isinstance(p, RetryPromptPart) and p.tool_name is None]:
        return messages[:-1], [p.model_response() for p in retries], _Replay([])  # e.g. output validation failed

    for i in reversed(range(len(messages))):
        if isinstance(messages[i], ModelRequest) and (
            prompts := [p for p in messages[i].parts if isinstance(p, UserPromptPart)]
        ):
            return messages[:i], [item for p in prompts for item in _user_items(p)], _recorded_tool_results(messages[i:])
    return messages, ["Continue, using the tool results above."], _Replay([])


def _recorded_tool_results(messages: list[ModelMessage]) -> _Replay:
    parts = [p for m in messages for p in m.parts]
    calls = {p.tool_call_id: p for p in parts if isinstance(p, ToolCallPart)}
    return _Replay(
        [
            (c.tool_name, _parse_json(c.args_as_json_str()), _tool_output(p))
            for p in parts
            if isinstance(p, ToolReturnPart) and (c := calls.get(p.tool_call_id))
        ]
    )


async def _transcript_dict(instructions: str | None, history: list[ModelMessage]) -> dict[str, Any]:
    """Message history in Apple's `FoundationModels.Transcript` JSON format."""

    async def contents(items: list[Any]) -> list[dict[str, Any]]:
        out = []
        for item in items:
            if isinstance(item, str):
                out.append({"type": "text", "id": _uid(), "text": item})
            else:  # the transcript only decodes JPEG data URIs
                jpeg = await asyncio.to_thread(_to_jpeg, *await _image_bytes(item))
                data = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
                out.append({"type": "attachment", "id": _uid(), "attachment": {"type": "image", "data": data}})
        return out

    entries = [{"role": "instructions", "id": _uid(), "contents": await contents([instructions])}] if instructions else []
    for message in history:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                match part:
                    case UserPromptPart():
                        entries.append({"role": "user", "id": _uid(), "options": {}, "contents": await contents(_user_items(part))})
                    case ToolReturnPart() | RetryPromptPart() if part.tool_name:
                        entries.append(
                            {
                                "role": "tool",
                                "id": part.tool_call_id,
                                "toolCallID": part.tool_call_id,
                                "toolName": part.tool_name,
                                "contents": await contents([_tool_output(part)]),
                            }
                        )
            continue
        if calls := [p for p in message.parts if isinstance(p, ToolCallPart)]:
            tool_calls = [{"id": c.tool_call_id, "name": c.tool_name, "arguments": c.args_as_json_str()} for c in calls]
            entries.append({"role": "response", "id": _uid(), "toolCalls": tool_calls})
        if text := "".join(p.content for p in message.parts if isinstance(p, TextPart)):
            entries.append({"role": "response", "id": _uid(), "assets": [], "contents": await contents([text])})
    return {"type": "FoundationModels.Transcript", "version": "1.1", "transcript": {"entries": entries}}


def _user_items(part: UserPromptPart) -> list[Any]:
    """The text and images of a user prompt; other content isn't supported."""
    if isinstance(part.content, str):
        return [part.content]
    return [c for c in part.content if isinstance(c, (str, ImageUrl)) or (isinstance(c, BinaryContent) and c.is_image)]


async def _prompt_components(items: list[Any], image_dir: Path) -> list[Any]:
    """Prompt items as Apple prompt components: text stays text, images become files on disk."""
    components = []
    for item in items:
        if isinstance(item, str):
            components.append(item)
            continue
        data, media_type = await _image_bytes(item)
        path = image_dir / f"{_uid()}{mimetypes.guess_extension(media_type) or '.img'}"
        path.write_bytes(data)
        components.append(fm.ImageAttachment(path))
    return components


async def _image_bytes(item: BinaryContent | ImageUrl) -> tuple[bytes, str]:
    if isinstance(item, BinaryContent):
        return item.data, item.media_type
    downloaded = await download_item(item, data_format="bytes")
    return downloaded["data"], downloaded["data_type"]


def _to_jpeg(data: bytes, media_type: str) -> bytes:
    if media_type == "image/jpeg":
        return data
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp, f"in{mimetypes.guess_extension(media_type) or '.img'}"), Path(tmp, "out.jpg")
        src.write_bytes(data)
        subprocess.run(["sips", "-s", "format", "jpeg", src, "--out", dst], check=True, capture_output=True)
        return dst.read_bytes()


def _uid() -> str:
    return str(uuid.uuid4())


def _tool_output(part: ToolReturnPart | RetryPromptPart) -> str:
    return part.model_response_str() if isinstance(part, ToolReturnPart) else part.model_response()


def _calls_text(calls: list[_ToolCall]) -> str:
    return "\n".join(f"{c.name}({c.arguments})" for c in calls)


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except ValueError:
        return value


def _generation_options(settings: ModelSettings | None) -> Optional[fm.GenerationOptions]:
    settings = settings or {}
    options = {"temperature": settings.get("temperature"), "maximum_response_tokens": settings.get("max_tokens")}
    return fm.GenerationOptions(**options) if any(v is not None for v in options.values()) else None


def _with_apple_object_keys(schema: Any, title: str = "Output") -> Any:
    """Add the `title`, `x-order` and `additionalProperties` keys Apple's decoder requires on object schemas."""
    if isinstance(schema, list):
        return [_with_apple_object_keys(s, title) for s in schema]
    if not isinstance(schema, dict):
        return schema
    schema = {k: _with_apple_object_keys(v, k) for k, v in schema.items()}
    if "properties" in schema:
        schema = {"title": title, "x-order": list(schema["properties"]), "additionalProperties": False, **schema}
    return schema


class _SchemaBuilder:
    """Translates a tool's JSON schema into an Apple `GenerationSchema`.

    Handles primitives, enums, arrays, optional fields (incl. `anyOf [X, null]`) and nested objects
    (incl. `$ref`/`$defs`). Apple requires every nested object schema on the root schema, so they're
    collected in `nested`; recursive models fall back to strings.
    """

    def __init__(self, tool: ToolDefinition):
        self.tool = tool
        self.defs: dict[str, Any] = tool.parameters_json_schema.get("$defs", {})
        self.nested: dict[str, fm.GenerationSchema] = {}
        self.building: set[str] = set()

    def build(self) -> fm.GenerationSchema:
        properties = self._properties(self.tool.parameters_json_schema, self.tool.name)
        return fm.GenerationSchema(
            type(f"{self.tool.name}_args", (), {}), self.tool.description, properties, list(self.nested.values())
        )

    def _properties(self, schema: dict[str, Any], path: str) -> list[Property]:
        required = set(schema.get("required", []))
        properties = []
        for name, prop in schema.get("properties", {}).items():
            py_type, guides = self._type(prop, f"{path}_{name}")
            description = prop.get("description") or self._resolve(prop).get("description")
            properties.append(Property(name, py_type if name in required else Optional[py_type], description, guides))
        return properties

    def _resolve(self, prop: dict[str, Any]) -> dict[str, Any]:
        while "$ref" in prop:
            prop = self.defs[prop["$ref"].rsplit("/", 1)[-1]]
        return prop

    def _type(self, prop: dict[str, Any], path: str) -> tuple[Any, list[fm.GenerationGuide]]:
        name = prop.get("$ref", path).rsplit("/", 1)[-1]
        prop = self._resolve(prop)
        if len(variants := [v for v in prop.get("anyOf", []) if v.get("type") != "null"]) == 1:
            return self._type(variants[0], path)
        if "enum" in prop:
            return str, [fm.GenerationGuide.anyOf([str(v) for v in prop["enum"]])]
        if prop.get("type") == "array":
            return list[self._type(prop.get("items", {}), f"{path}_item")[0]], []
        if prop.get("type") == "object" and "properties" in prop:
            return self._object(prop, name), []
        return _JSON_TYPES.get(prop.get("type", ""), str), []

    def _object(self, prop: dict[str, Any], name: str) -> Any:
        if name in self.building:
            return str
        if name not in self.nested:
            self.building.add(name)
            properties = self._properties(prop, name)  # registers child objects first
            self.building.discard(name)
            self.nested[name] = fm.GenerationSchema(
                type(name, (), {}), prop.get("description"), properties, list(self.nested.values())
            )
        return self.nested[name].type_class
