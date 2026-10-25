"""Evaluation of the Pydantic AI bridge on Apple's on-device Foundation Model.

Every scenario runs several times through Pydantic Evals. Deterministic checks grade each trial's trajectory (tool
calls, questions to the user, rounds, output); outcomes, transcripts and OpenTelemetry spans go to `results.json`.

    uv run --group evals python -m evals.run --repeat 10
"""

import argparse
import gc
import importlib.metadata
import json
import platform
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from operator import attrgetter
from pathlib import Path
from typing import Any, Literal

from mcp import types
from mcp.client import Client, ClientRequestContext
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.toolsets import FunctionToolset
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport, ReportCase, ReportCaseFailure

from weather_agent import model, weather_agent
from weather_mcp import QUESTION, server

HERE = Path(__file__).parent
SPANS = InMemorySpanExporter()
KEPT_ATTRIBUTES = (
    "gen_ai.tool.name",
    "gen_ai.tool.call.arguments",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
)


@dataclass
class Run:
    """What a scenario hands back besides the spans: the answer and what the user saw."""

    output: str
    asked: list[str] = field(default_factory=list)
    deltas: int = 0


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    replayed: bool


@dataclass
class Trial:
    """One run of a scenario, as graded and as shown on the results page."""

    output: str
    asked: list[str]
    deltas: int
    tool_calls: list[ToolCall]
    agent_runs: int
    input_tokens: int
    output_tokens: int
    transcript: list[Any]
    spans: list[dict[str, Any]]


Check = Callable[[Trial], bool]


@dataclass
class Scenario:
    title: str
    description: str
    prompt: str
    run: Callable[[str], Awaitable[Run]]
    checks: dict[str, Check]  # what the bridge must deliver; a trial passes when all hold
    findings: dict[str, Check] = field(default_factory=dict)  # how the model behaves, reported but not graded


def attributes(span: ReadableSpan) -> Mapping[str, Any]:
    if span.attributes is None:
        raise ValueError(f"Span {span.name!r} has no attributes")
    return span.attributes


def attribute(span: ReadableSpan, key: str) -> Any:
    return attributes(span)[key]


def span_record(span: ReadableSpan, started: int) -> dict[str, Any]:
    if span.start_time is None or span.end_time is None:
        raise ValueError(f"Span {span.name!r} has not ended")
    return {
        "name": span.name,
        "id": f"{span.context.span_id:016x}",
        "parent": f"{span.parent.span_id:016x}" if span.parent else None,
        "start_ms": round((span.start_time - started) / 1e6, 1),
        "end_ms": round((span.end_time - started) / 1e6, 1),
        "attributes": {k: v for k, v in attributes(span).items() if k in KEPT_ATTRIBUTES},
    }


def observe(started: int, run: Run) -> Trial:
    """Combine a scenario's answer with what its spans recorded: tool calls, rounds, tokens and transcript."""
    spans = sorted(SPANS.get_finished_spans(), key=attrgetter("start_time"))
    agents = [s for s in spans if s.name.startswith("invoke_agent")]
    chats = [s for s in spans if s.name.startswith("chat")]
    tools = [s for s in spans if s.name.startswith(("execute_tool", "replay_tool"))]
    return Trial(
        output=run.output,
        asked=run.asked,
        deltas=run.deltas,
        tool_calls=[
            ToolCall(
                attribute(s, "gen_ai.tool.name"),
                json.loads(attribute(s, "gen_ai.tool.call.arguments")),
                s.name.startswith("replay_tool"),
            )
            for s in tools
        ],
        agent_runs=len(agents),
        input_tokens=sum(int(attribute(s, "gen_ai.usage.input_tokens")) for s in chats),
        output_tokens=sum(int(attribute(s, "gen_ai.usage.output_tokens")) for s in chats),
        transcript=json.loads(attribute(agents[-1], "pydantic_ai.all_messages")) if agents else [],
        spans=[span_record(s, started) for s in spans],
    )


# --- checks -------------------------------------------------------------------------------------------------------


def calls(tool: str, **arguments: str) -> Check:
    def check(trial: Trial) -> bool:
        return any(
            c.name == tool
            and all(k in c.arguments and v.lower() in str(c.arguments[k]).lower() for k, v in arguments.items())
            for c in trial.tool_calls
        )

    return check


def does_not_call(tool: str) -> Check:
    def check(trial: Trial) -> bool:
        return all(c.name != tool for c in trial.tool_calls)

    return check


def replays(tool: str) -> Check:
    def check(trial: Trial) -> bool:
        return any(c.name == tool and c.replayed for c in trial.tool_calls)

    return check


def says(text: str) -> Check:
    def check(trial: Trial) -> bool:
        return text.lower() in trial.output.lower()

    return check


def asks(times: int) -> Check:
    def check(trial: Trial) -> bool:
        return trial.asked == [QUESTION] * times

    return check


def rounds(count: int) -> Check:
    def check(trial: Trial) -> bool:
        return trial.agent_runs == count

    return check


def overrides_looked_up_coordinates(trial: Trial) -> bool:
    """`get_lat_long` always answers (0.0, 0.0); a weather lookup with other coordinates ignores that answer."""
    return any(c.name == "get_weather" and c.arguments != {"lat": 0, "lon": 0} for c in trial.tool_calls)


def places_kampen_in_oslo(trial: Trial) -> bool:
    return "oslo" in trial.output.lower()


def streams_in_pieces(trial: Trial) -> bool:
    return trial.deltas > 1


def asks_in_its_answer(trial: Trial) -> bool:
    return trial.output.rstrip().endswith("?")


def city_is(city: str) -> Check:
    def check(trial: Trial) -> bool:
        return bool(json.loads(trial.output)["city"] == city)

    return check


@dataclass
class Checks(Evaluator[str, Trial, dict[str, Any]]):
    """Runs a scenario's checks and findings; `outcome` tells them apart again by name."""

    checks: dict[str, Check]

    def evaluate(self, ctx: EvaluatorContext[str, Trial, dict[str, Any]]) -> dict[str, bool]:
        return {name: check(ctx.output) for name, check in self.checks.items()}


# --- scenarios ----------------------------------------------------------------------------------------------------


class Plan(BaseModel):
    city: str
    needs_tent: bool


async def weather(prompt: str) -> Run:
    return Run((await weather_agent.run(prompt)).output)


async def structured(prompt: str) -> Run:
    result = await Agent(model, output_type=Plan).run(prompt)
    return Run(result.output.model_dump_json())


async def streaming(prompt: str) -> Run:
    async with Agent(model).run_stream(prompt) as result:
        deltas = [delta async for delta in result.stream_text(delta=True)]
    return Run("".join(deltas), deltas=len(deltas))


async def image(prompt: str) -> Run:
    picture = BinaryContent((HERE / "kampen-40.jpg").read_bytes(), media_type="image/jpeg")
    return Run((await Agent(model).run([prompt, picture])).output)


def over_mcp(mode: Literal["legacy", "auto"]) -> Callable[[str], Awaitable[Run]]:
    async def run(prompt: str) -> Run:
        asked: list[str] = []

        async def reply(context: ClientRequestContext, params: types.ElicitRequestParams) -> types.ElicitResult:
            asked.append(params.message)
            return types.ElicitResult(action="accept", content={"city": "Kampen"})

        async with Client(server, mode=mode, elicitation_callback=reply) as client:
            result = await client.call_tool("ask_weather", {"question": prompt})
        text = "".join(c.text for c in result.content if isinstance(c, types.TextContent))
        if result.is_error:
            raise RuntimeError(text)
        return Run(text, asked)

    return run


async def ask_user(question: str) -> str:
    """Ask the user a question and get their answer. The only way to get information from the user."""
    return "Celsius"


async def generic_question_tool(prompt: str) -> Run:
    return Run((await weather_agent.run(prompt, toolsets=[FunctionToolset([ask_user])])).output)


gc_agent = Agent(model, instructions="Use the tools to answer. Look up each location separately.")


@gc_agent.tool_plain
def get_lat_long(location: str) -> tuple[float, float]:
    """Get the coordinates of a location."""
    gc.collect()
    return 52.56, 5.91


@gc_agent.tool_plain
def get_weather(lat: float, lon: float) -> str:
    """Get current weather at a location."""
    gc.collect()
    return "sunny, 22 degrees Celsius"


async def collected_mid_generation(prompt: str) -> Run:
    return Run((await gc_agent.run(prompt)).output)


SCENARIOS: dict[str, Scenario] = {
    "tools": Scenario(
        "Tools",
        "The weather agent looks up two cities and converts to Celsius.",
        "What's the weather in Paris and London, in Celsius?",
        weather,
        {
            "looks up Paris": calls("get_lat_long", location="paris"),
            "looks up London": calls("get_lat_long", location="london"),
            "gets the weather": calls("get_weather"),
            "converts to Celsius": calls("convert_temp"),
            "reports 22.2 °C": says("22.2"),
        },
        {"overrides the coordinates from get_lat_long": overrides_looked_up_coordinates},
    ),
    "structured": Scenario(
        "Structured output",
        "A Pydantic model as output type, sent to Apple as a JSON schema.",
        "Party in Kampen in January. Plan it.",
        structured,
        {"city is Kampen": city_is("Kampen")},
    ),
    "streaming": Scenario(
        "Streaming",
        "Apple streams snapshots; the bridge turns them into deltas.",
        "Write three sentences about Kampen.",
        streaming,
        {"streams in pieces": streams_in_pieces, "mentions Kampen": says("Kampen")},
        {"takes Kampen to be the district in Oslo": places_kampen_in_oslo},
    ),
    "image": Scenario(
        "Image input",
        "The share image of the party site, sent as BinaryContent.",
        "What number is shown in large neon digits on this image?",
        image,
        {"reads 40": says("40")},
    ),
    "elicitation-classic": Scenario(
        "Elicitation, classic MCP",
        "Over MCP the model asks for your city while Apple's generation waits.",
        "What's the weather where I am?",
        over_mcp("legacy"),
        {"asks for the city once": asks(1), "answers for Kampen": says("Kampen"), "one round": rounds(1)},
    ),
    "elicitation-stateless": Scenario(
        "Elicitation, stateless MCP",
        "The call ends with input_required; the retry regenerates the turn and replays your answer.",
        "What's the weather where I am?",
        over_mcp("auto"),
        {
            "asks for the city once": asks(1),
            "answers for Kampen": says("Kampen"),
            "two rounds": rounds(2),
            "replays the answer": replays("get_user_location"),
        },
    ),
    "no-question": Scenario(
        "No question needed",
        "With the city in the question, the model should not ask.",
        "What's the weather in Paris?",
        over_mcp("auto"),
        {
            "asks nothing": asks(0),
            "one round": rounds(1),
            "does not ask for a location": does_not_call("get_user_location"),
        },
    ),
    "generic-question-tool": Scenario(
        "A generic ask_user tool",
        "Does the model use a tool to ask the user, or does it ask in its answer?",
        "What's the weather in Paris? Ask me which temperature unit I want first.",
        generic_question_tool,
        {},
        {"skips the ask_user tool": does_not_call("ask_user"), "asks its question in the answer": asks_in_its_answer},
    ),
    "garbage-collection": Scenario(
        "Garbage collection mid-generation",
        "Tools call gc.collect() while Apple's generation waits; this crashed before the proxy tools were kept alive.",
        "What's the weather in Kampen?",
        collected_mid_generation,
        {"looks up Kampen": calls("get_lat_long", location="kampen"), "gets the weather": calls("get_weather")},
    ),
}


async def trial(name: str) -> Trial:
    SPANS.clear()
    started = time.time_ns()
    scenario = SCENARIOS[name]
    return observe(started, await scenario.run(scenario.prompt))


# --- results ------------------------------------------------------------------------------------------------------


def outcome(case: ReportCase[str, Trial, dict[str, Any]]) -> dict[str, Any]:
    scenario = SCENARIOS[case.inputs]
    values = {name: bool(result.value) for name, result in case.assertions.items()}
    checks = {name: values[name] for name in scenario.checks}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "findings": {name: values[name] for name in scenario.findings},
        "duration_s": case.task_duration,
        **asdict(case.output),
    }


def failure(case: ReportCaseFailure[str, Trial, dict[str, Any]]) -> dict[str, Any]:
    return {"passed": False, "checks": {}, "findings": {}, "error": case.error_message}


def results(report: EvaluationReport[str, Trial, dict[str, Any]], names: Sequence[str], repeat: int) -> dict[str, Any]:
    trials: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for case in report.cases:
        trials[case.inputs].append(outcome(case))
    for failed in report.failures:
        trials[failed.inputs].append(failure(failed))
    packages = ("pydantic-ai-slim", "pydantic-evals", "mcp", "apple-fm-sdk")
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in packages},
            "model": "SystemLanguageModel (Apple Foundation Model, on device)",
        },
        "repeat": repeat,
        "scenarios": [
            {
                "id": name,
                "title": SCENARIOS[name].title,
                "description": SCENARIOS[name].description,
                "prompt": SCENARIOS[name].prompt,
                "checks": list(SCENARIOS[name].checks),
                "findings": list(SCENARIOS[name].findings),
                "trials": trials[name],
            }
            for name in names
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=10, help="trials per scenario")
    parser.add_argument("--only", nargs="+", choices=list(SCENARIOS), help="run only these scenarios")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    args = parser.parse_args()

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(SPANS))
    trace.set_tracer_provider(provider)
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))

    names: list[str] = list(SCENARIOS) if args.only is None else args.only
    dataset = Dataset[str, Trial, dict[str, Any]](
        name="fm-pydantic-ai",
        cases=[
            Case(
                name=n,
                inputs=n,
                metadata={"title": SCENARIOS[n].title},
                evaluators=(Checks(SCENARIOS[n].checks | SCENARIOS[n].findings),),
            )
            for n in names
        ],
    )
    report = dataset.evaluate_sync(trial, max_concurrency=1, repeat=args.repeat, progress=False)
    summary = results(report, names, args.repeat)
    args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    for scenario in summary["scenarios"]:
        trials = scenario["trials"]
        print(f"{scenario['title']:<40} {sum(t['passed'] for t in trials)}/{len(trials)} passed")
        for name in scenario["findings"]:
            print(f"  finding: {name:<46} {sum(t['findings'].get(name, False) for t in trials)}/{len(trials)}")


if __name__ == "__main__":
    main()
