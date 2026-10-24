"""Weather agent as a local MCP server: when the on-device model needs your location, it asks via MCP elicitation.

On the classic protocol the question goes over the open connection while Apple's generation waits. On the stateless
protocol (2026-07-28) the call ends with `input_required` and the message history travels in the sealed request
state; the retry regenerates the turn and the bridge replays your answer when the model asks for the location again.
"""

from mcp import types
from mcp.server.mcpserver import AcceptedElicitation, Context, MCPServer
from pydantic import BaseModel
from pydantic_ai import AgentRunResult, CallDeferred, DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ModelMessage
from pydantic_ai.toolsets import FunctionToolset

from weather_agent import model, weather_agent

STATELESS_PROTOCOL = "2026-07-28"
QUESTION = "In which city are you?"

server = MCPServer("weather", instructions="Answers weather questions with Apple's on-device Foundation Model.")


class Location(BaseModel):
    city: str


class PendingRun(BaseModel):
    """What the stateless protocol carries between rounds; the SDK seals it."""

    messages: list[ModelMessage]
    call_ids: list[str]


def declined(action: str) -> str:
    return f"The user did not share their location ({action})."


@server.tool()
async def ask_weather(question: str, ctx: Context) -> str | types.InputRequiredResult:
    """Ask the on-device weather agent a question. For questions like "weather where I am" it asks for your city."""
    if ctx.protocol_version == STATELESS_PROTOCOL:
        return await ask_stateless(question, ctx)
    return await ask_connected(question, ctx)


async def ask_connected(question: str, ctx: Context) -> str:
    async def get_user_location() -> str:
        """Get the city the user is in right now."""
        reply = await ctx.elicit(QUESTION, Location)
        return reply.data.city if isinstance(reply, AcceptedElicitation) else declined(reply.action)

    result = await weather_agent.run(question, toolsets=[FunctionToolset([get_user_location])])
    return result.output


async def get_user_location() -> str:
    """Get the city the user is in right now."""
    raise CallDeferred


async def ask_stateless(question: str, ctx: Context) -> str | types.InputRequiredResult:
    pending = PendingRun.model_validate_json(ctx.request_state) if ctx.request_state else None
    result: AgentRunResult[str | DeferredToolRequests] = await weather_agent.run(
        None if pending else question,
        output_type=[str, DeferredToolRequests],
        message_history=pending.messages if pending else None,
        deferred_tool_results=answers(pending, ctx) if pending else None,
        toolsets=[FunctionToolset([get_user_location])],
    )
    if isinstance(result.output, str):
        return result.output
    call_ids = [call.tool_call_id for call in result.output.calls]
    model.abandon(call_ids)  # the turn now lives in the request state, not in this process
    form = types.ElicitRequestFormParams(message=QUESTION, requested_schema=Location.model_json_schema())
    return types.InputRequiredResult(
        input_requests={"location": types.ElicitRequest(params=form)},
        request_state=PendingRun(messages=result.all_messages(), call_ids=call_ids).model_dump_json(),
    )


def answers(pending: PendingRun, ctx: Context) -> DeferredToolResults:
    if ctx.input_responses is None or not isinstance(reply := ctx.input_responses["location"], types.ElicitResult):
        raise ValueError("The retry carries no answer to the location question")
    if reply.action != "accept":
        return DeferredToolResults(calls=dict.fromkeys(pending.call_ids, declined(reply.action)))
    if reply.content is None:
        raise ValueError("The accepted location answer has no content")
    return DeferredToolResults(calls=dict.fromkeys(pending.call_ids, str(reply.content["city"])))


if __name__ == "__main__":
    server.run()
