"""Weather agent as a local MCP server: when the on-device model needs your location, it asks via MCP elicitation."""

from mcp.server.mcpserver import AcceptedElicitation, Context, MCPServer
from pydantic import BaseModel
from pydantic_ai.toolsets import FunctionToolset

from weather_agent import weather_agent

server = MCPServer("weather", instructions="Answers weather questions with Apple's on-device Foundation Model.")


class Location(BaseModel):
    city: str


@server.tool()
async def ask_weather(question: str, ctx: Context) -> str:
    """Ask the on-device weather agent a question. For questions like "weather where I am" it asks for your city."""

    async def get_user_location() -> str:
        """Get the city the user is in right now."""
        reply = await ctx.elicit("In which city are you?", Location)
        if isinstance(reply, AcceptedElicitation):
            return reply.data.city
        return f"The user did not share their location ({reply.action})."

    result = await weather_agent.run(question, toolsets=[FunctionToolset([get_user_location])])
    return result.output


if __name__ == "__main__":
    server.run()
