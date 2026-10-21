"""Weather agent: Pydantic AI with tools, on Apple's on-device Foundation Model."""

from typing import TypedDict

from pydantic_ai import Agent

from apple_fm_model import AppleFoundationModel

weather_agent = Agent(
    AppleFoundationModel(),
    instructions='Use the tools to answer. Look up each location separately.',
)


@weather_agent.tool_plain
def get_lat_long(location: str) -> tuple[float, float]:
    """Get the coordinates of a location.

    Returns: A tuple of `(latitude, longitude)`.
    """
    return 0.0, 0.0


class Weather(TypedDict):
    temp_f: int
    condition: str


@weather_agent.tool_plain
def get_weather(lat: float, lon: float) -> Weather:
    """Get current weather at a location."""
    return Weather(temp_f=72, condition='sunny')


@weather_agent.tool_plain
def convert_temp(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return round((fahrenheit - 32) * 5 / 9, 1)


if __name__ == '__main__':
    result = weather_agent.run_sync("What's the weather in Paris and London, in Celsius?")
    print(result.output)
