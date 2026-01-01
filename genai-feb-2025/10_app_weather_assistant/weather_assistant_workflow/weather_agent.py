# create a free API key at https://www.tomorrow.io/weather-api/
import time
from typing import Any
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from httpx import Client
import os
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass
from pydantic_ai.models.openai import OpenAIModel
from geocode_agent import GeoCoordinates


@dataclass
class WeatherDeps:
    url: str
    client: Client
    weather_api_key: str


class WeatherInfo(BaseModel):
    temperature: float
    description: str


weather_agent = Agent(
    model=OpenAIModel(
        model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    ),
    system_prompt="You are helpful assistant. You must use the tool 'get_weather' to find the weather at given coordinates.",
    deps_type=WeatherDeps,
    result_type=WeatherInfo,
    retries=3,
)


@weather_agent.tool
def get_weather(ctx: RunContext[WeatherDeps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    r = ctx.deps.client.get(
        ctx.deps.url,
        params={
            "apikey": ctx.deps.weather_api_key,
            "location": f"{lat},{lng}",
            "units": "metric",
        },
    )
    r.raise_for_status()
    data = r.json()
    values = data["data"]["values"]

    code_lookup = {
        1000: "Clear, Sunny",
        1100: "Mostly Clear",
        1101: "Partly Cloudy",
        1102: "Mostly Cloudy",
        1001: "Cloudy",
        2000: "Fog",
        2100: "Light Fog",
        4000: "Drizzle",
        4001: "Rain",
        4200: "Light Rain",
        4201: "Heavy Rain",
        5000: "Snow",
        5001: "Flurries",
        5100: "Light Snow",
        5101: "Heavy Snow",
        6000: "Freezing Drizzle",
        6001: "Freezing Rain",
        6200: "Light Freezing Rain",
        6201: "Heavy Freezing Rain",
        7000: "Ice Pellets",
        7101: "Heavy Ice Pellets",
        7102: "Light Ice Pellets",
        8000: "Thunderstorm",
    }
    return {
        "temperature": f'{values["temperatureApparent"]:0.0f}°C',
        "description": code_lookup.get(values["weatherCode"], "Unknown"),
    }


if __name__ == "__main__":
    load_dotenv(override=True)
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
    time.sleep(1)
    logfire.instrument_openai()
    logfire.instrument_httpx()

    deps = WeatherDeps(
        url="https://api.tomorrow.io/v4/weather/realtime",
        client=Client(),
        weather_api_key=os.getenv("WEATHER_API_KEY"),
    )
    coordinates = GeoCoordinates(lat=17.385044, lng=78.486671)
    result = weather_agent.run_sync(
        f"Find the weather at following coordinates: {coordinates.model_dump_json()}",
        deps=deps,
    )
    print(result.data)
