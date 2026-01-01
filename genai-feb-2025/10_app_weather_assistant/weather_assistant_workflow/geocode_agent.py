# create a free API key at https://geocode.maps.co/
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


@dataclass
class GeoCodeDeps:
    url: str
    client: Client
    geocode_api_key: str


class GeoCoordinates(BaseModel):
    lat: float
    lng: float


geo_code_agent = Agent(
    model=OpenAIModel(
        model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    ),
    system_prompt="You are helpful assistant. You must use the tool 'get_lat_lng' to find the longitude and latitude of any location.",
    deps_type=GeoCodeDeps,
    result_type=GeoCoordinates,
    retries=3,
)


@geo_code_agent.tool
def get_lat_lng(ctx: RunContext[GeoCodeDeps], location: str) -> str:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location: A description of a location.
    """
    try:
        r = ctx.deps.client.get(
            ctx.deps.url,
            params={
                "q": location,
                "api_key": ctx.deps.geocode_api_key,
            },
        )
        r.raise_for_status()
    except Exception as e:
        print(e)
        return {"lat": 17.385044, "lng": 78.486671}

    return r.json()


if __name__ == "__main__":
    load_dotenv(override=True)
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
    time.sleep(1)
    logfire.instrument_openai()
    logfire.instrument_httpx()

    deps = GeoCodeDeps(
        url="https://geocode.maps.co/search",
        client=Client(),
        geocode_api_key=os.getenv("GEO_API_KEY"),
    )
    result = geo_code_agent.run_sync("Find Hyderabad's current weather.", deps=deps)
    print(result.data)
