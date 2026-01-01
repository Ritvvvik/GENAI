from typing import Any
from pydantic_ai import Agent, ModelRetry, RunContext
from httpx import Client
import os
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass
from pydantic_ai.models.groq import GroqModel

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


@dataclass
class GeoCodeDeps:
    url: str
    client: Client
    api_key: str


system_prompt = """
    Be concise, reply with one sentence.
    Use the `get_lat_lng` tool to get the latitude and longitude of a location.
"""
agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
    deps_type=GeoCodeDeps,
)


@agent.tool
def get_lat_lng(ctx: RunContext[GeoCodeDeps], location: str) -> str:
    """Get the latitude and longitude of a location.

    Args:
        location: A description of a location.
    """
    r = ctx.deps.client.get(
        ctx.deps.url,
        params={
            "q": location,
            "api_key": ctx.deps.api_key,
        },
    )
    return r.json()


deps = GeoCodeDeps(
    url="https://geocode.maps.co/search",
    client=Client(),
    api_key=os.getenv("GEO_API_KEY"),
)
result = agent.run_sync("Find the longitude and lattitude of Hyderabad", deps=deps)
print(result.data)
