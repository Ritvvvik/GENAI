# create a free API key at https://geocode.maps.co/
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


system_prompt = """
    You are helpful assistant.
    Use the tool `get_lat_lng` to get the latitude and longitude of a location.
"""
agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)


@agent.tool_plain
def get_lat_lng(location: str) -> str:
    """Get the latitude and longitude of a location.

    Args:
        location: A description of a location.
    """
    client = Client()
    r = client.get(
        "https://geocode.maps.co/search",
        params={
            "q": location,
            "api_key": os.getenv("GEO_API_KEY"),
        },
    )
    return r.json()


result = agent.run_sync("Find the longitude and lattitude of Hyderabad")
print(result.data)
