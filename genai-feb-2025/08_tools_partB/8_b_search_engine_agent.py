from typing import Any, List
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
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
    You must use the tool `get_search` for any query.
"""

search_agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_REASONING_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)


@search_agent.tool_plain
def get_search(query: str) -> dict[str, Any]:
    """Get the search for a keyword query.

    Args:
        query: keywords to search.
    """
    print(f"Search query: {query}")
    results = DDGS(proxy=None).text(query, max_results=3)
    return results


# response = search_agent.run_sync("Is 26th feb 2025 holiday in india?")
response = search_agent.run_sync("What are the latest AI news announcements?")
print(response.data)
