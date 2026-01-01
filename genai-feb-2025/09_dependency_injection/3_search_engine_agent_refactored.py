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


@dataclass
class SearchDeps:
    user_name: str
    max_results: int


class SearchResult(BaseModel):
    title: str = Field(..., description="title of search result")
    href: str = Field(..., description="URL of search")
    body: str = Field(..., description="body of search result")


class SearchResults(BaseModel):
    results: List[SearchResult] = Field(..., description="search results")


system_prompt = """
    You are helpful assistant.
    You must use the tool `get_search` for any query.
"""

search_agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_REASONING_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
    deps_type=SearchDeps,
    # result_type=SearchResults,
)


@search_agent.system_prompt
def add_name(ctx: RunContext[SearchDeps]) -> str:
    return f"Always use the user's name in the response: {ctx.deps.user_name}"


@search_agent.tool
def get_search(ctx: RunContext[SearchDeps], query: str) -> dict[str, Any]:
    """Get the search for a keyword query.

    Args:
        query: keywords to search.
    """
    print(f"Search query: {query}")
    results = DDGS(proxy=None).text(query, max_results=ctx.deps.max_results)
    return results


deps = SearchDeps(user_name="Algorithmica", max_results=3)
# response = search_agent.run_sync("Is 26th feb 2025 holiday in india?", deps=deps)
response = search_agent.run_sync(
    "What are the latest AI news announcements?", deps=deps
)
print(response.data)
print(response.all_messages())
