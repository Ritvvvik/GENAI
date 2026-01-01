import asyncio
from pydantic_ai import Agent
from llm import build_model
from pydantic import BaseModel


class WebSearchItem(BaseModel):
    reason: str
    "Your reasoning for why this search is important to the query."

    query: str
    "The search term to use for the web search."


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """A list of web searches to perform to best answer the query."""


system_prompt = """
You are a helpful research assistant. Given a query, come up with a set of web searches
to perform to best answer the query. Output between 2 and 3 terms to query for.
"""

planner_agent = Agent(
    name="PlannerAgent",
    model=build_model(),
    system_prompt=system_prompt,
    result_type=WebSearchPlan,
    retries=3,
)


async def main(query: str):
    result = await planner_agent.run(query)
    print(result.data)


if __name__ == "__main__":
    query = "Caribbean vacation spots in April, optimizing for surfing, hiking and water sports"
    asyncio.run(main(query))
