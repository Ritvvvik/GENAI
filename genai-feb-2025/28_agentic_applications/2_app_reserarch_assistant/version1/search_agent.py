import asyncio
from pydantic_ai import Agent
from llm import build_model
from pydantic import BaseModel
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool


system_prompt = """
You are a research assistant. Given a search term, you search the web for that term and
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300
words. Capture the main points. Write succinctly, no need to have complete sentences or good
grammar. This will be consumed by someone synthesizing a report, so its vital you capture the
essence and ignore any fluff. Do not include any additional commentary other than the summary
itself.
"""

search_agent = Agent(
    name="SearchAgent",
    model=build_model(),
    system_prompt=system_prompt,
    tools=[duckduckgo_search_tool(max_results=2)],
    retries=3,
)


async def main(query: str):
    result = await search_agent.run(query)
    print(result.data)


if __name__ == "__main__":
    query = "surfing"
    asyncio.run(main(query))
