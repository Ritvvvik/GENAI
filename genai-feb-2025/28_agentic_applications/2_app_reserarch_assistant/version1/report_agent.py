import asyncio
from pydantic_ai import Agent
from llm import build_model
from pydantic import BaseModel
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool


class ReportData(BaseModel):
    short_summary: str
    """A short 2-3 sentence summary of the findings."""

    markdown_report: str
    """The final report"""

    # follow_up_questions: list[str]
    """Suggested topics to research further"""


system_prompt = """
You are a senior researcher tasked with writing a cohesive report for a research query.
You will be provided with the original query, and some initial research done by a research assistant.
You should first come up with an outline for the report that describes the structure and
flow of the report. Then, generate the report and return that as your final output.
The final output should be in markdown format, and it should be lengthy and detailed. 
Aim for 2-5 pages of content, at least 1000 words.
"""

report_agent = Agent(
    name="ReportAgent",
    model=build_model(),
    system_prompt=system_prompt,
    result_type=ReportData,
    retries=3,
)


async def main(query: str):
    result = await report_agent.run(query)
    print(result.data)


if __name__ == "__main__":
    query = "surfing"
    asyncio.run(main(query))
