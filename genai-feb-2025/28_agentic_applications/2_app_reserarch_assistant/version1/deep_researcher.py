from config_reader import *
from planner_agent import *
from search_agent import *
from report_agent import *
import asyncio
import logfire
import time


class DeepReseacher:
    def __init__(self):
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    async def plan_searches(self, query: str) -> WebSearchPlan:
        logfire.info("Planning searches...")
        search_plan = await planner_agent.run(query)
        logfire.info(f"Will perform {len(search_plan.data.searches)} searches...")
        return search_plan.data

    async def search(self, item: WebSearchItem) -> str | None:
        query = f"Search term: {item.query}\nReason for searching: {item.reason}"
        logfire.info(query)
        try:
            result = await search_agent.run(query)
            return result.data
        except Exception:
            return None

    async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        logfire.info("Searching Web...")
        num_completed = 0
        tasks = [
            asyncio.create_task(self.search(item)) for item in search_plan.searches
        ]
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            num_completed += 1
            logfire.info(f"Searching... {num_completed}/{len(tasks)} completed")
        return results

    async def generate_report(
        self, query: str, search_results: list[str]
    ) -> ReportData:
        logfire.info("Generating Report...")
        query = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await report_agent.run(query)
        return result.data

    async def do_research(self, query: str) -> tuple:
        logfire.info(f"Starting deep research for query:{query}")
        search_plan = await self.plan_searches(query)
        search_results = await self.perform_searches(search_plan)
        report = await self.generate_report(query, search_results)
        return (
            report.short_summary,
            report.markdown_report,
            # "\n".join(report.follow_up_questions),
        )


async def main(query: str):
    dr = DeepReseacher()
    result = await dr.do_research(query)
    print(result)


if __name__ == "__main__":
    query = "Caribbean vacation spots in April, optimizing for surfing, hiking and water sports"
    asyncio.run(main(query))
