from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
import os
import time
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import asyncio
from pydantic_ai.mcp import MCPServer
from pydantic_ai.mcp import ToolDefinition

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(1)
logfire.instrument_pydantic_ai()
logfire.instrument_openai()


class FilteringMCPServer(MCPServerStdio):
    async def list_tools(self) -> list[ToolDefinition]:
        tools = await super().list_tools()
        filter_value = ["browser_navigate", "browser_close"]
        return [t for t in tools if t.name in filter_value]


playwright_server = FilteringMCPServer(
    command="npx",
    args=["@playwright/mcp@latest"],
)

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_REASONING_MODEL"),
        provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")),
    ),
    mcp_servers=[playwright_server],
    instructions="You are an AI agent to perform browser based actions using available tools.",
)


async def main():
    queries = [
        # "Navigate to 'example.com' and click the hyperlink 'More information'",
        # "navigate to website https://example.com/article and extract all visible text from the page",
        # "Navigate to 'https://en.wikipedia.org/wiki/Internet' and scroll to the string 'The vast majority of computer'",
        "Search for news related to MCP on bing news. Open first link and summarize content",
        # """
        # Given I navigate to website http://eaapp.somee.com and click login link
        # And I enter username and password as "admin" and "password" respectively and perform login
        # Then click the Employee List page
        # And click "Create New" button and enter realistic employee details to create for Name, Salary, DurationWorked,
        # Select dropdown for Grade as CLevel and Email.
        # Close the browser.
        # """
        # """
        # navigate to website "https://example.com/report"
        # save the current page as a PDF in "/downloads" folder with name "report.pdf"
        # """
        # """
        # Go to opentable, and book resturant in Atlanta. Use my phone number 123–456–7890 for reservation
        # """
    ]
    history = []
    async with agent.run_mcp_servers():
        for query in queries:
            result = await agent.run(query, message_history=history)
            history = result.all_messages()
            print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
