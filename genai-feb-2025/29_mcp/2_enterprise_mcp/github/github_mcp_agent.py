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
from pydantic_ai.mcp import ToolDefinition

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(1)
logfire.instrument_pydantic_ai()
logfire.instrument_openai()


class FilteringMCPServer(MCPServerStdio):
    async def list_tools(self) -> list[ToolDefinition]:
        tools = await super().list_tools()
        filter_value = ["get_me", "list_commits", "get_commit"]
        return [t for t in tools if t.name in filter_value]


github_server = FilteringMCPServer(
    command="docker",
    args=[
        "run",
        "--rm",
        "-i",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server",
    ],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")},
)

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_REASONING_MODEL"),
        provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")),
    ),
    mcp_servers=[github_server],
    instructions="You are an AI agent to answer queries and perform operations on the github repository using tools.",
)


async def main():
    queries = [
        "get the owner of the repository",
        "get the summary of the commit with sha *bd0fe8ab5b486a717da35ce25e337d5832f6327f* of repo name *genai-feb-2025*",
    ]
    history = []
    async with agent.run_mcp_servers():
        for query in queries:
            result = await agent.run(query, message_history=history)
            history = result.all_messages()
            print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
