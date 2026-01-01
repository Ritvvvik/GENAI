# docker run -i --rm -v C:/Users/pc/Documents/genai-training-pydanticai/data/text-to-sql-qa-bot/db:/mcp mcp/sqlite --db-path /mcp/chinook.sqlite

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
import os
import time
from dotenv import load_dotenv
import logfire
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
        """List the tools available on the server.

        This method overrides the default behavior to filter tools based on the context variable.
        """
        tools = await super().list_tools()
        filter_value = ["list_tables", "describe_table"]
        return [t for t in tools if t.name in filter_value]


sqlite_server = FilteringMCPServer(
    command="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "-v",
        "C:/Users/pc/Documents/genai-training-pydanticai/data/text-to-sql-qa-bot/db:/mcp",
        "mcp/sqlite",
        "--db-path",
        "/mcp/chinook.sqlite",
    ],
)
agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_REASONING_MODEL"),
        provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")),
    ),
    instructions="Use the tools to answer questions based on the database.",
    mcp_servers=[sqlite_server],
)


async def main():
    queries = [
        "list the table names",
        "get the schema of table album",
        "list the columns of table artist",
    ]
    history = []
    async with agent.run_mcp_servers():
        for query in queries:
            result = await agent.run(query, conversation_history=history)
            history = result.new_messages()
            print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
