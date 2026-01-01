import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import logfire
import os
import time

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_pydantic_ai()

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"),
        provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")),
    ),
    system_prompt="You are a helpful AI assistant",
)


async def approach2() -> None:
    for _ in range(10):
        res = await agent.run("what is the capital of india")
        print(res.data)


start_time = time.perf_counter()
asyncio.run(approach2())
end_time = time.perf_counter()
print(f"Approach2: {end_time - start_time:.2f} seconds")
