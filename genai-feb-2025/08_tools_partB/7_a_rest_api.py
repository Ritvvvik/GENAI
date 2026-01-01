from typing import Any
from pydantic_ai import Agent, ModelRetry, RunContext
from httpx import Client
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
"""

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)

result = agent.run_sync("what are the longitude and lattitude of Hyderabad")
# result = agent.run_sync("what are the longitude and lattitude of Warangal")
print(result.data)
