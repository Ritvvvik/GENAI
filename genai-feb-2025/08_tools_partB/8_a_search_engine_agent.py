from datetime import datetime
from typing import Any, List
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
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

search_agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)

response = search_agent.run_sync("What is the most recent national holiday in india?")
# response = search_agent.run_sync("What are the latest AI news announcements?")
print(response.data)
