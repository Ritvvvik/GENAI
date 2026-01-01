from pathlib import Path
from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.messages import ImageUrl
import os
from dotenv import load_dotenv
from textwrap import dedent
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

system_prompt = """
    You are smart vision agent with text extraction skills
"""

agent = Agent(
    model=GeminiModel(
        model_name=os.getenv("GEMINI_CHAT_MODEL"), api_key=os.getenv("GEMINI_API_KEY")
    ),
    system_prompt=dedent(system_prompt),
)

response = agent.run_sync(
    [
        "What's in the image?",
        ImageUrl(url="https://iili.io/3Hs4FMg.png"),
    ]
)
print(response.data)
