from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

agent = Agent(
    model=GeminiModel(
        model_name=os.getenv("GEMINI_CHAT_MODEL"), api_key=os.getenv("GEMINI_API_KEY")
    ),
    system_prompt="You are a helpful assistant.",
)

with logfire.span("Calling Gemini model") as span:
    response = agent.run_sync("Write a haiku about recursion in programming.")
    print(response.data)
    response = agent.run_sync("What is recursion in programming.")
    print(response.data)
