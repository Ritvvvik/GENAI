# pip install pydanticai
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

agent = Agent(
    model=OpenAIModel(
        model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    ),
    system_prompt="You are a helpful assistant.",
)

response = agent.run_sync("What is recursion in programming.")
print(response.data)

response = agent.run_sync("Write a haiku about recursion in programming.")
print(response.data)

print(response.new_messages)
print()
print(response.all_messages)
print()
print(response.usage)
