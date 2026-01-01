from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel

load_dotenv(override=True)

agent = Agent(
    model=MistralModel(
        model_name=os.getenv("MISTRAL_CHAT_MODEL"), api_key=os.getenv("MISTRAL_API_KEY")
    ),
    system_prompt="You are a helpful assistant.",
)

response = agent.run_sync("Write a haiku about recursion in programming.")
print(response.data)

response = agent.run_sync("What is recursion in programming.")
print(response.data)
