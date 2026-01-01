from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

load_dotenv(override=True)

agent = Agent(
    model=OpenAIModel(
        model_name=os.getenv("DEEPSEEK_CHAT_MODEL"),
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    ),
    system_prompt="You are a helpful assistant.",
)

response = agent.run_sync("Write a haiku about recursion in programming.")
print(response.data)

response = agent.run_sync("What is recursion in programming.")
print(response.data)
