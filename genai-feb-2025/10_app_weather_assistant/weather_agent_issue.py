from pydantic_ai import Agent
import os
from dotenv import load_dotenv
import logfire
from pydantic_ai.models.groq import GroqModel

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


system_prompt = """
    You are a helpful assistant.
"""
agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)

# result = agent.run_sync("Find Berlin's current weather.")
result = agent.run_sync("Find Hyderabad's current weather.")
print(result.data)
