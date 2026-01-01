from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire
import os
import random

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

system_prompt = """
    You are helpful assistant. 
    Use the tool for the job.
"""


agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)


@agent.tool_plain
def toss_coin() -> int:
    """Toss a coin and return the result."""
    return random.randint(1, 2)


response = agent.run_sync("toss a coin and tell me the outcome")
print(response.data)
