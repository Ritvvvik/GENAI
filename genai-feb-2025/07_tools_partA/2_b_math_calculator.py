from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire
import os

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

system_prompt = """
    You are helpful assistant.
    Use 'add' tool to add two integers.
    Use 'mul' tool to multiply two integers.
"""
agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)


@agent.tool_plain
def add(a: int, b: int) -> int:
    """Adds two numbers"""
    return a + b


@agent.tool_plain
def mul(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a * b


response = agent.run_sync("10 plus 20")
print(response.data)

response = agent.run_sync("10 * 20")
print(response.data)
