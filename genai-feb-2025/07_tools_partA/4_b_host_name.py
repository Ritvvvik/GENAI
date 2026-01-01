from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire
import os
import subprocess

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

system_prompt = """
    You are helpful assistant.
    Use 'who_am_i' tool to find the active user of my system.
"""

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)


@agent.tool_plain
def who_am_i() -> str:
    """Find the current user of system"""
    result = subprocess.run("whoami", capture_output=True, text=True)
    return result.stdout


response = agent.run_sync("who is the active user of my system?")
print(response.data)
