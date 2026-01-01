# pip install whatismyip
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire
import os
import whatismyip

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

system_prompt = """
    You are helpful assistant. 
    Pick the right tool for the user request.
"""
agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)


@agent.tool_plain
def who_am_i() -> str:
    """Find the ip address of the host"""
    return whatismyip.whatismyip()


response = agent.run_sync("what is the ip address of my machine")
print(response.data)
