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
"""

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)


# response = agent.run_sync("is 13 prime number?")
# response = agent.run_sync("is 16 prime number?")
response = agent.run_sync("is 27 prime number?")
print(response.data)
