from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire
import os
from collections import deque
from pydantic_ai.messages import ModelMessage
import time

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(1)
logfire.instrument_openai()

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt="You are helpful assistant.",
)


response = agent.run_sync("tell me a joke")
print(response.data)

response = agent.run_sync(
    "explanin?",
    model=os.getenv("GROQ_REASONER_MODEL"),
    message_history=response.all_messages(),
)
print(response.data)
