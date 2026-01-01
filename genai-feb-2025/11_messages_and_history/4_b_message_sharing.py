from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire
import os
from collections import deque
from pydantic_ai.messages import ModelMessage


load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

agent1 = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt="You are helpful assistant.",
)

agent2 = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt="You are helpful assistant.",
)

response = agent1.run_sync("tell me a joke")
print(response.data)

response = agent2.run_sync("explanin?", message_history=response.all_messages())
print(response.data)
