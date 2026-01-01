from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
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
    model=OpenAIModel(
        model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    ),
    system_prompt="You are helpful assistant.",
)

result = agent.run_sync("tell me a joke, in short")
logfire.info("result data {data}", data=result.data)
logfire.info("new messages {msgs}", msgs=result.new_messages())
logfire.info("all messages {msgs}", msgs=result.all_messages())

result = agent.run_sync("tell me about yourself, in a sentence")
logfire.info("result data {data}", data=result.data)
logfire.info("new messages {msgs}", msgs=result.new_messages())
logfire.info("all messages {msgs}", msgs=result.all_messages())
