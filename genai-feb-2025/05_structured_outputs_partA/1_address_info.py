from datetime import date
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from textwrap import dedent
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


class AddressInfo(BaseModel):
    first_name: str
    last_name: str
    street: str
    house_number: str
    postal_code: str
    city: str
    state: str
    country: str


system_prompt = """
        You are an intelligent research agent. 
        Analyze user request carefully and provide structured responses.
    """

agent = Agent(
    model=OpenAIModel(
        model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    ),
    system_prompt=dedent(system_prompt),
    result_type=AddressInfo,
)
user_prompt = "Sherlock Holmes lives in the United Kingdom. His residence is in at 221B Baker Street, London, NW1 6XE."
response = agent.run_sync(user_prompt)
print(response.data)
