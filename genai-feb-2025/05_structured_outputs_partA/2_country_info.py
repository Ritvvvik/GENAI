from datetime import date
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from textwrap import dedent
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


class CountryInfo(BaseModel):
    continent_name: str
    country_name: str
    capital_name: str
    has_river: bool
    has_sea: bool
    weather: str = Field(description="Weather over the year")


system_prompt = """
        You are an intelligent research agent. 
        Analyze user request carefully and provide structured responses.
    """

agent = Agent(
    model=OpenAIModel(
        model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    ),
    system_prompt=dedent(system_prompt),
    result_type=CountryInfo,
)
response = agent.run_sync("tell me about India")
print(response.data)

response = agent.run_sync("tell me about Australia")
print(response.data)
