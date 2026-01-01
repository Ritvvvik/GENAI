from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from html_to_markdown import convert_to_markdown
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from textwrap import dedent
import logfire
from pydantic_ai.models.gemini import GeminiModel

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


def scrape_and_convert_to_md(url):
    content = requests.get(url).text
    soup = BeautifulSoup(content, "lxml")
    md = convert_to_markdown(str(soup))
    return md


class ProjectInformation(BaseModel):
    """Information about the project"""

    name: str = Field(description="Name of the project")
    tagline: str = Field(
        description="What this project is about",
    )
    benefits: List[str] = Field(
        description="A list of main benefits of the project including 3-5 words to summarize each one."
    )


system_prompt = """
        You are an intelligent research agent. 
        Analyze user request carefully and provide structured responses.
    """

agent = Agent(
    model=GeminiModel(
        model_name=os.getenv("GEMINI_CHAT_MODEL"), api_key=os.getenv("GEMINI_API_KEY")
    ),
    system_prompt=dedent(system_prompt),
    result_type=ProjectInformation,
)

url = "https://playwright.dev/"
page_content = scrape_and_convert_to_md(url)
user_prompt = f"Extract the information from the following web page. The raw data is {page_content}"
response = agent.run_sync(user_prompt)
print(response.data)
