import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from playwright.sync_api import sync_playwright
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
    pw = sync_playwright().start()
    browser = pw.firefox.launch(headless=False)
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = context.new_page()
    page.goto(url)
    content = page.content()
    soup = BeautifulSoup(content, "lxml")
    md = convert_to_markdown(str(soup))
    browser.close()
    pw.stop()
    return md


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str


class PropertyFeatures(BaseModel):
    bedrooms: int
    bathrooms: int
    square_footage: float
    lot_size: float  # in acres, but we can convert to sqft if needed


class AdditionalInfo(BaseModel):
    price: float
    listing_agent: str
    last_updated: datetime.date


class Property(BaseModel):
    address: Address
    info: AdditionalInfo
    type: str  # Single Family Home
    mls_id: int
    features: PropertyFeatures
    garage_spaces: int


system_prompt = """
        You are an intelligent research agent. 
        Analyze user request carefully and provide structured responses.
    """

agent = Agent(
    model=GeminiModel(
        model_name=os.getenv("GEMINI_CHAT_MODEL"), api_key=os.getenv("GEMINI_API_KEY")
    ),
    system_prompt=dedent(system_prompt),
    result_type=Property,
    result_retries=3,
)

url = "https://www.redfin.com/VA/Sterling/47516-Anchorage-Cir-20165/home/11931811"

page_content = scrape_and_convert_to_md(url)
user_prompt = f"Extract the information from the following web page. The raw data is {page_content}"
response = agent.run_sync(user_prompt)
print(response.data)
