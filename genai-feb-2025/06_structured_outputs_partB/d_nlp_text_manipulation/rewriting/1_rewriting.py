from datetime import date
from enum import Enum
import json
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from textwrap import dedent
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


class TextStyle(str, Enum):
    formal = "formal"
    informal = "informal"
    academic = "academic"
    professional = "professional"
    business = "business"


class NormalizedText(BaseModel):
    style: TextStyle = Field(..., description=("The style of the text normalization."))
    text: str


class NormalizedTexts(BaseModel):
    normalized_texts: List[NormalizedText]


system_prompt = (
    "Normalize the user-provided text into the following styles: "
    + json.dumps([style.value for style in TextStyle])
)

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=dedent(system_prompt),
    result_type=NormalizedTexts,
)

response = agent.run_sync(
    "Large Language Models are a powerful tool for natural language processing"
)
print(response.data)
