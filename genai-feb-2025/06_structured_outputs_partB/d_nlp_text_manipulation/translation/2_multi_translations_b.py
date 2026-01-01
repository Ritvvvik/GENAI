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
from pydantic_ai.models.gemini import GeminiModel

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


class TargetLanguage(str, Enum):
    de = "de"
    fr = "fr"
    it = "it"
    es = "es"
    he = "he"


class TranslatedString(BaseModel):
    input_language: TargetLanguage = Field(
        ..., description="The language of the original text, as 2-letter language code."
    )
    translation: str = Field(
        ..., description="The translated text in the target language."
    )


class Translations(BaseModel):
    translations: List[TranslatedString]


system_prompt = (
    "Translate the user-provided text into the following languages: "
    + json.dumps([language.value for language in TargetLanguage])
)

agent = Agent(
    model=GeminiModel(
        model_name=os.getenv("GEMINI_CHAT_MODEL"), api_key=os.getenv("GEMINI_API_KEY")
    ),
    system_prompt=dedent(system_prompt),
    result_type=TranslatedString,
)

user_prompt = (
    "Large Language Models are a powerful tool for natural language processing."
)

response = agent.run_sync(user_prompt)
print(response.data)
