from datetime import date
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


class TranslatedString(BaseModel):
    input_language: str = Field(
        ..., description="The language of the original text, as 2-letter language code."
    )
    translation: str = Field(
        ..., description="The translated text in the target language."
    )


system_prompt = """
        Detect the language of the original text and translate it into English.
    """
agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=dedent(system_prompt),
    result_type=TranslatedString,
)
user_prompt = "Sprachkenntnisse sind ein wichtiger Bestandteil der Kommunikation."

response = agent.run_sync(user_prompt)
print(response.data)
