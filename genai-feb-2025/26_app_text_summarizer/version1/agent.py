from pydantic_ai import Agent
from llm import build_model
from config_reader import settings

text_summarizer_agent = Agent(model=build_model(), system_prompt=settings.llm.prompt)
