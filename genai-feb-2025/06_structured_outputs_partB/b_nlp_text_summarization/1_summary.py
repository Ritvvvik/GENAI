from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
import os
from textwrap import dedent
import logfire
from pydantic_ai.models.gemini import GeminiModel

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


class Concept(BaseModel):
    title: str
    description: str


class ArticleSummary(BaseModel):
    invented_year: int
    summary: str
    inventors: list[str]
    description: str
    concepts: list[Concept]


def get_article_content(path):
    with open(path, "r") as f:
        content = f.read()
    return content


def print_summary(summary):
    print(f"Invented year: {summary.invented_year}\n")
    print(f"Summary: {summary.summary}\n")
    print("Inventors:")
    for i in summary.inventors:
        print(f"- {i}")
    print("\nConcepts:")
    for c in summary.concepts:
        print(f"- {c.title}: {c.description}")
    print(f"\nDescription: {summary.description}")


summarization_prompt = """
    You will be provided with content from an article about an invention.
    Your goal will be to summarize the article following the schema provided.
"""

agent = Agent(
    model=GeminiModel(
        model_name=os.getenv("GEMINI_CHAT_MODEL"), api_key=os.getenv("GEMINI_API_KEY")
    ),
    result_type=ArticleSummary,
    system_prompt=dedent(summarization_prompt),
)

path = "~/Documents/genai-training-pydanticai/data/articles/cnns.md"
content = get_article_content(os.path.expanduser(path))
result = agent.run_sync(content)
print_summary(result.data)

path = "~/Documents/genai-training-pydanticai/data/articles/moe.md"
content = get_article_content(os.path.expanduser(path))
result = agent.run_sync(content)
print_summary(result.data)
