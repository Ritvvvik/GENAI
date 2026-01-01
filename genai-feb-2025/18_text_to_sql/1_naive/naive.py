# https://inloop.github.io/sqlite-viewer/
# pip install sqlite3
from pathlib import Path
from pydantic_ai import Agent, RunContext
import os
import time
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass
from pydantic_ai.models.groq import GroqModel
from utils import get_schema_info

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(1)
logfire.instrument_openai()


@dataclass
class Deps:
    db_path: str


system_prompt = """
    You are an AI assistant that converts natural language queries into SQL.

    Follow these guidelines:
    - ALWAYS use the knowledge base that is provided in the context between <schema> </schema> tags to answer user questions.
    - Generate SQL query based ONLY on the information retrieved from the knowledge base. 
    - Provide only the SQL query in your response.
    """
text_to_sql_agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    deps_type=Deps,
    system_prompt=system_prompt,
)


@text_to_sql_agent.system_prompt
def add_context_system_prompt(ctx: RunContext[Deps]) -> str:
    res = get_schema_info(ctx.deps.db_path)
    return "\n<schema>\n" + res + "\n</schema>"


# Query sentences:
queries = [
    "How many employees are there?",
    "What are the names and salaries of employees in the Marketing department?",
    "What are the names and hire dates of employees in the Engineering department, ordered by their salary?",
    "What is the average salary of employees hired in 2022?",
]

src_dir = os.path.expanduser("~/Documents/genai-training-pydanticai/data/text-to-sql")
db_file = "emp.sqlite"
deps = Deps(kb_path=os.path.join(src_dir, db_file))

for query in queries:
    result = text_to_sql_agent.run_sync(query, deps=deps)
    print("\nQuery:", query)
    print("\nAnswer:", result.data)
