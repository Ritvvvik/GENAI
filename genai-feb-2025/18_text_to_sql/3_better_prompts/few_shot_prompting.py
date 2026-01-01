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


examples = """
    Example 1:
    <query>List all employees in the HR department.</<query>
    <output>SELECT e.name FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'HR';</output>

    Example 2:
    User: What is the average salary of employees in the Engineering department?
    SQL: SELECT AVG(e.salary) FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'Engineering';

    Example 3:
    User: Who is the oldest employee?
    SQL: SELECT name, age FROM employees ORDER BY age DESC LIMIT 1;
"""

few_shot_prompt = f"""
    You are an AI assistant that converts natural language queries into SQL.

    Follow these guidelines:
    - ALWAYS use the knowledge base that is provided in the context between <schema> </schema> tags to answer user questions.
    - Generate SQL query based ONLY on the information retrieved from the knowledge base. 
    - Provide only the SQL query in your response.

    Here are some examples of natural language queries and their corresponding SQL:
    <examples>
        {examples}
    </examples>
    """
text_to_sql_agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    deps_type=Deps,
    system_prompt=few_shot_prompt,
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
