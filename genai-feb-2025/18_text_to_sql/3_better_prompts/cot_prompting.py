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
    <example>
    <query>List all employees in the HR department.</query>
    <thought_process>
    1. We need to join the employees and departments tables.
    2. We'll match employees.department_id with departments.id.
    3. We'll filter for the HR department.
    4. We only need to return the employee names.
    </thought_process>
    <sql>SELECT e.name FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'HR';</sql>
    </example>

    <example>
    <query>What is the average salary of employees hired in 2022?</query>
    <thought_process>
    1. We need to work with the employees table.
    2. We need to filter for employees hired in 2022.
    3. We'll use the YEAR function to extract the year from the hire_date.
    4. We'll calculate the average of the salary column for the filtered rows.
    </thought_process>
    <sql>SELECT AVG(salary) FROM employees WHERE YEAR(hire_date) = 2022;</sql>
    </example>
"""

few_shot_prompt = f"""
    You are an AI assistant that converts natural language queries into SQL.

    Follow these guidelines:
    - ALWAYS use the knowledge base that is provided in the context between <schema> </schema> tags to answer user questions.
    - Generate SQL query based ONLY on the information retrieved from the knowledge base. 
    - Provide only the SQL query in your response.
    - Within <thought_process> tags, explain your thought process for creating the SQL query.
    Then, within <sql> tags, provide your output SQL query.

    Here are some examples of natural language queries, thought processes, and their corresponding SQL:
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
