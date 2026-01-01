from pydantic_ai import Agent
from llm.llm import build_model
from config.config_reader import settings
from pydantic import BaseModel
from typing import Annotated


class ProblemInfo(BaseModel):
    description: Annotated[str, ..., "Description of the problem"]
    code: Annotated[str, ..., "Python3 code to solve the problem"]
    plan: Annotated[str, ..., "Planning to solve the problem"]


class RertrieverResponse(BaseModel):
    similar_problems: Annotated[
        list[ProblemInfo],
        ...,
        "List of similar problems with their details. Each problem should have a description, python3 code, and planning.",
    ]
    algorithm_technique: Annotated[
        str,
        ...,
        "Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.",
    ]


instructions = """
Given a problem, provide 2 to 3 relevant problems and then identify the algorithm to solve the original problem.
# Important:
Provide problem description, plan and python3 code for each relevant problem.
"""

model = build_model(
    model_name=settings.llm_generator.name,
    api_key=settings.llm_generator.api_key,
    base_url=settings.llm_generator.base_url,
    temperature=settings.llm_generator.temperature,
    max_tokens=settings.llm_generator.max_tokens,
)

retriever_agent = Agent(
    model=model,
    instructions=instructions,
    output_type=RertrieverResponse,
)
