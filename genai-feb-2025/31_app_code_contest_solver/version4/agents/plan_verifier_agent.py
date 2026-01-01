from typing import Annotated
from pydantic import BaseModel
from pydantic_ai import Agent
from llm.llm import build_model
from config.config_reader import settings


class PlanVerifierResponse(BaseModel):
    explanation: Annotated[
        str,
        ...,
        "Discuss whether the given competitive programming problem is solvable by using the given plan.",
    ]
    confidence_score: Annotated[
        int,
        ...,
        "Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100.",
    ]


instructions = """
Given a competitive programming problem and a plan to solve the problem, tell whether the plan is correct to solve this problem
"""

model = build_model(
    model_name=settings.llm_generator.name,
    api_key=settings.llm_generator.api_key,
    base_url=settings.llm_generator.base_url,
    temperature=settings.llm_generator.temperature,
    max_tokens=settings.llm_generator.max_tokens,
)

plan_verifier_agent = Agent(
    model=model, instructions=instructions, output_type=PlanVerifierResponse
)
