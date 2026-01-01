from pydantic_ai import Agent
from llm.llm import build_model
from config.config_reader import settings

instructions = """
Given a competitive programming problem and a similar problem with its plan, generate a concrete planning to solve the problem.
# Important 
You should give only the planning to solve the problem. Do not add extra explanation or words.
"""

model = build_model(
    model_name=settings.llm_generator.name,
    api_key=settings.llm_generator.api_key,
    base_url=settings.llm_generator.base_url,
    temperature=settings.llm_generator.temperature,
    max_tokens=settings.llm_generator.max_tokens,
)

plan_generator_agent = Agent(model=model, instructions=instructions)
