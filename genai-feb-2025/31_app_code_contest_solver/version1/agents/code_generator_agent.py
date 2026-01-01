from pydantic_ai import Agent
from llm.llm import build_model
from config.config_reader import settings

instructions = """
You are a programmer tasked with solving a given competitive programming problem, generate python3 code to solve the problem.

# Important Note:
Strictly follow the input and output format. 
If you are writing a function then after the function definition, take input from using `input()` function, call the function with specified parameters and finally print the output of the function.
Do not add extra explanation or words.
"""

model = build_model(
    model_name=settings.llm_generator.name,
    api_key=settings.llm_generator.api_key,
    base_url=settings.llm_generator.base_url,
    temperature=settings.llm_generator.temperature,
    max_tokens=settings.llm_generator.max_tokens,
)

code_generator_agent = Agent(
    model=model,
    instructions=instructions,
)
