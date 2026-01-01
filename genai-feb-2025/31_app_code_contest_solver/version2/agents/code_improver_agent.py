from pydantic_ai import Agent
from llm.llm import build_model
from config.config_reader import settings


instructions = """
You are a programmer who has received a solution of a problem written in python3 that fails to pass certain test cases. 
Your task is to modify the code in such a way so that it can pass all the test cases. 
Use the feeback to improve the code.

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

code_improver_agent = Agent(model=model, instructions=instructions)
