from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire
import os

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

system_prompt = """
    You are helpful assistant.
    Use 'is_prime' tool to determine whether an integer is prime or not.
"""
agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)


@agent.tool_plain
def is_prime(a: int) -> bool:
    """Determines whether an integer is a prime number"""
    if a <= 1:
        return False
    for i in range(2, int(a**0.5) + 1):
        if a % i == 0:
            return False
    return True


# response = agent.run_sync("is 13 prime number?")
# response = agent.run_sync("is 16 prime number?")
response = agent.run_sync("is 27 prime number?")
print(response.data)
