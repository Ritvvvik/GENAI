import time
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire
import os
from rich.console import Console

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(2)
logfire.instrument_openai()

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt="You are helpful assistant.",
    instrument=True,
)


def main():
    console = Console()
    console.print(
        "Welcome to Algorithmica Chat Bot. How may I assist you today?",
        style="cyan",
        end="\n\n",
    )
    while True:
        user_message = input(">>")
        if user_message == "q":
            break
        console.print()
        result = agent.run_sync(user_message)
        console.print(result.data, style="cyan", end="\n\n")


if __name__ == "__main__":
    main()

# I am thimmareddy.
# What is your name?
# I am an applied researcher in Artificial Intelligence and exploring the science of consciouness.
# What are you good at?
# Tell me my name
# What do i do?
