from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import os
from pathlib import Path
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

system_prompt = """
    You are helpful assistant.
    Use the tool `list_all_files` to get all the files of a given directory.
"""

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    system_prompt=system_prompt,
)


@agent.tool_plain
def list_all_files(dir: str) -> list[str]:
    """
    List all files in the input directory

    Args:
        dir (str): The directory to list files from

    Returns:
        (list[str]): A list of all file paths in the input directory
    """
    logfire.info(f"listing files of {dir}")
    files = [str(path) for path in Path(dir).glob("**/*")]
    return files


dir = os.path.expanduser("~/Documents/genai-training-pydanticai/data/articles")
response = agent.run_sync(f"What are the files in the directory {dir}")
print(response.data)
print(response.all_messages())
