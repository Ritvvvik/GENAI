from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire
from openai import AsyncOpenAI

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()
logfire.instrument_pydantic_ai()

client = AsyncOpenAI(
    api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.github.ai/inference"
)

agent = Agent(
    model=OpenAIModel(
        model_name=os.getenv("GITHUB_MODEL"),
        provider=OpenAIProvider(openai_client=client),
    ),
    system_prompt="You are a helpful assistant.",
    retries=2,
)


@agent.tool_plain(retries=2)
def calc_volume(size: int) -> int:
    """calculates the volume of box

    Args:
        size: size of the box
    """
    if size == 42:
        return size**3
    else:
        print(f"Invalid size: {size}.")
        raise ModelRetry("Please try again with size=42.")


try:
    result = agent.run_sync("Please get me the volume of a box with size 6.")
except ModelHTTPError as e:
    print(e)
except UnexpectedModelBehavior as e:
    print(e.__cause__)
    print(e)
else:
    print(result.output)
