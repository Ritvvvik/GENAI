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


def cost(result):
    # Cost per million tokens in USD
    input_cost_per_million = 0.15
    response_cost_per_million = 0.60

    # Extract tokens from the result object
    request_tokens = result.request_tokens
    response_tokens = result.response_tokens

    # Calculate the cost for input tokens and response tokens
    input_cost = (request_tokens / 1_000_000) * input_cost_per_million
    response_cost = (response_tokens / 1_000_000) * response_cost_per_million

    # Calculate total cost for this interaction
    total_cost = input_cost + response_cost

    # Total calls (assuming 'requests' attribute is the total number of calls)
    total_calls = result.requests

    # Print the detailed output
    print(f"Interaction Details:")
    print(
        f"Total input tokens were: {request_tokens} and Response tokens were: {response_tokens}"
    )
    print(f"Total cost for this interaction: ${total_cost:.4f}")
    print(f"Total Calls to LLM: {total_calls}")


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


try:
    result = agent.run_sync("What is the capital of india")
    cost(result.usage())
except ModelHTTPError as e:
    print(e)
except UnexpectedModelBehavior as e:
    print(e.__cause__)
    print(e)
else:
    print(result.output)
