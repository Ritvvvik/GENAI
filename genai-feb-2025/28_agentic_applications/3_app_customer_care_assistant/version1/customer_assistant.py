from config_reader import *
from loan_agent import *
from support_agent import *
from triage_agent import *
import asyncio
import logfire
import time


class CustomerAssistant:
    def __init__(self):
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    async def process(self, query: str) -> tuple:
        result = await triage_agent.run(query)
        return result.output


async def main(query: str):
    ca = CustomerAssistant()
    result = await ca.process(query)
    print(result)


if __name__ == "__main__":
    query = "what is the loan status of customer with id=123"
    # query = "what is the account balance"
    asyncio.run(main(query))
