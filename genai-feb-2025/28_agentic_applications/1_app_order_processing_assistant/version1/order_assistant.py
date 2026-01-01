import logfire
import time
from config_reader import settings
from order_processing_agent import *
from pydantic_ai.messages import ModelMessage


class OrderAssistant:
    def __init__(self):
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()
        self.order_agent = create_order_agent()

    def process(self, user_query: str, history: list[ModelMessage]) -> OrderResponse:
        deps = Deps(order_db_service=OrderDBService())
        result = self.order_agent.run_sync(
            user_query, message_history=history, deps=deps
        )
        logfire.info(f"thoughts:{result.data.thoughts}")
        return result.data
