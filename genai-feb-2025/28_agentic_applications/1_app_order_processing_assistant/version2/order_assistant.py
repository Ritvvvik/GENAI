import logfire
import time
from config_reader import settings
from order_processing_agent import *
from pydantic_ai.messages import ModelMessage
from input_guard_rail_agent import *


class OrderAssistant:
    def __init__(self):
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()
        self.deps = Deps(order_db_service=OrderDBService())
        self.order_agent = create_order_agent()

    def get_query_category(self, user_query: str) -> str:
        result = input_gaurd_rail_agent.run_sync(user_query)
        return result.data.ticket_class.value

    def process(self, user_query: str, history: list[ModelMessage]) -> str:
        query_category = self.get_query_category(user_query)
        if query_category == RequestClassEnum.OT:
            return "Invalid Request"
        result = self.order_agent.run_sync(
            user_query, message_history=history, deps=self.deps
        )
        return result.data
