from typing import Annotated, Any
from uuid import uuid4
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from order_db_service import *
from llm import *
from config_reader import *


@dataclass
class Deps:
    order_db_service: OrderDBService


def get_order_details(ctx: RunContext[Deps], order_id: str) -> str:
    """Get the current status and details of an order."""
    order = ctx.deps.order_db_service.get_order(order_id)
    if not order:
        return "Order not found"

    return order.model_dump_json()


def update_shipping_address(
    ctx: RunContext[Deps],
    order_id: str,
    street: str,
    city: str,
    postal_code: str,
    country: str,
) -> str:
    """Update the shipping address for an order if possible."""
    order = ctx.deps.order_db_service.get_order(order_id)
    if not order:
        return "Order not found"

    if not order.can_modify:
        return f"Cannot modify order - current status is {order.status.value}"

    new_address = Address(
        street=street, city=city, postal_code=postal_code, country=country
    )

    if ctx.deps.order_db_service.update_shipping_address(order_id, new_address):
        return f"Successfully updated shipping address"
    return "Failed to update shipping address"


def cancel_order(ctx: RunContext[Deps], order_id: str) -> str:
    """Cancel an order if possible."""
    order = ctx.deps.order_db_service.get_order(order_id)
    if not order:
        return "Order not found"

    if not order.can_modify:
        return f"Cannot modify order - current status is {order.status.value}"

    if ctx.deps.order_db_service.update_order_status(order_id, OrderStatus.CANCELLED):
        return f"Successfully cancelled order"
    return "Failed to cancel order"


def request_return(ctx: RunContext[Deps], order_id: str, reason: ReturnReason) -> str:
    """Process a return request for an order."""
    order = ctx.deps.order_db_service.get_order(order_id)
    if not order:
        return "Order not found"

    if not order.can_return:
        if order.status != OrderStatus.DELIVERED:
            return f"Cannot return order - current status is {order.status.value}"
        return "Cannot return order - outside our 30-day return window."

    return_id = "RET-" + uuid4().hex[:12]

    return (
        f"Return request approved for order {order_id}:\n"
        f"Reason: {reason.value}\n\n"
        f"A return label with ID {return_id} has been emailed to you. Please ship items within 14 days with all original tags attached."
    )


def escalate_to_human(
    ctx: RunContext[Deps],
    reason: EscalationReason,
    high_priority: bool = False,
) -> str:
    """Escalate the conversation to a human.
    Set high_priority=True for urgent matters or when customer is clearly dissatisfied.
    """
    response_time = "1 hour" if high_priority else "24 hours"
    return f"This matter has been escalated to our support team. We will contact you within {response_time}."


class OrderResponse(BaseModel):
    thoughts: Annotated[
        str,
        ...,
        "Your thoughts to execute the task.",
    ]
    response: Annotated[str, ..., "The generated response."]


def create_order_agent():
    model = build_model(
        model_name=settings.llm.name,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )
    order_agent = Agent(
        model=model,
        system_prompt=settings.llm.prompt,
        tools=[
            get_order_details,
            update_shipping_address,
            cancel_order,
            request_return,
            escalate_to_human,
        ],
        result_type=OrderResponse,
        retries=3,
    )
    return order_agent


def main():
    deps = Deps(order_db_service=OrderDBService())
    user_message = "What is the status of order 001?"
    order_agent = create_order_agent()
    result = order_agent.run_sync(user_message, deps=deps)
    print(result.data)


if __name__ == "__main__":
    main()

test_messages = (
    "What is the status of order 001?"
    "What is the status of order 002?"
    "please update the shipping address of order 001: 10 Fifth Avenue, LosAngels, California, 10005, USA"
    "please update the shipping address of order 002: 10 Fifth Avenue, LosAngels, California, 10005, USA"
    "Cancel my order 001"
    "Cancel my order 002"
    "Requesting a return of item shipped as part of order 002, reason being wrong size"
    "Escalate to human for my order 001 since it is not resolved"
    "Escalate to human for my order 001 since customer is unhappy"
)
