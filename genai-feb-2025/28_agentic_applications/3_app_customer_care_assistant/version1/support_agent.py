from dataclasses import dataclass
from typing import Annotated
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from llm import *


# --- Fake Database for Customer Data ---
class CustomerDB:
    """This is a fake database for example purposes.

    In reality, you'd be connecting to an external database
    (e.g. PostgreSQL) to get information about customers.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return "John"

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        if id == 123:
            return 123.45
        else:
            raise ValueError("Customer not found")


@dataclass
class SupportDependencies:
    customer_id: int
    db: CustomerDB


class SupportResult(BaseModel):
    support_advice: Annotated[str, ..., "Advice returned to the customer"]
    block_card: Annotated[bool, ..., "Whether to block their card or not"]
    risk: Annotated[int, ..., "Risk level of query"]
    customer_tracking_id: Annotated[str, ..., "Tracking ID for customer"]


support_agent = Agent(
    model=build_model(),
    deps_type=SupportDependencies,
    output_type=SupportResult,
    system_prompt=(
        "You are a support agent in our bank, give the "
        "customer support and judge the risk level of their query. "
    ),
    output_retries=2,
)


@support_agent.tool()
async def block_card(ctx: RunContext[SupportDependencies], customer_name: str) -> str:
    return f"I'm sorry to hear that, {customer_name}. We are temporarily blocking your card to prevent unauthorized transactions."


@support_agent.tool()
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """Returns the customer's current account balance."""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f"${balance:.2f}"
