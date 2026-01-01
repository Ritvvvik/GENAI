from llm import *
from dataclasses import dataclass
from typing import Annotated
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


# --- Fake Database for Loan Data ---
class LoanDB:
    """This is a fake loan database for example purposes.
    In reality, you'd be connecting to an external database
    (e.g., PostgreSQL) to manage loan information.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return "John"

    @classmethod
    async def loan_status(cls, *, id: int) -> str | None:
        """Fetch the loan status of a customer by their ID."""
        if id == 123:
            return "Active"
        elif id == 124:
            return "Paid off"
        elif id == 125:
            return "Defaulted"
        else:
            return None

    @classmethod
    async def cancel_loan(cls, *, id: int) -> str:
        """Cancel a loan for a customer."""
        if id == 123:
            # Fake logic for canceling a loan
            return f"Loan for customer ID {id} has been canceled."
        else:
            raise ValueError(f"Customer with ID {id} does not have an active loan.")

    @classmethod
    async def add_loan(cls, *, id: int, amount: float, interest_rate: float) -> str:
        """Add a loan for a customer."""
        if id == 123:
            # Fake logic for adding a loan
            return f"Loan of ${amount} with an interest rate of {interest_rate}% has been added for customer ID {id}."
        else:
            raise ValueError(f"Customer with ID {id} cannot be found to add a loan.")

    @classmethod
    async def loan_balance(cls, *, id: int) -> float | None:
        """Fetch the remaining balance of a customer's loan."""
        if id == 123:
            return 5000.0  # Fake loan balance
        elif id == 124:
            return 0.0  # Loan paid off
        else:
            raise ValueError(f"Customer with ID {id} not found or no loan exists.")


@dataclass
class LoanDependencies:
    customer_id: int
    db: LoanDB


class LoanResult(BaseModel):
    loan_approval_status: Annotated[
        str, ..., "Approval status of the loan (e.g., Approved, Denied, Pending)"
    ]
    loan_balance: Annotated[float, ..., "Remaining balance of the loan"]


loan_agent = Agent(
    model=build_model(),
    deps_type=LoanDependencies,
    output_type=LoanResult,
    system_prompt=(
        "You are a support agent in our bank, assisting customers with loan-related inquiries. "
        "For every query, provide the following information: "
        "- Loan approval status (e.g., Approved, Denied, Pending) "
        "- Loan balance "
        "Please ensure that your response is clear and helpful for the customer. "
        "Always conclude by providing the customer’s name and capturing their information in the marking system using "
        "the tool `capture_customer_name`. "
        "Never generate data based on your internal knowledge; always rely on the provided tools to fetch the most "
        "accurate and up-to-date information."
    ),
    output_retries=2,
)


@loan_agent.tool()
async def loan_status(ctx: RunContext[LoanDependencies]) -> str:
    status = await ctx.deps.db.loan_status(id=ctx.deps.customer_id)
    return f"The loan status is {status!r}"


@loan_agent.tool()
async def cancel_loan(ctx: RunContext[LoanDependencies]) -> str:
    return await ctx.deps.db.cancel_loan(id=ctx.deps.customer_id)


@loan_agent.tool()
async def add_loan(
    ctx: RunContext[LoanDependencies], amount: float, interest_rate: float
) -> str:
    return await ctx.deps.db.add_loan(
        id=ctx.deps.customer_id, amount=amount, interest_rate=interest_rate
    )


@loan_agent.tool()
async def loan_balance(ctx: RunContext[LoanDependencies]) -> float:
    return await ctx.deps.db.loan_balance(id=ctx.deps.customer_id)
