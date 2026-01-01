from typing import Annotated, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from support_agent import *
from loan_agent import *
from llm import *


class TriageResult(BaseModel):
    department: Annotated[str, ..., "Department to direct the customer query to"]
    text_response: Annotated[str, ..., "Text response to the customer query"]


triage_agent = Agent(
    model=build_model(),
    system_prompt=(
        "You are a triage agent in our bank, responsible for directing customer queries to the appropriate department. "
        "For each query, determine whether it is related to support (e.g., balance, card, account-related queries) or "
        "loan services (e.g., loan status, application, and loan-related inquiries). "
        "If the query is related to support, direct the customer to the support team with an appropriate response. "
        "If the query is related to loans, direct the customer to the loan department with a relevant response. "
        "If the query is unclear or does not fit into either category, politely inform the customer and suggest they "
        "ask about loans or support. "
        "Always ensure that the response is clear, concise, and provides direction to the right department for further"
        " assistance."
        "Never generate data based on your internal knowledge; always rely on the provided tools to fetch the most "
        "accurate and up-to-date information."
    ),
    output_type=TriageResult,
    output_retries=2,
)


@triage_agent.tool()
async def call_support_agent(ctx: RunContext, prompt: str, customer_id: int) -> Any:
    support_deps = SupportDependencies(customer_id=customer_id, db=CustomerDB())
    result = await support_agent.run(prompt, deps=support_deps, usage=ctx.usage)
    return result.output


@triage_agent.tool()
async def call_loan_agent(ctx: RunContext, prompt: str, customer_id: int) -> Any:
    loan_deps = LoanDependencies(customer_id=customer_id, db=LoanDB())
    result = await loan_agent.run(prompt, deps=loan_deps, usage=ctx.usage)
    return result.output
