from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from langchain_community.vectorstores import FAISS
from llm import build_model
from config_reader import settings


@dataclass
class Deps:
    vector_db: FAISS
    query: str
    top_k: int


def get_agent() -> Agent:
    agent = Agent(
        model=build_model(),
        system_prompt=settings.llm.prompt,
        deps_type=Deps,
    )

    @agent.system_prompt
    def add_context_system_prompt(ctx: RunContext[Deps]) -> str:
        base_retriever = ctx.deps.vector_db.as_retriever(
            search_kwargs={"k": ctx.deps.top_k}
        )
        results = base_retriever.invoke(ctx.deps.query)
        context = "\n".join([result.page_content for result in results])
        return "\n<context>\n" + context + "\n</context>"

    return agent
