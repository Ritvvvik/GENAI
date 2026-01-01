from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from llm import build_model
from langchain_community.vectorstores import FAISS
from config_reader import settings


@dataclass
class Deps:
    vector_db: FAISS
    query: str
    top_k: int


def get_agent() -> Agent:
    text_to_sql_agent = Agent(
        model=build_model(),
        system_prompt=settings.llm.prompt,
        deps_type=Deps,
    )

    @text_to_sql_agent.system_prompt
    def add_context_system_prompt(ctx: RunContext[Deps]) -> str:
        base_retriever = ctx.deps.vector_db.as_retriever(
            search_kwargs={"k": ctx.deps.top_k}
        )
        results = base_retriever.invoke(ctx.deps.query)
        context = "\n".join([result.page_content for result in results])
        return context

    return text_to_sql_agent
