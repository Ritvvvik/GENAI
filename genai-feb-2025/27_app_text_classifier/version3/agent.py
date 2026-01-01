from dataclasses import dataclass
import textwrap
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
    text_classifier_agent = Agent(
        model=build_model(), system_prompt=settings.llm.prompt
    )

    @text_classifier_agent.system_prompt
    def add_context_system_prompt(ctx: RunContext[Deps]) -> str:
        base_retriever = ctx.deps.vector_db.as_retriever(
            search_kwargs={"k": ctx.deps.top_k}
        )
        examples = base_retriever.invoke(ctx.deps.query)
        rag_string = "Use the following examples to help you classify the query:"
        rag_string += "\n<examples>\n"
        for example in examples:
            rag_string += textwrap.dedent(
                f"""
            <example>
                <query>
                    "{example.metadata["question"]}"
                </query>
                <label>
                    {example.metadata["answer"]}
                </label>
            </example>
            """
            )
        rag_string += "\n</examples>"
        return rag_string

    return text_classifier_agent
