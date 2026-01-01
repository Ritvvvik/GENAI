from typing import Annotated
from pydantic_ai import Agent, RunContext
from llm import *
from config_reader import *
from dependencies import *


class GeneratorResponse(BaseModel):
    thoughts: Annotated[
        str,
        ...,
        "Your thoughts to execute the task.",
    ]
    sql_query: Annotated[str, ..., "The generated sql query."]


def create_sql_generator_agent():
    model = build_model(
        model_name=settings.llm_generator.name,
        api_key=settings.llm_generator.api_key,
        base_url=settings.llm_generator.base_url,
        temperature=settings.llm_generator.temperature,
        max_tokens=settings.llm_generator.max_tokens,
    )
    sql_generator_agent = Agent(
        model=model,
        instructions=settings.llm_generator.prompt,
        deps_type=Deps,
        output_type=GeneratorResponse,
        retries=3,
    )

    @sql_generator_agent.instructions
    def get_database_or_table_schema(ctx: RunContext[Deps]) -> str:
        base_retriever = ctx.deps.vector_db.as_retriever(
            search_kwargs={"k": ctx.deps.retriever_top_k}
        )
        results = base_retriever.invoke(ctx.deps.user_query)
        results_str = "\n".join([result.page_content for result in results])
        return "<schema>\n" + results_str + "</schema>"

    return sql_generator_agent
