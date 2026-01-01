from pydantic_ai import Agent, RunContext
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from llm.llm import build_model
from config.config_reader import settings
from dataclasses import dataclass


@dataclass
class Deps:
    vector_db: FAISS
    query: str
    reranker_model: str
    retriever_top_k: int
    reranker_top_k: int


model = build_model(
    model_name=settings.llm_generator.name,
    api_key=settings.llm_generator.api_key,
    base_url=settings.llm_generator.base_url,
    temperature=settings.llm_generator.temperature,
    max_tokens=settings.llm_generator.max_tokens,
)

code_improver_agent = Agent(model=model, deps_type=Deps)


@code_improver_agent.instructions()
def dynamic_instructions(ctx: RunContext[Deps]) -> str:
    tmp = """
You are a programmer who has received a solution of a problem written in python3 that fails to pass certain test cases. 
Your task is to modify the code in such a way so that it can pass all the test cases. 
Use the feeback to improve the code.
You will also be given multiple somewhat similar problems, as well as the solution to those similar problems.
Feel free to use the given information to aid your problem solving process if necessary.
"""
    base_retriever = ctx.deps.vector_db.as_retriever(
        search_kwargs={"k": ctx.deps.retriever_top_k}
    )
    compressor = CrossEncoderReranker(
        model=ctx.deps.reranker_model, top_n=ctx.deps.reranker_top_k
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    results = compression_retriever.invoke(ctx.deps.query)
    tmp += "\n# similar problems\n"
    for result in results:
        tmp += "\n## problem\n"
        tmp += result.page_content
        tmp += "\n## solution\n"
        tmp += result.metadata["source_code"]

    tmp += """
# Important Note:
Strictly follow the input and output format. 
If you are writing a function then after the function definition, take input from using `input()` function, call the function with specified parameters and finally print the output of the function.
Do not add extra explanation or words.
"""
    return tmp
