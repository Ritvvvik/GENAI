from langchain_huggingface import HuggingFaceEmbeddings
from pydantic_ai import Agent, RunContext
import os
import time
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass
from pydantic_ai.models.groq import GroqModel
from langchain_community.vectorstores import FAISS
from rich.console import Console

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(1)
logfire.instrument_openai()


@dataclass
class KnowledgeDeps:
    vector_db: FAISS
    top_k: int
    query: str


system_prompt = """
You are a helpful and knowledgeable assistant for the luxury fashion store Harvest & Mill.

Follow these guidelines:
- ALWAYS search the knowledge base that is provided in the context between <context> </context> tags to answer user questions. 
- Generate answer based ONLY on the information retrieved from the knowledge base.
- If information is not found in the knowledge base, politely acknowledge this.
"""

rag_agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY")
    ),
    deps_type=KnowledgeDeps,
    system_prompt=system_prompt,
)


@rag_agent.system_prompt
def add_context_system_prompt(ctx: RunContext[KnowledgeDeps]) -> str:
    base_retriever = ctx.deps.vector_db.as_retriever(
        search_kwargs={"k": ctx.deps.top_k}
    )
    hits = base_retriever.invoke(ctx.deps.query)
    res = "\n".join([hit.page_content for hit in hits])

    return "<context>\n" + res + "\n</context>"


vector_db_dir = os.path.expanduser(
    "~/Documents/genai-training-pydanticai/data/semantic-search/index/faiss2"
)
embeddings_model = HuggingFaceEmbeddings(
    model_name=os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"token": os.getenv("HUGGING_FACE_TOKEN")},
)
vector_db = FAISS.load_local(
    folder_path=vector_db_dir,
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True,
)

# Query sentences:
queries = [
    "How long does international shipping take?",
    "Show the Leather jackets in the store",
    "Contact details of Harvest & Mill store",
    "What is the return policy?",
]

for query in queries:
    deps = KnowledgeDeps(vector_db=vector_db, query=query, top_k=2)
    result = rag_agent.run_sync(query, deps=deps)
    print("\nQuery:", query)
    print("\nAnswer:", result.data)
