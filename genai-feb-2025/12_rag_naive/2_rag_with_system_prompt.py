from pathlib import Path
from pydantic_ai import Agent, RunContext
import os
import time
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass
from pydantic_ai.models.groq import GroqModel
from rich.console import Console

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(1)
logfire.instrument_openai()


@dataclass
class KnowledgeDeps:
    kb_path: str


system_prompt = """
You are a helpful and knowledgeable assistant for the luxury fashion store H&M.
Your role is to provide detailed information and assistance about the store and its products.

Follow these guidelines:
- ALWAYS search the knowledge base that is provided in the context between <context> </context> tags to answer user questions.
- Provide accurate product and policy information based ONLY on the information retrieved from the knowledge base. Never make assumptions or provide information not present in the knowledge base.
- Structure your responses in a clear, concise and professional manner, maintaining our premium brand standards
- Highlight unique features, materials, and care instructions when relevant.
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
    kb = "<context>\n"
    for file in Path(ctx.deps.kb_path).glob("*.txt"):
        content = file.read_text(encoding="utf-8")
        kb += content
        kb += "\n"
    kb += "</context>"
    return kb


def main():
    console = Console()
    path = os.path.expanduser("~/Documents/genai-training-pydanticai/data/kb1")
    deps = KnowledgeDeps(kb_path=path)
    console.print(
        "Welcome to Harvest & Mill. How may I assist you today?",
        style="cyan",
        end="\n\n",
    )
    while True:
        user_message = input()
        if user_message == "q":
            break
        console.print()
        result = rag_agent.run_sync(user_message, deps=deps)
        console.print(result.data, style="cyan", end="\n\n")


if __name__ == "__main__":
    main()

# How long does international shipping take?
# Show leather jackets
# How should i contact you?
