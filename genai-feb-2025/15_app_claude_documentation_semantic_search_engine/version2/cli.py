from dotenv import load_dotenv
import os
from semantic_searcher_with_rerank import SemanticSearcherWithRerank
from rich.console import Console


def main():
    load_dotenv(override=True)
    src_dir = os.path.expanduser(
        "~/Documents/genai-training-pydanticai/data/claude-doc-semenatic-search-engine"
    )
    vector_db_dir = "index"
    kb_dir = "kb"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    reranking_model_name = "BAAI/bge-reranker-base"
    retriever_top_k = 10
    reranker_top_k = 2
    ss = SemanticSearcherWithRerank(
        src_dir,
        kb_dir,
        vector_db_dir,
        embedding_model_name,
        reranking_model_name,
        retriever_top_k,
        reranker_top_k,
    )
    console = Console()
    console.print(
        "Welcome to SemanticSearch On Claude Documentation.  Ask questions and get semantically closest results.",
        style="cyan",
        end="\n\n",
    )
    while True:
        user_question = input(">>")
        if user_question == "q":
            break
        console.print()
        result = ss.retrieveContent(user_question)
        console.print(result, style="cyan", end="\n\n")


if __name__ == "__main__":
    main()

# supported llm models
# embedding models
# rate limits of models
# Pricing of models
# multi-modal models supported
# free tier models supported
# Text classification example
