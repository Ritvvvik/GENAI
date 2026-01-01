from langchain_community.vectorstores import FAISS
from dataclasses import dataclass


@dataclass
class Deps:
    vector_db: FAISS
    retriever_top_k: int
    db_file_path: str
    user_query: str
