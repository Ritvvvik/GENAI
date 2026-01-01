import logfire
import time
from config_reader import settings
from sql_agent import *
from pydantic_ai.messages import ModelMessage
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class DBExplorer:
    def __init__(self):
        vector_db_dir = os.path.join(
            settings.file_paths.src_dir, settings.file_paths.vector_db_dir
        )
        embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedder.model,
            encode_kwargs={"normalize_embeddings": settings.embedder.normalize},
            model_kwargs={"token": settings.embedder.token},
        )
        self.vector_db = FAISS.load_local(
            folder_path=vector_db_dir,
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True,
        )
        self.db_file_path = os.path.join(
            settings.file_paths.src_dir, settings.file_paths.db_file
        )
        self.sql_agent = create_sql_agent()
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    def process(
        self, user_query: str, history: list[ModelMessage], retriever_top_k: int
    ) -> str:
        deps = Deps(
            vector_db=self.vector_db,
            retriever_top_k=retriever_top_k,
            db_file_path=self.db_file_path,
        )
        result = self.sql_agent.run_sync(user_query, message_history=history, deps=deps)
        return result.data
