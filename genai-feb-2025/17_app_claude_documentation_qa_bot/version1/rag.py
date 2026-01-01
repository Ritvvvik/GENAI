from pathlib import Path
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from agent import *
from config_reader import *
import logfire
import time


class DocumentationRAG:
    def __init__(self):
        src_dir = settings.file_paths.src_dir
        self.kb_dir = os.path.join(src_dir, settings.file_paths.kb_dir)
        self.vector_db_dir = os.path.join(src_dir, settings.file_paths.vector_db_dir)
        self.agent = get_agent()
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedder.model,
            encode_kwargs={"normalize_embeddings": settings.embedder.normalize},
            model_kwargs={"token": settings.embedder.token},
        )
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()
        self.ingest()

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["source"] = record.get("chunk_link")
        metadata["title"] = record.get("chunk_heading")
        return metadata

    def ingest(self):
        print("Ingestion of data begins........")

        documents = []
        for file in Path(self.kb_dir).glob("**/*.json"):
            print("Ingesting file", file)
            loader = JSONLoader(
                file_path=file,
                jq_schema=".[]",
                content_key=".text",
                is_content_key_jq_parsable=True,
                metadata_func=self.metadata_func,
            )
            documents.extend(loader.load())

        vector_db = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings_model,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        vector_db.save_local(folder_path=self.vector_db_dir)
        self.vector_db = vector_db
        print("Ingestion of data ends........")

    def retrieveContext(
        self, query: str, retriever_top_k: int = settings.retriever.top_k
    ) -> list:
        base_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": retriever_top_k}
        )
        results = base_retriever.invoke(query)

        return [result.page_content for result in results]

    def execute(
        self, query: str, retriever_top_k: int = settings.retriever.top_k
    ) -> str:
        deps = Deps(self.vector_db, query, retriever_top_k)
        result = self.agent.run_sync(query, deps=deps)
        return result.data
