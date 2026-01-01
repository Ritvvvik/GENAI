from pathlib import Path
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import JSONLoader
import logfire
import time
from utils import *
import json


class Ingestor:
    def __init__(self):
        self.src_dir = settings.file_paths.src_dir
        self.kb_dir = os.path.join(self.src_dir, settings.file_paths.kb_dir)
        self.vector_db_dir = os.path.join(
            self.src_dir, settings.file_paths.vector_db_dir
        )
        self.retriever_top_k = settings.retriever.top_k
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedder.model,
            encode_kwargs={"normalize_embeddings": settings.embedder.normalize},
            model_kwargs={"token": settings.embedder.token},
        )
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    def chunk_schema(self):
        db_file = settings.file_paths.db_file
        db_file_path = os.path.join(self.src_dir, db_file)

        # get chunks for the database schema
        print(">>> Begin: Chunking...")
        chunk_strategy = settings.chunker.chunk_strategy
        if chunk_strategy == "table_level":
            chunks = get_chunks_by_table(db_file_path)
        else:
            chunks = get_chunks_by_field(db_file_path)

        # create the directory to store chunks
        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir)
        json_file_name = db_file.split(".")[0].split("/")[1] + "_schema.json"
        json_file_path = os.path.join(self.kb_dir, json_file_name)

        # store the chunks
        with open(json_file_path, "w") as f:
            json.dump(chunks, f, indent=2)
        print(">>> End: Chunking...")

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["table"] = record.get("table")
        metadata["column"] = record.get("column")
        metadata["type"] = record.get("type")
        return metadata

    def ingest(self):
        print(">>> Begin: Ingestion of data...")
        documents = []
        for file in Path(self.kb_dir).glob("*.json"):
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
        print(">>> End: Ingestion of data...")

    def execute_pipeline(self):
        self.chunk_schema()
        self.ingest()


if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.execute_pipeline()
