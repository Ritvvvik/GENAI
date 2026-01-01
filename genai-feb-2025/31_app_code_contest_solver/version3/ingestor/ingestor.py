from pathlib import Path
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import JSONLoader
import logfire
import time
from config.config_reader import settings


class KnowledgeIngestor:
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

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["prob_desc_description"] = record.get("prob_desc_description")
        metadata["src_uid"] = record.get("src_uid")
        metadata["source_code"] = record.get("source_code")
        return metadata

    def ingest(self):
        print(">>> Begin: Ingestion of data...")
        documents = []
        for file in Path(self.kb_dir).glob("*.jsonl"):
            print("Ingesting file", file)
            loader = JSONLoader(
                file_path=file,
                jq_schema=".",
                content_key="prob_desc_description",
                # is_content_key_jq_parsable=True,
                metadata_func=self.metadata_func,
                json_lines=True,
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


if __name__ == "__main__":
    ingestor = KnowledgeIngestor()
    ingestor.ingest()
