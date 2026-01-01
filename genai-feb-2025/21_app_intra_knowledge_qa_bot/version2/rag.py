from pathlib import Path
import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from agent import *
from config_reader import *
import logfire
import time


class IntraKnowledgeRAG:
    def __init__(self):
        src_dir = settings.file_paths.src_dir
        self.kb_dir = os.path.join(src_dir, settings.file_paths.kb_dir)
        self.vector_db_dir = os.path.join(src_dir, settings.file_paths.vector_db_dir)
        self.bm_25_dir = os.path.join(src_dir, settings.file_paths.bm_25_dir)
        self.agent = get_agent()
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedder.model,
            encode_kwargs={"normalize_embeddings": settings.embedder.normalize},
            model_kwargs={"token": settings.embedder.token},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunker.chunk_size,
            chunk_overlap=settings.chunker.chunk_overlap,
            length_function=len,
        )
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()
        if os.path.exists(self.vector_db_dir):
            self.vector_db = FAISS.load_local(
                folder_path=self.vector_db_dir,
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
            self.bm25_retriever = pickle.load(
                open(os.path.join(self.bm_25_dir, "bm25.pkl"), "rb")
            )
        else:
            self.ingest()

    def ingest(self):
        print("Ingestion of data begins........")

        documents = []
        for file in Path(self.kb_dir).glob("**/*.pdf"):
            print("Ingesting file", file)
            loader = PyPDFLoader(file)
            documents.extend(loader.load_and_split(self.text_splitter))

        vector_db = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings_model,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        vector_db.save_local(folder_path=self.vector_db_dir)
        self.vector_db = vector_db

        if not os.path.exists(self.bm_25_dir):
            os.makedirs(self.bm_25_dir)
        bm25_retriever = BM25Retriever.from_documents(documents)
        with open(os.path.join(self.bm_25_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(bm25_retriever, f)
        self.bm25_retriever = bm25_retriever
        print("Ingestion of data ends........")

    def retrieveContext(
        self, query: str, retriever_top_k: int = settings.retriever.top_k
    ) -> list:
        faiss_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": retriever_top_k}
        )
        self.bm25_retriever.k = retriever_top_k
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        results = ensemble_retriever.invoke(query)
        return [result.page_content for result in results]

    def retrieveContextForUI(
        self, query: str, retriever_top_k: int = settings.retriever.top_k
    ) -> str:
        results = self.retrieveContext(query, retriever_top_k)
        return "\n\n".join(results)

    def execute(
        self, query: str, retriever_top_k: int = settings.retriever.top_k
    ) -> str:
        deps = Deps(self.vector_db, query, retriever_top_k)
        result = self.agent.run_sync(query, deps=deps)
        return result.data
