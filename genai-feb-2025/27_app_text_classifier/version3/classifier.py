import os
from config_reader import *
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from agent import *
import logfire
import time


class TextClassifierRAG:
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
        if os.path.exists(self.vector_db_dir):
            self.vector_db = FAISS.load_local(
                folder_path=self.vector_db_dir,
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
        else:
            self.ingest()

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["question"] = record.get("question")
        metadata["answer"] = record.get("answer")
        return metadata

    def ingest(self):
        print(">>> Begin: Ingestion of data...")

        documents = []
        for file in Path(self.kb_dir).glob("**/*.json"):
            print("Ingesting file", file)
            loader = JSONLoader(
                file_path=file,
                jq_schema=".[]",
                content_key=".question",
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
        print(">>> End:Ingestion of data...")

    def retrieveContext(
        self, query: str, retriever_top_k: int = settings.retriever.top_k
    ) -> list:
        base_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": retriever_top_k}
        )
        examples = base_retriever.invoke(query)
        rag_string = "Use the following examples to help you classify the query:"
        rag_string += "\n<examples>\n"
        for example in examples:
            rag_string += textwrap.dedent(
                f"""
            <example>
                <query>
                    "{example.metadata["question"]}"
                </query>
                <label>
                    {example.metadata["answer"]}
                </label>
            </example>
            """
            )
        rag_string += "\n</examples>"
        return rag_string

    def predict(
        self, query: str, retriever_top_k: int = settings.retriever.top_k
    ) -> str:
        deps = Deps(self.vector_db, query, retriever_top_k)
        result = self.agent.run_sync(query, deps=deps)
        return result.data


if __name__ == "__main__":
    classifier = TextClassifierRAG()
    query = "I'm confused about a charge on my recent auto insurance bill that's higher than my usual premium payment."
    # print(classifier.retrieveContext(query))
    print(classifier.predict(query))
