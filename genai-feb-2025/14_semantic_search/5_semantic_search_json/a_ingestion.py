from pathlib import Path
import time
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import JSONLoader

load_dotenv(override=True)

# read all the json files
kb_dir = os.path.expanduser(
    "~/Documents/genai-training-pydanticai/data/semantic-search/kb3"
)

documents = []
for file in Path(kb_dir).glob("*.json"):
    loader = JSONLoader(
        file_path=file,
        jq_schema=".[]",
        # content_key=".description",
        # content_key=".name + .description",
        content_key="[.name,.description] | @tsv",
        is_content_key_jq_parsable=True,
    )
    documents.extend(loader.load())
print(len(documents))
print(documents)


# create embeddings for all the chunks and store them in vector database
embeddings_model = HuggingFaceEmbeddings(
    model_name=os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"token": os.getenv("HUGGING_FACE_TOKEN")},
)

vector_db = FAISS.from_documents(
    documents=documents,
    embedding=embeddings_model,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)
vector_db_dir = os.path.expanduser(
    "~/Documents/genai-training-pydanticai/data/semantic-search/index/faiss5"
)
vector_db.save_local(folder_path=vector_db_dir)
