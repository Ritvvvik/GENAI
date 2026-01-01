import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)

embeddings_model = HuggingFaceEmbeddings(
    model_name=os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"token": os.getenv("HUGGING_FACE_TOKEN")},
)

vector_db_dir = os.path.expanduser(
    "~/Documents/genai-training-pydanticai/data/semantic-search/index/faiss1"
)
vector_db = FAISS.load_local(
    folder_path=vector_db_dir,
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True,
)

# Query sentences:
queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]

# Find the closest 5 chunks for each query based on similarity measure
for query in queries:
    base_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    hits = base_retriever.invoke(query)

    print("\nQuery:", query)
    print("Top 5 most similar chunks:")
    print(hits)
    for hit in hits:
        print(hit.page_content)
