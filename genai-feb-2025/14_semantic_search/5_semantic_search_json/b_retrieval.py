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

# load the vector database
vector_db_dir = os.path.expanduser(
    "~/Documents/genai-training-pydanticai/data/semantic-search/index/faiss5"
)
vector_db = FAISS.load_local(
    folder_path=vector_db_dir,
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True,
)

# Query sentences:
queries = [
    "Sit on the ground with a bench behind you",
    "Step forward with one leg and lower your body until both knees are bent",
    "Lie face down on a leg curl machine",
]

for query in queries:
    base_retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    hits = base_retriever.invoke(query)

    print("\nQuery:", query)
    print("Top 2 most similar chunks in corpus/knowledge base:")
    # print(hits)
    for hit in hits:
        print(hit.page_content, f"\nSource:{hit.metadata.get("source")}")
        print()
