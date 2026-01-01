import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)
time.sleep(1)

embeddings_model = HuggingFaceEmbeddings(
    model_name=os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"token": os.getenv("HUGGING_FACE_TOKEN")},
)
res = embeddings_model.embed_query("king")
print(len(res))
print(res)

res = embeddings_model.embed_query("queen")
print(len(res))
print(res)

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
res = embeddings_model.embed_documents(sentences)
print(res)
