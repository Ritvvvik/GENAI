import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)
time.sleep(1)

embeddings_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
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
