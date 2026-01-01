import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)
time.sleep(1)

embeddings_model = SentenceTransformer(
    model_name_or_path=os.getenv("HF_EMBEDDINGS_MODEL")
)
res = embeddings_model.encode("king", normalize_embeddings=True)
print(res.shape)
print(res)

res = embeddings_model.encode("queen", normalize_embeddings=True)
print(res.shape)
print(res)

res = embeddings_model.encode("The weather is lovely today.", normalize_embeddings=True)
print(res.shape)
print(res)

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
res = embeddings_model.encode(
    sentences, normalize_embeddings=True, show_progress_bar=True
)
print(res.shape)
