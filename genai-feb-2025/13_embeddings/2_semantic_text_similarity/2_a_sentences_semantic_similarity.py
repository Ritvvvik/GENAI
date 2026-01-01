import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from plot_embeddings import plot_list_2d

load_dotenv(override=True)
time.sleep(1)

embeddings_model = SentenceTransformer(
    model_name_or_path=os.getenv("HF_EMBEDDINGS_MODEL")
)


# Two lists of sentences
sentences1 = [
    "The new movie is awesome",
    "The cat sits outside",
    "A man is playing guitar",
]

sentences2 = [
    "The dog plays in the garden",
    "The new movie is so great",
    "A woman watches TV",
]

# Compute embeddings for both lists
embeddings1 = embeddings_model.encode(
    sentences1, normalize_embeddings=True, show_progress_bar=True
)
embeddings2 = embeddings_model.encode(
    sentences2, normalize_embeddings=True, show_progress_bar=True
)

# Compute cosine similarities
similarities = embeddings_model.similarity(embeddings1, embeddings2)

# Output the pairs with their score
for i, sentence1 in enumerate(sentences1):
    print(sentence1)
    for j, sentence2 in enumerate(sentences2):
        print(f" - {sentence2: <30}: {similarities[i][j]:.4f}")
