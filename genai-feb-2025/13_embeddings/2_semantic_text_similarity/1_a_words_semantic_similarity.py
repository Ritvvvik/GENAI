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
words1 = [
    "rice",
    "king",
    "missile",
]

words2 = ["emperor", "queen", "war"]

# Compute embeddings for both lists
embeddings1 = embeddings_model.encode(
    words1, normalize_embeddings=True, show_progress_bar=True
)
embeddings2 = embeddings_model.encode(
    words2, normalize_embeddings=True, show_progress_bar=True
)

# Compute cosine similarities
similarities = embeddings_model.similarity(embeddings1, embeddings2)

# Output the pairs with their score
for i, word1 in enumerate(words1):
    print(word1)
    for j, word2 in enumerate(words2):
        print(f" - {word2: <30}: {similarities[i][j]:.4f}")
