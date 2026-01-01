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


sentences = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
    "technology is changing the facet of world",
    "technology also have side effects",
    "today is a sunny day",
    "i love coding",
    "coding is my passion",
    "the cat sat on the mat",
    "dogs are awesome",
    "the weather today is beautiful",
    "it is raining",
    "the weather is lovely today",
]
sentence_embeddings = embeddings_model.encode(
    sentences, normalize_embeddings=True, show_progress_bar=True
)
similarity_scores = embeddings_model.similarity(
    sentence_embeddings, sentence_embeddings
)
print(similarity_scores)
plot_list_2d(sentence_embeddings, sentences)
