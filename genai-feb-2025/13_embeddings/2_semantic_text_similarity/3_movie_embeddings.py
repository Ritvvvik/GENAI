import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from plot_embeddings import plot_df_2d, plot_df_3d
import pandas as pd


def embed_movies(dir: str, source_file: str, target_file: str):
    source_path = os.path.join(dir, source_file)
    target_path = os.path.join(dir, target_file)
    if not os.path.exists(target_path):
        embeddings_model = SentenceTransformer(
            model_name_or_path=os.getenv("HF_EMBEDDINGS_MODEL")
        )
        movies_df = pd.read_csv(source_path)
        movie_plots = (
            movies_df["short_plot"] + "\n" + movies_df["long_plot"]
        ).to_list()
        res = embeddings_model.encode(
            movie_plots, normalize_embeddings=True, show_progress_bar=True
        )
        print(res.shape)
        res_list = []
        for e in res:
            res_list.append(e.tolist())
        movies_df["embedding"] = res_list
        movies_df.to_csv(target_path, index=False)


if __name__ == "__main__":
    load_dotenv(override=True)
    time.sleep(1)

    dir = os.path.expanduser("~/Documents/genai-training-pydanticai/data/embeddings")
    source_file = "movies.csv"
    target_file = "movies_embeddings.csv"
    embed_movies(dir, source_file, target_file)
    plot_df_2d(dir, target_file)
