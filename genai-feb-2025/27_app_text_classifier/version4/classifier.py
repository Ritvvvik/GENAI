from pathlib import Path
import os
from langchain_huggingface import HuggingFaceEmbeddings
from config_reader import *
import joblib
from sklearn.linear_model import LogisticRegression
import json
import pandas as pd
import numpy as np


class TextClassifier:
    def __init__(self):
        src_dir = settings.file_paths.src_dir
        self.train_file = os.path.join(src_dir, settings.file_paths.train_file)
        self.embeddings_dir = os.path.join(src_dir, settings.file_paths.embeddings_dir)
        self.model_dir = os.path.join(src_dir, settings.file_paths.model_dir)

        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedder.model,
            encode_kwargs={"normalize_embeddings": settings.embedder.normalize},
            model_kwargs={"token": settings.embedder.token},
        )
        if not os.path.exists(self.embeddings_dir):
            self.generate_embeddings()
        if not os.path.exists(self.model_dir):
            self.build_model()

    def generate_embeddings(self):
        print(">> Begin: Generating Embeddings...")
        with open(self.train_file, "r") as f:
            eval_data = json.load(f)
        items = []
        for item in eval_data:
            embedding = self.embeddings_model.embed_query(item["question"])
            items.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "embedding": embedding,
                }
            )
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)
        with open(os.path.join(self.embeddings_dir, "train_embeddings.json"), "w") as f:
            json.dump(items, f, indent=2)
        print(">> End: Generating Embeddings...")

    def build_model(self):
        print(">> Begin: Model Training...")
        with open(os.path.join(self.embeddings_dir, "train_embeddings.json"), "r") as f:
            eval_data = json.load(f)
        df = pd.json_normalize(eval_data)
        clf = LogisticRegression(random_state=42)
        y = df["answer"]
        X = pd.DataFrame(df["embedding"].to_list())
        clf.fit(X, y)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        joblib.dump(clf, os.path.join(self.model_dir, "model.pkl"))
        print(">> End: Model Training...")

    def predict(self, query: str) -> str:
        query_embedding = np.array(self.embeddings_model.embed_query(query))
        query_embedding = query_embedding.reshape(1, -1)
        clf = joblib.load(os.path.join(self.model_dir, "model.pkl"))
        result = clf.predict(query_embedding)
        return result[0]


if __name__ == "__main__":
    classifier = TextClassifier()
    query = "I'm confused about a charge on my recent auto insurance bill that's higher than my usual premium payment."
    print(classifier.predict(query))
