# pip install plotly
import time
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from ast import literal_eval
import os
from langchain_huggingface import HuggingFaceEmbeddings


def plot_df_2d(dir: str, file: str):
    df = pd.read_csv(os.path.join(dir, file))
    embeddings = np.array(df["embedding"].apply(literal_eval).tolist())

    tsne = TSNE(perplexity=15, n_components=2)
    scaler = StandardScaler()
    embeddings_2d = tsne.fit_transform(embeddings)
    embeddings_2d = scaler.fit_transform(embeddings_2d)

    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="title",
        hover_data={"title": True, "x": False, "y": False},
        title="Similarity based on Embeddings",
    )
    fig.update_traces(textposition="top center", textfont=dict(size=6))
    fig.show()


def plot_df_3d(dir: str, file: str):
    df = pd.read_csv(os.path.join(dir, file))
    embeddings = np.array(df["embedding"].apply(literal_eval).tolist())

    tsne = TSNE(perplexity=15, n_components=3)
    scaler = StandardScaler()
    embeddings_3d = tsne.fit_transform(embeddings)
    embeddings_3d = scaler.fit_transform(embeddings_3d)

    df["x"] = embeddings_3d[:, 0]
    df["y"] = embeddings_3d[:, 1]
    df["z"] = embeddings_3d[:, 2]

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        text="title",
        hover_data={"title": True, "x": False, "y": False, "z": False},
        title="Similarity based on Embeddings",
    )
    fig.update_traces(textposition="top center", textfont=dict(size=6))
    fig.show()


def plot_list_2d(embeddings, titles):
    embeddings = np.array(embeddings)

    tsne = TSNE(perplexity=15, n_components=2)
    scaler = StandardScaler()
    embeddings_2d = tsne.fit_transform(embeddings)
    embeddings_2d = scaler.fit_transform(embeddings_2d)

    df = pd.DataFrame()
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]
    df["title"] = titles

    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="title",
        hover_data={"title": True, "x": False, "y": False},
        title="Similarity based on Embeddings",
    )
    fig.update_traces(textposition="top center", textfont=dict(size=6))
    fig.show()


def plot_list_3d(embeddings, titles):
    embeddings = np.array(embeddings)

    tsne = TSNE(perplexity=15, n_components=3)
    scaler = StandardScaler()
    embeddings_3d = tsne.fit_transform(embeddings)
    embeddings_3d = scaler.fit_transform(embeddings_3d)

    df = pd.DataFrame()
    df["x"] = embeddings_3d[:, 0]
    df["y"] = embeddings_3d[:, 1]
    df["z"] = embeddings_3d[:, 2]
    df["title"] = titles

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        text="title",
        hover_data={"title": True, "x": False, "y": False, "z": False},
        title="Similarity based on Embeddings",
    )
    fig.update_traces(textposition="top center", textfont=dict(size=6))
    fig.show()
