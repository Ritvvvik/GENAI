from config_reader import *
from agent import text_summarizer_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
import logfire
import time
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.cluster import KMeans
import numpy as np


class TextSummarizer:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunker.chunk_size,
            chunk_overlap=settings.chunker.chunk_overlap,
        )
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedder.model,
            encode_kwargs={"normalize_embeddings": settings.embedder.normalize},
            model_kwargs={"token": settings.embedder.token},
        )
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    async def summarize_short_docs(self, text: str) -> str:
        res = await text_summarizer_agent.run(text)
        return res.data

    async def summarize_medium_docs(
        self, text: str = None, chunks: list[str] = None
    ) -> str:
        if chunks == None:
            chunks = self.text_splitter.split_text(text)
        tasks = [text_summarizer_agent.run(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        result_data = [result.data for result in results]
        result_data_combined = "\n".join(result_data)
        res = await text_summarizer_agent.run(result_data_combined)
        return res.data

    async def summarize_long_docs(self, text: str) -> str:
        chunks = self.text_splitter.split_text(text)
        # get embeddings for each chunk in a batch_size of 2
        batch_size = 2
        vectors = []
        for i in range(0, len(chunks), batch_size):
            vectors.extend(
                self.embeddings_model.embed_documents(chunks[i : i + batch_size])
            )
        nclusters = 5
        model = KMeans(n_init=10, n_clusters=nclusters, random_state=0).fit(vectors)
        closest_indices = []
        # find the closest vector to each cluster center
        for i in range(nclusters):
            distances = np.linalg.norm(vectors - model.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)
        selected_indices = sorted(closest_indices)
        selected_chunks = [chunks[idx] for idx in selected_indices]
        return await self.summarize_medium_docs(chunks=selected_chunks)

    async def summarize(self, text: str) -> str:
        if len(text) <= settings.docs.short_doc_threshold:
            return await self.summarize_short_docs(text)
        elif len(text) <= settings.docs.medium_doc_threshold:
            return await self.summarize_medium_docs(text)
        elif len(text) <= settings.docs.long_doc_threshold:
            return await self.summarize_long_docs(text)
        else:
            return "Not Supported"
