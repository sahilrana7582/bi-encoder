import numpy as np
from langchain_openai import OpenAIEmbeddings

from src.llm import Settings


def load_model(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.llm_api_key,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_seconds,
    )


def load_documents(path):
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    documents = [
        doc.strip()
        for doc in content.split("\n\n")
        if doc.strip()
    ]

    return documents


def embed_query(model, query):
    return np.array(
        model.embed_query(query),
        dtype=np.float32,
    )


def create_embeddings(model, documents):
    embeddings = np.array(
        model.embed_documents(documents),
        dtype=np.float32,
    )

    return embeddings
