"""Step 4: store chunk embeddings in Qdrant Cloud so we can search them later."""

import os

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from hr_assistant import config

from hr_assistant.embeddings import get_embeddings_model

from hr_assistant.logger import get_logger

logger = get_logger(__name__)

# build_vector_store

def build_vector_store(chunks):
    """Embed every chunk and upload it into a Qdrant Cloud collection..."""

    logger.info("Embedding %d chunk(s) and uploading to Qdrant collection...",
                len(chunks),
                config.QDRANT_COLLECTION_NAME)

    embeddings_model = get_embeddings_model()

    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding = embeddings_model,
        url = config.QDRANT_URL,
        api_key = config.QDRANT_API_KEY,
        collection_name = config.QDRANT_COLLECTION_NAME

    )

    logger.info("Uploaded to Qdrant collection '%s'",config.QDRANT_COLLECTION_NAME)


    return vector_store



def load_vector_store():
    """
    Connect to an existing Qdrant Cloud collection.
    """

    logger.info(
        "Connecting to Qdrant collection '%s'",
        config.QDRANT_COLLECTION_NAME
    )

    embeddings_model = get_embeddings_model()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME
    )

    logger.info("Connected to Qdrant collection successfully.")

    return vector_store


def vector_store_exists():
    """Check if a qdrant store already exists"""

    client = QdrantClient(
        url= config.QDRANT_URL,
        api_key= config.QDRANT_API_KEY
    )

    return client.collection_exists(config.QDRANT_COLLECTION_NAME)

def get_retriever(vector_store,k: int = config.TOP_K_RESULTS):
    """Turn a vector store into retriever that returns top-k matching chunks."""

    logger.info("Creating a retriever with top_k=%d",k)
    
    return vector_store.as_retriever(search_kwargs={"k":k})

