"""
build_index.py
Builds ChromaDB embedding collections for FAQ and Database recommendation system.
Uses ChromaDB's built-in OpenAI embedding function for embedding generation.
"""

import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import List, Tuple
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    FAQ_PATH = "library_faq_clean.csv"
    DB_PATH = "Databases description.csv"
    CHROMA_DIR = "./chroma_db"
    EMBEDDING_MODEL = "text-embedding-3-small"

    FAQ_COLLECTION = "faq"
    DB_COLLECTION = "databases"


def get_chroma_client():
    """Create a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=Config.CHROMA_DIR)


def get_embedding_function():
    """Create OpenAI embedding function for ChromaDB."""
    return OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=Config.EMBEDDING_MODEL,
    )


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    name: str
) -> Tuple[pd.DataFrame, int]:
    """Validate and clean dataframe."""
    initial_len = len(df)

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name}: Missing columns {missing_cols}")

    for col in required_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    mask = df[required_cols].apply(lambda x: x != "").all(axis=1)
    df_clean = df[mask].reset_index(drop=True)

    dropped = initial_len - len(df_clean)
    if dropped > 0:
        logger.warning(f"{name}: Dropped {dropped} rows with empty fields")

    return df_clean, dropped


def build_faq_index(client, embedding_fn) -> None:
    """Build and save FAQ collection in ChromaDB."""
    logger.info("Building FAQ collection...")

    faq = pd.read_csv(Config.FAQ_PATH)
    faq, dropped = validate_dataframe(faq, ["question", "answer"], "FAQ")

    logger.info(f"Processing {len(faq)} FAQ entries")

    # Delete existing collection and recreate
    try:
        client.delete_collection(Config.FAQ_COLLECTION)
    except ValueError:
        pass

    collection = client.create_collection(
        name=Config.FAQ_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches (ChromaDB has a batch limit)
    batch_size = 100
    for i in range(0, len(faq), batch_size):
        batch = faq.iloc[i:i + batch_size]
        collection.upsert(
            ids=[f"faq_{j}" for j in range(i, i + len(batch))],
            documents=batch["question"].tolist(),
            metadatas=[
                {"question": row["question"], "answer": row["answer"]}
                for _, row in batch.iterrows()
            ],
        )
        logger.info(f"  Upserted FAQ batch {i}-{i + len(batch)}")

    logger.info(f"FAQ collection saved: {collection.count()} entries")


def build_database_index(client, embedding_fn) -> None:
    """Build and save database recommendation collection in ChromaDB."""
    logger.info("Building Database collection...")

    db = pd.read_csv(Config.DB_PATH)
    db, dropped = validate_dataframe(db, ["name", "description"], "Database")

    logger.info(f"Processing {len(db)} database entries")

    try:
        client.delete_collection(Config.DB_COLLECTION)
    except ValueError:
        pass

    collection = client.create_collection(
        name=Config.DB_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(db), batch_size):
        batch = db.iloc[i:i + batch_size]
        documents = (batch["name"] + ". " + batch["description"]).tolist()
        collection.upsert(
            ids=[f"db_{j}" for j in range(i, i + len(batch))],
            documents=documents,
            metadatas=[
                {"name": row["name"], "description": row["description"]}
                for _, row in batch.iterrows()
            ],
        )
        logger.info(f"  Upserted Database batch {i}-{i + len(batch)}")

    logger.info(f"Database collection saved: {collection.count()} entries")


def verify_indices(client) -> None:
    """Verify that all ChromaDB collections were created successfully."""
    for name in [Config.FAQ_COLLECTION, Config.DB_COLLECTION]:
        try:
            col = client.get_collection(name)
            count = col.count()
            if count == 0:
                raise ValueError(f"Collection '{name}' is empty")
            logger.info(f"  Collection '{name}': {count} entries")
        except ValueError:
            raise FileNotFoundError(f"Collection '{name}' not found")

    logger.info("All collections verified")


def main():
    """Main entry point for building indices."""
    try:
        logger.info("Starting index build process...")

        chroma_client = get_chroma_client()
        embedding_fn = get_embedding_function()

        build_faq_index(chroma_client, embedding_fn)
        build_database_index(chroma_client, embedding_fn)
        verify_indices(chroma_client)

        logger.info("Index build completed successfully!")

    except Exception as e:
        logger.error(f"Index build failed: {e}")
        raise


if __name__ == "__main__":
    main()
