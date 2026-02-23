"""
index_builder.py
Builds ChromaDB collections from source CSV files.
"""

import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import logging

from .chatbot import Config

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Builds ChromaDB indices if they don't exist."""

    @staticmethod
    def indices_exist() -> bool:
        """Check if ChromaDB collections exist with data."""
        try:
            client = chromadb.PersistentClient(path=Config.CHROMA_DIR)
            for name in [Config.FAQ_COLLECTION, Config.DB_COLLECTION]:
                col = client.get_collection(name)
                if col.count() == 0:
                    return False
            return True
        except (ValueError, Exception):
            return False

    @staticmethod
    def build_indices(api_key: str) -> None:
        """Build all indices from source CSV files into ChromaDB."""
        project_root = os.path.dirname(os.path.dirname(__file__))
        faq_path = os.path.join(project_root, "library_faq_clean.csv")
        db_path = os.path.join(project_root, "Databases description.csv")

        embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=Config.EMBEDDING_MODEL,
        )
        client = chromadb.PersistentClient(path=Config.CHROMA_DIR)

        # Build FAQ collection
        logger.info("Building FAQ collection...")
        faq = pd.read_csv(faq_path)
        faq["question"] = faq["question"].fillna("").astype(str).str.strip()
        faq["answer"] = faq["answer"].fillna("").astype(str).str.strip()
        faq = faq[(faq["question"] != "") & (faq["answer"] != "")].reset_index(drop=True)

        try:
            client.delete_collection(Config.FAQ_COLLECTION)
        except (ValueError, Exception):
            pass

        faq_col = client.create_collection(
            name=Config.FAQ_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 100
        for i in range(0, len(faq), batch_size):
            batch = faq.iloc[i:i + batch_size]
            faq_col.upsert(
                ids=[f"faq_{j}" for j in range(i, i + len(batch))],
                documents=batch["question"].tolist(),
                metadatas=[
                    {"question": row["question"], "answer": row["answer"]}
                    for _, row in batch.iterrows()
                ],
            )
        logger.info(f"FAQ collection: {faq_col.count()} entries")

        # Build Database collection
        logger.info("Building Database collection...")
        db = pd.read_csv(db_path)
        db["name"] = db["name"].fillna("").astype(str).str.strip()
        db["description"] = db["description"].fillna("").astype(str).str.strip()
        db = db[(db["name"] != "") & (db["description"] != "")].reset_index(drop=True)

        try:
            client.delete_collection(Config.DB_COLLECTION)
        except (ValueError, Exception):
            pass

        db_col = client.create_collection(
            name=Config.DB_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        for i in range(0, len(db), batch_size):
            batch = db.iloc[i:i + batch_size]
            documents = (batch["name"] + ". " + batch["description"]).tolist()
            db_col.upsert(
                ids=[f"db_{j}" for j in range(i, i + len(batch))],
                documents=documents,
                metadatas=[
                    {"name": row["name"], "description": row["description"]}
                    for _, row in batch.iterrows()
                ],
            )
        logger.info(f"Database collection: {db_col.count()} entries")
