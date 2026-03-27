"""
index_builder.py
Builds PostgreSQL tables from source CSV files using pgvector embeddings.
"""

import os
import numpy as np
import pandas as pd
import logging

from .chatbot import Config
from .database import get_connection
from .embeddings import embed_texts

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Builds pgvector indices from CSV source data."""

    @staticmethod
    def indices_exist() -> bool:
        """Check if tables have data."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    for table in ["faq", "databases"]:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        if cur.fetchone()[0] == 0:
                            return False
            return True
        except Exception:
            return False

    @staticmethod
    def build_indices(api_key: str) -> None:
        """Build all indices from source CSV files into PostgreSQL."""
        project_root = os.path.dirname(os.path.dirname(__file__))
        faq_path = os.path.join(project_root, "data", "library_faq_clean.csv")
        db_path = os.path.join(project_root, "data", "Databases description.csv")

        # Build FAQ table
        logger.info("Building FAQ table...")
        faq = pd.read_csv(faq_path)
        faq["question"] = faq["question"].fillna("").astype(str).str.strip()
        faq["answer"] = faq["answer"].fillna("").astype(str).str.strip()
        faq = faq[(faq["question"] != "") & (faq["answer"] != "")].reset_index(drop=True)

        documents = faq["question"].tolist()
        logger.info(f"Generating embeddings for {len(documents)} FAQ entries...")
        embeddings = embed_texts(documents)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE faq")
                for i, (_, row) in enumerate(faq.iterrows()):
                    cur.execute(
                        "INSERT INTO faq (id, document, embedding, question, answer) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (id) DO UPDATE SET "
                        "document=EXCLUDED.document, embedding=EXCLUDED.embedding, "
                        "question=EXCLUDED.question, answer=EXCLUDED.answer",
                        (
                            f"faq_{i}",
                            row["question"],
                            np.array(embeddings[i]),
                            row["question"],
                            row["answer"],
                        ),
                    )
            conn.commit()
        logger.info(f"FAQ table: {len(faq)} entries")

        # Build Database table
        logger.info("Building Database table...")
        db = pd.read_csv(db_path)
        db["name"] = db["name"].fillna("").astype(str).str.strip()
        db["description"] = db["description"].fillna("").astype(str).str.strip()
        db = db[(db["name"] != "") & (db["description"] != "")].reset_index(drop=True)

        documents = (db["name"] + ". " + db["description"]).tolist()
        logger.info(f"Generating embeddings for {len(documents)} database entries...")
        embeddings = embed_texts(documents)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE databases")
                for i, (_, row) in enumerate(db.iterrows()):
                    cur.execute(
                        "INSERT INTO databases (id, document, embedding, name, description) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (id) DO UPDATE SET "
                        "document=EXCLUDED.document, embedding=EXCLUDED.embedding, "
                        "name=EXCLUDED.name, description=EXCLUDED.description",
                        (
                            f"db_{i}",
                            documents[i],
                            np.array(embeddings[i]),
                            row["name"],
                            row["description"],
                        ),
                    )
            conn.commit()
        logger.info(f"Database table: {len(db)} entries")
