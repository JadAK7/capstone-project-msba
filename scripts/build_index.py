"""
build_index.py
Builds PostgreSQL embedding tables for FAQ and Database recommendation system.
Uses pgvector for vector storage and OpenAI for embedding generation.
"""

import os
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.database import init_db
from backend.embeddings import embed_texts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    _DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    FAQ_PATH = os.path.join(_DATA_DIR, "library_faq_clean.csv")
    DB_PATH = os.path.join(_DATA_DIR, "Databases description.csv")


def validate_dataframe(df, required_cols, name):
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


def build_faq_index():
    """Build FAQ table in PostgreSQL."""
    from backend.database import get_connection

    logger.info("Building FAQ table...")

    faq = pd.read_csv(Config.FAQ_PATH)
    faq, dropped = validate_dataframe(faq, ["question", "answer"], "FAQ")

    logger.info(f"Processing {len(faq)} FAQ entries")

    documents = faq["question"].tolist()
    logger.info("Generating FAQ embeddings...")
    embeddings = embed_texts(documents)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE faq")
            batch_size = 100
            for i in range(0, len(faq), batch_size):
                batch = faq.iloc[i:i + batch_size]
                for j, (_, row) in enumerate(batch.iterrows()):
                    idx = i + j
                    cur.execute(
                        "INSERT INTO faq (id, document, embedding, question, answer) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (
                            f"faq_{idx}",
                            row["question"],
                            np.array(embeddings[idx]),
                            row["question"],
                            row["answer"],
                        ),
                    )
                logger.info(f"  Inserted FAQ batch {i}-{i + len(batch)}")
        conn.commit()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM faq")
            count = cur.fetchone()[0]
    logger.info(f"FAQ table saved: {count} entries")


def build_database_index():
    """Build database recommendation table in PostgreSQL."""
    from backend.database import get_connection

    logger.info("Building Database table...")

    db = pd.read_csv(Config.DB_PATH)
    db, dropped = validate_dataframe(db, ["name", "description"], "Database")

    logger.info(f"Processing {len(db)} database entries")

    documents = (db["name"] + ". " + db["description"]).tolist()
    logger.info("Generating database embeddings...")
    embeddings = embed_texts(documents)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE databases")
            batch_size = 100
            for i in range(0, len(db), batch_size):
                batch = db.iloc[i:i + batch_size]
                for j, (_, row) in enumerate(batch.iterrows()):
                    idx = i + j
                    cur.execute(
                        "INSERT INTO databases (id, document, embedding, name, description) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (
                            f"db_{idx}",
                            documents[idx],
                            np.array(embeddings[idx]),
                            row["name"],
                            row["description"],
                        ),
                    )
                logger.info(f"  Inserted Database batch {i}-{i + len(batch)}")
        conn.commit()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM databases")
            count = cur.fetchone()[0]
    logger.info(f"Database table saved: {count} entries")


def verify_indices():
    """Verify that all tables were populated successfully."""
    from backend.database import get_connection

    for table in ["faq", "databases"]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                if count == 0:
                    raise ValueError(f"Table '{table}' is empty")
                logger.info(f"  Table '{table}': {count} entries")

    logger.info("All tables verified")


def main():
    """Main entry point for building indices."""
    try:
        logger.info("Starting index build process...")

        init_db()

        build_faq_index()
        build_database_index()
        verify_indices()

        logger.info("Index build completed successfully!")

    except Exception as e:
        logger.error(f"Index build failed: {e}")
        raise


if __name__ == "__main__":
    main()
