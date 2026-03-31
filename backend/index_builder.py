"""
index_builder.py
Builds PostgreSQL tables from source CSV files and scraped data.
Uses the new pipeline: extract → chunk → embed → store.
"""

import os
import numpy as np
import pandas as pd
import logging

from .chatbot import Config
from .database import get_connection
from .embeddings import embed_texts
from .content_extractor import extract_pages_batch
from .chunker import chunk_pages

logger = logging.getLogger(__name__)

# Site-specific noise selectors passed to the generic extractor.
# Keeps site config out of the extractor itself.
_AUB_NOISE_SELECTORS = [
    "#useful-tools", ".useful-tools",
    "#sideNavBox", ".ms-quickLaunch", ".ms-nav",
    "#DeltaTopNavigation", ".ms-topNavContainer",
    "#s4-breadcrumb",
]


class IndexBuilder:
    """Builds pgvector indices from CSV source data and scraped content."""

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

        # Embed the full Q+A pair for better semantic matching
        documents = (faq["question"] + " " + faq["answer"]).tolist()
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
                            documents[i],
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

    @staticmethod
    def build_chunks_from_scraped(scraped_data: list) -> int:
        """Process scraped data through the full pipeline and store chunks.

        Pipeline: raw pages → extract → chunk → embed → store.
        Returns the number of chunks created.
        """
        logger.info(f"Processing {len(scraped_data)} scraped pages through pipeline...")

        # Step 1: Extract structured blocks from raw pages
        extracted_pages = extract_pages_batch(
            scraped_data,
            extra_noise_selectors=_AUB_NOISE_SELECTORS,
        )
        logger.info(f"Step 1 (extract): {len(extracted_pages)} pages with content")

        # Step 2: Chunk pages into embeddable pieces
        chunks = chunk_pages(extracted_pages)
        logger.info(f"Step 2 (chunk): {len(chunks)} chunks created")

        if not chunks:
            logger.warning("No chunks produced from scraped data")
            return 0

        # Step 3: Generate embeddings for all chunks
        chunk_texts = [c["chunk_text"] for c in chunks]
        logger.info(f"Step 3 (embed): Generating embeddings for {len(chunk_texts)} chunks...")
        embeddings = embed_texts(chunk_texts)

        # Step 4: Store in document_chunks table (truncate first to clear old data)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE document_chunks")
                batch_size = 50
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    for j, chunk in enumerate(batch):
                        idx = i + j
                        cur.execute(
                            "INSERT INTO document_chunks "
                            "(id, chunk_text, embedding, page_url, page_title, "
                            " section_title, page_type, chunk_index) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                            (
                                f"chunk_{idx}",
                                chunk["chunk_text"],
                                np.array(embeddings[idx]),
                                chunk["page_url"],
                                chunk["page_title"],
                                chunk.get("section_title", ""),
                                chunk.get("page_type", "general"),
                                chunk.get("chunk_index", 0),
                            ),
                        )
            conn.commit()

        logger.info(f"Step 4 (store): {len(chunks)} chunks stored in document_chunks table")

        # Also store in legacy library_pages table for backward compatibility
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE library_pages")
                for i, page in enumerate(extracted_pages):
                    # Reconstruct clean text from blocks
                    clean_text = "\n\n".join(
                        b["content"] for b in page.get("blocks", [])
                    )
                    doc_text = f"{page.get('page_title', '')}\n\n{clean_text}"
                    page_url = page.get("url", "")
                    # Find an embedding from one of this page's chunks
                    page_chunks = [
                        idx for idx, c in enumerate(chunks)
                        if c["page_url"] == page_url
                    ]
                    emb_idx = page_chunks[0] if page_chunks else min(i, len(embeddings) - 1)
                    if emb_idx < len(embeddings):
                        cur.execute(
                            "INSERT INTO library_pages (id, document, embedding, url, title, content) "
                            "VALUES (%s, %s, %s, %s, %s, %s)",
                            (
                                f"page_{i}",
                                doc_text,
                                np.array(embeddings[emb_idx]),
                                page_url,
                                page.get("page_title", ""),
                                clean_text[:5000],
                            ),
                        )
            conn.commit()

        # Ensure full-text search index exists
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_chunks_fts
                        ON document_chunks
                        USING gin (to_tsvector('english', chunk_text))
                    """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to create FTS index on document_chunks: {e}")

        return len(chunks)
