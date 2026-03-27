"""
database.py
PostgreSQL connection pool and schema initialization for pgvector.
"""
from __future__ import annotations

import os
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)

_pool: pool.ThreadedConnectionPool | None = None

DEFAULT_DATABASE_URL = "postgresql://aub_library:aub_library_pass@localhost:5432/aub_library"

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS faq (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS databases (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS library_pages (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_faq_embedding ON faq USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_databases_embedding ON databases USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_library_pages_embedding ON library_pages USING hnsw (embedding vector_cosine_ops);

-- Chat conversations: every question + answer the chatbot produces
CREATE TABLE IF NOT EXISTS chat_conversations (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    chosen_source TEXT,
    faq_top_score REAL DEFAULT 0,
    db_top_score REAL DEFAULT 0,
    library_top_score REAL DEFAULT 0,
    response_time_ms REAL DEFAULT 0,
    retrieved_chunks JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_created ON chat_conversations (created_at DESC);

-- Admin feedback on conversations
CREATE TABLE IF NOT EXISTS chat_feedback (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),  -- -1 = thumbs down, 1 = thumbs up
    corrected_answer TEXT,  -- admin-provided correct answer (for thumbs down)
    comment TEXT,           -- optional admin comment
    embedding vector(1536), -- embedding of the original query (for similarity lookup)
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_conversation ON chat_feedback (conversation_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON chat_feedback (rating);
CREATE INDEX IF NOT EXISTS idx_feedback_embedding ON chat_feedback USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;
"""


def init_db() -> None:
    """Create the connection pool and ensure schema exists."""
    global _pool
    if _pool is not None:
        return

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    logger.info("Connecting to PostgreSQL...")
    _pool = pool.ThreadedConnectionPool(minconn=2, maxconn=10, dsn=database_url)

    # Bootstrap: create the vector extension before registering the type.
    # get_connection() calls register_vector(), which fails if the extension
    # doesn't exist yet, so we use a raw connection for the initial setup.
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        conn.commit()
    finally:
        _pool.putconn(conn)

    logger.info("Database schema initialized")


def close_db() -> None:
    """Close all pooled connections."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("Database connections closed")


@contextmanager
def get_connection():
    """Yield a pooled connection (auto-returned on exit).

    Registers the pgvector type on each connection so numpy arrays
    and Python lists are transparently serialized to vector columns.
    """
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db() first.")

    conn = _pool.getconn()
    try:
        register_vector(conn)
        yield conn
    finally:
        _pool.putconn(conn)
