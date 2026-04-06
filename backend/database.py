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

-- Custom notes: free-form text entries added by admins
CREATE TABLE IF NOT EXISTS custom_notes (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    label TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Semantic chunks from processed documents (for upgraded RAG pipeline)
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    page_url TEXT NOT NULL,
    page_title TEXT NOT NULL,
    section_title TEXT NOT NULL DEFAULT '',
    page_type TEXT NOT NULL DEFAULT 'general',
    chunk_index INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_faq_embedding ON faq USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_databases_embedding ON databases USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_library_pages_embedding ON library_pages USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_custom_notes_embedding ON custom_notes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_custom_notes_fts ON custom_notes USING gin (to_tsvector('english', document));
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON document_chunks USING gin (to_tsvector('english', chunk_text));
CREATE INDEX IF NOT EXISTS idx_chunks_page_type ON document_chunks (page_type);

-- Chat conversations: every question + answer the chatbot produces
-- Also stores all analytics/debugging fields (replaces chat_logs.json file)
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
    -- Analytics fields
    cache_hit BOOLEAN DEFAULT FALSE,
    query_word_count INTEGER DEFAULT 0,
    keyword_intent_fired BOOLEAN DEFAULT FALSE,
    sources_above_threshold INTEGER DEFAULT 0,
    top_faq_question TEXT DEFAULT '',
    top_db_name TEXT DEFAULT '',
    -- Hallucination debugging
    draft_answer TEXT DEFAULT '',
    verified_answer TEXT DEFAULT '',
    removed_claims JSONB DEFAULT '[]'::jsonb,
    context_sent_to_llm TEXT DEFAULT '',
    verification_passed BOOLEAN DEFAULT TRUE,
    -- Safety / guard fields
    guard_injection_detected BOOLEAN DEFAULT FALSE,
    guard_out_of_scope BOOLEAN DEFAULT FALSE,
    guard_refusal_reason TEXT DEFAULT '',
    guard_matched_patterns JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_created ON chat_conversations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_language ON chat_conversations (language);
CREATE INDEX IF NOT EXISTS idx_conversations_source ON chat_conversations (chosen_source);

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


_MIGRATION_COLUMNS = [
    ("cache_hit", "BOOLEAN DEFAULT FALSE"),
    ("query_word_count", "INTEGER DEFAULT 0"),
    ("keyword_intent_fired", "BOOLEAN DEFAULT FALSE"),
    ("sources_above_threshold", "INTEGER DEFAULT 0"),
    ("top_faq_question", "TEXT DEFAULT ''"),
    ("top_db_name", "TEXT DEFAULT ''"),
    ("draft_answer", "TEXT DEFAULT ''"),
    ("verified_answer", "TEXT DEFAULT ''"),
    ("removed_claims", "JSONB DEFAULT '[]'::jsonb"),
    ("context_sent_to_llm", "TEXT DEFAULT ''"),
    ("verification_passed", "BOOLEAN DEFAULT TRUE"),
    ("guard_injection_detected", "BOOLEAN DEFAULT FALSE"),
    ("guard_out_of_scope", "BOOLEAN DEFAULT FALSE"),
    ("guard_refusal_reason", "TEXT DEFAULT ''"),
    ("guard_matched_patterns", "JSONB DEFAULT '[]'::jsonb"),
]


def _migrate_chat_conversations(conn) -> None:
    """Add any missing columns to chat_conversations (safe to run repeatedly)."""
    try:
        with conn.cursor() as cur:
            for col_name, col_def in _MIGRATION_COLUMNS:
                cur.execute(
                    f"ALTER TABLE chat_conversations ADD COLUMN IF NOT EXISTS {col_name} {col_def}"
                )
            # Migrate chat_feedback: add source column (user vs admin)
            cur.execute(
                "ALTER TABLE chat_feedback "
                "ADD COLUMN IF NOT EXISTS feedback_source TEXT DEFAULT 'admin'"
            )
        conn.commit()
    except Exception as e:
        logger.warning(f"Migration of chat_conversations columns failed (non-critical): {e}")
        try:
            conn.rollback()
        except Exception:
            pass


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

    # Migrate: add new analytics columns to existing chat_conversations table.
    # ALTER TABLE ... ADD COLUMN IF NOT EXISTS is safe to run repeatedly.
    _migrate_chat_conversations(conn)

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
