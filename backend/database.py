"""
database.py
PostgreSQL connection pool and schema initialization for pgvector.

Embedding dimension is read from embeddings.py (which reads the
OPENAI_EMBEDDING_MODEL env var) and baked into the schema SQL at import time.
If the running DB has a different dimension, _migrate_embedding_dimensions()
detects it on startup and automatically resizes the columns.
"""
from __future__ import annotations

import os
import re
import threading
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)


class DatabaseUnavailableError(Exception):
    """Raised when the database connection cannot be acquired."""
    pass

_pool: pool.ThreadedConnectionPool | None = None

# Tracks connection objects on which register_vector() has already run.
# pgvector adapter registration is per-connection state, so we only need
# to do it once per pooled connection rather than on every checkout.
_vector_registered: set[int] = set()
_vector_registered_lock = threading.Lock()

DEFAULT_DATABASE_URL = "postgresql://aub_library:aub_library_pass@localhost:5433/aub_library"

# Import the dimension that was resolved from the env var in embeddings.py.
# embeddings.py only imports from llm_client.py, so there is no circular dep.
from .embeddings import EMBEDDING_DIM as _DIM  # noqa: E402

_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS faq (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding vector({_DIM}) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS databases (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding vector({_DIM}) NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS library_pages (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding vector({_DIM}) NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Custom notes: free-form text entries added by admins
CREATE TABLE IF NOT EXISTS custom_notes (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding vector({_DIM}) NOT NULL,
    label TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Semantic chunks from processed documents (for upgraded RAG pipeline)
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding vector({_DIM}) NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_faq_fts ON faq USING gin (to_tsvector('english', document));
CREATE INDEX IF NOT EXISTS idx_databases_fts ON databases USING gin (to_tsvector('english', document));
CREATE INDEX IF NOT EXISTS idx_library_pages_fts ON library_pages USING gin (to_tsvector('english', document));
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
    embedding vector({_DIM}), -- embedding of the original query (for similarity lookup)
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_conversation ON chat_feedback (conversation_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON chat_feedback (rating);
CREATE INDEX IF NOT EXISTS idx_feedback_embedding ON chat_feedback USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

-- Escalations: student requests forwarded to a librarian
CREATE TABLE IF NOT EXISTS escalations (
    id SERIAL PRIMARY KEY,
    student_email TEXT NOT NULL,
    student_name TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, answered, closed
    admin_response TEXT,
    response_sent_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations (status);
CREATE INDEX IF NOT EXISTS idx_escalations_created ON escalations (created_at DESC);
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
    ("stage_latencies", "JSONB DEFAULT '{}'::jsonb"),
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


# ---------------------------------------------------------------------------
# Embedding dimension migration
# ---------------------------------------------------------------------------

# Content tables: must be truncated when dimension changes (old vectors unusable)
_CONTENT_EMBEDDING_TABLES = [
    "faq",
    "databases",
    "library_pages",
    "document_chunks",
    "custom_notes",
]

# HNSW indexes on those tables (dropped automatically with the column, but we
# recreate them explicitly after the migration so queries work immediately)
_CONTENT_HNSW_INDEXES = {
    "faq":            "idx_faq_embedding",
    "databases":      "idx_databases_embedding",
    "library_pages":  "idx_library_pages_embedding",
    "document_chunks":"idx_chunks_embedding",
    "custom_notes":   "idx_custom_notes_embedding",
}


def _get_column_vector_dim(cur, table: str, column: str = "embedding") -> int | None:
    """Return the current vector(N) dimension for a column, or None if absent."""
    cur.execute(
        """
        SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = %s
          AND a.attname = %s
          AND a.attnum > 0
          AND NOT a.attisdropped
        """,
        (table, column),
    )
    row = cur.fetchone()
    if not row:
        return None
    m = re.match(r"vector\((\d+)\)", row[0] or "")
    return int(m.group(1)) if m else None


def _migrate_embedding_dimensions(conn, target_dim: int) -> None:
    """Detect and fix an embedding dimension mismatch between the DB and the
    currently configured OPENAI_EMBEDDING_MODEL.

    When the dimension changes:
      - Content tables (faq, databases, etc.) are TRUNCATED and their
        embedding column is replaced.  All data must be rebuilt via
        `python scripts/build_index.py`.
      - chat_feedback rows are PRESERVED; only the embedding column is
        replaced (values become NULL — they are regenerated when feedback
        is re-submitted).

    This runs automatically in init_db() and is a no-op when dimensions match.
    """
    from .embeddings import EMBEDDING_MODEL

    try:
        with conn.cursor() as cur:
            current_dim = _get_column_vector_dim(cur, "faq")

        if current_dim is None or current_dim == target_dim:
            return  # fresh install or dimensions already match

        logger.warning(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Embedding dimension mismatch detected!\n"
            "  Database has: vector(%d)\n"
            "  Configured model: %s → vector(%d)\n"
            "  Action: content tables will be TRUNCATED and embedding columns\n"
            "          resized.  Run 'python scripts/build_index.py' afterwards\n"
            "          to rebuild all vector indices.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            current_dim, EMBEDDING_MODEL, target_dim,
        )

        with conn.cursor() as cur:
            # 1. Content tables: truncate + resize embedding column
            for table in _CONTENT_EMBEDDING_TABLES:
                dim = _get_column_vector_dim(cur, table)
                if dim is None or dim == target_dim:
                    continue
                logger.info("Resizing %s.embedding  %d → %d  (data cleared)", table, dim, target_dim)
                cur.execute(f"TRUNCATE TABLE {table}")
                cur.execute(f"ALTER TABLE {table} DROP COLUMN embedding")
                cur.execute(f"ALTER TABLE {table} ADD COLUMN embedding vector({target_dim}) NOT NULL")

            # 2. chat_feedback: preserve rows, resize nullable embedding column
            fb_dim = _get_column_vector_dim(cur, "chat_feedback")
            if fb_dim is not None and fb_dim != target_dim:
                logger.info(
                    "Resizing chat_feedback.embedding  %d → %d  (rows preserved, embeddings cleared)",
                    fb_dim, target_dim,
                )
                cur.execute("ALTER TABLE chat_feedback DROP COLUMN embedding")
                cur.execute(f"ALTER TABLE chat_feedback ADD COLUMN embedding vector({target_dim})")

            # 3. Recreate HNSW indexes on content tables
            for table, idx_name in _CONTENT_HNSW_INDEXES.items():
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} "
                    f"USING hnsw (embedding vector_cosine_ops)"
                )

            # 4. Recreate conditional HNSW index on chat_feedback
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_embedding "
                "ON chat_feedback USING hnsw (embedding vector_cosine_ops) "
                "WHERE embedding IS NOT NULL"
            )

        conn.commit()
        logger.warning(
            "Embedding dimension migration complete.  "
            "Run 'python scripts/build_index.py' to rebuild all indices."
        )

    except Exception as e:
        logger.error("Embedding dimension migration failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        raise


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

        # Detect and fix embedding dimension mismatches before anything else runs.
        _migrate_embedding_dimensions(conn, _DIM)

    finally:
        _pool.putconn(conn)

    # Add any missing analytics/guard columns to chat_conversations.
    _migrate_chat_conversations(conn)

    logger.info("Database schema initialized (embedding dim=%d)", _DIM)


def close_db() -> None:
    """Close all pooled connections."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        with _vector_registered_lock:
            _vector_registered.clear()
        logger.info("Database connections closed")


@contextmanager
def get_connection():
    """Yield a pooled connection (auto-returned on exit).

    Registers the pgvector type on each connection so numpy arrays
    and Python lists are transparently serialized to vector columns.

    Raises DatabaseUnavailableError if the pool is not initialized or
    a connection cannot be acquired.
    """
    if _pool is None:
        raise DatabaseUnavailableError("Database pool not initialized. Call init_db() first.")

    try:
        conn = _pool.getconn()
    except psycopg2.Error as e:
        logger.error(f"Failed to acquire database connection: {e}")
        raise DatabaseUnavailableError(f"Cannot connect to database: {e}") from e

    try:
        cid = id(conn)
        with _vector_registered_lock:
            needs_register = cid not in _vector_registered
        if needs_register:
            register_vector(conn)
            with _vector_registered_lock:
                _vector_registered.add(cid)
        yield conn
    finally:
        _pool.putconn(conn)
