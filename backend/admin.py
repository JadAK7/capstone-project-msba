"""
admin.py
Admin API logic for managing PostgreSQL collections (CRUD operations).
"""

import time
import logging
import numpy as np
from typing import Optional

from .chatbot import Config
from .database import get_connection
from .embeddings import embed_text

logger = logging.getLogger(__name__)

# Track last index build time (in-memory; resets on server restart)
_last_index_build_time: Optional[float] = None
_server_start_time: float = time.time()


def set_last_index_build_time(ts: Optional[float] = None):
    """Record when indices were last built."""
    global _last_index_build_time
    _last_index_build_time = ts or time.time()


def get_last_index_build_time() -> Optional[float]:
    return _last_index_build_time


def get_server_start_time() -> float:
    return _server_start_time


# Map collection names to their table schemas
_TABLE_META = {
    Config.FAQ_COLLECTION: {
        "table": "faq",
        "metadata_cols": ["question", "answer"],
    },
    Config.DB_COLLECTION: {
        "table": "databases",
        "metadata_cols": ["name", "description"],
    },
    Config.LIBRARY_COLLECTION: {
        "table": "library_pages",
        "metadata_cols": ["url", "title", "content"],
    },
}


class AdminManager:
    """Manages direct PostgreSQL operations for admin CRUD."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Collection browsing
    # ------------------------------------------------------------------

    def list_collections(self) -> list:
        """Return list of collections with document counts."""
        results = []
        with get_connection() as conn:
            with conn.cursor() as cur:
                for name in [Config.FAQ_COLLECTION, Config.DB_COLLECTION, Config.LIBRARY_COLLECTION]:
                    table = _TABLE_META[name]["table"]
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                    except Exception:
                        conn.rollback()
                        count = 0
                    results.append({"name": name, "count": count})
        return results

    def get_collection_entries(self, collection_name: str, offset: int = 0, limit: int = 20) -> dict:
        """Return paginated entries from a table."""
        meta = _TABLE_META.get(collection_name)
        if meta is None:
            return {"entries": [], "total": 0, "offset": offset, "limit": limit}

        table = meta["table"]
        metadata_cols = meta["metadata_cols"]

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                total = cur.fetchone()[0]

                if total == 0:
                    return {"entries": [], "total": 0, "offset": offset, "limit": limit}

                cols = ", ".join(metadata_cols)
                cur.execute(
                    f"SELECT id, document, {cols} FROM {table} ORDER BY id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                rows = cur.fetchall()

        entries = []
        for row in rows:
            entry = {
                "id": row[0],
                "document": row[1],
                "metadata": {col: row[2 + i] for i, col in enumerate(metadata_cols)},
            }
            entries.append(entry)

        return {
            "entries": entries,
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    # ------------------------------------------------------------------
    # FAQ CRUD
    # ------------------------------------------------------------------

    def _next_faq_id(self) -> str:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM faq")
                all_ids = [row[0] for row in cur.fetchall()]
        max_num = -1
        for id_ in all_ids:
            try:
                num = int(id_.split("_", 1)[1])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                pass
        return f"faq_{max_num + 1}"

    def add_faq(self, question: str, answer: str) -> dict:
        """Add a new FAQ entry."""
        entry_id = self._next_faq_id()
        embedding = embed_text(question)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO faq (id, document, embedding, question, answer) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (entry_id, question, np.array(embedding), question, answer),
                )
            conn.commit()
        return {"id": entry_id, "question": question, "answer": answer}

    def update_faq(self, entry_id: str, question: str, answer: str) -> dict:
        """Update an existing FAQ entry."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM faq WHERE id = %s", (entry_id,))
                if cur.fetchone() is None:
                    raise KeyError(f"FAQ entry '{entry_id}' not found.")

        embedding = embed_text(question)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE faq SET document=%s, embedding=%s, question=%s, answer=%s WHERE id=%s",
                    (question, np.array(embedding), question, answer, entry_id),
                )
            conn.commit()
        return {"id": entry_id, "question": question, "answer": answer}

    def delete_faq(self, entry_id: str) -> dict:
        """Delete an FAQ entry."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM faq WHERE id = %s", (entry_id,))
            conn.commit()
        return {"id": entry_id, "deleted": True}

    # ------------------------------------------------------------------
    # Database CRUD
    # ------------------------------------------------------------------

    def _next_db_id(self) -> str:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM databases")
                all_ids = [row[0] for row in cur.fetchall()]
        max_num = -1
        for id_ in all_ids:
            try:
                num = int(id_.split("_", 1)[1])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                pass
        return f"db_{max_num + 1}"

    def add_database(self, name: str, description: str) -> dict:
        """Add a new database description entry."""
        entry_id = self._next_db_id()
        document = f"{name}. {description}"
        embedding = embed_text(document)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO databases (id, document, embedding, name, description) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (entry_id, document, np.array(embedding), name, description),
                )
            conn.commit()
        return {"id": entry_id, "name": name, "description": description}

    def update_database(self, entry_id: str, name: str, description: str) -> dict:
        """Update an existing database entry."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM databases WHERE id = %s", (entry_id,))
                if cur.fetchone() is None:
                    raise KeyError(f"Database entry '{entry_id}' not found.")

        document = f"{name}. {description}"
        embedding = embed_text(document)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE databases SET document=%s, embedding=%s, name=%s, description=%s WHERE id=%s",
                    (document, np.array(embedding), name, description, entry_id),
                )
            conn.commit()
        return {"id": entry_id, "name": name, "description": description}

    def delete_database(self, entry_id: str) -> dict:
        """Delete a database entry."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM databases WHERE id = %s", (entry_id,))
            conn.commit()
        return {"id": entry_id, "deleted": True}

    # ------------------------------------------------------------------
    # Library Pages (read-only + delete)
    # ------------------------------------------------------------------

    def delete_library_page(self, entry_id: str) -> dict:
        """Delete a library page entry."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM library_pages WHERE id = %s", (entry_id,))
            conn.commit()
        return {"id": entry_id, "deleted": True}

    # ------------------------------------------------------------------
    # System info
    # ------------------------------------------------------------------

    def get_system_info(self) -> dict:
        """Return system information for the admin dashboard."""
        collections = self.list_collections()
        return {
            "collections": collections,
            "embedding_model": Config.EMBEDDING_MODEL,
            "database": "PostgreSQL + pgvector",
            "last_index_build": get_last_index_build_time(),
            "server_start_time": get_server_start_time(),
            "server_uptime_seconds": time.time() - get_server_start_time(),
        }
