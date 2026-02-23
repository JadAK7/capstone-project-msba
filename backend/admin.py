"""
admin.py
Admin API logic for managing ChromaDB collections (CRUD operations).
"""

import os
import time
import logging
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from .chatbot import Config

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


class AdminManager:
    """Manages direct ChromaDB operations for admin CRUD."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=Config.EMBEDDING_MODEL,
        )
        self.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DIR)

    def _get_collection(self, name: str):
        """Get a ChromaDB collection by name, or None if it doesn't exist."""
        try:
            return self.chroma_client.get_collection(
                name=name,
                embedding_function=self.embedding_fn,
            )
        except (ValueError, Exception):
            return None

    # ------------------------------------------------------------------
    # Collection browsing
    # ------------------------------------------------------------------

    def list_collections(self) -> list:
        """Return list of collections with document counts."""
        results = []
        for name in [Config.FAQ_COLLECTION, Config.DB_COLLECTION, Config.LIBRARY_COLLECTION]:
            col = self._get_collection(name)
            count = col.count() if col else 0
            results.append({"name": name, "count": count})
        return results

    def get_collection_entries(self, collection_name: str, offset: int = 0, limit: int = 20) -> dict:
        """Return paginated entries from a collection."""
        col = self._get_collection(collection_name)
        if col is None:
            return {"entries": [], "total": 0, "offset": offset, "limit": limit}

        total = col.count()
        if total == 0:
            return {"entries": [], "total": 0, "offset": offset, "limit": limit}

        # ChromaDB get() returns all documents; we paginate manually
        all_data = col.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )

        entries = []
        ids = all_data.get("ids", [])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])

        for i in range(len(ids)):
            entries.append({
                "id": ids[i],
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
            })

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
        col = self._get_collection(Config.FAQ_COLLECTION)
        if col is None:
            return "faq_0"
        count = col.count()
        # Find the max existing numeric suffix to avoid collisions
        all_ids = col.get(include=[])["ids"]
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
        col = self._get_collection(Config.FAQ_COLLECTION)
        if col is None:
            raise ValueError("FAQ collection does not exist. Run re-index first.")

        entry_id = self._next_faq_id()
        col.upsert(
            ids=[entry_id],
            documents=[question],
            metadatas=[{"question": question, "answer": answer}],
        )
        return {"id": entry_id, "question": question, "answer": answer}

    def update_faq(self, entry_id: str, question: str, answer: str) -> dict:
        """Update an existing FAQ entry."""
        col = self._get_collection(Config.FAQ_COLLECTION)
        if col is None:
            raise ValueError("FAQ collection does not exist.")

        # Verify entry exists
        existing = col.get(ids=[entry_id], include=["documents"])
        if not existing["ids"]:
            raise KeyError(f"FAQ entry '{entry_id}' not found.")

        col.upsert(
            ids=[entry_id],
            documents=[question],
            metadatas=[{"question": question, "answer": answer}],
        )
        return {"id": entry_id, "question": question, "answer": answer}

    def delete_faq(self, entry_id: str) -> dict:
        """Delete an FAQ entry."""
        col = self._get_collection(Config.FAQ_COLLECTION)
        if col is None:
            raise ValueError("FAQ collection does not exist.")

        col.delete(ids=[entry_id])
        return {"id": entry_id, "deleted": True}

    # ------------------------------------------------------------------
    # Database CRUD
    # ------------------------------------------------------------------

    def _next_db_id(self) -> str:
        col = self._get_collection(Config.DB_COLLECTION)
        if col is None:
            return "db_0"
        all_ids = col.get(include=[])["ids"]
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
        col = self._get_collection(Config.DB_COLLECTION)
        if col is None:
            raise ValueError("Databases collection does not exist. Run re-index first.")

        entry_id = self._next_db_id()
        document = f"{name}. {description}"
        col.upsert(
            ids=[entry_id],
            documents=[document],
            metadatas=[{"name": name, "description": description}],
        )
        return {"id": entry_id, "name": name, "description": description}

    def update_database(self, entry_id: str, name: str, description: str) -> dict:
        """Update an existing database entry."""
        col = self._get_collection(Config.DB_COLLECTION)
        if col is None:
            raise ValueError("Databases collection does not exist.")

        existing = col.get(ids=[entry_id], include=["documents"])
        if not existing["ids"]:
            raise KeyError(f"Database entry '{entry_id}' not found.")

        document = f"{name}. {description}"
        col.upsert(
            ids=[entry_id],
            documents=[document],
            metadatas=[{"name": name, "description": description}],
        )
        return {"id": entry_id, "name": name, "description": description}

    def delete_database(self, entry_id: str) -> dict:
        """Delete a database entry."""
        col = self._get_collection(Config.DB_COLLECTION)
        if col is None:
            raise ValueError("Databases collection does not exist.")

        col.delete(ids=[entry_id])
        return {"id": entry_id, "deleted": True}

    # ------------------------------------------------------------------
    # Library Pages (read-only + delete)
    # ------------------------------------------------------------------

    def delete_library_page(self, entry_id: str) -> dict:
        """Delete a library page entry."""
        col = self._get_collection(Config.LIBRARY_COLLECTION)
        if col is None:
            raise ValueError("Library pages collection does not exist.")

        col.delete(ids=[entry_id])
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
            "chroma_dir": Config.CHROMA_DIR,
            "last_index_build": get_last_index_build_time(),
            "server_start_time": get_server_start_time(),
            "server_uptime_seconds": time.time() - get_server_start_time(),
        }
