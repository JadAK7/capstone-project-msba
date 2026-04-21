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
from .embeddings import embed_text, embed_texts

logger = logging.getLogger(__name__)

# Track last index build / scrape times (in-memory; resets on server restart)
_last_index_build_time: Optional[float] = None
_last_scrape_time: Optional[float] = None
_server_start_time: float = time.time()


def set_last_index_build_time(ts: Optional[float] = None):
    """Record when indices were last built."""
    global _last_index_build_time
    _last_index_build_time = ts or time.time()


def get_last_index_build_time() -> Optional[float]:
    return _last_index_build_time


def set_last_scrape_time(ts: Optional[float] = None):
    """Record when the library website was last scraped."""
    global _last_scrape_time
    _last_scrape_time = ts or time.time()


def get_last_scrape_time() -> Optional[float]:
    return _last_scrape_time


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
    "document_chunks": {
        "table": "document_chunks",
        "metadata_cols": ["page_url", "page_title", "section_title", "page_type", "chunk_index"],
    },
    "custom_notes": {
        "table": "custom_notes",
        "metadata_cols": ["label", "content"],
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
                for name in [Config.FAQ_COLLECTION, Config.DB_COLLECTION, Config.LIBRARY_COLLECTION, "document_chunks", "custom_notes"]:
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
                # For custom_notes, hide chunk rows (note_X_cN) — only show base rows
                chunk_filter = ""
                if table == "custom_notes":
                    chunk_filter = "WHERE id NOT LIKE '%\\_c%' ESCAPE '\\'"

                cur.execute(f"SELECT COUNT(*) FROM {table} {chunk_filter}")
                total = cur.fetchone()[0]

                if total == 0:
                    return {"entries": [], "total": 0, "offset": offset, "limit": limit}

                doc_col = "chunk_text" if table == "document_chunks" else "document"
                cols = ", ".join(metadata_cols)
                cur.execute(
                    f"SELECT id, {doc_col}, {cols} FROM {table} {chunk_filter} ORDER BY id LIMIT %s OFFSET %s",
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
    # Custom Notes CRUD (with chunking)
    #
    # A single admin note can contain multiple topics (hours, borrowing
    # policies, etc.).  We split the content into semantic chunks so each
    # piece of information gets its own embedding and can be retrieved
    # independently.
    #
    # Storage model:
    #   - Each note gets a base ID like "note_3"
    #   - Its chunks are stored as "note_3", "note_3_c1", "note_3_c2", …
    #   - The base row stores the FULL content (label + content) for display
    #     in the admin dashboard.  Its embedding covers the first chunk.
    #   - Extra chunk rows store individual pieces with focused embeddings.
    #   - Deleting/updating the base ID cascades to all its chunks.
    # ------------------------------------------------------------------

    # Chunking config for custom notes
    _NOTE_CHUNK_SIZE = 600       # target chars per chunk
    _NOTE_CHUNK_OVERLAP = 80     # overlap between chunks
    _NOTE_MIN_CHUNK_SIZE = 40    # drop tiny trailing chunks

    def _next_note_id(self) -> str:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM custom_notes")
                all_ids = [row[0] for row in cur.fetchall()]
        max_num = -1
        for id_ in all_ids:
            # Parse base id (note_3, note_3_c1 → 3)
            base = id_.split("_c")[0] if "_c" in id_ else id_
            try:
                num = int(base.split("_", 1)[1])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                pass
        return f"note_{max_num + 1}"

    @staticmethod
    def _chunk_note_content(label: str, content: str,
                            chunk_size: int = 600,
                            overlap: int = 80,
                            min_size: int = 40) -> list:
        """Split a custom note into semantic chunks.

        Strategy:
          1. Split on double newlines (paragraph boundaries) first
          2. If a paragraph is still too long, split on single newlines
          3. If still too long, split on sentence boundaries
          4. Each chunk gets the label as a context prefix so the embedding
             knows what topic area the chunk belongs to

        Returns list of dicts: [{text, label, chunk_index}, ...]
        """
        import re

        # Normalize whitespace
        text = content.strip()
        if not text:
            return [{"text": f"{label}.", "label": label, "chunk_index": 0}]

        # Split into paragraphs (double newline or multiple newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # If the entire content fits in one chunk, return as-is
        if len(text) <= chunk_size:
            return [{"text": f"{label}\n\n{text}", "label": label, "chunk_index": 0}]

        # Build chunks by accumulating paragraphs
        chunks = []
        current = ""

        for para in paragraphs:
            # If this paragraph alone exceeds chunk_size, split it further
            if len(para) > chunk_size:
                # Flush current buffer first
                if current.strip():
                    chunks.append(current.strip())
                    current = ""
                # Split long paragraph on sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if len(current) + len(sent) + 1 > chunk_size and current.strip():
                        chunks.append(current.strip())
                        # Overlap: keep tail of previous chunk
                        current = current[-overlap:] + " " if overlap and current else ""
                    current += sent + " "
                if current.strip():
                    chunks.append(current.strip())
                    current = ""
            elif len(current) + len(para) + 2 > chunk_size and current.strip():
                # Adding this paragraph would exceed chunk_size → flush
                chunks.append(current.strip())
                # Overlap: keep tail of previous chunk
                current = current[-overlap:] + "\n\n" if overlap and current else ""
                current += para + "\n\n"
            else:
                current += para + "\n\n"

        if current.strip():
            chunks.append(current.strip())

        # Drop tiny trailing chunks
        chunks = [c for c in chunks if len(c) >= min_size]

        if not chunks:
            return [{"text": f"{label}\n\n{text}", "label": label, "chunk_index": 0}]

        # Prepend label as context to each chunk
        result = []
        for i, chunk_text in enumerate(chunks):
            result.append({
                "text": f"{label}\n\n{chunk_text}",
                "label": label,
                "chunk_index": i,
            })
        return result

    def add_custom_note(self, label: str, content: str) -> dict:
        """Add a new custom note, chunked for better retrieval."""
        base_id = self._next_note_id()
        chunks = self._chunk_note_content(
            label, content,
            self._NOTE_CHUNK_SIZE, self._NOTE_CHUNK_OVERLAP, self._NOTE_MIN_CHUNK_SIZE,
        )

        # Embed all chunks in one batch call
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts)

        full_document = f"{label}. {content}"

        with get_connection() as conn:
            with conn.cursor() as cur:
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    if i == 0:
                        # Base row: stores full content for admin display
                        row_id = base_id
                        doc = full_document
                    else:
                        # Chunk rows: store only the chunk text
                        row_id = f"{base_id}_c{i}"
                        doc = chunk["text"]
                    cur.execute(
                        "INSERT INTO custom_notes (id, document, embedding, label, content) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (row_id, doc, np.array(emb), label, chunk["text"]),
                    )
            conn.commit()

        logger.info(f"Custom note '{base_id}' added with {len(chunks)} chunk(s)")
        return {"id": base_id, "label": label, "content": content, "chunks": len(chunks)}

    def update_custom_note(self, entry_id: str, label: str, content: str) -> dict:
        """Update a custom note: delete old chunks, re-chunk, re-embed."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM custom_notes WHERE id = %s", (entry_id,))
                if cur.fetchone() is None:
                    raise KeyError(f"Custom note '{entry_id}' not found.")

        # Delete old base + chunk rows
        self._delete_note_and_chunks(entry_id)

        # Re-chunk and re-embed
        chunks = self._chunk_note_content(
            label, content,
            self._NOTE_CHUNK_SIZE, self._NOTE_CHUNK_OVERLAP, self._NOTE_MIN_CHUNK_SIZE,
        )
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts)

        full_document = f"{label}. {content}"

        with get_connection() as conn:
            with conn.cursor() as cur:
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    if i == 0:
                        row_id = entry_id
                        doc = full_document
                    else:
                        row_id = f"{entry_id}_c{i}"
                        doc = chunk["text"]
                    cur.execute(
                        "INSERT INTO custom_notes (id, document, embedding, label, content) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (row_id, doc, np.array(emb), label, chunk["text"]),
                    )
            conn.commit()

        logger.info(f"Custom note '{entry_id}' updated with {len(chunks)} chunk(s)")
        return {"id": entry_id, "label": label, "content": content, "chunks": len(chunks)}

    def _delete_note_and_chunks(self, entry_id: str):
        """Delete a note's base row and all its chunk rows."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Delete base row + any chunk rows (note_3, note_3_c1, note_3_c2, …)
                cur.execute(
                    "DELETE FROM custom_notes WHERE id = %s OR id LIKE %s",
                    (entry_id, f"{entry_id}_c%"),
                )
            conn.commit()

    def delete_custom_note(self, entry_id: str) -> dict:
        """Delete a custom note and all its chunks."""
        self._delete_note_and_chunks(entry_id)
        return {"id": entry_id, "deleted": True}

    # ------------------------------------------------------------------
    # Word Documents (stored in custom_notes with [DOC] label prefix)
    #
    # Uploaded .docx files are parsed to plain text, then chunked and
    # embedded exactly like custom notes.  The label is always prefixed
    # with "[DOC] " so they can be filtered separately in the admin UI.
    # ------------------------------------------------------------------

    _DOC_PREFIX = "[DOC] "

    def upload_document(self, filename: str, content: str) -> dict:
        """Store a parsed Word document as a custom note with [DOC] prefix."""
        label = f"{self._DOC_PREFIX}{filename}"
        return self.add_custom_note(label, content)

    def list_documents(self) -> list:
        """Return all uploaded documents (base rows only, no chunk rows)."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, label, content, created_at FROM custom_notes "
                    "WHERE label LIKE %s AND id NOT LIKE %s "
                    "ORDER BY created_at DESC",
                    (f"{self._DOC_PREFIX}%", "%\\_c%"),
                )
                rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "filename": row[1][len(self._DOC_PREFIX):],
                "label": row[1],
                "preview": row[2][:200] if row[2] else "",
                "created_at": row[3].isoformat() if row[3] else None,
            }
            for row in rows
        ]

    def delete_document(self, entry_id: str) -> dict:
        """Delete an uploaded document and all its chunks."""
        return self.delete_custom_note(entry_id)

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
            "last_scrape": get_last_scrape_time(),
            "server_start_time": get_server_start_time(),
            "server_uptime_seconds": time.time() - get_server_start_time(),
        }
