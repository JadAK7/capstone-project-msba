"""
main.py
FastAPI backend for the AUB Libraries Assistant chatbot.
Run with: uvicorn backend.main:app --reload --port 8000
"""

import os
import time
import logging
import threading
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel

import json as _json

from .chatbot import LibraryChatbot
from .database import init_db, close_db, get_connection
from .index_builder import IndexBuilder
from .admin import AdminManager, set_last_index_build_time, set_last_scrape_time
from .analytics import ChatLogger, AnalyticsComputer
from .evaluation import evaluate_single, run_evaluation_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AUB Libraries Assistant API")

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
_chatbot: Optional[LibraryChatbot] = None
_admin_manager: Optional[AdminManager] = None
_chat_logger = ChatLogger()
_analytics = AnalyticsComputer()

# Scrape task state
_scrape_status = {
    "running": False,
    "message": "",
    "pages_scraped": 0,
    "error": None,
}


def get_chatbot() -> LibraryChatbot:
    global _chatbot
    if _chatbot is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

        # Auto-build indices if needed
        if not IndexBuilder.indices_exist():
            logger.info("Building indices for first time setup...")
            IndexBuilder.build_indices(api_key)
            set_last_index_build_time()
            logger.info("Indices built successfully")

        _chatbot = LibraryChatbot(api_key)
    return _chatbot


def get_admin_manager() -> AdminManager:
    global _admin_manager
    if _admin_manager is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        _admin_manager = AdminManager(api_key)
    return _admin_manager


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None 
    history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    answer: str
    debug: dict
    detected_language: str
    conversation_id: Optional[int] = None


class FAQRequest(BaseModel):
    question: str
    answer: str


class DatabaseRequest(BaseModel):
    name: str
    description: str


class CustomNoteRequest(BaseModel):
    label: str
    content: str


class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: int  # 1 = thumbs up, -1 = thumbs down
    corrected_answer: Optional[str] = None
    comment: Optional[str] = None
    source: Optional[str] = "admin"  # "admin" or "user"


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Initialize database pool and chatbot eagerly on startup."""
    try:
        init_db()
        get_chatbot()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot on startup: {e}")


@app.on_event("shutdown")
async def shutdown():
    close_db()


# ---------------------------------------------------------------------------
# Existing chatbot endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """Check if the chatbot is initialized and indices are ready."""
    try:
        bot = get_chatbot()
        return {
            "status": "ok",
            "faq_count": bot.get_collection_count("faq"),
            "db_count": bot.get_collection_count("databases"),
            "library_available": bot.library_available,
            "supported_languages": ["en", "ar"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message and get a chatbot response.

    Language is always auto-detected from the message text on a per-message
    basis. The `language` field is accepted for backwards compatibility but
    is not used for response language determination.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    bot = get_chatbot()
    query_text = req.message.strip()

    start_time = time.time()
    answer, debug = bot.answer(query_text, history=req.history)
    response_time_ms = (time.time() - start_time) * 1000

    detected_language = debug.get("detected_language", "en")

    # Log the interaction for analytics (best-effort, non-blocking)
    conversation_id = None
    try:
        faq_scores = debug.get("faq_scores", [])
        db_scores = debug.get("db_scores", [])
        library_scores = debug.get("library_scores", [])
        faq_metas = debug.get("faq_metadatas", [])
        db_metas = debug.get("db_metadatas", [])

        conversation_id = _chat_logger.log(
            query=query_text,
            language=detected_language,
            intent_source=debug.get("chosen_source", "unknown"),
            response_length=len(answer),
            faq_top_score=faq_scores[0][1] if faq_scores else 0.0,
            db_top_score=db_scores[0][1] if db_scores else 0.0,
            library_top_score=library_scores[0][1] if library_scores else 0.0,
            cache_hit=debug.get("cache_hit", False),
            response_time_ms=response_time_ms,
            query_word_count=len(query_text.split()),
            keyword_intent_fired=debug.get("is_db_intent", False),
            sources_above_threshold=sum([
                debug.get("show_faq", False),
                debug.get("show_db", False),
                debug.get("show_library", False),
            ]),
            top_faq_question=faq_metas[0].get("question", "") if faq_metas else "",
            top_db_name=db_metas[0].get("name", "") if db_metas else "",
            generated_answer=answer,
            retrieved_chunks=debug.get("retrieved_chunks", []),
            # Hallucination debugging fields
            draft_answer=debug.get("draft_answer", ""),
            verified_answer=answer if debug.get("removed_claims") else "",
            removed_claims=debug.get("removed_claims", []),
            context_sent_to_llm=debug.get("context_sent_to_llm", ""),
            verification_passed=debug.get("verified", True),
            # Safety / guard fields
            guard_injection_detected=debug.get("guard_injection_detected", False),
            guard_out_of_scope=debug.get("guard_out_of_scope", False),
            guard_refusal_reason=debug.get("guard_refusal_reason", ""),
            guard_matched_patterns=debug.get("guard_matched_patterns", []),
        )
    except Exception as e:
        logger.warning(f"Chat logging failed (non-critical): {e}")

    return ChatResponse(
        answer=answer,
        debug=debug,
        detected_language=detected_language,
        conversation_id=conversation_id,
    )


@app.get("/api/status")
def status():
    """Check if indices exist."""
    return {"indices_ready": IndexBuilder.indices_exist()}


# ---------------------------------------------------------------------------
# Admin endpoints -- Collection browsing
# ---------------------------------------------------------------------------

@app.get("/api/admin/collections")
def admin_list_collections():
    """Return list of collections with document counts."""
    try:
        mgr = get_admin_manager()
        return mgr.list_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/collections/{collection_name}/entries")
def admin_get_entries(collection_name: str, offset: int = 0, limit: int = 20):
    """Return paginated entries from a collection."""
    valid_names = ["faq", "databases", "library_pages", "document_chunks", "custom_notes"]
    if collection_name not in valid_names:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection name. Must be one of: {valid_names}",
        )
    try:
        mgr = get_admin_manager()
        return mgr.get_collection_entries(collection_name, offset, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Admin endpoints -- FAQ CRUD
# ---------------------------------------------------------------------------

@app.post("/api/admin/faq")
def admin_add_faq(req: FAQRequest):
    """Add a new FAQ entry."""
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(status_code=400, detail="Question and answer are required.")
    try:
        mgr = get_admin_manager()
        return mgr.add_faq(req.question.strip(), req.answer.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/faq/{entry_id}")
def admin_update_faq(entry_id: str, req: FAQRequest):
    """Update an existing FAQ entry."""
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(status_code=400, detail="Question and answer are required.")
    try:
        mgr = get_admin_manager()
        return mgr.update_faq(entry_id, req.question.strip(), req.answer.strip())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/faq/{entry_id}")
def admin_delete_faq(entry_id: str):
    """Delete an FAQ entry."""
    try:
        mgr = get_admin_manager()
        return mgr.delete_faq(entry_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Admin endpoints -- Database CRUD
# ---------------------------------------------------------------------------

@app.post("/api/admin/database")
def admin_add_database(req: DatabaseRequest):
    """Add a new database description entry."""
    if not req.name.strip() or not req.description.strip():
        raise HTTPException(status_code=400, detail="Name and description are required.")
    try:
        mgr = get_admin_manager()
        return mgr.add_database(req.name.strip(), req.description.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/database/{entry_id}")
def admin_update_database(entry_id: str, req: DatabaseRequest):
    """Update an existing database entry."""
    if not req.name.strip() or not req.description.strip():
        raise HTTPException(status_code=400, detail="Name and description are required.")
    try:
        mgr = get_admin_manager()
        return mgr.update_database(entry_id, req.name.strip(), req.description.strip())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/database/{entry_id}")
def admin_delete_database(entry_id: str):
    """Delete a database entry."""
    try:
        mgr = get_admin_manager()
        return mgr.delete_database(entry_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Admin endpoints -- Custom Notes CRUD
# ---------------------------------------------------------------------------

@app.post("/api/admin/custom-note")
def admin_add_custom_note(req: CustomNoteRequest):
    """Add a new custom note entry."""
    if not req.label.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="Label and content are required.")
    try:
        mgr = get_admin_manager()
        return mgr.add_custom_note(req.label.strip(), req.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/custom-note/{entry_id}")
def admin_update_custom_note(entry_id: str, req: CustomNoteRequest):
    """Update an existing custom note entry."""
    if not req.label.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="Label and content are required.")
    try:
        mgr = get_admin_manager()
        return mgr.update_custom_note(entry_id, req.label.strip(), req.content.strip())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/custom-note/{entry_id}")
def admin_delete_custom_note(entry_id: str):
    """Delete a custom note entry."""
    try:
        mgr = get_admin_manager()
        return mgr.delete_custom_note(entry_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Admin endpoints -- Library Pages (delete only)
# ---------------------------------------------------------------------------

@app.get("/api/admin/document-chunks/search")
def admin_search_document_chunks(q: str, offset: int = 0, limit: int = 20):
    """Search document chunks by text content."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    try:
        search_term = f"%{q.strip()}%"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM document_chunks WHERE chunk_text ILIKE %s",
                    (search_term,),
                )
                total = cur.fetchone()[0]

                cur.execute(
                    "SELECT id, chunk_text, page_url, page_title, section_title, page_type, chunk_index "
                    "FROM document_chunks WHERE chunk_text ILIKE %s "
                    "ORDER BY id LIMIT %s OFFSET %s",
                    (search_term, limit, offset),
                )
                rows = cur.fetchall()

        entries = []
        for row in rows:
            entries.append({
                "id": row[0],
                "document": row[1],
                "metadata": {
                    "page_url": row[2],
                    "page_title": row[3],
                    "section_title": row[4],
                    "page_type": row[5],
                    "chunk_index": row[6],
                },
            })

        return {"entries": entries, "total": total, "offset": offset, "limit": limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/document-chunk/{entry_id}")
def admin_delete_document_chunk(entry_id: str):
    """Delete a document chunk entry."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE id = %s", (entry_id,))
            conn.commit()
        return {"id": entry_id, "deleted": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/library-page/{entry_id}")
def admin_delete_library_page(entry_id: str):
    """Delete a library page entry."""
    try:
        mgr = get_admin_manager()
        return mgr.delete_library_page(entry_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Admin endpoints -- Re-indexing
# ---------------------------------------------------------------------------

@app.post("/api/admin/reindex")
def admin_reindex():
    """Trigger full re-index of all collections from CSV source files."""
    global _chatbot, _admin_manager
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

        # Clear the response cache before re-indexing so stale answers
        # are not served if a request arrives between reset and re-init.
        if _chatbot is not None:
            _chatbot.clear_cache()
            logger.info("Response cache cleared before re-index")

        logger.info("Admin triggered full re-index...")
        IndexBuilder.build_indices(api_key)
        set_last_index_build_time()

        # Reset singletons so they pick up fresh collections
        # (the new chatbot instance will have a fresh empty cache)
        _chatbot = None
        _admin_manager = None

        logger.info("Re-index complete.")
        return {"status": "ok", "message": "Re-indexing completed successfully."}
    except Exception as e:
        logger.error(f"Re-index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Admin endpoints -- System info
# ---------------------------------------------------------------------------

@app.get("/api/admin/system-info")
def admin_system_info():
    """Return system information for the admin dashboard."""
    try:
        mgr = get_admin_manager()
        return mgr.get_system_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/cache-stats")
def admin_cache_stats():
    """Return response cache statistics (hits, misses, size, hit rate)."""
    try:
        bot = get_chatbot()
        return bot.cache_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/clear-cache")
def admin_clear_cache():
    """Clear the response cache."""
    try:
        bot = get_chatbot()
        bot.clear_cache()
        logger.info("Response cache cleared via admin endpoint")
        return {"status": "ok", "message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------

@app.get("/api/admin/analytics/summary")
def analytics_summary():
    """Return analytics summary."""
    try:
        return _analytics.summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/analytics/trends")
def analytics_trends():
    """Return daily conversation trends for the last 30 days."""
    try:
        return _analytics.trends(days=30)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/analytics/top-queries")
def analytics_top_queries():
    """Return top queried topics."""
    try:
        return _analytics.top_queries(limit=20)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/analytics/unanswered-queries")
def analytics_unanswered_queries():
    """Return queries the bot could not answer."""
    try:
        return _analytics.unanswered_queries(limit=50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/analytics/charts")
def analytics_charts():
    """Generate all analytics charts as base64-encoded PNGs."""
    try:
        charts = _analytics.compute_charts()
        extended_summary = _analytics.compute_extended_summary()
        return {
            "charts": charts,
            "extended_summary": extended_summary,
        }
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Admin endpoints -- Rescrape library website
# ---------------------------------------------------------------------------

def _run_scrape():
    """Background task: scrape AUB library website and rebuild library_pages."""
    global _chatbot, _admin_manager, _scrape_status
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
        from scrape_aub_library import AUBLibraryScraper, \
            START_URLS, ALLOWED_DOMAIN, ALLOWED_PATHS, MAX_PAGES

        _scrape_status["message"] = "Crawling AUB library website..."
        _scrape_status["pages_scraped"] = 0

        scraper = AUBLibraryScraper(
            start_urls=START_URLS,
            allowed_domain=ALLOWED_DOMAIN,
            allowed_paths=ALLOWED_PATHS,
        )
        scraped_data = scraper.crawl(max_pages=MAX_PAGES)
        _scrape_status["pages_scraped"] = len(scraped_data)

        if not scraped_data:
            _scrape_status["message"] = "No pages scraped."
            _scrape_status["error"] = "Scraper returned no data."
            _scrape_status["running"] = False
            return

        _scrape_status["message"] = f"Processing {len(scraped_data)} pages through pipeline..."
        count = IndexBuilder.build_chunks_from_scraped(scraped_data)

        # Reset singletons so they pick up fresh data
        if _chatbot is not None:
            _chatbot.clear_cache()
        _chatbot = None
        _admin_manager = None

        set_last_scrape_time()
        _scrape_status["message"] = f"Scraping complete. {count} chunks indexed from {len(scraped_data)} pages."
        _scrape_status["error"] = None
        logger.info(f"Rescrape complete: {count} library pages indexed.")
    except Exception as e:
        logger.error(f"Rescrape failed: {e}")
        _scrape_status["message"] = "Scraping failed."
        _scrape_status["error"] = str(e)
    finally:
        _scrape_status["running"] = False


@app.post("/api/admin/rescrape")
def admin_rescrape():
    """Trigger a background rescrape of the AUB library website."""
    global _scrape_status
    if _scrape_status["running"]:
        raise HTTPException(status_code=409, detail="A scrape is already in progress.")

    _scrape_status = {
        "running": True,
        "message": "Starting scrape...",
        "pages_scraped": 0,
        "error": None,
    }
    thread = threading.Thread(target=_run_scrape, daemon=True)
    thread.start()
    return {"status": "started", "message": "Scraping started in background."}


@app.get("/api/admin/rescrape/status")
def admin_rescrape_status():
    """Check the status of a running or completed scrape."""
    return _scrape_status


# ---------------------------------------------------------------------------
# Conversations & Feedback endpoints
# ---------------------------------------------------------------------------


@app.get("/api/admin/conversations")
def list_conversations(offset: int = 0, limit: int = 30, rating_filter: Optional[str] = None):
    """List all chat conversations with optional feedback info.

    rating_filter: 'positive', 'negative', 'unreviewed', or None (all).
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Build query with optional filter
                base_query = """
                    SELECT c.id, c.query, c.answer, c.language, c.chosen_source,
                           c.faq_top_score, c.db_top_score, c.library_top_score,
                           c.response_time_ms, c.created_at,
                           f.id AS feedback_id, f.rating, f.corrected_answer,
                           f.comment, f.created_at AS feedback_at,
                           f.feedback_source
                    FROM chat_conversations c
                    LEFT JOIN chat_feedback f ON f.conversation_id = c.id
                """
                conditions = []
                params = []

                if rating_filter == "positive":
                    conditions.append("f.rating = 1")
                elif rating_filter == "negative":
                    conditions.append("f.rating = -1")
                elif rating_filter == "unreviewed":
                    conditions.append("f.id IS NULL")
                elif rating_filter == "user_feedback":
                    conditions.append("f.feedback_source = 'user'")

                if conditions:
                    base_query += " WHERE " + " AND ".join(conditions)

                base_query += " ORDER BY c.created_at DESC OFFSET %s LIMIT %s"
                params.extend([offset, limit])

                cur.execute(base_query, params)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]

                # Get total count
                count_query = "SELECT COUNT(*) FROM chat_conversations c LEFT JOIN chat_feedback f ON f.conversation_id = c.id"
                if conditions:
                    count_query += " WHERE " + " AND ".join(conditions)
                cur.execute(count_query, [])
                total = cur.fetchone()[0]

        conversations = []
        for row in rows:
            conv = dict(zip(columns, row))
            # Serialize timestamps
            if conv.get("created_at"):
                conv["created_at"] = conv["created_at"].isoformat() + "Z"
            if conv.get("feedback_at"):
                conv["feedback_at"] = conv["feedback_at"].isoformat() + "Z"
            conversations.append(conv)

        return {"conversations": conversations, "total": total}
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/conversations/{conversation_id}")
def get_conversation(conversation_id: int):
    """Get a single conversation with its retrieved chunks and feedback."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT c.*, f.id AS feedback_id, f.rating,
                              f.corrected_answer, f.comment,
                              f.created_at AS feedback_at
                       FROM chat_conversations c
                       LEFT JOIN chat_feedback f ON f.conversation_id = c.id
                       WHERE c.id = %s""",
                    (conversation_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Conversation not found")
                columns = [desc[0] for desc in cur.description]
                conv = dict(zip(columns, row))
                if conv.get("created_at"):
                    conv["created_at"] = conv["created_at"].isoformat() + "Z"
                if conv.get("feedback_at"):
                    conv["feedback_at"] = conv["feedback_at"].isoformat() + "Z"
                return conv
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/feedback")
def submit_feedback(req: FeedbackRequest):
    """Submit admin feedback (thumbs up/down + optional corrected answer) for a conversation."""
    if req.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="Rating must be 1 (thumbs up) or -1 (thumbs down)")

    if req.rating == -1 and not req.corrected_answer and not req.comment:
        raise HTTPException(
            status_code=400,
            detail="Negative feedback requires a corrected answer or comment",
        )

    try:
        from .embeddings import embed_text as _embed

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Verify conversation exists and get query text for embedding
                cur.execute(
                    "SELECT query FROM chat_conversations WHERE id = %s",
                    (req.conversation_id,),
                )
                conv_row = cur.fetchone()
                if not conv_row:
                    raise HTTPException(status_code=404, detail="Conversation not found")

                # Delete existing feedback for this conversation (allow re-rating)
                cur.execute(
                    "DELETE FROM chat_feedback WHERE conversation_id = %s",
                    (req.conversation_id,),
                )

                # Always generate embedding for the original query so feedback
                # lookup works for both positive and negative ratings.
                # Positive feedback confirms the answer is good (used to skip
                # regeneration).  Negative + corrected_answer provides the fix.
                query_text = conv_row[0]
                embedding = None
                try:
                    embedding = _embed(query_text)
                except Exception as e:
                    logger.warning(f"Failed to embed query for feedback: {e}")

                feedback_source = req.source if req.source in ("admin", "user") else "admin"
                cur.execute(
                    """INSERT INTO chat_feedback
                       (conversation_id, rating, corrected_answer, comment, embedding, feedback_source)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (
                        req.conversation_id,
                        req.rating,
                        req.corrected_answer.strip() if req.corrected_answer else None,
                        req.comment.strip() if req.comment else None,
                        embedding,
                        feedback_source,
                    ),
                )
                feedback_id = cur.fetchone()[0]
            conn.commit()

        # Clear the response cache so the corrected answer is used on next query
        bot = get_chatbot()
        if bot:
            bot.clear_cache()

        return {"status": "ok", "feedback_id": feedback_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/conversations/{conversation_id}")
def delete_conversation(conversation_id: int):
    """Delete a chat conversation and its associated feedback."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_feedback WHERE conversation_id = %s", (conversation_id,))
                cur.execute("DELETE FROM chat_conversations WHERE id = %s RETURNING id", (conversation_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Conversation not found")
            conn.commit()
        return {"status": "ok", "id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/feedback/all")
def delete_all_feedback():
    """Delete all feedback entries and clear response cache."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_feedback")
                count = cur.rowcount
            conn.commit()
        # Clear cache so stale feedback-based answers aren't served
        bot = get_chatbot()
        bot.clear_cache()
        logger.info(f"All feedback deleted: {count} entries removed, cache cleared")
        return {"status": "ok", "deleted": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/feedback/{feedback_id}")
def delete_feedback(feedback_id: int):
    """Delete a feedback entry."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_feedback WHERE id = %s RETURNING id", (feedback_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Feedback not found")
            conn.commit()
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/feedback/stats")
def feedback_stats():
    """Return feedback statistics."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) AS total_conversations,
                        COUNT(f.id) AS total_reviewed,
                        COUNT(*) FILTER (WHERE f.rating = 1) AS positive,
                        COUNT(*) FILTER (WHERE f.rating = -1) AS negative,
                        COUNT(*) FILTER (WHERE f.corrected_answer IS NOT NULL) AS with_corrections
                    FROM chat_conversations c
                    LEFT JOIN chat_feedback f ON f.conversation_id = c.id
                """)
                row = cur.fetchone()
                columns = [desc[0] for desc in cur.description]
                stats = dict(zip(columns, row))
                stats["unreviewed"] = stats["total_conversations"] - stats["total_reviewed"]
                return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Evaluation endpoints
# ---------------------------------------------------------------------------


class EvaluationRequest(BaseModel):
    questions: List[str]
    language: Optional[str] = None


class SingleEvalRequest(BaseModel):
    question: str
    language: Optional[str] = None


@app.post("/api/admin/evaluation/run")
def run_evaluation(req: EvaluationRequest):
    """Run the evaluation pipeline on a list of test questions.

    Returns per-question metrics and aggregate scores.
    """
    if not req.questions:
        raise HTTPException(status_code=400, detail="At least one question is required")
    if len(req.questions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 questions per run")

    try:
        bot = get_chatbot()
        results = run_evaluation_pipeline(
            questions=req.questions,
            chatbot=bot,
            language=req.language,
        )
        return results
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/evaluation/single")
def evaluate_single_question(req: SingleEvalRequest):
    """Evaluate a single question through the chatbot and return all metrics."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        bot = get_chatbot()
        answer, debug = bot.answer(req.question.strip(), language=req.language)

        result = evaluate_single(
            query=req.question.strip(),
            answer=answer,
            retrieved_chunks=debug.get("retrieved_chunks", []),
            chosen_source=debug.get("chosen_source", ""),
            context_sent_to_llm=debug.get("context_sent_to_llm", ""),
        )
        result["full_answer"] = answer
        result["debug"] = {
            "chosen_source": debug.get("chosen_source"),
            "detected_language": debug.get("detected_language"),
            "faq_scores": debug.get("faq_scores", []),
            "db_scores": debug.get("db_scores", []),
            "library_scores": debug.get("library_scores", []),
        }
        return result
    except Exception as e:
        logger.error(f"Single evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Serve React build in production
# ---------------------------------------------------------------------------

_frontend_build = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "build")
if os.path.isdir(_frontend_build):

    @app.middleware("http")
    async def no_cache_html(request: Request, call_next):
        response: Response = await call_next(request)
        # Prevent Safari (and other browsers) from caching index.html
        # JS/CSS files have content hashes in their filenames so caching those is fine
        if request.url.path == "/" or request.url.path.endswith(".html"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    from fastapi.responses import FileResponse

    # Serve static assets (JS, CSS, images) directly
    app.mount("/static", StaticFiles(directory=os.path.join(_frontend_build, "static")), name="static")

    # Catch-all: serve index.html for any non-API route (React client-side routing)
    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        # If the path points to an actual file in build/, serve it
        file_path = os.path.join(_frontend_build, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        # Otherwise serve index.html — React Router handles the route
        return FileResponse(os.path.join(_frontend_build, "index.html"))
