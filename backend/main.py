"""
main.py
FastAPI backend for the AUB Libraries Assistant chatbot.
Run with: uvicorn backend.main:app --reload --port 8000
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .chatbot import LibraryChatbot, LanguageDetector
from .index_builder import IndexBuilder
from .admin import AdminManager, set_last_index_build_time
from .analytics import ChatLogger, AnalyticsComputer

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
    language: Optional[str] = None  # "en", "ar", or None for auto-detect


class ChatResponse(BaseModel):
    answer: str
    debug: dict
    detected_language: str  # The language used for the response


class FAQRequest(BaseModel):
    question: str
    answer: str


class DatabaseRequest(BaseModel):
    name: str
    description: str


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Initialize chatbot eagerly on startup."""
    try:
        get_chatbot()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot on startup: {e}")


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
            "faq_count": bot.faq_collection.count(),
            "db_count": bot.db_collection.count(),
            "library_available": bot.library_collection is not None,
            "supported_languages": ["en", "ar"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message and get a chatbot response.

    The `language` field is optional:
    - "en" -- force English response
    - "ar" -- force Arabic response
    - null/omitted -- auto-detect from the message text
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    bot = get_chatbot()
    answer, debug = bot.answer(req.message.strip(), language=req.language)

    # The detected (or overridden) language is stored in debug by chatbot.py
    detected_language = debug.get("detected_language", "en")

    # Log the interaction for analytics (best-effort, non-blocking)
    try:
        faq_scores = debug.get("faq_scores", [])
        db_scores = debug.get("db_scores", [])
        library_scores = debug.get("library_scores", [])
        _chat_logger.log(
            query=req.message.strip(),
            language=detected_language,
            intent_source=debug.get("chosen_source", "unknown"),
            response_length=len(answer),
            faq_top_score=faq_scores[0][1] if faq_scores else 0.0,
            db_top_score=db_scores[0][1] if db_scores else 0.0,
            library_top_score=library_scores[0][1] if library_scores else 0.0,
        )
    except Exception as e:
        logger.warning(f"Chat logging failed (non-critical): {e}")

    return ChatResponse(
        answer=answer,
        debug=debug,
        detected_language=detected_language,
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
    valid_names = ["faq", "databases", "library_pages"]
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
# Admin endpoints -- Library Pages (delete only)
# ---------------------------------------------------------------------------

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

        logger.info("Admin triggered full re-index...")
        IndexBuilder.build_indices(api_key)
        set_last_index_build_time()

        # Reset singletons so they pick up fresh collections
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


# ---------------------------------------------------------------------------
# Serve React build in production
# ---------------------------------------------------------------------------

_frontend_build = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "build")
if os.path.isdir(_frontend_build):
    app.mount("/", StaticFiles(directory=_frontend_build, html=True), name="frontend")
