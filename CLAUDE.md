# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AUB Libraries Assistant -- a bilingual (Arabic & English) RAG-based chatbot that answers library FAQs and provides personalized database recommendations. Uses OpenAI embeddings (text-embedding-3-small) with cosine similarity search via ChromaDB, and GPT-4o-mini for synthesizing library page answers and Arabic translation. React frontend with FastAPI backend, plus an admin dashboard for data management and analytics.

## Tech Stack

- **Frontend**: React 18, react-markdown, react-router-dom
- **Backend**: FastAPI + Uvicorn (Python 3.8+)
- **Vector DB**: ChromaDB (embedded, persistent) with OpenAI embedding function
- **APIs**: OpenAI (embeddings + chat completions)
- **Scraping**: Playwright (optional)

## Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Build embedding indices (required before first run, or after CSV changes)
python build_index.py

# Run backend (auto-builds indices if missing)
uvicorn backend.main:app --reload --port 8000

# Run frontend dev server (proxies API to :8000)
cd frontend && npm start

# Run CLI chatbot version
python chat.py

# Build frontend for production
cd frontend && npm run build
# Then: uvicorn backend.main:app --port 8000 (serves React build + API)

# Scrape AUB library website (optional)
python scrape_aub_library.py
```

No test suite exists yet.

## Architecture

### Data Flow
```
React UI --> POST /api/chat { message, language? } --> FastAPI
  --> ChromaDB query (auto-embeds via OpenAI)
  --> Parallel Search (FAQ, DB, Library Pages)
  --> Intent Detection (bilingual keyword regex)
  --> Language Resolution (explicit param > auto-detect)
  --> Response Router
  --> JSON { answer, debug, detected_language }
  --> React renders markdown (RTL for Arabic)
```

### API Endpoints

**Chat:**
- `POST /api/chat` -- Send message, get answer + debug info. Accepts `{ message, language? }`
- `GET /api/health` -- Check backend status and collection counts
- `GET /api/status` -- Check if indices exist

**Admin (under `/api/admin/`):**
- `GET /collections` -- List collections with document counts
- `GET /collections/{name}/entries` -- Paginated collection entries
- `POST /faq`, `PUT /faq/{id}`, `DELETE /faq/{id}` -- FAQ CRUD
- `POST /database`, `PUT /database/{id}`, `DELETE /database/{id}` -- Database CRUD
- `DELETE /library-page/{id}` -- Delete library page entry
- `POST /reindex` -- Trigger full re-index from CSV sources
- `GET /system-info` -- System information
- `GET /analytics/summary`, `/analytics/trends`, `/analytics/top-queries` -- Analytics

### Key Directories
- **`backend/`** -- FastAPI application
  - `main.py` -- FastAPI app, endpoints, startup initialization
  - `chatbot.py` -- `LibraryChatbot`, `IntentDetector`, `LanguageDetector`, `EmbeddingUtils`, `Config`
  - `index_builder.py` -- `IndexBuilder` for building ChromaDB collections
  - `admin.py` -- `AdminManager` for CRUD operations on collections
  - `analytics.py` -- `ChatLogger`, `AnalyticsComputer` for usage tracking
- **`frontend/`** -- React application
  - `src/App.js` -- Main shell, state management, API calls
  - `src/components/` -- Header, Footer, ChatWindow, ChatInput, MessageBubble, DebugPanel
  - `src/pages/AdminDashboard.js` -- Admin dashboard (System Status, Data Management, Analytics tabs)
  - `src/api.js` -- Fetch wrapper for backend API
  - `src/App.css` -- AUB-branded styles (includes RTL support for Arabic)
  - `src/i18n/` -- Translation files (en.js, ar.js)
  - `src/LanguageContext.js` -- Language state context
- **`build_index.py`** -- Standalone index builder (same logic as IndexBuilder)
- **`chat.py`** -- CLI chatbot (standalone, bilingual, no web server)
- **`scrape_aub_library.py`** -- Playwright scraper, stores in ChromaDB

### Key Classes (backend/)
- **`LibraryChatbot`** -- Main orchestrator: queries ChromaDB, routes through intent detection, bilingual
- **`IntentDetector`** -- Keyword regex-based intent classification (English + Arabic patterns)
- **`LanguageDetector`** -- Unicode character analysis for Arabic vs English detection
- **`EmbeddingUtils`** -- HTML cleaning and library content boilerplate stripping
- **`IndexBuilder`** -- Builds ChromaDB collections from CSV source files
- **`AdminManager`** -- CRUD operations on ChromaDB collections for admin dashboard
- **`ChatLogger`** -- Appends chat interaction logs to JSON-lines file
- **`AnalyticsComputer`** -- Reads chat logs and computes analytics on demand
- **`Config`** -- ChromaDB paths, collection names, confidence thresholds, language constants

### ChromaDB Collections (stored in `./chroma_db/`)
| Collection | Documents (embedded text) | Metadata |
|---|---|---|
| `faq` | question text | `{"question": "...", "answer": "..."}` |
| `databases` | `"{name}. {description}"` | `{"name": "...", "description": "..."}` |
| `library_pages` | `"{title}\n\n{content}"` | `{"url": "...", "title": "...", "content": "..."}` |

### Source Data Files
- **`library_faq_clean.csv`** -- FAQ source data (question/answer pairs)
- **`Databases description.csv`** -- Database recommendations (1000+ records)

## Configuration

- **`OPENAI_API_KEY`** -- Required; set in `.env` file or as environment variable
- Confidence thresholds defined in `backend/chatbot.py` Config class
- React dev server proxies to backend via `"proxy": "http://localhost:8000"` in package.json
- AUB brand colors: maroon #840132, red #ee3524
- Routes: `/` = chat interface, `/admin` = admin dashboard

## Important Notes

- Never commit API keys or `.env` files
- Rebuild indices (`python build_index.py`) when CSV source files change
- Library pages scraping is optional; the app works without it
- `chroma_db/` directory is gitignored
- `chat_logs.json` is generated at runtime and gitignored
- `chat.py` and `backend/chatbot.py` have parallel implementations (CLI vs API) -- keep in sync
- IntentDetector uses keyword regex, not LLM -- fast but needs manual keyword maintenance for new languages
- For production: build frontend (`npm run build`), then run FastAPI (serves static files)
