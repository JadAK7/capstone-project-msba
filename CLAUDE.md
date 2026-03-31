# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AUB Libraries Assistant -- a bilingual (Arabic & English) RAG-based chatbot that answers library FAQs and provides personalized database recommendations. Uses OpenAI embeddings (text-embedding-3-small) with cosine similarity search via PostgreSQL + pgvector, and GPT-4o-mini for synthesizing library page answers and Arabic translation. React frontend with FastAPI backend, plus an admin dashboard for data management and analytics.

## Tech Stack

- **Frontend**: React 18, react-markdown, react-router-dom
- **Backend**: FastAPI + Uvicorn (Python 3.8+)
- **Vector DB**: PostgreSQL 16 + pgvector (via Docker Compose)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions), centralized in `backend/embeddings.py`
- **APIs**: OpenAI (embeddings + chat completions)
- **Scraping**: Playwright (optional)

## Commands

```bash
# Start PostgreSQL + pgvector (required)
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Build embedding indices (required before first run, or after CSV changes)
python scripts/build_index.py

# Run backend (auto-builds indices if missing)
uvicorn backend.main:app --reload --port 8000

# Run frontend dev server (proxies API to :8000)
cd frontend && npm start

# Run CLI chatbot version
python scripts/chat.py

# Build frontend for production
cd frontend && npm run build
# Then: uvicorn backend.main:app --port 8000 (serves React build + API)

# Scrape AUB library website (optional)
python scripts/scrape_aub_library.py

# Run everything with Docker (single command)
docker-compose up --build
```

No test suite exists yet.

## Architecture

### Data Flow (Pipeline v3: Query Rewriting + Cross-Encoder Reranking)
```
React UI --> POST /api/chat { message, language? } --> FastAPI
  --> 1. Input Guards (prompt injection detection, domain scope check)
  --> 2. Query Rewriting (LLM-based via gpt-4o-mini)
        --> Follow-up resolution using conversation history
        --> Arabic → English translation for retrieval
        --> Short/vague query expansion
        --> Fast-path skip for well-formed English queries (>5 words)
  --> Cache lookup (rewritten_query, lang)
  --> HIT: return cached (answer, debug) immediately
  --> MISS:
    --> Admin feedback correction lookup (pgvector similarity >= 0.85)
    --> FOUND: use admin-corrected answer via LLM
    --> NOT FOUND:
      --> 3. Intent Classification → table/page_type pre-filtering
            --> hours → hours_contact pages, database → databases table, etc.
      --> 4. Hybrid Retrieval (per table: vector + keyword search)
            --> Vector: embed_text(rewritten_query) -> pgvector cosine similarity
            --> Keyword: phraseto_tsquery (boosted 2x) + plainto_tsquery + synonym expansion
            --> Merge: Reciprocal Rank Fusion (65% vector, 35% keyword)
            --> page_type WHERE clause for document_chunks when intent is specific
      --> 5. LLM Reranking (gpt-4o-mini relevance scoring 0-1)
            --> Dedup (Jaccard 0.85) → top 15 → score relevance → min_score 0.5
      --> 6. Context Sufficiency Check
            --> top_score >= 0.55 → confident answer
            --> top_score 0.40-0.55 → partial answer (extra caution prompt)
            --> top_score < 0.40 → abstain (refuse to answer)
      --> 7. Grounded Answer Generation (uses ORIGINAL query for user language)
            --> System prompt: role-locked, self-check, cite-or-remove
            --> temperature=0.0, top_p=0.85
            --> Partial-context warning injected when context is borderline
      --> 8. Claim Verification (LLM-based strict checking + regex safety)
            --> Fail-safe: if verifier errors, return cautious fallback (not raw draft)
      --> Cache store (answer, debug)
  --> JSON { answer, debug, detected_language }
  --> React renders markdown (RTL for Arabic)
```

### Ingestion Pipeline (Scraping -> Chunks)
```
Playwright Scraper --> Raw HTML pages (preserves innerHTML + innerText)
  --> scraper_cleaner.py: BeautifulSoup HTML parsing, noise removal,
      preserves headings/tables/lists/structure, page type detection
  --> document_processor.py: page type routing, FAQ Q&A pair extraction,
      hours/schedule parsing per-location, section splitting
  --> chunker.py: structure-aware chunking (tables stay intact, FAQ pairs
      become individual chunks, lists not split mid-item, overlap on text only)
  --> embeddings.py: OpenAI text-embedding-3-small
  --> PostgreSQL: document_chunks table (+ legacy library_pages with cleaned text)
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
- `POST /reindex` -- Trigger full re-index from CSV sources (also clears response cache)
- `POST /rescrape` -- Trigger background rescrape of AUB library website
- `GET /rescrape/status` -- Check rescrape progress (running, message, pages_scraped, error)
- `GET /system-info` -- System information
- `GET /cache-stats` -- Response cache statistics (hits, misses, size, hit rate)
- `GET /analytics/summary`, `/analytics/trends`, `/analytics/top-queries` -- Analytics
- `GET /analytics/unanswered-queries` -- Questions the bot could not answer
- `GET /analytics/charts` -- All matplotlib charts as base64 PNGs + extended summary stats

### Key Directories
- **`backend/`** -- FastAPI application
  - `main.py` -- FastAPI app, endpoints, startup/shutdown (init_db/close_db)
  - `chatbot.py` -- `LibraryChatbot`, `IntentDetector`, `LanguageDetector`, `EmbeddingUtils`, `Config`
  - `database.py` -- PostgreSQL connection pool (`ThreadedConnectionPool`), schema initialization
  - `embeddings.py` -- Centralized OpenAI embedding generation (`embed_text`, `embed_texts`)
  - `cache.py` -- `ResponseCache` in-memory TTL+LRU cache for chatbot responses
  - `index_builder.py` -- `IndexBuilder` for building PostgreSQL tables from CSV data + chunk pipeline
  - `admin.py` -- `AdminManager` for CRUD operations on PostgreSQL tables
  - `scraper_cleaner.py` -- HTML cleaning, noise removal, structured document extraction
  - `document_processor.py` -- Page type detection, FAQ extraction, section splitting, LLM enrichment
  - `chunker.py` -- Semantic chunking by headings/sections with overlap and metadata (500 char target)
  - `retriever.py` -- Hybrid retrieval: vector + keyword (phrase+plain tsquery, synonym expansion) + RRF + intent pre-filtering
  - `reranker.py` -- LLM-based reranking (gpt-4o-mini relevance scoring, min_score=0.5)
  - `query_rewriter.py` -- LLM-based query rewriting (follow-up resolution, Arabic→English, expansion)
  - `input_guard.py` -- Pre-processing safety: prompt injection detection + domain scope filtering
  - `verifier.py` -- Post-generation claim verification + safety validation
  - `analytics.py` -- `ChatLogger`, `AnalyticsComputer` for usage tracking
  - `chart_generator.py` -- `ChartGenerator` for matplotlib chart generation (22 charts, base64 PNG)
- **`frontend/`** -- React application
  - `src/App.js` -- Main shell, state management, API calls
  - `src/components/` -- Header, Footer, ChatWindow, ChatInput, MessageBubble, DebugPanel
  - `src/pages/AdminDashboard.js` -- Admin dashboard (System Status, Data Management, Analytics tabs)
  - `src/api.js` -- Fetch wrapper for backend API
  - `src/App.css` -- AUB-branded styles (includes RTL support for Arabic)
  - `src/i18n/` -- Translation files (en.js, ar.js)
  - `src/LanguageContext.js` -- Language state context
- **`data/`** -- Source data files (CSVs)
  - `library_faq_clean.csv` -- FAQ source data (question/answer pairs)
  - `Databases description.csv` -- Database recommendations (1000+ records)
- **`scripts/`** -- Standalone CLI scripts
  - `build_index.py` -- Standalone index builder (same logic as IndexBuilder)
  - `chat.py` -- CLI chatbot (standalone, bilingual, no web server)
  - `scrape_aub_library.py` -- Playwright scraper, stores in PostgreSQL
- **`docker-compose.yml`** -- PostgreSQL 16 + pgvector + app container

### Key Classes (backend/)
- **`LibraryChatbot`** -- Main orchestrator: queries pgvector tables, routes through intent detection, bilingual. Uses ResponseCache.
- **`ResponseCache`** -- In-memory TTL+LRU cache keyed on (search_query, language). 256 entries max, 1hr TTL. Cleared on reindex.
- **`IntentDetector`** -- Keyword regex-based intent classification (English + Arabic patterns)
- **`LanguageDetector`** -- Unicode character analysis for Arabic vs English detection
- **`EmbeddingUtils`** -- HTML cleaning and library content boilerplate stripping
- **`IndexBuilder`** -- Builds PostgreSQL tables from CSV source files with OpenAI embeddings
- **`AdminManager`** -- CRUD operations on PostgreSQL tables for admin dashboard
- **`ChatLogger`** -- Appends chat interaction logs to JSON-lines file (includes cache_hit field)
- **`AnalyticsComputer`** -- Reads chat logs and computes analytics on demand
- **`Config`** -- Collection names, confidence thresholds, language constants

### PostgreSQL Tables (pgvector, stored in Docker volume)
| Table | Documents (embedded text) | Metadata Columns |
|---|---|---|
| `faq` | question + answer text | `question`, `answer` |
| `databases` | `"{name}. {description}"` | `name`, `description` |
| `library_pages` | `"{title}\n\n{content}"` | `url`, `title`, `content` |
| `document_chunks` | chunk_text (semantic chunks) | `page_url`, `page_title`, `section_title`, `page_type`, `chunk_index` |
| `chat_conversations` | query + answer logs | `query`, `answer`, `language`, `chosen_source`, scores, `retrieved_chunks` (JSONB) |
| `chat_feedback` | admin feedback | `conversation_id`, `rating`, `corrected_answer`, `comment`, `embedding` |

All embedding tables have HNSW indexes on `embedding vector(1536)` using `vector_cosine_ops`.

### Source Data Files (under `data/`)
- **`library_faq_clean.csv`** -- FAQ source data (question/answer pairs)
- **`Databases description.csv`** -- Database recommendations (1000+ records)

## Configuration

- **`OPENAI_API_KEY`** -- Required; set in `.env` file or as environment variable
- **`DATABASE_URL`** -- PostgreSQL connection string (default: `postgresql://aub_library:aub_library_pass@localhost:5432/aub_library`)
- Confidence thresholds defined in `backend/chatbot.py` Config class
- React dev server proxies to backend via `"proxy": "http://localhost:8000"` in package.json
- AUB brand colors: maroon #840132, red #ee3524
- Routes: `/` = chat interface, `/admin` = admin dashboard

## Important Notes

- Never commit API keys or `.env` files
- Start PostgreSQL first: `docker-compose up -d`
- Rebuild indices (`python scripts/build_index.py`) when CSV source files change
- Library pages scraping is optional; the app works without it
- `chat_logs.json` is generated at runtime and gitignored
- `scripts/chat.py` and `backend/chatbot.py` have parallel implementations (CLI vs API) -- keep in sync
- IntentDetector uses keyword regex, not LLM -- fast but needs manual keyword maintenance for new languages
- For production: build frontend (`npm run build`), then run FastAPI (serves static files)
- Connection pooling: psycopg2 `ThreadedConnectionPool` (min=2, max=10)
