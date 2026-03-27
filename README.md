# AUB Libraries Assistant

A bilingual (Arabic & English) RAG-based chatbot that answers library FAQs, provides personalized database recommendations, and serves information scraped from the AUB Libraries website. Built with a React frontend, FastAPI backend, and PostgreSQL + pgvector for vector search.

## Features

- **FAQ Answering** -- Instant answers to common library questions using semantic search
- **Database Recommendations** -- Suggests relevant research databases based on topic and intent detection
- **Library Page Retrieval** -- Synthesizes answers from scraped AUB library website content using GPT-4o-mini
- **Bilingual Support** -- Arabic and English with automatic per-message language detection and RTL rendering
- **Admin Dashboard** -- Manage collections, view analytics, add/edit/delete entries, trigger re-indexing and re-scraping
- **Response Caching** -- In-memory TTL+LRU cache for fast repeated queries
- **Debug Panel** -- Inspect retrieval scores, chosen source, and language detection for each response

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, react-router-dom, react-markdown |
| Backend | FastAPI + Uvicorn (Python 3.8+) |
| Vector DB | PostgreSQL 16 + pgvector (cosine similarity, HNSW indexes) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| LLM | GPT-4o-mini (library page synthesis, Arabic translation) |
| Scraping | Playwright (headless Chromium) |
| Containerization | Docker Compose |

## Prerequisites

- Docker and Docker Compose
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

For local development without Docker:
- Python 3.8+
- Node.js 16+ and npm

## Quick Start (Docker)

The easiest way to run everything with a single command:

### 1. Configure environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-api-key-here
```

### 2. Run

```bash
docker-compose up --build
```

This starts PostgreSQL + pgvector and the application (backend + frontend) together. The app is available at `http://localhost:8000`. Embedding indices are built automatically on first startup.

## Local Development

### 1. Install dependencies

```bash
# Start PostgreSQL + pgvector
docker-compose up -d db

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Configure environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Build embedding indices

Required before first run, or after CSV source files change:

```bash
python scripts/build_index.py
```

### 4. Run the application

**Development (two terminals):**

```bash
# Terminal 1: Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend (proxies API calls to :8000)
cd frontend && npm start
```

The app opens at `http://localhost:3000`. The backend API is at `http://localhost:8000`.

**Production (single server):**

```bash
cd frontend && npm run build && cd ..
uvicorn backend.main:app --port 8000
```

FastAPI serves the React build and API from the same port.

### 5. (Optional) Scrape library website

To populate the library pages table with content from the AUB Libraries website:

```bash
python scripts/scrape_aub_library.py
```

This requires Playwright browsers (`playwright install chromium`). You can also trigger a re-scrape from the admin dashboard.

## Project Structure

```
.
├── backend/                    # FastAPI application
│   ├── __init__.py
│   ├── main.py                 # App entrypoint, API endpoints, startup
│   ├── chatbot.py              # LibraryChatbot, IntentDetector, LanguageDetector
│   ├── database.py             # PostgreSQL connection pool, schema init
│   ├── embeddings.py           # Centralized OpenAI embedding generation
│   ├── cache.py                # In-memory TTL+LRU response cache
│   ├── index_builder.py        # IndexBuilder for pgvector tables
│   ├── admin.py                # AdminManager for CRUD operations
│   └── analytics.py            # ChatLogger, AnalyticsComputer
├── frontend/                   # React application
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js              # Main shell, state, API calls
│       ├── App.css             # AUB-branded styles (includes RTL)
│       ├── api.js              # Fetch wrapper for backend API
│       ├── index.js            # React entry point
│       ├── LanguageContext.js   # Language state context
│       ├── i18n.js             # i18n configuration
│       ├── i18n/               # Translation files (en.js, ar.js)
│       ├── components/         # Header, Footer, ChatWindow, ChatInput,
│       │                       #   MessageBubble, DebugPanel
│       └── pages/
│           └── AdminDashboard.js
├── data/                       # Source data files
│   ├── library_faq_clean.csv   # FAQ source data (question/answer pairs)
│   └── Databases description.csv  # Database source data (1000+ records)
├── scripts/                    # Standalone CLI scripts
│   ├── build_index.py          # Build embedding indices
│   ├── chat.py                 # CLI chatbot (standalone, bilingual)
│   └── scrape_aub_library.py   # AUB library website scraper
├── docker-compose.yml          # PostgreSQL + pgvector + app containers
├── Dockerfile                  # Multi-stage build (React + Python)
├── requirements.txt            # Python dependencies
└── .env                        # API keys (not committed)
```

**Generated at runtime (not committed):**
- `chat_logs.json` -- Chat interaction logs for analytics

## Architecture

```
User (React UI or CLI)
    │
    ▼
POST /api/chat  { message }
    │
    ▼
FastAPI Backend
    │
    ├─── Response Cache (in-memory TTL+LRU)
    │    ├── HIT  → return cached answer immediately
    │    └── MISS → continue to retrieval pipeline
    │
    ├─── pgvector Query (cosine similarity via HNSW indexes)
    │    ├── faq table              (question text)
    │    ├── databases table        (name + description)
    │    └── library_pages table    (title + content)
    │
    ├─── Intent Detection (keyword regex, bilingual)
    │
    ├─── Language Detection
    │    (auto-detect from query text per message)
    │
    └─── Response Router
         ├── Database intent detected → DB recommendations
         ├── Library pages match      → GPT-4o-mini synthesis
         ├── FAQ match                → Direct answer
         ├── DB semantic match        → DB recommendations
         └── No match                 → Clarification prompt
    │
    ▼
JSON { answer, debug, detected_language }
    │
    ▼
React renders markdown (RTL for Arabic)
```

## API Endpoints

### Chat
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Send message, get answer + debug info |
| `GET` | `/api/health` | Backend status and collection counts |
| `GET` | `/api/status` | Check if indices exist |

### Admin
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/admin/collections` | List collections with counts |
| `GET` | `/api/admin/collections/{name}/entries` | Paginated collection entries |
| `POST` | `/api/admin/faq` | Add FAQ entry |
| `PUT` | `/api/admin/faq/{id}` | Update FAQ entry |
| `DELETE` | `/api/admin/faq/{id}` | Delete FAQ entry |
| `POST` | `/api/admin/database` | Add database entry |
| `PUT` | `/api/admin/database/{id}` | Update database entry |
| `DELETE` | `/api/admin/database/{id}` | Delete database entry |
| `DELETE` | `/api/admin/library-page/{id}` | Delete library page entry |
| `POST` | `/api/admin/reindex` | Rebuild all indices from CSV sources |
| `POST` | `/api/admin/rescrape` | Trigger background library website scrape |
| `GET` | `/api/admin/rescrape/status` | Check scrape progress |
| `GET` | `/api/admin/system-info` | System information |
| `GET` | `/api/admin/cache-stats` | Response cache statistics |
| `GET` | `/api/admin/analytics/summary` | Analytics summary |
| `GET` | `/api/admin/analytics/trends` | Daily conversation trends (30 days) |
| `GET` | `/api/admin/analytics/top-queries` | Most frequent queries |
| `GET` | `/api/admin/analytics/unanswered-queries` | Questions the bot could not answer |

## Configuration

### Environment Variables
| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | -- | OpenAI API key |
| `DATABASE_URL` | No | `postgresql://aub_library:aub_library_pass@localhost:5432/aub_library` | PostgreSQL connection string |

### Confidence Thresholds

Defined in `backend/chatbot.py` (`Config` class):

| Threshold | Default | Purpose |
|---|---|---|
| `FAQ_HIGH_CONFIDENCE` | 0.70 | High-confidence FAQ match |
| `FAQ_MIN_CONFIDENCE` | 0.60 | Minimum FAQ relevance |
| `DB_MIN_CONFIDENCE` | 0.45 | Minimum database relevance |
| `LIBRARY_MIN_CONFIDENCE` | 0.35 | Minimum library page relevance |

### PostgreSQL Tables (pgvector)

| Table | Embedded Text | Metadata Columns |
|---|---|---|
| `faq` | question text | `question`, `answer` |
| `databases` | `"{name}. {description}"` | `name`, `description` |
| `library_pages` | `"{title}\n\n{content}"` | `url`, `title`, `content` |

All tables have HNSW indexes on the `embedding vector(1536)` column using `vector_cosine_ops`.

## Rebuilding Indices

After updating CSV source files in `data/`:

```bash
python scripts/build_index.py
```

Or use the admin dashboard's "Re-index All Collections" button.

## Cost Estimation

Using `text-embedding-3-small` (~$0.02 per 1M tokens):
- Average query embedding: ~50 tokens
- Cost per query: approximately $0.000001

GPT-4o-mini is used only for library page synthesis and Arabic translation, adding minimal cost per relevant query.

Monitor usage at: https://platform.openai.com/usage

## License

This project is part of a capstone project at the American University of Beirut.
