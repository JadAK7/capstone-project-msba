# AUB Libraries Assistant

A bilingual (Arabic & English) RAG-based chatbot that answers library FAQs, provides personalized database recommendations, and serves information scraped from the AUB Libraries website. Built with a React frontend, FastAPI backend, and ChromaDB vector database.

## Features

- **FAQ Answering** -- Instant answers to common library questions using semantic search
- **Database Recommendations** -- Suggests relevant research databases based on topic and intent detection
- **Library Page Retrieval** -- Synthesizes answers from scraped AUB library website content using GPT-4o-mini
- **Bilingual Support** -- Arabic and English with automatic language detection and on-the-fly translation
- **Admin Dashboard** -- Manage collections, view analytics, add/edit/delete FAQ and database entries, trigger re-indexing
- **Debug Panel** -- Inspect retrieval scores, chosen source, and language detection for each response

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, react-router-dom, react-markdown |
| Backend | FastAPI + Uvicorn (Python 3.8+) |
| Vector DB | ChromaDB (embedded, persistent, cosine similarity) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| LLM | GPT-4o-mini (library page synthesis, Arabic translation) |
| Scraping | Playwright (headless Chromium) |

## Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd <project-directory>

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
python build_index.py
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

To populate the library pages collection with content from the AUB Libraries website:

```bash
python scrape_aub_library.py
```

This requires Playwright browsers to be installed (`playwright install chromium`).

## Project Structure

```
.
├── backend/                    # FastAPI application
│   ├── __init__.py
│   ├── main.py                 # App entrypoint, API endpoints, startup
│   ├── chatbot.py              # LibraryChatbot, IntentDetector, LanguageDetector
│   ├── index_builder.py        # IndexBuilder for ChromaDB collections
│   ├── admin.py                # AdminManager for CRUD operations
│   └── analytics.py            # ChatLogger, AnalyticsComputer
├── frontend/                   # React application
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js              # Main shell, state, API calls
│   │   ├── App.css             # AUB-branded styles
│   │   ├── api.js              # Fetch wrapper for backend API
│   │   ├── index.js            # React entry point
│   │   ├── LanguageContext.js   # Language state context
│   │   ├── i18n.js             # i18n configuration
│   │   ├── i18n/               # Translation files (en.js, ar.js)
│   │   ├── components/         # Header, Footer, ChatWindow, ChatInput,
│   │   │                       #   MessageBubble, DebugPanel
│   │   └── pages/
│   │       └── AdminDashboard.js
│   ├── package.json
│   └── .gitignore
├── build_index.py              # Standalone index builder (CLI)
├── chat.py                     # CLI chatbot (standalone)
├── scrape_aub_library.py       # AUB library website scraper
├── library_faq_clean.csv       # FAQ source data (question/answer pairs)
├── Databases description.csv   # Database source data (1000+ records)
├── requirements.txt            # Python dependencies
├── CLAUDE.md                   # AI assistant project instructions
├── .env                        # API keys (not committed)
└── .gitignore
```

**Generated at runtime (not committed):**
- `chroma_db/` -- ChromaDB persistent storage
- `chat_logs.json` -- Chat interaction logs for analytics

## Architecture

```
User (React UI or CLI)
    │
    ▼
POST /api/chat  { message, language? }
    │
    ▼
FastAPI Backend
    │
    ├─── ChromaDB Query (auto-embeds via OpenAI)
    │    ├── FAQ Collection       (question text)
    │    ├── Databases Collection  (name + description)
    │    └── Library Pages Collection (title + content)
    │
    ├─── Intent Detection (keyword regex, bilingual)
    │
    ├─── Language Resolution
    │    (explicit param > auto-detect from query text)
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
| `POST` | `/api/admin/reindex` | Trigger full re-index from CSV sources |
| `GET` | `/api/admin/system-info` | System information |
| `GET` | `/api/admin/analytics/summary` | Analytics summary |
| `GET` | `/api/admin/analytics/trends` | Daily conversation trends (30 days) |
| `GET` | `/api/admin/analytics/top-queries` | Most frequent queries |

## Configuration

### Environment Variables
- `OPENAI_API_KEY` -- Required. Set in `.env` or as a shell environment variable.

### Confidence Thresholds

Defined in `backend/chatbot.py` (`Config` class):

| Threshold | Default | Purpose |
|---|---|---|
| `FAQ_HIGH_CONFIDENCE` | 0.70 | High-confidence FAQ match |
| `FAQ_MIN_CONFIDENCE` | 0.60 | Minimum FAQ relevance |
| `DB_MIN_CONFIDENCE` | 0.45 | Minimum database relevance |
| `LIBRARY_MIN_CONFIDENCE` | 0.35 | Minimum library page relevance |

### ChromaDB Collections

| Collection | Embedded Text | Metadata |
|---|---|---|
| `faq` | question text | `{ question, answer }` |
| `databases` | `"{name}. {description}"` | `{ name, description }` |
| `library_pages` | `"{title}\n\n{content}"` | `{ url, title, content }` |

## Rebuilding Indices

After updating CSV source files:

```bash
python build_index.py
```

Or use the admin dashboard's "Re-index" button, which triggers the same operation via the API.

## Cost Estimation

Using `text-embedding-3-small` (~$0.02 per 1M tokens):
- Average query embedding: ~50 tokens
- Cost per query: approximately $0.000001

GPT-4o-mini is used only for library page synthesis and Arabic translation, adding minimal cost per relevant query.

Monitor usage at: https://platform.openai.com/usage

## License

This project is part of a capstone project at the American University of Beirut.
