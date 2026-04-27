# AUB Libraries Assistant

A bilingual (Arabic & English) RAG-based chatbot for the American University of Beirut Libraries. Answers FAQs, recommends research databases, and synthesises responses from scraped library website content. Built on a hybrid retrieval pipeline (vector + keyword + RRF), cross-encoder reranking, grounded generation, and post-generation claim verification — all guarded by an injection / scope filter and backed by a feedback loop that lets librarians correct bad answers.

## Features

- **Hybrid retrieval** — Dense (pgvector cosine) + sparse (PostgreSQL FTS with phrase/synonym expansion), merged via Reciprocal Rank Fusion
- **Query rewriting** — LLM-based follow-up resolution, Arabic→English translation for retrieval, expansion of short queries
- **Cross-encoder reranking** — Local reranker model scores top-K candidates; min-score threshold gates abstention
- **Multi-source priority** — `custom_notes` (faculty text) > `document_chunks` (scraped pages) > `faq` > `databases`, with trust boosts applied at rerank time
- **Grounded generation + claim verification** — Inline evidence-plan generation; abstain when context is insufficient; post-hoc verifier strips unsupported claims
- **Input guards** — Regex + embedding-based prompt-injection detection and out-of-scope filtering
- **Bilingual** — Per-message Arabic/English detection with RTL rendering; cross-lingual threshold offset for Arabic queries against English-indexed data
- **Two-tier cache** — In-memory exact-key + semantic similarity (cosine ≥ 0.95) cache, persisted to disk between restarts
- **Feedback loop** — Admin can correct answers; corrections retrieved by similarity (≥ 0.85) and replayed on near-duplicate questions
- **Escalations** — Students can hand off questions to a librarian; admins respond from the dashboard
- **Admin dashboard** — Auth-gated UI for collection CRUD, re-indexing, re-scraping, freshness checks, conversations, feedback, escalations, analytics, and 22 generated charts
- **Per-stage latency instrumentation** — `StageTimer` records timing for every pipeline stage; surfaced in debug payload and analytics

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, react-router-dom, react-markdown, rehype-sanitize |
| Backend | FastAPI + Uvicorn (Python 3.8+), slowapi rate limiting |
| Vector DB | PostgreSQL 16 + pgvector (HNSW, cosine), GIN FTS indexes |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim), configurable via env |
| LLM | OpenAI `gpt-4o-mini` (configurable; supports any OpenAI-compatible provider) |
| Resilience | tenacity retry + per-stage circuit breakers (`backend/llm_client.py`) |
| Scraping | Playwright (headless Chromium) |
| Eval | Custom LLM-as-judge metrics + Ragas; orchestrator runs full suite |

## Prerequisites

- Docker and Docker Compose
- An OpenAI API key

For local development without Docker:
- Python 3.8+
- Node.js 16+ and npm

## Quick Start (Docker)

### 1. Configure environment

Create a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
ADMIN_PASSWORD=your-admin-password   # required to log into the admin dashboard
DATABASE_URL=postgresql://aub_library:aub_library_pass@db:5432/aub_library
```

### 2. Run

```bash
docker-compose up --build
```

This starts PostgreSQL + pgvector and the application together. Visit `http://localhost:8000`. Indices are built automatically on first startup.

## Local Development

### 1. Install dependencies

```bash
# PostgreSQL + pgvector (host port 5433 → container port 5432)
docker-compose up -d db

# Python deps
pip install -r requirements.txt

# Frontend deps
cd frontend && npm install && cd ..
```

### 2. Configure environment

Create `.env` at project root. For local dev the DB host is `localhost:5433`:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://aub_library:aub_library_pass@localhost:5433/aub_library
ADMIN_PASSWORD=your-admin-password
```

### 3. Build embedding indices

Required before first run, and any time CSV source files change:

```bash
python scripts/build_index.py
```

### 4. Run

Two terminals (recommended for development):

```bash
# Terminal 1 — backend (auto-builds indices if missing)
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend (proxies API to :8000)
cd frontend && npm start
```

App opens at `http://localhost:3000`. API at `http://localhost:8000`.

Single-server production mode:

```bash
cd frontend && npm run build && cd ..
uvicorn backend.main:app --port 8000
```

FastAPI serves the React build and the API from the same port.

### 5. (Optional) Scrape the library website

```bash
python scripts/scrape_aub_library.py
```

Requires `playwright install chromium` once. Re-scrape can also be triggered from the admin dashboard.

### 6. (Optional) Content freshness check

```bash
python scripts/freshness_check.py            # check + auto-rescrape if >20% drift
python scripts/freshness_check.py --dry-run  # check only
```

### 7. CLI chatbot (no web server)

```bash
python scripts/chat.py
```

## Project Structure

```
.
├── backend/                       # FastAPI application
│   ├── main.py                    # FastAPI app, all 54 endpoints, startup/shutdown
│   ├── chatbot.py                 # LibraryChatbot orchestrator, LanguageDetector, Config
│   ├── intent_classifier.py       # Unified intent rules (EN + AR keyword regex)
│   ├── retriever.py               # Hybrid vector + keyword retrieval + RRF
│   ├── reranker.py                # LLM-based reranking, source-trust boosts
│   ├── query_rewriter.py          # Follow-up resolution, AR→EN, expansion
│   ├── input_guard.py             # Injection detection + domain scope filter
│   ├── grounding.py               # Answerability classifier + inline-verified generation
│   ├── verifier.py                # Post-generation claim-level verification
│   ├── evaluation.py              # RAG metrics (groundedness, faithfulness, etc.)
│   ├── source_config.py           # Multi-source priority + trust weights
│   ├── stage_timer.py             # Per-stage latency instrumentation
│   ├── llm_client.py              # Resilient chat-completion gateway (tenacity + breakers)
│   ├── embeddings.py              # Centralized embedding generation
│   ├── cache.py                   # Two-tier (exact + semantic) response cache, disk-persisted
│   ├── content_extractor.py       # HTML/text → structured documents (replaces scraper_cleaner)
│   ├── document_parser.py         # .docx → plain text for ingestion
│   ├── chunker.py                 # Structure-aware semantic chunking
│   ├── index_builder.py           # CSV + scraped pages → pgvector tables
│   ├── admin.py                   # AdminManager for CRUD operations
│   ├── analytics.py               # ChatLogger + AnalyticsComputer
│   ├── chart_generator.py         # 22 matplotlib charts as base64 PNGs
│   └── database.py                # psycopg2 ThreadedConnectionPool, schema init
│
├── frontend/                      # React application
│   └── src/
│       ├── App.js                 # Main shell, state, API calls
│       ├── api.js                 # Fetch wrapper
│       ├── LanguageContext.js     # Language state context
│       ├── i18n/                  # en.js, ar.js translation files
│       ├── components/            # Header, Footer, ChatWindow, ChatInput,
│       │                          # MessageBubble, DebugPanel, EscalationModal
│       └── pages/AdminDashboard.js
│
├── data/
│   ├── library_faq_clean.csv      # FAQ source (Q/A pairs)
│   ├── Databases description.csv  # Database recommendations (1000+ rows)
│   ├── golden_set.json            # Eval golden questions
│   ├── ablation_stress_set.json   # Ablation stress questions
│   ├── feedback_similarity_set.json
│   ├── guard_redteam_set.json     # Injection / red-team test cases
│   └── intent_labels.json         # Intent classifier labels
│
├── scripts/
│   ├── build_index.py             # Build all pgvector tables from CSVs + scraped pages
│   ├── chat.py                    # CLI chatbot (no web server)
│   ├── scrape_aub_library.py      # Playwright scraper
│   ├── freshness_check.py         # Live-content drift detection + auto-rescrape
│   └── eval/                      # Evaluation suite (eval_*, run_*, generate_*, etc.)
│
├── docker-compose.yml             # PostgreSQL 16 + pgvector + app
├── Dockerfile                     # Multi-stage React + Python build
├── requirements.txt               # Python dependencies
└── .env                           # Secrets (gitignored)
```

**Generated at runtime (gitignored):**
- `data/.response_cache.pkl` — persisted response cache
- `eval_run_*/` — evaluation result directories
- `chat_logs.json` — legacy chat log file (current logs live in the `chat_conversations` table)

## Pipeline (v3)

```
React UI / CLI
    │
    ▼  POST /api/chat { message, language?, history? }
FastAPI
    │
    ├── 1. Input guards (regex + embedding injection detection, scope filter)
    ├── 2. Query rewriting  (LLM: follow-up resolution, AR→EN, expansion)
    ├──────────  Cache lookup (exact key, then semantic ≥ 0.95)
    │            HIT → return immediately
    ├──── MISS:
    │       ├── Admin-feedback correction lookup (pgvector ≥ 0.85)
    │       │   FOUND → use corrected answer (re-rendered by LLM)
    │       └── NOT FOUND:
    │             ├── 3. Intent classification → table / page_type pre-filter
    │             ├── 4. Hybrid retrieval per source
    │             │      • Vector: single embedding reused across tables
    │             │      • Keyword: phraseto_tsquery (×2 boost) + plainto_tsquery + synonyms
    │             │      • RRF merge (65% vector / 35% keyword) — pure quality, no trust bias
    │             ├── 5. Cross-encoder rerank + source-trust boost
    │             │      Dedup (Jaccard 0.85) → top 15 → score → min_score gate
    │             ├── 6. Sufficiency check (top-score gates: confident / partial / abstain)
    │             ├── 7. Grounded generation (uses ORIGINAL query for user language)
    │             │      System prompt: role-locked, self-check, cite-or-remove
    │             │      temperature=0.0, top_p=0.85
    │             └── 8. Claim verification (LLM strict + regex safety, fail-safe fallback)
    │
    └─→  Cache store, log conversation
        │
        ▼  JSON { answer, debug, detected_language, conversation_id }
React renders markdown (RTL for Arabic)
```

## API Endpoints

### Public

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Send message, get answer + debug info |
| `POST` | `/api/feedback` | Submit feedback on a conversation |
| `POST` | `/api/escalate` | Student hands off question to a librarian |
| `GET`  | `/api/health` | Status + collection counts |
| `GET`  | `/api/status` | Whether indices exist |

### Admin (auth-required, under `/api/admin/`)

Authentication: `POST /api/admin/login` returns a bearer token; all admin routes require `Authorization: Bearer <token>`.

Collections & ingest: `collections`, `collections/{name}/entries`, `faq` (CRUD), `database` (CRUD), `custom-note` (CRUD), `documents/upload`, `documents`, `document-chunks/search`, `document-chunk/{id}` (DELETE), `library-page/{id}` (DELETE), `reindex`, `rescrape`, `rescrape/status`, `freshness-check`, `freshness-status`, `system-info`, `cache-stats`, `clear-cache`, `backup`, `restore`.

Conversations & feedback: `conversations`, `conversations/{id}`, `feedback`, `feedback/all`, `feedback/{id}`, `feedback/stats`.

Analytics: `analytics/summary`, `analytics/trends`, `analytics/top-queries`, `analytics/unanswered-queries`, `analytics/latency`, `analytics/charts`.

Evaluation: `evaluation/run`, `evaluation/single`.

Escalations: `escalations`, `escalations/{id}` (DELETE).

54 endpoints in total — see `backend/main.py` for canonical definitions.

## Configuration

### Required environment variables

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `ADMIN_PASSWORD` | — | Required to use the admin dashboard |
| `DATABASE_URL` | `postgresql://aub_library:aub_library_pass@localhost:5433/aub_library` | host port 5433 (Docker maps `5433:5432`) |

### Optional

| Variable | Default | Purpose |
|---|---|---|
| `ADMIN_USERNAME` | `admin` | |
| `ADMIN_TOKEN_SECRET` | random per-process | Set explicitly to keep tokens valid across restarts |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | `text-embedding-3-large` (3072-dim) is also supported; schema auto-resizes |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Any OpenAI-compatible model |
| `LLM_BASE_URL` | (OpenAI) | Point at Groq, Together, Ollama, OpenRouter, Azure, etc. |
| `LLM_API_KEY` | falls back to `OPENAI_API_KEY` | Provider key if `LLM_BASE_URL` is set |

### PostgreSQL tables

| Table | Embedded text | Notes |
|---|---|---|
| `faq` | question + answer | FAQ pairs from CSV |
| `databases` | `"{name}. {description}"` | Database recommendations |
| `library_pages` | `"{title}\n\n{content}"` | Whole pages from scraper |
| `document_chunks` | chunk_text | Semantic chunks (preferred over `library_pages` for retrieval) |
| `custom_notes` | `"{label}\n\n{content}"` | Faculty-authored highest-trust source |
| `chat_conversations` | — | Logged chat turns (with retrieved chunks JSONB) |
| `chat_feedback` | — | Admin/student feedback, with embedding for similarity lookup |
| `escalations` | — | Librarian hand-off requests |

All embedding tables have HNSW indexes (`vector_cosine_ops`) and GIN indexes for full-text search.

### Confidence thresholds

Defined in `backend/chatbot.py` (`Config` class) and `backend/source_config.py`. Top-level retrieval gating now happens at the rerank stage, not at these per-collection scores — see `backend/reranker.py` for the active min-score logic.

## Rebuilding indices

After CSV / scraped content updates:

```bash
python scripts/build_index.py
```

Or click "Re-index All Collections" in the admin dashboard. Re-indexing also clears the response cache.

## Evaluation

The eval suite lives under `scripts/eval/`. Most common entry points:

```bash
# Full suite — produces an eval_run_<timestamp>/ directory
python scripts/eval/run_all_evals.py

# Faster subset
python scripts/eval/run_all_evals.py --quick

# Specific evals only
python scripts/eval/run_all_evals.py --only golden baselines threshold

# Just the curated golden set
python scripts/eval/run_golden_eval.py

# Ablation study (disable one stage at a time)
python scripts/eval/run_ablation.py
```

Individual eval scripts in `scripts/eval/eval_*.py` are runnable on their own and accept `--output <path>`.

## License

Capstone project at the American University of Beirut.
