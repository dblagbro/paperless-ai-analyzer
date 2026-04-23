# Architecture — Paperless AI Analyzer

> Last updated: 2026-04-23 — post-refactor v3.9.0 → v3.9.2

---

## Overview

Paperless AI Analyzer is a Flask-based AI assistant for legal/court case document analysis.
It sits alongside a Paperless-NGX instance, pulls documents from it, embeds them into a
Chroma vector store, and exposes a web UI for chat-based querying, Case Intelligence (CI)
pipeline runs, court document importing, and project management.

---

## Deployment Topology

Three permanent nodes, all running the same codebase at the same version:

| Node | Container | Port | Image |
|------|-----------|------|-------|
| Dev | `paperless-ai-analyzer-dev` | 8052 | locally built |
| QA (Jacob) | `paperless-ai-analyzer-jacob` | 8053 | locally built |
| Prod | `paperless-ai-analyzer` | 8051 | `dblagbro/paperless-ai-analyzer:latest` (Docker Hub) |

All three share the same host server. Deploy flow: **dev → jacob → prod**.
Prod image is pushed to Docker Hub, then pulled by the prod container.

---

## Source Layout

```
paperless-ai-analyzer/
├── analyzer/                        # Python package — all application code
│   ├── __init__.py                  # Package version (__version__)
│   ├── app.py                       # Flask app instance, middleware, auth, factories
│   ├── web_ui.py                    # Thin orchestrator: registers blueprints, server entry points
│   ├── main.py                      # DocumentAnalyzer class, poll loop, CLI entry point
│   ├── auth.py                      # Flask-Login user model + DB init
│   ├── db.py                        # Core SQLite (users, chat sessions, messages, upload history)
│   ├── paperless_client.py          # Paperless-NGX REST client
│   ├── vector_store.py              # ChromaDB wrapper (multi-project collections)
│   ├── state.py                     # Per-project state persistence (state_{slug}.json)
│   ├── profile_loader.py            # YAML analysis profile loader
│   ├── project_manager.py           # Project CRUD (SQLite), per-project Paperless config
│   ├── smart_upload.py              # AI-powered document upload with metadata extraction
│   ├── llm_usage_tracker.py         # Per-model token/cost tracking
│   ├── url_poller.py                # URL polling for new documents
│   ├── remote_downloader.py         # Download documents from remote URLs
│   ├── court_db.py                  # Court-specific SQLite tables (jobs, dockets)
│   │
│   ├── routes/                      # Flask Blueprint modules — one per API domain
│   │   ├── __init__.py
│   │   ├── auth.py                  # /login, /logout
│   │   ├── status.py                # /api/status, /api/recent, /health, /api/about
│   │   ├── profiles.py              # /api/profiles, /api/staging/*, /api/active/*
│   │   ├── chat.py                  # /api/chat, /api/chat/compare, /api/chat/sessions/*
│   │   ├── vector.py                # /api/vector/*
│   │   ├── documents.py             # /api/reprocess, /api/reconcile, /api/trigger, /api/logs, /api/search, /api/tag-evidence
│   │   ├── projects.py              # /api/projects/* (CRUD + config + provisioning + migration + documents)
│   │   ├── upload.py                # /api/upload/*
│   │   ├── ai_config.py             # /api/ai-config/*, /api/llm/*, /api/llm-usage/*
│   │   ├── users.py                 # /api/users, /api/me, /api/change-password
│   │   ├── system.py                # /api/containers, /api/smtp-settings, /api/bug-report, /api/system-health
│   │   ├── ci.py                    # /api/ci/*
│   │   ├── court.py                 # /api/court/*, /api/projects/<slug>/analyze-missing
│   │   ├── forms.py                 # /api/ai-form/parse
│   │   └── docs.py                  # /docs/*, /api/docs/ask
│   │
│   ├── services/                    # Cross-cutting business logic (no Flask, no routes)
│   │   ├── __init__.py
│   │   ├── ai_config_service.py     # load/save/get AI config, project config resolution
│   │   ├── smtp_service.py          # SMTP send helpers, welcome/manual email templates
│   │   ├── web_research_service.py  # DuckDuckGo search, URL fetch, Justia→CourtListener resolver (extracted from routes/chat.py, v3.9.1)
│   │   ├── vision_service.py        # Vision-AI PDF page extraction for RAG (extracted from routes/chat.py, v3.9.1)
│   │   ├── chat_branch_service.py   # Chat branch-tree computation (extracted from routes/chat.py, v3.9.1)
│   │   └── project_provisioning_service.py  # Docker-compose + nginx + Postgres provisioning for per-project Paperless instances (extracted from routes/projects.py, v3.9.1)
│   │
│   ├── case_intelligence/           # CI pipeline — all /api/ci/* backend logic
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # CIOrchestrator: composition-only (inherits 7 mixins from ci_phases/)
│   │   ├── ci_phases/               # CI phase implementations as mixins (v3.9.2 split)
│   │   │   ├── __init__.py
│   │   │   ├── directors_mixin.py   # D1 plan, Q questions, D2 synthesis
│   │   │   ├── managers_mixin.py    # parallel domain managers + workers
│   │   │   ├── specialist_mixin.py  # Tier 3+ forensic / discovery / witness
│   │   │   ├── tier4_mixin.py       # Senior Partner review
│   │   │   ├── tier5_mixin.py       # White Glove (deep forensics / trial / multi-model)
│   │   │   ├── writeback_mixin.py   # Paperless write-back, finding embedding
│   │   │   └── utils_mixin.py       # budget checkpoints, status, doc fetching
│   │   ├── db/                      # CI data access layer (split from single flat db.py)
│   │   │   ├── __init__.py          # Re-exports all public symbols
│   │   │   ├── schema.py            # init_ci_db(), recover_orphaned_runs()
│   │   │   ├── runs.py              # run lifecycle + shares + questions
│   │   │   ├── analysis.py          # entities, timeline, contradictions, theories
│   │   │   ├── authorities.py       # authority corpus ingestion + embedding
│   │   │   └── reports.py           # report CRUD
│   │   ├── web_researcher.py        # WebResearcher: web search + page fetching
│   │   ├── war_room.py              # WarRoom + TrialStrategist
│   │   ├── deep_financial_forensics.py
│   │   ├── multi_model_synthesis.py
│   │   ├── report_generator.py
│   │   ├── theory_planner.py
│   │   ├── entity_extractor.py
│   │   ├── entity_merger.py
│   │   ├── timeline_builder.py
│   │   ├── contradiction_engine.py
│   │   ├── discovery_analyst.py
│   │   ├── forensic_accountant.py
│   │   ├── financial_extractor.py
│   │   ├── witness_analyst.py
│   │   ├── authority_ingester.py
│   │   ├── authority_retriever.py
│   │   ├── jurisdiction.py
│   │   ├── task_registry.py
│   │   ├── job_manager.py
│   │   ├── budget_manager.py
│   │   └── provenance.py
│   │
│   ├── court_connectors/            # Court system integrations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pacer.py                 # PACER / ECF
│   │   ├── nyscef.py                # New York State Courts
│   │   ├── federal.py               # Federal court lookup
│   │   ├── recap_courtlistener.py   # CourtListener / RECAP (free)
│   │   ├── deduplicator.py
│   │   ├── credential_store.py
│   │   └── import_job.py
│   │
│   ├── cloud_adapters/              # Cloud storage integrations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── google_drive.py
│   │   ├── s3.py
│   │   ├── onedrive.py
│   │   └── dropbox_adapter.py
│   │
│   ├── llm/                         # LLM client abstraction
│   │   ├── __init__.py
│   │   └── llm_client.py            # Multi-provider client (Anthropic, OpenAI, Ollama, etc.)
│   │
│   ├── extract/                     # Document text extraction
│   │   └── unstructured_extract.py
│   │
│   ├── checks/                      # Deterministic analysis checks
│   │   └── deterministic.py
│   │
│   ├── forensics/                   # Risk scoring
│   │   └── risk_score.py
│   │
│   ├── templates/
│   │   ├── dashboard.html           # Single-page app shell (~2,900 lines HTML only)
│   │   ├── docs.html
│   │   ├── login.html
│   │   └── chat_export.html
│   │
│   └── static/
│       ├── css/
│       │   └── dashboard.css        # All dashboard styles (extracted from dashboard.html)
│       └── js/
│           ├── utils.js             # Shared: apiFetch, apiUrl, showToast, escapeHtml
│           ├── overview.js          # Overview tab + stats
│           ├── config.js            # Config tab: vector store, AI config, LLM, profiles
│           ├── chat.js              # Chat tab: sessions, messages, branching, compare
│           ├── upload.js            # Upload tab: file/URL/cloud/court import
│           ├── ci.js                # Case Intelligence tab
│           ├── users.js             # Users admin panel
│           ├── ai_form_filler.js    # AIFormFiller reusable widget
│           └── init.js              # DOMContentLoaded init, global tab switching
│
├── Dockerfile
├── requirements.txt
├── architecture.md                  # This file
├── design.md                        # UI/UX patterns and frontend conventions
├── contributing.md                  # Dev workflow, style guide, refactor rules
└── refactor-log.md                  # Running log of architectural changes
```

---

## Application Boot Sequence

```
main.py::main()
  ├── load_config()                   # Load .env / environment variables
  ├── StateManager()                  # Load/create state_{slug}.json files
  ├── ProfileLoader()                 # Load YAML analysis profiles
  ├── PaperlessClient()               # Connect to Paperless-NGX API
  ├── VectorStore()                   # Connect to ChromaDB
  ├── LLMClient()                     # Initialize LLM provider
  ├── ProjectManager()                # Init project SQLite DB
  ├── SmartUploader()
  ├── DocumentAnalyzer()              # Main analysis engine
  ├── start_web_server_thread()       # Start Flask in background thread
  │     └── web_ui.py → app.py        # create_app() → register blueprints
  └── DocumentAnalyzer.run()          # Main poll loop (foreground)
```

---

## Data Storage

| Store | Technology | Contents |
|-------|------------|----------|
| Core DB | SQLite (`analyzer.db`) | Users, chat sessions/messages, upload history |
| CI DB | SQLite (`ci.db`) | CI runs, findings, reports, authorities |
| Court DB | SQLite (`court.db`) | Import jobs, docket cache |
| Project DB | SQLite (`projects.db`) | Project metadata, Paperless configs |
| Vector Store | ChromaDB | Document embeddings per project (`paperless_docs_{slug}`) |
| State files | JSON (`state_{slug}.json`) | Sync state per project |
| Profiles | YAML (`profiles/`) | Analysis rule profiles |
| AI Config | JSON (`ai_config.json`) | Global + per-project LLM config |

---

## Key Design Decisions

### Global Flask App (not application factory pattern)
The Flask `app` object is created as a module-level global in `analyzer/app.py`.
Route modules import `app` via `from flask import current_app` (Blueprint pattern).
Dependencies (state_manager, project_manager, etc.) are set as `app.*` attributes in
`create_app()` and accessed via `current_app.*` in route handlers.

### Per-Project Isolation
Each project gets:
- Its own ChromaDB collection: `paperless_docs_{slug}`
- Its own state file: `state_{slug}.json`
- Optional: its own Paperless-NGX instance (URL + token)
- Its own LLM configuration (AI config per project per use-case)

### Case Intelligence Tiers
CI runs execute a phased pipeline:
- **Tier 1** — Basic extraction (entities, timeline, contradictions)
- **Tier 2** — Deep analysis (theories, disputed facts)
- **Tier 3** — Specialist reports (forensic accounting, discovery gaps, witness cards)
- **Tier 4** — Senior Partner review (war room synthesis)
- **Tier 5 White Glove** — Deep financial forensics + trial strategy + multi-model synthesis (Claude + GPT-4o parallel)

### Frontend Architecture
Single-page app built in vanilla JS inside a Jinja2 template.
Flask-injected variables are surfaced through `window.APP_CONFIG` in a small inline
`<script>` block; all feature JS lives in external static files.

---

## Runtime Configuration

All configuration via environment variables (`.env` or Docker Compose):

| Variable | Purpose |
|----------|---------|
| `PAPERLESS_URL` | Paperless-NGX base URL |
| `PAPERLESS_TOKEN` | Paperless-NGX API token |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) API key |
| `OPENAI_API_KEY` | OpenAI API key (optional, for multi-model) |
| `CHROMA_HOST` / `CHROMA_PORT` | ChromaDB connection |
| `URL_PREFIX` | Nginx reverse proxy path prefix |
| `PORT` | Flask listen port (default: 8051) |
| `SMTP_*` | Email notification settings |

---

## Next Recommended Refactor Targets

Tracked in `project_paperless_backlog.md`; summarised here for visibility:

1. **Split `analyzer/main.py`** (1,598 lines) — the `DocumentAnalyzer` class mixes
   poll-loop control with per-document processing. Natural split: `poller.py` +
   `document_processor.py`.
2. **Split `routes/ci.py`** (1,793 lines, 40 top-level functions) by CI phase:
   setup / runs / findings / reports. Currently one monolithic blueprint module.
3. **Split `routes/chat.py`** further (now 1,068 lines) by grouping the 20+ handlers:
   core `/api/chat`, session CRUD, branch/edit/leaf, sharing, export.
4. ~~Split `case_intelligence/orchestrator.py`~~ — **COMPLETED v3.9.2** (2026-04-23).
   Split into 7 mixin files under `ci_phases/`. orchestrator.py now 334 lines.

---

## Version History

See `CHANGELOG.md` for detailed per-version changes.
See `refactor-log.md` for architectural change history.
