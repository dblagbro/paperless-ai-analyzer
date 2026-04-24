# Architecture — Paperless AI Analyzer

> Last updated: 2026-04-23 — post-refactor v3.9.0 → v3.9.3 (all backlog items completed)

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
│   ├── main.py                      # CLI entry point + DocumentAnalyzer composition (inherits 2 mixins)
│   ├── poller.py                    # PollerMixin: poll loop, re-analysis, stale-embedding check (v3.9.3)
│   ├── document_processor.py        # DocumentProcessorMixin: per-doc analysis, vision AI, tag compilation (v3.9.3)
│   ├── auth.py                      # Flask-Login user model + DB init
│   ├── db.py                        # Core SQLite (users, chat sessions, messages, upload history)
│   ├── paperless_client.py          # Paperless-NGX REST client (default 15s session timeout + 10s health_check cache, v3.9.6)
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
│   │   ├── chat/                    # /api/chat, /api/chat/compare, /api/chat/sessions/* (v3.9.3 package split)
│   │   │   ├── __init__.py          # blueprint aggregator
│   │   │   ├── core.py              # /api/chat, /api/chat/compare
│   │   │   ├── sessions.py          # session CRUD, share/unshare, rename
│   │   │   ├── branching.py         # message edit, branch, set-leaf
│   │   │   └── export.py            # PDF export
│   │   ├── vector.py                # /api/vector/*
│   │   ├── documents.py             # /api/reprocess, /api/reconcile, /api/trigger, /api/logs, /api/search, /api/tag-evidence
│   │   ├── projects.py              # /api/projects/* (CRUD + config + provisioning + migration + documents)
│   │   ├── upload.py                # /api/upload/*
│   │   ├── ai_config.py             # /api/ai-config/*, /api/llm/*, /api/llm-usage/*
│   │   ├── users.py                 # /api/users, /api/me, /api/change-password
│   │   ├── system.py                # /api/containers, /api/smtp-settings, /api/bug-report, /api/system-health
│   │   ├── ci/                      # /api/ci/* (v3.9.3 package split)
│   │   │   ├── __init__.py          # blueprint aggregator
│   │   │   ├── helpers.py           # shared notification + LLM-client helpers
│   │   │   ├── setup.py             # status, jurisdictions, detect, goal/key guides, cost, authority
│   │   │   ├── runs.py              # run CRUD + lifecycle (start/cancel/interrupt/rerun/shares/questions)
│   │   │   ├── findings.py          # findings + tier-specific report views
│   │   │   └── reports.py           # custom report generation + PDF download
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
│   │   └── project_provisioning_service.py  # Docker-compose + nginx + Postgres provisioning for per-project Paperless instances (extracted from routes/projects.py, v3.9.1; throttled FIFO queue + single worker added v3.9.6, PROVISION_MIN_INTERVAL_SECS default 180s)
│   │
│   ├── routes/projects/              # Projects blueprint as a package (v3.9.7 split from 988-line projects.py)
│   │   ├── __init__.py               # Blueprint + side-effect imports
│   │   ├── core.py                   # CRUD + archive/unarchive + current-project (~260 lines)
│   │   ├── paperless_config.py       # per-project Paperless config + test-connect + doc-link (~100 lines)
│   │   ├── provisioning.py           # provision-snippets + provision-status + reprovision (~170 lines)
│   │   ├── migration.py              # migrate-to-own-paperless + migration-status + migrate-documents (~230 lines)
│   │   └── documents.py              # list/delete project docs, orphan-documents, assign-project, reanalyze (~240 lines)
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
│   │   ├── web_researcher.py        # Re-export shim (v3.9.8) → web_researchers/
│   │   ├── web_researchers/         # Split from 1,362-line web_researcher.py (v3.9.8)
│   │   │   ├── __init__.py                 # WebResearcher class: composes mixins + 3 orchestration methods (~210 lines)
│   │   │   ├── constants.py                # _RATE, _STATE_TO_CL, _ROLE_* mappings (~75 lines)
│   │   │   ├── http_utils.py               # _http_get, _http_post_json helpers (~60 lines)
│   │   │   ├── base.py                     # WebResearcherBaseMixin: __init__, throttle, jur helpers, dedup (~65 lines)
│   │   │   ├── providers_legal.py          # CourtListener + Caselaw + Lexis + vLex + Westlaw + Docket Alarm + UniCourt (~365 lines)
│   │   │   ├── providers_general.py        # DDG + GDELT + Brave + Google CSE + Exa + Perplexity + NewsAPI + Tavily + Serper (~330 lines)
│   │   │   └── providers_entities.py       # BOP + OFAC + SEC EDGAR + FEC + OpenSanctions + OpenCorporates + CLEAR (~285 lines)
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
│   │   ├── dashboard.html           # Single-page app shell (~1,100 lines after v3.9.7 partial extraction)
│   │   ├── partials/                # Dashboard tab includes — extracted from dashboard.html (v3.9.7)
│   │   │   ├── tab_config.html            # Config tab (~625 lines)
│   │   │   ├── tab_upload.html            # Smart Upload tab + Court Import wizard (~395 lines)
│   │   │   └── tab_case_intelligence.html # CI tab — Setup / Findings / Specialists / Tier 5 (~890 lines)
│   │   ├── docs.html                # User-manual shell (~590 lines after v3.9.9 page extraction)
│   │   ├── docs_pages/              # Per-page manual content — extracted from docs.html (v3.9.9)
│   │   │   ├── overview.html              # 68 lines
│   │   │   ├── getting_started.html       # 55
│   │   │   ├── projects.html              # 175
│   │   │   ├── upload.html                # 84
│   │   │   ├── chat.html                  # 110
│   │   │   ├── search.html                # 56
│   │   │   ├── anomaly_detection.html     # 63
│   │   │   ├── tools.html                 # 89
│   │   │   ├── configuration.html         # 108
│   │   │   ├── users.html                 # 73
│   │   │   ├── llm_usage.html             # 47
│   │   │   ├── api.html                   # 101
│   │   │   ├── case_intelligence.html     # 120
│   │   │   └── court_import.html          # 77
│   │   ├── login.html
│   │   └── chat_export.html
│   │
│   └── static/
│       ├── css/
│       │   └── dashboard.css        # All dashboard styles (extracted from dashboard.html)
│       └── js/
│           ├── utils.js             # Shared: apiFetch, apiUrl, showToast, escapeHtml
│           ├── overview.js          # Overview tab + stats
│           ├── config/              # Config tab — split from 2,361-line config.js (v3.9.8)
│           │   ├── core.js                 # Sub-tabs, tools, AI usage, vector store, SMTP (~465 lines)
│           │   ├── projects.js             # Projects CRUD + Paperless modal + provisioning (~905 lines)
│           │   ├── search.js               # Search & Analysis tab (~285 lines)
│           │   └── profiles_ai.js          # LLM profiles + AI config management (~705 lines)
│           ├── chat.js              # Chat tab: sessions, messages, branching, compare
│           ├── upload.js            # Upload tab: file/URL/cloud/court import
│           ├── ci/                  # Case Intelligence tab — split from 2,229-line ci.js (v3.9.8)
│           │   ├── setup.js                # Tier selector, 5-tier config, findings + elapsed timer (~755 lines)
│           │   ├── goal_assist.js          # Goal Assistant, meta header, web research (~650 lines)
│           │   ├── specialists.js          # Forensic, Discovery, Witnesses, War Room (~365 lines)
│           │   ├── tier5.js                # Deep Forensics, Trial Strategy, Multi-Model, Settlement (~385 lines)
│           │   └── report.js               # Report builder + tab open hook (~70 lines)
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

Historical splits (all shipped):
- ✅ `routes/chat.py` — 3 service modules (v3.9.1) + 4-file package (v3.9.3)
- ✅ `routes/projects.py` — provisioning extracted to service (v3.9.1)
- ✅ `case_intelligence/orchestrator.py` — 7 mixin files under `ci_phases/` (v3.9.2)
- ✅ `analyzer/main.py` — `poller.py` + `document_processor.py` mixins (v3.9.3)
- ✅ `routes/ci.py` — 5-file package under `routes/ci/` (v3.9.3)
- ✅ `routes/projects.py` — 5-file package under `routes/projects/` (v3.9.7)
- ✅ `templates/dashboard.html` — tab-config, tab-upload, tab-case-intelligence
  extracted to `templates/partials/` (v3.9.7)
- ✅ `static/js/config.js` — 4-file package under `static/js/config/` (v3.9.8)
- ✅ `static/js/ci.js` — 5-file package under `static/js/ci/` (v3.9.8)
- ✅ `case_intelligence/web_researcher.py` — 7-file mixin package under
  `case_intelligence/web_researchers/` (v3.9.8); original file kept as a
  2-line re-export shim for backward compatibility.
- ✅ `templates/docs.html` — 14 page partials under `templates/docs_pages/` (v3.9.9)
- ✅ `routes/court.py` — 5-file package under `routes/court/` (v3.9.9)

Outstanding candidates (ranked by current size vs. cost to split):

1. **`case_intelligence/ci_phases/managers_mixin.py` — 1,004 lines.** Cohesive
   (all methods implement the "Manager phase" of the CI pipeline and share
   extensive `self.*` state). Revisit only if it crosses ~1,500 lines.

2. **`analyzer/paperless_client.py` — 680 lines.** Single class with method
   groups (docs, tags, documents, polling). Not urgent; split only if it
   passes ~1,000 lines.

3. ~~`analyzer/routes/court.py` — 847 lines.~~ **Done v3.9.9** — now a
   5-file package at `routes/court/`.

4. **`analyzer/poller.py` — 623 lines.** Similar — single class, one concern.

No file in the code base is currently above 1,400 lines after the v3.9.7 +
v3.9.8 passes. The next split pass should be triggered by either (a) a file
passing 1,200 lines, or (b) a single file having five-plus distinct
responsibilities (e.g. the old `routes/chat.py`).

---

## Version History

See `CHANGELOG.md` for detailed per-version changes.
See `refactor-log.md` for architectural change history.
