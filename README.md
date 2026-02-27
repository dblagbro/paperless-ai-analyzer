# Paperless AI Analyzer

[![Docker Hub](https://img.shields.io/docker/v/dblagbro/paperless-ai-analyzer?label=Docker%20Hub)](https://hub.docker.com/r/dblagbro/paperless-ai-analyzer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Advanced AI-powered anomaly detection and risk analysis microservice for [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx). Automatically analyzes financial documents, bank statements, and invoices for inconsistencies, tampering, and anomalies using deterministic checks, image forensics, and optional LLM-powered analysis — with a secure multi-user web dashboard and persistent AI chat.

> **Looking for v1.x docs?** See [README-v1.md](README-v1.md)

---

## What's New in v3.6.2

### Chat Editing, Dual LLM, Log Details — and All Features Now Built-In

v3.6.2 adds quality-of-life improvements to AI Chat and import history, and removes all feature-flag gates so every feature works out of the box.

**AI Chat improvements:**
- **Edit past messages** — hover any user message to reveal a ✏️ pencil; click to edit the text in-place. "↩ Resend" discards the old assistant response and regenerates from the edited message.
- **Stop button** — a red ■ Stop button appears while a request is in-flight. Clicking it aborts immediately via `AbortController`.
- **Dual LLM compare ⚖️** — toggle compare mode to send the same question to both configured providers simultaneously. Results appear as tabbed panels (e.g. Anthropic | OpenAI) side-by-side. The primary response is saved to session history for continuity.
- **Clickable document links** — the system prompt now instructs the AI to always write `[Document #NNN]` so every document reference becomes a clickable "View in Paperless" link. The linkifier also catches fallback formats like `Doc 123`.

**Import history improvements:**
- **Court import log drawer** — click 📋 on any past import job to expand an inline log showing duration and the last 15 log lines.
- **Upload history "Link" column** — successfully uploaded files now show a 🔗 View link directly to Paperless. Failed imports show ❌ with an error tooltip on hover. Filenames link to the source URL when available.

**All features now standard (no env-var gating):**
- `COURT_IMPORT_ENABLED` and `CASE_INTELLIGENCE_ENABLED` flags removed — both features are always on.
- `LLM_ENABLED` now defaults to `true` — LLM runs automatically when an API key is present.

---

## What's New in v3.6.1

### Fully Automated Per-Project Paperless Provisioning

v3.6.1 completes the per-project isolation story: provisioning is now **100% automated** — no Docker Compose editing, no SQL commands, no manual steps.

**Zero-touch project creation:**
- Creating a project automatically starts a background provisioner that spins up `paperless-web-{slug}` and `paperless-consumer-{slug}`, creates a dedicated Postgres database and Redis DB index, generates credentials, writes an nginx location block, and wires the project — all in ~45 seconds.
- A live banner on the project card shows provisioning progress (`⏳ Provisioning… Waiting for Paperless web to start (12s elapsed, up to 8 min)…`) and turns green when complete.

**Reprovision button:**
- Any project with a Provision tab now has an **⚡ Auto-Provision Now** button that triggers a full reprovision in the background (safe to run on already-provisioned projects).

**Live health monitoring:**
- A 7th dot ("Project Instances") appears in the title bar health widget — it turns green/yellow/red based on the aggregate running status of all per-project `paperless-web-{slug}` containers.
- The Container Manager (Debug & Tools tab) now dynamically lists all per-project containers alongside the core stack. Per-project containers are added/removed automatically as projects are provisioned.
- Restart and log-view operations work for per-project containers.

**URL deduplication for batch imports:**
- The file URL importer (`Upload → From URL`) now checks `import_history` before downloading. If the exact file URL was already successfully imported, it returns `{duplicate: true}` and skips the download entirely — preventing duplicates when rescanning the same directory.

**Zero-migration upgrade:**
- Two new encrypted columns (`paperless_secret_key_enc`, `paperless_admin_pass_enc`) are added via idempotent `ALTER TABLE` on startup — no manual SQL required. All existing data (chat history, case intelligence, ChromaDB, documents) is preserved.

---

## What's New in v3.6.0

### Per-Project Paperless-ngx Instances (100% Back-End Separation)

v3.6.0 eliminates the single shared Paperless-ngx instance risk: each project can now have its own dedicated Paperless web + consumer containers. There is no longer any path for a tag accident, API quirk, or bug to contaminate another project's documents.

**Infrastructure provisioning (UI-assisted):**
- Click **⚙️ Paperless** on any project card → the **Provision** tab generates ready-to-paste Docker Compose services, nginx location block, and Postgres SQL for a new per-project instance.
- Auto-assigns the next free Redis DB index (0 = shared, 1+ = per-project).

**Connection management:**
- **Connect** tab — enter the internal container URL, API token (stored AES-256-GCM encrypted), and public base URL for "View in Paperless" links. Includes a **Test Connection** button.
- Credentials are encrypted using the same key derivation as court court credential storage.
- Each project's client is factory-cached (5-minute TTL) so credential changes are picked up automatically.

**Dedicated polling threads:**
- Each configured project gets its own daemon polling thread at startup, using its own `PaperlessClient` and `StateManager`. The default project continues using the main loop.

**"View in Paperless" links everywhere:**
- Manage Projects document table, Search/Analysis tab, and AI chat now all show clickable per-project links.
- AI chat automatically linkifies `[Document #NNN]` and `Document #NNN` patterns to the correct project's Paperless URL.

**One-click document migration:**
- The **Migrate** tab in the Configure Paperless modal starts a background migration: downloads all documents from the shared instance (tagged `project:{slug}`), uploads each to the new dedicated instance, waits for OCR, re-keys ChromaDB embeddings to the new doc IDs, updates `processed_documents`, patches chat history references, and updates `court_imported_docs`. Live progress bar shows `N / total` documents migrated.

---

## What's New in v3.5.5

### Reliable Upload-to-Analysis Pipeline

Court imports (and any upload that returns a Paperless task UUID) now automatically trigger full AI analysis as soon as OCR completes:

- **Background analysis thread** — after the upload loop finishes, a daemon thread resolves each task UUID to a Paperless doc ID (waiting up to 3 min for OCR), then runs AI analysis on every newly added document. The import job log shows live task-resolution progress and a final "AI analysis complete" line.
- **`paperless_task_id` tracking** — stored in `court_imported_docs` so partially-resolved imports survive restarts.

### Historical Backlog Recovery

If documents were uploaded in the past but never analyzed (e.g. the 746-doc federal court backlog):

- **"🔍 Analyze Missing" button** on each project card in Manage Projects — triggers a background scan of all Paperless docs tagged with that project, compares against ChromaDB, and runs AI analysis on anything not yet embedded.
- **`POST /api/projects/<slug>/analyze-missing`** — REST endpoint powering the button.

### Pipeline Visibility

Two new stat cards appear on Overview when relevant:

- **Awaiting OCR** — documents uploaded to Paperless but OCR not yet complete (resolves automatically as tasks finish).
- **Awaiting AI** — OCR complete in Paperless but not yet AI-analyzed (use Analyze Missing to recover).

The "Documents Analyzed" card is renamed **"AI Analyzed"** to clearly indicate it reflects the ChromaDB-embedded count.

---

## What's New in v3.1

### Case Intelligence AI *(Advanced/Admin role required)*
- **Director → Manager → Worker orchestrator** analyzes all project documents as a coordinated group across six domains: Entities, Timeline, Financial, Contradictions, Theories, Authorities
- **Goal Assistant** — "✨ Refine with AI" chat in the Setup tab helps you write a precise, targeted goal statement before starting a run
- **CI run sharing** — share any run with specific users; shared runs appear in their dropdown annotated with the owner's name
- **Scientific paper report** — Director synthesizes a 12-section report (Executive Summary through Appendices A–C); PDF export via Report Builder
- **Budget checkpoints** — email alerts fire at every 10% completion with projected cost vs. budget (under/on/over status)

### Per-Project AI Configuration
- Each project independently sets provider + model for Document Analysis, AI Chat, and Case Intelligence
- Primary + fallback per use-case; fallback chain: per-project → global key → system default
- Admin: copy a project's full config to another project with one click

### User Roles & Profile
- **Advanced role** — between Basic and Admin; enables Case Intelligence and power-user features. Set via Configuration → Users
- **👤 My Profile** — header button replaces "Change Password"; opens a full profile form for name, email, phone, address, job title, and password management

### Bug Fixes
- Fixed spurious "No AI API key configured" warning in AI Chat when global keys are configured
- Fixed Search & Analysis "Failed to load documents" crash caused by anomalies field type mismatch

---

## What's New in v2.1

### Complete Project Isolation
- Each project has its own **separate Chroma vector store collection** — AI Chat only searches the project you have selected
- **Chat history is project-scoped** — switching projects shows only conversations about that project
- **Document tracking is per-project** — analyzed-document counts and skip-lists are isolated per project
- Archived projects now appear in a dedicated **"🗄️ Archived"** section below active projects instead of disappearing

### User Manual
- Built-in **12-page user manual** at `/docs/` — accessible from the **📖 Users Manual** button in the header
- Each tab's **? Help** panel includes a direct link to the relevant manual section
- **Email Manual** button on the Edit User modal so admins can send the manual to any user
- New-user welcome emails include links to all major manual sections
- AI Chat automatically links to relevant manual pages when answering how-to questions

### Reconcile Index *(Debug & Tools tab)*
- New **🔁 Reconcile Now** button — cleans up stale records for documents deleted from Paperless
- Removes orphaned `processed_documents` DB records without re-analyzing anything
- Removes orphaned Chroma embeddings for the same deleted docs
- Reports how many documents are not yet analyzed or embedded

### AI Usage & Cost Sub-Tab *(v2.1.5)*
- **AI Usage** sub-tab under ⚙️ Configuration shows a daily bar chart of API token usage and cost
- Rendered in-browser with Canvas 2D (no external library) — blue bars for cost, amber overlay for call volume
- Data sourced from the persistent LLM usage tracker database

### Search Now Queries All Documents *(v2.1.6)*
- **Search & Analysis** previously only searched the last 100 documents analyzed in the current container session — results were empty after every restart
- Now queries the **Chroma vector store directly** (all embedded docs, full metadata) whenever a filter or text query is active
- Searches work immediately after startup — by document ID, keyword, title, anomaly type, or risk level
- Semantic similarity search (Cohere embeddings) is used for free-text queries with substring matching as a secondary pass

### Bug Fixes
- Fixed `session` not imported in Flask app — caused `NameError` on vector store API calls
- Fixed all `VectorStore()` calls in the web layer to pass `project_slug` from session
- Fixed docs sidebar links using bare paths instead of sub-path prefix
- Fixed welcome and manual emails sending links to the host root instead of the app sub-path
- Fixed Config sub-tabs (Vector Store, Notifications, Users) showing 130–420 px whitespace gap when switching — caused by a mismatched `</div>` that displaced the panes outside the config container
- Fixed Smart Upload silently dropping the project tag when uploading documents

---

## What's New in v2.0

### Multi-User Authentication
- Login-protected dashboard with **admin** and **basic** user roles
- Persistent sessions (7-day cookies, survive browser close/reopen)
- Per-deployment unique session cookies — running prod and dev on the same domain no longer causes session collisions

### Persistent AI Chat
- **ChatGPT-style sidebar** lists all your previous chat sessions
- Sessions are saved to SQLite and survive container restarts
- Auto-titles new chats from the first message
- **Rename** and **delete** sessions from the sidebar

### Chat Sharing & Export
- Share any chat session with another logged-in user by username
- Admin users can view all chats grouped by user
- **Export any chat to PDF** — rendered markdown, code blocks, tables, and full Unicode preserved

### User Management
- Admins manage users in the **Configuration tab** (add, edit role, deactivate)
- Full **CLI tool** (`manage_users.py`) for headless/scripted user management
- Soft-delete (deactivate) preserves history; reactivate at any time

### Bug Fixes
- Fixed JavaScript crash on dashboard load that prevented stats/refresh from running
- Fixed session cookies expiring on browser close
- Fixed prod/dev cookie name collision on shared domain
- API key inputs are now plain-text (no more browser password manager prompts)

---

## Features

### Core Analysis (v1.0+)
- **Deterministic Anomaly Detection**: Balance verification, duplicate detection, date validation
- **Image Forensics**: PDF tampering detection with risk scoring (0–100%)
- **AI-Assisted Analysis**: Optional Claude/GPT integration for narrative summaries
- **Profile-Based Processing**: YAML-configured document type matching
- **Automated Tagging**: Adds structured tags back to Paperless documents
- **Idempotent Processing**: Safe to re-run on the same documents

### Multi-Tenant & Management (v1.5+)
- **Multi-Project Support**: Isolated document collections with per-project tagging
- **Document Migration**: Move documents between projects with automatic tag management
- **LLM Usage Tracking**: Token counting, cost calculation, and usage analytics

### Smart Ingestion (v1.5+)
- **Smart Upload**: Drag-and-drop upload with AI-powered tagging and project assignment
- **URL Downloads**: Fetch documents from web URLs, Google Drive, Dropbox, OneDrive
- **Automated Polling**: Monitor URLs for new documents and auto-import

### Secure Multi-User Dashboard (v2.0)
- **Login Required**: All pages protected; redirects to `/login` when unauthenticated
- **Role-Based Access**: `admin` sees all users' chats and user management; `basic` sees only their own
- **Persistent Chat**: Sessions stored in SQLite, survive restarts and redeploys
- **Chat Sharing**: Share sessions with named users (not public links)
- **PDF Export**: Download any chat as a formatted PDF
- **Admin UI**: User management panel inside the Configuration tab

---

## Table of Contents

- [Quick Start](#quick-start)
- [First-Time Setup — Creating the Admin User](#first-time-setup--creating-the-admin-user)
- [Configuration — All Environment Variables](#configuration--all-environment-variables)
- [docker-compose.yml Reference](#docker-composeyml-reference)
- [User Management CLI](#user-management-cli)
- [Web UI Tour](#web-ui-tour)
- [Built-In User Manual](#built-in-user-manual)
- [Document Profiles](#document-profiles)
- [Nginx Reverse Proxy](#nginx-reverse-proxy)
- [How It Works](#how-it-works)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Security Notes](#security-notes)
- [FAQ](#faq)
- [License](#license)

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- An existing [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) installation
- A Paperless API token ([generate here](https://docs.paperless-ngx.com/api/))

### Minimal docker-compose.yml

```yaml
services:
  paperless-ai-analyzer:
    image: dblagbro/paperless-ai-analyzer:latest
    container_name: paperless-ai-analyzer
    restart: unless-stopped
    environment:
      PAPERLESS_API_BASE_URL: http://paperless-web:8000
      PAPERLESS_API_TOKEN: your_paperless_token_here
      WEB_UI_ENABLED: "true"
      WEB_HOST: 0.0.0.0
      WEB_PORT: 8051
    volumes:
      - ./profiles:/app/profiles
      - analyzer_data:/app/data
      - /path/to/paperless/media:/paperless/media:ro
    ports:
      - "8051:8051"

volumes:
  analyzer_data:
```

### Start and create your admin account

```bash
# Start the container
docker compose up -d paperless-ai-analyzer

# Create the first admin user
docker exec -it paperless-ai-analyzer python3 /app/manage_users.py create \
  --username admin --password yourpassword --role admin

# Open the dashboard
open http://localhost:8051
```

Log in with the credentials you just created. Done.

---

## First-Time Setup — Creating the Admin User

On a fresh install the database has no users. The web UI will show the login page immediately. You **must** create at least one admin user via the CLI before you can log in:

```bash
docker exec -it paperless-ai-analyzer python3 /app/manage_users.py create \
    --username admin \
    --password "YourSecurePassword!" \
    --role admin \
    --display-name "Administrator"
```

After that, log in at the dashboard URL. Additional users can be created via the CLI or through the **Configuration → User Management** panel (admin only).

---

## Configuration — All Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `PAPERLESS_API_BASE_URL` | Paperless-ngx API base URL, e.g. `http://paperless-web:8000` |
| `PAPERLESS_API_TOKEN` | API authentication token from Paperless |

### Analysis Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `POLL_INTERVAL_SECONDS` | `30` | How often to check for new documents (seconds) |
| `BALANCE_TOLERANCE` | `0.01` | Dollar tolerance for balance mismatch checks |
| `FORENSICS_DPI` | `300` | DPI for PDF rendering during forensics (150–600) |
| `STATE_PATH` | `/app/data/state.json` | Path to the processing state file |
| `PROFILES_DIR` | `/app/profiles` | Directory containing YAML document profiles |
| `ARCHIVE_PATH` | `/paperless/media/documents/archive` | Path to Paperless archived PDFs (read-only mount) |

### LLM / AI Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENABLED` | `true` | Set `"false"` to disable AI-assisted analysis entirely |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `LLM_API_KEY` | — | Your Anthropic (`sk-ant-...`) or OpenAI (`sk-...`) API key |
| `LLM_MODEL` | *(auto)* | Override the model, e.g. `claude-sonnet-4-5-20250929` or `gpt-4o` |
| `COHERE_API_KEY` | — | Optional Cohere API key for enhanced RAG embeddings |

### Web UI & Auth Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_UI_ENABLED` | `true` | Enable the web dashboard |
| `WEB_HOST` | `0.0.0.0` | Interface to bind the web server |
| `WEB_PORT` | `8051` | Port for the web server |
| `URL_PREFIX` | — | Sub-path prefix when serving behind a reverse proxy, e.g. `/paperless-ai-analyzer`. **Required for reverse proxy deployments.** Automatically namespaces session cookies to prevent collisions when multiple instances share a domain. |
| `FLASK_SECRET_KEY` | *(auto-generated)* | Flask session signing key. Auto-generated and saved to `/app/data/.flask_secret_key` on first run. Set explicitly if you need deterministic keys (e.g. multi-replica deployments). |

### Resource Limits (docker-compose deploy section)

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

---

## docker-compose.yml Reference

### Full example with all settings

```yaml
services:
  paperless-ai-analyzer:
    image: dblagbro/paperless-ai-analyzer:latest
    container_name: paperless-ai-analyzer
    restart: unless-stopped

    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

    environment:
      # --- Paperless connection (required) ---
      PAPERLESS_API_BASE_URL: http://paperless-web:8000
      PAPERLESS_API_TOKEN: ${PAPERLESS_API_TOKEN}

      # --- Analysis tuning ---
      POLL_INTERVAL_SECONDS: 30
      BALANCE_TOLERANCE: 0.01
      FORENSICS_DPI: 300
      ARCHIVE_PATH: /paperless/media/documents/archive
      STATE_PATH: /app/data/state.json
      PROFILES_DIR: /app/profiles

      # --- LLM / AI (optional) ---
      LLM_ENABLED: "true"
      LLM_PROVIDER: anthropic          # or: openai
      LLM_API_KEY: ${LLM_API_KEY}
      LLM_MODEL: claude-sonnet-4-5-20250929
      COHERE_API_KEY: ${COHERE_API_KEY} # optional, for Cohere embeddings

      # --- Web UI ---
      WEB_UI_ENABLED: "true"
      WEB_HOST: 0.0.0.0
      WEB_PORT: 8051

      # --- Reverse proxy (remove if not using one) ---
      URL_PREFIX: /paperless-ai-analyzer

      # --- Session security (optional: set explicitly for production) ---
      # FLASK_SECRET_KEY: ${FLASK_SECRET_KEY}

    ports:
      - "8051:8051"

    volumes:
      # Paperless media (read-only)
      - /path/to/paperless/media:/paperless/media:ro
      # Persistent data: state, ChromaDB, SQLite users/chat DB, secret key
      - analyzer_data:/app/data
      # Document profiles (YAML)
      - ./profiles:/app/profiles
      # Hot-reload analyzer code during development (remove in production)
      # - ./analyzer:/app/analyzer:ro

    networks:
      - paperless_network   # same network as Paperless-ngx

volumes:
  analyzer_data:

networks:
  paperless_network:
    external: true   # if Paperless runs in a separate compose project
```

### Running prod and dev side-by-side

Both instances can share the same domain safely. Set a different `URL_PREFIX` for each — the session and remember-me cookies are automatically namespaced using the prefix, so they never collide:

```yaml
  paperless-ai-analyzer:          # production
    environment:
      WEB_PORT: 8051
      URL_PREFIX: /paperless-ai-analyzer
    volumes:
      - analyzer_data:/app/data

  paperless-ai-analyzer-dev:      # development / staging
    environment:
      WEB_PORT: 8052
      URL_PREFIX: /paperless-ai-analyzer-dev
    volumes:
      - analyzer_data_dev:/app/data   # separate data volume
```

---

## User Management CLI

`manage_users.py` is a CLI tool that runs inside the container. Use it for scripted/headless user management or initial setup.

### Synopsis

```
docker exec -it <container> python3 /app/manage_users.py <command> [options]
```

### Commands

#### `create` — Create a new user

```bash
docker exec -it paperless-ai-analyzer python3 /app/manage_users.py create \
    --username alice \
    --password "SecureP@ss1" \
    --role admin \           # optional: admin or basic (default: basic)
    --display-name "Alice"   # optional: friendly display name
```

#### `list` — List all users

```bash
docker exec paperless-ai-analyzer python3 /app/manage_users.py list
```

Output:
```
ID    Username             Display Name              Role     Last Login           Active
--------------------------------------------------------------------------------------------
1     admin                Administrator             admin    2026-02-19 01:59     yes
2     alice                Alice                     basic    2026-02-18 14:30     yes
3     bob                  Bob                       basic    never                yes
```

#### `reset-password` — Change a user's password

```bash
docker exec -it paperless-ai-analyzer python3 /app/manage_users.py reset-password \
    --username alice \
    --password "NewP@ss2"
```

Works on both active and inactive users.

#### `deactivate` — Disable a user (soft delete)

```bash
docker exec -it paperless-ai-analyzer python3 /app/manage_users.py deactivate \
    --username bob
```

The user can no longer log in. Their chat history and sessions are preserved.

#### `activate` — Re-enable a deactivated user

```bash
docker exec -it paperless-ai-analyzer python3 /app/manage_users.py activate \
    --username bob
```

### User Roles

| Role | Can log in | Sees own chats | Sees all users' chats | Manages users |
|------|-----------|---------------|----------------------|---------------|
| `basic` | Yes | Yes | No | No |
| `admin` | Yes | Yes | Yes | Yes |

---

## Web UI Tour

### Login Page

All pages require authentication. Unauthenticated requests redirect to `/login` (or `/<url_prefix>/login` behind a proxy). Log in with your username and password.

### Dashboard Sidebar

The left sidebar lists your chat sessions, newest first:
- **+ New Chat** starts a fresh session
- Click any session title to reload its full history
- Hover a session to reveal **rename** (✏️) and **delete** (🗑️) actions
- Sessions shared with you show a **shared** badge
- Admins see all sessions grouped by username

### AI Chat

Type a natural-language question about your documents:
- "Show me all documents with balance mismatches"
- "What are the highest-risk documents in the last 30 days?"
- "Summarize the anomalies for vendor Acme Corp"

The chat uses RAG (retrieval-augmented generation) over your document corpus — answers are grounded in your actual documents.

**Chat header buttons:**
- **Share** 🔗 — share this session with another user by their username
- **Export PDF** 📥 — download the full conversation as a formatted PDF

### Header Bar

- **Current Project** selector — switches the active project context for AI Chat and all vector store operations
- **📖 Users Manual** — opens the built-in 12-page user manual in a new tab
- **? Help** — toggles a contextual help panel at the top of the current tab
- **🐛 Report Issue** — sends a bug report with optional log attachment
- **🔑 Change Password** / **Sign Out**

### Overview Tab

Real-time statistics: documents analyzed, anomalies detected, high-risk count. Vector store doc count reflects the currently selected project.

### Search & Analysis Tab

Search and filter all analyzed documents across the entire vector store — by document ID, title keyword, anomaly type, risk level, or free-text semantic query. Results are drawn directly from Chroma and are available immediately after startup. Click any row to expand the full AI summary and evidence.

### AI Chat Tab

Natural-language chat over your document corpus using RAG. Chat history is **project-scoped** — each project has its own isolated chat sessions.

### Manage Projects Tab

Create, edit, archive, and delete projects. Move documents between projects. Archived projects appear in a separate section below active ones.

### Smart Upload Tab

Upload documents via file drag-and-drop, direct URL, or cloud share link (Google Drive, Dropbox, OneDrive). Optional Smart Metadata pre-analyzes the document before upload.

### Configuration Tab

- **AI Settings** — Configure OpenAI or Anthropic API keys, model selection, and test connectivity
- **Vector Store** — View, search, and manage Chroma embeddings for the current project; manually trigger stale re-embedding
- **AI Usage** — Daily bar chart of LLM API token usage and cost sourced from the persistent usage tracker
- **Profiles** — Browse and activate document profiles
- **Notifications** — SMTP settings for email alerts and bug reports
- **User Management** *(admin only)* — Add, edit, deactivate users; send the user manual by email

### Debug & Tools Tab

- **Reprocess All** — clears analysis state and re-queues all documents
- **Reprocess Document** — re-queues a single document by Paperless ID
- **🔁 Reconcile Index** — removes stale DB records and Chroma embeddings for documents deleted from Paperless (no re-analysis)
- **Live Logs** — real-time tail of the last 200 analyzer log lines

---

## Built-In User Manual

A comprehensive 12-page user manual is built into the app at `/docs/` (or `/<url_prefix>/docs/`). Click **📖 Users Manual** in the header to open it.

Sections:
| Page | URL |
|------|-----|
| Overview | `/docs/overview` |
| Quick Start | `/docs/getting-started` |
| Projects | `/docs/projects` |
| Smart Upload | `/docs/upload` |
| AI Chat | `/docs/chat` |
| Search & Analysis | `/docs/search` |
| Anomaly Detection | `/docs/anomaly-detection` |
| Debug & Tools | `/docs/tools` |
| Configuration | `/docs/configuration` |
| User Management | `/docs/users` |
| LLM Usage & Cost | `/docs/llm-usage` |
| API Reference | `/docs/api` |

---

## Document Profiles

Profiles define how to classify and validate specific document types. They live in `profiles/active/` as YAML files.

### Profile Structure

```yaml
name: Chase Bank Statement
description: Personal checking account statements from Chase Bank
min_score: 0.4

matching:
  keywords:
    - CHASE
    - Checking Account
  patterns:
    - "Statement Period: \\d{2}/\\d{2}/\\d{4}"
  mime_types:
    - application/pdf

extraction:
  mode: hi_res           # hi_res | fast | elements
  table_min_rows: 3
  date_formats:
    - "%m/%d/%Y"
    - "%m/%d/%y"

validation:
  check_balance: true
  balance_tolerance: 0.01
  check_duplicates: true
  check_date_order: true

forensics:
  enabled: true
  min_risk_threshold: 40

ai_prompt: |
  Analyze this bank statement for unusual patterns.
  Focus on large transactions and suspicious activity.
```

### Creating a Custom Profile

```bash
# Copy an example
cp profiles/examples/bank_statement_generic.yaml profiles/active/my_bank.yaml

# Edit to match your document
vi profiles/active/my_bank.yaml

# Restart to load the new profile
docker compose restart paperless-ai-analyzer
```

### Staging Workflow

When a document doesn't match any active profile, the analyzer:
1. Generates a suggested profile in `profiles/staging/` based on document content
2. Tags the document with `needs_profile:unmatched`

Review the staging profile, move it to `profiles/active/` once customized, and restart.

---

## Nginx Reverse Proxy

To serve behind nginx at a sub-path, set `URL_PREFIX` in the container environment and add this to your nginx config:

```nginx
location /paperless-ai-analyzer/ {
    rewrite ^/paperless-ai-analyzer/(.*) /$1 break;
    proxy_pass http://localhost:8051;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

The `URL_PREFIX` environment variable tells the Flask app its own prefix so that `url_for()` generates correct URLs (e.g. `/paperless-ai-analyzer/login` instead of `/login`). It also automatically scopes session cookies to that path, which is how multiple instances can safely coexist on the same domain.

---

## How It Works

### Document Processing Pipeline

```
Paperless-ngx API
      │  Poll every POLL_INTERVAL_SECONDS
      ▼
Profile Matcher ──(no match)──► Generate Staging Profile
      │ match
      ▼
Unstructured Extract (tables, transactions, OCR)
      │
      ├──────────────────────────────┐
      ▼                              ▼
Deterministic Checks          Forensics Risk Scoring
- Balance verify              - Compression artifacts
- Duplicate detection         - Noise patterns
- Date ordering               - Edge anomalies
      │                              │
      └──────────────┬───────────────┘
                     ▼
             LLM Analysis (optional)
             - Narrative summaries
             - Additional anomaly flags
                     │
                     ▼
             Tag Compilation
                     │
                     ▼
          Write Tags → Paperless-ngx
```

### RAG / Vector Store

Every analyzed document is embedded into a ChromaDB vector store. The AI chat uses these embeddings to retrieve relevant documents before answering, grounding responses in your actual data.

### Session & Cookie Architecture

- Flask sessions are stored as signed cookies (server secret key in `/app/data/.flask_secret_key`)
- Sessions last 7 days by default (permanent cookies, survive browser close)
- When `URL_PREFIX` is set, cookies are automatically scoped to that path and given a unique name derived from the prefix — allowing multiple instances on the same domain without interference

### Tag Naming Scheme

| Prefix | Example | Meaning |
|--------|---------|---------|
| `analyzed:deterministic:v1` | — | Deterministic analysis complete |
| `analyzed:ai:v1` | — | AI analysis complete |
| `anomaly:*` | `anomaly:balance_mismatch` | Deterministic anomaly found |
| `aianomaly:*` | `aianomaly:suspicious_pattern` | AI-flagged anomaly |
| `forensics:risk_score:*` | `forensics:risk_score:high` | Forensics risk category |
| `needs_profile:*` | `needs_profile:unmatched` | No matching profile |

---

## Monitoring

### Web Dashboard

Access `http://localhost:8051` (or your proxy URL) after logging in.

### Command Line

```bash
# Follow live logs
docker compose logs -f paperless-ai-analyzer

# View current processing state
docker exec paperless-ai-analyzer cat /app/data/state_default.json

# List staging profiles
ls profiles/staging/
```

### Health Check

```bash
docker compose ps paperless-ai-analyzer

# Test Paperless connectivity from inside the container
docker exec paperless-ai-analyzer \
    curl -s -H "Authorization: Token YOUR_TOKEN" \
    http://paperless-web:8000/api/documents/?page_size=1 | python3 -m json.tool
```

---

## Troubleshooting

### Can't Log In / Login Page Loops

**No users exist yet** (fresh install):
```bash
docker exec -it paperless-ai-analyzer python3 /app/manage_users.py create \
    --username admin --password yourpassword --role admin
```

**Forgot password**:
```bash
docker exec -it paperless-ai-analyzer python3 /app/manage_users.py reset-password \
    --username admin --password newpassword
```

**Session not persisting after login**:
Check that `URL_PREFIX` matches your nginx `location` path exactly (including no trailing slash). A mismatch causes the cookie's `Path` to not match the requests the browser makes.

### Stats Show Zero / Dashboard Not Refreshing

This is usually a JavaScript initialization error. Check the browser console (F12 → Console) for `TypeError` messages. If upgrading from v1.x, hard-refresh the page (Ctrl+Shift+R) to clear cached JS.

### Logging Into Dev Kicks You Out of Prod (or Vice Versa)

Set a different `URL_PREFIX` for each instance. The URL prefix is used to namespace the session cookie name and path — without it, both containers write to the same `session` cookie on `Path=/` and overwrite each other.

```yaml
# Prod
URL_PREFIX: /paperless-ai-analyzer

# Dev
URL_PREFIX: /paperless-ai-analyzer-dev
```

### No Documents Being Processed

```bash
# Check API connectivity
docker exec paperless-ai-analyzer \
    curl -H "Authorization: Token YOUR_TOKEN" \
    http://paperless-web:8000/api/documents/?page_size=1

# Reset state to reprocess all documents
docker exec paperless-ai-analyzer rm /app/data/state_default.json
docker compose restart paperless-ai-analyzer
```

### Profile Not Matching

```bash
# Check match scores in logs
docker compose logs paperless-ai-analyzer | grep "Profile match"

# Check staging profiles for auto-generated suggestions
ls -la profiles/staging/
```

### LLM Not Working

```bash
# Verify LLM is enabled
docker exec paperless-ai-analyzer printenv | grep LLM

# Test Anthropic library
docker exec paperless-ai-analyzer python3 -c "import anthropic; print('OK')"
```

### High Memory Usage

```yaml
environment:
  FORENSICS_DPI: "150"        # Lower quality, faster
  POLL_INTERVAL_SECONDS: "60" # Poll less frequently
  LLM_ENABLED: "false"        # Disable if not needed
```

---

## Performance

### Resource Usage

| Component | Typical | Peak |
|-----------|---------|------|
| Memory | 500 MB – 1 GB | 2–4 GB (large PDFs) |
| CPU | Low (idle) | 4 cores (forensics + LLM) |
| Disk | 1–5 GB | Depends on corpus size |

### Benchmarks

Typical processing times (Intel i7, 16 GB RAM):

| Document Type | Pages | Time |
|---------------|-------|------|
| Bank Statement | 2 | 8–12 s |
| Credit Card | 5 | 20–30 s |
| Tax Form | 10 | 45–60 s |
| Invoice | 1 | 4–6 s |

*Add 2–5 s per document when LLM is enabled.*

---

## Security Notes

1. **Authentication**: All dashboard routes require login. There are no anonymous-accessible API endpoints.

2. **Role separation**: Basic users cannot see other users' chats, access user management, or view admin-only API endpoints (returns 403).

3. **Session signing**: The Flask secret key is stored in `/app/data/.flask_secret_key` (mode 0600). The data volume should be on encrypted storage in production. Alternatively, supply `FLASK_SECRET_KEY` as an environment variable from a secrets manager.

4. **LLM privacy**: Only extracted facts and anomaly findings are sent to LLM APIs. Raw document content and full PDF text never leave your infrastructure.

5. **Read-only media**: The Paperless media volume is mounted `:ro`. The analyzer never modifies your original documents.

6. **No code in profiles**: Profiles are pure YAML configuration. No arbitrary code is executed from profile files.

7. **Network isolation**: Recommended to run in the same Docker network as Paperless-ngx with no public exposure. Use nginx or another reverse proxy with TLS for external access.

8. **Audit trail**: All analysis actions are logged with timestamps and document IDs.

---

## FAQ

**Q: Does this modify my original documents?**
A: No. It only adds tags to Paperless via the API. Original PDFs are never modified.

**Q: Can I reprocess all documents?**
A: Yes. Delete `/app/data/state_default.json` inside the container and restart. The analyzer is idempotent.

**Q: What if I forget the admin password?**
A: Use `manage_users.py reset-password` from the host via `docker exec`.

**Q: Can I run without any LLM API key?**
A: Yes. Deterministic checks and forensics work fully offline. LLM is purely additive.

**Q: How accurate is the forensics risk scoring?**
A: It's a heuristic, not forensic proof. Use as an additional signal alongside human review.

**Q: Can I share a chat link publicly?**
A: No. Chat sharing in v2.0 is user-to-user only (both must have accounts). Public link sharing is not implemented.

**Q: Does the chat export preserve formatting?**
A: Yes. The PDF export renders markdown (bold, italic, code blocks, tables, bullet lists) and handles full Unicode.

**Q: How does cookie namespacing work with `URL_PREFIX`?**
A: Setting `URL_PREFIX=/paperless-ai-analyzer` causes the session cookie to be named `paperless_ai_analyzer_session` with `Path=/paperless-ai-analyzer/`. A second instance with `URL_PREFIX=/paperless-ai-analyzer-dev` gets `paperless_ai_analyzer_dev_session` with its own path — completely independent.

---

## Contributing

Contributions welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit (`git commit -m 'Add amazing feature'`)
5. Push (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Credits

Built with:
- [Unstructured](https://unstructured.io/) for document extraction
- [OpenCV](https://opencv.org/) for image analysis
- [Anthropic Claude](https://anthropic.com/) / [OpenAI GPT](https://openai.com/) for optional AI analysis
- [Flask](https://flask.palletsprojects.com/) + [Flask-Login](https://flask-login.readthedocs.io/) for web UI and authentication
- [ChromaDB](https://www.trychroma.com/) for vector search
- [WeasyPrint](https://weasyprint.org/) for PDF export
- [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) for document management

---

## Support

- **Issues**: [GitHub Issues](https://github.com/dblagbro/paperless-ai-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dblagbro/paperless-ai-analyzer/discussions)

---

*Built for professionals who need automated document analysis without compromising security or privacy.*
