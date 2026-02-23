# Changelog

All notable changes to Paperless AI Analyzer are documented here.

---

## v3.0.0 — 2026-02-23

### Added
- **Per-project AI configuration** — each project independently configures provider/model
  for Document Analysis, AI Chat, and Case Intelligence. Global API keys remain as fallback.
  Admin: copy config between projects. Fallback chain: per-project → global → system default.
- **Hierarchical CI Orchestrator** — Director → Managers → Workers replaces the linear
  5-phase runner. Director LLM plans domain assignments; N managers run in parallel
  (auto = min(6, docs÷20)); K workers per manager (auto = budget-scaled). Deterministic
  fallback plan used if Director LLM call fails.
- **CI budget checkpoints** — notifications fire every 10% completion (under/on/over_budget
  status); completion notification fires on run finish. Requires SMTP configuration.
- **Scientific paper CI report** — Director D2 synthesizes a 12-section report (Sections
  I–IX + Appendices A–C). PDF export via weasyprint.
- **Advanced user role** — between Basic and Admin. Enables CI and power-user features.
  Set via Configuration → Users.
- **ci_manager_reports table** — tracks per-domain manager status, findings, cost, timing.
- **7 new ci_runs columns** — director_count, manager_count, workers_per_manager,
  notification_email, notify_on_complete, notify_on_budget, last_budget_checkpoint_pct.

### Changed
- AI Settings (Configuration tab) redesigned: collapsible Global API Keys (admin-only) +
  per-project config table with 3 use-cases × primary/fallback provider+model.
- CI Setup tab: ORCHESTRATION TIERS and NOTIFICATIONS sections added.
- In-pane help panels: 8 new sub-tab panels for all config sub-tabs and both CI sub-tabs;
  `_refreshHelpPanel()` now context-aware for config and CI sub-tabs.
- Built-in manual (/docs/): Configuration page updated; User Management gains Advanced role;
  new Case Intelligence page added.

---

## [2.1.6] — 2026-02-22

### Bug Fixes

**Search & Analysis — Chroma-backed search**
- `api_search` was only searching `ui_state['recent_analyses']`, an in-memory list capped at 100 entries that resets to empty on every container restart — after any restart all searches returned no results and the **Has Anomalies** filter showed nothing
- Fix: when any query or filter is provided the endpoint now queries the Chroma vector store directly (all embedded docs with full metadata), falling back to the in-memory cache only if Chroma is unavailable
- Searching by document ID, keyword, title substring, anomaly type, or risk level now works immediately after startup — no warm-up period required
- Semantic similarity search (via Cohere embeddings) is used for free-text queries; exact substring matching is applied across title, brief summary, full summary, and anomaly fields as a secondary pass

---

## [2.1.5] — 2026-02-21

### New Features

**AI Usage & Cost sub-tab** *(Configuration tab)*
- New **AI Usage** sub-tab under ⚙️ Configuration showing daily token and cost history
- Bar chart rendered with the Canvas 2D API (no external library) — blue bars show daily API cost, amber dashed overlay shows API call volume
- Chart pulls data from the LLM usage tracker database and updates on tab open

**Move Documents — UX improvements**
- Move Documents dialog now closes automatically on success instead of requiring a manual dismiss
- Success action shows a non-blocking toast notification confirming the move

### Bug Fixes

**Config sub-tab layout — whitespace gap eliminated**
- Vector Store, Notifications, and Users sub-tabs showed 130–420 px of blank space between the sub-nav and content when switching tabs
- Root cause: an extra `</div>` tag inside the Profiles section was prematurely closing `#tab-config`, causing `cfg-vectorstore`, `cfg-smtp`, and `cfg-users` to be rendered as siblings of the tab container instead of children — each silently received a `flex: 1` share of the full panel height, leaving the active pane crammed into the bottom half
- Fix: removed the orphaned closing tag; `#tab-config` is now a flex-column where the sub-nav is a fixed header and the active pane scrolls in the remaining space; switching sub-tabs resets scroll position to the top
- Debug & Tools tab top padding reduced to match other tabs
- Config sub-nav no longer bleeds into adjacent tabs when the Config tab is hidden

**Smart Upload — project tag dropped on upload**
- Uploaded documents were not receiving their project tag in Paperless
- Root cause: `get_or_create_tag` returns a tag ID integer, not a dict; the upload handler was calling `.get('id')` on an integer (always `None`), so the project tag was silently discarded before the Paperless API call
- Fix: check `if proj_tag is not None` and append the integer directly

---

## [2.1.4] — 2026-02-20

### Bug Fixes

- **Tag creation failure no longer aborts document analysis** — if `get_or_create_tag` fails (e.g. Paperless returns 400 because the tag already exists but was missed by a paginated GET), the failed tag is now skipped with a warning instead of sending `None` as a tag ID to Paperless, which previously triggered a retry loop and caused the entire document analysis to fail
- Added recovery GET on 400 from tag POST — handles cases where the tag exists but wasn't found in the initial paginated lookup
- Removed dead duplicate `get_or_create_tag` definition that was silently overridden by the v1.5.0 version
- **Overview "total analyzed" count is now project-scoped** — previously always showed the total across all projects regardless of which project was selected; now reflects only documents analyzed within the current project
- **AI Chat pane** now shows a **"Title:"** label before the session title

---

## [2.1.3] — 2026-02-20

### Reconcile Index

New **🔁 Reconcile Index** button in the Debug & Tools tab:
- Fetches all current document IDs from Paperless
- Removes `processed_documents` DB records for docs that have been deleted from Paperless
- Removes Chroma embeddings for the same deleted docs
- Reports how many documents are not yet analyzed or not yet embedded
- Does **not** re-analyze or modify any documents — pure index cleanup
- New `POST /api/reconcile` endpoint (admin only), scoped to current project

---

## [2.1.2] — 2026-02-20

### Minor

- Renamed **Manual** header button to **📖 Users Manual**
- Fixed prod container requiring image rebuild to pick up template changes (no bind mounts)

---

## [2.1.1] — 2026-02-20

### Complete Project Isolation

Every layer of the stack is now fully isolated per project:

**Vector store (Chroma)**
- All `VectorStore()` calls in the web layer now pass `project_slug` from the Flask session
- AI Chat, vector document listing, delete, clear, and status endpoints all scope to the currently selected project
- Fixed: `session` was not imported in Flask app, causing `NameError` on all vector store API calls

**Chat history**
- `chat_sessions` table gained a `project_slug` column (migration-safe `ALTER TABLE`)
- `get_sessions()` filters by project — switching projects shows only that project's chat history
- `create_session()` stores the current project slug

**Document tracking**
- `processed_documents` table gained a `project_slug` column (migration-safe)
- `mark_document_processed()` stores the project slug per document
- `count_processed_documents()` and `get_analyzed_doc_ids()` accept optional `project_slug` filter
- Startup gap-fill scoped to config project slug

**Archived projects UI**
- Archived projects now appear in a dedicated "🗄️ Archived (N)" section below active projects
- `GET /api/projects` always returns all projects (active + archived); filtering is client-side
- Archived cards show only Restore and Delete buttons

---

## [2.1.0] — 2026-02-20

### User Manual

- 12-page built-in user manual at `/docs/` with sidebar navigation
- Pages: overview, getting-started, projects, upload, chat, search, anomaly-detection, tools, configuration, users, llm-usage, api
- **📖 Users Manual** link in the header (username row), always visible
- Each tab's **? Help** panel has a "📖 Full manual for this section →" link
- **📧 Email Manual** button on the Edit User modal — sends all 12 section URLs to the user
- Welcome email expanded to include links to all major manual sections
- AI Chat system prompt updated to include doc URLs so it references them in how-to answers

### Bug Fixes

- Fixed docs sidebar links rendering bare paths (missing URL sub-path prefix) — `url_prefix` in the docs route now uses `request.script_root` instead of `app.config.get('URL_PREFIX')`
- Fixed welcome and manual emails linking to host root instead of app sub-path — both now use `request.host_url + request.script_root`
- Fixed `'sqlite3.Row' object has no attribute 'get'` in the send-manual endpoint — `get_user_by_id()` result converted to `dict` before attribute access

---

## [2.0.4] — 2026-02-19

### Stale RAG embedding detection

When a document is embedded in the vector store (Chroma) before Paperless OCR has completed, the embedding captures empty or near-empty content. Once OCR finishes, Paperless updates the document's `modified` timestamp — but the vector store entry is never refreshed automatically.

This release adds automatic detection and repair of stale embeddings:

- **`paperless_modified` field in Chroma metadata** — every new embedding stores the Paperless document's `modified` ISO timestamp. This lets the analyzer compare what was current at embed time vs what Paperless reports now.
- **`check_stale_embeddings()` method** — scans all Chroma entries and re-analyzes (and re-embeds) documents where:
  - The stored `paperless_modified` is older than the current Paperless `modified` (OCR or an edit updated the document after embedding), OR
  - No `paperless_modified` is stored (embedded before v2.0.4) AND the Chroma document text is < 200 characters (indicating empty OCR at embed time).
  - Caps at 50 documents per run and filters to docs modified within the past 7 days to avoid flooding the Paperless API.
- **Periodic auto-check** — fires on the 1st incremental poll after reprocess-all completes, then every 10 subsequent polls (roughly every 5 minutes with the default 30 s poll interval). Skipped during active reprocess-all runs.
- **Manual trigger** — `POST /api/vector/reembed-stale` endpoint + **⟳ Re-embed Stale** button in Config → Vector Store Management.

---

## [2.0.3] — 2026-02-19

### Upload tab — multi-file / folder URL support

- **File or Folder URL** — The URL mode now accepts a directory URL in addition to a single file link. When a folder/index URL is entered, the backend fetches the HTML page, parses all `<a href>` links, and filters to compatible document types (PDF, images, Word, Excel, ODT, TXT, EML, and more). Executables, binaries, media files, and archives are excluded.
- **File-picker panel** — A checklist panel appears showing discovered files with coloured extension badges, filename, and file size. All files are pre-checked; a "Select all / none" toggle and per-file checkboxes let you choose exactly what to upload. Files upload sequentially with per-row ✅ / ❌ / ⏳ status icons.
- **New backend route** `POST /api/upload/scan-url` — probes a URL, returns `{type: "single"}` or `{type: "directory", files: [...]}`.

### Bug fixes

- **Help panel on AI Chat tab** — The `switchTab` function was identifying the active tab button via text-content matching with an incorrect `replace('-', ' & ')` transform, so `ai-chat` never matched `💬 AI Chat`. Replaced with `onclick`-attribute matching, which is exact for all tabs.
- **Help button label** — Button now shows **"? Help: Off"** / **"? Help: On"** to make the toggle state obvious.
- **Analyzed count persists across restarts** — `total_analyzed` was an in-memory counter rebuilt from the anomaly-detector's Paperless tag (applied by a different service). Added `processed_documents` table to `app.db`; every document analyzed by the AI is recorded there. On startup the count is loaded from the database instead of querying Paperless tags, so restarts no longer reset the counter.

---

## [2.0.2] — 2026-02-19

### About / Help / Bug Report

- **ℹ About modal** — Header button shows app version, component versions, and a link to GitHub.
- **? Help toggle** — Header button shows/hides a contextual help panel at the top of the currently active tab. Each tab has its own help text.
- **🐛 Report Issue modal** — Header button opens a bug report form with:
  - Severity selector (Low / Medium / High / Critical)
  - Free-text description field
  - Optional contact email for follow-up
  - Optional HAR file attachment (browser network capture)
  - Option to include last 60 lines of server logs
  - Sends email to `dblagbro@voipguru.org` via SMTP with all details
- **📧 Notifications sub-tab** (admin only, under ⚙️ Configuration) — SMTP settings form:
  - Host, Port, STARTTLS toggle, Username, Password, From Address, HELO hostname, Bug Report To address
  - Save and Send Test Email buttons
  - Settings stored in `/app/data/smtp_settings.json`
- New backend routes: `GET /api/about`, `GET|POST /api/smtp-settings`, `POST /api/smtp-settings/test`, `POST /api/bug-report`

## [2.0.1] — 2026-02-19

### Upload Tab Redesign

- Three-mode upload card: **File** (drag-and-drop), **URL** (with Basic/Token auth), **Cloud Link** (Google Drive, Dropbox, OneDrive share links auto-transformed to direct downloads)
- New `POST /api/upload/transform-url` endpoint — detects and rewrites cloud share links (Google Drive file/Docs/Sheets/Slides, Dropbox `?dl=0→dl=1`, OneDrive `1drv.ms` pass-through)
- New `GET /api/upload/history` endpoint — returns last 20 imports per user
- Upload now routes through the analyzer backend (`SmartUploader` / `paperless_client.upload_document`) instead of posting directly to Paperless
- Optional **Smart Metadata** toggle — AI analyzes document before upload and shows a preview card (title, type, tags, suggested project) for confirmation before submitting
- **Import history panel** below the upload card — filename, color-coded source badge (file / url / google_drive / dropbox / onedrive), timestamp, status; auto-refreshes every 10 s when the tab is active
- New `import_history` table in `app.db` with `log_import()` and `get_import_history()` helpers

### Manage Projects Tab Redesign

- **New Project button** in tab header — opens a modal with name, auto-generated slug (editable before save, locked after), description, and color swatch picker (10 presets + custom color input)
- Per-project **Edit** button — updates name, description, and color via `PUT /api/projects/<slug>`
- Per-project **Move Documents** button — opens a modal to migrate all or specific documents (by comma-separated IDs) from one project to another; calls the existing `POST /api/projects/migrate-documents` background job
- Per-project **Archive / Restore** toggle — soft-archives projects with a visual "Archived" badge and reduced opacity
- Per-project **Delete** button (suppressed for the `default` project) — confirmation modal with option to also remove analyzer state and vector data
- Project cards now show slug chip, storage size, and creation date alongside document count

---

## [2.0.0] — 2026-02-19

### Major: Multi-User Authentication & Persistent Chat

#### New Features

**Authentication & Authorization**
- Login-protected dashboard — all routes require authentication; unauthenticated requests redirect to `/login`
- Two user roles: `admin` (full access) and `basic` (own chats only)
- Persistent session cookies (7-day lifetime, survive browser close/reopen)
- Per-deployment cookie namespacing via `URL_PREFIX` — prod and dev on the same domain no longer overwrite each other's sessions
- Flask secret key auto-generated on first run and persisted to `/app/data/.flask_secret_key`

**Persistent AI Chat**
- ChatGPT-style left sidebar listing all chat sessions
- Sessions stored in SQLite (`/app/data/app.db`), survive container restarts
- Auto-titles new sessions from the first 60 characters of the first user message
- Rename and delete sessions from the sidebar
- Admins see all users' sessions grouped by username

**Chat Sharing**
- Share any chat session with another logged-in user by username
- Shared sessions appear in the recipient's sidebar with a "shared" badge
- Owner can remove sharing at any time via the share modal

**Chat Export to PDF**
- Export any session as a formatted PDF (`/api/chat/sessions/<id>/export`)
- Renders markdown: bold, italic, code blocks, tables, bullet lists
- Full Unicode support, page headers with session title and date

**Admin User Management**
- User management panel in the Configuration tab (admin-only)
- Add users, change roles, change passwords, deactivate users from the web UI
- Soft-delete (deactivate) preserves chat history; accounts can be reactivated

**CLI User Management (`manage_users.py`)**
- `create` — create a new user with username, password, role, optional display name
- `list` — tabular list of all users with last login and active status
- `reset-password` — reset any user's password (works on inactive users)
- `deactivate` — soft-disable a user account
- `activate` — re-enable a deactivated account

**New API Routes**
- `GET/POST /api/chat/sessions` — list or create chat sessions
- `GET/DELETE/PATCH /api/chat/sessions/<id>` — get, delete, or rename a session
- `POST /api/chat/sessions/<id>/share` — share with a user
- `DELETE /api/chat/sessions/<id>/share/<user_id>` — unshare
- `GET /api/chat/sessions/<id>/export` — download session as PDF
- `GET/POST/PATCH/DELETE /api/users` — user CRUD (admin only)

**New Files**
- `analyzer/auth.py` — Flask-Login `UserMixin` and `LoginManager` setup
- `analyzer/db.py` — SQLite schema and CRUD: users, chat_sessions, chat_messages, chat_shares
- `analyzer/templates/login.html` — standalone login page matching dashboard theme
- `analyzer/templates/chat_export.html` — PDF export template with inline CSS
- `manage_users.py` — CLI user management tool

#### Bug Fixes

- **Fixed**: JavaScript `TypeError` crash on dashboard load — `document.getElementById('share-modal')` was called before the element existed in the DOM, halting all initialization code (stats refresh, interval polling, session loading). Replaced with safe event delegation via `document.addEventListener`.
- **Fixed**: Stats and charts showing all zeros — caused by the JavaScript crash above; `refresh()` and `setInterval(refresh, 10000)` never ran.
- **Fixed**: Session cookies lost on browser close — added `session.permanent = True` via `before_request` hook and `PERMANENT_SESSION_LIFETIME = timedelta(days=7)`.
- **Fixed**: Multiple instances on the same domain (prod + dev) invalidating each other's sessions — sessions are now namespaced: cookie name and path are derived from `URL_PREFIX`.
- **Fixed**: Browser password manager prompting to save API keys — changed OpenAI and Anthropic API key inputs from `type="password"` to `type="text" autocomplete="off"`.

#### Dependencies Added
- `flask-login>=0.6.3` — user session management
- `mistune>=3.0.2` — server-side markdown-to-HTML for PDF export
- `weasyprint>=62.0` — HTML+CSS to PDF conversion

#### System Packages Added (Dockerfile)
- `libpango-1.0-0`, `libharfbuzz0b`, `libpangoft2-1.0-0`, `libpangocairo-1.0-0`
- `libcairo2`, `libgdk-pixbuf-2.0-0`, `libffi-dev`, `shared-mime-types`
*(required by WeasyPrint for PDF rendering)*

---

## [1.5.2] — 2026-02

### Rich Anomaly Evidence Display
- Anomaly evidence now shows actual dollar values and specific transaction details
- Clickable evidence tags expand to show the full anomaly context

---

## [1.5.1] — 2026-02

### Nginx Subpath Routing Fix
- Fixed URL generation when running behind a reverse proxy at a sub-path
- Restored project selector in the web UI after routing refactor
- Added `_ReverseProxied` WSGI middleware that reads `URL_PREFIX` env var and sets `SCRIPT_NAME`

---

## [1.0.2] — 2026-01

### Document Integrity Analysis
- Added document integrity checks to the deterministic analyzer
- Clickable evidence tags in the web UI for legal review workflows

---

## [1.0.1] — 2026-01

### Pagination & Document Links
- Fixed pagination for document lists > 100 items
- Made document entries in the dashboard link directly to Paperless-ngx

---

## [1.0.0] — 2025-12

### Initial Release
- Deterministic anomaly detection (balance, duplicates, date ordering)
- Image/PDF forensics risk scoring
- Optional LLM analysis (Claude / GPT)
- YAML document profiles with auto-staging
- Flask web dashboard
- ChromaDB vector store with RAG chat
- Multi-project support
- Cloud integrations (Google Drive, Dropbox, OneDrive)
- URL polling for automated document import
