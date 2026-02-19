# Changelog

All notable changes to Paperless AI Analyzer are documented here.

---

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
