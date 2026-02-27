# Changelog

All notable changes to Paperless AI Analyzer are documented here.

---

## v3.6.2 — 2026-02-27

### Added
- **Chat message editing** — hover over any user message to reveal a ✏️ pencil button; click to edit the message in-place with a textarea. "↩ Resend" truncates the conversation at that point and regenerates the assistant response.
- **Stop button** — a red "■ Stop" button appears next to Send while a request is in-flight; clicking it aborts the request via AbortController and shows "⚠️ Request stopped."
- **Dual LLM comparison** — ⚖️ toggle button in the chat input bar enables compare mode. Sends the query to both configured LLM providers in parallel; results shown as tabbed Anthropic / OpenAI panels side-by-side. Primary response is saved to session history for conversation continuity.
- **Court import log drawer** — each row in the Court Import History table now has a 📋 button that expands an inline log drawer showing duration, error message (if any), and the last 15 log lines from `job_log_json`.
- **Upload history improvements** — added "Link" column with 🔗 View links to Paperless for successfully uploaded documents; failed imports show ❌ with error tooltip on hover; filenames linked to `original_url` when available.

### Changed
- **AI document references** — system prompt now formats document lists as `[Document #NNN]` and explicitly instructs the AI to use this format for all references, enabling consistent click-to-open links.
- **Linkify regex expanded** — `_linkifyDocRefs` now also catches `Doc NNN` and `Doc. #NNN` fallback patterns in addition to the canonical `[Document #NNN]` format.
- **`append_message()`** — now returns the `lastrowid` of the inserted message so the frontend can attach edit buttons to user bubbles.
- **`api_chat_session_get`** — messages now include `id` field so the edit endpoint can be targeted correctly after session restore.

### Fixed
- Removed test projects (playwright-test, banner-test, banner-test2, court-import-test): containers stopped/removed, Postgres DBs dropped, nginx confs deleted, host data dirs removed, DB records deleted.

---

## v3.6.1 — 2026-02-27

### Added
- **Automated per-project Paperless provisioning** — when a new project is created (or "Auto-Provision Now" is clicked in the Paperless modal), the analyzer automatically:
  1. Reads shared config from the running `paperless-web` container via Docker SDK
  2. Assigns the next free Redis DB index by scanning running `paperless-web-*` containers
  3. Generates a random `secret_key` (64-hex) and `admin_password` (URL-safe 16-char)
  4. Creates the per-project Postgres database on `paperless-postgres` (idempotent)
  5. Creates host directories under `/mnt/s/documents/paperless-{slug}/`
  6. Starts `paperless-web-{slug}` and `paperless-consumer-{slug}` containers with `unless-stopped` restart policy
  7. Waits up to 3 minutes for the web container to become ready (HTTP 200/302/401/403)
  8. Obtains an API token via `POST /api/token/`
  9. Writes a nginx location block to `/app/nginx-projects-locations.d/paperless-{slug}.conf` and reloads nginx
  10. Saves `paperless_url`, `paperless_token`, `paperless_doc_base_url`, `paperless_secret_key`, and `paperless_admin_password` to the project DB (all sensitive values AES-256-GCM encrypted)
- **`GET /api/projects/<slug>/provision-status`** — live polling endpoint; returns `{status, phase, error, doc_base_url}`. Status values: `idle`, `queued`, `running`, `complete`, `error`.
- **`POST /api/projects/<slug>/reprovision`** — trigger auto-provisioning for an existing project (used by the "Auto-Provision Now" / "Reprovision" button).
- **`_provision_status` dict** — in-memory provisioning state, mirrors the `_migration_status` pattern.
- **`paperless_secret_key_enc` / `paperless_admin_pass_enc` DB columns** — two new encrypted columns in the `projects` table (idempotent migrations). Stored and retrieved via `update_project` / `get_paperless_config`.
- **Auto-Provision button in the Provision tab** — replaces the "manual steps" description with a prominent green "⚡ Auto-Provision Now" button. Manual snippets collapsed into an expandable `<details>` section.
- **Provisioning progress banner** — after project creation or clicking Auto-Provision, the Paperless modal shows a live yellow→green/red status banner polling every 3 s. On completion, shows a direct "Open Paperless →" link.
- **Provision status resume** — reopening the Paperless modal while provisioning is in progress resumes the polling banner automatically.
- **`nginx/projects-locations.d/`** — new host directory; nginx container now mounts it at `/etc/nginx/projects-locations.d/` (rw), and the analyzer-dev container at `/app/nginx-projects-locations.d/` (rw). The nginx.conf 443 server block now includes `projects-locations.d/*.conf`.

### Changed
- **`provision-snippets` infra names** — compose snippet now uses correct names (`paperless-postgres`, `paperless-redis`, host-path volumes under `/mnt/s/documents/paperless-{slug}/`) instead of the old generic `postgres-master`/`redis` names.
- **`update_project`** — now accepts `paperless_secret_key` and `paperless_admin_password` (plaintext → encrypted on write). Allowed-fields list updated.
- **`get_paperless_config`** — returns `secret_key` and `admin_password` (decrypted) in addition to existing fields.
- **`api_create_project`** — now immediately queues auto-provisioning in a daemon thread and opens the Paperless modal with the provision progress banner after creation.

### Fixed
- **Provision snippets postgres_sql** — removed misleading `CREATE USER` with placeholder password; the shared `paperless` DB user is reused and the DB is created automatically by the provisioner.

---

## v3.6.0 — 2026-02-26

### Added
- **Per-project Paperless-ngx instances** — each project can now point to its own dedicated Paperless web + consumer containers. 100% back-end document separation with zero tag-accident risk.
- **`projects` table schema** — 3 new columns (`paperless_url`, `paperless_token_enc`, `paperless_doc_base_url`) added via idempotent migrations. Token stored AES-256-GCM encrypted using the same key derivation as court credentials.
- **`ProjectManager.get_paperless_config(slug)`** — returns `{url, token, doc_base_url}` with decrypted token; never exposes the encrypted BLOB in public dicts.
- **`_get_project_client(slug)` helper** — module-level TTL-cached factory in `web_ui.py`. Returns a dedicated `PaperlessClient` for projects with their own URL+token, falls back to the global client for projects without.
- **`POST /api/projects/<slug>/paperless-config`** — save internal URL, API token (encrypted on write), and public base URL. Clears client cache entry.
- **`GET /api/projects/<slug>/paperless-config`** — return current config (token masked to `token_set: true/false`).
- **`GET /api/projects/<slug>/provision-snippets`** — auto-generates ready-to-paste Docker Compose services, nginx location block, and Postgres SQL for a new per-project Paperless instance. Redis DB index auto-assigned.
- **`POST /api/projects/<slug>/paperless-health-check`** — test URL + token without saving; used by the Connect tab.
- **`GET /api/projects/<slug>/doc-link/<int:doc_id>`** — resolve a doc ID to its public Paperless URL for a given project.
- **"View in Paperless" links** — all document tables (Manage Projects docs, Search/Analysis tab) now show per-project "↗ View" or clickable `#ID` links using `paperless_doc_base_url`. Falls back gracefully when no base URL is configured.
- **AI chat doc reference linkification** — `[Document #NNN]`, `Document #NNN`, and `doc #NNN` patterns in assistant messages are automatically wrapped in `<a href>` links to the per-project Paperless URL.
- **Per-project polling threads** — `DocumentAnalyzer.start_project_pollers()` launches a daemon thread (`_poll_project_loop`) for each project that has its own Paperless URL+token. The default project continues using the main poll loop.
- **"Configure Paperless" modal** — gear button on each project card. Tabbed UI with Provision (infra snippets + copy buttons), Connect (URL/token/base URL form + test connection), and Migrate tabs.
- **Document migration** — `POST /api/projects/<slug>/migrate-to-own-paperless` starts a background migration: download from shared Paperless → upload to new instance → wait OCR → re-key ChromaDB embeddings → update `processed_documents` → patch chat history doc ID references → update `court_imported_docs`.
- **`GET /api/projects/<slug>/migration-status`** — live polling endpoint for migration progress (total, migrated, failed, phase, status).
- **Migration progress UI** — progress bar + live count in the Migrate tab of the Configure Paperless modal. Polls every 2 seconds.
- **`upload_document_bytes(filename, content, title, tag_ids, created)`** — new `PaperlessClient` method for uploading raw bytes without a temp file on disk; used by migration.

### Changed
- **All project-scoped API endpoints** now use `_get_project_client(slug)` instead of `app.paperless_client`: delete document, `_analyze_missing_for_project`, `_post_import_analyze`, `/api/status` awaiting-counts.
- **`GET /api/projects` response** now includes `paperless_doc_base_url`, `paperless_url`, `paperless_configured`, and `global_paperless_base_url` per project.
- **Project card buttons** — "⚙️ Paperless" button added; turns green when instance is configured.
- **`update_project()`** — now accepts `paperless_url`, `paperless_token` (plaintext → encrypts on write), `paperless_doc_base_url`.
- **`PAPERLESS_PUBLIC_BASE_URL` env var** — new env var used as global fallback for all "View in Paperless" links when a project has no dedicated `paperless_doc_base_url`. Both `api_list_project_documents` and `api_recent` use this fallback.
- **`_paperlessDocUrl(slug, docId)` JS helper** — now falls back to `_globalPaperlessBase` (from `GET /api/projects` response) when the project has no per-project Paperless URL.
- **Analysis tab doc links** — previously used a hardcoded voipguru.org URL; now dynamically resolved from `analysis.paperless_link` or `_globalPaperlessBase`.

### Fixed
- **Project migration missing CI and court data** — `api_migrate_documents` (the "Move" feature on Manage Projects) now migrates ALL project-scoped data in addition to Chroma embeddings and Paperless tags: (5) Case Intelligence runs (`ci_runs.project_slug` in `case_intelligence.db` — child tables follow via `run_id`), (6) court import jobs and imported-doc records (`court_import_jobs` and `court_imported_docs` in `projects.db`). Previously only Chroma, Paperless tags, `processed_documents`, and `chat_sessions` were moved, leaving CI and court import history orphaned on the source project.
- **Court import "ghost imported" bug** — `_run_court_import` was recording documents as `status='imported'` even when `upload_document()` returned `None` (upload silently failed). The misleading `_log("uploaded as...")` also ran in this case. Now raises a `RuntimeError` for None results so the document is correctly recorded as `status='failed'`, preserving the ability to re-import via URL dedup.
- **PACER fee-gate Playwright download** — `_download_via_playwright` now correctly captures PACER's temp PDF URL from the response interceptor (avoiding `response.body()` which fails with "No resource with given identifier found"), then downloads the PDF via `context.request.get()` which shares the browser's cookies. This resolves the fee-confirmation flow: PACER loads `/doc1/<old>` (fee gate) → click "View Document" → navigates to `/doc1/<new>` (HTML wrapper) → browser loads `/cgi-bin/show_temp.pl?file=<random>.pdf` (actual PDF).
- **PACER docket form submission wait** — `_get_docket_direct` and `_get_docket_playwright` now use `wait_for_load_state('load')` after submitting the docket options form, replacing `'domcontentloaded'` which was resolving too early (before PACER finished writing the full docket HTML) and causing 0 entries to be returned.
- **ECF URL routing** — `FederalConnector.download_document()` now routes PACER ECF URLs (containing `uscourts.gov`) directly to the PACER connector, bypassing `CourtListenerConnector` which was downloading the fee-confirmation HTML and returning it as a valid path.
- **HTML detection in RECAP connector** — `CourtListenerConnector.download_document()` now detects HTML content (fee-gate pages or auth redirects) by inspecting `Content-Type` and magic bytes, returning `None` instead of saving HTML to a temp file.
- **PACER date format** — `_parse_pacer_docket_html` now normalises entry dates from `MM/DD/YYYY` (PACER format) to `YYYY-MM-DD` (ISO 8601) at the source. `_run_court_import` applies the same conversion as a safeguard before passing dates to Paperless-ngx.
- **Paperless upload error details** — `PaperlessClient.upload_document()` exception handler now includes the Paperless response body (first 400 chars) in the error log, making upload rejections (e.g., `{"document":["File type text/html not supported"]}`) immediately visible.

---

## v3.5.5 — 2026-02-26

### Added
- **Post-import AI analysis pipeline** — after a court import upload loop completes, a daemon thread (`_post_import_analyze`) resolves each Paperless task UUID to a doc ID (waiting up to 3 minutes per task for OCR to finish), then runs AI analysis on every newly uploaded document. Import job log now shows task-resolution progress and analysis completion lines.
- **`resolve_task_to_doc_id(task_id)`** — new `PaperlessClient` method that polls `GET /api/tasks/?task_id=<uuid>` until Paperless reports `SUCCESS` or timeout, returning the resolved `related_document` integer ID. Handles the Paperless-ngx v2+ behavior of returning a task UUID instead of a full doc dict on upload.
- **`get_project_document_count(project_slug)`** — new `PaperlessClient` method for a fast single-request count of all Paperless documents tagged `project:<slug>` (used by `/api/status`).
- **`_analyze_missing_for_project(project_slug)`** — scans all Paperless docs tagged `project:<slug>`, compares against ChromaDB IDs, and runs AI analysis on any not yet embedded. Returns the count analyzed.
- **`POST /api/projects/<slug>/analyze-missing`** — new endpoint that fires `_analyze_missing_for_project` in a background thread. Enables one-click recovery of the historical 746-doc backlog.
- **"🔍 Analyze Missing" button** — added to each project card in Manage Projects. Calls the new endpoint and shows a toast confirming the scan has started.
- **`paperless_task_id` column** — added to `court_imported_docs` table via migration in `init_court_db()`. Stores the Paperless upload task UUID so it can be polled for OCR completion.
- **`get_pending_ocr_count(project_slug)`** — returns count of court docs uploaded (task_id recorded) but not yet resolved to a Paperless doc_id.
- **`update_court_doc_task_resolved(task_id, doc_id)`** — marks a court_imported_doc as resolved by writing the Paperless doc_id for a given task_id.
- **"Awaiting OCR" stat card** — shown on Overview when `awaiting_ocr > 0`. Counts court docs whose Paperless task hasn't resolved yet.
- **"Awaiting AI" stat card** — shown on Overview when `awaiting_ai > 0`. Counts docs present in Paperless but not yet embedded in ChromaDB (computed as `paperless_total - chroma_count - awaiting_ocr`).

### Changed
- **`log_court_doc()`** — accepts new optional `paperless_task_id: str = ''` parameter written to the new column.
- **`_run_court_import()`** — now collects `task_id` (Paperless-ngx v2+) or `doc_id` (v1.x) from each upload result and fires the post-import analysis thread when the loop completes.
- **`/api/status`** — now returns `awaiting_ocr` and `awaiting_ai` counts alongside existing fields.
- **"Documents Analyzed" → "AI Analyzed"** — renamed stat card on Overview dashboard for clarity; distinguishes ChromaDB-analyzed count from raw Paperless document counts.

---

## v3.5.4 — 2026-02-26

### Fixed
- **Court-imported documents now visible in project view** — `GET /api/projects` and `GET /api/projects/<slug>` now include a `court_doc_count` field sourced from the `court_imported_docs` table (count of successfully imported entries for that project). Previously, the project "Manage Projects" card showed only the ChromaDB-analyzed document count, which is always 0 for court-imported documents (court docs are uploaded to Paperless-ngx but not run through AI analysis). Result: Manage Projects cards now show "X analyzed · Y court" counts separately so court-imported documents are visible.
- **Overview "Court Imported" stat card** — `/api/status` now returns `court_doc_count` for the current project. A new "Court Imported" stat card is shown on the Overview dashboard whenever `court_doc_count > 0`, hidden otherwise.

---

## v3.5.3 — 2026-02-26

### Fixed
- **PACER direct docket fetching** — `FederalConnector` now falls back to PACER CM/ECF when CourtListener returns 0 entries (previously only fell back on 403). Removes the incorrect assumption that 0 entries means no documents exist.
- **PACER docket date range** — the CM/ECF docket options form pre-fills a "last 2 weeks" date window by default. The connector now clears all date-range inputs before submitting, so the full case history is returned (not just recent entries).
- **PACER `pacer_case_id` lookup** — `CourtListenerConnector` now fetches the numeric `pacer_case_id` from the CL dockets API before the PACER fallback, enabling direct `DktRpt.pl?{pacer_case_id}` navigation. Avoids the `iquery.pl?1-L_0_0-1` endpoint which returns HTTP 500 on NYSB.
- **Duplicate `get_docket` stub** — a dead `return []` method in `PACERConnector` shadowed the real implementation (Python uses the last definition). Removed the stub; PACER docket fetching now executes.
- **`court_id` vs `court` in compound case_id** — `CourtListenerConnector._search_result_to_case()` was encoding the display name ("United States Bankruptcy Court, S.D. New York") instead of the short court code ("nysb"). The `ecf.{court_code}.uscourts.gov` URL requires the short code.
- **HTML tag attribute bleed in document titles** — `<td[^>]*>` split (was `<td[\s>]`) now strips full opening tag attributes so table cell attribute text (e.g. `valign="bottom">`) no longer appears in parsed titles.
- **PACER fallback also triggers on empty CL result** — `FederalConnector.get_docket()` now attempts PACER when CourtListener returns an empty list (in addition to the existing 403 RuntimeError path).

---

## v3.5.2 — 2026-02-26

### Fixed
- **NYSCEF connector reliability** — changed all Playwright waits from `networkidle` → `domcontentloaded` (Cloudflare blocks `networkidle`). County field in case search now uses `fill()` + autocomplete instead of `select_option()` (it is a text input, not a `<select>`). `get_docket()` now follows the DocumentList redirect when it is not the initial landing page. `authenticate()` now proceeds to login even when `public_only=true` if credentials are provided, instead of going anonymous and hitting CAPTCHA.
- **Delete document button** — `confirmDeleteDoc()` onclick attribute was silently broken: `JSON.stringify(title)` produces double-quotes that terminate the `onclick=""` HTML attribute, so the browser discarded it. Fixed by wrapping with `_escHtml()` → `&quot;` is properly decoded by the browser when the handler fires.
- **Document count consistency** — Overview dashboard and Manage Projects now always show the same count. `/api/status` previously used SQLite `count_processed_documents()` (drifts) while Manage Projects used ChromaDB `collection.count()` (live). Both now use ChromaDB.
- **Direct-port access with URL prefix** — `_ReverseProxied` WSGI middleware now strips the URL prefix from `PATH_INFO` when present. Previously, direct requests (not via nginx) failed with 404 because `PATH_INFO` still contained the prefix that Flask's router didn't expect.

### Improved
- **Project migration** — `api_migrate_documents()` now also migrates chat sessions (with their messages via ON DELETE CASCADE), uses a single batch SQL UPDATE for full-project migrations instead of a per-document loop, and refreshes the cached document count on both source and destination projects when done.
- **NYSCEF Pro Se / Party access UI** — "Pro Se / Party access" checkbox in the credential wizard step 2 for NYSCEF. Parties, defendants, and plaintiffs who are not attorneys create a free account at NYSCEF → Unrepresented Litigants. The credential fields stay visible (Pro Se users still need a username + password). Password show/hide toggle added to both PACER and NYSCEF password fields.
- **onnxruntime version constraint** — `==1.15.1` → `>=1.16.0`. 1.15.1 crashes with "cannot enable executable stack" on Linux kernels with strict NX enforcement (Ubuntu 22.04+). 1.24.x works fine despite the unstructured-inference `<1.16` spec warning.

---

## v3.5.1 — 2026-02-25

### Added
- **"📋 Paste credentials" shortcut button** — visible directly on the Court Import panel alongside the ⚙️ Manage button. Opens the credential wizard immediately in AI Paste mode, skipping step 1 entirely. Solves the discoverability problem where the AI paste feature was previously buried 4+ clicks deep.
- **Generic `POST /api/ai-form/parse` route** — replaces the court-specific `/api/court/credentials/parse` for AI field extraction. Accepts a dynamic schema so any form can use it. Uses `get_project_ai_config()` for per-project model selection and provider fallback; supports OpenAI and Anthropic.
- **`AIFormFiller` JS class** — reusable widget that renders a paste panel, manages multi-turn AI chat bubbles, and auto-fills form inputs from the AI response. Extracted as a standalone library at [github.com/dblagbro/ai-form-filler](https://github.com/dblagbro/ai-form-filler).

### Changed
- **Court wizard AI paste refactored** to use the new `AIFormFiller` class. Removed the one-off `_callCourtParse`, `_renderCourtChat`, `courtParseCredentials`, `courtPasteReply`, and `_courtPasteAutofill` functions and their state variables. The paste panel HTML simplifies to an empty container div; the widget injects its own UI.

---

## v3.5.0 — 2026-02-25

### Added
- **AI Paste credential entry** — "📋 AI Paste" tab in the court credential wizard step 2 lets users paste a raw email, Slack message, or attorney notes. AI extracts `court_system`, `username`, `password`, `pacer_client_code`, `courtlistener_api_token`, `nyscef_county` from free-form text, explains what it found in plain English, asks one follow-up question at a time if needed, and auto-fills the manual form when complete. New `POST /api/court/credentials/parse` route handles multi-turn conversation history; supports OpenAI and Anthropic providers.
- **Court Document Importer** — pull entire case files directly from federal courts (PACER / CourtListener RECAP) and NYS NYSCEF into a Paperless-ngx project library without manual downloading. Gated by `COURT_IMPORT_ENABLED=true` env var.
- **Free federal access via CourtListener RECAP** — the free CourtListener REST API (no auth required) is the primary source for federal documents; supports `COURTLISTENER_API_TOKEN` in `extra_config_json` to raise the 5K req/day anonymous limit.
- **PACER direct fallback** — when a document is not in the RECAP archive and PACER credentials are configured, the `FederalConnector` falls back to PACER session-cookie auth for downloading; 1-second rate limit between downloads; realistic user-agent header.
- **NYSCEF connector (Playwright)** — headless Chromium login → case search → docket scrape → cookie-replayed downloads. Requires `INCLUDE_PLAYWRIGHT=true` Docker build arg. All selectors centralised in `NYSCEF_SELECTORS` dict for easy one-line DOM-change fixes. Gracefully errors with a clear message when Playwright is not installed.
- **AES-256-GCM credential encryption** — court passwords encrypted at rest in `projects.db` using a key derived from the Flask secret key file. API responses never return the raw password.
- **3-tier deduplication** — Tier 1: source URL match in `court_imported_docs`; Tier 2: SHA-256 hash match; Tier 3: Paperless `title__icontains` search. Re-running an import on an existing case skips already-imported documents with a `skip_reason` logged.
- **Background import jobs** — `CourtImportJobManager` mirrors the `CIJobManager` daemon-thread + `threading.Event` pattern. Cancel signal propagates within one document boundary.
- **Setup wizard** — 3-step modal: select court → enter credentials + test connection → confirmation. "Test Connection" validates credentials before saving; PACER/NYSCEF test details shown inline.
- **Docket viewer with filter bar** — date/title filter, source badge filter (RECAP / PACER / NYSCEF), checkbox selection for partial imports, "Import Selected" and "Import All" buttons.
- **Live progress bar** — animated progress bar, imported/skipped/failed counters, last 20 log lines with auto-scroll, Cancel button.
- **Import history table** — case number, court system, doc counts (imported/skipped/failed), date, status badge, "Sync Again" link that reloads the docket.
- **11 new API routes**: `POST/GET/DELETE /api/court/credentials`, `POST /api/court/credentials/test`, `POST /api/court/search`, `GET /api/court/docket/<court>/<id>`, `POST /api/court/import/start`, `GET /api/court/import/status/<job_id>`, `POST /api/court/import/cancel/<job_id>`, `GET /api/court/import/history`.
- **3 new DB tables** in `projects.db`: `court_credentials`, `court_import_jobs`, `court_imported_docs` (with URL/hash/project indexes). All created via `init_court_db()` with `CREATE TABLE IF NOT EXISTS` + WAL mode.
- **`COURT_IMPORT_ENABLED`** injected into every Jinja2 template context to feature-flag the tab button and tab content.
- **Optional Playwright Docker build arg** — `ARG INCLUDE_PLAYWRIGHT=false`; set to `true` in dev service `docker-compose.yml` to include Chromium (+350–500 MB). Federal-only users omit it; NYSCEF features show a clear "Playwright not available" error.
- `cryptography>=42.0.0` and `playwright>=1.40.0` added to `requirements.txt`.

---

## v3.2.0 — 2026-02-24

### Added
- **CI email notifications fixed** — `notification_email`, `notify_on_complete`, and
  `notify_on_budget` are now correctly passed from the UI through the API and saved to the
  DB on run creation. Budget checkpoint and completion emails now fire as configured.
- **Enhanced CI progress bar** — live status line now shows active manager/worker counts,
  cumulative token usage (in+out), elapsed time, and ETA (linear extrapolation shown after
  ≥10% complete with a "~" prefix).
- **CI findings RAG-embedded** — on run completion all findings (entities, timeline, financial,
  contradictions, theories, authorities, disputed facts) are embedded into the project's Chroma
  vector store. AI Chat will cite CI findings when relevant; the Director skips re-extraction
  of already-known facts on subsequent runs of the same project.
- **War room briefing (Phase 1 → Phase 2 knowledge handoff)** — after Phase 1 managers
  (entities/timeline/financial) finish, the orchestrator builds a compact briefing of all
  extracted facts and injects it into every Phase 2 manager's context (contradictions,
  theories, authorities). Phase 2 agents now start with full situational awareness instead
  of deriving facts from scratch.
- **Opposing theory pass** — `_manager_theories()` generates a second set of theories from
  the opposing role (e.g. defense theories when your role is plaintiff) to surface the
  strongest counter-arguments. Saved with `role_perspective` set to the opposing role.
- **`opposing_theory_generation` task** added to the task registry (Tier 3, gpt-4o primary,
  Claude escalation, fixed-cost per run).
- **Vector store enrichment for CI workers (Lever 1)** — `_fetch_case_documents()` now
  bulk-retrieves prior AI analysis (brief summary, full summary, document type) from the
  project's Chroma vector store and attaches it to each document before extraction.
  `_run_worker()` prepends this pre-computed analysis as `[PRIOR AI ANALYSIS]` context
  ahead of the raw OCR text, dramatically improving extraction quality on large documents
  where OCR alone is truncated. 745 of 748 docs enriched in live testing.
- **`VectorStore.get_documents_metadata()`** — new bulk retrieval method that returns a
  `{doc_id: metadata_dict}` map for a list of Paperless document IDs, used by Lever 1.

### Fixed
- **CI contradiction engine now receives Phase 1 entities/events** — `_manager_contradictions()`
  previously passed hardcoded empty lists for `entities` and `events` on every document. Now
  queries the DB post-Phase-1 and groups results by document ID, so the contradiction engine
  receives real extracted data.
- **`_build_docs_summary()` now emits rich content** — the disputed facts matrix prompt
  previously received only entity/event counts per document (`"entities=3, events=2"`). It
  now receives content snippets, key party names, and key event descriptions per document,
  giving the LLM enough signal to identify genuine factual disputes.
- **Financial data no longer re-extracted in `_manager_theories()`** — the theory manager
  previously called `financial_extractor.extract()` again on up to 5 documents, duplicating
  Phase 1 work. It now reads financial facts from the war room briefing if available,
  falling back to re-extraction only for backward compatibility.
- **Budget notification `pct_complete` bug** — `_send_ci_budget_notification()` was
  computing `int(round(pct_complete * 100))` but `pct_complete` is already 0–100, producing
  email subjects like "1000% complete". Fixed to `int(round(pct_complete))`.
- **Duplicate `const pct` JS error** — `ciUpdateStatusBar()` declared `const pct` twice
  in the same function scope, throwing a `SyntaxError` that silently prevented the entire
  CI script block from executing (no jurisdiction auto-load, findings tab unresponsive).
- **`sqlite3.Row.get()` errors in Phase 2** — `_build_case_context()`,
  `_manager_contradictions()`, and `_paperless_writeback()` all called `.get()` on raw
  `sqlite3.Row` objects returned by `get_ci_entities()`, `get_ci_timeline()`,
  `get_ci_contradictions()`, and `get_ci_theories()`. Now converted to plain dicts at each
  call site. This was causing Phase 2 to fail entirely after Phase 1 completed.
- **Theory output truncated at token limit** — with 15 theories × rich supporting evidence,
  JSON output exceeded the 8,192-token Anthropic output limit, producing an unterminated
  JSON response and zero theories saved. Reduced to 8 theories with a conciseness instruction.

### Changed
- `contradiction_engine._build_docs_context()`: `max_per_doc` increased 1500 → 3000 chars.
- `theory_planner.generate_theories()`: truncation limits increased
  (entities/timeline 2000 → 3500, financial/contradictions 1500 → 2500,
  authorities 1500 → 2000). Maximum theories per run set to 8 (up from 10, but
  constrained by output token budget).
- `entity_extractor.py` content truncation: 6,000 → 15,000 chars.
- `timeline_builder.py` content truncation: 6,000 → 15,000 chars.
- `financial_extractor.py` content truncation: 7,000 → 20,000 chars.
- `estimate_run_cost()`: `theory_generation` and `opposing_theory_generation` now counted
  as fixed-cost (per-run) tasks rather than per-doc, matching actual billing behavior.

---

## v3.1.0 — 2026-02-23

### Added
- **CI run sharing** — run owners can share Case Intelligence runs with specific users via
  the 👥 Share button in CI Findings. Shared runs appear in the recipient's run dropdown
  annotated with the owner's name. Admins always see all runs.
- **Goal Assistant** — "✨ Refine with AI" chat in CI Setup asks clarifying questions and
  produces a structured goal statement; apply it with one click.
- **My Profile modal** — header button (previously "Change Password") is now "👤 My Profile".
  Opens a full profile form where users can update name, email, phone, address, and job title,
  plus change their password in the same modal.
- **Advanced user role (UI)** — Add User and Edit User forms now include the Advanced role
  option. Role badges display in purple throughout the interface.

### Changed
- CI Findings: run dropdown annotates runs by owner (admin view) and shows "(shared by X)"
  for runs shared with the current user.
- AI Chat: key detection now correctly reads the v2 `global` config format — the spurious
  "No AI API key configured" warning no longer appears when global keys are set.
- Help panels updated for CI Setup, CI Findings, Configuration → Users, and CI tab overview.
- Built-in manual (/docs/): Configuration page updated for v2 per-project AI config format;
  User Management page updated with Advanced role and My Profile; new Case Intelligence page.

---

## v3.0.1 — 2026-02-23

### Fixed
- **Search & Analysis tab** — "Failed to load documents" error caused by `anomalies` field
  being returned as a raw comma-separated string instead of a list. The JS `.join()` call
  threw a TypeError on any document with anomalies, aborting the entire load.

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
