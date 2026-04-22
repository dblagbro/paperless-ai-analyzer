# Refactor Log — Paperless AI Analyzer

---

## Entry 001 — 2026-04-22

### Scope
Full architectural refactor of the v3.7.4 codebase.
Classification: **Heavy refactor** — major restructuring of two primary files and one data layer.

### Motivation
- `web_ui.py` had grown to 9,812 lines containing ~30 distinct domain areas with no separation
  of concerns. A single AI context load of this file consumed ~40% of a typical context window,
  making feature work on any single domain expensive and error-prone.
- `dashboard.html` was 12,621 lines: 12,300 lines of JavaScript and 1,554 lines of CSS embedded
  inline in a Jinja2 template. The `static/js/` and `static/css/` directories existed but were
  completely empty.
- `case_intelligence/db.py` was a 1,482-line flat file containing ~15 distinct data domains
  (runs, shares, questions, entities, timeline, contradictions, theories, authorities, reports).
- No architecture documentation existed. Every AI session re-derived project structure from scratch.

### Key Changes

#### Phase 1 — Documentation
Created:
- `architecture.md` — full system architecture, boot sequence, data stores, design decisions
- `design.md` — frontend architecture, API conventions, CSS patterns, error handling
- `contributing.md` — dev workflow, file size limits, rules for adding routes/components
- `refactor-log.md` — this file

#### Phase 2 — web_ui.py Blueprint Decomposition

**New files created:**
- `analyzer/app.py` — Flask app instance, middleware (ReverseProxied, LogBufferHandler),
  auth decorators (admin_required, advanced_required, _ci_gate), before/after request hooks,
  global ui_state, create_app() factory, _get_project_client() cache,
  update_ui_stats(), run_web_server(), start_web_server_thread()
- `analyzer/services/__init__.py`
- `analyzer/services/ai_config_service.py` — load_ai_config, save_ai_config,
  get_project_ai_config, _migrate_old_ai_config, get_default_ai_config
- `analyzer/services/smtp_service.py` — _load/_save_smtp_settings, _smtp_send,
  _send_welcome_email, _send_manual_email, _send_ci_budget_notification,
  _send_ci_complete_notification
- `analyzer/routes/__init__.py`
- `analyzer/routes/auth.py` — /login, /logout
- `analyzer/routes/status.py` — /api/status, /api/recent, /health, /api/about
- `analyzer/routes/profiles.py` — /api/profiles, /api/staging/*, /api/active/*
- `analyzer/routes/chat.py` — /api/chat, /api/chat/compare, /api/chat/sessions/*
- `analyzer/routes/vector.py` — /api/vector/*, /api/scan/process-unanalyzed
- `analyzer/routes/documents.py` — /api/reprocess, /api/reconcile, /api/trigger,
  /api/logs, /api/search, /api/tag-evidence
- `analyzer/routes/projects.py` — /api/projects/* full suite including provisioning,
  migration, health checks, orphan documents, document listing
- `analyzer/routes/upload.py` — /api/upload/*
- `analyzer/routes/ai_config.py` — /api/ai-config/*, /api/llm/*, /api/llm-usage/*
- `analyzer/routes/users.py` — /api/users, /api/me, /api/change-password
- `analyzer/routes/system.py` — /api/containers, /api/smtp-settings, /api/bug-report,
  /api/system-health, Docker health checks
- `analyzer/routes/ci.py` — /api/ci/* (kept as one Blueprint, ~1,600 lines, single domain)
- `analyzer/routes/court.py` — /api/court/*, /api/projects/<slug>/analyze-missing
- `analyzer/routes/forms.py` — /api/ai-form/parse
- `analyzer/routes/docs.py` — /docs/*, /api/docs/ask

**Modified:**
- `analyzer/web_ui.py` — reduced from 9,812 lines to ~40-line thin orchestrator
  (imports app, registers all blueprints, re-exports entry points for main.py compat)
- `analyzer/main.py` — import updated from `web_ui` to `app` for `start_web_server_thread`
  and `update_ui_stats`

**Pattern change:**
All `app.X` attribute accesses in route handlers converted to `current_app.X`
(118 occurrences total across all route modules).

#### Phase 3 — dashboard.html JS/CSS Extraction

**New files created:**
- `analyzer/static/css/dashboard.css` — 1,554 lines extracted CSS
- `analyzer/static/js/utils.js` — shared utilities (apiFetch, apiUrl, showToast, escapeHtml)
- `analyzer/static/js/overview.js` — Overview tab + stats bar
- `analyzer/static/js/config.js` — Config tab: vector store, AI config, LLM settings, profiles
- `analyzer/static/js/chat.js` — Chat tab: sessions, messages, branching, compare
- `analyzer/static/js/upload.js` — Upload tab: file/URL/cloud/court import
- `analyzer/static/js/ci.js` — Case Intelligence tab
- `analyzer/static/js/users.js` — Users admin panel
- `analyzer/static/js/ai_form_filler.js` — AIFormFiller widget class
- `analyzer/static/js/init.js` — DOMContentLoaded bootstrap, tab switching

**Modified:**
- `analyzer/templates/dashboard.html` — reduced from 12,621 to ~2,900 lines (HTML only)
  Added `window.APP_CONFIG` injection block for Flask→JS variable bridging.
  All `<script>` and `<style>` blocks replaced with external file references.

#### Phase 4 — case_intelligence/db.py Split

**New files created:**
- `analyzer/case_intelligence/db/__init__.py` — re-exports all public symbols
- `analyzer/case_intelligence/db/schema.py` — init_ci_db(), recover_orphaned_runs()
- `analyzer/case_intelligence/db/runs.py` — run lifecycle, shares, questions
- `analyzer/case_intelligence/db/analysis.py` — entities, timeline, contradictions,
  disputed facts, theories
- `analyzer/case_intelligence/db/authorities.py` — authority corpus management
- `analyzer/case_intelligence/db/reports.py` — report CRUD

**Deleted:**
- `analyzer/case_intelligence/db.py` — replaced by db/ package

### Architectural Decisions

1. **Flask Blueprints over shared `app` import** — Chosen for proper Flask architecture and
   `current_app` proxy correctness. Each Blueprint is independently importable.

2. **`ci.py` kept as one Blueprint** — The CI domain (~1,600 lines) is cohesive: all
   `/api/ci/*`. Splitting by tier would create tightly-coupled micro-files. Acceptable at
   current size.

3. **`window.APP_CONFIG` pattern for JS variable injection** — All Flask template variables
   needed by JS are surfaced through one small inline block. External `.js` files remain
   pure JavaScript with no Jinja2 syntax, enabling linting, syntax highlighting, and future
   bundling.

4. **Services layer for cross-cutting logic** — `ai_config_service.py` and `smtp_service.py`
   are the only true shared services. All other helpers are co-located with their route
   module (they are private to that domain).

5. **`__init__.py` re-export pattern for db/ split** — All `from analyzer.case_intelligence.db
   import X` statements across the codebase continue to work unchanged. No call-site changes
   required for the db split.

### Files Impacted
- Deleted/replaced: `analyzer/web_ui.py` (9,812 lines), `analyzer/case_intelligence/db.py` (1,482 lines)
- Modified: `analyzer/main.py`, `analyzer/templates/dashboard.html`
- Created: 28 new files across `routes/`, `services/`, `case_intelligence/db/`, `static/js/`, `static/css/`

### Risks
- Blueprint `current_app.X` conversion: all 118 occurrences must be correct — verified by
  grepping for `app.state_manager` etc. after completion
- JS extraction: any missed `jinja2` expression in extracted JS would be a runtime error —
  verified by grepping for `{{` in all `.js` files after extraction

### Remaining Issues / Technical Debt
- `case_intelligence/orchestrator.py` (2,386 lines) is large but has a single clear responsibility.
  If the CI pipeline grows significantly, consider splitting by phase group (phases 1–2 vs 3A vs 3B).
- `main.py` `DocumentAnalyzer` class (1,500 lines) mixes poll loop and per-document processing.
  Decompose in a future pass: `poller.py` (loop management) + `document_processor.py` (per-doc logic).
- No automated test suite. Integration tests for key API routes would significantly reduce
  regression risk during future refactors.

### Next Recommended Actions
1. Test all tabs in the dev UI end-to-end after Phase 3 (JS extraction is the highest visual risk)
2. Promote to jacob for QA once dev is stable
3. Consider adding pytest integration tests for the Blueprint route handlers
4. Consider splitting `orchestrator.py` if Tier 6+ is added
