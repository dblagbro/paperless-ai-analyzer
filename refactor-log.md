# Refactor Log — Paperless AI Analyzer

---

## Entry 006 — 2026-04-23 (v3.9.2 — CI orchestrator split via mixin pattern)

### Scope
Completed the refactor deferred from Entry 005: splitting the 2,371-line
single-class `case_intelligence/orchestrator.py` into a coordinator + 7 mixin
files under `ci_phases/`. Zero behavior change — same `self` instance, same
state, pure physical file split via Python multiple inheritance.

### Approach chosen: mixin pattern (not module-function extraction)

Considered two approaches:
1. **Module-level function extraction** — phases become free functions that
   take an `orchestrator` argument. Requires passing the orchestrator through
   every phase call, plus all its state attributes. High churn, high risk.
2. **Mixin class inheritance** — phases extracted as classes whose methods
   use `self.*` exactly like before. `CIOrchestrator` inherits from all mixins;
   methods resolve via MRO; no signature or state changes. Low churn, low risk.

Chose #2 because the 30 methods share heavy state (`self.llm_clients`,
`self.budget_manager`, `self.usage_tracker`, `self._token_lock`, 13 extractor
instances). Approach #2 preserves exact runtime semantics.

### Files changed

| File | Lines before | Lines after | Delta |
|------|--------------|-------------|-------|
| `case_intelligence/orchestrator.py` | 2,371 | 334 | –2,037 |
| `case_intelligence/ci_phases/directors_mixin.py` | (new) | 208 | +208 |
| `case_intelligence/ci_phases/managers_mixin.py` | (new) | 1,004 | +1,004 |
| `case_intelligence/ci_phases/specialist_mixin.py` | (new) | 257 | +257 |
| `case_intelligence/ci_phases/tier4_mixin.py` | (new) | 104 | +104 |
| `case_intelligence/ci_phases/tier5_mixin.py` | (new) | 263 | +263 |
| `case_intelligence/ci_phases/writeback_mixin.py` | (new) | 259 | +259 |
| `case_intelligence/ci_phases/utils_mixin.py` | (new) | 312 | +312 |
| `case_intelligence/ci_phases/__init__.py` | (new) | 0 | +0 |
| `architecture.md` | updated | — | — |
| `refactor-log.md` | this entry | — | — |

**Net code change:** +370 lines of import/header boilerplate across 7 new
files, but each file is now focused on one concern. Reading a 200–1,000-line
file to find one phase is significantly easier than a 2,371-line monolith.

### Method distribution

| Mixin | Methods | Lines | Concern |
|-------|---------|-------|---------|
| `DirectorPhasesMixin` | 3 | 159 | Director D1 plan, Q questions, D2 synthesis |
| `ManagerPhasesMixin` | 8 | 950 | Parallel domain managers, workers, specialist managers |
| `SpecialistPhasesMixin` | 2 | 209 | Tier 3+ forensic/discovery/witness coordination |
| `Tier4PhaseMixin` | 1 | 57 | Senior Partner review |
| `Tier5PhaseMixin` | 1 | 216 | White Glove (deep forensics + trial + multi-model) |
| `WritebackPhasesMixin` | 3 | 210 | Paperless write-back, CI findings embedding |
| `OrchestratorUtilsMixin` | 10 | 256 | Budget checkpoints, status, doc fetching, cancellation |

Kept on `CIOrchestrator` itself: `__init__` (58 lines) + `execute_run` (132
lines). These form the public surface that all callers (CIJobManager,
route handlers) interact with.

### Why this helps

- **Navigability**: reading the 2,371-line file required constant Ctrl-F to
  jump between phase methods. New files are focused on one concern.
- **Review friction**: PR reviews for a bug in Tier 5 no longer scroll past
  1,600 lines of unrelated Director/Manager/Writeback code.
- **Parallel work**: two developers can safely edit `tier5_mixin.py` and
  `managers_mixin.py` without merge conflicts.
- **Testability**: mixins can be instantiated with minimal stubs for
  unit-testing specific phases without loading the full orchestrator.

### Verification

- Import-sanity on dev container: all 28 inherited methods resolve via
  `CIOrchestrator` instance
- CI CRUD smoke-test on dev + jacob: runs list, status, jurisdictions,
  create/read/delete all return 200
- Medium-test `/tmp/instance_medium.py` passes on dev + jacob with identical
  failure pattern (known cosmetic health-probe bug + hardcoded test version)
- Zero new JS console errors on CI tab in either instance

### Deployment path

- Dev (8052): ✅ v3.9.2 via volume-mounted source + restart
- Jacob (8053): ✅ v3.9.2 via volume-mounted source + restart
- Prod (8051): bundled with next Docker Hub image push (pending user approval)

### Next recommended refactor targets

See `architecture.md`. Remaining candidates:
1. `analyzer/main.py` (1,598 lines) → `poller.py` + `document_processor.py`
2. `routes/ci.py` (1,793 lines) → split by CI phase groups
3. `routes/chat.py` (1,068 lines) → group handlers by concern

---

## Entry 005 — 2026-04-23 (v3.9.1 — maintainability refactor: split oversized route files)

### Scope
Incremental architectural refactor focused on maintainability and future
development speed. Target: the two largest route files which had mixed HTTP
and business-logic concerns. No behavior change; pure mechanical extraction.

### What was improved

**Refactor 1: `routes/chat.py` (1,443 → 1,068 lines, –26%)**
Extracted 7 business-logic helper functions from the top of `routes/chat.py`
into three focused service modules. The route module now contains only HTTP
handlers; all web-research, vision-OCR, and branch-tree logic lives in
`services/`.

- **`services/web_research_service.py`** (262 lines, NEW) —
  `ddg_search`, `fetch_url_text`, `resolve_court_docket_url`,
  `load_session_web_context`, `save_session_web_context`,
  `SEARCH_INTENT_PHRASES` constant
- **`services/vision_service.py`** (102 lines, NEW) —
  `vision_extract_doc`
- **`services/chat_branch_service.py`** (82 lines, NEW) —
  `compute_branch_data`

**Refactor 2: `routes/projects.py` (1,420 → 947 lines, –33%)**
Extracted the 477-line docker-compose + nginx + Postgres provisioning pipeline
and the document-migration pipeline into a single service module. The route
module now contains only HTTP handlers.

- **`services/project_provisioning_service.py`** (505 lines, NEW) —
  `_provision_status`, `_migration_status` state dicts;
  `_get_docker_client`, `_provision_log`, `_migration_log`;
  `_provision_project_paperless`, `_migrate_project_to_own_paperless`

### Files changed

| File | Lines before | Lines after | Delta |
|------|--------------|-------------|-------|
| `routes/chat.py` | 1,443 | 1,068 | –375 |
| `routes/projects.py` | 1,420 | 947 | –473 |
| `services/web_research_service.py` | (new) | 262 | +262 |
| `services/vision_service.py` | (new) | 102 | +102 |
| `services/chat_branch_service.py` | (new) | 82 | +82 |
| `services/project_provisioning_service.py` | (new) | 505 | +505 |
| `architecture.md` | updated | — | — |
| `refactor-log.md` | this entry | — | — |

### Why this helps

- **Testability**: services are plain Python functions with typed arguments
  and typed returns. They can be unit-tested without booting Flask, without
  a request context, without a DB connection (for the pure functions). The
  original helpers required spinning up the full Flask app to exercise.
- **Reusability**: `ddg_search` and `fetch_url_text` can be called from
  Case Intelligence's `web_researcher` (currently duplicates its own search
  code). Future refactor can de-duplicate.
- **Review friction**: reading the 1,400-line route files required mentally
  skipping past 400 lines of helpers to find the HTTP handler being discussed
  in a bug report. The new layout is "one file per concern".
- **Discoverability**: a new contributor asking "where does the AI chat
  decide to fetch a URL?" can search `services/` for `fetch_url_text` and
  find it in a 260-line file instead of a 1,400-line file.

### Verification

- `rsync` + container restart on dev
- Full `/tmp/instance_medium.py` medium-test run — **35/43 passed on all 3 instances** (identical to pre-refactor baseline)
- End-to-end chat + docs-ask smoke tests pass:
  - dev: `'refactor1-ok'`, `'refactor2-ok'`
  - jacob: `'jacob-refactor-ok'`
- Projects API `/provision-status` + `/migration-status` return expected `{"status":"idle"}`

### Why the CI orchestrator split was NOT done here

`case_intelligence/orchestrator.py` (2,371 lines, single `CIOrchestrator`
class with 30 methods spanning 6 phases) was identified as refactor candidate
#3 but deferred. Reasoning:

- It's the highest-risk code in the codebase (real billing, parallel workers,
  long-running) — a mistake here could silently corrupt in-flight runs
- The refactor is significantly more surgical (phase extraction with proper
  method signature changes) vs the mechanical move-and-import pattern used in
  #1 and #2
- It deserves its own PR with its own CI-focused regression pass
- Tracking in `project_paperless_backlog.md` so it's not lost

### Next recommended refactor targets

See `architecture.md` "Next Recommended Refactor Targets" section. Top three:

1. **`case_intelligence/orchestrator.py`** — split into `ci_phases/` submodules (deferred from this pass)
2. **`analyzer/main.py`** (1,598 lines) — split `DocumentAnalyzer` class into `poller.py` + `document_processor.py`
3. **`routes/ci.py`** (1,793 lines, 40 functions) — split by CI phase (setup / runs / findings / reports)

---

## Entry 004 — 2026-04-23 (v3.9.0 — LLM Proxy pool + LMRH)

### Scope
Introduce a multi-provider fallback layer between paperless-ai-analyzer and the
LLM providers. Replaces all 29 direct Anthropic/OpenAI SDK call sites with a
single unified chokepoint that routes through llm-proxy/llm-proxy2 endpoints
with circuit breakers and LMRH headers.

### New files
- `analyzer/llm/proxy_manager.py` (~130 lines) — circuit breaker + v1/v2 client builder
- `analyzer/llm/lmrh.py` (~90 lines) — LMRH header builder with 17 task presets
- `analyzer/llm/proxy_call.py` (~280 lines) — `call_llm()` + `call_llm_json()` helpers
- `analyzer/routes/llm_proxy.py` — admin CRUD + test endpoint blueprint
- `analyzer/static/js/llm_proxy.js` — admin sub-tab UI logic

### Modified
- `analyzer/db.py` — new `llm_proxy_endpoints` table + seed + 5 CRUD helpers
- `analyzer/web_ui.py` — register llm_proxy blueprint
- `analyzer/llm/llm_client.py` — `_call_llm` body reduced from 180 lines to 17
- `analyzer/routes/{chat,docs,forms,ci}.py` — provider branches → single `call_llm` call
- 13 `analyzer/case_intelligence/*.py` — `_call_llm` bodies delegated to `call_llm_json`
- `analyzer/templates/dashboard.html` — new Config sub-tab button + panel + JS include
- `analyzer/static/js/config.js` — lazy-load hook for new sub-tab
- `docker-compose.yml` — `LLM_PROXY_KEY` + `LLM_PROXY_URL` + `LLM_PROXY2_URL` env vars
  on all three paperless services

### Pattern source
Ported 1:1 from `/home/dblagbro/docker/devingpt/services/proxy_manager.py` and
`blueprints/admin_config.py`. Devingpt has been running this pattern for months.

### LMRH task map
| Task type | `model-pref` | Extra |
|-----------|-------------|-------|
| `chat` | `claude-sonnet-4-6` | `modality=vision` if images |
| `qa` | — | — |
| `analysis` | `claude-sonnet-4-6` | `fallback-chain=anthropic,openai` |
| `extraction` | `gpt-4o` | |
| `classification` | `gpt-4o-mini` | |
| `reasoning`, `theory`, `warroom`, `report`, `settlement` | `claude-opus-4-7` | `quality=high` |
| `entity`, `timeline`, `financial`, `contradiction` | `claude-sonnet-4-6` | |
| `forensic`, `discovery`, `witness` | `claude-sonnet-4-6` | `quality=high` |

### Testing
- **Phase 1 verified on dev**: DB table created + seeded; 2 endpoints present (v1 enabled, v2 disabled)
- **Phase 4 verified on dev**: `call_llm()` returns `pong` via llm-proxy-manager (gemini-2.5-flash)
- **Phase 5b verified on dev**: chat, docs-ask, forms all return 200 through the proxy
- **Phase 5c verified on dev**: all 14 CI submodules import cleanly
- **Phase 6 verified on dev**: admin sub-tab renders 2 rows, Test button returns `✓ gemini-2.5-flash`
- Full regression (`/tmp/full_regression_v2.py`) and medium check (`/tmp/instance_medium.py`) pending in Phase 8

### Fallback hierarchy (defense in depth)
1. Healthy llm-proxy endpoints in priority order (v1 enabled → v2 if enabled)
2. Direct Anthropic/OpenAI SDK using keys from `ai_config.json` (existing behavior)
3. Raise `LLMUnavailableError` → caller returns HTTP 503 with `source: 'llm-pool-exhausted'`

### Risks + mitigations
- Usage tracking: all proxy calls now reported as `provider='llm-proxy'` in
  `LLMUsageTracker`. Cost math unchanged (proxy returns same token counts).
- Case Intelligence parallel synthesis: `multi_model_synthesis` preserves its
  ThreadPoolExecutor but each branch uses `fallback_chain=<provider>` to pin
  the proxy to a single upstream per branch.

---

## Entry 002 — 2026-04-22 (v3.8.1 bug-fix pass)

### Scope
Post-refactor Playwright regression testing and bug fixes. No structural changes.

### Changes
- `analyzer/routes/documents.py` — reconcile performance + crash fix; trigger content-type hardening
- `analyzer/routes/system.py` — bug-report dual JSON/form accept
- `analyzer/routes/ci.py` — new `/interrupt` endpoint; `auto_start` + `start_url` in create response
- `analyzer/templates/dashboard.html` — admin Users shortcut tab button
- `analyzer/static/js/init.js` — `goToUsersAdmin()` helper for Users tab
- `contributing.md` — API conventions documented

### Test coverage
Full 99-test Playwright regression suite at `/tmp/full_regression.py`: **99/99 pass**.

---

## Entry 003 — 2026-04-22 (v3.8.1 expanded regression pass)

### Scope
Expanded regression test suite covering all 15 new features added since the original 296-step plan.
No code changes in this entry — documentation and test artifact only.

### Changes
- `/tmp/full_regression_v2.py` — 712-test Playwright suite (36 phases) replacing the 99-test v1 suite:
  Chat CRUD/branching/sharing/export/compare, Upload file/URL/cloud/directory-scan, Court
  credentials/search/import lifecycle, Case Intelligence run CRUD/lifecycle/findings/reports/
  sharing/authority corpus, stale RAG re-embedding, multi-user RBAC cross-role flows,
  cross-feature end-to-end workflows, and error-handling edge cases (malformed JSON, unicode,
  SQL injection patterns, large payloads, zero-byte upload)
- `README.md` — "What's New" updated from v3.7.3 → v3.8.1
- `CHANGELOG.md` — Testing section added to v3.8.1 entry

### Test results
**712 total | 674 passed | 20 failed | 18 skipped** (94.7% pass rate)

Open issues queued for v3.8.2 (selected):
- `POST` with malformed JSON body → 500 (`'str' object has no attribute 'get'`) — 3 endpoints
- `POST /api/upload/from-url` with reachable URL → 500
- `POST /api/ci/runs` with `auto_start=true` → 500
- `GET /api/court/docket/pacer/<case_id>` → 500 (unknown court system)
- `POST /api/vector/delete-document` → wrong status code
- Zero-byte file upload → 500
- Chat session share API sends `uid` but endpoint expects `username`
- `POST /api/projects` intermittent timeout under load (affects 3 derived test failures)

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
