# Refactor Log — Paperless AI Analyzer

---

## Entry 012 — 2026-04-23 (v3.9.9 — docs.html pages + routes/court package)

### Scope
Last two maintainability splits worth doing. Every file in the codebase is
now well under 1,000 lines except for cohesive large files that would not
benefit from splitting.

### Changes

**`templates/docs.html` (1,820 lines) → 591-line shell + 14 page partials:**
`docs.html` kept the sidebar, layout, AI help widget, and final scripts; each
`{% elif page == 'X' %}` branch now reads `{% include 'docs_pages/X.html' %}`.
Per-page files:

| Page | Lines |
|------|------:|
| overview | 68 |
| getting_started | 55 |
| projects | 175 |
| upload | 84 |
| chat | 110 |
| search | 56 |
| anomaly_detection | 63 |
| tools | 89 |
| configuration | 108 |
| users | 73 |
| llm_usage | 47 |
| api | 101 |
| case_intelligence | 120 |
| court_import | 77 |

**`routes/court.py` (847 lines) → `routes/court/` package (5 files):**

| File | Lines | Concern |
|------|-------|---------|
| `__init__.py`    |  22 | Blueprint + side-effect imports |
| `helpers.py`     | 400 | `_court_gate`, `_build_court_connector`, `_post_import_analyze`, `_analyze_missing_for_project`, `_run_court_import` (import job worker) |
| `credentials.py` | 243 | save / list / test / delete / paste-parse |
| `search.py`      |  97 | search + docket fetch |
| `imports.py`     | 145 | start / status / cancel / history + analyze-missing |

Pattern matches `routes/chat/`, `routes/ci/`, `routes/projects/`: `__init__.py`
defines the blueprint, submodules import `bp` and register routes via
side-effect decorators.

### Verification
- All 14 `/docs/<slug>` URLs return 200 with expected content.
- `/api/court/credentials` GET 200, `/api/court/import/history` GET 200,
  `/api/court/search` POST (empty body) 400 (expected validation).
- Playwright UI smoke: login + all 8 tabs + 4 config sub-tabs + every doc
  page; no real console errors (one pre-existing 404 on `/x` is unrelated
  to the refactor).

### State after 4 consecutive refactor passes (v3.9.7 → v3.9.9)
| File | Before | After |
|------|-------:|------:|
| `templates/dashboard.html` | 2,994 | ~1,100 |
| `templates/docs.html` | 1,820 | 591 |
| `static/js/config.js` | 2,361 | 4-file package |
| `static/js/ci.js` | 2,229 | 5-file package |
| `routes/projects.py` | 988 | 5-file package |
| `routes/court.py` | 847 | 5-file package |
| `case_intelligence/web_researcher.py` | 1,362 | 6-line shim + 7-file package |

Largest single file now: `case_intelligence/ci_phases/managers_mixin.py` at
1,004 lines (cohesive — one concern, revisit > 1,500).

### Files touched
- `analyzer/__init__.py` — version → 3.9.9
- `analyzer/routes/court.py` — **deleted** (replaced by package)
- `analyzer/routes/court/` — new package (5 files)
- `analyzer/templates/docs.html` — 1,820 → 591 lines
- `analyzer/templates/docs_pages/` — new directory (14 files)
- `architecture.md` — tree + next-targets updated
- `refactor-log.md` — this entry
- `CHANGELOG.md` — v3.9.9 entry

---

## Entry 011 — 2026-04-23 (v3.9.8 — config.js, ci.js, web_researcher splits)

### Scope
Completes the maintainability pass kicked off in Entry 010. Three of the last
four large-file refactor targets shipped; the fourth (`managers_mixin.py`) is
still cohesive and deferred.

### Changes

**`static/js/config.js` (2,361 lines) → `static/js/config/` package (4 files):**

| File | Lines | Concern |
|------|-------|---------|
| `core.js`        |  465 | Sub-tab switcher, Tools panel, AI usage + chart, vector store, SMTP |
| `projects.js`    |  905 | Projects CRUD, new/edit modal, Paperless modal, auto-provisioning, delete/move modals |
| `search.js`      |  285 | Search & Analysis tab |
| `profiles_ai.js` |  705 | LLM settings, profile CRUD, AI config management |

**`static/js/ci.js` (2,229 lines) → `static/js/ci/` package (5 files):**

| File | Lines | Concern |
|------|-------|---------|
| `setup.js`        | 755 | 5-tier selector, tier config, findings subtab, elapsed timer helpers |
| `goal_assist.js`  | 650 | Goal Assistant, run metadata header, web research panel |
| `specialists.js`  | 365 | Forensic, Discovery, Witnesses, War Room panels |
| `tier5.js`        | 385 | Deep Forensics, Trial Strategy, Multi-Model Compare, Settlement |
| `report.js`       |  70 | Final report builder + tab-open hook |

Both JS splits rely on classic-script globals: `let`/`const` at top level is
shared across sequentially-loaded classic scripts (not `type="module"`), so
cross-file state (e.g. `vsAllDocs`, tier selector state) continues to work
without any `window.` globals or module exports. Load order is preserved in
`dashboard.html`: `core.js` first (it defines shared state), then siblings.

**`case_intelligence/web_researcher.py` (1,362 lines) → `case_intelligence/web_researchers/` mixin package (7 files):**

| File | Lines | Concern |
|------|-------|---------|
| `__init__.py`             | 210 | `WebResearcher` class composes 4 mixins + 3 public methods (`search_legal_authorities`, `research_entity`, `search_general`) |
| `constants.py`            |  75 | `_RATE`, `_STATE_TO_CL`, `_ROLE_AUTHORITY_PREFIX`, `_ROLE_ENTITY_*` |
| `http_utils.py`           |  60 | `_http_get`, `_http_post_json` — requests → urllib fallback |
| `base.py`                 |  65 | `WebResearcherBaseMixin`: `__init__`, `_throttle`, `_jur_to_cl`, `_jur_to_caselaw`, `_dedup` |
| `providers_legal.py`      | 365 | `LegalProvidersMixin`: CourtListener, Harvard Caselaw, Lexis-Nexis, vLex, Westlaw, Docket Alarm, UniCourt |
| `providers_general.py`    | 330 | `GeneralSearchProvidersMixin`: DDG, GDELT, Brave, Google CSE, Exa, Perplexity, NewsAPI, Tavily, Serper |
| `providers_entities.py`   | 285 | `EntityResearchProvidersMixin`: BOP, OFAC, SEC EDGAR, FEC, OpenSanctions, OpenCorporates, CLEAR |

`web_researcher.py` kept as a 2-line re-export shim so existing callers
(`from analyzer.case_intelligence.web_researcher import WebResearcher`) keep
working unchanged. MRO is `WebResearcher → LegalProvidersMixin →
GeneralSearchProvidersMixin → EntityResearchProvidersMixin →
WebResearcherBaseMixin → object`.

### Why the mixin pattern
`WebResearcher` is one logical class — `search_legal_authorities` calls into
provider methods, provider methods call into `_throttle` and `_http_get`.
Splitting into separate non-mixin classes would have required either pushing
all state to a context object or duplicating orchestration. Mixins preserve
the single-class API while letting each provider group live in its own file.
This mirrors the `ci_phases/` pattern shipped in v3.9.2.

### Deferred
**`case_intelligence/ci_phases/managers_mixin.py` (1,004 lines)** — every
method implements the same architectural concept (the "Manager phase" of the
CI pipeline) and shares extensive `self.*` state. Splitting would not reduce
edit context meaningfully — whoever touches `_manager_theories` will often
need to cross-reference `_run_manager` and `_run_worker` anyway. Revisit only
if the file crosses ~1,500 lines.

### Verification
- Python import smoke: `WebResearcher({})` constructs; all provider methods
  discoverable via `hasattr`; MRO matches expected order.
- HTTP smoke (after restart): dashboard 200, `/api/status` 200, `/api/ci/runs`
  200, `/api/orphan-documents` 200, all 9 new JS files serve 200, old
  monolithic `config.js` and `ci.js` return 404.

### Files touched
- `analyzer/__init__.py` — version → 3.9.8
- `analyzer/case_intelligence/web_researcher.py` — **rewritten as 2-line shim**
- `analyzer/case_intelligence/web_researchers/` — new package (7 files)
- `analyzer/static/js/config.js` — **deleted** (replaced by package)
- `analyzer/static/js/config/` — new directory (4 files)
- `analyzer/static/js/ci.js` — **deleted** (replaced by package)
- `analyzer/static/js/ci/` — new directory (5 files)
- `analyzer/templates/dashboard.html` — script tags updated
- `architecture.md` — tree + next-targets updated
- `refactor-log.md` — this entry
- `CHANGELOG.md` — v3.9.8 entry

---

## Entry 010 — 2026-04-23 (v3.9.7 — projects route package + dashboard partials)

### Scope
Maintainability pass focused on the two worst file-size offenders outside
`case_intelligence/`. No behavior changes; pure reorganisation.

### Changes

**`routes/projects.py` (988 lines) → `routes/projects/` package (5 files, ~250 avg):**

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py`           |  22 | Blueprint + side-effect imports |
| `core.py`               | 260 | CRUD + archive/unarchive + current-project |
| `paperless_config.py`   | 100 | per-project Paperless config + test + doc-link |
| `provisioning.py`       | 170 | provision-snippets + status + reprovision (delegates to service) |
| `migration.py`          | 230 | migrate-to-own-paperless + migration-status + migrate-documents |
| `documents.py`          | 240 | list/delete project docs + orphan-documents + assign-project + reanalyze |

Pattern follows `routes/chat/` and `routes/ci/` — `__init__.py` creates the
blueprint; each submodule imports `bp` from the package and registers routes
by side-effect. External import surface is unchanged: `from analyzer.routes import projects`
still works; `projects.bp` is registered in `web_ui.py` as before.

Drive-by fix while in the file: `GET /api/orphan-documents` no longer 500s when
Paperless returns a non-dict inside the results list (pre-existing `'int'
object is not subscriptable` bug, logged for weeks). Malformed entries are now
skipped rather than aborting the whole response.

**`templates/dashboard.html` (2,994 lines) → shell + 3 partials:**

| File | Lines | Notes |
|------|-------|-------|
| `dashboard.html`                                 | 1,089 | Shell: head, tab bar, 4 small panels, project modals, trailing scripts |
| `partials/tab_config.html`                       |   625 | Whole Config tab: AI settings, profiles, vector store, SMTP, LLM proxy, Users |
| `partials/tab_upload.html`                       |   395 | Smart Upload panel + Court Import wizard |
| `partials/tab_case_intelligence.html`            |   888 | Entire CI tab (Setup / Findings / Specialists / Tier 5) |

Jinja `{% include %}` inherits the parent context by default, so `{% if
is_admin %}`, `{{ request.script_root }}`, and other template variables work
unchanged inside partials.

Skipped from this pass:
- `tab-overview` (61 lines), `tab-projects` (29 lines; modals live outside the
  panel), `tab-ai-chat` (91), `tab-search` (117), `tab-tools` (119) — all below
  the threshold for a dedicated partial.
- Project-related modals (248–497 in the old file) — kept inline for now; would
  be a second pass if the modal stack grows further.

### Verification
- Dashboard renders 200 with all 8 tab panels present (grep-verified after restart).
- 9/10 project endpoint smoke tests pass; the 10th (`/api/orphan-documents`)
  only failed due to a pre-existing malformed-doc serialization bug which was
  fixed as part of the move.

### Deferred to future passes (see architecture.md § Next Recommended Refactor Targets)
- `static/js/config.js` (2,361 lines) — split plan drafted but not executed;
  JS-specific risk (shared top-level `let` state across scripts) needs a small
  proof-of-concept before full execution.
- `static/js/ci.js` (2,229 lines) — same pattern as config.js once proven.
- `case_intelligence/web_researcher.py` (1,362 lines) — mixin split planned
  (legal / general / entities providers).

### Files touched (10)
- `analyzer/routes/projects.py` — **deleted** (replaced by package)
- `analyzer/routes/projects/__init__.py` — new
- `analyzer/routes/projects/core.py` — new
- `analyzer/routes/projects/paperless_config.py` — new
- `analyzer/routes/projects/provisioning.py` — new
- `analyzer/routes/projects/migration.py` — new
- `analyzer/routes/projects/documents.py` — new (includes orphan-documents bugfix)
- `analyzer/templates/dashboard.html` — trimmed 2,994 → 1,089 lines
- `analyzer/templates/partials/tab_config.html` — new
- `analyzer/templates/partials/tab_upload.html` — new
- `analyzer/templates/partials/tab_case_intelligence.html` — new
- `architecture.md` — tree updated, "Next targets" rewritten
- `refactor-log.md` — this entry

---

## Entry 009 — 2026-04-23 (v3.9.6 — Paperless-slow hardening + provision throttling)

### Scope
Two related fixes in one pass. (1) Every endpoint that called the Paperless
client on the request thread could block Waitress for 15–45s when Paperless
was slow (not unreachable — slow enough that `health_check` passed but the
next call still hung on tenacity retries). (2) Rapid project creation spun
up new Paperless stacks so fast the host CPU/RAM saturated and starved
earlier stacks before they finished warming up.

### Root-cause patterns

**Pattern A — synchronous Paperless calls on the request thread.** 5
routes were affected: `/api/status`, `/api/reconcile`, `/api/orphan-documents`,
`POST /api/projects` (tag creation), `/api/scan/process-unanalyzed`. All
used tenacity-wrapped calls that retried 3× with exponential backoff, and
the shared `requests.Session` had no default timeout. A slow upstream
therefore parked a Waitress thread for the full retry budget. With only
4 Waitress threads, even fast admin requests queued up behind.

**Pattern B — unbounded concurrent provisioning.** The old
`POST /api/projects` fire-and-forgot a thread that called
`_provision_project_paperless` immediately. Tests (and impatient humans)
could fire 3+ creates in seconds, each starting its own Paperless stack
before the previous one finished warming up.

### Changes

**Pattern A fixes:**
- `PaperlessClient.__init__` installs a default 15s per-request timeout on
  the shared `requests.Session` — a monkey-patch on `session.request` that
  preserves per-call `timeout=` overrides.
- `PaperlessClient.health_check()` accepts a `timeout` (default 3s) and
  caches the result for 10s (`_HEALTH_CACHE_SECS`).
- `get_documents_without_project()` now scans `page_size=100`, single page
  (the calling route caps at 100 anyway). Previously `page_size=1000`,
  which timed out on instances with many docs.
- `POST /api/reconcile` paginates at `page_size=200` for the same reason.
- 5 routes now short-circuit with a 503 / `paperless_available: false`
  response when `health_check()` fails.
- `POST /api/projects` tag creation moved to a background daemon thread.
- Waitress `threads=4` → `threads=16` in `analyzer/app.py` — 4 was too few
  to survive even one slow upstream call.

**Pattern B fix — `services/project_provisioning_service.py`:**
- New module-level `PROVISION_MIN_INTERVAL_SECS = 180` (env-tunable).
- New FIFO queue + single worker thread. First request starts immediately;
  subsequent requests wait out the cooldown before their turn.
- New public `enqueue_provision(slug) -> dict` — replaces direct
  `Thread(target=_provision_project_paperless).start()` in routes.
- `_provision_status[slug]` gains `queue_position` and `eta_seconds`
  (refreshed every 5s while queued so the UI countdown stays live).
- `status='queued_waiting'` is a new state distinct from `queued`.

**Routes touched:**
- `POST /api/projects` — returns `provision: {status, queue_position,
  eta_seconds, throttle_interval_secs}` in 201 response.
- `POST /api/projects/<slug>/reprovision` — also uses the queue.

**UI touched (`static/js/config.js`):**
- Card banner renders `⏸️ Waiting in provisioning queue (#N)` for the
  new `queued_waiting` state.
- Toast on project create reports the wait time immediately if queued.

### Why not a full-blown job scheduler?
A queue + sleep + one worker is ~130 lines. A real scheduler would add
a dependency, a DB table, and operational surface for something that only
exists to gate one long-running op. If/when other throttled ops show up,
this generalizes — not before.

### Verification
- 3 back-to-back `POST /api/projects` returned 201 in < 50ms each with
  `queue_position` 1/2/3 and `eta_seconds` 0/179/359.
- `GET /api/projects/<slug>/provision-status` returns live ETAs.
- `GET /api/status` response time down from 10+s (when Paperless slow)
  to sub-5s (first call, uncached) and sub-0.1s (health-cached).
- `POST /api/orphan-documents`, `POST /api/reconcile`,
  `POST /api/scan/process-unanalyzed`, `POST /api/projects` all now
  respond in < 3s under a slow Paperless.

### Files touched (8)
- `analyzer/__init__.py` — version → 3.9.6
- `analyzer/app.py` — Waitress threads=16
- `analyzer/paperless_client.py` — default session timeout + health cache + orphan pagination
- `analyzer/routes/documents.py` — reconcile + process-unanalyzed health gating
- `analyzer/routes/projects.py` — enqueue_provision; async tag creation; reprovision queue
- `analyzer/routes/status.py` — health-gate Paperless calls
- `analyzer/services/project_provisioning_service.py` — queue + worker + enqueue API
- `analyzer/static/js/config.js` — queued_waiting banner + toast ETA

---

## Entry 008 — 2026-04-23 (v3.9.4 + v3.9.5 — regression cleanup)

### Scope
Not a structural refactor — this is a bug-fix pass that clears the 20 known
regression failures from v3.8.2 documented in
`project_paperless_regression_state.md`. Bundled together because most share
two root-cause patterns worth fixing systematically: (a) Flask's
`request.get_json()` returning non-dict types, and (b) `current_app` accessed
from background threads outside request context.

### Added

- **`analyzer.app.safe_json_body()`** — one-line helper that returns `{}`
  when the request body isn't a JSON object. Replaces 55 instances of
  `request.json or {}` / `request.get_json() or {}` across 18 route modules.

### Fixed (19 of 20 regression items)

| Test ID | Fix |
|---------|-----|
| 34.2 / 15.10 / 15.12 / 34.x | Malformed-JSON → 400 (was 500) via `safe_json_body` |
| 14.8 | `PATCH /api/users/<bad_uid>` → 404 (was 200) |
| 4.15 | `POST /api/reprocess/<bad_id>` → 404 (was 500 `AnalyzerState` attr error) |
| 23.14 | `GET /api/court/docket/pacer/<case>` → 400 with supported-systems hint (was 500) |
| 26.5 | `POST /api/ci/runs auto_start=true` → 201 (was 500 `current_app` import) |
| 21.4 | `POST /api/upload/from-url` upstream fail → 502 (was 500) |
| 4.18 | `POST /api/scan/process-unanalyzed` background thread captures deps |
| 3.6 | `/api/status` aliases `total_documents` / `analyzed_documents` / `analyzed_count` |
| 9.4 / 9.13 | `POST /api/vector/delete-document` string `doc_id` → 400 (was 500 int-cast) |
| 19.1 | `POST /api/chat/sessions/<id>/share` accepts `uid` or `username` (was only username) |
| System-health | Probes capture `current_app` deps on request thread (was always `error`) |
| Stale embeddings | `check_stale_embeddings` skips non-numeric CI composite IDs |
| 25.11 / 25.12 / 25.25 | Goal-assistant stale import `_fetch_url_text` from `routes.chat` (moved to services/ in v3.9.1) |
| 8.2 | `/api/active/duplicates` O(n²) similarity-scan gated to `?deep=1` for > 500 profiles |
| 8.11 | `POST /api/active/duplicates/remove` missing `safe_json_body` import |
| 34.9 | Zero-byte file upload → 400 "File is empty" (was 500 upstream) |

### Remaining (1 of 20 — test-hygiene only)

- **36.6 / 36.7 cleanup race** — regression tests create `pw-*` test artifacts
  and assert they don't remain at the end. Flagged `is_bug=False` in the
  suite. Minor race between DELETE and the next GET under high concurrency.
  Not a product defect.

### Patterns applied

**Pattern A — Background-thread app-context capture:**
```python
# Before (crashes outside request context):
def _run():
    current_app.document_analyzer.do_work()

# After:
da = current_app.document_analyzer
def _run(document_analyzer):
    document_analyzer.do_work()
Thread(target=_run, args=(da,)).start()
```
Applied to 3 call sites: system-health probes, vector/reembed-stale,
scan/process-unanalyzed.

**Pattern B — JSON body validation:**
```python
# Before (crashes on valid-but-non-object JSON):
data = request.json or {}
x = data.get('x')  # AttributeError: 'str' object has no attribute 'get'

# After (via shared helper):
data = safe_json_body()
x = data.get('x')  # always safe — helper returns {} for non-dicts
```
Applied to 55 call sites across 18 route files.

### Verification
- Full `/tmp/full_regression_v2.py` run: 19/20 of the targeted failures
  resolved. The remaining 1 (cleanup race) is tagged `is_bug=False`.
- Medium-test pass rate: 37/43 → 40/43 on dev.
- No new regressions introduced by the fixes.

---

## Entry 007 — 2026-04-23 (v3.9.3 — final trio: main.py, routes/ci.py, routes/chat.py)

### Scope
Completed the three remaining backlog refactors in a single pass:
1. **`analyzer/main.py`** (1,598 → 267 lines, –83%) — DocumentAnalyzer mixin split
2. **`routes/ci.py`** (1,793 → 5-file package) — CI handlers grouped by concern
3. **`routes/chat.py`** (1,068 → 4-file package) — chat handlers grouped by concern

### main.py split
- New `analyzer/poller.py` — `PollerMixin` + module-level `_poll_project_loop`
  (566 lines total). Owns poll loop, per-project pollers, re-analysis,
  stale-embedding check, OCR quality check.
- New `analyzer/document_processor.py` — `DocumentProcessorMixin` (852 lines).
  Owns vision AI extraction, the 538-line `analyze_document`, PDF path, tag
  compilation, AI note formatting, severity helpers.
- `main.py` retains CLI entry point + `DocumentAnalyzer(PollerMixin, DocumentProcessorMixin)` composition (267 lines total).

### routes/ci.py split (converted to `routes/ci/` package)
- `routes/ci/__init__.py` — blueprint aggregator; exports `bp = Blueprint('ci', ...)`
- `routes/ci/helpers.py` (254 lines) — `_send_ci_budget_notification`,
  `_send_ci_complete_notification`, `_match_jurisdiction_profile`,
  `_ci_elapsed_seconds`, `_build_ci_llm_clients`
- `routes/ci/setup.py` (536 lines) — status, jurisdictions, detect-jurisdiction,
  goal-assistant, key-guide, cost-estimate, authority ingest/status
- `routes/ci/runs.py` (590 lines) — runs list/create/get/update/delete,
  lifecycle (start/cancel/interrupt/rerun/status), shares, questions/answers
- `routes/ci/findings.py` (418 lines) — run findings + all tier-specific
  report views (forensic/discovery/witness/war-room/deep-forensics/
  trial-strategy/multi-model/settlement-valuation)
- `routes/ci/reports.py` (138 lines) — custom report CRUD + PDF download

### routes/chat.py split (converted to `routes/chat/` package)
- `routes/chat/__init__.py` — blueprint aggregator
- `routes/chat/core.py` (729 lines) — `/api/chat` + `/api/chat/compare`
- `routes/chat/sessions.py` (227 lines) — session CRUD + share/unshare + rename
- `routes/chat/branching.py` (165 lines) — message edit, branch, set-leaf
- `routes/chat/export.py` (93 lines) — PDF export

### Why two different patterns (mixin vs package)?

**Mixin pattern (main.py)**: DocumentAnalyzer is a single class with heavy
shared state (`self.llm_client`, `self.paperless`, `self.vector_store`, etc.).
Extracting methods into mixins preserves `self` resolution exactly as before
via Python's MRO. Same pattern used for CIOrchestrator in v3.9.2.

**Package pattern (routes/ci, routes/chat)**: Flask Blueprint handlers don't
share instance state — each is a free function decorated with `@bp.route`.
Convert the file to a package with `__init__.py` creating the shared `bp`;
submodules do `from analyzer.routes.{ci,chat} import bp` and attach handlers.
Zero external API change (`web_ui.py` still imports `ci.bp` and `chat.bp`).

### Files changed

| File | Lines before | Lines after | Delta |
|------|--------------|-------------|-------|
| `analyzer/main.py` | 1,598 | 267 | –1,331 |
| `analyzer/poller.py` | (new) | 566 | +566 |
| `analyzer/document_processor.py` | (new) | 852 | +852 |
| `analyzer/routes/ci.py` | 1,793 | → package | – |
| `analyzer/routes/ci/__init__.py` | (new) | 17 | +17 |
| `analyzer/routes/ci/helpers.py` | (new) | 254 | +254 |
| `analyzer/routes/ci/setup.py` | (new) | 536 | +536 |
| `analyzer/routes/ci/runs.py` | (new) | 590 | +590 |
| `analyzer/routes/ci/findings.py` | (new) | 418 | +418 |
| `analyzer/routes/ci/reports.py` | (new) | 138 | +138 |
| `analyzer/routes/chat.py` | 1,068 | → package | – |
| `analyzer/routes/chat/__init__.py` | (new) | 15 | +15 |
| `analyzer/routes/chat/core.py` | (new) | 729 | +729 |
| `analyzer/routes/chat/sessions.py` | (new) | 227 | +227 |
| `analyzer/routes/chat/branching.py` | (new) | 165 | +165 |
| `analyzer/routes/chat/export.py` | (new) | 93 | +93 |

### Verification

- Import-sanity: all 14 DocumentAnalyzer methods + all CI + chat handlers
  resolve through the new structure
- End-to-end smoke on dev + jacob:
  - chat/core: `'all-ok'`, `'jacob-r6-ok'`, `'r4-ok'`
  - chat/sessions: rename → 200
  - chat/branching: message-edit → 200
  - chat/export: 200 with 12,175-byte response
  - ci/setup: status/jurisdictions/cost-estimate/authority all 200
  - ci/runs: POST/GET/DELETE on test run all 200
  - ci/findings: 200
  - main/status: `svc=running paperless=72`

### Deployment status (all instances)

| Node | Port | v3.9.3 deployed |
|------|------|-----------------|
| Dev | 8052 | ✅ |
| Jacob/QA | 8053 | ✅ |
| Prod | 8051 | Pending image push |

### Backlog complete

All items from Entry 005's "next recommended refactor targets" are now done.
No further large-file splits warranted — remaining files are either at
appropriate size for their concern or represent cohesive single
responsibilities.

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
