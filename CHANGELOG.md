# Changelog

All notable changes to Paperless AI Analyzer are documented here.

---

## v3.9.16 — 2026-05-01

### Admin "Test" button uses /health, not a chat completion

Per the LLM-Proxy team's guidance: ``GET /health`` is the canonical
liveness probe for v2 endpoints, has been since v2.x, won't move.
``test_endpoint()`` route now hits ``/health``, returns the proxy's
own status payload (version, total/healthy provider counts), and
falls through to a chat-completion ping only on 404 (legacy v1
endpoints that don't expose ``/health``). Cheaper, more informative,
no LLM call billed.

### Verified working at the call_llm layer

End-to-end smoke confirmed today after the proxy team enabled their
openai + google providers (and answered Q3 "stale info — already
enabled"):

- ``task=classification`` → ``gpt-4o-mini`` → proxy → "Receipt." in 25→2 tokens
- ``task=extraction`` → ``gpt-4o`` JSON mode → proxy → ``{"answer":"PONG"}``
- ``task=chat`` / ``task=analysis`` → claude-sonnet-4-6 → proxy ``/v1/messages`` → claude-oauth (free)

So all four major call classes now route through the proxy: Anthropic
Claude via ``/v1/messages``, OpenAI GPT via ``/v1/chat/completions``.
Direct-API fallback paths preserved for outages.

### Backlog notes from the proxy team's reply

- ``cache_control: ephemeral`` is forwarded by the proxy verbatim (Q2),
  but ``claude-oauth`` subscription routing (which we use today) does
  not return cache savings — observed both
  ``cache_creation_input_tokens`` and ``cache_read_input_tokens`` as 0.
  Wiring prompt-caching for CI director-tier deferred until either we
  switch that traffic to a direct-API provider or claude-oauth gains
  cache support.
- ``safety-min`` is honored at routing-time (Q4); v3.9.15's
  ``safety-min=3`` on doc analysis is enforced by the proxy scorer.
- No rate-limit ceiling on our key today (Q5); ``rate_limit_rpm`` and
  ``spending_cap_usd`` available if we want them.

---

## v3.9.15 — 2026-05-01

### Poller no longer hard-exits on initial Paperless health-check failure

When `paperless-web` was down and the analyzer container restarted (e.g.
during a v3.9.14 image deploy), the poller would `sys.exit(1)`,
docker-compose restart-policy would respawn us, the next start would
also fail health-check — infinite restart loop. Discovered today during
the v3.9.14 prod deploy.

`analyzer/poller.py:309` now logs **"DEGRADED MODE"** and proceeds. Web
UI, AI Chat, RAG search, LLM-proxy admin, and document upload all work
without Paperless. Document polling retries on its normal cadence and
resumes automatically when Paperless returns. No restart-loop.

### llm-proxy2 (v2) promoted to pool primary

The v2 endpoint was seeded at v3.9.0 but disabled by default. Per the
proxy team's confirmation today, the **claude-oauth subscription path
on v2 is free for chat-class traffic** — every call we route through
v2 takes pressure off the depleted Anthropic direct-API key.

Migration applied via SQL update on all 3 instances:

  - `llm-proxy2 (v2)`        priority **10**, enabled (was: 20, disabled)
  - `llm-proxy-manager (v1)` priority **20**, disabled (was: 10, enabled)

v1's container has been Exited (0) for ~6h; demoting it removes the
"try v1 first, fall through" 1-second tax on every LLM call. The pool's
circuit breaker still isolates v2 if it ever misbehaves and the row can
be re-enabled by an admin via Config → 🛰 LLM Proxy.

---

## v3.9.14 — 2026-05-01

### LMRH builder upgraded to dim-based routing

Aligns with the LMRH 1.0 spec the proxy team finalized. Old builder
hardcoded model names per call-site (``model-pref=claude-sonnet-4-6``)
which forced a code change every time a new model shipped. New builder
emits cognitive-task dims and lets the proxy pick the model:

  - ``cost`` (economy / standard / premium) — per-task default in TASK_PRESETS
  - ``safety-min`` (1-5) — set on document-analysis paths (legal docs)
  - ``context-length`` — minimum tokens; CI director-tier hints 60k-80k

19 task presets covering core analysis, chat, and all Case Intelligence
specialists (entity / timeline / financial / contradiction / theory /
forensic / discovery / witness / warroom / report / settlement).

Backwards compatible: the old ``model_pref=`` and ``fallback_chain=``
kwargs still work and emit the deprecated dims (proxy treats as soft
preference). ``proxy_call.py`` call site unchanged.

### New: native Anthropic SDK clients pointed at the proxy

`proxy_manager.build_anthropic_client()` + `get_all_anthropic_clients()`.
Required for callers that need Anthropic-native features the OpenAI
compat shim can't express — primarily prompt caching with
`cache_control` blocks, used by Case Intelligence director-tier
(theory_planner, war_room, deep_financial_forensics) calls. The
Anthropic SDK passes through to the proxy's `/v1/messages` endpoint.
LMRH hint is pre-attached via `default_headers`.

### New: `lmrh.get_hint(task)` convenience

Operator can override any task's emitted hint at runtime by setting
`lmrh.hint.<task>` in the settings table — tunes routing without a
code change. Mirrors the coordinator-hub `get_lmrh_hint()` pattern.

---

## v3.9.13 — 2026-04-27

### Fixes

- **Project deletion now actually deletes the per-project Paperless stack.**
  Previously `DELETE /api/projects/<slug>` removed only the analyzer DB row
  and ChromaDB collection, leaving `paperless-web-<slug>` and
  `paperless-consumer-<slug>` containers running, the `paperless_<slug>`
  postgres database in place, and the auto-generated nginx location block
  serving 502s. Across the regression test runs that load this code path
  hardest, those orphans accumulated until the host was running 6+ stale
  Paperless stacks.

  New helper `deprovision_project_paperless(slug)` in
  `analyzer/services/project_provisioning_service.py`. Idempotent — safe to
  call when only some resources exist. Returns a structured summary that
  the route surfaces in the API response under the `deprovision` key:

  ```json
  {
    "success": true,
    "deprovision": {
      "containers_removed": ["paperless-web-foo", "paperless-consumer-foo"],
      "container_errors": [],
      "db_dropped": "paperless_foo",
      "db_error": null,
      "nginx_block_removed": true,
      "nginx_reloaded": true,
      "nginx_error": null
    }
  }
  ```

  The deprovision call runs *before* `project_manager.delete_project()`, so
  if anything fails the project still removes from the analyzer DB but the
  caller sees what was left behind.

### Drive-by fix (manual, not in code)

- Dev's `robinhoodproperties` project had `paperless_doc_base_url` set to
  `https://www.voipguru.org/paperless-robinhoodproperties` — an nginx
  location whose config file is `.disabled`. Updated to
  `https://www.voipguru.org/paperless` (the working shared route — doc 5
  and friends still live there with the `project:robinhoodproperties` tag).
  Fixes the "Document not found" symptom on Search & Analysis link clicks.

---

## v3.9.12 — 2026-04-27

### Fixes

- **`/api/system-health` no longer reports false `error` for `paperless_api`
  and `chromadb` under load.** The per-component timeout was 1.8s — too
  tight for a Paperless TCP+HTTP roundtrip or a Chroma `collection.count()`
  walk on a project with thousands of rows. Probes that simply ran longer
  than 1.8s were caught at the `future.result(timeout=…)` layer and
  reported as `error: Check failed: …`, which lit the dashboard red even
  though the services were healthy.
- Bumped `HEALTH_TIMEOUT` 1.8s → 6s (real failures still surface; slow ≠
  down).
- Distinguish probe-timeout from probe-exception: timeouts now return
  `warning` with detail `"Probe exceeded 6s — component slow, not down"`
  instead of being conflated with actual exceptions.

---

## v3.9.11 — 2026-04-24

### Fixes

- **Manage Projects "analyzed" count no longer inflated by CI findings.**
  Prod's `default` project was showing `6023 analyzed` while Paperless held
  810 docs and the Search & Analysis tab listed 808. Root cause:
  `_embed_ci_run_findings` (writeback_mixin) stores Case Intelligence
  artifacts (entities, timeline events, contradictions, theories,
  authorities, disputed facts) in the same Chroma collection as document
  embeddings, using string IDs like `ci:<run>:timeline:<id>`. `collection.
  count()` was counting those alongside real docs.

  `get_statistics()` now partitions the collection: numeric IDs → real
  documents (displayed as "📄 N analyzed"), everything else → CI findings
  (displayed as "🧠 N CI findings" when non-zero). Counts on Manage
  Projects now agree with Search & Analysis.

- **New admin action: "🧹 Clean Stale" per project.** Purges Chroma rows
  whose numeric `document_id` is no longer present in Paperless — fixes
  the symptom where clicking a document tile in Search & Analysis returns
  a Paperless 404 because the doc was deleted after being embedded. Uses
  the project-specific Paperless client when available and falls back to
  the global client filtered by `project:<slug>` tag when the per-project
  token is stale. CI findings (non-numeric IDs) are never touched.
  Response returns `purged`, `docs_after`, and `source` so the toast can
  surface what happened.

### API changes

- `GET /api/projects` — each project now includes `ci_finding_count`.
- `POST /api/projects/<slug>/cleanup-stale-embeddings` (new, admin-only).

---

## v3.9.10 — 2026-04-24

### Docker image diet — 7.73 GB → 3.84 GB (50% reduction)

Same app, same dependencies, same behavior. The giant was GPU libraries that
the CPU-only server never touches.

**Dockerfile:** install CPU-only `torch` + `torchvision` from
`https://download.pytorch.org/whl/cpu` BEFORE the main `pip install -r
requirements.txt`. The transitive torch dep pulled by `unstructured[pdf]`
(via `timm`, `torchvision`, `effdet`) is now satisfied without fetching the
default CUDA wheel. Removed from the image: `nvidia-*` CUDA runtime
(~2.7 GB), `triton` GPU compiler (~640 MB), and `torch` shrinks
~1.2 GB → ~250 MB. `torch.cuda.is_available()` returns `False` —
confirmed no code path relies on CUDA.

**`.dockerignore`** (new): excludes build-context cruft that never needs to
enter the image — `.git/`, `backups/`, `__pycache__/`, `*.pyc`, logs, editor
files, and `profiles/staging/` (1,443 AI-suggested YAML files accumulated
over months, ~5.8 MB, that reference real customer document IDs; now
generated fresh per-instance at runtime).

**Why it matters**
- Docker Hub push time drops roughly proportionally (4.5 GB less to upload).
- Prod pull on recreate is ~50% faster.
- Fewer bytes shipped that could accidentally leak customer metadata.

**No behavior changes.** Version bumped from 3.9.9 → 3.9.10 to mark the
image change so rollback tags are unambiguous.

---

## v3.9.9 — 2026-04-23

### Refactored — two more splits (see refactor-log Entry 012)
- **`templates/docs.html` (1,820 lines) → 591-line shell + 14 page partials**
  in `templates/docs_pages/` (`overview.html`, `getting_started.html`,
  `projects.html`, `upload.html`, `chat.html`, `search.html`,
  `anomaly_detection.html`, `tools.html`, `configuration.html`,
  `users.html`, `llm_usage.html`, `api.html`, `case_intelligence.html`,
  `court_import.html`). Each page < 200 lines.
- **`routes/court.py` (847 lines) → `routes/court/` package** —
  5 files: `__init__.py` (blueprint), `helpers.py` (400 — internal job worker
  + credential gate + connector builder), `credentials.py` (243), `search.py`
  (97), `imports.py` (145).

### Why
After v3.9.8 all files below 1,400 lines; these two were the next biggest
editing-friction hotspots. The docs.html split was especially high-value for
AI work: touching a single doc page no longer requires loading 1,800 lines
of sibling pages.

---

## v3.9.8 — 2026-04-23

### Refactored — three large files split (see refactor-log Entry 011)
- **`static/js/config.js` (2,361 lines) → `static/js/config/` package** —
  4 modules: `core.js` (465), `projects.js` (905), `search.js` (285),
  `profiles_ai.js` (705). Loaded sequentially via `<script>` tags; shared
  top-level `let`/`const` state continues to work across classic scripts.
- **`static/js/ci.js` (2,229 lines) → `static/js/ci/` package** —
  5 modules: `setup.js` (755), `goal_assist.js` (650), `specialists.js` (365),
  `tier5.js` (385), `report.js` (70).
- **`case_intelligence/web_researcher.py` (1,362 lines) → `case_intelligence/web_researchers/` mixin package** —
  7 files: `constants.py`, `http_utils.py`, `base.py`, `providers_legal.py`,
  `providers_general.py`, `providers_entities.py`, `__init__.py`. Main class
  composes 3 provider mixins + 1 base mixin. `web_researcher.py` kept as a
  2-line re-export shim so every `from analyzer.case_intelligence.web_researcher
  import WebResearcher` continues to work.

### Why
All three files were above the "one edit reloads the whole file" threshold
for AI-assisted work. After v3.9.7 + v3.9.8, no file in the codebase exceeds
1,400 lines; the largest remaining single file is the CI `managers_mixin.py`
at 1,004 lines, which is cohesive (one concern: the Manager phase of the CI
pipeline) and deferred until it crosses ~1,500 lines.

---

## v3.9.7 — 2026-04-23

### Refactored — maintainability pass
- **`analyzer/routes/projects.py` (988 lines) → package of 5 cohesive
  modules** (`core`, `paperless_config`, `provisioning`, `migration`,
  `documents`). External API unchanged — `web_ui.py` still registers
  `projects.bp`. See refactor-log Entry 010.
- **`templates/dashboard.html` (2,994 lines) → 1,089-line shell + 3 Jinja
  partials** in `templates/partials/`: `tab_config.html` (625),
  `tab_upload.html` (395), `tab_case_intelligence.html` (888). Tab panels
  under ~150 lines stay inline.

### Fixed — during refactor drive-by
- `GET /api/orphan-documents` no longer 500s (`'int' object is not
  subscriptable`) when Paperless returns a malformed entry in the results
  list. Non-dict items are now skipped rather than aborting the whole
  response.

### Why
`routes/projects.py` had grown back to 988 lines with 24 routes covering
six distinct concerns — any edit pulled in the full file. `dashboard.html`
at ~3,000 lines regularly overflowed AI context windows when touching any
UI area. Both splits are pure reorganisation; no behavior changes beyond
the orphan-documents bugfix.

---

## v3.9.6 — 2026-04-23

### Added — project provision throttling
- **Sequential container provisioning with 180s cooldown.** Spinning up a
  dedicated Paperless-ngx stack (web + consumer + postgres + redis) is
  expensive during the first minute. Back-to-back project creates (e.g.
  from a regression run) used to saturate the host CPU/RAM and starve
  earlier stacks before they finished warming up. New behaviour:
  - First project after boot starts immediately.
  - Every subsequent project waits `PROVISION_MIN_INTERVAL_SECS`
    (default 180s, env-tunable) after the previous provision started.
  - A single background worker drains the queue FIFO, sleeping out the
    cooldown between items.
- **UI feedback.** Project cards show `⏸️ Waiting in provisioning queue
  (#N). Starting in ~M mins — host throttled to one Paperless stack at a
  time.` while `status=queued_waiting`. Toast on project create reports
  the ETA immediately: `Project created — queued for Paperless
  provisioning (#N in line, starts in ~M min).`
- New fields on `GET /api/projects/<slug>/provision-status`:
  `queue_position`, `eta_seconds` (both refresh live every 5s).
- New field on `POST /api/projects` response: `provision.status`,
  `provision.queue_position`, `provision.eta_seconds`,
  `provision.throttle_interval_secs`.

### Fixed — Paperless-slow failures that stalled /api/status, /api/reconcile, /api/orphan-documents, POST /api/projects, /api/scan/process-unanalyzed
- **Root cause.** Several endpoints called the Paperless client on the
  request thread. With a slow or unreachable Paperless, the `tenacity`
  decorator retried for up to 45s, saturating the 4-thread Waitress
  pool and making even fast admin requests time out at 10–15s.
- **Fixes:**
  - `PaperlessClient.health_check()` now accepts a `timeout` (default 3s)
    and caches the result for 10s so hot paths don't ping repeatedly.
  - `PaperlessClient.__init__` installs a default 15s per-request timeout
    on the shared `requests.Session` — a slow upstream can no longer hang
    a Waitress thread forever.
  - `GET /api/orphan-documents`, `POST /api/reconcile`,
    `POST /api/scan/process-unanalyzed`, `POST /api/projects`,
    `GET /api/status` all short-circuit with a structured 503 /
    `paperless_available: false` response when `health_check()` fails
    instead of triggering the retry loop.
  - `get_documents_without_project()` now scans one page of 100 docs
    (the route caps results at 100 anyway) instead of page_size=1000 —
    the 1000-row query was timing out against instances with many docs.
  - `POST /api/reconcile` paginates with `page_size=200` instead of 1000
    for the same reason.
  - `POST /api/projects` tag creation moved off the request thread —
    a slow shared Paperless used to block project create for 15s+
    waiting on `get_or_create_tag`. Now returns in < 50ms; the tag is
    created asynchronously (the provision worker also ensures it).
  - `analyzer/app.py` — Waitress `threads=4` → `threads=16`. 4 was too
    few: a single slow Paperless call could block the whole pool.

### Why
This pass closes the remaining regression failures where Paperless itself
was slow, not unreachable — `health_check()` returned True, but the next
call still took 15–45s. Bounding every request with a default timeout +
caching the health check + moving slow ops off the request thread is the
consistent fix across all affected endpoints. The provisioning throttle
prevents the test-driven reproduction of the original Waitress saturation
by limiting how fast new Paperless stacks can be spun up.

---

## v3.9.5 — 2026-04-23

### Fixed — the last regression stragglers
- **CI goal-assistant 500** — `routes/ci/setup.py` tried to
  `from analyzer.routes.chat import _fetch_url_text`, but chat was converted
  to a package in v3.9.3 (that symbol moved to
  `analyzer.services.web_research_service.fetch_url_text`). Stale import
  fixed — all 3 goal-assistant tests (25.11 / 25.12 / 25.25) now pass.
- **`POST /api/active/duplicates/remove` crashed** with `NameError:
  safe_json_body is not defined` — the bulk-edit that introduced
  `safe_json_body` across routes missed adding the import in
  `routes/profiles.py`. Fixed.
- **`GET /api/active/duplicates` timeout on large profile sets** — the
  O(n²) similarity-scan loop runs 3.3M iterations on dev's 2,580 profiles.
  Fix: skip the similarity pass when `len(profiles) > 500` unless
  `?deep=1` is passed. Fast exact-hash scan still runs; response now
  returns in < 1s with a `scan_note` field indicating the skip.
- **Zero-byte file upload returned 500** — `POST /api/upload/submit` with
  an empty file bubbled an upstream error into a 500. Fix: explicit
  `os.path.getsize == 0` check, clean 400 "File is empty (0 bytes)".

### Why
These were the last four items on the v3.8.2 regression backlog after v3.9.4
cleared the main 12. The goal-assistant stale import was a post-refactor
artifact — my v3.9.3 package split missed this one caller in CI setup.
The duplicates perf fix is a proper optimization: O(n²) against a growing
profile directory would have timed out more frequently as more suggested
profiles accumulated.

---

## v3.9.4 — 2026-04-23

### Fixed — dashboard noise
- **`/api/system-health` false negatives** — `paperless_api` and `chromadb`
  components always reported `error` with detail
  "Working outside of application context", making the dashboard health widget
  permanently red. Root cause: the probes were submitted to a
  `ThreadPoolExecutor` whose worker threads don't have Flask's app context.
  Fix: capture the client, analyzer, and state-manager on the request thread,
  pass them as arguments to the probe functions.
- **`/api/vector/reembed-stale` background crash** — same class of bug:
  `_run()` closure used `current_app.document_analyzer` from a background
  thread. Fix: capture the analyzer on the request thread.
- **`check_stale_embeddings` startup ValueError on CI composite chroma IDs**
  — `analyzer/poller.py` did `int(chroma_id)` over all IDs, crashing on
  CI findings like `ci:<run>:timeline:N`. Fix: try/except, skip non-numeric IDs.

### Fixed — regression failures (12 categories addressed)
- **Malformed-JSON body handling** — five endpoints previously crashed with
  `'str' object has no attribute 'get'` when sent a valid-but-non-object
  JSON body (e.g. `"plain string"`). Added `analyzer.app.safe_json_body()`
  helper that returns `{}` for non-dict bodies, and bulk-replaced 55 call
  sites across 18 route modules. Affected endpoints (now return 400/422
  instead of 500): `/api/projects`, `/api/trigger`, `/api/docs/ask`,
  `/api/ai-form/parse`, `/api/chat` — plus background robustness in ~50
  other POST/PATCH routes.
- **`PATCH /api/users/<bad_uid>` silently succeeded** — missing existence
  check returned 200 for any uid. Fix: return 404 if uid not found.
- **`POST /api/reprocess/<bad_id>` → 500** — `AnalyzerState` object has no
  `.get` method. Fix: verify document exists in Paperless first; return
  404 on any lookup failure (not 503).
- **`GET /api/court/docket/<unknown_system>/<case>` → 500** — `RuntimeError`
  "Unknown court system: X" surfaced as 500. Fix: catch and return 400 with
  `{supported: ['federal', 'nyscef']}` hint.
- **`POST /api/ci/runs` with `auto_start=true` → 500** `"name 'current_app'
  is not defined"` — missing module-level import in `routes/ci/runs.py`
  after the v3.9.3 package split. Fix: add `current_app` to the flask import.
- **`POST /api/upload/from-url` returning 500 on upstream failure** — was
  an internal-error code for what is clearly a bad-gateway situation (Paperless
  rejected the upload). Fix: return 502 with structured error body.
- **`POST /api/scan/process-unanalyzed` background-task app-context crash**
  — same class as reembed-stale. Fix: capture `paperless_client` and
  `document_analyzer` on request thread, pass to background worker.
- **`/api/status` missing canonical key names** — external consumers / test
  harnesses expected `total_documents` / `analyzed_documents` / `analyzed_count`
  at the top level. Fix: add aliases (underlying values unchanged).
- **`POST /api/vector/delete-document` crashed on string doc_id** —
  `int(doc_id)` raised `ValueError` when test sent `{"doc_id": "nonexistent"}`.
  Fix: explicit try/except with clean 400 response.
- **`POST /api/chat/sessions/<id>/share` only accepted `username` field** —
  older API consumers sent `uid` (int). Fix: accept either — resolve to the
  same user record.

### Added
- **`analyzer.app.safe_json_body()`** helper — returns `{}` when request body
  isn't a JSON object (string, number, malformed, missing). Every route that
  previously did `request.json or {}` or `request.get_json() or {}` now uses
  this — eliminates the "str has no attribute 'get'" class of bug at every
  POST/PATCH handler.

### Why
The v3.8.2 regression triage identified 20 test failures. Most were small,
repetitive crashes caused by two patterns: (1) Flask's `request.get_json()`
returning a non-dict when the client sent valid-but-non-object JSON, and
(2) `current_app` accesses from background threads outside request context.
This release addresses both classes systematically plus several
point-fixes. Regression pass rate should jump from 94.7% (674/712) to
~97% (691/712) with the remaining ~20 failures queued for v3.9.5 (upload
submit, duplicates profile scan, CI goal-assistant contextual, DELETE CI
run — all smaller individual fixes).

---

## v3.9.0 — 2026-04-23

### Added
- **LLM Proxy pool with LMRH routing** — all 29 LLM call sites across the app
  (chat, docs-ask, form parsing, anomaly analysis, 13 Case Intelligence submodules)
  now route through an ordered pool of llm-proxy endpoints with per-endpoint
  circuit breakers (3 failures → 60s cooldown). Each call emits an `LLM-Hint`
  header (LMRH protocol) with task-specific `model-pref` so the proxy can pick
  the best upstream provider. Direct Anthropic/OpenAI SDK calls are preserved
  as absolute-last-resort fallback when the proxy pool is exhausted.
- **`analyzer/llm/proxy_manager.py`** — ported from devingpt's
  `services/proxy_manager.py` (the canonical pattern). In-memory circuit breaker,
  `get_healthy_endpoints()`, `build_client()` with v1 (Bearer) / v2 (x-api-key) auth.
- **`analyzer/llm/lmrh.py`** — LLM Model Routing Hint builder with presets for
  17 task types (analysis, chat, qa, extraction, classification, reasoning,
  entity, timeline, financial, contradiction, theory, forensic, discovery,
  witness, warroom, report, settlement).
- **`analyzer/llm/proxy_call.py`** — unified `call_llm()` chokepoint + `call_llm_json()`
  helper for Case Intelligence submodules. Retry loop catches connection/timeout
  errors, marks failure, tries next endpoint.
- **New admin UI: Config → 🛰 LLM Proxy sub-tab** — table of endpoints with
  inline edit (label, url, version, priority, enabled), per-row Test button
  (shows `✓ <model>` green or `✗ <error>` red for 3s), Delete, and circuit
  breaker live status. Add Endpoint form with v1/v2 version selector.
- **`GET/POST/PATCH/DELETE /api/llm-proxy/endpoints`** + `POST /.../<id>/test` — admin-only
  blueprint at `analyzer/routes/llm_proxy.py`. API keys masked in responses.
- **DB table `llm_proxy_endpoints`** (SQLite, `/app/data/app.db`) — seeded on
  first boot from `LLM_PROXY_KEY` env var with two entries: llm-proxy-manager
  v1 (enabled, priority 10) and llm-proxy2 v2 (disabled, priority 20 — admin
  enables once v2 API keys are provisioned).
- **New env vars for all three Docker services** — `LLM_PROXY_KEY`,
  `LLM_PROXY2_URL`, `LLM_PROXY_URL` sourced from `.env`'s `LLM_PROXY_KEY_DEVINGPT`.

### Changed
- **`analyzer/llm/llm_client.py`** — `_call_llm()` simplified from 180-line
  multi-provider retry loop to a single `call_llm()` call. Public API
  (`analyze_anomalies`, `generate_document_summary`, etc.) unchanged.
- **All Case Intelligence submodules** (`entity_extractor`, `entity_merger`,
  `timeline_builder`, `financial_extractor`, `contradiction_engine`,
  `theory_planner`, `discovery_analyst`, `forensic_accountant`,
  `deep_financial_forensics`, `witness_analyst`, `war_room`, `report_generator`,
  `multi_model_synthesis`, `orchestrator`) — each `_call_llm()` body replaced with
  a short wrapper calling the proxy pool. Provider-specific branches removed;
  LMRH `fallback-chain` + `model-pref` drive routing. Direct-provider fallback
  via `direct_provider`/`direct_api_key`/`direct_model` parameters preserves
  the original behavior when the pool is empty.
- **Route handlers** (`chat.py`, `docs.py`, `forms.py`, `ci.py`) — replaced
  inline `if provider == 'openai' ... elif provider == 'anthropic'` blocks
  with single `call_llm()` call. Pool-exhausted failures return structured
  HTTP 503 with `source: 'llm-pool-exhausted'` (not opaque 500s).
- `multi_model_synthesis.py` keeps the parallel ThreadPoolExecutor but each
  branch now uses `call_llm()` with `fallback_chain=<provider>` to pin the
  proxy to a single upstream per branch.

### Why
The 2026-04-22 prod outage (Anthropic credits depleted, no OpenAI fallback
configured, Waitress queue flooded to depth 87+) proved that direct provider
keys are a single point of failure. Devingpt and coordinator-hub have been
using llm-proxy for multi-provider failover for months; this adopts the same
pattern. Existing `ai_config.json` global keys are preserved as the last-resort
fallback so behavior is strictly additive for current deployments.

---

## v3.8.1 — 2026-04-22

### Fixed
- **`POST /api/reconcile` crash and hang** — Fetching all Chroma embeddings to find orphans
  caused a >30 s hang (807 docs × 7 s/page) followed by a `ValueError` crash when CI
  analysis embeddings (composite IDs like `ci:uuid:type:N`) were present. Fixed: now fetches
  only Chroma IDs (`include=[]`, nearly instant), skips composite IDs silently, and uses
  `page_size=1000` to collapse the Paperless pagination from 9 round-trips to 1.
- **`POST /api/trigger` returns 500 for non-existent documents** — Two issues: (1) calling
  without `Content-Type: application/json` caused an `AttributeError` (`NoneType.get('doc_id')`)
  that surfaced as a 500; (2) a missing document raised a `RetryError` surfaced as an opaque
  500. Fixed: `get_json(force=True, silent=True)` accepts any Content-Type; network/404
  exceptions now return a clean 404 JSON response.
- **`POST /api/bug-report` rejected JSON callers** — Sole POST endpoint that used
  `request.form` exclusively; any API consumer sending `application/json` got a silent 400
  `"Please describe the problem"`. Fixed: dual-accept — JSON path handled via `request.is_json`
  branch; FormData path (UI, with optional HAR file upload) unchanged.

### Added
- **`POST /api/ci/runs/<run_id>/interrupt`** — New endpoint to soft-stop a running or queued
  CI run and set its status to `interrupted`. Unlike `/cancel` (terminal, no restart),
  `interrupted` runs can be restarted via `/start` or `/rerun`. Respects the same authorization
  and CI-gate checks as all other run endpoints.
- **`auto_start` param and `start_url` hint in `POST /api/ci/runs`** — Passing
  `auto_start: true` creates the run and immediately launches the orchestrator in one call
  (returns `status: "queued"` and HTTP 201). All responses now include `start_url` pointing
  to the `/start` endpoint for clients that prefer two-step flow.
- **Top-level 👥 Users tab** — Admin users now see a **👥 Users** button in the main tab bar
  as a shortcut to the Config → Users sub-tab. Implemented via `goToUsersAdmin()` in `init.js`
  which activates the config panel and users sub-tab directly, highlighting only the Users
  button (not the Configuration button).

### Documentation
- **API conventions** added to `contributing.md`: `GET /api/users` returns a wrapped
  `{"users": [...]}` object (not a bare array) intentionally; `PATCH`/`DELETE /api/users/<int:uid>`
  require the integer `id`, not the username string.

### Testing
- Expanded Playwright regression suite from 99 tests (15 phases) to **712 tests across 36 phases**
  (`/tmp/full_regression_v2.py`). New coverage includes: Chat branching/sharing/export/compare,
  Upload directory scan and cloud links, Court credentials/import lifecycle, Case Intelligence
  run CRUD/lifecycle/findings/reports/sharing/authority, stale RAG re-embedding, multi-user
  cross-role interaction flows, cross-feature end-to-end workflows, and error-handling edge cases.
- **Result: 674 passed / 20 failed / 18 skipped** — 20 open issues queued for v3.8.2.

---

## v3.7.4 — 2026-03-13

### Fixed
- **New project modal stays open after save** — Two bugs combined: (1) `get_or_create_tag` on Paperless could throw an exception after the project was already committed to the DB, causing a 500 response that left the modal open showing an error (project existed but wasn't visible until refresh); (2) `closeProjModal()` was called after `res.json()` parsing, so any JSON/network error after a successful 2xx also blocked modal close. Fixed: tag creation is now best-effort (wrapped in try/except, logs a warning on failure); modal now closes immediately on any `res.ok` response before list reload.

---

## v3.7.3 — 2026-03-11

### Added
- **CI Re-run button for interrupted runs** — When a CI run is killed mid-flight by a container restart, it now shows `⚡ Interrupted` status with a **🔄 Re-run Same Settings** banner and button that creates a new run with identical parameters, skips clarifying questions, and starts immediately.
- **Graceful shutdown for active CI runs** — `stop_grace_period: 600s` in docker-compose gives containers up to 10 minutes to finish before Docker force-kills. A SIGTERM handler in `main.py` waits up to 540 seconds for any in-flight CI job threads before exiting cleanly, preventing mid-run data corruption.
- **POST `/api/ci/runs/<run_id>/rerun`** — New endpoint: copies all parameters from a completed/interrupted/failed run, creates a new run record, immediately starts the orchestrator, and returns the new `run_id`.

### Fixed
- **AI Key Guide modal — input/send box invisible** — The modal used a fixed `height:600px` with the chat area taking `flex:1`, leaving the input pinned far below the AI's first message. Fixed by switching to `max-height` on the modal and capping the chat area to `max-height:min(320px,45vh)` so the modal shrinks to content size and the input is always immediately below the conversation.
- **Clickable URLs in AI chat bubbles** — AI messages were rendered with `textContent`, so URLs appeared as plain, unclickable text. Fixed: all AI bubbles now use a `_akgmLinkify()` helper that HTML-escapes content then converts `https://` URLs to proper `<a target="_blank">` links.
- **Google CSE guidance — `*` wildcard rejected** — The AI-guided API key setup was advising users to add `*` as a site-to-include pattern for whole-web search; Google rejects this as an invalid public-suffix pattern. Updated `SETUP_NOTES['gcse']` with accurate guidance: the whole-web toggle is a section-level control (deprecated in newer PSE UIs), not a site pattern. Guidance now recommends Brave Search API or Serper.dev as reliable alternatives when the toggle is absent.
- **Search & Analysis showing 0 documents** — The `/api/projects/<id>/documents` endpoint crashed with `invalid literal for int() with base 10: 'ci:...'` when CI run embeddings (with non-numeric `document_id` metadata) were stored in the same Chroma collection as Paperless documents. Fixed with a `try/except` around the `int()` cast; CI embeddings are silently skipped and all real documents are returned correctly.
- **`recover_orphaned_runs` marks runs `interrupted` (not `failed`)** — On startup, any run left in `running`/`queued` state from a previous crash is now marked `interrupted` with a progress-aware message and directed to use the Re-run button, instead of the generic `failed` status.

---

## v3.7.2 — 2026-03-10

### Added
- **Tier 5 White Glove — Deep Financial Forensics** (`deep_financial_forensics.py`) — Goes beyond Tier 3 forensic accounting with: Benford's Law first-digit analysis (deterministic chi-squared test; detects fabricated/manipulated amounts), beneficial ownership tracing (who ultimately controls each entity through shell chains), round-trip transaction detection (A→B→C→A money circles), shell entity identification (offshore jurisdictions, no business purpose, pass-through patterns), advanced layering analysis (multi-hop obscuring of fund origin), suspicious cluster detection (same-date coordinated transactions), and financial crime risk score (0–100). New **Deep Financial Forensics** accordion tab (White Glove badge).
- **Tier 5 White Glove — Trial Strategy** (`TrialStrategist` class in `war_room.py`) — Comprehensive trial preparation memo: opening statement theme (one sentence), our vs. their narrative arc, witness order with strategic rationale and cross risk, top 10 key exhibits with introduction strategy, motions in limine with legal basis and likelihood of success, closing argument themes, jury selection profile (favorable/unfavorable juror types + voir dire questions), and top 3 trial risks with contingency plans. New **Trial Strategy** accordion tab (White Glove badge).
- **Tier 5 White Glove — Multi-Model Synthesis** (`multi_model_synthesis.py`) — Runs theory generation simultaneously on both Anthropic claude-opus-4-6 and OpenAI gpt-4o in parallel (ThreadPoolExecutor), then runs a synthesis pass to: identify agreed findings (both models independently found — higher confidence), surface model-unique findings (may be novel insights or hallucinations — flagged), flag direct disagreements (high-uncertainty areas), and produce a merged summary with model agreement rate. New **Multi-Model Analysis** accordion tab with agreement rate bar and disagreement flags.
- **Phase 3B** in orchestrator — Tier 5 White Glove phases run in parallel after Phase 3A (senior partner review): deep forensics + trial strategy + multi-model synthesis all fire concurrently via ThreadPoolExecutor.
- **3 new DB tables**: `ci_deep_forensics`, `ci_trial_strategy`, `ci_multi_model_comparison`
- **3 new API endpoints**: `/api/ci/runs/<id>/deep-forensics`, `/api/ci/runs/<id>/trial-strategy`, `/api/ci/runs/<id>/multi-model`
- **Tier 5 UI sections**: all three new accordions are hidden until a Tier 5 run populates them; no UI change for Tier 1–4 runs

---

## v3.7.1 — 2026-03-10

### Added
- **Budget overage policy** — Two new checkboxes in CI Setup: "Allow up to 20% overage" (hard stop at 120% of budget) and "Allow unlimited overage" (budget is a goal only, run never blocked). Checkboxes are mutually exclusive. Stored as `allow_overage_pct` on the run (0 = hard 100% block, 20 = 120% ceiling, -1 = unlimited).
- **Smarter budget notifications** — Changed from every-10% to targeted checkpoints: 50%, 70%, 80%, 90%. Notifications at 80% and 90% get `URGENT:` subject-line prefix. Email body includes current cost, projected total, and budget overage policy.
- **Senior Partner Notes plain-text fallback** — War Room tab now renders senior partner review notes stored as plain markdown text (pre-structured-JSON format) with a pre-wrap fallback display path, in addition to the structured field rendering.
- **All external links in AI Chat open in new tab** — `target="_blank" rel="noopener noreferrer"` applied to all `https://` links in both chat message and compare-message rendered markdown.
- **Key Findings → Key Findings and Rulings** — All instances of "Key Findings" renamed to "Key Findings and Rulings" throughout findings display, report generator, and PDF output.

### Fixed
- Specialist accordion render functions unwrap the `{present, data}` API response wrapper before accessing fields — was previously passing the wrapper object directly, causing all Tier 3/4 accordions to remain hidden.

---

## v3.7.0 — 2026-03-09

### Added
- **Branching chat history** — editing a user message no longer destroys the original exchange. Instead, a new branch is created and both the original and edited conversation threads are preserved. You can navigate between them with **← 1/2 →** arrows that appear on the left of any edited message.
- **Variant navigation** — each fork point in a chat session shows the variant counter and left/right arrows. Clicking switches the full view (user message + all subsequent AI replies) to the other branch. You can continue chatting from any branch independently.
- **Persistent tree structure** — the `chat_messages` table now uses a `parent_id` linked-list; `chat_sessions` tracks `active_leaf_id`. Old sessions continue to work unchanged (linear order, no arrows shown).

---

## v3.6.9 — 2026-03-09

### Added
- **Web search in AI Chat** — say "search for...", "find online...", "look up...", or any similar phrase and the AI now runs a live DuckDuckGo search and incorporates the results. No manual URL needed.
- **Persistent web context across chat turns** — URLs fetched and searches performed in earlier turns of a conversation are remembered and available for follow-up questions. The AI no longer "forgets" what it fetched from a prior message.
- **AI web capability statement** — the chat system prompt now explicitly instructs the AI that it has web access via this system, preventing responses that claim it "cannot browse the internet."

---

## v3.6.8 — 2026-03-09

### Added
- **Smart court docket URL resolver** — Justia docket URLs (which block automated fetches with 403) are now automatically resolved through CourtListener's free public API, returning case name, parties, attorneys, judge, cause, and nature of suit.

---

## v3.6.7 — 2026-03-09

### Added
- **Web URL fetching in AI Chat** — when a user includes a URL in a chat message, the AI now fetches the page content and uses it in its answer. Up to 3 URLs per message; content truncated to 4,000 chars each. HTML, `<script>`, and `<style>` blocks stripped to clean text.
- **Honest URL failure responses** — if a URL cannot be fetched (domain not found, SSL error, timeout, etc.), the AI explicitly tells the user it could not access that URL and gives the reason, rather than hallucinating or answering with irrelevant content.
- **URL fetching in CI Goal Assistant** — the "Refine with AI" goal assistant chat also now fetches URLs shared by the user, so they can paste a case link and have it inform the goal statement.

---

## v3.6.6 — 2026-02-27

### Added
- **5-Tier Analysis System** — Replaces the prior 3-tier `max_tier` select with a 5-card visual tier selector:
  - Tier 1 Junior Associate: extraction only (entities, timeline, financial)
  - Tier 2 Senior Associate: + contradictions, basic theories, free web research
  - Tier 3 Partner ★ (default): + forensic accounting, discovery gap analysis, adversarial theory testing
  - Tier 4 Senior Partner: + witness intelligence dossiers, war room opposing-counsel simulation, senior partner review
  - Tier 5 White Glove: + multi-model comparison, deep financial forensics, trial strategy
  - Live cost estimate badge updates as tier or document count changes
- **Phase 1M — Entity Merge Pass** — After extraction, merges duplicate entity names (deterministic normalization + AI-assisted fuzzy match); shows `(also: J. Smith, ...)` aliases in the entities display; merged duplicates count shown in status
- **Phase 2F — Forensic Accounting** (Tier 3+) — New `ForensicAccountant` module: detects structuring, round-number anomalies, timing correlations, cash-flow reconciliation, balance discontinuities, transaction chain tracing; new **Forensic Accounting** accordion tab in findings
- **Phase 2D — Discovery Gap Analysis** (Tier 3+) — New `DiscoveryAnalyst` module: identifies missing document types, custodian gaps, spoliation indicators, RFP list, subpoena targets; new **Discovery Strategy** accordion tab with Copy RFP List button
- **Phase 2W — Witness Intelligence** (Tier 4+) — New `WitnessAnalyst` module: builds per-witness dossiers with credibility score, impeachment points, prior inconsistent statements, financial interests, and key deposition questions; new **Witness Intelligence** accordion tab with collapsible per-witness cards and credibility bars
- **Phase 2R — War Room** (Tier 4+) — New `WarRoom` module: simulates full opposing-counsel case theory, identifies top 3 dangerous arguments with client responses, ranks client vulnerabilities with mitigations, flags documents dangerous to client, generates settlement valuation range; new **War Room** accordion tab
- **Phase 3A — Senior Partner Review** (Tier 4+) — Second LLM pass that challenges the Director synthesis: finds missed issues, unsupported conclusions, theories that won't survive cross-examination; Senior Partner notes displayed at bottom of War Room tab
- **Enhanced Theory Generation** — 12 theories max (up from 8); each theory now includes legal element mapping, 2-paragraph legal argument memo, knowledge cutoff date, companion theories, and discovery implications
- **Enhanced Contradiction Detection** — Contradiction engine now detects patterns of conduct (3+ instances), behavioral tells (hedging language, formality shifts, CC drops, phone shifts), communication gaps, and knowledge cutoff per disputed fact
- **4 New DB Tables**: `ci_forensic_report`, `ci_discovery_gaps`, `ci_witness_cards`, `ci_war_room`
- **5 New API Endpoints**: `/api/ci/runs/<id>/forensic-report`, `/api/ci/runs/<id>/discovery-gaps`, `/api/ci/runs/<id>/witness-cards`, `/api/ci/runs/<id>/war-room`, `/api/ci/cost-estimate`
- **New Specialist Modules**: `entity_merger.py`, `forensic_accountant.py`, `discovery_analyst.py`, `witness_analyst.py`, `war_room.py`

### Changed
- **"Key Findings" → "Key Findings and Rulings"** — Renamed everywhere in the findings panel
- `estimate_run_cost()` now returns `{total_usd, breakdown_by_task}` dict; `estimate_run_cost_simple()` added for backward compat
- `task_registry.py` — 9 new task types added with tier assignments and per-run cost estimates
- `ci_theory_ledger` — new columns: `legal_element_mapping`, `theory_legal_memo`, `companion_theories`, `discovery_needed`, `model_source`
- `ci_entities` — new column: `merged_into` (FK to canonical entity), `aliases` (JSON array of name variants)
- `ci_runs` — new column: `analysis_tier`
- Entity display now shows `(also: ...)` aliases under canonical name

---

## v3.6.5 — 2026-02-28 (updated 2026-02-28 — pass 4)

### Added
- **CI Web Research (Phase W)** — New phase runs between Phase 1 (extraction) and Phase 2 (synthesis):
  - **CourtListener** (free): Searches federal case law and opinions via the CourtListener REST API; returns binding/persuasive cases with excerpts and direct links.
  - **Harvard Caselaw Access Project** (free): Searches 6.7M US state and federal cases; official citations + court details.
  - **DuckDuckGo web search** (free): General web search for news, background, recent legal developments — no API key required.
  - **Entity background research** — After extracting entities (persons, orgs), automatically searches court records and news to build character/background profiles that are injected into theory generation. Defense mode looks for impeachment material; plaintiff/prosecution mode looks for prior bad acts and criminal history.
  - **Role-aware query bias** — All searches are biased toward strengthening the user's litigation position: defense searches favor acquittal/suppression/dismissal precedents; plaintiff/prosecution searches favor judgments/convictions.
  - **Optional paid sources**: Tavily AI search (free tier: 1k/mo), Serper.dev (Google results), and Lexis-Nexis enterprise integration — each configurable per run with an API key field in the UI.
  - **Web research section in CI setup** — Collapsible "🌐 WEB RESEARCH" card with individual checkboxes for each free source and key-entry fields for paid sources.
  - **Web Research accordion in findings** — Dedicated section in CI findings panel showing retrieved case law, entity background profiles (court history + web mentions), and general web results.
  - Web-sourced legal authorities are automatically added to the `ci_authorities` table and injected into the Authorities manager results.
  - Entity background summaries are injected into the Theory Ledger generation prompt as additional context.
  - New DB table `ci_web_research` stores all search results per run with search type, query, source, and JSON results.
- **Expanded Web Research sources** — 12+ new data source integrations across 5 categories, each with checkboxes and key fields:
  - **Case Law**: Docket Alarm (675M fed+state dockets, $99/mo), UniCourt (normalized court data, OAuth)
  - **Web Search**: Brave Search (independent index, $5/1k), Google Custom Search (100/day free), Exa AI (neural/semantic, $7/1k), Perplexity Sonar (AI-synthesized answers with citations)
  - **Public Records & Background** (new category): BOP Federal Inmate Locator (free, criminal history since 1982), OFAC/Treasury Sanctions (free SDN list), SEC EDGAR full-text (free, securities filings), FEC campaign finance (free key), OpenSanctions (€0.10/call, sanctions+PEPs), OpenCorporates (200M business entities), CLEAR by Thomson Reuters (enterprise background intelligence)
  - **News & Media** (new category): GDELT global news (free, real-time, 65+ languages), NewsAPI (150k-source archive)
  - **Enterprise Legal**: vLex (global case law, 100+ countries), Westlaw Edge (Thomson Reuters)
  - WEB RESEARCH card reorganized into 5 labeled collapsible subsections with clear FREE vs paid labels
  - All new keys wired into `ciGetConfig()` → sent to `WebResearcher` in Phase W
- **Entity grouping in findings** — Entities section now shows groups instead of one flat list:
  - 9 groups: People, Organizations, Law Firms, Courts, Bank Accounts, Addresses & Properties, Locations, Documents & Filings, Other
  - Each group is a `<details>` accordion showing the item count; click to expand
  - First group auto-expanded; all others collapsed — solves the 7,904-item single-list problem
- **Chronology by year** — Timeline accordion now groups events by year (newest year open, others collapsed); each year header shows event count
- **Key Findings — Judgments & Rulings surface** — Key Findings section now always surfaces:
  - **Judgments & Rulings** block: filters timeline events matching judgment/ruling/order/verdict/conviction/dismissal/settlement types and displays them prominently even when Director synthesis was not generated
  - **Financial Amounts** block: highlights events mentioning dollar amounts
  - Count label updated to reflect both AI findings + extracted rulings count
- **Authority Corpus management UI** — New "📚 AUTHORITY CORPUS" collapsible card in the CI Setup sub-tab:
  - Displays live corpus status (vector count, Cohere availability) on page load and on sub-tab switch.
  - Per-source checkboxes: NYS Senate Open Legislation (statutes), eCFR federal regulations, CourtListener opinions — all free, no keys required.
  - "⚡ Populate / Update Corpus" button triggers background ingestion via `/api/ci/authority/ingest`.
  - "↺ Refresh Status" button re-polls `/api/ci/authority/status` after ingestion completes.
  - Empty-state message in findings now includes a direct button that switches to the Setup sub-tab and scrolls to the corpus card.
- **API Key Guide modal** — every API key field in the Web Research card and Authority Corpus card now has a "🔑 get key" link that opens a floating modal:
  - Shows service name, pricing tier, and a brief description of what the source provides.
  - Direct "🔗 Open Registration Page ↗" button opens the official signup page in a new tab.
  - "🤖 Get AI Step-by-Step Guide" button calls the new `/api/ci/key-guide` endpoint, which uses the configured LLM to generate concise numbered instructions specific to that service (registration steps, where to find the key, recommended plan for legal research use).
  - Covers 18 services: Brave, Google CSE, Exa, Perplexity, Tavily, Serper, Docket Alarm, UniCourt, FEC, OpenSanctions, OpenCorporates, CLEAR, NewsAPI, LexisNexis, vLex, Westlaw, NY Senate Open Legislation, CourtListener, and Cohere.
  - Dismiss with ✕ button, Escape key, or clicking the overlay.
- **Authority corpus fixed & seeded** — Fixed broken `authority_ingester.py` endpoints (eCFR search API, Federal Register term search, CourtListener v4 citation list); successfully ingested and embedded 195 legal authorities (89 eCFR regulations, 120 Federal Register documents, 1 court opinion) into ChromaDB for semantic retrieval.

## v3.6.4 — 2026-02-28

### Fixed
- **CI theory evidence `paperless_doc_id: null`** — Theories generated by Case Intelligence were frequently emitting `null` document IDs in supporting and counter evidence items, causing the UI to show "Unspecified document" instead of a clickable document title. Root cause: the entity and timeline summaries passed to the LLM stripped out all provenance/doc-ID information. Fixed by appending `[Doc #NNN]` tags to each entity and timeline summary line in `_manager_theories()` (and the war room context path) so the LLM always has the source document ID available when generating evidence citations.
- **Theory/adversarial prompts enforce doc IDs** — Both `THEORY_GENERATION_PROMPT` and `ADVERSARIAL_TESTING_PROMPT` now include an explicit rule forbidding `null` paperless_doc_id values.

---

## v3.6.3 — 2026-02-27

### Fixed
- **Chat edit button always visible** — ✏️ Edit button is now permanently visible to the left of each user message bubble (no hover required, not hidden behind opacity:0). Works on both current-session messages and restored sessions.
- **Edit button placement** — moved outside the message bubble as a flex sibling, so it appears cleanly to the left of the blue bubble instead of inside it.
- **Edit mode visual** — the message bubble switches to a white background with gray border when in edit mode, making the textarea clearly readable instead of sitting on a blue background.
- **Compare LLMs button label** — button now reads "⚖️ Compare LLMs" (was an unlabelled ⚖️ icon) and shows "⚖️ Comparing — ON" when active, making its purpose obvious.
- **Edit button after Stop** — stopping a request now immediately attaches a ✏️ Edit button to the stopped message, so you can fix and resend without retyping the whole message.

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
