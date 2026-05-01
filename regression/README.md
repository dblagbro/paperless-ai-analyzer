# Regression Test Artifacts

Persistent home for the previously-`/tmp`-only regression scripts and logs.

## Scripts
- `full_regression_v2.py` — 712-test Playwright + API regression suite. Target:
  `http://localhost:8052/paperless-ai-analyzer-dev`. Includes the 4 stale-assertion
  fixes from 2026-04-27 (11.6, 19.1, 21.4, 26.12) and the login helper hardening
  (`domcontentloaded` + `no_wait_after=True` + 15s default timeout).
- `instance_medium.py` — ~35-43 check medium-depth smoke across all three
  instances (auth, status, system_health, projects, users/RBAC, search, LLM
  chat ping, reconcile, chat CRUD, CI, upload, court, profiles, vector,
  docs/ask, UI tab clicks).

## Latest results
- `run-v3.9.16-20260501.log` — **648 pass / 1 fail** through phase 33 of 36.
  Phase 34 (Error Handling) couldn't start its first test under host load
  (Playwright `page.fill` timed out at 15s — load-induced, not a code regression).
  The 1 failure is `33.3 Reconcile after upload → HTTP 503` (the known
  reconcile-60s optimization candidate, queue item #4).
- `run-v3.9.16-20260501.watchdog.log` — load tracker; shows host going from
  load=8 to load>30 in ~4 min once two paperless-web test stacks were
  concurrently provisioned. Watchdog spaced 30s missed the spike.

## Running

```bash
# Pre-flight: make sure load < 5 and no orphan paperless-web-pw-* containers
uptime; sudo docker ps --format '{{.Names}}' | grep '^paperless-web-pw-'

# Run from /home/dblagbro/docker/paperless-ai-analyzer/regression/
python3 -u full_regression_v2.py > run-$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

Tail the log live; watch concurrent test stacks. Two simultaneous stacks
saturate this host. v3.9.13's `deprovision_project_paperless` cleans up
stacks on `DELETE /api/projects/<slug>`, but during a run two test
projects can briefly coexist (Phase 8 + Phase 11 each provision one).
