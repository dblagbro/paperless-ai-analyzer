# Contributing — Paperless AI Analyzer

---

## Development Workflow

### Node layout
- **Dev** (`paperless-ai-analyzer-dev`, port 8052) — all work starts here
- **Jacob** (`paperless-ai-analyzer-jacob`, port 8053) — QA/acceptance testing
- **Prod** (`paperless-ai-analyzer`, port 8051) — live, Docker Hub image

### Change flow
```
Edit code on host → build dev container → verify on dev →
test on jacob → push image to Docker Hub → pull on prod
```

### Rebuild dev after code changes
```bash
cd /home/dblagbro/docker
sudo docker compose up -d --force-recreate --no-deps paperless-ai-analyzer-dev
sudo docker logs -f paperless-ai-analyzer-dev
```

The source is volume-mounted, so for pure Python changes a container restart is enough:
```bash
sudo docker restart paperless-ai-analyzer-dev
```

### Versioning
Bump `__version__` in `analyzer/__init__.py` before any meaningful release.
Format: `MAJOR.MINOR.PATCH`. Update `CHANGELOG.md` with the new version entry.

---

## Code Style

- Python 3.10+, no type stubs required but type hints are encouraged on public functions
- 4-space indentation, 120-character line limit (soft)
- No docstrings on trivial functions; short one-line comments only where the WHY is non-obvious
- All new route handlers: `@login_required` minimum; add `@admin_required` for admin ops
- All new routes go into the appropriate Blueprint in `analyzer/routes/`
- All new cross-cutting service logic goes in `analyzer/services/`
- No business logic in route handlers — route handlers validate input, call services/models, return JSON

---

## Adding a New Route

1. Find the right Blueprint in `analyzer/routes/`
2. Add the route to that module
3. Access app dependencies via `current_app`:
   ```python
   from flask import current_app
   state = current_app.state_manager
   pm = current_app.project_manager
   ```
4. No new imports of `app` directly — always use `current_app` inside Blueprint functions
5. Register any new Blueprint in `web_ui.py` (it won't be active otherwise)

---

## Adding a New CI Analysis Component

1. Create the module in `analyzer/case_intelligence/`
2. Add its DB tables to `case_intelligence/db/schema.py` (run `init_ci_db()` which is idempotent)
3. Add corresponding DB access functions to the appropriate `case_intelligence/db/*.py` module
4. Wire the component into `orchestrator.py` at the correct phase
5. Add result API endpoint(s) to `analyzer/routes/ci.py`
6. Add UI accordion/section to `static/js/ci.js`

---

## Adding a New Court Connector

1. Subclass `BaseCourt` in `analyzer/court_connectors/base.py`
2. Add the connector to `analyzer/court_connectors/__init__.py`
3. Register the court system in `_build_court_connector()` in `analyzer/routes/court.py`

---

## Adding a New Cloud Adapter

1. Subclass `BaseCloudAdapter` in `analyzer/cloud_adapters/base.py`
2. Register in `analyzer/cloud_adapters/__init__.py`
3. Wire into the upload pipeline in `analyzer/routes/upload.py`

---

## Frontend Changes

- All CSS belongs in `static/css/dashboard.css`
- All feature JS belongs in the appropriate file in `static/js/`
- No Jinja2 template expressions in `.js` files — use `window.APP_CONFIG.*` for server-injected values
- If a new Flask variable is needed in JS, add it to the `window.APP_CONFIG` block in `dashboard.html`
- Tab structure is defined in `dashboard.html`; switching logic is in `init.js`

---

## File Size Limits

| Context | Limit |
|---------|-------|
| Python modules | 2,000 lines (target ≤1,600) |
| JavaScript files | 2,500 lines |
| HTML templates | No embedded JS or CSS; HTML-only |

If a file approaches the limit, split it before adding more features.

---

## Documentation

After any structural change:
- Update `architecture.md` if the folder layout changes
- Update `design.md` if frontend patterns change
- Append an entry to `refactor-log.md`
- Update `CHANGELOG.md` with a user-facing summary

---

## Docker Rules (Hard Limits)

- **NEVER** run `docker compose down` or any command that stops the full stack
- **NEVER** touch `docker-paperless-ai-analyzer` (prod) — only dev and jacob
- Safe single-container rebuild: `sudo docker compose up -d --force-recreate --no-deps <name>`
- `docker restart <name>` for code-only changes (no image rebuild needed)
