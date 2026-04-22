# Design — Paperless AI Analyzer

> Last updated: 2026-04-22

---

## Frontend Architecture

The UI is a single-page application (SPA) built in vanilla JavaScript, rendered by a
single Jinja2 template (`dashboard.html`). There is no build pipeline, no npm, no bundler.
Files are served directly as static assets by Flask.

### File Roles

| File | Responsibility |
|------|---------------|
| `templates/dashboard.html` | HTML structure only: tab shells, modal skeletons, sidebar layout |
| `static/css/dashboard.css` | All dashboard styles |
| `static/js/utils.js` | Shared helpers used by every other JS module |
| `static/js/overview.js` | Overview tab: stats, status bar, system health |
| `static/js/config.js` | Config tab: vector store, AI config, LLM settings, profiles |
| `static/js/chat.js` | Chat tab: sessions, messages, branching, compare mode |
| `static/js/upload.js` | Upload tab: file/URL/cloud/court import |
| `static/js/ci.js` | Case Intelligence tab: all CI UI |
| `static/js/users.js` | Users admin panel (admin-only) |
| `static/js/ai_form_filler.js` | AIFormFiller reusable widget (credential wizard, etc.) |
| `static/js/init.js` | DOMContentLoaded bootstrap, top-level tab switching |

### Flask → JS Variable Injection

Jinja2 variables used by JavaScript are injected through a single `window.APP_CONFIG`
object in a small inline `<script>` block at the top of `dashboard.html`:

```html
<script>
window.APP_CONFIG = {
  basePath: "{{ request.script_root }}",
  isAdmin: {{ 'true' if is_admin else 'false' }},
  isAdvanced: {{ 'true' if is_advanced else 'false' }},
  currentProject: "{{ session.get('current_project', 'default') | e }}",
  currentUserEmail: "{{ current_user.email | default('') | e }}"
};
</script>
```

All JS files reference `window.APP_CONFIG.*`. No other Jinja2 expressions appear in JS files.

### Script Load Order

Scripts must be loaded in dependency order (utils first, init last):

```html
<script src="{{ url_for('static', filename='js/utils.js') }}"></script>
<script src="{{ url_for('static', filename='js/overview.js') }}"></script>
<script src="{{ url_for('static', filename='js/config.js') }}"></script>
<script src="{{ url_for('static', filename='js/chat.js') }}"></script>
<script src="{{ url_for('static', filename='js/upload.js') }}"></script>
<script src="{{ url_for('static', filename='js/ci.js') }}"></script>
<script src="{{ url_for('static', filename='js/users.js') }}"></script>
<script src="{{ url_for('static', filename='js/ai_form_filler.js') }}"></script>
<script src="{{ url_for('static', filename='js/init.js') }}"></script>
```

### AIFormFiller Widget

`ai_form_filler.js` exports the `AIFormFiller` class — a self-contained widget that
auto-fills any HTML form from free-form pasted text via the `/api/ai-form/parse` endpoint.
To add it to a new form: instantiate with a schema, field selectors, and container.
See the court credentials wizard for a reference implementation.

---

## Tab Structure

The dashboard uses a flat tab model with one level of sub-tabs in some areas:

```
Top-level tabs:
  Overview        — system status, recent activity
  Projects        — project CRUD, smart upload, court import
  Chat            — AI chat sessions (per project)
  Case Intel      — CI pipeline setup + findings + reports
  Config          — AI config, LLM settings, vector store, profiles
  Users           — user management (admin only)
  Docs            — embedded documentation
```

Sub-tabs exist within: Config (AI config / LLM / vector store / usage), Case Intelligence
(setup / runs / findings).

Tab switching is handled by `switchTab()` in `init.js`. Sub-tab switching is feature-local
(e.g. `ciSwitchSub()` in `ci.js`, `switchConfigTab()` in `config.js`).

---

## API Conventions

### URL Structure
All API routes use `/api/` prefix. Exceptions:
- `/login`, `/logout` — auth pages
- `/health` — container health probe
- `/docs` — documentation viewer

### Response format
All JSON responses follow:
```json
{ "success": true, "data": ... }       // success
{ "error": "human-readable message" }  // failure
```

HTTP status codes are used correctly (200, 201, 400, 401, 403, 404, 500, 503).

### Auth
- All API routes require `@login_required` (Flask-Login)
- Admin routes add `@admin_required`
- CI routes add `@_ci_gate()` (checks feature flag + subscription tier)
- Court routes add `@_court_gate()`

### Project scoping
Most routes operate in the context of a project slug. The active project for the session
is stored in `flask.session['current_project']`. Routes that operate on a specific project
take `<slug>` as a URL parameter.

---

## CSS Conventions

- No CSS framework — all custom CSS in `dashboard.css`
- Color palette: neutral grays + blue accent (`#2563eb`) + red danger (`#dc2626`) + green success (`#16a34a`)
- Dark sidebar (`#1e293b`), light content area (`#f8fafc`)
- Modals use a `.modal-overlay` + `.modal` pattern; shown/hidden via `.hidden` class
- Cards use `.card` with `border-radius: 8px` and `box-shadow`
- Responsive breakpoints exist but the app is primarily desktop-targeted

---

## Error Handling Patterns

### Frontend
- `apiFetch()` in `utils.js` handles 401 → redirect to login
- `showToast(msg, type)` for user-facing notifications (success/error/info)
- Inline error spans for form validation

### Backend
All route handlers follow:
```python
try:
    ...
    return jsonify({...})
except Exception as e:
    logger.error(f"Context: {e}")
    return jsonify({'error': str(e)}), 500
```

Input validation happens at the route level before any service calls.

---

## Notification System

Outbound email notifications are sent for:
- CI run budget checkpoints (50%, 70%, 80%, 90%)
- CI run completion
- New user welcome email
- Manual re-invitation

All email logic lives in `analyzer/services/smtp_service.py`.
SMTP settings are stored in `smtp_settings.json` and configurable via the UI.
