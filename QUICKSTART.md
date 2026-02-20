# Quick Start Guide — Paperless AI Analyzer

Get up and running in about 5 minutes.

---

## Prerequisites

- Docker and Docker Compose
- A running [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) instance
- A Paperless API token (Settings → API Token in your Paperless UI)

---

## Step 1: Create a docker-compose.yml

```yaml
services:
  paperless-ai-analyzer:
    image: dblagbro/paperless-ai-analyzer:latest
    container_name: paperless-ai-analyzer
    restart: unless-stopped
    environment:
      PAPERLESS_API_BASE_URL: http://paperless-web:8000   # adjust to your Paperless URL
      PAPERLESS_API_TOKEN: your_paperless_token_here
      WEB_UI_ENABLED: "true"
      WEB_HOST: 0.0.0.0
      WEB_PORT: 8051
    volumes:
      - ./profiles:/app/profiles
      - analyzer_data:/app/data
      - /path/to/paperless/media:/paperless/media:ro   # optional — for local PDF access
    ports:
      - "8051:8051"

volumes:
  analyzer_data:
```

> **Behind a reverse proxy?** Add `URL_PREFIX: /paperless-ai-analyzer` (or whatever sub-path you use) to the `environment` section so all links and cookies are correctly scoped.

---

## Step 2: Start the container

```bash
docker compose up -d paperless-ai-analyzer
```

The first startup initializes the SQLite database, creates the Chroma vector store, and starts polling Paperless for new documents.

---

## Step 3: Create the admin user

The web UI requires authentication. Use the bundled CLI to create your first admin account:

```bash
docker exec paperless-ai-analyzer python manage_users.py create \
  --username admin \
  --password changeme \
  --role admin \
  --display-name "Admin"
```

> Change the password after first login via **Configuration → Users**.

---

## Step 4: Open the dashboard

Navigate to `http://your-host:8051` (or your reverse-proxy URL) and log in with the admin credentials you just created.

You'll land on the **Dashboard** tab showing real-time stats, recent analyses, and anomaly activity.

---

## Step 5: Configure analysis

### Connect to Paperless (verify)

The analyzer starts polling automatically. Check the **Debug & Tools** tab → **Server Logs** to confirm:

```
INFO - Paperless API health check passed
INFO - Starting polling loop (interval=30s)
```

### Enable AI Analysis (optional)

Add these to your `environment` block and restart:

```yaml
LLM_ENABLED: "true"
LLM_PROVIDER: anthropic          # or openai
LLM_API_KEY: sk-ant-your-key-here
LLM_MODEL: claude-sonnet-4-6     # or gpt-4o, etc.
```

Documents will then receive AI-generated narrative summaries and `aianomaly:*` tags in Paperless.

---

## Step 6: Upload or wait for documents

### Automatic: let it poll
The analyzer checks Paperless every 30 seconds for new or modified documents. Any document added to Paperless will be picked up automatically.

### Manual: use the Upload tab
Go to the **📤 Upload** tab to:
- Drag-and-drop a file directly
- Fetch a document from a URL (including Google Drive, Dropbox, OneDrive share links)
- Browse a directory URL and select which files to import

---

## Using the Web UI

| Tab | What it does |
|-----|--------------|
| 📊 Dashboard | Live stats, recent analyses, anomaly activity chart |
| 📋 Anomaly Detector | Browse all analyzed documents, filter by risk level |
| 💬 AI Chat | Chat with your documents using RAG (scoped to the selected project) |
| 🔍 Search | Full-text semantic search across your document vector store |
| 📤 Upload | Import documents from file, URL, or cloud links |
| 🗂️ Manage Projects | Create/archive projects, see document counts, migrate docs between projects |
| ⚙️ Configuration | API keys, SMTP, LLM model selection, vector store management, user management (admin) |
| 🛠️ Debug & Tools | Live logs, reprocess-all, reconcile index, Chroma status |

The **📖 Users Manual** button in the header opens the built-in 12-page manual at `/docs/`.

---

## User Management

```bash
# List all users
docker exec paperless-ai-analyzer python manage_users.py list

# Add a basic (non-admin) user
docker exec paperless-ai-analyzer python manage_users.py create \
  --username alice --password secret --role basic

# Reset a password
docker exec paperless-ai-analyzer python manage_users.py reset-password alice newpassword

# Deactivate a user
docker exec paperless-ai-analyzer python manage_users.py deactivate alice
```

Or manage users through **Configuration → Users** in the web UI (admin only).

---

## Reprocessing Documents

To re-analyze all documents from scratch:

1. Go to **Debug & Tools** → click **🔄 Reprocess All**
2. Or delete `/app/data/state_default.json` and restart the container

To clean up stale index records after deleting docs from Paperless:

- **Debug & Tools** → **🔁 Reconcile Now**

---

## Troubleshooting

### Container won't start
```bash
docker logs paperless-ai-analyzer
```

### "Cannot reach Paperless API"
- Verify `PAPERLESS_API_BASE_URL` is reachable from inside the container
- Check that `PAPERLESS_API_TOKEN` is correct
- If Paperless is on the same Docker network, use its service name (e.g. `http://paperless-web:8000`)

### Dashboard shows all zeros after login
- Wait ~30 seconds for the first poll cycle to complete
- Check logs for errors

### Links in emails go to the wrong URL
- Set `URL_PREFIX` to your sub-path (e.g. `/paperless-ai-analyzer`) if running behind a reverse proxy
- Set `SMTP_*` environment variables in Configuration → Notifications

---

## Next Steps

- Read the full **[README.md](README.md)** for all environment variables, profile docs, and architecture details
- Open the **📖 Users Manual** in the dashboard for tab-by-tab guidance
- Set up [SMTP notifications](README.md#smtp--notifications) to receive welcome emails and bug reports
- Create **document profiles** (`profiles/active/*.yaml`) to tune anomaly detection for your document types
