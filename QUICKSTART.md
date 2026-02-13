# Quick Start Guide

Get the Paperless AI Analyzer running in 5 minutes.

## Prerequisites

- Running Paperless-ngx instance
- Docker and Docker Compose
- Paperless API token

## Step 1: Build the Analyzer

```bash
cd /home/dblagbro/docker
docker compose build paperless-ai-analyzer
```

This will:
- Install Python dependencies
- Set up Unstructured for PDF extraction
- Install OpenCV for forensics analysis

**Note**: Initial build may take 5-10 minutes due to large dependencies.

## Step 2: Activate a Profile

The analyzer needs at least one active profile to match documents:

```bash
# Copy the generic bank statement profile
cp paperless-ai-analyzer/profiles/examples/bank_statement_generic.yaml \
   paperless-ai-analyzer/profiles/active/
```

You can add more profiles later. See `profiles/README.md` for details.

## Step 3: Start the Service

```bash
docker compose up -d paperless-ai-analyzer
```

## Step 4: Verify It's Working

Watch the logs:

```bash
docker compose logs -f paperless-ai-analyzer
```

You should see:
```
INFO - Starting polling loop (interval=30s)
INFO - Paperless API health check passed
INFO - Loaded 1 active profiles
INFO - Polling for new documents...
```

## Step 5: Access the Web UI

Open your browser to:
```
http://localhost:8051
```

You should see the Paperless AI Analyzer dashboard with:
- Real-time status
- Statistics (will be 0 initially)
- Active profiles list
- Recent analyses (empty until documents are processed)

To access via your domain (after nginx setup):
```
https://www.voipguru.org/paperless-ai-analyzer
```

See `UI_SETUP.md` for nginx configuration.

## Step 6: Run the Smoke Test

```bash
./paperless-ai-analyzer/smoke_test.sh 146
```

Replace `146` with any document ID from your Paperless instance.

The smoke test will:
1. Check dependencies
2. Verify API connectivity
3. Analyze the document (dry run)
4. Show any detected anomalies

## Step 6: Check Results

### View Tags in Paperless UI

1. Go to your Paperless UI
2. Open the document you tested
3. Look for new tags:
   - `analyzed:deterministic:v1`
   - `anomaly:*` (if anomalies were found)
   - `needs_profile:unmatched` (if no profile matched)

### Check Staging Profiles

If documents didn't match any profile, check for auto-generated suggestions:

```bash
ls -la paperless-ai-analyzer/profiles/staging/
```

Review and promote useful profiles:

```bash
# Review a suggested profile
cat paperless-ai-analyzer/profiles/staging/suggested_*.yaml

# If it looks good, promote it
mv paperless-ai-analyzer/profiles/staging/suggested_20240130_doc_147.yaml \
   paperless-ai-analyzer/profiles/active/my_custom_profile.yaml

# Restart to load new profile
docker compose restart paperless-ai-analyzer
```

## Optional: Enable AI Analysis

To enable Claude/GPT-assisted analysis:

1. Edit `docker-compose.yml`:
   ```yaml
   environment:
     LLM_ENABLED: "true"
     LLM_PROVIDER: anthropic
     LLM_API_KEY: sk-ant-your-key-here
   ```

2. Restart:
   ```bash
   docker compose up -d paperless-ai-analyzer
   ```

3. Documents will now get `analyzed:ai:v1` and `aianomaly:*` tags.

## Troubleshooting

### "No documents to process"

The analyzer only processes documents modified after it started. To reprocess existing documents:

```bash
# Reset state
docker exec paperless-ai-analyzer rm /app/data/state.json
docker compose restart paperless-ai-analyzer
```

### "No active profiles found"

Copy at least one example profile to `profiles/active/`:

```bash
cp paperless-ai-analyzer/profiles/examples/*.yaml \
   paperless-ai-analyzer/profiles/active/
docker compose restart paperless-ai-analyzer
```

### Analysis fails with "unstructured not available"

Rebuild the container:

```bash
docker compose build --no-cache paperless-ai-analyzer
docker compose up -d paperless-ai-analyzer
```

## Next Steps

- **Customize profiles**: Edit `profiles/active/*.yaml` to match your document types
- **Monitor continuously**: `docker compose logs -f paperless-ai-analyzer`
- **Adjust sensitivity**: Change `BALANCE_TOLERANCE` in docker-compose.yml
- **Review staging profiles**: Check `profiles/staging/` for auto-generated suggestions

## Common Use Cases

### Analyzing Specific Documents

```bash
# Dry run (no tags written)
docker exec paperless-ai-analyzer python -m analyzer.main --doc-id 146 --dry-run

# Actually write tags
docker exec paperless-ai-analyzer python -m analyzer.main --doc-id 146
```

### Reprocessing All Documents

```bash
# Clear state to reprocess everything
docker exec paperless-ai-analyzer rm /app/data/state.json
docker compose restart paperless-ai-analyzer

# The analyzer will now process all documents
```

### Checking What Got Analyzed

```bash
# View state
docker exec paperless-ai-analyzer cat /app/data/state.json | python3 -m json.tool

# Search logs for specific document
docker compose logs paperless-ai-analyzer | grep "document 146"
```

## Support

- README: `paperless-ai-analyzer/README.md`
- Profile docs: `paperless-ai-analyzer/profiles/README.md`
- View logs: `docker compose logs paperless-ai-analyzer`
