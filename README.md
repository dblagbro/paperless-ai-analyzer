# Paperless AI Analyzer

Advanced anomaly detection and risk analysis microservice for Paperless-ngx.

## Features

### Web Dashboard

Real-time monitoring and management interface:
- Live status updates and statistics
- Recent analysis results with risk scores
- Profile management (active and staging)
- Manual document analysis triggers
- **Access at**: `https://www.voipguru.org/paperless-ai-analyzer` (after nginx setup)

See `UI_SETUP.md` for configuration details.

### Core Capabilities

1. **Deterministic Anomaly Detection**
   - Running balance verification with configurable tolerance
   - Page totals validation
   - Cross-page continuity checks
   - Duplicate transaction detection
   - Date ordering validation

2. **AI-Assisted Anomaly Tagging** (Optional)
   - LLM-powered analysis using Claude or GPT
   - Generates narrative summaries
   - Suggests additional anomaly tags with `aianomaly:` prefix
   - Never invents data - only analyzes provided evidence

3. **Image/PDF Forensics Risk Scoring**
   - Compression artifact detection
   - Noise pattern inconsistency analysis
   - Edge anomaly detection
   - Produces 0-100% risk score (not binary)
   - Provides contributing signal details

4. **Profile-Based Document Type Matching**
   - YAML-based profile system
   - Auto-matches documents to extraction/validation rules
   - Generates staging profiles for unmatched documents
   - No code modification needed to add new document types

5. **Unstructured Integration**
   - Advanced PDF/image parsing
   - Table extraction with layout analysis
   - Multi-page document handling
   - Handles scanned and digital documents

## Architecture

```
┌─────────────────────┐
│  Paperless-ngx API  │
└──────────┬──────────┘
           │ Poll for new/modified docs
           ▼
┌─────────────────────┐
│  Profile Matcher    │──┐
└──────────┬──────────┘  │ No match
           │             ▼
           │ Match  ┌────────────────┐
           │        │ Generate       │
           ▼        │ Staging Profile│
┌─────────────────────┐ └────────────┘
│ Unstructured Extract│
└──────────┬──────────┘
           │
           ├─────────────────────────────┐
           ▼                             ▼
┌────────────────────┐      ┌─────────────────────┐
│ Deterministic      │      │ Forensics          │
│ Checks             │      │ Risk Scoring       │
│ - Balance verify   │      │ - Compression      │
│ - Duplicates       │      │ - Noise patterns   │
│ - Date order       │      │ - Edge anomalies   │
└──────────┬─────────┘      └──────────┬──────────┘
           │                           │
           └───────────┬───────────────┘
                       ▼
            ┌─────────────────────┐
            │ LLM Analysis        │
            │ (Optional)          │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │ Tag Compilation     │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │ Write Back to       │
            │ Paperless (Tags)    │
            └─────────────────────┘
```

## Quick Start

### 1. Build and Start

```bash
cd /home/dblagbro/docker

# Build the analyzer
docker compose build paperless-ai-analyzer

# Start the analyzer
docker compose up -d paperless-ai-analyzer

# Check logs
docker compose logs -f paperless-ai-analyzer
```

### 2. Verify Operation

```bash
# Check that it's running
docker compose ps paperless-ai-analyzer

# Access the web UI (direct)
open http://localhost:8051

# Test on a specific document (dry run)
docker exec paperless-ai-analyzer python -m analyzer.main --doc-id 146 --dry-run
```

### 3. Configure Profiles

```bash
# View active profiles
ls -la paperless-ai-analyzer/profiles/active/

# Copy example profiles to active
cp paperless-ai-analyzer/profiles/examples/*.yaml paperless-ai-analyzer/profiles/active/

# Restart to reload profiles
docker compose restart paperless-ai-analyzer
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPERLESS_API_BASE_URL` | `http://paperless-web:8000` | Paperless API endpoint |
| `PAPERLESS_API_TOKEN` | (required) | API authentication token |
| `POLL_INTERVAL_SECONDS` | `30` | Polling interval |
| `BALANCE_TOLERANCE` | `0.01` | Balance mismatch tolerance ($) |
| `FORENSICS_DPI` | `300` | DPI for PDF rendering |
| `LLM_ENABLED` | `false` | Enable AI-assisted analysis |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `LLM_API_KEY` | - | LLM API key (if enabled) |
| `LLM_MODEL` | (auto) | Model name (optional) |
| `ARCHIVE_PATH` | `/paperless/media/documents/archive` | Path to archived PDFs |
| `STATE_PATH` | `/app/data/state.json` | State file location |
| `PROFILES_DIR` | `/app/profiles` | Profiles directory |

### Enabling LLM Analysis

To enable AI-assisted analysis:

1. Edit `docker-compose.yml`:
   ```yaml
   environment:
     LLM_ENABLED: "true"
     LLM_PROVIDER: anthropic
     LLM_API_KEY: your_anthropic_key_here
   ```

2. Restart the service:
   ```bash
   docker compose up -d paperless-ai-analyzer
   ```

## How It Works

### Polling Strategy

The analyzer uses an efficient polling strategy:

1. Polls Paperless API every `POLL_INTERVAL_SECONDS`
2. Queries for documents with `modified > last_seen_modified`
3. Processes documents in order
4. Tracks `(modified_timestamp, document_id)` tuples to handle ties
5. Persists state to survive restarts

**Idempotency**: Re-running analysis on the same document is safe and will not create duplicate tags.

### Tag Naming Scheme

| Tag Prefix | Usage | Example |
|------------|-------|---------|
| `analyzed:deterministic:v1` | Analysis version marker | `analyzed:deterministic:v1` |
| `analyzed:ai:v1` | AI analysis version marker | `analyzed:ai:v1` |
| `anomaly:*` | Deterministic anomalies | `anomaly:balance_mismatch` |
| `aianomaly:*` | AI-suggested anomalies | `aianomaly:suspicious_pattern` |
| `needs_profile:*` | Document needs profile | `needs_profile:unmatched` |

### Profile Matching

1. Loads all profiles from `profiles/active/`
2. Scores each profile against document content using:
   - Keyword matching (0.1 per match)
   - Regex matching (0.15 per match)
   - MIME type matching (0.2)
3. Selects highest-scoring profile if above `min_score` threshold
4. If no match, generates a staging profile suggestion

### Risk Scoring

The forensics analyzer produces a 0-100% risk score based on:

- **Compression Artifacts** (max 25 points): Inconsistent compression patterns
- **Noise Inconsistency** (max 20 points): Varying noise levels across regions
- **Edge Anomalies** (max 15 points): Suspicious edge patterns near text

**Important**: This is a heuristic assessment, not definitive proof of tampering.

### Extraction Process

1. Renders PDF pages to images (if needed)
2. Uses Unstructured library to partition document
3. Identifies tables and extracts rows
4. Normalizes to canonical transaction schema:
   ```python
   {
     'date': '01/15/2024',
     'description': 'PAYMENT RECEIVED',
     'debit': None,
     'credit': 1500.00,
     'balance': 2345.67,
     'page_num': 1,
     'row_index': 5,
     'confidence': 0.85
   }
   ```

## Document Profiles

See `profiles/README.md` for detailed documentation on:

- Creating custom profiles
- Profile matching rules
- Extraction configuration
- Validation settings
- Staging profile workflow

## Smoke Test

Run the smoke test to verify everything works:

```bash
./paperless-ai-analyzer/smoke_test.sh
```

This will:
1. Check that all dependencies are installed
2. Analyze a sample document
3. Verify tags are created correctly
4. Test staging profile generation

## Monitoring

### View Logs

```bash
# Follow live logs
docker compose logs -f paperless-ai-analyzer

# View recent logs
docker compose logs --tail=100 paperless-ai-analyzer

# Search logs for specific document
docker compose logs paperless-ai-analyzer | grep "doc_id:146"
```

### Check State

```bash
# View current state
docker exec paperless-ai-analyzer cat /app/data/state.json

# Reset state (start fresh)
docker exec paperless-ai-analyzer rm /app/data/state.json
docker compose restart paperless-ai-analyzer
```

### Check Staging Profiles

```bash
# List generated staging profiles
ls -la paperless-ai-analyzer/profiles/staging/

# View a staging profile
cat paperless-ai-analyzer/profiles/staging/suggested_*.yaml
```

## Troubleshooting

### No Documents Being Processed

1. Check API connectivity:
   ```bash
   docker exec paperless-ai-analyzer curl -H "Authorization: Token f7207347bff7a7b44676f4bbb5354e64189e952d" \
     http://paperless-web:8000/api/documents/
   ```

2. Check state file:
   ```bash
   docker exec paperless-ai-analyzer cat /app/data/state.json
   ```

3. Reset state to reprocess all documents:
   ```bash
   docker exec paperless-ai-analyzer rm /app/data/state.json
   docker compose restart paperless-ai-analyzer
   ```

### Profile Not Matching

1. Check match scores in logs:
   ```bash
   docker compose logs paperless-ai-analyzer | grep "Profile match scores"
   ```

2. Lower `min_score` threshold in profile YAML

3. Add more match keywords to profile

### Extraction Issues

1. Check Unstructured is working:
   ```bash
   docker exec paperless-ai-analyzer python -c "from unstructured.partition.pdf import partition_pdf; print('OK')"
   ```

2. Try different extraction modes in profile:
   ```yaml
   extraction:
     mode: hi_res  # or 'fast', 'elements'
   ```

### LLM Not Working

1. Verify API key is set:
   ```bash
   docker exec paperless-ai-analyzer printenv | grep LLM
   ```

2. Check LLM library is installed:
   ```bash
   docker exec paperless-ai-analyzer pip list | grep anthropic
   ```

3. Test LLM directly:
   ```bash
   docker exec paperless-ai-analyzer python -c "import anthropic; print('OK')"
   ```

## Security Notes

1. **API Token**: The token is visible in logs only at startup. Never logs full tokens afterward.
2. **LLM Privacy**: Only sends extracted facts and findings to LLM, never raw document content.
3. **No Code Execution**: Profiles are pure YAML configuration, never executed as code.
4. **Read-Only Access**: Archive volume is mounted read-only.

## Performance

- **Memory**: ~500MB-1GB depending on document size
- **CPU**: Spikes during PDF rendering and forensics analysis
- **Disk**: State file < 1KB, staging profiles < 100KB total

To reduce resource usage:
- Increase `POLL_INTERVAL_SECONDS`
- Set `FORENSICS_DPI` to 150 (lower quality but faster)
- Disable LLM if not needed

## Development

### Local Testing

```bash
# Install dependencies
pip install -r paperless-ai-analyzer/requirements.txt

# Run analyzer locally
cd paperless-ai-analyzer
python -m analyzer.main --doc-id 146 --dry-run
```

### Adding New Checks

1. Edit `analyzer/checks/deterministic.py`
2. Add new check method to `DeterministicChecker` class
3. Update `check_all()` to call your new check
4. Add corresponding anomaly tag to profile YAML

### Extending Forensics

1. Edit `analyzer/forensics/risk_score.py`
2. Add new signal detection method
3. Update `_analyze_page()` to call it
4. Adjust weight calculation in `_calculate_overall_score()`

## Credits

Built with:
- [Unstructured](https://unstructured.io/) for document extraction
- [OpenCV](https://opencv.org/) for image analysis
- [Anthropic Claude](https://anthropic.com/) / [OpenAI GPT](https://openai.com/) for AI analysis

## License

MIT License - See LICENSE file for details
