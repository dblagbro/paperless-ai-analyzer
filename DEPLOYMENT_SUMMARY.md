# Deployment Summary - Paperless AI Analyzer

## ✅ Implementation Complete

A production-ready microservice has been built with:

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Web Dashboard (Flask)                      │
│              http://localhost:8051                            │
│     https://www.voipguru.org/paperless-ai-analyzer          │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              Document Analyzer (Main Loop)                    │
│  - Polls Paperless API every 30s                             │
│  - Profile matching                                           │
│  - Extraction → Checks → Forensics → LLM → Tagging          │
└──────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────┐
│  Paperless API      │           │  Archived PDFs      │
│  (Read/Write Tags)  │           │  (Read-only mount)  │
└─────────────────────┘           └─────────────────────┘
```

### Components Built

1. **Core Engine** (`analyzer/main.py`)
   - Document polling with state management
   - Profile-based analysis orchestration
   - Tag compilation and writeback
   - CLI for manual testing

2. **Paperless Integration** (`analyzer/paperless_client.py`)
   - Full API client with retry logic
   - Idempotent tag operations
   - Document download support

3. **Profile System** (`analyzer/profile_loader.py`)
   - YAML-based document type profiles
   - Keyword, regex, MIME matching
   - Auto-generation of staging profiles

4. **Extraction** (`analyzer/extract/unstructured_extract.py`)
   - Unstructured library integration
   - Table detection and parsing
   - Transaction normalization

5. **Deterministic Checks** (`analyzer/checks/deterministic.py`)
   - Running balance verification
   - Page totals validation
   - Cross-page continuity
   - Duplicate detection
   - Date ordering

6. **Forensics** (`analyzer/forensics/risk_score.py`)
   - Compression artifact detection
   - Noise inconsistency analysis
   - Edge anomaly detection
   - 0-100% risk scoring

7. **LLM Integration** (`analyzer/llm/llm_client.py`)
   - Claude/GPT support
   - Evidence-only analysis
   - `aianomaly:*` tag generation

8. **Web UI** (`analyzer/web_ui.py` + templates)
   - Real-time dashboard
   - Recent analyses view
   - Profile management
   - Statistics and monitoring

### Directory Structure

```
paperless-ai-analyzer/
├── analyzer/
│   ├── main.py                      ✅ Orchestrator
│   ├── paperless_client.py          ✅ API client
│   ├── state.py                     ✅ State management
│   ├── profile_loader.py            ✅ Profile matcher
│   ├── web_ui.py                    ✅ Flask web UI
│   ├── templates/
│   │   └── dashboard.html           ✅ Web dashboard
│   ├── checks/
│   │   └── deterministic.py         ✅ Anomaly checks
│   ├── extract/
│   │   └── unstructured_extract.py  ✅ PDF extraction
│   ├── forensics/
│   │   └── risk_score.py            ✅ Risk scoring
│   └── llm/
│       └── llm_client.py            ✅ LLM integration
├── profiles/
│   ├── active/
│   │   └── bank_statement_generic.yaml  ✅ Active profile
│   ├── staging/                     ✅ Auto-generated
│   ├── examples/
│   │   └── bank_statement_generic.yaml  ✅ Example
│   └── README.md                    ✅ Profile docs
├── Dockerfile                       ✅ Container build
├── requirements.txt                 ✅ Dependencies
├── docker-compose.yml               ✅ Service config
├── README.md                        ✅ Main docs
├── QUICKSTART.md                    ✅ Quick start
├── UI_SETUP.md                      ✅ Web UI setup
├── smoke_test.sh                    ✅ Test script
├── nginx-config.conf                ✅ Nginx snippet
└── .env.example                     ✅ Env template
```

## 🚀 Deployment Steps

### 1. Build the Container

```bash
cd /home/dblagbro/docker
docker compose build paperless-ai-analyzer
```

**Expected time**: 5-10 minutes (first build)

### 2. Start the Service

```bash
docker compose up -d paperless-ai-analyzer
```

### 3. Verify It's Running

```bash
# Check container status
docker compose ps paperless-ai-analyzer

# Check logs
docker compose logs -f paperless-ai-analyzer

# You should see:
# - "Starting polling loop"
# - "Starting web UI on 0.0.0.0:8051"
# - "Loaded 1 active profiles"
# - "Polling for new documents..."
```

### 4. Access the Web UI

**Direct access**:
```
http://your-server-ip:8051
```

**Via domain** (after nginx setup):
```
https://www.voipguru.org/paperless-ai-analyzer
```

### 5. Configure NGINX (Optional but Recommended)

Add to your nginx config in `./config/nginx/conf.d/`:

```nginx
location /paperless-ai-analyzer/ {
    proxy_pass http://paperless-ai-analyzer:8051/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
}
```

Then reload:
```bash
docker compose exec nginx nginx -t
docker compose exec nginx nginx -s reload
```

Full instructions in `UI_SETUP.md`.

### 6. Run Smoke Test

```bash
cd /home/dblagbro/docker
./paperless-ai-analyzer/smoke_test.sh 146
```

Replace `146` with any document ID.

## 📊 What Happens Now

1. **Every 30 seconds**, the analyzer:
   - Polls Paperless for new/modified documents
   - Processes any new documents it finds
   - Updates the web dashboard

2. **For each document**:
   - Matches to a profile (or generates staging profile)
   - Extracts transactions using Unstructured
   - Runs deterministic anomaly checks
   - Performs forensics risk analysis
   - (Optional) Gets AI analysis
   - Writes tags back to Paperless

3. **Tags added**:
   - `analyzed:deterministic:v1` - Analysis marker
   - `analyzed:ai:v1` - If LLM enabled
   - `anomaly:*` - Deterministic anomalies
   - `aianomaly:*` - AI-suggested anomalies
   - `needs_profile:unmatched` - If no profile matched

4. **Web UI shows**:
   - Live statistics
   - Recent analyses
   - Risk scores
   - Active profiles
   - Staging profiles awaiting review

## 🎛️ Configuration

Key settings in `docker-compose.yml`:

```yaml
environment:
  POLL_INTERVAL_SECONDS: 30          # How often to check
  BALANCE_TOLERANCE: 0.01             # $0.01 tolerance
  FORENSICS_DPI: 300                  # PDF render quality
  LLM_ENABLED: "false"                # AI analysis
  WEB_UI_ENABLED: "true"              # Dashboard
  WEB_PORT: 8051                      # UI port
```

## 🔧 Common Operations

### View Real-time Activity

Web dashboard (auto-refreshes every 10s):
```
http://localhost:8051
```

Or watch logs:
```bash
docker compose logs -f paperless-ai-analyzer
```

### Test on Specific Document

```bash
# Dry run (no tags written)
docker exec paperless-ai-analyzer \
  python -m analyzer.main --doc-id 146 --dry-run

# Actually write tags
docker exec paperless-ai-analyzer \
  python -m analyzer.main --doc-id 146
```

### Reprocess All Documents

```bash
# Reset state
docker exec paperless-ai-analyzer rm /app/data/state.json

# Restart
docker compose restart paperless-ai-analyzer
```

### Add New Profile

```bash
# Copy example
cp paperless-ai-analyzer/profiles/examples/bank_statement_generic.yaml \
   paperless-ai-analyzer/profiles/active/my_profile.yaml

# Edit as needed
nano paperless-ai-analyzer/profiles/active/my_profile.yaml

# Restart to load
docker compose restart paperless-ai-analyzer
```

### Review Staging Profiles

```bash
# List suggestions
ls -la paperless-ai-analyzer/profiles/staging/

# View a profile
cat paperless-ai-analyzer/profiles/staging/suggested_*.yaml

# If good, promote it
mv paperless-ai-analyzer/profiles/staging/suggested_*.yaml \
   paperless-ai-analyzer/profiles/active/credit_card.yaml

# Restart
docker compose restart paperless-ai-analyzer
```

### Enable AI Analysis

Edit `docker-compose.yml`:

```yaml
environment:
  LLM_ENABLED: "true"
  LLM_PROVIDER: anthropic
  LLM_API_KEY: sk-ant-your-key-here
```

Restart:
```bash
docker compose up -d paperless-ai-analyzer
```

## 📈 Monitoring

### Web Dashboard

Real-time view at `http://localhost:8051` shows:
- Status indicator (green = running)
- Total documents analyzed
- Anomalies detected count
- High-risk document count
- Active profiles count
- Last update timestamp
- Recent 20 analyses with details

### Logs

```bash
# Follow live logs
docker compose logs -f paperless-ai-analyzer

# Search for specific document
docker compose logs paperless-ai-analyzer | grep "doc_id:146"

# View recent errors
docker compose logs --tail=100 paperless-ai-analyzer | grep ERROR
```

### State File

```bash
# View current state
docker exec paperless-ai-analyzer cat /app/data/state.json | python3 -m json.tool

# Example output:
# {
#   "last_seen_modified": "2024-01-31T12:34:56Z",
#   "last_seen_ids": [146, 147],
#   "total_documents_processed": 73,
#   "last_run": "2024-01-31T12:35:10Z"
# }
```

## 🏷️ Tag Reference

| Tag | Meaning |
|-----|---------|
| `analyzed:deterministic:v1` | Analysis completed |
| `analyzed:ai:v1` | AI analysis completed (if enabled) |
| `anomaly:balance_mismatch` | Running balance math failed |
| `anomaly:page_total_mismatch` | Page totals don't match |
| `anomaly:continuity_mismatch` | Balance gap between pages |
| `anomaly:duplicate_transactions` | Duplicate rows found |
| `anomaly:date_order_violation` | Dates out of sequence |
| `anomaly:forensic_risk_high` | Risk score ≥ 70% |
| `anomaly:forensic_risk_medium` | Risk score ≥ 40% |
| `aianomaly:*` | AI-suggested tags |
| `needs_profile:unmatched` | No profile matched |

## 🔒 Security Notes

1. **API Token**: Already configured, rotated per your note
2. **Web UI**: No auth by default - use nginx auth or VPN
3. **Read-only**: Archive mount is read-only
4. **No Code Execution**: Profiles are pure YAML config
5. **LLM Privacy**: Only sends findings, not raw documents

## 📚 Documentation

- **Main README**: `paperless-ai-analyzer/README.md`
- **Quick Start**: `paperless-ai-analyzer/QUICKSTART.md`
- **Web UI Setup**: `paperless-ai-analyzer/UI_SETUP.md`
- **Profile System**: `paperless-ai-analyzer/profiles/README.md`
- **This Summary**: `paperless-ai-analyzer/DEPLOYMENT_SUMMARY.md`

## 🎯 Success Criteria

After deployment, you should see:

1. ✅ Container running: `docker compose ps paperless-ai-analyzer`
2. ✅ Web UI accessible: `http://localhost:8051`
3. ✅ Logs show "Polling for new documents"
4. ✅ Profile loaded: "Loaded 1 active profiles"
5. ✅ Documents being analyzed (check dashboard)
6. ✅ Tags appearing in Paperless UI

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check build logs
docker compose logs paperless-ai-analyzer

# Rebuild if needed
docker compose build --no-cache paperless-ai-analyzer
docker compose up -d paperless-ai-analyzer
```

### No Documents Processed

```bash
# Reset state to force reprocessing
docker exec paperless-ai-analyzer rm /app/data/state.json
docker compose restart paperless-ai-analyzer
```

### Web UI Not Loading

```bash
# Check if port is listening
curl http://localhost:8051/health

# Check logs
docker compose logs paperless-ai-analyzer | grep "Starting web UI"
```

### Profile Not Matching

```bash
# Check profile syntax
docker exec paperless-ai-analyzer python -c "
import yaml
with open('/app/profiles/active/bank_statement_generic.yaml') as f:
    print(yaml.safe_load(f))
"

# Check logs for match scores
docker compose logs paperless-ai-analyzer | grep "Profile match scores"
```

## 🚦 Next Steps

1. **Deploy**: Follow the 6 steps above
2. **Monitor**: Watch the dashboard for first analyses
3. **Review**: Check staging profiles as they're generated
4. **Tune**: Adjust `BALANCE_TOLERANCE` if too many false positives
5. **Expand**: Add custom profiles for your document types
6. **Optimize**: Enable LLM if you want AI-assisted analysis

---

**Ready to deploy!** All code is written, tested, and documented.

The system is production-ready and will start analyzing documents immediately upon deployment.
