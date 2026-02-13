# Paperless AI Analyzer

[![Docker Hub](https://img.shields.io/docker/v/dblagbro/paperless-ai-analyzer?label=Docker%20Hub)](https://hub.docker.com/r/dblagbro/paperless-ai-analyzer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Advanced AI-powered anomaly detection and risk analysis microservice for [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx). Automatically analyzes financial documents, bank statements, and invoices for inconsistencies, tampering, and anomalies using deterministic checks, image forensics, and optional LLM-powered analysis.

## 🌟 Key Features

- **🔍 Deterministic Anomaly Detection**: Balance verification, duplicate detection, date validation
- **🖼️ Image Forensics**: PDF tampering detection with risk scoring (0-100%)
- **🤖 AI-Assisted Analysis**: Optional Claude/GPT integration for narrative summaries
- **📊 Profile-Based Processing**: YAML-configured document type matching
- **🎯 Automated Tagging**: Adds structured tags to Paperless documents
- **📈 Web Dashboard**: Real-time monitoring and manual triggers
- **🔄 Idempotent Processing**: Safe to re-run on same documents
- **🚀 Plug-and-Play**: Integrates seamlessly with existing Paperless-ngx installations

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Use Cases](#use-cases)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Document Profiles](#document-profiles)
- [How It Works](#how-it-works)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Performance](#performance)
- [Security](#security-notes)
- [FAQ](#faq)
- [License](#license)

## Features

### Web Dashboard

Real-time monitoring and management interface:
- **Live Status**: Current processing state and statistics
- **Recent Analysis**: View documents with risk scores and anomalies
- **Profile Management**: Manage active and staging profiles
- **Manual Triggers**: Force analysis on specific documents
- **Vector Store**: Document embeddings and similarity search
- **Logs Viewer**: Real-time log streaming

**Access**: `http://localhost:8051` (or via reverse proxy at `/paperless-ai-analyzer`)

See [UI_SETUP.md](UI_SETUP.md) for nginx configuration.

### Core Capabilities

#### 1. 🔍 Deterministic Anomaly Detection

Catches common document issues without AI:

- **Running Balance Verification**: Ensures debits/credits match balance changes
- **Page Totals Validation**: Verifies page-level subtotals
- **Cross-Page Continuity**: Checks balance carries forward correctly
- **Duplicate Transaction Detection**: Identifies repeated entries
- **Date Ordering Validation**: Ensures chronological order
- **Configurable Tolerance**: Customize thresholds per document type

**Tags Generated**: `anomaly:balance_mismatch`, `anomaly:duplicate_transaction`, etc.

#### 2. 🖼️ Image/PDF Forensics Risk Scoring

Advanced visual analysis to detect tampering:

- **Compression Artifact Detection**: Identifies inconsistent JPEG compression
- **Noise Pattern Analysis**: Detects varying noise levels across regions
- **Edge Anomaly Detection**: Finds suspicious edges near text
- **Produces Risk Score**: 0-100% with detailed signal breakdown
- **Per-Page Analysis**: Identifies which pages are suspicious

**Output**: Numeric risk score + contributing factors for transparency

**Tags Generated**: `forensics:risk_score:high` (if score > 60%)

#### 3. 🤖 AI-Assisted Anomaly Tagging (Optional)

Uses Claude or GPT to provide narrative insights:

- **Evidence-Based Analysis**: Only analyzes extracted facts
- **Narrative Summaries**: Human-readable explanations
- **Additional Anomaly Suggestions**: With `aianomaly:` prefix
- **Never Invents Data**: Strictly analyzes provided evidence
- **Custom Instructions**: Per-profile AI prompts

**Requirements**: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment variable

**Tags Generated**: `aianomaly:suspicious_pattern`, `analyzed:ai:v1`

#### 4. 📊 Profile-Based Document Type Matching

Flexible YAML configuration system:

- **Auto-Matching**: Scores profiles based on keywords, regex, MIME type
- **No Code Changes Needed**: Add document types via YAML files
- **Staging Workflow**: Auto-generates profile suggestions for unmatched docs
- **Extraction Rules**: Customize how tables and transactions are parsed
- **Validation Settings**: Configure balance tolerance, date formats, etc.

**Example Use Cases**:
- Bank statements (Chase, Bank of America, Wells Fargo)
- Credit card statements
- Utility bills
- Invoices
- Tax documents

#### 5. 📄 Unstructured Integration

Powered by [Unstructured](https://unstructured.io/):

- **Advanced PDF Parsing**: Layout-aware table extraction
- **Multi-Page Documents**: Handles statements with 10+ pages
- **Scanned Documents**: Works with both digital and scanned PDFs
- **Table Detection**: Identifies transaction tables automatically
- **OCR Support**: Via tesseract for image-based PDFs

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

## Use Cases

### 💼 Financial Auditing
- **Scenario**: Reviewing bank statements for expense reports
- **Benefit**: Automatically detect balance errors, duplicate transactions, and suspicious modifications
- **Tags**: `anomaly:balance_mismatch`, `forensics:risk_score:high`

### 🏢 Accounts Payable/Receivable
- **Scenario**: Processing invoices from multiple vendors
- **Benefit**: Validate totals, detect duplicate invoices, flag unusual patterns
- **Tags**: `anomaly:duplicate_transaction`, `aianomaly:unusual_amount`

### 🔒 Compliance & Fraud Detection
- **Scenario**: Document retention with fraud risk assessment
- **Benefit**: Image forensics identifies potentially tampered PDFs
- **Tags**: `forensics:risk_score:high`, `needs_review`

### 📊 Document Classification
- **Scenario**: Large document backlog from scanning project
- **Benefit**: Auto-generate profiles for unknown document types
- **Tags**: `needs_profile:unmatched`, staging profiles created

### 🤝 Vendor Statement Reconciliation
- **Scenario**: Monthly statements from 50+ vendors
- **Benefit**: Automated validation with custom profiles per vendor
- **Tags**: `analyzed:deterministic:v1`, profile-specific anomalies

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Existing Paperless-ngx installation
- Paperless API token ([generate here](https://docs.paperless-ngx.com/api/))

### Installation

#### Using Docker Hub

```yaml
services:
  paperless-ai-analyzer:
    image: dblagbro/paperless-ai-analyzer:latest
    container_name: paperless-ai-analyzer
    restart: unless-stopped
    environment:
      PAPERLESS_API_BASE_URL: http://paperless-web:8000
      PAPERLESS_API_TOKEN: your_token_here
      POLL_INTERVAL_SECONDS: 30
      LLM_ENABLED: "false"  # Set to "true" for AI analysis
    volumes:
      - ./paperless-ai-analyzer/profiles:/app/profiles
      - ./paperless-ai-analyzer/data:/app/data
      - /path/to/paperless/media:/paperless/media:ro
    ports:
      - "8051:8051"
```

#### From Source

```bash
git clone https://github.com/dblagbro/paperless-ai-analyzer.git
cd paperless-ai-analyzer
docker build -t paperless-ai-analyzer .
```

### Initial Setup

1. **Start the service**:
   ```bash
   docker compose up -d paperless-ai-analyzer
   ```

2. **Verify it's running**:
   ```bash
   docker compose logs -f paperless-ai-analyzer
   ```

3. **Access the web UI**:
   ```
   http://localhost:8051
   ```

4. **Add example profiles**:
   ```bash
   cp profiles/examples/*.yaml profiles/active/
   docker compose restart paperless-ai-analyzer
   ```

5. **Test on a document**:
   ```bash
   docker exec paperless-ai-analyzer python -m analyzer.main --doc-id 123 --dry-run
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPERLESS_API_BASE_URL` | `http://paperless-web:8000` | Paperless API endpoint |
| `PAPERLESS_API_TOKEN` | *(required)* | API authentication token |
| `POLL_INTERVAL_SECONDS` | `30` | Polling interval |
| `BALANCE_TOLERANCE` | `0.01` | Balance mismatch tolerance ($) |
| `FORENSICS_DPI` | `300` | DPI for PDF rendering (150-600) |
| `LLM_ENABLED` | `false` | Enable AI-assisted analysis |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `LLM_API_KEY` | - | LLM API key (required if LLM enabled) |
| `LLM_MODEL` | *(auto)* | Override model (e.g., `claude-3-sonnet-20240229`) |
| `ARCHIVE_PATH` | `/paperless/media/documents/archive` | Path to archived PDFs |
| `STATE_PATH` | `/app/data/state.json` | State file location |
| `PROFILES_DIR` | `/app/profiles` | Profiles directory |

### Enabling LLM Analysis

Add to your docker-compose environment:

```yaml
environment:
  LLM_ENABLED: "true"
  LLM_PROVIDER: anthropic
  LLM_API_KEY: sk-ant-api03-xxx  # Your Anthropic API key
```

Or for OpenAI:

```yaml
environment:
  LLM_ENABLED: "true"
  LLM_PROVIDER: openai
  LLM_API_KEY: sk-proj-xxx  # Your OpenAI API key
  LLM_MODEL: gpt-4-turbo-preview
```

Then restart:
```bash
docker compose up -d paperless-ai-analyzer
```

## 📄 Document Profiles

Profiles define how to process specific document types. See [profiles/README.md](profiles/README.md) for detailed documentation.

### Profile Structure

```yaml
name: Chase Bank Statement
description: Personal checking account statements from Chase Bank
min_score: 0.4

matching:
  keywords:
    - CHASE
    - Checking Account
    - Account Number
  patterns:
    - "Statement Period: \\d{2}/\\d{2}/\\d{4}"
  mime_types:
    - application/pdf

extraction:
  mode: hi_res
  table_min_rows: 3
  date_formats:
    - "%m/%d/%Y"
    - "%m/%d/%y"

validation:
  check_balance: true
  balance_tolerance: 0.01
  check_duplicates: true
  check_date_order: true

forensics:
  enabled: true
  min_risk_threshold: 40

ai_prompt: |
  Analyze this bank statement for unusual patterns.
  Focus on large transactions and suspicious activity.
```

### Creating Custom Profiles

1. Copy example: `cp profiles/examples/bank_statement_generic.yaml profiles/active/my_bank.yaml`
2. Edit matching criteria (keywords, patterns)
3. Adjust validation rules
4. Test: `docker exec paperless-ai-analyzer python -m analyzer.main --doc-id 123`
5. Restart service to reload profiles

### Staging Workflow

When a document doesn't match any profile:

1. Analyzer generates a staging profile in `profiles/staging/`
2. Profile includes suggested keywords from document content
3. Review and move to `profiles/active/` after customization
4. Restart service to activate

## 🔍 How It Works

### Polling Strategy

1. Polls Paperless API every `POLL_INTERVAL_SECONDS`
2. Queries for documents with `modified > last_seen_modified`
3. Processes documents in chronological order
4. Tracks `(modified_timestamp, document_id)` tuples to handle ties
5. Persists state to survive restarts

**Idempotency**: Re-running analysis on the same document is safe and won't create duplicate tags.

### Tag Naming Scheme

| Tag Prefix | Usage | Example |
|------------|-------|---------|
| `analyzed:deterministic:v1` | Analysis version marker | `analyzed:deterministic:v1` |
| `analyzed:ai:v1` | AI analysis version marker | `analyzed:ai:v1` |
| `anomaly:*` | Deterministic anomalies | `anomaly:balance_mismatch` |
| `aianomaly:*` | AI-suggested anomalies | `aianomaly:suspicious_pattern` |
| `forensics:risk_score:*` | Risk score categories | `forensics:risk_score:high` |
| `needs_profile:*` | Document needs profile | `needs_profile:unmatched` |

### Profile Matching Algorithm

1. Loads all profiles from `profiles/active/`
2. Scores each profile against document:
   - Keyword match: +0.1 per keyword
   - Regex match: +0.15 per pattern
   - MIME type match: +0.2
3. Selects highest-scoring profile if score ≥ `min_score`
4. If no match, generates staging profile suggestion

### Risk Scoring Components

The forensics analyzer produces a 0-100% risk score:

- **Compression Artifacts** (max 25 points): Inconsistent JPEG compression patterns
- **Noise Inconsistency** (max 20 points): Varying noise levels across regions
- **Edge Anomalies** (max 15 points): Suspicious edge patterns near text

**Interpretation**:
- **0-30%**: Low risk, likely unmodified
- **30-60%**: Medium risk, review recommended
- **60-100%**: High risk, likely modified or poor quality scan

**Important**: This is a heuristic assessment, not definitive proof of tampering.

### Extraction Process

1. Renders PDF pages to images at configured DPI
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

## 📊 Monitoring

### Web Dashboard

Access `http://localhost:8051` to view:
- System status and statistics
- Recent analysis results
- Active and staging profiles
- Real-time logs
- Vector store documents

### Command Line

```bash
# Follow live logs
docker compose logs -f paperless-ai-analyzer

# View recent logs
docker compose logs --tail=100 paperless-ai-analyzer

# Search logs for specific document
docker compose logs paperless-ai-analyzer | grep "doc_id:146"

# View current state
docker exec paperless-ai-analyzer cat /app/data/state.json

# List staging profiles
ls -la profiles/staging/
```

### Health Check

```bash
# Check container is running
docker compose ps paperless-ai-analyzer

# Test API connectivity
docker exec paperless-ai-analyzer curl -H "Authorization: Token YOUR_TOKEN" \
  http://paperless-web:8000/api/documents/?page_size=1
```

## 🔧 Troubleshooting

### No Documents Being Processed

**Symptoms**: No logs showing document analysis

**Solutions**:
1. Verify API connectivity:
   ```bash
   docker exec paperless-ai-analyzer curl -H "Authorization: Token YOUR_TOKEN" \
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

**Symptoms**: Documents tagged with `needs_profile:unmatched`

**Solutions**:
1. Check match scores in logs:
   ```bash
   docker compose logs paperless-ai-analyzer | grep "Profile match scores"
   ```

2. Lower `min_score` threshold in profile YAML

3. Add more keywords to profile matching section

4. Check staging profiles for suggestions:
   ```bash
   ls -la profiles/staging/
   ```

### Extraction Issues

**Symptoms**: No transactions extracted, empty table data

**Solutions**:
1. Verify Unstructured is working:
   ```bash
   docker exec paperless-ai-analyzer python -c \
     "from unstructured.partition.pdf import partition_pdf; print('OK')"
   ```

2. Try different extraction modes:
   ```yaml
   extraction:
     mode: hi_res  # Options: hi_res, fast, elements
   ```

3. Check PDF rendering:
   ```bash
   docker exec paperless-ai-analyzer python -c \
     "import pdf2image; print('OK')"
   ```

### LLM Not Working

**Symptoms**: No `aianomaly:*` tags, AI-related errors in logs

**Solutions**:
1. Verify LLM is enabled:
   ```bash
   docker exec paperless-ai-analyzer printenv | grep LLM
   ```

2. Check API key is valid:
   - Anthropic: Test at https://console.anthropic.com
   - OpenAI: Test at https://platform.openai.com

3. Test LLM library:
   ```bash
   docker exec paperless-ai-analyzer python -c \
     "import anthropic; print('OK')"
   ```

4. Check for rate limits in logs

### High Memory Usage

**Symptoms**: Container using > 2GB RAM

**Solutions**:
1. Lower forensics DPI:
   ```yaml
   environment:
     FORENSICS_DPI: "150"  # Lower quality but faster
   ```

2. Increase poll interval:
   ```yaml
   environment:
     POLL_INTERVAL_SECONDS: "60"
   ```

3. Disable LLM if not needed:
   ```yaml
   environment:
     LLM_ENABLED: "false"
   ```

## 💻 Development

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run analyzer locally
cd analyzer
python main.py --doc-id 146 --dry-run

# Run specific checks
python -m checks.deterministic
python -m forensics.risk_score
```

### Running Tests

```bash
# Run smoke test
./smoke_test.sh

# Run unit tests (if available)
python -m pytest tests/

# Lint code
python -m pylint analyzer/
```

### Adding New Checks

1. Edit `analyzer/checks/deterministic.py`
2. Add new check method to `DeterministicChecker` class:
   ```python
   def check_my_new_validation(self, transactions):
       """Check for my new anomaly."""
       anomalies = []
       # Your logic here
       return anomalies
   ```
3. Update `check_all()` to call your new check
4. Add corresponding tag handling in main analyzer
5. Test thoroughly

### Extending Forensics

1. Edit `analyzer/forensics/risk_score.py`
2. Add new signal detection method:
   ```python
   def _detect_my_signal(self, image):
       """Detect my new tampering signal."""
       score = 0.0
       details = []
       # Your analysis here
       return score, details
   ```
3. Update `_analyze_page()` to call it
4. Adjust weight in `_calculate_overall_score()`
5. Test on known good/bad samples

## 📈 Performance

### Resource Usage

- **Memory**: 500MB-2GB depending on document size and DPI
- **CPU**: Spikes during PDF rendering and forensics (10-30 seconds per document)
- **Disk**:
  - State file: < 1KB
  - Staging profiles: < 100KB total
  - Logs: Grows over time (rotate recommended)

### Optimization Tips

**For High Volume**:
```yaml
environment:
  POLL_INTERVAL_SECONDS: "60"  # Reduce polling frequency
  FORENSICS_DPI: "150"         # Lower DPI for faster processing
  LLM_ENABLED: "false"         # Disable if not needed
```

**For Quality**:
```yaml
environment:
  FORENSICS_DPI: "600"         # Higher DPI for better analysis
  LLM_ENABLED: "true"          # Enable AI insights
```

### Benchmarks

Typical processing times (Intel i7, 16GB RAM):

| Document Type | Pages | Processing Time |
|---------------|-------|-----------------|
| Bank Statement | 2 | 8-12 seconds |
| Credit Card | 5 | 20-30 seconds |
| Tax Form | 10 | 45-60 seconds |
| Invoice | 1 | 4-6 seconds |

*With LLM enabled, add 2-5 seconds per document*

## 🔒 Security Notes

1. **API Token**: Token is visible in logs only at startup. Mask it or use Docker secrets for production.

2. **LLM Privacy**: Only extracted facts and findings are sent to LLM APIs. Raw document content never leaves your infrastructure.

3. **No Code Execution**: Profiles are pure YAML configuration. No code is executed from profiles.

4. **Read-Only Access**: Archive volume is mounted read-only (`:ro`) to prevent accidental modifications.

5. **Network Isolation**: Recommend running in same Docker network as Paperless for security.

6. **Audit Trail**: All analysis actions are logged with timestamps and document IDs.

## ❓ FAQ

**Q: Does this modify my original documents?**
A: No. It only adds tags to Paperless. Original PDFs are never modified.

**Q: Can I reprocess documents?**
A: Yes. Delete `state.json` and restart. The analyzer is idempotent.

**Q: What happens if I don't have an LLM API key?**
A: The analyzer works fine without it. You'll get deterministic checks and forensics, just not AI narrative summaries.

**Q: How accurate is the forensics risk scoring?**
A: It's a heuristic tool, not forensic proof. Use it as an additional signal, not sole evidence.

**Q: Can I customize what tags are created?**
A: Yes, through profiles. You can also modify the core code to change tag naming.

**Q: Does this work with scanned documents?**
A: Yes, it uses tesseract OCR via Unstructured. Quality depends on scan quality.

**Q: How do I update profiles without restarting?**
A: Currently requires restart. We may add hot-reload in the future.

**Q: Can I analyze documents from other sources?**
A: It's designed for Paperless-ngx, but you could adapt the extraction/analysis code for other uses.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Credits

Built with:
- [Unstructured](https://unstructured.io/) for document extraction
- [OpenCV](https://opencv.org/) for image analysis
- [Anthropic Claude](https://anthropic.com/) / [OpenAI GPT](https://openai.com/) for optional AI analysis
- [Flask](https://flask.palletsprojects.com/) for web UI
- [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) for document management

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/dblagbro/paperless-ai-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dblagbro/paperless-ai-analyzer/discussions)
- **Documentation**: See [QUICKSTART.md](QUICKSTART.md), [UI_SETUP.md](UI_SETUP.md), [VISION_AI_README.md](VISION_AI_README.md)

---

**Built for professionals who need automated document analysis without compromising security or privacy.**
