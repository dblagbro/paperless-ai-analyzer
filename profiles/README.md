# Document Type Profiles

This directory contains YAML profiles that define how different document types should be analyzed.

## Directory Structure

- **`active/`** - Profiles that are currently used by the analyzer
- **`staging/`** - Auto-generated profile suggestions awaiting human review
- **`examples/`** - Example profiles that can be copied to `active/`

## How Profiles Work

### 1. Profile Matching

When a document is analyzed, the system:
1. Loads all profiles from `active/`
2. Scores each profile against the document using match rules
3. Selects the highest-scoring profile (if above `min_score` threshold)
4. Applies that profile's extraction and validation rules

### 2. Match Rules

Profiles use multiple matching strategies:

```yaml
match:
  keywords:
    any: ["statement period", "ending balance"]  # Match if any found
    all: ["invoice", "due date"]                 # Match only if all found

  regex:
    any: ["\\bending balance\\b", "\\bstatement\\b"]

  mime_types:
    - "application/pdf"

  min_score: 0.6  # Minimum score (0-1) required to use this profile
```

**Scoring logic:**
- Each keyword match: +0.1
- Each regex match: +0.15
- MIME type match: +0.2
- Score normalized to 0-1 range

### 3. Extraction Configuration

Controls how data is extracted from documents:

```yaml
extraction:
  engine: unstructured
  mode: elements  # or hi_res, fast

  table_hints:
    column_keywords: ["date", "description", "amount"]
    date_regex: "\\b\\d{1,2}/\\d{1,2}/\\d{4}\\b"
```

### 4. Validation Rules

Define tolerances and constraints:

```yaml
validation:
  currency: "USD"
  running_balance_tolerance: 0.01  # $0.01 tolerance
  allow_future_dates: false
```

### 5. Enabled Checks

Controls which anomaly checks run:

```yaml
checks_enabled:
  - running_balance
  - page_totals
  - continuity
  - duplicates
  - date_order
  - image_forensics
```

### 6. Tagging Rules

Defines how anomalies are tagged:

```yaml
tagging:
  deterministic_prefix: "anomaly:"
  ai_prefix: "aianomaly:"

  anomaly_tags:
    balance_mismatch: "anomaly:balance_mismatch"
    high_forensic_risk: "anomaly:forensic_risk_high"
```

## Creating a New Profile

### Option 1: Copy Example

```bash
cp examples/bank_statement_generic.yaml active/my_bank_statement.yaml
# Edit the file to customize
```

### Option 2: Let the System Suggest

When a document doesn't match any profile:
1. System tags it with `needs_profile:<slug>`
2. Generates a suggested profile in `staging/`
3. Review the suggestion
4. Move to `active/` and customize

### Example Staging Profile

```bash
# Auto-generated: staging/suggested_2024-01-30_doc_146.yaml
profile_id: suggested_doc_146
display_name: "Credit Card Statement"
version: 1

match:
  keywords:
    any: ["merchant number", "total amount processed"]
  # ... rest of suggested config
```

## Promoting Staging Profiles to Active

1. **Review the staging profile:**
   ```bash
   cat profiles/staging/suggested_*.yaml
   ```

2. **Test it (optional):**
   ```bash
   # Run analyzer with --dry-run on specific document
   docker exec paperless-ai-analyzer python -m analyzer.main --dry-run --doc-id 146
   ```

3. **Promote to active:**
   ```bash
   mv profiles/staging/suggested_doc_146.yaml profiles/active/credit_card_statement.yaml
   ```

4. **Restart analyzer:**
   ```bash
   docker compose restart paperless-ai-analyzer
   ```

## Profile Versioning

Profiles include a `version` field. When you update a profile:

1. Increment the version number
2. Update the `analyzed:deterministic:v{N}` tag in `auto_tags`
3. Documents will be re-analyzed with the new rules

## Best Practices

1. **Start Generic** - Use broad match rules, then narrow down
2. **Test Thoroughly** - Use `--dry-run` mode to test on sample documents
3. **Document Changes** - Add comments to profiles explaining customizations
4. **Version Control** - Keep profiles in git to track changes
5. **Regular Review** - Check `staging/` directory for new suggestions

## Security Note

Profiles are pure configuration (YAML). The analyzer:
- ✅ Reads profiles to configure behavior
- ❌ Never executes code from profiles
- ❌ Never writes code (only YAML configs)
- ✅ Validates all profile fields against strict schema

This ensures the "self-improving" system remains safe and auditable.

## Troubleshooting

### Profile Not Matching

Check the logs for match scores:
```bash
docker compose logs paperless-ai-analyzer | grep "Profile match scores"
```

Lower the `min_score` threshold or add more match keywords.

### False Positives

Adjust validation tolerances:
```yaml
validation:
  running_balance_tolerance: 0.05  # Increase if too strict
```

Disable specific checks:
```yaml
checks_enabled:
  - running_balance
  # - page_totals  # Commented out to disable
```

### Extraction Issues

Try different extraction modes:
```yaml
extraction:
  mode: hi_res  # More accurate but slower
  # or
  mode: fast    # Faster but less accurate
```
