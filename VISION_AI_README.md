# Vision AI Fallback for Poor OCR Documents

## Overview

The AI Analyzer now includes **Vision AI fallback** to handle documents where Paperless OCR fails or produces poor-quality results. This feature uses Claude 3.5 Sonnet's vision capabilities to read and extract text from scanned PDF images.

## How It Works

### Automatic Detection & Fallback

1. **Document arrives** → Paperless processes it normally with OCR
2. **AI Analyzer fetches** OCR text from Paperless
3. **Quality check** → Analyzer evaluates OCR quality:
   - Is content very short (< 200 chars for multi-page docs)?
   - Are dollar amounts missing from financial documents?
   - Is content highly repetitive (same lines repeated)?
   - Is it mostly just headers/page numbers?
4. **If OCR is poor** → Trigger Vision AI fallback:
   - Download PDF from Paperless
   - Convert pages to images (max 10 pages)
   - Use Claude Vision to extract text, tables, and financial data
   - Use Vision-extracted content instead of OCR
5. **Embed & index** → Store better content in vector store
6. **Tag document** → Add `aianomaly:vision_ai_extracted` tag

### Non-Interfering Design

✅ Paperless OCR continues unchanged
✅ No modifications to Paperless configuration
✅ Vision AI only runs when OCR is poor (cost-effective)
✅ Paperless retains its OCR for search/indexing
✅ AI Analyzer gets better data for analysis

## OCR Quality Detection Criteria

A document is flagged for Vision AI fallback if ANY of these conditions are met:

1. **Empty content**: No OCR text at all
2. **Very short content**: < 200 chars for multi-page documents
3. **Missing financial data**: Financial document (statement, invoice, report) but no dollar amounts found
4. **High repetition**: Same lines repeated 5+ times (headers only)
5. **Mostly headers**: 70%+ of lines are page headers/case numbers

## Features

### Automatic Processing (Ongoing)

All **new documents** are automatically checked. If OCR is poor, Vision AI runs automatically.

No manual intervention needed!

### Backfill Script (One-Time)

Reprocess **existing documents** with poor OCR:

```bash
# Run inside container
docker exec paperless-ai-analyzer python3 /app/backfill_vision_ai.py
```

The script will:
1. Scan all documents in vector store
2. Identify those with poor OCR quality
3. Reprocess them with Vision AI
4. Re-index with better content
5. Tag with `aianomaly:vision_ai_extracted`

## Configuration

### Required Environment Variables

```bash
# Claude API key (for Vision AI)
ANTHROPIC_API_KEY=your-api-key-here

# Paperless connection (already configured)
PAPERLESS_BASE_URL=http://paperless-web:8000
PAPERLESS_API_TOKEN=your-token
```

### Cost Management

Vision AI processes **max 10 pages per document** to control costs.

Typical costs (Claude 3.5 Sonnet):
- ~$0.015 per image (PNG from PDF page)
- 5-page document: ~$0.075
- 10-page document: ~$0.15

Only runs when OCR is actually poor, minimizing unnecessary API calls.

## Monitoring

### Check Vision AI Usage

Find documents processed with Vision AI:

```bash
# In Paperless UI, search for tag:
aianomaly:vision_ai_extracted
```

### Logs

Vision AI activity is logged:

```bash
docker logs paperless-ai-analyzer | grep "Vision AI"
```

Example log output:
```
INFO - Poor OCR quality detected for document 150, attempting Vision AI fallback
INFO - Using Vision AI fallback for document 150: doc 88 little falls
INFO - Converted 12 pages to images for Vision AI
INFO - Vision AI extracted 8432 chars from page 1
INFO - Vision AI total extraction: 45678 chars from 12 pages
INFO - Vision AI extraction successful: 45678 chars vs 1034 from Paperless OCR
```

## Examples

### Example: Court Filing (doc 150)

**Before Vision AI:**
- Paperless OCR: 1,034 chars (just headers repeated)
- Vector store: "No extractable text"
- AI Chat: Cannot answer questions about content

**After Vision AI:**
- Vision extraction: 45,000+ chars with full text
- Vector store: Complete MOR data with tables and amounts
- AI Chat: Can analyze financial figures accurately

## Troubleshooting

### Vision AI Not Running

Check:
1. Is `ANTHROPIC_API_KEY` set in environment?
2. Is LLM enabled in analyzer? (`llm_enabled: true`)
3. Check logs for errors during Vision AI extraction

### Vision AI Fails

Common issues:
- **PDF is encrypted**: Cannot convert to images
- **Images too large**: Reduce page count or resolution
- **API rate limits**: Slow down processing or use smaller batches
- **Non-text content**: Handwriting or poor quality scans may not extract well

### Backfill Script Errors

If backfill fails:
```bash
# Check logs
docker logs paperless-ai-analyzer | grep ERROR

# Run with verbose logging
docker exec paperless-ai-analyzer python3 -u /app/backfill_vision_ai.py
```

## Architecture

```
┌─────────────────┐
│  Paperless NGX  │
│  (OCR Process)  │
└────────┬────────┘
         │
         │ 1. Fetch OCR text
         ▼
┌─────────────────┐
│  AI Analyzer    │
│  Quality Check  │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Is OCR  │
    │  poor?  │
    └─┬────┬──┘
      │NO  │YES
      │    │
      │    ▼
      │  ┌──────────────────┐
      │  │  Vision AI       │
      │  │  (Claude Vision) │
      │  │  - Convert PDF   │
      │  │  - Extract text  │
      │  │  - Extract tables│
      │  └────────┬─────────┘
      │           │
      ▼           ▼
┌─────────────────────────┐
│   Vector Store          │
│   (Better Content)      │
└─────────────────────────┘
```

## Version History

- **2026-02-04**: Vision AI fallback implemented
  - OCR quality detection
  - Claude Vision integration
  - Automatic fallback in pipeline
  - Backfill script for existing documents
