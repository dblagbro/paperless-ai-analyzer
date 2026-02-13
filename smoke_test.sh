#!/bin/bash
#
# Smoke Test for Paperless AI Analyzer
#
# This script verifies that the analyzer is working correctly.
#

set -e

echo "==================================="
echo "Paperless AI Analyzer - Smoke Test"
echo "==================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check container is running
echo "Test 1: Checking if container is running..."
if docker compose ps paperless-ai-analyzer | grep -q "Up"; then
    echo -e "${GREEN}✓${NC} Container is running"
else
    echo -e "${RED}✗${NC} Container is not running"
    echo "Run: docker compose up -d paperless-ai-analyzer"
    exit 1
fi
echo

# Test 2: Check dependencies
echo "Test 2: Checking dependencies..."
if docker exec paperless-ai-analyzer python -c "from unstructured.partition.pdf import partition_pdf; from PIL import Image; import cv2; print('OK')" 2>&1 | grep -q "OK"; then
    echo -e "${GREEN}✓${NC} All dependencies available"
else
    echo -e "${YELLOW}⚠${NC} Some dependencies may be missing (check logs)"
fi
echo

# Test 3: Check Paperless API connectivity
echo "Test 3: Checking Paperless API connectivity..."
if docker exec paperless-ai-analyzer python -c "
from analyzer.paperless_client import PaperlessClient
import os
client = PaperlessClient(
    os.getenv('PAPERLESS_API_BASE_URL'),
    os.getenv('PAPERLESS_API_TOKEN')
)
if client.health_check():
    print('OK')
" 2>&1 | grep -q "OK"; then
    echo -e "${GREEN}✓${NC} Paperless API is accessible"
else
    echo -e "${RED}✗${NC} Cannot connect to Paperless API"
    exit 1
fi
echo

# Test 4: Check profiles are loaded
echo "Test 4: Checking profiles..."
PROFILE_COUNT=$(docker exec paperless-ai-analyzer ls -1 /app/profiles/active/*.yaml 2>/dev/null | wc -l)
if [ "$PROFILE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Found $PROFILE_COUNT active profile(s)"
    docker exec paperless-ai-analyzer ls -1 /app/profiles/active/ | sed 's/^/  - /'
else
    echo -e "${YELLOW}⚠${NC} No active profiles found"
    echo "  Copy example profiles: cp paperless-ai-analyzer/profiles/examples/*.yaml paperless-ai-analyzer/profiles/active/"
fi
echo

# Test 5: Check state file
echo "Test 5: Checking state persistence..."
if docker exec paperless-ai-analyzer test -f /app/data/state.json; then
    echo -e "${GREEN}✓${NC} State file exists"
    docker exec paperless-ai-analyzer cat /app/data/state.json | python3 -m json.tool 2>/dev/null | sed 's/^/  /'
else
    echo -e "${YELLOW}⚠${NC} No state file yet (will be created on first run)"
fi
echo

# Test 6: Run analysis on a document (if ID provided)
if [ ! -z "$1" ]; then
    DOC_ID=$1
    echo "Test 6: Running analysis on document $DOC_ID (dry run)..."

    # Get document title first
    DOC_TITLE=$(docker exec paperless-ai-analyzer python -c "
from analyzer.paperless_client import PaperlessClient
import os
client = PaperlessClient(os.getenv('PAPERLESS_API_BASE_URL'), os.getenv('PAPERLESS_API_TOKEN'))
doc = client.get_document($DOC_ID)
print(doc.get('title', 'Unknown'))
" 2>&1 | tail -1)

    echo "  Document: $DOC_TITLE"
    echo

    # Run analysis
    if docker exec paperless-ai-analyzer python -m analyzer.main --doc-id $DOC_ID --dry-run 2>&1 | tee /tmp/analyzer_test.log | grep -q "Successfully analyzed"; then
        echo -e "${GREEN}✓${NC} Analysis completed"

        # Check for anomalies
        if grep -q "Anomalies found:" /tmp/analyzer_test.log; then
            echo "  Anomalies detected:"
            grep "Anomalies found:" /tmp/analyzer_test.log | sed 's/^/    /'
        fi

        # Check for risk score
        if grep -q "Forensic risk score:" /tmp/analyzer_test.log; then
            grep "Forensic risk score:" /tmp/analyzer_test.log | sed 's/^/  /'
        fi

        echo
        echo -e "${GREEN}Success!${NC} The analyzer is working correctly."
        echo
        echo "Next steps:"
        echo "  1. View tags in Paperless UI (run without --dry-run to apply)"
        echo "  2. Check staging profiles: ls paperless-ai-analyzer/profiles/staging/"
        echo "  3. Monitor logs: docker compose logs -f paperless-ai-analyzer"
    else
        echo -e "${RED}✗${NC} Analysis failed"
        echo "Check logs: docker compose logs paperless-ai-analyzer"
        exit 1
    fi
else
    echo "Test 6: Skipped (provide document ID as argument)"
    echo "  Usage: ./smoke_test.sh <document_id>"
    echo "  Example: ./smoke_test.sh 146"
fi
echo

# Summary
echo "==================================="
echo "Smoke Test Complete"
echo "==================================="
echo
echo "To run a full analysis on a document:"
echo "  docker exec paperless-ai-analyzer python -m analyzer.main --doc-id <ID>"
echo
echo "To start the polling loop:"
echo "  docker compose restart paperless-ai-analyzer"
echo
