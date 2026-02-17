#!/usr/bin/env python3
"""
Backfill Script: Reprocess Existing Poor-Quality OCR Documents with Vision AI

This script identifies documents in the vector store with poor OCR quality
and reprocesses them using Vision AI extraction, then re-indexes them.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add analyzer to path
sys.path.insert(0, '/app')

from analyzer.paperless_client import PaperlessClient
from analyzer.vector_store import VectorStore
from analyzer.main import DocumentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def identify_poor_quality_documents(vector_store: VectorStore, paperless: PaperlessClient) -> List[int]:
    """
    Identify documents in vector store that have poor OCR quality.

    Args:
        vector_store: Vector store instance
        paperless: Paperless client

    Returns:
        List of document IDs with poor quality OCR
    """
    if not vector_store.enabled:
        logger.error("Vector store not enabled")
        return []

    try:
        # Get all documents from vector store
        all_docs = vector_store.collection.get(include=['documents', 'metadatas'])

        poor_quality_doc_ids = []

        for i, doc_id_str in enumerate(all_docs['ids']):
            doc_id = int(doc_id_str)
            content = all_docs['documents'][i]
            metadata = all_docs['metadatas'][i]

            # Check for indicators of poor OCR
            is_poor = False
            reason = None

            # Check 1: Explicitly marked as no extractable text
            if "(No extractable text" in content or "(Scanned image PDF" in content:
                is_poor = True
                reason = "Marked as scanned image with no text"

            # Check 2: Very short content
            elif len(content) < 300:
                is_poor = True
                reason = f"Very short content ({len(content)} chars)"

            # Check 3: No financial data in financial documents
            elif "statement" in metadata.get('title', '').lower() or "report" in metadata.get('title', '').lower():
                import re
                has_amounts = bool(re.search(r'\$\s*[\d,]+\.\d{2}|\b[\d,]+\.\d{2}\b', content))
                if not has_amounts:
                    is_poor = True
                    reason = "Financial document but no dollar amounts"

            if is_poor:
                logger.info(f"Doc {doc_id} ({metadata.get('title', 'N/A')[:50]}): Poor OCR - {reason}")
                poor_quality_doc_ids.append(doc_id)

        logger.info(f"Found {len(poor_quality_doc_ids)} documents with poor OCR quality")
        return poor_quality_doc_ids

    except Exception as e:
        logger.error(f"Failed to identify poor quality documents: {e}", exc_info=True)
        return []


def reprocess_document_with_vision_ai(
    doc_id: int,
    analyzer: DocumentAnalyzer,
    paperless: PaperlessClient,
    vector_store: VectorStore
) -> bool:
    """
    Reprocess a single document using Vision AI and re-index in vector store.

    Args:
        doc_id: Document ID to reprocess
        analyzer: Document analyzer instance
        paperless: Paperless client
        vector_store: Vector store instance

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Reprocessing document {doc_id} with Vision AI...")

        # Get document from Paperless
        try:
            document = paperless.get_document(doc_id)
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                logger.warning(f"Document {doc_id} no longer exists in Paperless (deleted), skipping")
                return False
            raise

        if not document:
            logger.error(f"Failed to retrieve document {doc_id} from Paperless")
            return False

        doc_title = document.get('title', f'Document {doc_id}')
        logger.info(f"Document: {doc_title}")

        # Extract with Vision AI
        vision_content = analyzer.extract_with_vision_ai(document)

        if not vision_content:
            logger.error(f"Vision AI extraction failed for document {doc_id}")
            return False

        logger.info(f"Vision AI extracted {len(vision_content)} chars")

        # Prepare content for vector store
        content_parts = [
            f"Document Content (extracted via Vision AI):\n{vision_content}"
        ]

        # Add metadata
        metadata_parts = []
        if document.get('page_count'):
            metadata_parts.append(f"{document['page_count']} pages")
        if document.get('correspondent'):
            metadata_parts.append(f"Correspondent: {document['correspondent']}")
        if document.get('created'):
            metadata_parts.append(f"Created: {document['created']}")
        if metadata_parts:
            content_parts.append(f"\nDocument Metadata: {', '.join(metadata_parts)}")

        final_content = "\n\n".join(content_parts)

        # Re-index in vector store
        success = vector_store.embed_document(
            document_id=doc_id,
            title=doc_title,
            content=final_content,
            metadata={
                'risk_score': 0,
                'anomalies': [],
                'timestamp': document.get('modified', ''),
                'document_type': document.get('document_type', 'unknown')
            }
        )

        if success:
            logger.info(f"✓ Successfully reprocessed and re-indexed document {doc_id}")

            # Tag document to indicate Vision AI was used
            try:
                paperless.update_document_tags(doc_id, ['aianomaly:vision_ai_extracted'])
            except Exception as e:
                logger.warning(f"Failed to add vision_ai tag: {e}")

            return True
        else:
            logger.error(f"✗ Failed to re-index document {doc_id}")
            return False

    except Exception as e:
        logger.error(f"Failed to reprocess document {doc_id}: {e}", exc_info=True)
        return False


def main():
    """Main backfill execution."""
    logger.info("=" * 80)
    logger.info("Vision AI Backfill Script - Reprocessing Poor Quality OCR Documents")
    logger.info("=" * 80)

    # Initialize components
    config = {
        'paperless_api_base_url': os.environ.get('PAPERLESS_BASE_URL', 'http://paperless-web:8000'),
        'paperless_api_token': os.environ.get('PAPERLESS_API_TOKEN', ''),
        'llm_enabled': True,
        'llm_provider': 'anthropic',
        'llm_api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
        'llm_model': 'claude-sonnet-4-5-20250929'
    }

    logger.info("Initializing components...")
    paperless = PaperlessClient(
        base_url=config['paperless_api_base_url'],
        api_token=config['paperless_api_token']
    )
    vector_store = VectorStore()
    analyzer = DocumentAnalyzer(config)

    if not vector_store.enabled:
        logger.error("Vector store not enabled. Cannot proceed with backfill.")
        return 1

    # Step 1: Identify poor quality documents
    logger.info("\nStep 1: Identifying documents with poor OCR quality...")
    poor_quality_doc_ids = identify_poor_quality_documents(vector_store, paperless)

    if not poor_quality_doc_ids:
        logger.info("No poor quality documents found. Nothing to do.")
        return 0

    # Step 2: Reprocess each document
    logger.info(f"\nStep 2: Reprocessing {len(poor_quality_doc_ids)} documents with Vision AI...")
    logger.info("This may take several minutes depending on document count and page numbers.")
    logger.info("")

    success_count = 0
    failed_count = 0

    for idx, doc_id in enumerate(poor_quality_doc_ids, 1):
        logger.info(f"\n[{idx}/{len(poor_quality_doc_ids)}] Processing document {doc_id}...")

        if reprocess_document_with_vision_ai(doc_id, analyzer, paperless, vector_store):
            success_count += 1
        else:
            failed_count += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Backfill Complete!")
    logger.info(f"  Total processed: {len(poor_quality_doc_ids)}")
    logger.info(f"  ✓ Successful: {success_count}")
    logger.info(f"  ✗ Failed: {failed_count}")
    logger.info("=" * 80)

    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
