#!/usr/bin/env python3
"""
Bulk migrate all documents from 'default' project to 'robinhood-fonda'
This script:
1. Fetches all documents from Paperless
2. Adds 'project:robinhood-fonda' tag to each
3. Triggers re-analysis for the project
"""

import os
import sys
import requests
import time

# Configuration
PAPERLESS_URL = os.environ.get('PAPERLESS_API_BASE_URL', 'http://paperless-web:8000')
PAPERLESS_TOKEN = os.environ.get('PAPERLESS_API_TOKEN', '')

if not PAPERLESS_TOKEN:
    print("ERROR: PAPERLESS_API_TOKEN not found in environment")
    sys.exit(1)

headers = {
    'Authorization': f'Token {PAPERLESS_TOKEN}',
    'Content-Type': 'application/json'
}

def get_or_create_project_tag():
    """Get or create the 'project:robinhood-fonda' tag."""
    tag_name = 'project:robinhood-fonda'

    # Search for existing tag
    response = requests.get(
        f'{PAPERLESS_URL}/api/tags/',
        headers=headers,
        params={'name': tag_name}
    )

    if response.status_code == 200:
        results = response.json()['results']
        if results:
            tag_id = results[0]['id']
            print(f"✓ Found existing tag '{tag_name}' (ID: {tag_id})")
            return tag_id

    # Create tag
    response = requests.post(
        f'{PAPERLESS_URL}/api/tags/',
        headers=headers,
        json={
            'name': tag_name,
            'color': '#e74c3c',
            'is_inbox_tag': False
        }
    )

    if response.status_code == 201:
        tag_id = response.json()['id']
        print(f"✓ Created tag '{tag_name}' (ID: {tag_id})")
        return tag_id
    else:
        print(f"ERROR: Could not create tag: {response.text}")
        sys.exit(1)

def get_all_documents():
    """Fetch all documents from Paperless."""
    all_docs = []
    page = 1

    print("\n📄 Fetching documents from Paperless...")

    while True:
        response = requests.get(
            f'{PAPERLESS_URL}/api/documents/',
            headers=headers,
            params={'page': page, 'page_size': 100}
        )

        if response.status_code != 200:
            print(f"ERROR: Could not fetch documents: {response.text}")
            sys.exit(1)

        data = response.json()
        docs = data['results']
        all_docs.extend(docs)

        print(f"  Page {page}: {len(docs)} documents (total so far: {len(all_docs)})")

        if not data['next']:
            break

        page += 1

    print(f"✓ Found {len(all_docs)} total documents")
    return all_docs

def add_tag_to_document(doc_id, tag_id, existing_tags):
    """Add project tag to a document."""
    # Add tag_id to existing tags if not already present
    if tag_id in existing_tags:
        return True  # Already has the tag

    updated_tags = existing_tags + [tag_id]

    response = requests.patch(
        f'{PAPERLESS_URL}/api/documents/{doc_id}/',
        headers=headers,
        json={'tags': updated_tags}
    )

    return response.status_code == 200

def main():
    print("=" * 70)
    print("  Bulk Document Migration: default → Robinhood Fonda")
    print("=" * 70)

    # Step 1: Get or create project tag
    print("\n1. Setting up project tag...")
    tag_id = get_or_create_project_tag()

    # Step 2: Get all documents
    print("\n2. Fetching all documents...")
    documents = get_all_documents()

    if len(documents) == 0:
        print("\n⚠ No documents found. Nothing to migrate.")
        return

    # Step 3: Add tag to each document
    print(f"\n3. Adding 'project:robinhood-fonda' tag to {len(documents)} documents...")
    print("   (This may take a few minutes...)")

    success_count = 0
    skipped_count = 0
    error_count = 0

    for i, doc in enumerate(documents, 1):
        doc_id = doc['id']
        doc_title = doc['title'][:50]  # Truncate long titles
        existing_tags = doc.get('tags', [])

        # Progress indicator every 10 documents
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(documents)} ({int(i/len(documents)*100)}%)")

        if tag_id in existing_tags:
            skipped_count += 1
            continue

        success = add_tag_to_document(doc_id, tag_id, existing_tags)

        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"   ⚠ Failed to tag document {doc_id}: {doc_title}")

        # Rate limiting: small delay to avoid overwhelming API
        time.sleep(0.1)

    print(f"\n✓ Migration complete!")
    print(f"   - Successfully tagged: {success_count} documents")
    print(f"   - Already had tag: {skipped_count} documents")
    print(f"   - Errors: {error_count} documents")

    # Step 4: Instructions for re-analysis
    print("\n4. Next steps:")
    print("   - Go to the AI Analyzer web UI (http://localhost:8051)")
    print("   - Navigate to the Projects tab")
    print("   - Select 'Robinhood Fonda' from the dropdown")
    print("   - Click '🔄 Re-analyze Current Project' button")
    print("   - This will rebuild the vector store with all documents")

    print("\n" + "=" * 70)
    print("✓ Bulk migration completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
