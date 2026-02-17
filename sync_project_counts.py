#!/usr/bin/env python3
"""
Sync project document counts with actual vector store counts.

This fixes the issue where projects show 0 documents even though
documents exist in the vector store collections.
"""

import chromadb
import sqlite3
from pathlib import Path

def sync_project_counts():
    """Sync project document counts from ChromaDB collections."""

    # Initialize ChromaDB
    persist_directory = Path('/app/data/chroma')
    client = chromadb.PersistentClient(path=str(persist_directory))

    # Connect to projects database
    conn = sqlite3.connect('/app/data/projects.db')
    cursor = conn.cursor()

    # Get all project-specific collections
    collections = client.list_collections()

    print("=" * 70)
    print("  Syncing Project Document Counts")
    print("=" * 70)
    print()

    for collection in collections:
        # Skip non-project collections
        if not collection.name.startswith('paperless_docs_'):
            continue

        # Extract project slug from collection name
        project_slug = collection.name.replace('paperless_docs_', '')

        # Get document count from collection
        doc_count = collection.count()

        print(f"Collection: {collection.name}")
        print(f"  Project slug: {project_slug}")
        print(f"  Document count: {doc_count}")

        # Update project document_count in database
        cursor.execute(
            """UPDATE projects
               SET document_count = ?,
                   last_analyzed_at = CURRENT_TIMESTAMP
               WHERE slug = ?""",
            (doc_count, project_slug)
        )

        rows_affected = cursor.rowcount

        if rows_affected > 0:
            print(f"  ✓ Updated project '{project_slug}' count to {doc_count}")
        else:
            print(f"  ⚠ Project '{project_slug}' not found in database")

        print()

    conn.commit()
    conn.close()

    print("=" * 70)
    print("✓ Project counts synchronized!")
    print("=" * 70)

if __name__ == '__main__':
    sync_project_counts()
