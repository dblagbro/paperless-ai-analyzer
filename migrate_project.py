#!/usr/bin/env python3
"""
Migrate documents from one project to another
Usage: docker exec paperless-ai-analyzer python3 /app/migrate_project.py
"""

import sqlite3
import sys

def create_project(name, description, color='#3498db'):
    """Create a new project if it doesn't exist."""
    conn = sqlite3.connect('/app/data/projects.db')
    cursor = conn.cursor()

    # Check if project exists
    cursor.execute('SELECT id FROM projects WHERE slug = ?', (name.lower().replace(' ', '-'),))
    existing = cursor.fetchone()

    if existing:
        print(f"✓ Project '{name}' already exists (ID: {existing[0]})")
        conn.close()
        return existing[0]

    # Create project
    slug = name.lower().replace(' ', '-')
    cursor.execute('''
        INSERT INTO projects (slug, name, description, color, created_at, updated_at)
        VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
    ''', (slug, name, description, color))

    project_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"✓ Created project '{name}' (ID: {project_id}, slug: {slug})")
    return project_id

def migrate_vector_store(from_project, to_project):
    """Migrate vector store entries."""
    from_db = f'/app/data/vector_store_{from_project}.db'
    to_db = f'/app/data/vector_store_{to_project}.db'

    try:
        # Copy the entire database file
        import shutil
        shutil.copy2(from_db, to_db)
        print(f"✓ Copied vector store from '{from_project}' to '{to_project}'")

        # Get document count
        conn = sqlite3.connect(to_db)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM documents')
        count = cursor.fetchone()[0]
        conn.close()

        print(f"  {count} documents in vector store")
        return count
    except Exception as e:
        print(f"⚠ Warning: Could not copy vector store: {e}")
        return 0

def update_paperless_tags(project_name):
    """Add project tag to all documents in Paperless."""
    print(f"\n📋 To update Paperless documents with project tag:")
    print(f"   1. Go to Paperless web UI")
    print(f"   2. Select all documents (or filter by existing tags)")
    print(f"   3. Bulk add tag: 'project:{project_name.lower().replace(' ', '-')}'")
    print(f"   OR use Paperless API to bulk update")

def main():
    print("=" * 60)
    print("  Project Migration Tool")
    print("=" * 60)

    # Create "Robinhood Fonda" project
    project_name = "Robinhood Fonda"
    project_slug = project_name.lower().replace(' ', '-')
    description = "Robinhood Fonda case documents"

    print(f"\n1. Creating project '{project_name}'...")
    project_id = create_project(project_name, description, color='#e74c3c')

    print(f"\n2. Migrating vector store from 'default' to '{project_slug}'...")
    doc_count = migrate_vector_store('default', project_slug)

    print(f"\n3. Next steps:")
    print(f"   - Vector store migrated: {doc_count} documents")
    print(f"   - New documents will need project tag: 'project:{project_slug}'")
    print(f"   - You can assign existing Paperless documents to this project via tags")

    update_paperless_tags(project_name)

    print("\n" + "=" * 60)
    print("✓ Migration setup complete!")
    print("=" * 60)

    print(f"\n💡 Next: Go to Projects tab in web UI and select '{project_name}'")

if __name__ == '__main__':
    main()
