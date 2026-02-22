"""
Project Manager

Manages multi-tenant projects with SQLite backend.
Each project represents an isolated workspace (case, client, matter).
"""

import sqlite3
import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)


class ProjectManager:
    """Manages projects with SQLite backend."""

    def __init__(self, db_path: str = '/app/data/projects.db'):
        """
        Initialize project manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.lock = Lock()
        self._initialize_database()
        self._ensure_default_project()

    def _initialize_database(self):
        """Create database schema if not exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    color TEXT DEFAULT '#3498db',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_analyzed_at TIMESTAMP,
                    document_count INTEGER DEFAULT 0,
                    is_archived BOOLEAN DEFAULT 0,
                    metadata TEXT
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_slug ON projects(slug)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_archived ON projects(is_archived)")

            conn.commit()
            logger.info(f"Project database initialized at {self.db_path}")

    def _ensure_default_project(self):
        """Ensure 'default' project exists for backward compatibility."""
        # No lock needed here - get_project() and create_project() handle their own locking
        default = self.get_project('default')
        if not default:
            logger.info("Creating default project for backward compatibility")
            self.create_project(
                slug='default',
                name='Default Project',
                description='Migrated from v1.0.x - all existing documents',
                color='#95a5a6',
                metadata={'migrated': True, 'version': '1.0.2'}
            )

    def create_project(self, slug: str, name: str, description: str = "",
                      color: str = None, metadata: Dict = None) -> Dict:
        """
        Create new project.

        Args:
            slug: URL-safe identifier (e.g., 'case-2024-123')
            name: Display name (e.g., 'Case 2024-123: Fairbridge')
            description: Optional description
            color: Hex color for UI (default: '#3498db')
            metadata: Additional metadata dict

        Returns:
            Dict with project details

        Raises:
            ValueError: If slug already exists or invalid format
        """
        # Validate slug format
        if not self._validate_slug(slug):
            raise ValueError(
                f"Invalid slug '{slug}'. Must be lowercase alphanumeric with dashes/underscores, "
                "2-50 characters, start and end with alphanumeric."
            )

        # Check if slug already exists
        if self.get_project(slug):
            raise ValueError(f"Project with slug '{slug}' already exists")

        # Default color if not provided
        if not color:
            color = self._generate_color()

        # Serialize metadata
        metadata_json = json.dumps(metadata) if metadata else None

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO projects (slug, name, description, color, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (slug, name, description, color, metadata_json))

                conn.commit()
                project_id = cursor.lastrowid

        logger.info(f"Created project: {slug} ({name})")

        return self.get_project(slug)

    def get_project(self, slug: str) -> Optional[Dict]:
        """
        Get project by slug.

        Args:
            slug: Project slug

        Returns:
            Project dict or None if not found
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM projects WHERE slug = ?
                """, (slug,))

                row = cursor.fetchone()
                if not row:
                    return None

                return self._row_to_dict(row)

    def list_projects(self, include_archived: bool = False) -> List[Dict]:
        """
        List all projects.

        Args:
            include_archived: Include archived projects

        Returns:
            List of project dicts
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                if include_archived:
                    cursor = conn.execute("SELECT * FROM projects ORDER BY created_at DESC")
                else:
                    cursor = conn.execute("""
                        SELECT * FROM projects
                        WHERE is_archived = 0
                        ORDER BY created_at DESC
                    """)

                return [self._row_to_dict(row) for row in cursor.fetchall()]

    def update_project(self, slug: str, **updates) -> Dict:
        """
        Update project metadata.

        Args:
            slug: Project slug
            **updates: Fields to update (name, description, color, metadata)

        Returns:
            Updated project dict

        Raises:
            ValueError: If project not found
        """
        project = self.get_project(slug)
        if not project:
            raise ValueError(f"Project '{slug}' not found")

        # Build update query dynamically
        allowed_fields = ['name', 'description', 'color', 'is_archived']
        update_fields = []
        update_values = []

        for field, value in updates.items():
            if field == 'metadata':
                # Special handling for metadata (merge, not replace)
                current_metadata = json.loads(project['metadata']) if project['metadata'] else {}
                if isinstance(value, dict):
                    current_metadata.update(value)
                update_fields.append('metadata = ?')
                update_values.append(json.dumps(current_metadata))
            elif field in allowed_fields:
                update_fields.append(f'{field} = ?')
                update_values.append(value)

        if not update_fields:
            return project

        # Add updated_at timestamp
        update_fields.append('updated_at = CURRENT_TIMESTAMP')

        update_values.append(slug)  # For WHERE clause

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                query = f"UPDATE projects SET {', '.join(update_fields)} WHERE slug = ?"
                conn.execute(query, update_values)
                conn.commit()

        logger.info(f"Updated project: {slug}")

        return self.get_project(slug)

    def delete_project(self, slug: str, delete_data: bool = True) -> bool:
        """
        Delete project.

        Args:
            slug: Project slug
            delete_data: If True, deletes vector store and state files

        Returns:
            True if successful

        Raises:
            ValueError: If trying to delete default project
        """
        if slug == 'default':
            raise ValueError("Cannot delete default project")

        project = self.get_project(slug)
        if not project:
            logger.warning(f"Project '{slug}' not found, nothing to delete")
            return False

        # Delete associated data files if requested
        if delete_data:
            try:
                # Delete state file
                state_file = Path(f'/app/data/state_{slug}.json')
                if state_file.exists():
                    state_file.unlink()
                    logger.info(f"Deleted state file: {state_file}")

                # Delete vector store collection (handled by VectorStore class)
                # Note: We don't delete the ChromaDB collection here directly
                # because it requires the VectorStore instance. The caller
                # should handle this.

            except Exception as e:
                logger.error(f"Error deleting data files for project {slug}: {e}")

        # Delete from database
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM projects WHERE slug = ?", (slug,))
                conn.commit()

        logger.info(f"Deleted project: {slug}")

        return True

    def archive_project(self, slug: str) -> Dict:
        """
        Archive project (soft delete).

        Args:
            slug: Project slug

        Returns:
            Updated project dict
        """
        return self.update_project(slug, is_archived=True)

    def unarchive_project(self, slug: str) -> Dict:
        """
        Unarchive project.

        Args:
            slug: Project slug

        Returns:
            Updated project dict
        """
        return self.update_project(slug, is_archived=False)

    def get_statistics(self, slug: str) -> Dict:
        """
        Get project statistics.

        Args:
            slug: Project slug

        Returns:
            Dict with statistics (doc_count, last_analyzed, storage_size)
        """
        project = self.get_project(slug)
        if not project:
            raise ValueError(f"Project '{slug}' not found")

        # Query live Chroma count so the UI always reflects reality
        # (the cached document_count in projects.db is only updated by explicit
        # upload paths, not by the normal analysis loop)
        try:
            from analyzer.vector_store import VectorStore
            vs = VectorStore(project_slug=slug)
            live_count = vs.collection.count()
        except Exception:
            live_count = project['document_count']  # fall back to cached

        # Sync the DB cache if it drifted
        if live_count != project['document_count']:
            self.update_document_count(slug, live_count)

        stats = {
            'document_count': live_count,
            'last_analyzed_at': project['last_analyzed_at'],
            'created_at': project['created_at'],
            'updated_at': project['updated_at']
        }

        # Calculate per-project storage size using the ChromaDB VECTOR segment directory.
        # ChromaDB stores each collection's HNSW index in a UUID-named subdirectory whose
        # UUID is the *segment* ID (not the collection ID). Look it up via the segments table.
        try:
            import sqlite3 as _sqlite3
            chroma_db = Path('/app/data/chroma/chroma.sqlite3')
            chroma_dir = Path('/app/data/chroma')
            storage_size = 0

            if chroma_db.exists():
                col_id = str(vs.collection.id)
                _conn = _sqlite3.connect(str(chroma_db))
                seg_row = _conn.execute(
                    "SELECT id FROM segments WHERE collection = ? AND scope = 'VECTOR'",
                    (col_id,)
                ).fetchone()
                _conn.close()
                if seg_row:
                    seg_dir = chroma_dir / seg_row[0]
                    if seg_dir.exists():
                        storage_size += sum(
                            f.stat().st_size for f in seg_dir.rglob('*') if f.is_file()
                        )

            # Add per-project state file if present
            state_file = Path(f'/app/data/state_{slug}.json')
            if state_file.exists():
                storage_size += state_file.stat().st_size

            stats['storage_size_bytes'] = storage_size
            stats['storage_size_mb'] = round(storage_size / (1024 * 1024), 2)

        except Exception as e:
            logger.warning(f"Could not calculate storage size for {slug}: {e}")
            stats['storage_size_bytes'] = 0
            stats['storage_size_mb'] = 0

        return stats

    def update_document_count(self, slug: str, count: int):
        """
        Update cached document count.

        Args:
            slug: Project slug
            count: Document count
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE projects
                    SET document_count = ?, last_analyzed_at = CURRENT_TIMESTAMP
                    WHERE slug = ?
                """, (count, slug))
                conn.commit()

    def increment_document_count(self, slug: str, delta: int = 1):
        """
        Increment document count.

        Args:
            slug: Project slug
            delta: Amount to increment (can be negative)
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE projects
                    SET document_count = document_count + ?,
                        last_analyzed_at = CURRENT_TIMESTAMP
                    WHERE slug = ?
                """, (delta, slug))
                conn.commit()

    def suggest_slug(self, name: str) -> str:
        """
        Generate URL-safe slug from project name.

        Args:
            name: Project name

        Returns:
            URL-safe slug
        """
        # Convert to lowercase
        slug = name.lower()

        # Replace spaces and special chars with dashes
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)

        # Trim dashes from ends
        slug = slug.strip('-')

        # Limit length
        slug = slug[:50]

        # Ensure uniqueness
        original_slug = slug
        counter = 1
        while self.get_project(slug):
            slug = f"{original_slug}-{counter}"
            counter += 1

        return slug

    def _validate_slug(self, slug: str) -> bool:
        """
        Validate slug format.

        Args:
            slug: Slug to validate

        Returns:
            True if valid
        """
        # Must be lowercase alphanumeric with dashes/underscores
        # 2-50 characters, start and end with alphanumeric
        pattern = r'^[a-z0-9][a-z0-9_-]{0,48}[a-z0-9]$'
        return bool(re.match(pattern, slug))

    def _generate_color(self) -> str:
        """
        Generate a random color for new project.

        Returns:
            Hex color string
        """
        import random
        colors = [
            '#3498db',  # Blue
            '#e74c3c',  # Red
            '#2ecc71',  # Green
            '#f39c12',  # Orange
            '#9b59b6',  # Purple
            '#1abc9c',  # Turquoise
            '#34495e',  # Dark gray
            '#e67e22',  # Carrot
            '#16a085',  # Green sea
            '#c0392b',  # Pomegranate
        ]
        return random.choice(colors)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """
        Convert SQLite row to dict.

        Args:
            row: SQLite row

        Returns:
            Dict with row data
        """
        data = dict(row)

        # Parse metadata JSON
        if data.get('metadata'):
            try:
                data['metadata'] = json.loads(data['metadata'])
            except json.JSONDecodeError:
                data['metadata'] = {}
        else:
            data['metadata'] = {}

        # Convert boolean
        data['is_archived'] = bool(data['is_archived'])

        return data
