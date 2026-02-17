"""
URL Poller

Periodically checks URLs for new/changed documents and auto-imports them.
"""

import logging
import sqlite3
import threading
import time
import hashlib
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class URLPoller:
    """Manages periodic polling of URLs for document imports."""

    def __init__(self, db_path: str = '/app/data/url_poller.db',
                 downloader=None, smart_uploader=None):
        """
        Initialize URL poller.

        Args:
            db_path: Path to SQLite database
            downloader: RemoteFileDownloader instance
            smart_uploader: SmartUploader instance
        """
        self.db_path = Path(db_path)
        self.downloader = downloader
        self.smart_uploader = smart_uploader
        self.lock = threading.RLock()
        self.running = False
        self.thread = None

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Create tracked_urls table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracked_urls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL UNIQUE,
                    auth_type TEXT DEFAULT 'none',
                    username TEXT,
                    password TEXT,
                    token TEXT,
                    custom_headers TEXT,
                    project_slug TEXT NOT NULL,
                    poll_interval_hours INTEGER DEFAULT 24,
                    last_checked TEXT,
                    last_content_hash TEXT,
                    last_modified TEXT,
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            ''')

            # Create import_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS import_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracked_url_id INTEGER NOT NULL,
                    imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT,
                    size_bytes INTEGER,
                    content_hash TEXT,
                    paperless_doc_id INTEGER,
                    status TEXT,
                    error TEXT,
                    FOREIGN KEY (tracked_url_id) REFERENCES tracked_urls(id)
                )
            ''')

            conn.commit()
            conn.close()

            logger.info(f"URL poller database initialized at {self.db_path}")

    def add_tracked_url(self, url: str, project_slug: str,
                       auth_type: str = 'none',
                       username: str = None, password: str = None,
                       token: str = None, custom_headers: str = None,
                       poll_interval_hours: int = 24,
                       notes: str = None) -> int:
        """
        Add URL to tracking list.

        Args:
            url: URL to track
            project_slug: Project to import documents to
            auth_type: Authentication type
            username: Username for auth
            password: Password for auth
            token: Bearer/OAuth2 token
            custom_headers: JSON string of custom headers
            poll_interval_hours: Check every X hours
            notes: Optional notes

        Returns:
            ID of tracked URL
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO tracked_urls
                (url, auth_type, username, password, token, custom_headers,
                 project_slug, poll_interval_hours, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (url, auth_type, username, password, token, custom_headers,
                  project_slug, poll_interval_hours, notes))

            url_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.info(f"Added tracked URL: {url} (check every {poll_interval_hours}h)")
            return url_id

    def remove_tracked_url(self, url_id: int):
        """Remove URL from tracking."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('DELETE FROM tracked_urls WHERE id = ?', (url_id,))
            cursor.execute('DELETE FROM import_history WHERE tracked_url_id = ?', (url_id,))

            conn.commit()
            conn.close()

            logger.info(f"Removed tracked URL ID: {url_id}")

    def list_tracked_urls(self) -> List[Dict]:
        """Get all tracked URLs."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, url, auth_type, project_slug, poll_interval_hours,
                       last_checked, enabled, created_at, notes
                FROM tracked_urls
                ORDER BY created_at DESC
            ''')

            urls = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return urls

    def get_tracked_url(self, url_id: int) -> Optional[Dict]:
        """Get specific tracked URL with full details."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM tracked_urls WHERE id = ?', (url_id,))
            row = cursor.fetchone()
            conn.close()

            return dict(row) if row else None

    def update_tracked_url(self, url_id: int, **kwargs):
        """Update tracked URL settings."""
        with self.lock:
            allowed_fields = ['poll_interval_hours', 'enabled', 'notes', 'project_slug']
            updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

            if not updates:
                return

            set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
            values = list(updates.values()) + [url_id]

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(f'''
                UPDATE tracked_urls
                SET {set_clause}
                WHERE id = ?
            ''', values)

            conn.commit()
            conn.close()

            logger.info(f"Updated tracked URL ID {url_id}: {updates}")

    def get_import_history(self, url_id: int, limit: int = 50) -> List[Dict]:
        """Get import history for a tracked URL."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM import_history
                WHERE tracked_url_id = ?
                ORDER BY imported_at DESC
                LIMIT ?
            ''', (url_id, limit))

            history = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return history

    def _record_import(self, url_id: int, filename: str, size: int,
                      content_hash: str, paperless_doc_id: int = None,
                      status: str = 'success', error: str = None):
        """Record import attempt in history."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO import_history
                (tracked_url_id, filename, size_bytes, content_hash,
                 paperless_doc_id, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (url_id, filename, size, content_hash, paperless_doc_id,
                  status, error))

            conn.commit()
            conn.close()

    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def check_url(self, url_id: int) -> bool:
        """
        Check a specific URL and import if new/changed.

        Returns:
            True if document was imported
        """
        if not self.downloader or not self.smart_uploader:
            logger.warning("Downloader or SmartUploader not configured")
            return False

        url_config = self.get_tracked_url(url_id)
        if not url_config:
            logger.warning(f"Tracked URL {url_id} not found")
            return False

        if not url_config['enabled']:
            logger.debug(f"Tracked URL {url_id} is disabled, skipping")
            return False

        try:
            logger.info(f"Checking URL: {url_config['url']}")

            # Download file
            file_path, download_meta = self.downloader.download_from_url(
                url=url_config['url'],
                auth_type=url_config['auth_type'],
                username=url_config.get('username'),
                password=url_config.get('password'),
                token=url_config.get('token'),
                custom_headers=None  # Parse from JSON if needed
            )

            # Calculate content hash
            content_hash = self._calculate_hash(file_path)

            # Check if content changed
            if content_hash == url_config.get('last_content_hash'):
                logger.info(f"URL content unchanged: {url_config['url']}")
                self.downloader.cleanup(file_path)

                # Update last_checked
                with self.lock:
                    conn = sqlite3.connect(str(self.db_path))
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE tracked_urls
                        SET last_checked = ?
                        WHERE id = ?
                    ''', (datetime.utcnow().isoformat(), url_id))
                    conn.commit()
                    conn.close()

                return False

            # Analyze document
            metadata = await self.smart_uploader.analyze_document(file_path)

            # Upload to Paperless
            result = await self.smart_uploader.upload_to_paperless(
                file_path=file_path,
                project_slug=url_config['project_slug'],
                metadata=metadata
            )

            # Clean up
            self.downloader.cleanup(file_path)

            if result:
                # Record success
                self._record_import(
                    url_id=url_id,
                    filename=download_meta['filename'],
                    size=download_meta['size_bytes'],
                    content_hash=content_hash,
                    paperless_doc_id=result.get('id'),
                    status='success'
                )

                # Update tracked URL
                with self.lock:
                    conn = sqlite3.connect(str(self.db_path))
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE tracked_urls
                        SET last_checked = ?,
                            last_content_hash = ?,
                            last_modified = ?
                        WHERE id = ?
                    ''', (datetime.utcnow().isoformat(), content_hash,
                          datetime.utcnow().isoformat(), url_id))
                    conn.commit()
                    conn.close()

                logger.info(f"Imported new/changed document from: {url_config['url']}")
                return True
            else:
                raise Exception("Upload to Paperless failed")

        except Exception as e:
            logger.error(f"Failed to check URL {url_id}: {e}")

            # Record failure
            self._record_import(
                url_id=url_id,
                filename='',
                size=0,
                content_hash='',
                status='error',
                error=str(e)
            )

            return False

    def start_polling(self, check_interval_seconds: int = 300):
        """
        Start background polling thread.

        Args:
            check_interval_seconds: How often to check for URLs that need polling (default 5 min)
        """
        if self.running:
            logger.warning("URL poller already running")
            return

        self.running = True

        def polling_loop():
            logger.info("URL poller started")

            while self.running:
                try:
                    # Get URLs that need checking
                    urls_to_check = self._get_urls_due_for_check()

                    for url_config in urls_to_check:
                        if not self.running:
                            break

                        try:
                            import asyncio
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(self.check_url(url_config['id']))
                            loop.close()
                        except Exception as e:
                            logger.error(f"Error checking URL {url_config['id']}: {e}")

                except Exception as e:
                    logger.error(f"Error in polling loop: {e}")

                # Sleep
                time.sleep(check_interval_seconds)

            logger.info("URL poller stopped")

        self.thread = threading.Thread(target=polling_loop, daemon=True)
        self.thread.start()

    def stop_polling(self):
        """Stop background polling."""
        if not self.running:
            return

        logger.info("Stopping URL poller...")
        self.running = False

        if self.thread:
            self.thread.join(timeout=10)

    def _get_urls_due_for_check(self) -> List[Dict]:
        """Get URLs that are due for checking."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get URLs where:
            # - enabled = 1
            # - last_checked is NULL OR (now - last_checked) >= poll_interval_hours

            cursor.execute('''
                SELECT *
                FROM tracked_urls
                WHERE enabled = 1
                  AND (
                      last_checked IS NULL
                      OR datetime(last_checked, '+' || poll_interval_hours || ' hours') <= datetime('now')
                  )
            ''')

            urls = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return urls
