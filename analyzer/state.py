"""
State Management

Tracks which documents have been analyzed and when.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Optional
from dataclasses import dataclass, asdict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerState:
    """State tracking for the analyzer."""
    last_seen_modified: Optional[str] = None  # ISO datetime
    last_seen_ids: Set[int] = None  # IDs sharing the same modified timestamp
    total_documents_processed: int = 0
    last_run: Optional[str] = None  # ISO datetime
    reprocess_all_mode: bool = False  # When True, reprocess all documents

    def __post_init__(self):
        if self.last_seen_ids is None:
            self.last_seen_ids = set()


class StateManager:
    """Manages persistent state for the analyzer."""

    def __init__(self, state_dir: str = '/app/data', project_slug: str = 'default'):
        """
        Initialize state manager for specific project.

        Args:
            state_dir: Directory for state files
            project_slug: Project identifier for state file naming (default: 'default')
        """
        self.project_slug = project_slug
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Migrate legacy state.json to state_default.json
        self._migrate_legacy_state()

        # Use project-specific state file
        self.state_path = self.state_dir / f"state_{project_slug}.json"

        self.lock = Lock()
        self.state = self._load_state()

    def _migrate_legacy_state(self):
        """
        Migrate old state.json to state_default.json for backward compatibility.
        Only runs once on first v1.5.0 startup.
        """
        legacy_path = self.state_dir / "state.json"
        new_path = self.state_dir / "state_default.json"

        if legacy_path.exists() and not new_path.exists() and self.project_slug == 'default':
            try:
                import shutil
                shutil.copy2(legacy_path, new_path)
                logger.info(f"Migrated legacy state.json to state_default.json")

                # Optionally remove legacy file (or keep as backup)
                # legacy_path.unlink()

            except Exception as e:
                logger.warning(f"Failed to migrate legacy state file: {e}")

    def _load_state(self) -> AnalyzerState:
        """Load state from disk."""
        if not self.state_path.exists():
            logger.info("No existing state file, starting fresh")
            return AnalyzerState()

        try:
            with open(self.state_path, 'r') as f:
                data = json.load(f)

            # Convert list to set for last_seen_ids
            if 'last_seen_ids' in data and isinstance(data['last_seen_ids'], list):
                data['last_seen_ids'] = set(data['last_seen_ids'])
            else:
                data['last_seen_ids'] = set()

            state = AnalyzerState(**data)
            logger.info(f"Loaded state: last_seen_modified={state.last_seen_modified}, "
                       f"processed={state.total_documents_processed}")
            return state

        except Exception as e:
            logger.error(f"Failed to load state: {e}, starting fresh")
            return AnalyzerState()

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            # Convert set to list for JSON serialization
            data = asdict(self.state)
            data['last_seen_ids'] = list(data['last_seen_ids'])

            with open(self.state_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved state to {self.state_path}")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def update_last_seen(self, modified_datetime: str, document_ids: Set[int]) -> None:
        """
        Update the last seen modified timestamp and associated document IDs.

        Args:
            modified_datetime: ISO datetime string
            document_ids: Set of document IDs with this modified time
        """
        with self.lock:
            if modified_datetime != self.state.last_seen_modified:
                # New timestamp, reset the seen IDs
                self.state.last_seen_modified = modified_datetime
                self.state.last_seen_ids = document_ids.copy()
            else:
                # Same timestamp, merge IDs
                self.state.last_seen_ids.update(document_ids)

            self._save_state()

    def should_process_document(self, doc_modified: str, doc_id: int) -> bool:
        """
        Check if a document should be processed.

        Args:
            doc_modified: Document's modified datetime (ISO string)
            doc_id: Document ID

        Returns:
            True if document should be processed
        """
        with self.lock:
            # In reprocess_all_mode, check if document has been processed this session
            if self.state.reprocess_all_mode:
                return doc_id not in self.state.last_seen_ids

            # No state yet, process everything
            if self.state.last_seen_modified is None:
                return True

            # Document modified after last seen
            if doc_modified > self.state.last_seen_modified:
                return True

            # Same modified time, check if we've seen this ID
            if doc_modified == self.state.last_seen_modified:
                return doc_id not in self.state.last_seen_ids

            # Document is older, skip
            return False

    def mark_processed(self) -> None:
        """Mark that a processing run has completed."""
        with self.lock:
            self.state.total_documents_processed += 1
            self.state.last_run = datetime.utcnow().isoformat()
            self._save_state()

    def get_stats(self) -> Dict:
        """Get current state statistics."""
        with self.lock:
            return {
                'last_seen_modified': self.state.last_seen_modified,
                'total_documents_processed': self.state.total_documents_processed,
                'last_run': self.state.last_run,
                'pending_ids_count': len(self.state.last_seen_ids)
            }

    def reset(self) -> None:
        """Reset state and enable reprocess all mode."""
        with self.lock:
            self.state = AnalyzerState(reprocess_all_mode=True)
            self._save_state()
            logger.warning("State has been reset - reprocess all mode enabled")
