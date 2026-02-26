"""
Court Import Job Manager — thread tracking and cancellation signals.

Each import job gets one background thread and one cancellation Event.
Status is persisted in court_import_jobs (Flask routes poll without
accessing the thread directly).

Mirrors the CIJobManager pattern exactly.
"""

import threading
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CourtImportJobManager:
    """
    Tracks active court import jobs as background threads with cancellation events.

    Thread lifecycle:
      start_job()  → thread started, event added
      cancel_job() → event.set() signals the import worker to stop
      Thread completion → cleanup() removes from active dict
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}
        self._cancel_events: Dict[str, threading.Event] = {}

    def start_job(self, job_id: str, import_fn, *args, **kwargs) -> bool:
        """
        Start an import job in a background daemon thread.

        Returns:
            True if started, False if already active.
        """
        with self._lock:
            if job_id in self._threads and self._threads[job_id].is_alive():
                logger.warning(f"Court import job {job_id} is already active")
                return False

            cancel_event = threading.Event()
            self._cancel_events[job_id] = cancel_event

            thread = threading.Thread(
                target=self._run_with_cleanup,
                args=(job_id, cancel_event, import_fn) + args,
                kwargs=kwargs,
                name=f"court-import-{job_id[:8]}",
                daemon=True,
            )
            self._threads[job_id] = thread
            thread.start()
            logger.info(f"Court import job {job_id} thread started")
            return True

    def _run_with_cleanup(self, job_id: str, cancel_event: threading.Event,
                          import_fn, *args, **kwargs):
        """Wrapper that cleans up after the import worker finishes."""
        try:
            # Inject job_id so the worker can log and update its own status
            import_fn(*args, job_id=job_id, cancel_event=cancel_event, **kwargs)
        except Exception as e:
            logger.error(f"Court import job {job_id} thread crashed: {e}", exc_info=True)
            try:
                from analyzer.court_db import update_import_job
                update_import_job(job_id,
                                  status='failed',
                                  error_message=f"Thread crash: {str(e)[:500]}",
                                  completed_at="datetime('now')")
            except Exception:
                pass
        finally:
            with self._lock:
                self._threads.pop(job_id, None)
                self._cancel_events.pop(job_id, None)
            logger.info(f"Court import job {job_id} thread cleaned up")

    def cancel_job(self, job_id: str) -> bool:
        """
        Signal a job to cancel. The import worker polls cancel_event.is_set().

        Returns True if signal was sent, False if job not found.
        """
        with self._lock:
            event = self._cancel_events.get(job_id)
            if not event:
                return False
            event.set()
        logger.info(f"Court import job {job_id} cancel signal sent")
        try:
            from analyzer.court_db import update_import_job
            update_import_job(job_id, status='cancelled')
        except Exception:
            pass
        return True

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(job_id)
            return thread is not None and thread.is_alive()

    def list_active_jobs(self):
        with self._lock:
            return [jid for jid, t in self._threads.items() if t.is_alive()]


# Singleton used by web_ui.py and import workers
_job_manager_instance: Optional[CourtImportJobManager] = None


def get_job_manager() -> CourtImportJobManager:
    """Return the singleton CourtImportJobManager, creating it on first call."""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = CourtImportJobManager()
    return _job_manager_instance
