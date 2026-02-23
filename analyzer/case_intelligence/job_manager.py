"""
CI Job Manager — thread tracking and cancellation signals.

Each CI run gets one background thread and one cancellation Event.
Status is persisted in ci_runs (so Flask routes can poll it without
accessing the thread directly).
"""

import threading
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CIJobManager:
    """
    Tracks active CI runs as background threads with cancellation events.

    Thread lifecycle:
      start_run() → thread started, event added
      cancel_run() → event.set() signals the orchestrator to stop
      Thread completion → cleanup() removes from active dict
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}
        self._cancel_events: Dict[str, threading.Event] = {}

    def start_run(self, run_id: str, orchestrator_fn, *args, **kwargs) -> bool:
        """
        Start a CI run in a background thread.

        Args:
            run_id: The CI run UUID
            orchestrator_fn: Callable — typically CIOrchestrator.execute_run
            *args, **kwargs: Passed to orchestrator_fn

        Returns:
            True if started, False if a run with this ID is already active.
        """
        with self._lock:
            if run_id in self._threads and self._threads[run_id].is_alive():
                logger.warning(f"CI run {run_id} is already active")
                return False

            cancel_event = threading.Event()
            self._cancel_events[run_id] = cancel_event

            thread = threading.Thread(
                target=self._run_with_cleanup,
                args=(run_id, cancel_event, orchestrator_fn) + args,
                kwargs=kwargs,
                name=f"ci-run-{run_id[:8]}",
                daemon=True,
            )
            self._threads[run_id] = thread
            thread.start()
            logger.info(f"CI run {run_id} thread started")
            return True

    def _run_with_cleanup(self, run_id: str, cancel_event: threading.Event,
                           orchestrator_fn, *args, **kwargs):
        """Wrapper that cleans up after the orchestrator finishes."""
        try:
            orchestrator_fn(*args, cancel_event=cancel_event, **kwargs)
        except Exception as e:
            logger.error(f"CI run {run_id} thread crashed: {e}", exc_info=True)
            try:
                from analyzer.case_intelligence.db import update_ci_run
                update_ci_run(run_id, status='failed',
                              error_message=f"Thread crash: {str(e)[:500]}")
            except Exception:
                pass
        finally:
            with self._lock:
                self._threads.pop(run_id, None)
                self._cancel_events.pop(run_id, None)
            logger.info(f"CI run {run_id} thread cleaned up")

    def cancel_run(self, run_id: str) -> bool:
        """
        Signal a run to cancel. The orchestrator polls cancel_event.is_set()
        and stops gracefully when it sees the signal.

        Returns True if a cancel signal was sent, False if run not found.
        """
        with self._lock:
            event = self._cancel_events.get(run_id)
            if not event:
                return False
            event.set()
        logger.info(f"CI run {run_id} cancel signal sent")
        # Update DB status immediately
        try:
            from analyzer.case_intelligence.db import update_ci_run
            update_ci_run(run_id, status='cancelled')
        except Exception:
            pass
        return True

    def is_active(self, run_id: str) -> bool:
        """Return True if the run thread is alive."""
        with self._lock:
            thread = self._threads.get(run_id)
            return thread is not None and thread.is_alive()

    def get_cancel_event(self, run_id: str) -> Optional[threading.Event]:
        """Return the cancellation event for a run, or None."""
        with self._lock:
            return self._cancel_events.get(run_id)

    def list_active_runs(self):
        """Return list of run_ids that are currently active."""
        with self._lock:
            return [rid for rid, t in self._threads.items() if t.is_alive()]


# Singleton used by web_ui.py and orchestrator
_job_manager_instance: Optional[CIJobManager] = None


def get_job_manager() -> CIJobManager:
    """Return the singleton CIJobManager, creating it on first call."""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = CIJobManager()
    return _job_manager_instance
