"""
Paperless AI Analyzer - Main Entry Point

Orchestrates document analysis pipeline.
"""

import os
import sys
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from analyzer.paperless_client import PaperlessClient
from analyzer.state import StateManager
from analyzer.profile_loader import ProfileLoader
from analyzer.project_manager import ProjectManager  # v1.5.0
from analyzer.smart_upload import SmartUploader  # v1.5.0
# NOTE: Deterministic checks and forensics handled by paperless-anomaly-detector
# from analyzer.extract.unstructured_extract import UnstructuredExtractor
# from analyzer.checks.deterministic import DeterministicChecker
# from analyzer.forensics.risk_score import ForensicsAnalyzer
from analyzer.vector_store import VectorStore
from analyzer.llm.llm_client import LLMClient
from analyzer.llm_usage_tracker import LLMUsageTracker
from analyzer.web_ui import start_web_server_thread, update_ui_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


from analyzer.poller import PollerMixin, _poll_project_loop  # noqa: F401  (re-exported for callers that imported from main)
from analyzer.document_processor import DocumentProcessorMixin

class DocumentAnalyzer(PollerMixin, DocumentProcessorMixin):
    """Main document analysis service.

    Implementation split across mixin files for maintainability:
      - PollerMixin             (analyzer/poller.py)
      - DocumentProcessorMixin  (analyzer/document_processor.py)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize components
        self.paperless = PaperlessClient(
            base_url=config['paperless_api_base_url'],
            api_token=config['paperless_api_token']
        )

        # Use default project for now (full multi-tenancy in later phase)
        project_slug = config.get('project_slug', 'default')
        self.state_manager = StateManager(
            state_dir=config.get('state_dir', '/app/data'),
            project_slug=project_slug
        )

        self.profile_loader = ProfileLoader(
            profiles_dir=config.get('profiles_dir', '/app/profiles')
        )

        # Initialize vector store for RAG (project-aware)
        self.vector_store = VectorStore(project_slug=project_slug)

        # NOTE: Deterministic checks and forensics are handled by paperless-anomaly-detector
        # This container ONLY does AI/LLM analysis
        # self.extractor = UnstructuredExtractor()  # Not needed for AI-only analysis
        # self.deterministic_checker = DeterministicChecker()  # Handled by anomaly-detector
        # self.forensics_analyzer = ForensicsAnalyzer()  # Handled by anomaly-detector

        # Initialize LLM usage tracker
        logger.info("Initializing LLM Usage Tracker...")
        self.usage_tracker = LLMUsageTracker(
            db_path=config.get('usage_db_path', '/app/data/llm_usage.db')
        )
        logger.info("LLM Usage Tracker initialized successfully")

        # Optional LLM client
        self.llm_enabled = config.get('llm_enabled', False)
        if self.llm_enabled:
            self.llm_client = LLMClient(
                provider=config.get('llm_provider', 'anthropic'),
                api_key=config.get('llm_api_key'),
                model=config.get('llm_model'),
                usage_tracker=self.usage_tracker
            )
        else:
            self.llm_client = None

        # v1.5.0: Project management and smart upload
        logger.info("Initializing ProjectManager...")
        self.project_manager = ProjectManager()
        logger.info("ProjectManager initialized successfully")

        logger.info("Initializing SmartUploader...")
        self.smart_uploader = SmartUploader(
            llm_client=self.llm_client,
            paperless_client=self.paperless,
            project_manager=self.project_manager
        ) if self.llm_enabled else None
        logger.info("SmartUploader initialized successfully")

        # v1.5.0: Re-analysis tracking for project-wide context
        self.last_document_time = {}  # project_slug -> timestamp of last processed doc
        self.last_reanalysis_time = {}  # project_slug -> timestamp of last re-analysis
        self.reanalysis_delay_seconds = 300  # 5 minutes after last doc before re-analyzing
        self.documents_processed_this_cycle = 0

        # v2.0.4: Stale embedding detection
        self._stale_check_counter = 0  # Incremented each poll; triggers check every 10 polls

        # v2.0.5: Guard so re_analyze_project never runs concurrently with itself
        self._reanalysis_running = False

        self.archive_path = config.get('archive_path', '/paperless/media/documents/archive')
        logger.info("DocumentAnalyzer initialization complete")


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    return {
        'paperless_api_base_url': os.getenv('PAPERLESS_API_BASE_URL', 'http://paperless-web:8000'),
        'paperless_api_token': os.getenv('PAPERLESS_API_TOKEN', ''),
        'poll_interval_seconds': int(os.getenv('POLL_INTERVAL_SECONDS', '30')),
        'state_path': os.getenv('STATE_PATH', '/app/data/state.json'),
        'profiles_dir': os.getenv('PROFILES_DIR', '/app/profiles'),
        # NOTE: archive_path, balance_tolerance, forensics_dpi no longer needed
        # Deterministic checks and forensics handled by paperless-anomaly-detector
        'llm_enabled': os.getenv('LLM_ENABLED', 'true').lower() == 'true',
        'llm_provider': os.getenv('LLM_PROVIDER', 'anthropic'),
        'llm_api_key': os.getenv('LLM_API_KEY'),
        'llm_model': os.getenv('LLM_MODEL'),
        'web_ui_enabled': os.getenv('WEB_UI_ENABLED', 'true').lower() == 'true',
        'web_host': os.getenv('WEB_HOST', '0.0.0.0'),
        'web_port': int(os.getenv('WEB_PORT', '8051')),
    }


def _install_sigterm_handler():
    """
    Install a SIGTERM handler that waits for active CI runs before exiting.

    Without this, daemon threads (CI orchestrator jobs) are killed the instant
    the main thread receives SIGTERM — causing in-progress CI runs to be marked
    as failed on next startup. With this handler, the process stays alive up to
    the container's stop_grace_period (600s) so running CI jobs can finish.
    """
    def _handler(signum, frame):
        logger.info("SIGTERM received — checking for active CI runs before shutdown...")
        try:
            from analyzer.case_intelligence.job_manager import get_job_manager
            jm = get_job_manager()
            active = jm.list_active_runs()
            if active:
                logger.info(f"Waiting for {len(active)} active CI run(s) to complete: {active}")
                # Wait up to 540s (9 min) — container stop_grace_period is 600s
                deadline = time.time() + 540
                while time.time() < deadline:
                    still_active = jm.list_active_runs()
                    if not still_active:
                        logger.info("All CI runs completed — shutting down cleanly.")
                        break
                    logger.info(f"Still waiting on CI runs: {still_active}")
                    time.sleep(10)
                else:
                    logger.warning("Shutdown timeout — CI run(s) still active, exiting anyway.")
            else:
                logger.info("No active CI runs — shutting down immediately.")
        except Exception as e:
            logger.error(f"Error in SIGTERM handler: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handler)
    logger.info("SIGTERM handler installed (will wait up to 540s for active CI runs)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Paperless AI Analyzer')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no tag writes)')
    parser.add_argument('--doc-id', type=int, help='Analyze single document by ID')
    args = parser.parse_args()

    config = load_config()

    if not config['paperless_api_token']:
        logger.error("PAPERLESS_API_TOKEN not set")
        sys.exit(1)

    _install_sigterm_handler()

    analyzer = DocumentAnalyzer(config)

    if args.doc_id:
        # Single document mode
        analyzer.analyze_single_document(args.doc_id, dry_run=args.dry_run)
    else:
        # Polling mode
        analyzer.run_polling_loop()


if __name__ == '__main__':
    main()
