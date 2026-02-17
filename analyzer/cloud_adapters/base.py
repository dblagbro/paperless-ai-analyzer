"""
Base Cloud Service Adapter

Abstract base class for cloud storage service integrations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CloudServiceAdapter(ABC):
    """Base class for cloud service adapters."""

    def __init__(self, credentials: Dict):
        """
        Initialize adapter with credentials.

        Args:
            credentials: Dict containing service-specific credentials
        """
        self.credentials = credentials
        self.authenticated = False

    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the cloud service.

        Returns:
            True if authentication successful

        Raises:
            Exception if authentication fails
        """
        pass

    @abstractmethod
    async def list_files(self, folder_path: Optional[str] = None,
                        page_token: Optional[str] = None) -> Dict:
        """
        List files in a folder.

        Args:
            folder_path: Path/ID of folder (None for root)
            page_token: Token for pagination

        Returns:
            Dict with 'files' list and optional 'next_page_token'
            File dict contains: {
                'id': 'file_id',
                'name': 'filename.pdf',
                'size': 12345,
                'modified': '2024-01-01T00:00:00Z',
                'mime_type': 'application/pdf',
                'is_folder': False
            }
        """
        pass

    @abstractmethod
    async def download_file(self, file_id: str, output_path: str) -> str:
        """
        Download file to local path.

        Args:
            file_id: Unique file identifier
            output_path: Local path to save file

        Returns:
            Local file path

        Raises:
            Exception if download fails
        """
        pass

    @abstractmethod
    async def get_file_metadata(self, file_id: str) -> Dict:
        """
        Get file metadata.

        Args:
            file_id: Unique file identifier

        Returns:
            Dict with file metadata
        """
        pass

    async def search_files(self, query: str, folder_path: Optional[str] = None) -> List[Dict]:
        """
        Search for files (optional, not all services support).

        Args:
            query: Search query
            folder_path: Limit search to folder

        Returns:
            List of file dicts
        """
        return []

    def disconnect(self):
        """Clean up connections."""
        self.authenticated = False
