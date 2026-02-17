"""
Dropbox Adapter

Integration with Dropbox API for file downloads.
"""

import logging
from typing import Dict, List, Optional
from .base import CloudServiceAdapter

logger = logging.getLogger(__name__)


class DropboxAdapter(CloudServiceAdapter):
    """Dropbox cloud storage adapter."""

    def __init__(self, credentials: Dict):
        """
        Initialize Dropbox adapter.

        Credentials dict should contain:
        - 'access_token': Dropbox access token (required)

        Optional (for OAuth flow):
        - 'app_key': Dropbox app key
        - 'app_secret': Dropbox app secret
        - 'refresh_token': OAuth refresh token
        """
        super().__init__(credentials)
        self.dbx = None

    async def authenticate(self) -> bool:
        """
        Authenticate with Dropbox API.

        Returns:
            True if authentication successful
        """
        try:
            import dropbox

            access_token = self.credentials.get('access_token')
            if not access_token:
                raise ValueError("Dropbox credentials must include 'access_token'")

            # Initialize Dropbox client
            self.dbx = dropbox.Dropbox(access_token)

            # Test authentication
            self.dbx.users_get_current_account()

            self.authenticated = True
            logger.info("Dropbox authentication successful")
            return True

        except ImportError:
            logger.error("Dropbox library not installed")
            raise Exception(
                "Dropbox integration requires: pip install dropbox"
            )
        except Exception as e:
            logger.error(f"Dropbox authentication failed: {e}")
            self.authenticated = False
            raise

    async def list_files(self, folder_path: Optional[str] = None,
                        page_token: Optional[str] = None) -> Dict:
        """
        List files in Dropbox folder.

        Args:
            folder_path: Folder path (empty string or None for root)
            page_token: Cursor for pagination

        Returns:
            Dict with 'files' list and optional 'next_page_token'
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            import dropbox

            # Default to root
            if not folder_path:
                folder_path = ''

            # List folder
            if page_token:
                result = self.dbx.files_list_folder_continue(page_token)
            else:
                result = self.dbx.files_list_folder(folder_path)

            # Convert to standard format
            formatted_files = []
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata):
                    formatted_files.append({
                        'id': entry.id,
                        'name': entry.name,
                        'size': entry.size,
                        'modified': entry.client_modified.isoformat() if entry.client_modified else None,
                        'mime_type': 'application/octet-stream',  # Dropbox doesn't provide MIME
                        'is_folder': False,
                        'path': entry.path_display
                    })
                elif isinstance(entry, dropbox.files.FolderMetadata):
                    formatted_files.append({
                        'id': entry.id,
                        'name': entry.name,
                        'size': 0,
                        'modified': None,
                        'mime_type': 'application/vnd.dropbox.folder',
                        'is_folder': True,
                        'path': entry.path_display
                    })

            return {
                'files': formatted_files,
                'next_page_token': result.cursor if result.has_more else None
            }

        except Exception as e:
            logger.error(f"Failed to list Dropbox files: {e}")
            raise

    async def download_file(self, file_id: str, output_path: str) -> str:
        """
        Download file from Dropbox.

        Note: file_id should be the path, not the ID.

        Args:
            file_id: Dropbox file path (e.g., '/documents/file.pdf')
            output_path: Local path to save file

        Returns:
            Local file path
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Download file
            metadata, response = self.dbx.files_download(file_id)

            # Write to output path
            with open(output_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded Dropbox file: {file_id} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to download Dropbox file {file_id}: {e}")
            raise

    async def get_file_metadata(self, file_id: str) -> Dict:
        """
        Get Dropbox file metadata.

        Args:
            file_id: Dropbox file path

        Returns:
            Dict with file metadata
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            import dropbox

            metadata = self.dbx.files_get_metadata(file_id)

            if isinstance(metadata, dropbox.files.FileMetadata):
                return {
                    'id': metadata.id,
                    'name': metadata.name,
                    'size': metadata.size,
                    'modified': metadata.client_modified.isoformat() if metadata.client_modified else None,
                    'mime_type': 'application/octet-stream',
                    'is_folder': False,
                    'path': metadata.path_display
                }
            elif isinstance(metadata, dropbox.files.FolderMetadata):
                return {
                    'id': metadata.id,
                    'name': metadata.name,
                    'size': 0,
                    'modified': None,
                    'mime_type': 'application/vnd.dropbox.folder',
                    'is_folder': True,
                    'path': metadata.path_display
                }

        except Exception as e:
            logger.error(f"Failed to get Dropbox file metadata: {e}")
            raise

    async def search_files(self, query: str, folder_path: Optional[str] = None) -> List[Dict]:
        """
        Search Dropbox files.

        Args:
            query: Search query
            folder_path: Folder path to search in

        Returns:
            List of matching files
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            import dropbox

            # Search
            search_options = dropbox.files.SearchOptions(
                path=folder_path or '',
                max_results=50
            )

            results = self.dbx.files_search_v2(query, options=search_options)

            # Convert to standard format
            formatted_files = []
            for match in results.matches:
                metadata = match.metadata.get_metadata()

                if isinstance(metadata, dropbox.files.FileMetadata):
                    formatted_files.append({
                        'id': metadata.id,
                        'name': metadata.name,
                        'size': metadata.size,
                        'modified': metadata.client_modified.isoformat() if metadata.client_modified else None,
                        'mime_type': 'application/octet-stream',
                        'is_folder': False,
                        'path': metadata.path_display
                    })

            return formatted_files

        except Exception as e:
            logger.error(f"Failed to search Dropbox: {e}")
            raise

    def disconnect(self):
        """Clean up Dropbox connection."""
        self.dbx = None
        self.authenticated = False
        logger.debug("Dropbox disconnected")
