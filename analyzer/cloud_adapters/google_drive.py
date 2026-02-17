"""
Google Drive Adapter

Integration with Google Drive API for file downloads.
Supports both OAuth2 user authentication and service account authentication.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional
from .base import CloudServiceAdapter

logger = logging.getLogger(__name__)


class GoogleDriveAdapter(CloudServiceAdapter):
    """Google Drive cloud storage adapter."""

    def __init__(self, credentials: Dict):
        """
        Initialize Google Drive adapter.

        Credentials dict should contain ONE of:
        - 'access_token': OAuth2 access token (user authentication)
        - 'service_account_json': Path to service account JSON file
        - 'service_account_info': Service account JSON as dict

        Optional:
        - 'refresh_token': OAuth2 refresh token
        - 'client_id': OAuth2 client ID
        - 'client_secret': OAuth2 client secret
        """
        super().__init__(credentials)
        self.service = None
        self.creds = None

    async def authenticate(self) -> bool:
        """
        Authenticate with Google Drive API.

        Returns:
            True if authentication successful
        """
        try:
            from google.oauth2.credentials import Credentials
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            # Option 1: OAuth2 user token
            if 'access_token' in self.credentials:
                logger.debug("Authenticating with OAuth2 access token")

                token = self.credentials['access_token']
                refresh_token = self.credentials.get('refresh_token')
                client_id = self.credentials.get('client_id')
                client_secret = self.credentials.get('client_secret')

                self.creds = Credentials(
                    token=token,
                    refresh_token=refresh_token,
                    token_uri='https://oauth2.googleapis.com/token',
                    client_id=client_id,
                    client_secret=client_secret
                )

            # Option 2: Service account JSON file
            elif 'service_account_json' in self.credentials:
                logger.debug("Authenticating with service account JSON file")

                json_path = self.credentials['service_account_json']
                self.creds = service_account.Credentials.from_service_account_file(
                    json_path,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )

            # Option 3: Service account info dict
            elif 'service_account_info' in self.credentials:
                logger.debug("Authenticating with service account info")

                info = self.credentials['service_account_info']
                self.creds = service_account.Credentials.from_service_account_info(
                    info,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )

            else:
                raise ValueError(
                    "Google Drive credentials must include 'access_token', "
                    "'service_account_json', or 'service_account_info'"
                )

            # Build Drive API service
            self.service = build('drive', 'v3', credentials=self.creds)

            # Test authentication with a simple API call
            self.service.about().get(fields='user').execute()

            self.authenticated = True
            logger.info("Google Drive authentication successful")
            return True

        except ImportError as e:
            logger.error(f"Google Drive libraries not installed: {e}")
            raise Exception(
                "Google Drive integration requires: "
                "pip install google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib"
            )
        except Exception as e:
            logger.error(f"Google Drive authentication failed: {e}")
            self.authenticated = False
            raise

    async def list_files(self, folder_path: Optional[str] = None,
                        page_token: Optional[str] = None) -> Dict:
        """
        List files in Google Drive folder.

        Args:
            folder_path: Folder ID (None for root/My Drive)
            page_token: Token for pagination

        Returns:
            Dict with 'files' list and optional 'next_page_token'
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Build query
            query = "trashed = false"
            if folder_path:
                query += f" and '{folder_path}' in parents"

            # List files
            results = self.service.files().list(
                q=query,
                pageSize=100,
                pageToken=page_token,
                fields="nextPageToken, files(id, name, size, modifiedTime, mimeType)",
                orderBy='name'
            ).execute()

            files = results.get('files', [])
            next_token = results.get('nextPageToken')

            # Convert to standard format
            formatted_files = []
            for file in files:
                is_folder = file.get('mimeType') == 'application/vnd.google-apps.folder'

                formatted_files.append({
                    'id': file['id'],
                    'name': file['name'],
                    'size': int(file.get('size', 0)) if not is_folder else 0,
                    'modified': file.get('modifiedTime'),
                    'mime_type': file.get('mimeType'),
                    'is_folder': is_folder
                })

            return {
                'files': formatted_files,
                'next_page_token': next_token
            }

        except Exception as e:
            logger.error(f"Failed to list Google Drive files: {e}")
            raise

    async def download_file(self, file_id: str, output_path: str) -> str:
        """
        Download file from Google Drive.

        Args:
            file_id: Google Drive file ID
            output_path: Local path to save file

        Returns:
            Local file path
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io

            # Check if it's a Google Workspace file that needs export
            file_metadata = await self.get_file_metadata(file_id)
            mime_type = file_metadata.get('mime_type', '')

            # Google Workspace files need to be exported
            if mime_type.startswith('application/vnd.google-apps.'):
                logger.info(f"Exporting Google Workspace file: {mime_type}")

                # Map Google types to export formats
                export_formats = {
                    'application/vnd.google-apps.document': 'application/pdf',
                    'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    'application/vnd.google-apps.presentation': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                }

                export_mime = export_formats.get(mime_type, 'application/pdf')
                request = self.service.files().export_media(fileId=file_id, mimeType=export_mime)
            else:
                # Regular file download
                request = self.service.files().get_media(fileId=file_id)

            # Download file
            fh = io.FileIO(output_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    if progress % 25 == 0:  # Log every 25%
                        logger.debug(f"Download progress: {progress}%")

            fh.close()
            logger.info(f"Downloaded Google Drive file: {file_id} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to download Google Drive file {file_id}: {e}")
            raise

    async def get_file_metadata(self, file_id: str) -> Dict:
        """
        Get Google Drive file metadata.

        Args:
            file_id: Google Drive file ID

        Returns:
            Dict with file metadata
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            file = self.service.files().get(
                fileId=file_id,
                fields='id, name, size, modifiedTime, mimeType, parents'
            ).execute()

            is_folder = file.get('mimeType') == 'application/vnd.google-apps.folder'

            return {
                'id': file['id'],
                'name': file['name'],
                'size': int(file.get('size', 0)) if not is_folder else 0,
                'modified': file.get('modifiedTime'),
                'mime_type': file.get('mimeType'),
                'is_folder': is_folder,
                'parents': file.get('parents', [])
            }

        except Exception as e:
            logger.error(f"Failed to get Google Drive file metadata: {e}")
            raise

    async def search_files(self, query: str, folder_path: Optional[str] = None) -> List[Dict]:
        """
        Search Google Drive files.

        Args:
            query: Search query (file name contains)
            folder_path: Folder ID to search in

        Returns:
            List of matching files
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Build search query
            search_query = f"name contains '{query}' and trashed = false"
            if folder_path:
                search_query += f" and '{folder_path}' in parents"

            results = self.service.files().list(
                q=search_query,
                pageSize=50,
                fields="files(id, name, size, modifiedTime, mimeType)",
                orderBy='name'
            ).execute()

            files = results.get('files', [])

            # Convert to standard format
            formatted_files = []
            for file in files:
                is_folder = file.get('mimeType') == 'application/vnd.google-apps.folder'

                formatted_files.append({
                    'id': file['id'],
                    'name': file['name'],
                    'size': int(file.get('size', 0)) if not is_folder else 0,
                    'modified': file.get('modifiedTime'),
                    'mime_type': file.get('mimeType'),
                    'is_folder': is_folder
                })

            return formatted_files

        except Exception as e:
            logger.error(f"Failed to search Google Drive: {e}")
            raise

    def disconnect(self):
        """Clean up Google Drive connection."""
        self.service = None
        self.creds = None
        self.authenticated = False
        logger.debug("Google Drive disconnected")
