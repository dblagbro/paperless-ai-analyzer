"""
OneDrive/SharePoint Adapter

Integration with Microsoft OneDrive and SharePoint via Microsoft Graph API.
"""

import logging
import requests
from typing import Dict, List, Optional
from .base import CloudServiceAdapter

logger = logging.getLogger(__name__)


class OneDriveAdapter(CloudServiceAdapter):
    """OneDrive/SharePoint cloud storage adapter."""

    def __init__(self, credentials: Dict):
        """
        Initialize OneDrive adapter.

        Credentials dict should contain:
        - 'access_token': Microsoft Graph access token (required)

        Optional (for OAuth flow):
        - 'client_id': Azure AD app client ID
        - 'client_secret': Azure AD app client secret
        - 'tenant_id': Azure AD tenant ID
        - 'refresh_token': OAuth refresh token
        """
        super().__init__(credentials)
        self.graph_url = 'https://graph.microsoft.com/v1.0'
        self.headers = {}

    async def authenticate(self) -> bool:
        """
        Authenticate with Microsoft Graph API.

        Returns:
            True if authentication successful
        """
        try:
            access_token = self.credentials.get('access_token')
            if not access_token:
                raise ValueError("OneDrive credentials must include 'access_token'")

            # Set up headers
            self.headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }

            # Test authentication with a simple API call
            response = requests.get(
                f'{self.graph_url}/me/drive',
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()

            self.authenticated = True
            logger.info("OneDrive authentication successful")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"OneDrive authentication failed: {e}")
            self.authenticated = False
            raise
        except Exception as e:
            logger.error(f"OneDrive authentication error: {e}")
            self.authenticated = False
            raise

    async def list_files(self, folder_path: Optional[str] = None,
                        page_token: Optional[str] = None) -> Dict:
        """
        List files in OneDrive folder.

        Args:
            folder_path: Folder ID or path (None for root)
            page_token: URL for next page

        Returns:
            Dict with 'files' list and optional 'next_page_token'
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Build URL
            if page_token:
                # Use pagination URL
                url = page_token
            elif folder_path:
                # Specific folder
                url = f'{self.graph_url}/me/drive/items/{folder_path}/children'
            else:
                # Root folder
                url = f'{self.graph_url}/me/drive/root/children'

            # Add query parameters
            params = {
                '$top': 100,
                '$select': 'id,name,size,lastModifiedDateTime,file,folder'
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Convert to standard format
            formatted_files = []
            for item in data.get('value', []):
                is_folder = 'folder' in item

                formatted_files.append({
                    'id': item['id'],
                    'name': item['name'],
                    'size': item.get('size', 0),
                    'modified': item.get('lastModifiedDateTime'),
                    'mime_type': item.get('file', {}).get('mimeType', 'application/octet-stream'),
                    'is_folder': is_folder
                })

            return {
                'files': formatted_files,
                'next_page_token': data.get('@odata.nextLink')
            }

        except Exception as e:
            logger.error(f"Failed to list OneDrive files: {e}")
            raise

    async def download_file(self, file_id: str, output_path: str) -> str:
        """
        Download file from OneDrive.

        Args:
            file_id: OneDrive file ID
            output_path: Local path to save file

        Returns:
            Local file path
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Get download URL
            url = f'{self.graph_url}/me/drive/items/{file_id}/content'

            # Download file with streaming
            response = requests.get(url, headers=self.headers, stream=True, timeout=300)
            response.raise_for_status()

            # Write to output path
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Downloaded OneDrive file: {file_id} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to download OneDrive file {file_id}: {e}")
            raise

    async def get_file_metadata(self, file_id: str) -> Dict:
        """
        Get OneDrive file metadata.

        Args:
            file_id: OneDrive file ID

        Returns:
            Dict with file metadata
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            url = f'{self.graph_url}/me/drive/items/{file_id}'
            params = {'$select': 'id,name,size,lastModifiedDateTime,file,folder'}

            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            item = response.json()

            is_folder = 'folder' in item

            return {
                'id': item['id'],
                'name': item['name'],
                'size': item.get('size', 0),
                'modified': item.get('lastModifiedDateTime'),
                'mime_type': item.get('file', {}).get('mimeType', 'application/octet-stream'),
                'is_folder': is_folder
            }

        except Exception as e:
            logger.error(f"Failed to get OneDrive file metadata: {e}")
            raise

    async def search_files(self, query: str, folder_path: Optional[str] = None) -> List[Dict]:
        """
        Search OneDrive files.

        Args:
            query: Search query
            folder_path: Folder ID to search in

        Returns:
            List of matching files
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Build search URL
            if folder_path:
                url = f'{self.graph_url}/me/drive/items/{folder_path}/search(q=\'{query}\')'
            else:
                url = f'{self.graph_url}/me/drive/root/search(q=\'{query}\')'

            params = {
                '$top': 50,
                '$select': 'id,name,size,lastModifiedDateTime,file,folder'
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Convert to standard format
            formatted_files = []
            for item in data.get('value', []):
                # Skip folders in search results
                if 'folder' in item:
                    continue

                formatted_files.append({
                    'id': item['id'],
                    'name': item['name'],
                    'size': item.get('size', 0),
                    'modified': item.get('lastModifiedDateTime'),
                    'mime_type': item.get('file', {}).get('mimeType', 'application/octet-stream'),
                    'is_folder': False
                })

            return formatted_files

        except Exception as e:
            logger.error(f"Failed to search OneDrive: {e}")
            raise

    def disconnect(self):
        """Clean up OneDrive connection."""
        self.headers = {}
        self.authenticated = False
        logger.debug("OneDrive disconnected")
