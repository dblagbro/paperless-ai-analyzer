"""
Paperless API Client

Handles all interactions with the Paperless-ngx API.
"""

import logging
import requests
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class PaperlessClient:
    """Client for interacting with Paperless-ngx API."""

    def __init__(self, base_url: str, api_token: str):
        """
        Initialize the Paperless API client.

        Args:
            base_url: Base URL of Paperless API (e.g., http://paperless-web:8000)
            api_token: API authentication token
        """
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Token {api_token}',
            'Content-Type': 'application/json'
        })

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_documents(self,
                      ordering: str = '-modified',
                      page_size: int = 100,
                      page: int = 1,
                      modified_after: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch documents from Paperless API.

        Args:
            ordering: Sort order (e.g., '-modified' for newest first)
            page_size: Number of results per page
            page: Page number
            modified_after: ISO datetime string to filter by modified date

        Returns:
            API response with results, count, next, previous
        """
        url = f'{self.base_url}/api/documents/'
        params = {
            'ordering': ordering,
            'page_size': page_size,
            'page': page
        }

        if modified_after:
            params['modified__gt'] = modified_after

        logger.debug(f"Fetching documents: {params}")
        response = self.session.get(url, params=params)
        response.raise_for_status()

        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_document(self, document_id: int) -> Dict[str, Any]:
        """
        Fetch a single document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document details including content, tags, custom fields
        """
        url = f'{self.base_url}/api/documents/{document_id}/'
        logger.debug(f"Fetching document {document_id}")

        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def download_document(self, document_id: int, archived: bool = True) -> bytes:
        """
        Download document PDF.

        Args:
            document_id: Document ID
            archived: If True, download archived (OCR'd) version

        Returns:
            PDF content as bytes
        """
        if archived:
            url = f'{self.base_url}/api/documents/{document_id}/download/'
        else:
            url = f'{self.base_url}/api/documents/{document_id}/preview/'

        logger.debug(f"Downloading document {document_id} (archived={archived})")
        response = self.session.get(url)
        response.raise_for_status()

        return response.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_or_create_tag(self, tag_name: str) -> int:
        """
        Get tag ID by name, or create if it doesn't exist.

        Args:
            tag_name: Tag name (e.g., 'anomaly:balance_mismatch')

        Returns:
            Tag ID
        """
        # First, try to find existing tag
        url = f'{self.base_url}/api/tags/'
        params = {'name': tag_name}

        response = self.session.get(url, params=params)
        response.raise_for_status()
        results = response.json()

        if results['count'] > 0:
            tag_id = results['results'][0]['id']
            logger.debug(f"Found existing tag '{tag_name}' with ID {tag_id}")
            return tag_id

        # Create new tag
        data = {'name': tag_name}
        response = self.session.post(url, json=data)
        response.raise_for_status()
        tag_id = response.json()['id']

        logger.info(f"Created new tag '{tag_name}' with ID {tag_id}")
        return tag_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def update_document_tags(self, document_id: int, tag_names: List[str], add_only: bool = True) -> None:
        """
        Update document tags.

        Args:
            document_id: Document ID
            tag_names: List of tag names to add
            add_only: If True, only add tags (don't remove existing)
        """
        # Get current document to fetch existing tags
        doc = self.get_document(document_id)
        existing_tag_ids = set(doc.get('tags', []))

        # Get or create tag IDs
        new_tag_ids = set()
        for tag_name in tag_names:
            tag_id = self.get_or_create_tag(tag_name)
            new_tag_ids.add(tag_id)

        # Combine with existing if add_only
        if add_only:
            all_tag_ids = list(existing_tag_ids | new_tag_ids)
        else:
            all_tag_ids = list(new_tag_ids)

        # Only update if there are new tags
        if new_tag_ids - existing_tag_ids:
            url = f'{self.base_url}/api/documents/{document_id}/'
            data = {'tags': all_tag_ids}

            logger.info(f"Updating document {document_id} with tags: {tag_names}")
            response = self.session.patch(url, json=data)
            response.raise_for_status()
        else:
            logger.debug(f"Document {document_id} already has all specified tags")

    def get_document_tags(self, document_id: int) -> List[str]:
        """
        Get list of tag names for a document.

        Args:
            document_id: Document ID

        Returns:
            List of tag names
        """
        doc = self.get_document(document_id)
        tag_ids = doc.get('tags', [])

        # Fetch tag names
        tag_names = []
        for tag_id in tag_ids:
            url = f'{self.base_url}/api/tags/{tag_id}/'
            response = self.session.get(url)
            if response.ok:
                tag_names.append(response.json()['name'])

        return tag_names

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def add_document_note(self, document_id: int, note: str, append: bool = True) -> None:
        """
        Store AI analysis. Since notes API has permission issues,
        we'll store in vector database and use custom fields for summary.

        Args:
            document_id: Document ID
            note: Note text to add
            append: Not used
        """
        logger.info(f"AI analysis for document {document_id} will be stored in vector database")
        # Note: The actual storage happens in main.py via vector_store.embed_document()
        # This function is kept for API compatibility but doesn't write to Paperless
        # The full analysis is in the vector store, accessible via chat

    def health_check(self) -> bool:
        """
        Check if Paperless API is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            url = f'{self.base_url}/api/documents/'
            response = self.session.get(url, params={'page_size': 1})
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
