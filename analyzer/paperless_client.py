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

    # ==================== Project Management Methods (v1.5.0) ====================

    def get_or_create_tag(self, tag_name: str, color: str = None) -> Optional[int]:
        """
        Get existing tag ID or create new tag.

        Args:
            tag_name: Tag name (e.g., 'project:case-123')
            color: Hex color (e.g., '#3498db')

        Returns:
            Tag ID or None if failed
        """
        try:
            # First try to get existing tag — use name__iexact for exact match,
            # then filter client-side as defence against Paperless versions that
            # ignore the filter and return all tags.
            url = f'{self.base_url}/api/tags/'
            response = self.session.get(url, params={'name__iexact': tag_name})
            response.raise_for_status()

            results = response.json().get('results', [])
            # Filter client-side: only accept an exact name match
            matching = [r for r in results if r['name'].lower() == tag_name.lower()]
            if matching:
                tag_id = matching[0]['id']
                logger.debug(f"Found existing tag '{tag_name}': ID {tag_id}")
                return tag_id

            # Tag doesn't exist, create it
            color = color or '#3498db'  # Default blue
            payload = {
                'name': tag_name,
                'color': color,
                'is_inbox_tag': False
            }

            response = self.session.post(url, json=payload)
            response.raise_for_status()

            tag_id = response.json()['id']
            logger.info(f"Created tag '{tag_name}': ID {tag_id}")
            return tag_id

        except Exception as e:
            logger.error(f"Failed to get/create tag '{tag_name}': {e}")
            return None

    def get_documents_by_project(self, project_slug: str, **filters) -> Dict[str, Any]:
        """
        Get documents tagged with specific project.

        Args:
            project_slug: Project identifier (e.g., 'case-123')
            **filters: Additional Paperless API filters

        Returns:
            API response with documents
        """
        tag_name = f"project:{project_slug}"
        tag_id = self.get_or_create_tag(tag_name)

        if not tag_id:
            logger.warning(f"Could not find/create tag for project '{project_slug}'")
            return {'results': [], 'count': 0}

        # Add tag filter to existing filters
        filters['tags__id__all'] = [tag_id]

        return self.get_documents(**filters)

    def add_project_tag(self, document_id: int, project_slug: str, color: str = None) -> bool:
        """
        Add project tag to document.

        Args:
            document_id: Document ID
            project_slug: Project identifier
            color: Optional color for new tag

        Returns:
            True if successful
        """
        tag_name = f"project:{project_slug}"
        tag_id = self.get_or_create_tag(tag_name, color=color)

        if not tag_id:
            return False

        try:
            # Get current document
            doc = self.get_document(document_id)
            if not doc:
                return False

            # Add tag if not already present
            current_tags = doc.get('tags', [])
            if tag_id not in current_tags:
                current_tags.append(tag_id)

                # Update document
                url = f'{self.base_url}/api/documents/{document_id}/'
                response = self.session.patch(url, json={'tags': current_tags})
                response.raise_for_status()

                logger.info(f"Added tag '{tag_name}' to document {document_id}")
                return True
            else:
                logger.debug(f"Document {document_id} already has tag '{tag_name}'")
                return True

        except Exception as e:
            logger.error(f"Failed to add tag to document {document_id}: {e}")
            return False

    def remove_project_tag(self, document_id: int, project_slug: str) -> bool:
        """
        Remove project tag from document.

        Args:
            document_id: Document ID
            project_slug: Project identifier

        Returns:
            True if successful
        """
        tag_name = f"project:{project_slug}"
        tag_id = self.get_or_create_tag(tag_name)

        if not tag_id:
            return False

        try:
            # Get current document
            doc = self.get_document(document_id)
            if not doc:
                return False

            # Remove tag if present
            current_tags = doc.get('tags', [])
            if tag_id in current_tags:
                current_tags.remove(tag_id)

                # Update document
                url = f'{self.base_url}/api/documents/{document_id}/'
                response = self.session.patch(url, json={'tags': current_tags})
                response.raise_for_status()

                logger.info(f"Removed tag '{tag_name}' from document {document_id}")
                return True
            else:
                logger.debug(f"Document {document_id} doesn't have tag '{tag_name}'")
                return True

        except Exception as e:
            logger.error(f"Failed to remove tag from document {document_id}: {e}")
            return False

    def get_documents_without_project(self, **filters) -> List[Dict]:
        """
        Get documents without any project tag.

        Args:
            **filters: Additional Paperless API filters

        Returns:
            List of documents without project: tags
        """
        try:
            # Get all documents
            all_docs = self.get_documents(page_size=1000, **filters)

            # Filter for documents without project: tags
            orphans = []
            for doc in all_docs.get('results', []):
                tags = [t['name'] for t in doc.get('tags', [])]
                has_project = any(t.startswith('project:') for t in tags)

                if not has_project:
                    orphans.append(doc)

            logger.info(f"Found {len(orphans)} documents without project tags")
            return orphans

        except Exception as e:
            logger.error(f"Failed to get orphan documents: {e}")
            return []

    def upload_document(self, file_path: str, title: str = None,
                       correspondent: int = None, document_type: int = None,
                       tags: List[int] = None, created: str = None,
                       **metadata) -> Optional[Dict]:
        """
        Upload document to Paperless.

        Args:
            file_path: Path to file
            title: Document title
            correspondent: Correspondent ID
            document_type: Document type ID
            tags: List of tag IDs
            created: Document date (ISO format)
            **metadata: Additional metadata

        Returns:
            Document dict from Paperless API or None if failed
        """
        try:
            url = f'{self.base_url}/api/documents/post_document/'

            # Prepare multipart form data
            with open(file_path, 'rb') as f:
                files = {'document': f}

                data = {}
                if title:
                    data['title'] = title
                if correspondent:
                    data['correspondent'] = correspondent
                if document_type:
                    data['document_type'] = document_type
                if tags:
                    data['tags'] = tags
                if created:
                    data['created'] = created

                # Add any additional metadata
                data.update(metadata)

                # Remove Content-Type from session temporarily so requests can
                # set multipart/form-data boundary automatically.  Session.post()
                # merges headers, so popping from a copy is not enough.
                saved_ct = self.session.headers.pop('Content-Type', None)
                try:
                    response = self.session.post(url, files=files, data=data)
                finally:
                    if saved_ct is not None:
                        self.session.headers['Content-Type'] = saved_ct
                response.raise_for_status()

                result = response.json()
                logger.info(f"Uploaded document: {title or file_path}")
                # Paperless-ngx v2+ returns a task UUID string instead of a doc dict
                if isinstance(result, str):
                    return {'task_id': result}
                return result

        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            return None

    def get_documents_by_tag(self, tag_id: int) -> List[Dict[str, Any]]:
        """
        Get all documents that have a specific tag.

        Args:
            tag_id: Tag ID to filter by

        Returns:
            List of document dicts
        """
        try:
            all_docs = []
            page = 1

            while True:
                url = f'{self.base_url}/api/documents/'
                params = {
                    'tags__id__in': tag_id,
                    'page': page,
                    'page_size': 100
                }

                response = self.session.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                docs = data.get('results', [])
                all_docs.extend(docs)

                # Check if there are more pages
                if not data.get('next'):
                    break

                page += 1

            logger.info(f"Found {len(all_docs)} documents with tag ID {tag_id}")
            return all_docs

        except Exception as e:
            logger.error(f"Failed to get documents by tag: {e}")
            return []

    def update_document(self, document_id: int, data: Dict[str, Any]) -> bool:
        """
        Update document metadata (tags, title, etc.).

        Args:
            document_id: Document ID
            data: Dict with fields to update (e.g., {'tags': [1, 2, 3]})

        Returns:
            True if successful, False otherwise
        """
        try:
            url = f'{self.base_url}/api/documents/{document_id}/'
            response = self.session.patch(url, json=data)
            response.raise_for_status()

            logger.debug(f"Updated document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
