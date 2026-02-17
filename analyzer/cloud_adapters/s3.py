"""
Amazon S3 Adapter

Integration with Amazon S3 for file downloads.
"""

import logging
from typing import Dict, List, Optional
from .base import CloudServiceAdapter

logger = logging.getLogger(__name__)


class S3Adapter(CloudServiceAdapter):
    """Amazon S3 cloud storage adapter."""

    def __init__(self, credentials: Dict):
        """
        Initialize S3 adapter.

        Credentials dict should contain:
        - 'aws_access_key_id': AWS access key (required)
        - 'aws_secret_access_key': AWS secret key (required)
        - 'bucket_name': S3 bucket name (required)

        Optional:
        - 'region_name': AWS region (default: us-east-1)
        - 'endpoint_url': Custom S3 endpoint (for S3-compatible services)
        """
        super().__init__(credentials)
        self.s3_client = None
        self.bucket_name = credentials.get('bucket_name')

    async def authenticate(self) -> bool:
        """
        Authenticate with Amazon S3.

        Returns:
            True if authentication successful
        """
        try:
            import boto3
            from botocore.exceptions import ClientError

            aws_access_key = self.credentials.get('aws_access_key_id')
            aws_secret_key = self.credentials.get('aws_secret_access_key')
            region = self.credentials.get('region_name', 'us-east-1')
            endpoint_url = self.credentials.get('endpoint_url')

            if not aws_access_key or not aws_secret_key or not self.bucket_name:
                raise ValueError(
                    "S3 credentials must include 'aws_access_key_id', "
                    "'aws_secret_access_key', and 'bucket_name'"
                )

            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region,
                endpoint_url=endpoint_url
            )

            # Test authentication by listing bucket
            self.s3_client.head_bucket(Bucket=self.bucket_name)

            self.authenticated = True
            logger.info(f"S3 authentication successful (bucket: {self.bucket_name})")
            return True

        except ImportError:
            logger.error("boto3 library not installed")
            raise Exception(
                "S3 integration requires: pip install boto3"
            )
        except Exception as e:
            logger.error(f"S3 authentication failed: {e}")
            self.authenticated = False
            raise

    async def list_files(self, folder_path: Optional[str] = None,
                        page_token: Optional[str] = None) -> Dict:
        """
        List files in S3 bucket.

        Args:
            folder_path: S3 prefix/folder (None for bucket root)
            page_token: Continuation token for pagination

        Returns:
            Dict with 'files' list and optional 'next_page_token'
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Build list_objects parameters
            params = {
                'Bucket': self.bucket_name,
                'MaxKeys': 1000,
                'Delimiter': '/'  # Treat prefixes as folders
            }

            if folder_path:
                params['Prefix'] = folder_path.rstrip('/') + '/'

            if page_token:
                params['ContinuationToken'] = page_token

            # List objects
            response = self.s3_client.list_objects_v2(**params)

            formatted_files = []

            # Add folders (common prefixes)
            for prefix in response.get('CommonPrefixes', []):
                folder_name = prefix['Prefix'].rstrip('/').split('/')[-1]
                formatted_files.append({
                    'id': prefix['Prefix'],
                    'name': folder_name,
                    'size': 0,
                    'modified': None,
                    'mime_type': 'application/x-directory',
                    'is_folder': True
                })

            # Add files
            for obj in response.get('Contents', []):
                # Skip the folder itself
                if obj['Key'].endswith('/'):
                    continue

                # Extract filename
                filename = obj['Key'].split('/')[-1]

                formatted_files.append({
                    'id': obj['Key'],
                    'name': filename,
                    'size': obj['Size'],
                    'modified': obj['LastModified'].isoformat() if obj.get('LastModified') else None,
                    'mime_type': 'application/octet-stream',  # S3 doesn't always provide MIME
                    'is_folder': False
                })

            return {
                'files': formatted_files,
                'next_page_token': response.get('NextContinuationToken')
            }

        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            raise

    async def download_file(self, file_id: str, output_path: str) -> str:
        """
        Download file from S3.

        Args:
            file_id: S3 object key
            output_path: Local path to save file

        Returns:
            Local file path
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Download file
            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=file_id,
                Filename=output_path
            )

            logger.info(f"Downloaded S3 file: s3://{self.bucket_name}/{file_id} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to download S3 file {file_id}: {e}")
            raise

    async def get_file_metadata(self, file_id: str) -> Dict:
        """
        Get S3 file metadata.

        Args:
            file_id: S3 object key

        Returns:
            Dict with file metadata
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Get object metadata
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=file_id
            )

            return {
                'id': file_id,
                'name': file_id.split('/')[-1],
                'size': response['ContentLength'],
                'modified': response['LastModified'].isoformat() if response.get('LastModified') else None,
                'mime_type': response.get('ContentType', 'application/octet-stream'),
                'is_folder': False
            }

        except Exception as e:
            logger.error(f"Failed to get S3 file metadata: {e}")
            raise

    async def search_files(self, query: str, folder_path: Optional[str] = None) -> List[Dict]:
        """
        Search S3 files (basic prefix search).

        Note: S3 doesn't support full-text search, only prefix matching.

        Args:
            query: Search query (used as prefix)
            folder_path: S3 prefix to search in

        Returns:
            List of matching files
        """
        if not self.authenticated:
            await self.authenticate()

        try:
            # Build prefix
            prefix = folder_path or ''
            if prefix and not prefix.endswith('/'):
                prefix += '/'
            prefix += query

            # List objects with prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=100
            )

            formatted_files = []
            for page in pages:
                for obj in page.get('Contents', []):
                    # Skip folders
                    if obj['Key'].endswith('/'):
                        continue

                    filename = obj['Key'].split('/')[-1]

                    formatted_files.append({
                        'id': obj['Key'],
                        'name': filename,
                        'size': obj['Size'],
                        'modified': obj['LastModified'].isoformat() if obj.get('LastModified') else None,
                        'mime_type': 'application/octet-stream',
                        'is_folder': False
                    })

            return formatted_files

        except Exception as e:
            logger.error(f"Failed to search S3: {e}")
            raise

    def disconnect(self):
        """Clean up S3 connection."""
        self.s3_client = None
        self.authenticated = False
        logger.debug("S3 disconnected")
