"""
Remote File Downloader

Download files from URLs with various authentication mechanisms.
Supports Basic Auth, Bearer tokens, OAuth2, custom headers, etc.
"""

import logging
import os
import tempfile
import re
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, unquote
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class RemoteFileDownloader:
    """Download files from URLs with authentication support."""

    def __init__(self, timeout: int = 300, max_size_mb: int = 500):
        """
        Initialize remote file downloader.

        Args:
            timeout: Request timeout in seconds (default 5 minutes)
            max_size_mb: Maximum file size in MB (default 500MB)
        """
        self.timeout = timeout
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Paperless-AI-Analyzer/1.5.0'
        })

    def download_from_url(
        self,
        url: str,
        auth_type: str = 'none',
        username: str = None,
        password: str = None,
        token: str = None,
        custom_headers: Dict[str, str] = None
    ) -> Tuple[str, Dict]:
        """
        Download file from URL with specified authentication.

        Args:
            url: URL to download from
            auth_type: Authentication type - 'none', 'basic', 'bearer', 'digest', 'custom'
            username: Username for basic/digest auth
            password: Password for basic/digest auth
            token: Bearer token or OAuth2 token
            custom_headers: Dictionary of custom headers

        Returns:
            Tuple of (local_file_path, metadata_dict)

        Raises:
            Exception: If download fails
        """
        try:
            logger.info(f"Downloading from URL: {url} (auth: {auth_type})")

            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")

            # Build request with authentication
            headers = {}
            auth = None

            # Apply authentication
            if auth_type == 'basic':
                if not username or not password:
                    raise ValueError("Basic auth requires username and password")
                auth = (username, password)
                logger.debug("Using Basic authentication")

            elif auth_type == 'bearer':
                if not token:
                    raise ValueError("Bearer auth requires token")
                headers['Authorization'] = f'Bearer {token}'
                logger.debug("Using Bearer token authentication")

            elif auth_type == 'digest':
                if not username or not password:
                    raise ValueError("Digest auth requires username and password")
                from requests.auth import HTTPDigestAuth
                auth = HTTPDigestAuth(username, password)
                logger.debug("Using Digest authentication")

            elif auth_type == 'oauth2':
                # OAuth2 is essentially bearer token
                if not token:
                    raise ValueError("OAuth2 requires access token")
                headers['Authorization'] = f'Bearer {token}'
                logger.debug("Using OAuth2 token authentication")

            elif auth_type == 'custom':
                if custom_headers:
                    headers.update(custom_headers)
                    logger.debug(f"Using custom headers: {list(custom_headers.keys())}")

            elif auth_type != 'none':
                raise ValueError(f"Unsupported auth type: {auth_type}")

            # Perform HEAD request first to check size and content type
            head_response = self.session.head(
                url,
                auth=auth,
                headers=headers,
                timeout=30,
                allow_redirects=True
            )

            # If HEAD not allowed, proceed with GET
            if head_response.status_code == 405:
                logger.debug("HEAD not allowed, proceeding with GET")
            elif head_response.status_code >= 400:
                raise Exception(f"HTTP {head_response.status_code}: {head_response.reason}")
            else:
                # Check file size
                content_length = head_response.headers.get('content-length')
                if content_length:
                    size_bytes = int(content_length)
                    if size_bytes > self.max_size_bytes:
                        raise Exception(
                            f"File too large: {size_bytes / (1024*1024):.1f}MB "
                            f"(max: {self.max_size_bytes / (1024*1024):.0f}MB)"
                        )

            # Download file with streaming
            response = self.session.get(
                url,
                auth=auth,
                headers=headers,
                timeout=self.timeout,
                stream=True,
                allow_redirects=True
            )
            response.raise_for_status()

            # Determine filename
            filename = self._extract_filename(response, url)

            # Create temp file
            temp_dir = tempfile.mkdtemp(prefix='paperless_remote_')
            local_path = os.path.join(temp_dir, filename)

            # Download with size check
            downloaded_size = 0
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > self.max_size_bytes:
                            os.remove(local_path)
                            os.rmdir(temp_dir)
                            raise Exception(
                                f"File exceeds size limit during download "
                                f"({downloaded_size / (1024*1024):.1f}MB)"
                            )
                        f.write(chunk)

            # Build metadata
            metadata = {
                'url': url,
                'filename': filename,
                'size_bytes': downloaded_size,
                'size_mb': round(downloaded_size / (1024*1024), 2),
                'content_type': response.headers.get('content-type', 'unknown'),
                'status_code': response.status_code,
                'final_url': response.url  # After redirects
            }

            logger.info(f"Downloaded {filename} ({metadata['size_mb']}MB) from {url}")
            return local_path, metadata

        except requests.exceptions.Timeout:
            raise Exception(f"Download timeout after {self.timeout} seconds")
        except requests.exceptions.ConnectionError as e:
            raise Exception(f"Connection error: {str(e)}")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error: {e.response.status_code} - {e.response.reason}")
        except Exception as e:
            logger.error(f"Failed to download from URL: {e}", exc_info=True)
            raise

    def _is_valid_url(self, url: str) -> bool:
        """
        Validate URL format and security.

        Args:
            url: URL to validate

        Returns:
            True if valid
        """
        try:
            parsed = urlparse(url)

            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False

            # Only allow HTTP/HTTPS (no file://, ftp://, etc. for security)
            if parsed.scheme.lower() not in ['http', 'https']:
                logger.warning(f"Blocked non-HTTP(S) URL: {parsed.scheme}://")
                return False

            # Block localhost/internal IPs to prevent SSRF
            netloc_lower = parsed.netloc.lower()
            blocked_hosts = [
                'localhost', '127.0.0.1', '0.0.0.0',
                '169.254.169.254',  # AWS metadata
                '::1', '[::1]'  # IPv6 localhost
            ]

            for blocked in blocked_hosts:
                if netloc_lower.startswith(blocked):
                    logger.warning(f"Blocked internal/localhost URL: {parsed.netloc}")
                    return False

            # Block private IP ranges (basic check)
            if any(netloc_lower.startswith(prefix) for prefix in ['10.', '172.16.', '192.168.']):
                logger.warning(f"Blocked private IP URL: {parsed.netloc}")
                return False

            return True

        except Exception as e:
            logger.warning(f"URL validation failed: {e}")
            return False

    def _extract_filename(self, response: requests.Response, url: str) -> str:
        """
        Extract filename from response or URL.

        Args:
            response: HTTP response
            url: Original URL

        Returns:
            Filename string
        """
        # Try Content-Disposition header first
        content_disp = response.headers.get('content-disposition', '')
        if content_disp:
            # Look for filename="..." or filename*=UTF-8''...
            match = re.search(r'filename[*]?=["\']?([^"\';\r\n]+)', content_disp)
            if match:
                filename = match.group(1).strip()
                # Handle RFC 5987 encoding (filename*=UTF-8''...)
                if filename.startswith("UTF-8''"):
                    filename = unquote(filename[7:])
                else:
                    filename = unquote(filename)
                logger.debug(f"Extracted filename from Content-Disposition: {filename}")
                return filename

        # Fall back to URL path
        parsed_url = urlparse(url)
        path = parsed_url.path
        filename = os.path.basename(path)

        if filename:
            filename = unquote(filename)
            logger.debug(f"Extracted filename from URL: {filename}")
            return filename

        # Last resort: generate from content type
        content_type = response.headers.get('content-type', 'application/octet-stream')
        ext = self._get_extension_from_mime(content_type)
        filename = f'download_{hash(url) % 10000}{ext}'
        logger.debug(f"Generated filename: {filename}")
        return filename

    def _get_extension_from_mime(self, content_type: str) -> str:
        """
        Get file extension from MIME type.

        Args:
            content_type: MIME type string

        Returns:
            File extension with dot (e.g., '.pdf')
        """
        # Simple mapping of common types
        mime_map = {
            'application/pdf': '.pdf',
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/gif': '.gif',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.ms-excel': '.xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'text/plain': '.txt',
            'text/html': '.html',
            'application/zip': '.zip'
        }

        # Extract base MIME type (remove charset, etc.)
        base_type = content_type.split(';')[0].strip().lower()
        return mime_map.get(base_type, '.bin')

    def cleanup(self, file_path: str):
        """
        Clean up downloaded file and temp directory.

        Args:
            file_path: Path to file to clean up
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)

                # Remove temp directory if empty
                temp_dir = os.path.dirname(file_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)

                logger.debug(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")
