"""
Cloud Service Adapters

Provides unified interface for downloading files from various cloud services.
"""

from .base import CloudServiceAdapter
from .google_drive import GoogleDriveAdapter
from .dropbox_adapter import DropboxAdapter
from .onedrive import OneDriveAdapter
from .s3 import S3Adapter

__all__ = [
    'CloudServiceAdapter',
    'GoogleDriveAdapter',
    'DropboxAdapter',
    'OneDriveAdapter',
    'S3Adapter'
]
