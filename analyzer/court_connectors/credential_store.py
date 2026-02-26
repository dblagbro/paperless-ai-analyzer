"""
AES-256-GCM credential encryption/decryption.

Key is derived from the Flask secret key file so rotating the secret key
(which should be rare) automatically invalidates stored court passwords.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_KEY_FILE = Path('/app/data/.flask_secret_key')
_CONTEXT = b'court-credentials'
_KEY_DERIVATION_PREFIX = b'court-import-credential-key-v1:'


def _derive_key() -> bytes:
    """Derive a 32-byte AES key from the Flask secret key file."""
    raw_hex = _KEY_FILE.read_text().strip()
    return hashlib.sha256(_KEY_DERIVATION_PREFIX + bytes.fromhex(raw_hex)).digest()


def encrypt_password(password: str) -> bytes:
    """
    Encrypt a plain-text password to AES-256-GCM blob.

    Returns:
        12-byte nonce || ciphertext blob
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = _derive_key()
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, password.encode('utf-8'), _CONTEXT)
    return nonce + ciphertext


def decrypt_password(blob: bytes) -> Optional[str]:
    """
    Decrypt an AES-256-GCM blob to a plain-text password.

    Returns:
        Decrypted password string, or None if decryption fails.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if len(blob) < 13:
            return None
        key = _derive_key()
        nonce = blob[:12]
        ciphertext = blob[12:]
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, _CONTEXT)
        return plaintext.decode('utf-8')
    except Exception as e:
        logger.error(f"Credential decryption failed: {e}")
        return None


def is_cryptography_available() -> bool:
    """Return True if the cryptography package is installed."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa
        return True
    except ImportError:
        return False
