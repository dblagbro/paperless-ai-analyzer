"""HTTP helpers used by every provider — thin wrappers around requests /
urllib that return parsed JSON or None on failure.

Extracted from web_researcher.py during the v3.9.8 split."""
import json
import logging
import urllib.parse
import urllib.request
from typing import Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)

def _http_get(url: str, params: dict = None, timeout: int = 10,
              extra_headers: dict = None) -> Optional[dict]:
    """HTTP GET returning parsed JSON, or None on failure."""
    headers = {'User-Agent': 'Paperless-AI-Analyzer/3.6 (legal-research)'}
    if extra_headers:
        headers.update(extra_headers)

    if _HAS_REQUESTS:
        try:
            r = _requests.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code == 200:
                return r.json()
            logger.debug(f"GET {url} returned {r.status_code}")
        except Exception as e:
            logger.debug(f"requests GET {url} failed: {e}")

    # urllib fallback
    try:
        full_url = url
        if params:
            full_url = url + '?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(full_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8', errors='replace'))
    except Exception as e:
        logger.debug(f"urllib GET {url} failed: {e}")
    return None


def _http_post_json(url: str, payload: dict,
                    headers: dict = None, timeout: int = 10) -> Optional[dict]:
    """HTTP POST with JSON body, returning parsed JSON or None."""
    if _HAS_REQUESTS:
        try:
            h = {'Content-Type': 'application/json',
                 'User-Agent': 'Paperless-AI-Analyzer/3.6 (legal-research)',
                 **(headers or {})}
            r = _requests.post(url, json=payload, headers=h, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            logger.debug(f"POST {url} returned {r.status_code}")
        except Exception as e:
            logger.debug(f"requests POST {url} failed: {e}")
    return None
