"""
LLM proxy endpoint pool with in-memory circuit breaker.

Maintains an ordered list of llm-proxy nodes (v1 or v2) from the
`llm_proxy_endpoints` DB table, picks the best healthy one per request,
and applies a 3-failure / 60-second cooldown per endpoint.

Ported from /home/dblagbro/docker/devingpt/services/proxy_manager.py
(the canonical reference) with minor adaptation to this app's db module.

v1 (Node.js llm-proxy-manager):     Authorization: Bearer <key>
v2 (Python/FastAPI llm-proxy2):     x-api-key: <key>
"""
import time
import threading
import logging
from typing import Optional

from openai import OpenAI

from analyzer.db import llm_proxy_list_endpoints

logger = logging.getLogger(__name__)

# Circuit breaker: 3 failures within the window → 60 s cooldown
_FAILURE_THRESHOLD = 3
_COOLDOWN_SECS = 60

_lock = threading.Lock()
_state: dict[str, dict] = {}  # {endpoint_id: {"failures": int, "cooldown_until": float}}


def _ep_state(eid: str) -> dict:
    if eid not in _state:
        _state[eid] = {"failures": 0, "cooldown_until": 0.0}
    return _state[eid]


def mark_success(endpoint_id: str) -> None:
    """Reset circuit-breaker state for an endpoint after a successful call."""
    with _lock:
        _ep_state(endpoint_id)["failures"] = 0
        _ep_state(endpoint_id)["cooldown_until"] = 0.0


def mark_failure(endpoint_id: str) -> None:
    """Record a failure. Trips breaker after _FAILURE_THRESHOLD failures."""
    with _lock:
        st = _ep_state(endpoint_id)
        st["failures"] += 1
        if st["failures"] >= _FAILURE_THRESHOLD:
            st["cooldown_until"] = time.time() + _COOLDOWN_SECS
            logger.warning(
                f"[llm-proxy] endpoint {endpoint_id} tripped — "
                f"cooling down {_COOLDOWN_SECS}s"
            )


def _is_healthy(endpoint_id: str) -> bool:
    with _lock:
        st = _ep_state(endpoint_id)
        if st["cooldown_until"] and time.time() < st["cooldown_until"]:
            return False
        if st["cooldown_until"] and time.time() >= st["cooldown_until"]:
            # Cooldown expired: reset and grant one retry
            st["failures"] = 0
            st["cooldown_until"] = 0.0
    return True


def get_breaker_status(endpoint_id: str) -> dict:
    """Return circuit-breaker state for admin UI display."""
    with _lock:
        st = _ep_state(endpoint_id).copy()
    now = time.time()
    cooldown_remaining = max(0, st["cooldown_until"] - now) if st["cooldown_until"] else 0
    return {
        "failures": st["failures"],
        "tripped": cooldown_remaining > 0,
        "cooldown_remaining_sec": int(cooldown_remaining),
    }


def get_endpoints() -> list[dict]:
    """Return all enabled endpoints ordered by priority ASC (fall back to [] on error)."""
    try:
        return llm_proxy_list_endpoints()
    except Exception as e:
        logger.error(f"[llm-proxy] failed to load endpoints: {e}")
        return []


def get_healthy_endpoints() -> list[dict]:
    """Return enabled endpoints NOT currently in breaker cooldown."""
    return [ep for ep in get_endpoints() if _is_healthy(ep["id"])]


def build_client(endpoint: dict) -> OpenAI:
    """Build an OpenAI-compatible client for a proxy endpoint.

    v1 uses `Authorization: Bearer`; v2 uses `x-api-key` header.
    """
    url = endpoint["url"].rstrip("/")
    key = endpoint["api_key"]
    version = int(endpoint.get("version", 1))
    if version == 2:
        # llm-proxy2: x-api-key header, SDK api_key is ignored
        return OpenAI(
            base_url=url,
            api_key="not-used",
            default_headers={"x-api-key": key},
        )
    # v1: standard Bearer token via api_key
    return OpenAI(base_url=url, api_key=key)


def get_chat_client() -> tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, endpoint_id) for the best available proxy endpoint,
    or (None, None) if no healthy endpoint exists.
    """
    endpoints = get_healthy_endpoints()
    if endpoints:
        ep = endpoints[0]
        return build_client(ep), ep["id"]
    return None, None


def get_all_clients() -> list[tuple[OpenAI, str]]:
    """Return (client, endpoint_id) for ALL healthy endpoints, priority ASC.

    Used by the retry loop in proxy_call.call_llm(): iterate, try each,
    on connection failure mark_failure(eid) and try the next.
    """
    return [(build_client(ep), ep["id"]) for ep in get_healthy_endpoints()]


def build_anthropic_client(endpoint: dict, lmrh_hint: str = ""):
    """Build a native Anthropic SDK client pointed at a proxy endpoint.

    Required for callers that depend on Anthropic-native features the
    OpenAI compat shim can't express — primarily prompt caching with
    ``cache_control`` blocks (used by Case Intelligence director-tier
    calls and theory_planner). The Anthropic SDK passes through to
    ``<base_url>/v1/messages`` exactly as it would against
    api.anthropic.com.

    v2 endpoints want the API key in ``x-api-key``. The Anthropic SDK
    already sends ``x-api-key`` from its own ``api_key`` argument, so
    for v2 we pass the real key through. For v1 (Bearer) we override
    with a ``default_headers={"Authorization": ...}`` and pass a
    placeholder.
    """
    import anthropic  # lazy import — anthropic is only needed when caller asks for prompt caching
    url = endpoint["url"].rstrip("/")
    # Anthropic SDK appends /v1 itself if base_url omits it; trim if present
    if url.endswith("/v1"):
        base = url[:-3]
    else:
        base = url
    key = endpoint["api_key"]
    version = int(endpoint.get("version", 2))
    extra_headers: dict = {}
    if lmrh_hint:
        extra_headers["LLM-Hint"] = lmrh_hint
    if version == 2:
        return anthropic.Anthropic(
            base_url=base, api_key=key,
            default_headers=extra_headers or None,
        )
    # v1: Bearer token
    extra_headers["Authorization"] = f"Bearer {key}"
    return anthropic.Anthropic(
        base_url=base, api_key="ignored-v1",
        default_headers=extra_headers,
    )


def get_all_anthropic_clients(operation: str = "") -> list[tuple]:
    """Return [(Anthropic client, endpoint_id), ...] for ALL healthy endpoints.

    Each client is pre-configured with the LMRH hint for ``operation`` so
    every ``messages.create()`` call carries the hint. ``operation`` is
    the same key used in :func:`analyzer.llm.lmrh.get_hint` (one of the
    TASK_PRESETS keys, e.g. ``"theory"``, ``"reasoning"``, ``"warroom"``).
    Pass an empty string to send no hint.
    """
    from analyzer.llm.lmrh import get_hint
    hint = get_hint(operation) if operation else ""
    return [(build_anthropic_client(ep, hint), ep["id"])
            for ep in get_healthy_endpoints()]
