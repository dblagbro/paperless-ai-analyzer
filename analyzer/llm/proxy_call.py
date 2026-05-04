"""
Unified LLM call helper — the single chokepoint for every LLM call in
paperless-ai-analyzer.

Routing strategy (proxy-first, direct-provider as absolute last resort):

    1. Iterate `proxy_manager.get_all_clients()` (ordered by priority).
       For each (client, endpoint_id):
         - Call client.chat.completions.create(..., extra_headers={'LLM-Hint': ...})
         - On success: mark_success(eid), log usage, return result.
         - On connection/timeout error: mark_failure(eid), try next.
         - On non-retriable HTTP error: mark_failure(eid), try next.

    2. If no healthy proxy endpoints remain OR all failed, call the direct
       Anthropic/OpenAI SDK using keys from ai_config.json. This is the
       legacy path — preserves current behavior if the proxy pool is down.

    3. If direct-provider also fails, raise LLMUnavailableError. Callers
       should catch this and convert to a structured HTTP 503 response,
       not 500.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Any

import httpx

from analyzer.llm import proxy_manager
from analyzer.llm.lmrh import build_lmrh_header

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised when every proxy endpoint AND every direct-provider fallback failed.

    Includes an `attempted` list for diagnostic rendering.
    """

    def __init__(self, message: str, attempted: list[str] | None = None):
        super().__init__(message)
        self.attempted = attempted or []


# Exceptions that indicate the endpoint itself is unreachable / broken
# (as opposed to e.g. a 400 invalid-input error from the model).
_CONNECTION_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    ConnectionError,
    OSError,
)


def _default_model_for_task(task: str, fallback: str = "claude-haiku-4-5") -> str:
    """Return the cheapest model name to put in the SDK request body, picked
    from the task's LMRH ``cost=`` tier.

    Why this looks like "hardcoding model tiers" but isn't:

    The proxy's LMRH 1.0 scorer picks the **provider** based on dims, but
    when the chosen provider has a valid match for the request-body
    ``model=`` field, the proxy honors that exact model rather than
    substituting. (Confirmed empirically 2026-05-04: with
    ``cost=standard, fallback-chain=anthropic;require``, ``LLM-Capability``
    reported ``model=claude-sonnet-4-6, chosen-because=score`` — proxy
    chose Sonnet — but the actual model served was ``claude-haiku-4-5``
    because that's what we sent in the request body.) So the SDK
    request-body model effectively gates which model in the chosen
    provider we land on.

    To honor the LMRH spirit while working with how the proxy actually
    behaves, we map the *cost-tier* (which the LMRH preset declares per
    task) → the *cheapest model in that tier*. Code never names a model
    per task; it names a model per cost-tier, and the cost-tier is the
    LMRH dim. If Anthropic ships a cheaper Haiku tomorrow we update one
    line here, not 17 task entries.

    Cost-tier → model mapping (cheapest viable in tier):
      economy  → claude-haiku-4-5
      standard → claude-sonnet-4-6
      premium  → claude-opus-4-7

    Caller can still pass ``model=`` explicitly to override.
    """
    from analyzer.llm.lmrh import TASK_PRESETS
    tier = TASK_PRESETS.get(task, {}).get("cost", "economy")
    return {
        "economy":  "claude-haiku-4-5",
        "standard": "claude-sonnet-4-6",
        "premium":  "claude-opus-4-7",
    }.get(tier, fallback)


def _log_usage(usage_tracker, provider: str, model: str, operation: str,
                input_tokens: int, output_tokens: int,
                document_id: int | None = None, success: bool = True,
                endpoint_id: str | None = None) -> None:
    """Forward usage to LLMUsageTracker if provided. Ignores errors silently."""
    if not usage_tracker:
        return
    try:
        usage_tracker.log_usage(
            provider=provider,
            model=model,
            operation=operation,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            document_id=document_id,
            success=success,
        )
    except Exception as e:  # pragma: no cover — tracker errors must not break LLM flow
        logger.warning(f"[llm-usage] log_usage failed (endpoint={endpoint_id}): {e}")


def call_llm(
    messages: list[dict],
    *,
    task: str,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: Optional[float] = None,
    response_format: Optional[dict] = None,
    usage_tracker=None,
    operation: str = "call_llm",
    document_id: Optional[int] = None,
    project_slug: str = "default",
    model_pref: Optional[str] = None,
    fallback_chain: Optional[str] = None,
    quality: Optional[str] = None,
    has_images: bool = False,
    lmrh_extras: Optional[dict] = None,
    # Optional — if caller has a specific direct-provider pair to use when
    # the proxy pool is exhausted. Otherwise we fall back on OpenAI from
    # ai_config.json.
    direct_provider: Optional[str] = None,
    direct_api_key: Optional[str] = None,
    direct_model: Optional[str] = None,
    timeout: float = 90.0,
    # v3.9.21: when True, wrap the joined system content with
    # ``cache_control: {type: "ephemeral"}`` so claude-oauth caches the
    # stable prefix between calls. Default False (no cache) preserves
    # current semantics for callers without a stable prefix. Use this on
    # high-volume call sites that share the same system instructions
    # across many documents (analyze_document_integrity, etc.) — the
    # proxy team confirmed Anthropic Pro Max OAuth honors caller-side
    # cache_control regardless of the auto-cache token threshold.
    cache_system: bool = False,
) -> dict[str, Any]:
    """Send a chat/completion request through the proxy pool.

    Returns dict:
        {
            "content":      str,    # assistant message text
            "provider":     str,    # "llm-proxy" | "anthropic" | "openai"
            "model":        str,    # actual model used by upstream
            "endpoint_id":  str | None,  # proxy endpoint id, or None for direct
            "input_tokens": int,
            "output_tokens": int,
            "attempted":    list[str],   # diagnostic trail
        }
    """
    attempted: list[str] = []
    lmrh = build_lmrh_header(
        task,
        model_pref=model_pref,
        fallback_chain=fallback_chain,
        quality=quality,
        has_images=has_images,
        extras=lmrh_extras,
    )
    send_model = model or _default_model_for_task(task)

    # v3.9.18 emergency: pin EVERY call to the Anthropic /v1/messages path.
    # Previously the model-aware dispatch routed gpt-* requests to the
    # OpenAI-compat /v1/chat/completions endpoint, which hit the proxy's
    # OpenAI provider — bound to the operator's personal subscription.
    # Two days of bulk legal-review polling burned $151.43 of personal
    # credits before the proxy team flagged it (2026-05-02).
    #
    # Until paperless-ai-analyzer provisions its own paid OpenAI/Google
    # key, ALL traffic must use claude-oauth (free) via /v1/messages.
    # `_default_model_for_task()` was also flipped to claude-* defaults
    # so the request body's `model` field matches the routed provider.
    # If a caller explicitly passes a gpt-* model, the LMRH
    # `fallback-chain=anthropic;require` hint will cause the proxy to
    # reject the call rather than route to OpenAI.
    is_anthropic = True

    # ── Proxy pool path ────────────────────────────────────────────────
    if is_anthropic:
        # Native Anthropic SDK against /v1/messages — what claude-oauth wants.
        for ant_client, eid in proxy_manager.get_all_anthropic_clients(operation=task):
            label = f"proxy[{eid}]:messages"
            attempted.append(label)
            started = time.time()
            try:
                ant_messages = [m for m in messages if m.get("role") != "system"]
                system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
                ant_kwargs: dict[str, Any] = {
                    "model": send_model,
                    "max_tokens": max_tokens,
                    "messages": ant_messages,
                    "timeout": timeout,
                }
                if system_msgs:
                    if cache_system:
                        # v3.9.21: emit system as a content-blocks list with
                        # cache_control ephemeral on the joined stable prefix
                        # so claude-oauth caches it. Anthropic SDK accepts
                        # either a string or a list of TextBlockParam dicts
                        # for ``system``.
                        ant_kwargs["system"] = [{
                            "type": "text",
                            "text": "\n\n".join(system_msgs),
                            "cache_control": {"type": "ephemeral"},
                        }]
                    else:
                        ant_kwargs["system"] = "\n\n".join(system_msgs)
                if temperature is not None:
                    ant_kwargs["temperature"] = temperature
                # v3.9.19: capture LLM-Capability response header so we can
                # log what the proxy actually picked. Uses with_raw_response
                # so we get headers without losing the parsed message.
                raw = ant_client.messages.with_raw_response.create(**ant_kwargs)
                resp = raw.parse()
                cap_header = raw.headers.get("LLM-Capability") or ""
                content = resp.content[0].text if resp.content else ""
                in_tok = resp.usage.input_tokens
                out_tok = resp.usage.output_tokens
                model_used = getattr(resp, "model", send_model) or send_model

                # v3.9.20: cross-family-fallback guard. The proxy's v3.0.36
                # may substitute a different model family when the requested
                # one isn't available on the chosen provider (e.g. gpt-4o →
                # gpt-5.5 via codex-oauth). For legal-review work we cannot
                # silently accept a different model — fail fast and let the
                # caller decide. Detected via `chosen-because=cross-family-
                # fallback` in LLM-Capability per the proxy team's contract.
                if "cross-family-fallback" in cap_header:
                    proxy_manager.mark_failure(eid)
                    raise LLMUnavailableError(
                        f"Proxy substituted a different model family — "
                        f"requested={send_model}, capability={cap_header}. "
                        f"AI Analyzer fails fast on cross-family substitution "
                        f"so legal-review output isn't silently served by an "
                        f"unintended model. Pass `LLM-Hint: provider-hint=...;"
                        f"require` or provision a direct provider key.",
                        attempted=attempted,
                    )

                # v3.9.20: parse cache token counts. The proxy's v3.0.42 auto-
                # injects cache_control on stable system prefixes for
                # claude-oauth, so cache_creation_input_tokens /
                # cache_read_input_tokens land on the usage payload without
                # us doing anything. Surface them in the log so the operator
                # can see the savings without scraping /api/activity.
                cache_creation = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
                cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0

                proxy_manager.mark_success(eid)
                _log_usage(usage_tracker, "llm-proxy", model_used, operation,
                            in_tok, out_tok, document_id=document_id,
                            success=True, endpoint_id=eid)
                cache_part = ""
                if cache_creation or cache_read:
                    cache_part = f" cache_create={cache_creation} cache_read={cache_read}"
                logger.info(
                    f"[llm-proxy] {eid} (messages) model={model_used} task={task} "
                    f"in={in_tok} out={out_tok}{cache_part} "
                    f"{time.time()-started:.2f}s"
                    + (f" capability={cap_header}" if cap_header else "")
                )
                return {
                    "content": content,
                    "provider": "llm-proxy",
                    "model": model_used,
                    "endpoint_id": eid,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                    "attempted": attempted,
                }
            except _CONNECTION_EXCEPTIONS as e:
                logger.warning(f"[llm-proxy] {eid} (messages) connection error: {e}")
                proxy_manager.mark_failure(eid)
                continue
            except Exception as e:
                logger.warning(f"[llm-proxy] {eid} (messages) call failed: {str(e)[:200]}")
                proxy_manager.mark_failure(eid)
                continue
    else:
        # OpenAI-format dispatch via /v1/chat/completions — needs the proxy
        # to have an openai/google provider enabled.
        for client, eid in proxy_manager.get_all_clients():
            label = f"proxy[{eid}]:chat"
            attempted.append(label)
            started = time.time()
            try:
                kwargs: dict[str, Any] = {
                    "model": send_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "extra_headers": {"LLM-Hint": lmrh},
                    "timeout": timeout,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if response_format is not None:
                    kwargs["response_format"] = response_format
                resp = client.chat.completions.create(**kwargs)
                choice = resp.choices[0] if resp.choices else None
                content = (choice.message.content or "") if choice and choice.message else ""
                usage = getattr(resp, "usage", None)
                in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
                out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
                model_used = getattr(resp, "model", send_model) or send_model
                proxy_manager.mark_success(eid)
                _log_usage(usage_tracker, "llm-proxy", model_used, operation,
                            in_tok, out_tok, document_id=document_id,
                            success=True, endpoint_id=eid)
                logger.info(
                    f"[llm-proxy] {eid} model={model_used} task={task} "
                    f"in={in_tok} out={out_tok} {time.time()-started:.2f}s"
                )
                return {
                    "content": content,
                    "provider": "llm-proxy",
                    "model": model_used,
                    "endpoint_id": eid,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "attempted": attempted,
                }
            except _CONNECTION_EXCEPTIONS as e:
                logger.warning(f"[llm-proxy] {eid} connection error: {e} — trying next")
                proxy_manager.mark_failure(eid)
                continue
            except Exception as e:
                msg = str(e)
                logger.warning(f"[llm-proxy] {eid} call failed: {msg[:200]}")
                proxy_manager.mark_failure(eid)
                continue

    # ── Direct-provider last-resort path ───────────────────────────────
    logger.warning(
        f"[llm-proxy] all proxy endpoints exhausted or empty pool. "
        f"Falling through to direct-provider for task={task}"
    )

    resolved = _resolve_direct_provider(
        task=task,
        project_slug=project_slug,
        direct_provider=direct_provider,
        direct_api_key=direct_api_key,
        direct_model=direct_model,
    )
    if not resolved:
        raise LLMUnavailableError(
            "No proxy endpoint reachable and no direct-provider API key configured",
            attempted=attempted,
        )
    provider, api_key, dmodel = resolved

    try:
        if provider == "anthropic":
            return _direct_anthropic(messages, dmodel, api_key, max_tokens,
                                      temperature, usage_tracker, operation,
                                      document_id, attempted)
        elif provider == "openai":
            return _direct_openai(messages, dmodel, api_key, max_tokens,
                                   temperature, response_format,
                                   usage_tracker, operation, document_id,
                                   attempted)
    except Exception as e:
        attempted.append(f"direct[{provider}]:{type(e).__name__}")
        logger.error(f"[llm-direct] {provider} {dmodel} failed: {e}")
        raise LLMUnavailableError(
            f"All proxy endpoints failed and direct {provider} fallback also failed: {e}",
            attempted=attempted,
        )

    raise LLMUnavailableError(
        f"Unsupported direct provider: {provider}",
        attempted=attempted,
    )


def _resolve_direct_provider(
    *,
    task: str,
    project_slug: str,
    direct_provider: Optional[str],
    direct_api_key: Optional[str],
    direct_model: Optional[str],
) -> tuple[str, str, str] | None:
    """Figure out which (provider, api_key, model) to use for the
    direct-provider fallback. Returns None if no usable combination exists.
    """
    if direct_provider and direct_api_key and direct_model:
        return direct_provider, direct_api_key, direct_model

    # Fall back to ai_config: chat uses 'chat' use_case, others use document_analysis
    try:
        from analyzer.services.ai_config_service import get_project_ai_config
        use_case = "chat" if task in ("chat", "qa") else "document_analysis"
        cfg = get_project_ai_config(project_slug, use_case)
        provider = cfg.get("provider", "openai")
        api_key = cfg.get("api_key", "")
        dmodel = cfg.get("model") or ("gpt-4o" if provider == "openai" else "claude-sonnet-4-5-20250929")
        if api_key:
            return provider, api_key, dmodel
    except Exception as e:
        logger.warning(f"[llm-direct] ai_config lookup failed: {e}")
    return None


def _direct_anthropic(messages, model, api_key, max_tokens, temperature,
                       usage_tracker, operation, document_id, attempted):
    import anthropic
    attempted.append(f"direct[anthropic:{model}]")
    client = anthropic.Anthropic(api_key=api_key)
    # Anthropic SDK differs: no 'system' role in messages — strip to system parameter.
    system_prompt = None
    msgs = []
    for m in messages:
        if m.get("role") == "system":
            system_prompt = (system_prompt + "\n\n" if system_prompt else "") + (m.get("content") or "")
        else:
            msgs.append(m)
    kw = {"model": model, "max_tokens": max_tokens, "messages": msgs}
    if system_prompt:
        kw["system"] = system_prompt
    if temperature is not None:
        kw["temperature"] = temperature
    resp = client.messages.create(**kw)
    content = resp.content[0].text if resp.content else ""
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) if usage else 0
    out_tok = getattr(usage, "output_tokens", 0) if usage else 0
    _log_usage(usage_tracker, "anthropic", model, operation,
                in_tok, out_tok, document_id=document_id, success=True)
    logger.info(f"[llm-direct] anthropic {model} in={in_tok} out={out_tok}")
    return {
        "content": content,
        "provider": "anthropic",
        "model": model,
        "endpoint_id": None,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "attempted": attempted,
    }


def call_llm_json(
    prompt: str,
    *,
    task: str,
    max_tokens: int = 4096,
    operation: str = "ci_call",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    document_id: Optional[int] = None,
    usage_tracker=None,
    project_slug: str = "default",
    extra_messages: Optional[list[dict]] = None,
) -> Optional[dict]:
    """Shorthand helper for Case Intelligence submodules: single user prompt,
    JSON response, parsed into a dict. Returns None if any step fails.

    Handles:
      - Building the [{'role':'user','content':prompt}] message list
      - Routing through call_llm (proxy-first, direct-provider fallback)
      - Stripping ```json code fences
      - Parsing JSON, returning None on parse failure

    The `provider`, `api_key`, `model` args become direct-provider hints for
    the last-resort fallback path. LMRH `model-pref` is driven by `task`.
    """
    import json as _json
    messages = list(extra_messages or [])
    messages.append({"role": "user", "content": prompt})

    try:
        result = call_llm(
            messages=messages,
            task=task,
            max_tokens=max_tokens,
            usage_tracker=usage_tracker,
            operation=operation,
            document_id=document_id,
            project_slug=project_slug,
            direct_provider=provider,
            direct_api_key=api_key,
            direct_model=model,
            response_format={"type": "json_object"} if provider == "openai" else None,
        )
    except LLMUnavailableError as e:
        logger.warning(f"{operation}: LLM unavailable: {e}")
        return None
    except Exception as e:  # pragma: no cover
        logger.error(f"{operation}: unexpected LLM error: {e}")
        return None

    text = result.get("content") or ""
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        return _json.loads(text)
    except _json.JSONDecodeError as e:
        logger.warning(f"{operation}: JSON parse error: {e} text[:200]={text[:200]!r}")
        return None


def _direct_openai(messages, model, api_key, max_tokens, temperature,
                    response_format, usage_tracker, operation,
                    document_id, attempted):
    import openai
    attempted.append(f"direct[openai:{model}]")
    client = openai.OpenAI(api_key=api_key)
    kw: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kw["temperature"] = temperature
    if response_format is not None:
        kw["response_format"] = response_format
    resp = client.chat.completions.create(**kw)
    choice = resp.choices[0] if resp.choices else None
    content = (choice.message.content or "") if choice and choice.message else ""
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
    _log_usage(usage_tracker, "openai", model, operation,
                in_tok, out_tok, document_id=document_id, success=True)
    logger.info(f"[llm-direct] openai {model} in={in_tok} out={out_tok}")
    return {
        "content": content,
        "provider": "openai",
        "model": model,
        "endpoint_id": None,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "attempted": attempted,
    }
