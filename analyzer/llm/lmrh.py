"""
LLM Model Routing Hint (LMRH) header builder.

Spec: /home/dblagbro/llm-proxy-v2/v1-reference/LMRH-PROTOCOL.md
Format: RFC 8941 structured dictionary, semicolon-separated key=value pairs.

Emit via `extra_headers={'LLM-Hint': build_lmrh_header(...)}` on OpenAI SDK calls.
llm-proxy2 parses this header and uses it to score provider candidates. Unknown
dimensions are ignored (soft preference) unless marked `;require`.
"""

from typing import Optional


# Task presets per call-site in paperless-ai-analyzer. Keep this table in sync
# with the plan at /home/dblagbro/.claude/plans/linear-prancing-squirrel.md.
TASK_PRESETS: dict[str, dict] = {
    # Core anomaly + extraction
    "analysis":         {"model_pref": "claude-sonnet-4-6", "fallback_chain": "anthropic,openai"},
    "chat":             {"model_pref": "claude-sonnet-4-6"},
    "qa":               {},
    "extraction":       {"model_pref": "gpt-4o"},
    "classification":   {"model_pref": "gpt-4o-mini"},

    # Case Intelligence specialists
    "reasoning":        {"model_pref": "claude-opus-4-7", "quality": "high"},
    "entity":           {"model_pref": "claude-sonnet-4-6"},
    "timeline":         {"model_pref": "claude-sonnet-4-6"},
    "financial":        {"model_pref": "claude-sonnet-4-6"},
    "contradiction":    {"model_pref": "claude-sonnet-4-6"},
    "theory":           {"model_pref": "claude-opus-4-7", "quality": "high"},
    "forensic":         {"model_pref": "claude-sonnet-4-6", "quality": "high"},
    "discovery":        {"model_pref": "claude-sonnet-4-6", "quality": "high"},
    "witness":          {"model_pref": "claude-sonnet-4-6", "quality": "high"},
    "warroom":          {"model_pref": "claude-opus-4-7", "quality": "high"},
    "report":           {"model_pref": "claude-opus-4-7", "quality": "high"},
    "settlement":       {"model_pref": "claude-opus-4-7", "quality": "high"},
}


def build_lmrh_header(
    task: str,
    *,
    model_pref: Optional[str] = None,
    fallback_chain: Optional[str] = None,
    quality: Optional[str] = None,
    has_images: bool = False,
    extras: Optional[dict] = None,
) -> str:
    """Build the LLM-Hint header value.

    Arguments:
        task: semantic label (e.g. "chat", "analysis", "reasoning"). Looked up
              in TASK_PRESETS to fill model_pref / quality defaults. Explicit
              kwargs override presets.
        model_pref: soft preference for a specific provider model name
        fallback_chain: comma-separated provider preference order
        quality: "low" | "standard" | "high" — provider quality tier hint
        has_images: if True, adds `modality=vision`
        extras: dict of extra key=value pairs appended verbatim

    Returns:
        A string like `task=chat; model-pref=claude-sonnet-4-6` suitable for
        the `LLM-Hint` HTTP header. Callers attach it via
        `extra_headers={'LLM-Hint': ...}`.
    """
    preset = TASK_PRESETS.get(task, {})
    pref = model_pref or preset.get("model_pref")
    fb = fallback_chain or preset.get("fallback_chain")
    q = quality or preset.get("quality")

    parts: list[str] = [f"task={task}"]
    if pref:
        parts.append(f"model-pref={pref}")
    if fb:
        parts.append(f"fallback-chain={fb}")
    if q:
        parts.append(f"quality={q}")
    if has_images:
        parts.append("modality=vision")
    if extras:
        for k, v in extras.items():
            # Normalise key to LMRH style (lowercase, hyphen-separated)
            key = str(k).lower().replace("_", "-")
            parts.append(f"{key}={v}")

    return "; ".join(parts)
