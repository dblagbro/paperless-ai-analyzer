"""LMRH (LLM Model Routing Hint) header builder for AI Analyzer.

Spec: https://www.voipguru.org/llm-proxy2/lmrh.md (v1.0 RFC draft)
Local copy: /home/dblagbro/llm-proxy-v2/v1-reference/LMRH-PROTOCOL.md
Format: RFC 8941 structured field list, comma-separated `dim=value` pairs.

Emit via ``extra_headers={'LLM-Hint': build_lmrh_header(...)}`` on the OpenAI
SDK, or via ``default_headers={'LLM-Hint': ...}`` when constructing the
Anthropic client (used by Case Intelligence director-tier calls that need
prompt caching). llm-proxy2 parses this header and uses it to score
provider candidates. Unknown dims are ignored (soft preference) unless
marked ``;require``.

**Ops directive (2026-04-30, ported from tax-ai 2026-05-01):**
  - Do NOT hardcode model names per call-site. Let the proxy pick the best
    model+provider based on ``task=`` + ``cost=`` + ``safety-min=`` +
    ``context-length=``.
  - If Anthropic ships a cheaper Haiku tomorrow, the proxy auto-picks it.
    AI Analyzer code does not change.

**Backwards compatibility:** the builder still accepts the legacy
``model_pref=`` and ``fallback_chain=`` kwargs that were used by paperless
v3.9.0. They emit the deprecated ``model-pref`` / ``fallback-chain`` dims;
the proxy honors them as soft preferences. New code should prefer the
dim-based ``cost=`` / ``safety_min=`` / ``context_length=`` parameters.

Recognized LMRH 1.0 dims:
  task            chat | reasoning | analysis | code | creative | audio
                  | vision | summarize | classify | extract
  cost            economy | standard | premium
  latency         low | medium | high
  safety-min      1..5         (with optional `;require` for hard constraint)
  safety-max      1..5
  context-length  positive int (token count needed)
  modality        text | vision | audio
  region          us | eu | asia | <ISO-3166-1 alpha-2>
  refusal-rate    permissive | standard | strict | maximum
"""

from __future__ import annotations

from typing import Optional


# Per-task default preset.
#
# v3.9.19 cost-discipline rebalance after the 2026-05-02 burn incident:
# **every task that runs in a high-volume loop is `cost=economy`**.
# Only multi-document synthesis tasks the operator explicitly invokes
# (Case Intelligence director-tier, Tier-5 reports) escalate to
# `cost=standard`. No task escalates to `cost=premium` by default —
# operators can opt in via the explicit kwarg if a specific report
# truly needs Opus-class quality.
#
# Why economy is fine for analysis: the doc-analysis pipeline runs over
# *every* document the user uploads. At 6 calls/min sustained on the
# poller, even sonnet-class pricing is meaningful spend. Haiku-class
# is plenty for "extract amount/date/category/tags from a financial
# document". The LMRH `task=analysis, cost=economy` hint tells the
# proxy "give me the cheapest provider that's solid at analysis" —
# typically claude-haiku, which costs ~10× less than sonnet.
#
# safety-min=3 is set on document analysis paths — these often handle
# legal documents and we want providers with reasonable refusal floors.
TASK_PRESETS: dict[str, dict] = {
    # ── Core anomaly + document analysis pipeline (high-volume polling) ──
    "analysis":         {"cost": "economy", "safety-min": 3},
    "extraction":       {"cost": "economy"},
    "classification":   {"cost": "economy"},
    "summarize":        {"cost": "economy"},

    # ── Chat + Q&A (interactive, latency matters more than depth) ────
    "chat":             {"cost": "economy"},
    "qa":               {"cost": "economy"},

    # ── Case Intelligence specialists (per-doc, high-volume) ─────────
    "entity":           {"cost": "economy"},
    "timeline":         {"cost": "economy"},
    "financial":        {"cost": "economy"},
    "contradiction":    {"cost": "economy"},

    # ── Director / synthesis tier — operator-invoked, multi-doc input ─
    # These run a small number of times per case and synthesise across
    # everything the specialists produced. Sonnet-class quality is the
    # right tradeoff. Long context-length tells the proxy to skip
    # short-context models even if they'd score higher otherwise.
    "reasoning":        {"cost": "standard", "context-length": 80000},
    "theory":           {"cost": "standard", "context-length": 60000},
    "warroom":          {"cost": "standard", "context-length": 80000},
    "report":           {"cost": "standard", "context-length": 60000},
    "settlement":       {"cost": "standard"},

    # ── Specialist analysts — operator-invoked drill-downs ───────────
    "forensic":         {"cost": "standard", "context-length": 60000},
    "discovery":        {"cost": "standard", "context-length": 60000},
    "witness":          {"cost": "standard"},

    # ── Vision / embeddings ──────────────────────────────────────────
    "vision":           {"cost": "economy"},
    "embed":            {"cost": "economy"},
}


def build_lmrh_header(
    task: str,
    *,
    cost: Optional[str] = None,
    quality: Optional[str] = None,
    safety_min: Optional[int] = None,
    context_length: Optional[int] = None,
    has_images: bool = False,
    extras: Optional[dict] = None,
    # v3.9.18 emergency: when True (default), append `fallback-chain=anthropic
    # ;require` so the proxy hard-rejects any routing decision that would land
    # on a non-Anthropic provider. Prevents inadvertent OpenAI/Google billing
    # to the operator's personal subscription. Set False ONLY if AI Analyzer
    # has provisioned its own paid OpenAI/Google key on the proxy.
    pin_to_anthropic: bool = True,
    # ── Backwards-compat kwargs (paperless v3.9.0 era) ──────────────
    model_pref: Optional[str] = None,
    fallback_chain: Optional[str] = None,
) -> str:
    """Build the LLM-Hint header value.

    Arguments:
        task: cognitive type — one of the LMRH ``task=`` tokens. Looked up
              in TASK_PRESETS to fill cost/safety-min/context-length defaults.
              Explicit kwargs override presets.
        cost: economy | standard | premium. Defaults to preset["cost"].
        quality: low | standard | high. Optional pass-through.
        safety_min: 1..5. Hard constraint via ``;require``.
        context_length: int — minimum tokens of context the model must support.
        has_images: if True, adds ``modality=vision``.
        extras: dict of extra dim=value pairs appended verbatim.
        model_pref: (deprecated, soft) specific model name preference.
        fallback_chain: (deprecated, soft) comma-separated provider order.

    Returns:
        A string like ``task=analysis, cost=standard, safety-min=3`` suitable
        for the LLM-Hint header. Empty string if ``task`` is empty.

    Per the LMRH 1.0 spec, unknown dims are ignored by the proxy. We keep
    things conservative — emit only well-formed dims.
    """
    if not task:
        return ""

    preset = TASK_PRESETS.get(task, {})
    eff_cost = cost or preset.get("cost")
    eff_ctx = context_length if context_length is not None else preset.get("context-length")
    eff_safety = safety_min if safety_min is not None else preset.get("safety-min")

    parts: list[str] = [f"task={task}"]
    if eff_cost:
        parts.append(f"cost={eff_cost}")
    if quality:
        parts.append(f"quality={quality}")
    if eff_safety is not None:
        parts.append(f"safety-min={int(eff_safety)}")
    if eff_ctx is not None:
        parts.append(f"context-length={int(eff_ctx)}")
    if has_images:
        parts.append("modality=vision")
    if model_pref:
        parts.append(f"model-pref={model_pref}")
    if fallback_chain:
        parts.append(f"fallback-chain={fallback_chain}")
    elif pin_to_anthropic:
        # Hard-constraint: proxy MUST route to an Anthropic provider, no
        # exceptions. Free claude-oauth subscription only. Without this,
        # default-model fallbacks to gpt-* would hit the operator's
        # personal OpenAI subscription. v3.9.18 emergency.
        parts.append("fallback-chain=anthropic;require")
    if extras:
        for k, v in extras.items():
            key = str(k).lower().replace("_", "-")
            parts.append(f"{key}={v}")

    # Spec calls for comma+space separator on the structured-field list.
    return ", ".join(parts)


def get_hint(operation_or_task: str, **overrides) -> str:
    """Convenience: look up an operation/task name and return the hint string.

    First checks for an operator override stored in db settings under
    ``lmrh.hint.<task>`` (mirrors the coordinator-hub get_lmrh_hint pattern),
    then falls through to ``build_lmrh_header(task=...)``. This lets the
    operator tune routing per task without a code change.
    """
    try:
        from analyzer import db as _db
        if hasattr(_db, "get_setting"):
            ovr = (_db.get_setting(f"lmrh.hint.{operation_or_task}") or "").strip()
            if ovr:
                return ovr
    except Exception:
        pass
    return build_lmrh_header(operation_or_task, **overrides)


def list_tasks() -> list[str]:
    """Task names with built-in presets (used by admin UI / docs)."""
    return sorted(TASK_PRESETS.keys())
