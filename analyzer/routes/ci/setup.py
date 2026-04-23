"""CI setup routes (status, jurisdictions, cost estimate, authority, key guide).

Extracted from routes/ci.py during the v3.9.3 maintainability refactor.
"""
import json
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, session, make_response
from flask_login import login_required, current_user

from analyzer.app import admin_required, advanced_required, _ci_gate, _ci_can_read, _ci_can_write
from analyzer.db import get_user_by_id, get_user_by_username
from analyzer.services.ai_config_service import load_ai_config, get_project_ai_config
from analyzer.services.smtp_service import (
    load_smtp_settings as _load_smtp_settings,
    smtp_send as _smtp_send,
)

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# CI Notification helpers
# ---------------------------------------------------------------------------

from analyzer.routes.ci import bp
from analyzer.routes.ci.helpers import (
    _send_ci_budget_notification,
    _send_ci_complete_notification,
    _match_jurisdiction_profile,
    _ci_elapsed_seconds,
    _build_ci_llm_clients,
)

@bp.route('/api/ci/status')
@login_required
def ci_status():
    """Feature status + authority corpus stats."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_authority_corpus_stats
        from analyzer.case_intelligence.job_manager import get_job_manager
        corpus_stats = get_authority_corpus_stats()
        active_runs = get_job_manager().list_active_runs()
        return jsonify({
            'enabled': True,
            'authority_corpus': corpus_stats,
            'active_runs': len(active_runs),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/jurisdictions')
@login_required
def ci_jurisdictions():
    """List pre-built jurisdiction profiles."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.jurisdiction import list_jurisdiction_profiles
        return jsonify({'jurisdictions': list_jurisdiction_profiles()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/detect-jurisdiction', methods=['POST'])
@login_required
def ci_detect_jurisdiction():
    """Detect court jurisdiction from project documents using LLM."""
    from flask import current_app
    ok, err = _ci_gate()
    if not ok:
        return err
    import json as _json
    try:
        project_slug = session.get('current_project', 'default')
        from analyzer.vector_store import VectorStore
        from analyzer.case_intelligence.jurisdiction import JURISDICTION_PROFILES

        vs = VectorStore(project_slug=project_slug)
        if not vs.enabled:
            return jsonify({'detected': False, 'reason': 'Vector store not enabled for this project'})

        # Search for jurisdiction-relevant document excerpts
        results = vs.search(
            "court supreme district county filed plaintiff defendant address jurisdiction caption index number",
            n_results=8
        )
        if not results:
            return jsonify({'detected': False, 'reason': 'No documents found in project'})

        # Build document excerpts (first 500 chars each)
        excerpts = []
        for r in results[:6]:
            title = r.get('title', 'Untitled')
            content = (r.get('content') or r.get('document') or '')[:500].strip()
            if content:
                excerpts.append(f"[{title}]\n{content}")
        if not excerpts:
            return jsonify({'detected': False, 'reason': 'Documents found but content unavailable'})
        excerpts_text = "\n\n---\n\n".join(excerpts)

        # Use the app's existing LLM client
        llm = getattr(current_app, 'llm_client', None)
        if not llm or not llm.client:
            return jsonify({'detected': False, 'reason': 'LLM client not available'})

        prompt = (
            "You are analyzing legal documents to identify their court jurisdiction.\n\n"
            "Examine these document excerpts and identify:\n"
            "- The specific court (e.g. 'NYS Supreme Court', 'S.D.N.Y.', 'E.D.N.Y. Bankruptcy Court')\n"
            "- The county (for NY state cases, e.g. 'Kings', 'New York', 'Queens')\n"
            "- The state (two-letter code, e.g. 'NY')\n"
            "- Whether it is a federal case (SDNY, EDNY, etc.)\n"
            "- Whether it is a bankruptcy case, and if so, chapter number (7 or 11)\n"
            "- Whether it is a Family Court or Surrogate's Court case\n"
            "- Addresses of parties (to confirm location)\n\n"
            "Document excerpts:\n"
            f"{excerpts_text[:3500]}\n\n"
            "Respond with ONLY valid JSON, no other text:\n"
            '{"court_name": "string or null", "county": "string or null", '
            '"state": "NY", "is_federal": false, "is_bankruptcy": false, '
            '"bankruptcy_chapter": null, "is_family_court": false, '
            '"is_surrogate": false, "confidence": 0.85, '
            '"reasoning": "one sentence explanation"}'
        )

        from analyzer.llm.proxy_call import call_llm, LLMUnavailableError
        try:
            result = call_llm(
                messages=[{'role': 'user', 'content': prompt}],
                task='classification',
                max_tokens=350,
                temperature=0,
                response_format={'type': 'json_object'} if llm.provider == 'openai' else None,
                direct_provider=llm.provider,
                direct_api_key=llm.api_key,
                direct_model='gpt-4o-mini' if llm.provider == 'openai' else 'claude-haiku-4-5-20251001',
                operation='ci_jurisdiction_detect',
            )
            raw = result['content']
        except LLMUnavailableError as e:
            logger.warning(f"ci/detect-jurisdiction: {e}")
            return jsonify({'detected': False, 'reason': 'LLM unavailable'}), 503

        if not raw or not raw.strip():
            return jsonify({'detected': False, 'reason': 'LLM returned an empty response'})

        # Strip markdown code fences if the model wrapped the JSON
        raw_stripped = raw.strip()
        if raw_stripped.startswith('```'):
            raw_stripped = raw_stripped.split('\n', 1)[-1]
            raw_stripped = raw_stripped.rsplit('```', 1)[0].strip()

        extracted = _json.loads(raw_stripped)
        profile_id = _match_jurisdiction_profile(extracted)
        profile = JURISDICTION_PROFILES.get(profile_id)

        return jsonify({
            'detected': True,
            'jurisdiction_id': profile_id,
            'display_name': profile.display_name if profile else 'Custom',
            'court': profile.court if profile else (extracted.get('court_name') or ''),
            'county': extracted.get('county'),
            'confidence': extracted.get('confidence', 0.5),
            'reason': extracted.get('reasoning', ''),
        })

    except Exception as e:
        logger.error(f"CI detect-jurisdiction error: {e}", exc_info=True)
        return jsonify({'detected': False, 'reason': f'Detection failed: {str(e)}'})


@bp.route('/api/ci/goal-assistant', methods=['POST'])
@login_required
@advanced_required
def ci_goal_assistant():
    """Lightweight AI chat that helps the user write a focused CI goal statement.
    Stateless — caller maintains conversation history in the browser.
    """
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.routes.chat import _fetch_url_text
        data = request.json or {}
        messages = data.get('messages', [])          # [{role, content}, ...]
        ctx = data.get('context', {})
        role = ctx.get('role', 'neutral')
        jurisdiction = ctx.get('jurisdiction', 'Not specified')
        draft_goal = ctx.get('draft_goal', '').strip()

        # Fetch any URLs the user included in their messages so the AI can reference them
        import re as _re_goal
        _goal_url_pat = _re_goal.compile(r'https?://[^\s<>"\']+', _re_goal.IGNORECASE)
        _goal_url_context = ''
        _seen_urls = set()
        for _msg in messages:
            for _u in _goal_url_pat.findall(_msg.get('content', ''))[:2]:
                if _u not in _seen_urls:
                    _seen_urls.add(_u)
                    _txt, _err = _fetch_url_text(_u, max_chars=3000)
                    if _txt:
                        _goal_url_context += f"\n\n--- Content from {_u} ---\n{_txt}"
                    else:
                        _goal_url_context += f"\n\n[Could not fetch {_u}: {_err}]"

        system_prompt = (
            "You are a legal case strategy advisor helping an attorney write a clear, focused "
            "goal statement for a document-analysis AI called Case Intelligence.\n\n"
            "Case Intelligence will analyze legal documents to:\n"
            "- Extract people, organizations, accounts, and key properties\n"
            "- Build a chronological timeline of events\n"
            "- Detect financial flows and amounts\n"
            "- Find contradictions between documents\n"
            "- Generate factual and legal theories\n"
            "- Identify relevant legal authorities\n\n"
            "A great goal statement tells Case Intelligence EXACTLY what to find:\n"
            "  - Who the client represents and the core legal dispute\n"
            "  - What specific evidence or patterns to surface\n"
            "  - What outcome the attorney needs (support damages, build defense, find smoking-gun)\n\n"
            f"CURRENT SETUP:\n"
            f"  Role: {role}\n"
            f"  Jurisdiction: {jurisdiction}\n"
        )
        if draft_goal:
            system_prompt += f"  Draft goal: \"{draft_goal}\"\n"
        if _goal_url_context:
            system_prompt += f"\nWEB CONTENT PROVIDED BY USER:{_goal_url_context}\nUse the above web content when relevant to help craft the goal statement.\n"
        system_prompt += (
            "\nINSTRUCTIONS:\n"
            "1. On the first turn, briefly acknowledge the context and ask 2-3 targeted "
            "questions to understand the case better. Be concise.\n"
            "2. After you have enough information, produce a polished goal statement.\n"
            "3. When you are ready to suggest a goal, end your message with exactly:\n"
            "Suggested Goal:\n[goal text — 2-4 sentences, specific and actionable]\n"
            "4. Only produce ONE suggested goal block per message. Keep the rest of your "
            "message brief.\n"
            "5. If the user wants changes, revise the suggested goal in the same format."
        )

        project_slug = session.get('current_project', 'default')
        chat_cfg = get_project_ai_config(project_slug, 'chat')

        # Synthetic opener when browser sends empty message list
        if not messages:
            messages = [{'role': 'user', 'content': 'Hello, I need help writing a focused goal statement for my Case Intelligence analysis.'}]

        from analyzer.llm.proxy_call import call_llm, LLMUnavailableError
        try:
            result = call_llm(
                messages=[{'role': 'system', 'content': system_prompt}] + messages,
                task='chat',
                max_tokens=600,
                project_slug=project_slug,
                operation='ci_goal_assistant',
                direct_provider=chat_cfg.get('provider'),
                direct_api_key=chat_cfg.get('api_key'),
                direct_model=chat_cfg.get('model'),
            )
            reply = result['content']
        except LLMUnavailableError as e:
            logger.warning(f"ci/goal-assistant: {e}")
            return jsonify({'error': 'No AI provider available — check Configuration → AI Settings.',
                             'source': 'llm-pool-exhausted'}), 503

        # Extract suggested goal if present
        suggested_goal = None
        marker = 'Suggested Goal:'
        if marker in reply:
            suggested_goal = reply.split(marker, 1)[1].strip()

        return jsonify({'response': reply, 'suggested_goal': suggested_goal})
    except Exception as e:
        logger.error(f"CI goal assistant error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/key-guide', methods=['POST'])
@login_required
def ci_key_guide():
    """Conversational AI agent to help users obtain and activate API keys for CI web research services."""
    import json as _json

    CREDENTIALS_MAP = {
        'brave':         {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'gcse':          {'fields': {'key': 'API Key', 'cx': 'Custom Search Engine ID (CX)'}, 'creds_desc': 'two credentials: an API Key (from https://console.cloud.google.com/apis/credentials) and a Custom Search Engine ID / CX (from https://programmablesearchengine.google.com/controlpanel/all → your engine → Overview page)'},
        'exa':           {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'perplexity':    {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'tavily':        {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'serper':        {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'fec':           {'fields': {'key': 'API key'}, 'creds_desc': 'one API key (email signup at api.data.gov)'},
        'opensanctions': {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'opencorp':      {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'docket':        {'fields': {'user': 'username', 'pass': 'password'}, 'creds_desc': 'username and password'},
        'unicourt':      {'fields': {'client_id': 'Client ID', 'client_secret': 'Client Secret'}, 'creds_desc': 'OAuth2 Client ID and Client Secret'},
        'newsapi':       {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'courtlistener': {'fields': {'key': 'API token'}, 'creds_desc': 'an API token (or leave blank for unauthenticated access)'},
    }

    # Service-specific setup notes injected into system prompt
    SETUP_NOTES = {
        'gcse': (
            "GOOGLE CUSTOM SEARCH — CRITICAL SETUP NOTES:\n"
            "1. The user needs TWO things: an API Key AND a Search Engine ID (CX).\n"
            "2. API Key: go to https://console.cloud.google.com/apis/credentials → Create Credentials → API Key. "
            "The Programmable Search JSON API must be enabled at https://console.cloud.google.com/apis/library/customsearch.googleapis.com\n"
            "3. Search Engine ID (CX): go to https://programmablesearchengine.google.com/controlpanel/all → "
            "create or open an engine → the CX/Search engine ID is shown on the Overview page.\n"
            "4. TO SEARCH THE ENTIRE WEB: In the engine control panel, open your engine → Edit → "
            "'Basics' or 'Setup' tab. Look for a toggle or radio button labelled 'Search the entire web' "
            "or 'Search the entire web but emphasize included sites'. This is a section-level toggle SEPARATE "
            "from the 'Sites to search' list — do NOT try to add * or any wildcard as a site pattern, that will fail. "
            "IMPORTANT: As of 2025 Google has been deprecating the whole-web PSE feature. "
            "If the toggle is absent or greyed out, the PSE engine is SITE-RESTRICTED ONLY. "
            "In that case, honest advice is: use Brave Search API (https://brave.com/search/api/) or "
            "Serper.dev (https://serper.dev/) instead — both search the full web without domain restrictions, "
            "are much simpler to set up, and have free tiers.\n"
            "5. Walk the user through BOTH credentials one at a time. Ask for the API Key first, then the CX.\n"
            "6. Always use full clickable URLs when directing the user somewhere."
        ),
    }

    SERVICE_URLS = {
        'brave':         'https://brave.com/search/api/',
        'gcse':          'https://programmablesearchengine.google.com/controlpanel/create',
        'exa':           'https://dashboard.exa.ai/',
        'perplexity':    'https://www.perplexity.ai/settings/api',
        'tavily':        'https://app.tavily.com/home',
        'serper':        'https://serper.dev/',
        'fec':           'https://api.data.gov/signup/',
        'opensanctions': 'https://www.opensanctions.org/api/',
        'opencorp':      'https://opencorporates.com/api_accounts/new',
        'docket':        'https://www.docketalarm.com/accounts/register/',
        'unicourt':      'https://unicourt.com/api',
        'newsapi':       'https://newsapi.org/register',
        'courtlistener': 'https://www.courtlistener.com/sign-in/',
    }

    try:
        data = request.json or {}
        service = (data.get('service') or '').strip()
        service_name = (data.get('service_name') or service).strip()
        messages_in = data.get('messages') or []
        if not service:
            return jsonify({'error': 'service required'}), 400

        svc_info = CREDENTIALS_MAP.get(service)
        creds_desc = svc_info['creds_desc'] if svc_info else 'the required credentials'
        fields = svc_info['fields'] if svc_info else {'key': 'API key'}
        url = SERVICE_URLS.get(service, '')

        # Build json_fields_example for CREDENTIALS_READY signal
        json_fields_example = _json.dumps({k: f'<{v}>' for k, v in fields.items()})

        extra_notes = SETUP_NOTES.get(service, '')
        system_prompt = (
            f"You are an AI assistant embedded in a legal document analysis platform. "
            f"Your sole job right now is to help the user obtain and activate their {service_name} API key.\n\n"
            f"RULES:\n"
            f"- Be conversational and direct. Maximum 3 sentences per reply.\n"
            f"- Ask ONE question or give ONE instruction at a time — never dump a full list.\n"
            f"- Guide them step by step: first ask if they have an account, then walk through account creation or key generation as needed.\n"
            f"- Always provide full URLs as plain text so the user can click them.\n"
            f"- Registration page: {url}\n"
            f"- This service requires: {creds_desc}\n"
            + (f"\n{extra_notes}\n" if extra_notes else '')
            + f"- When the user gives you real credential value(s), repeat them back to confirm, then output EXACTLY this on its own line (no other text after it):\n"
            f"  CREDENTIALS_READY:{json_fields_example}\n"
            f"- CRITICAL: Only output CREDENTIALS_READY when you have ALL required real values from the user — never with placeholder text.\n"
            f"- After outputting CREDENTIALS_READY, say \"Done! I've filled in your credentials and enabled this source.\"\n"
            f"- Do not tell the user to \"paste it into the form\" — you will do that automatically."
        )

        chat_cfg = get_project_ai_config(
            session.get('current_project', 'default'), 'chat'
        )
        _full_cfg = load_ai_config()

        def _global_key(p):
            return _full_cfg.get('global', {}).get(p, {}).get('api_key', '').strip()

        # Build message list for LLM
        llm_messages = list(messages_in)  # [{role, content}, ...]
        if not llm_messages:
            llm_messages = [{'role': 'user', 'content': 'Hello'}]

        from analyzer.llm.proxy_call import call_llm, LLMUnavailableError
        try:
            result = call_llm(
                messages=[{'role': 'system', 'content': system_prompt}] + llm_messages,
                task='settlement',
                max_tokens=600,
                project_slug=session.get('current_project', 'default'),
                operation='ci_key_guide',
                direct_provider=chat_cfg.get('provider'),
                direct_api_key=chat_cfg.get('api_key'),
                direct_model=chat_cfg.get('model'),
            )
            raw_reply = result['content']
        except LLMUnavailableError as e:
            logger.warning(f"ci/key-guide: {e}")
            return jsonify({'error': 'No AI provider available — check Configuration → AI Settings.',
                             'source': 'llm-pool-exhausted'}), 503

        # Parse CREDENTIALS_READY signal from response
        credentials = None
        display_lines = []
        after_ready = []
        found_ready = False
        for line in raw_reply.splitlines():
            stripped = line.strip()
            if stripped.startswith('CREDENTIALS_READY:'):
                found_ready = True
                json_str = stripped[len('CREDENTIALS_READY:'):]
                try:
                    credentials = _json.loads(json_str)
                    # Filter out placeholder values
                    placeholders = {'<the actual key>', '<the actual cx>', '<username>', '<password>',
                                    '<client_id>', '<client_secret>', '<api token>'}
                    if any(v.lower().startswith('<') and v.lower().endswith('>') for v in credentials.values()):
                        credentials = None
                except Exception:
                    credentials = None
            elif found_ready:
                after_ready.append(line)
            else:
                display_lines.append(line)

        # Build display text: lines before CREDENTIALS_READY + any lines after it
        display_text = '\n'.join(display_lines).strip()
        if after_ready:
            suffix = '\n'.join(after_ready).strip()
            if suffix:
                display_text = (display_text + '\n' + suffix).strip()

        return jsonify({'response': display_text, 'credentials': credentials})
    except Exception as e:
        logger.error(f"CI key guide error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/cost-estimate')
@login_required
def ci_cost_estimate():
    """
    Return estimated cost for a CI run.
    Query params: docs=N&tier=N
    """
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        docs = max(1, int(request.args.get('docs', 10)))
        tier = max(1, min(5, int(request.args.get('tier', 3))))
        from analyzer.case_intelligence.task_registry import estimate_run_cost, TIER_INFO
        estimate = estimate_run_cost(docs, tier)
        tier_meta = TIER_INFO.get(tier, {})
        return jsonify({
            'estimated_usd': estimate['total_usd'],
            'breakdown_by_task': estimate['breakdown_by_task'],
            'tier': tier,
            'tier_name': tier_meta.get('name', f'Tier {tier}'),
            'docs': docs,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@bp.route('/api/ci/authority/ingest', methods=['POST'])
@login_required
@admin_required
def ci_ingest_authorities():
    """Trigger authority corpus ingestion (admin only)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import os as _os
        import threading as _threading
        from analyzer.case_intelligence.authority_ingester import AuthorityIngester
        from analyzer.case_intelligence.authority_retriever import AuthorityRetriever
        from analyzer.case_intelligence.db import init_ci_db

        init_ci_db()
        data = request.json or {}
        sources = data.get('sources', ['nysenate', 'ecfr', 'courtlistener'])

        def _ingest():
            ingester = AuthorityIngester(
                courtlistener_token=_os.environ.get('COURTLISTENER_API_TOKEN'),
                nysenate_token=_os.environ.get('NYSENATE_API_TOKEN'),
            )
            results = ingester.ingest_all(sources=sources)
            logger.info(f"Authority ingestion complete: {results}")

            # Embed newly added authorities
            retriever = AuthorityRetriever(
                cohere_api_key=_os.environ.get('COHERE_API_KEY'),
            )
            if retriever.enabled:
                embedded = retriever.embed_pending_authorities(batch_size=200)
                logger.info(f"Embedded {embedded} new authorities")

        _threading.Thread(target=_ingest, daemon=True, name='ci-authority-ingest').start()
        return jsonify({'status': 'ingestion_started', 'sources': sources})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/authority/status')
@login_required
def ci_authority_status():
    """Authority corpus statistics."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_authority_corpus_stats
        from analyzer.case_intelligence.authority_retriever import AuthorityRetriever
        import os as _os

        db_stats = get_authority_corpus_stats()
        retriever = AuthorityRetriever(cohere_api_key=_os.environ.get('COHERE_API_KEY'))
        chroma_stats = retriever.get_corpus_stats()
        return jsonify({
            'db': db_stats,
            'chroma': chroma_stats,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


