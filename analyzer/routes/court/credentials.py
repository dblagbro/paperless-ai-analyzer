"""Court credential management — save, list, test, delete, and AI-assisted paste-parse."""
import logging

from flask import jsonify, request, session
from flask_login import current_user, login_required

from analyzer.app import advanced_required, safe_json_body
from . import bp
from .helpers import _court_gate, _build_court_connector, _get_current_project_slug

logger = logging.getLogger(__name__)

@bp.route('/api/court/credentials', methods=['POST'])
@login_required
@advanced_required
def court_save_credentials():
    """Save (upsert) court credentials for the current project."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = safe_json_body()
    court_system = data.get('court_system', '')
    username = data.get('username', '').strip()
    password = data.get('password', '')
    extra_config = data.get('extra_config', {})
    project_slug = data.get('project_slug', '') or 'default'

    if court_system not in ('federal', 'nyscef'):
        return jsonify({'error': 'court_system must be "federal" or "nyscef"'}), 400
    # NYSCEF public-only access has no username — allow empty username in that case
    if not username and not (court_system == 'nyscef' and extra_config.get('public_only')):
        return jsonify({'error': 'username is required'}), 400

    try:
        from analyzer.court_connectors.credential_store import encrypt_password, is_cryptography_available
        from analyzer.court_db import save_credentials
        if not is_cryptography_available():
            return jsonify({'error': 'cryptography package not installed — cannot encrypt credentials'}), 500
        encrypted = encrypt_password(password) if password else b''
        save_credentials(project_slug, court_system, username, encrypted, extra_config)
        return jsonify({'ok': True, 'message': f'{court_system} credentials saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/court/credentials', methods=['GET'])
@login_required
def court_list_credentials():
    """List configured court systems for the current project (no passwords)."""
    ok, err = _court_gate()
    if not ok:
        return err
    project_slug = request.args.get('project_slug', 'default')
    try:
        from analyzer.court_db import list_credentials
        creds = list_credentials(project_slug)
        return jsonify({'credentials': creds})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/court/credentials/test', methods=['POST'])
@login_required
@advanced_required
def court_test_credentials():
    """Test court credentials and return account info."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = safe_json_body()
    court_system = data.get('court_system', '')
    username = data.get('username', '').strip()
    password = data.get('password', '')
    extra_config = data.get('extra_config', {})
    project_slug = data.get('project_slug', 'default')

    if court_system not in ('federal', 'nyscef'):
        return jsonify({'error': 'court_system must be "federal" or "nyscef"'}), 400

    try:
        # Build a temporary credential dict for testing (don't require DB save first)
        import json as _json
        temp_creds = {
            'username': username,
            'extra_config_json': _json.dumps(extra_config),
        }
        if court_system == 'federal':
            from analyzer.court_connectors.federal import FederalConnector
            connector = FederalConnector(project_slug, temp_creds, pacer_password=password)
        else:
            from analyzer.court_connectors.nyscef import NYSCEFConnector
            connector = NYSCEFConnector(project_slug, temp_creds, password=password)

        result = connector.test_connection()

        # Update last_tested_at in DB if credentials already exist
        try:
            from analyzer.court_db import update_credential_test
            update_credential_test(project_slug, court_system, result['ok'])
        except Exception:
            pass

        return jsonify(result)
    except Exception as e:
        return jsonify({'ok': False, 'account_info': '', 'error': str(e)}), 500


@bp.route('/api/court/credentials/<court_system>', methods=['DELETE'])
@login_required
@advanced_required
def court_delete_credentials(court_system):
    """Remove court credentials for the current project."""
    ok, err = _court_gate()
    if not ok:
        return err
    project_slug = request.args.get('project_slug', 'default')
    try:
        from analyzer.court_db import delete_credentials
        deleted = delete_credentials(project_slug, court_system)
        return jsonify({'ok': deleted,
                        'message': f'{court_system} credentials removed' if deleted
                                   else 'No credentials found'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Search / docket routes
# ---------------------------------------------------------------------------

@bp.route('/api/court/credentials/parse', methods=['POST'])
@login_required
@advanced_required
def court_parse_credentials():
    """
    Use AI to parse free-form text (email, Slack message, lawyer notes) into
    structured court credential fields.

    Request JSON:
        raw_text   (str)  — optional on follow-up turns
        conversation (list) — [{role, content}, ...] full history including
                              the latest user message to send

    Returns JSON:
        court_system, username, password, pacer_client_code,
        courtlistener_api_token, nyscef_county, public_only,
        summary, follow_up, complete, notes
    """
    from flask import current_app
    ok, err = _court_gate()
    if not ok:
        return err

    data = safe_json_body()
    raw_text     = (data.get('raw_text') or '').strip()
    conversation = data.get('conversation') or []

    if not raw_text and not conversation:
        return jsonify({'error': 'raw_text or conversation required'}), 400

    llm = getattr(current_app, 'llm_client', None)
    if not llm or not llm.client:
        return jsonify({
            'error': 'AI not configured — set up an AI provider in Settings first'
        }), 503

    system_prompt = (
        "You are an expert at extracting court system login credentials from "
        "unstructured text (emails, Slack messages, attorney notes, etc.).\n\n"
        "Supported court systems:\n"
        "  - \"federal\": Uses PACER (username + password + optional billing "
        "client code) and/or a free CourtListener API token.\n"
        "  - \"nyscef\": New York state courts — NY Attorney Registration # + "
        "NYSCEF e-Filing password + optional default county.\n\n"
        "RULES:\n"
        "  - Phrases like 'public access', 'no login required', 'free access', "
        "'I am a party', 'I am a defendant', 'I am a plaintiff' (not an attorney) "
        "mean the user has no professional credentials — set public_only:true.\n"
        "  - For federal + public_only: CourtListener works without PACER.\n"
        "  - For nyscef + public_only: parties/defendants/plaintiffs can use the "
        "public NYSCEF portal with just an index number — no attorney login needed.\n"
        "  - Extract usernames/passwords even if labelled differently "
        "(e.g. 'login: X', 'user: X', 'pw: Y').\n"
        "  - If court system is unclear, ask ONE clarifying question.\n"
        "  - Ask follow-up questions ONE AT A TIME — never ask multiple at once.\n"
        "  - Set complete:true when you have enough to configure the system "
        "(public_only + court_system is sufficient for both public federal and "
        "public NYSCEF; no password required in public_only mode).\n\n"
        "Respond with ONLY valid JSON — no markdown fences, no extra text:\n"
        '{"court_system":"federal"|"nyscef"|null,'
        '"username":null,"password":null,'
        '"pacer_client_code":null,"courtlistener_api_token":null,'
        '"nyscef_county":null,"public_only":false,'
        '"summary":"plain English of what was found",'
        '"follow_up":"single question or null",'
        '"complete":false,'
        '"notes":"any other important observations or null"}'
    )

    # Build message list — client sends the full conversation including the
    # latest user turn, so we just pass it through.
    if conversation:
        messages = conversation
    elif raw_text:
        messages = [{'role': 'user',
                     'content': f"Please parse these court credentials:\n\n{raw_text}"}]
    else:
        return jsonify({'error': 'No input to parse'}), 400

    try:
        import json as _json
        raw_response = ''

        if llm.provider == 'openai':
            resp = llm.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'system', 'content': system_prompt}] + messages,
                temperature=0,
                max_tokens=600,
            )
            raw_response = resp.choices[0].message.content or ''
        else:
            # Anthropic — system param is separate
            resp = llm.client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=600,
                system=system_prompt,
                messages=messages,
            )
            raw_response = resp.content[0].text if resp.content else ''

        # Strip markdown code fences if present
        raw_stripped = raw_response.strip()
        if raw_stripped.startswith('```'):
            raw_stripped = raw_stripped.split('\n', 1)[1]
            raw_stripped = raw_stripped.rsplit('```', 1)[0]

        parsed = _json.loads(raw_stripped)
        return jsonify(parsed)

    except Exception as e:
        logger.error(f"Court credential parse failed: {e}")
        return jsonify({'error': str(e)}), 500
