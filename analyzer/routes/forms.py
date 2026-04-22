import logging
from flask import Blueprint, request, jsonify, session
from flask_login import login_required

from analyzer.services.ai_config_service import load_ai_config, get_project_ai_config

logger = logging.getLogger(__name__)

bp = Blueprint('forms', __name__)


@bp.route('/api/ai-form/parse', methods=['POST'])
@login_required
def ai_form_parse():
    """
    Generic AI form-field extractor.  Accepts free-form text (or a multi-turn
    conversation) plus a field schema, and returns structured field values.

    Request JSON:
        schema       (list)  — [{name, label, description, secret, required}]
        conversation (list)  — [{role, content}, ...] full history including
                               the latest user message to send
        project_slug (str)   — optional, default 'default'

    Response JSON:
        fields       {fieldName: value|null, ...}
        summary      str
        follow_up    str|null
        complete     bool
        notes        str|null
    """
    import json as _json

    data         = request.get_json(force=True) or {}
    schema       = data.get('schema') or []
    conversation = data.get('conversation') or []
    project_slug = (data.get('project_slug') or
                    session.get('current_project', 'default') or 'default')

    if not conversation:
        return jsonify({'error': 'conversation required'}), 400
    if not schema:
        return jsonify({'error': 'schema required'}), 400

    field_lines = []
    field_names = []
    for f in schema:
        name   = f.get('name', '')
        label  = f.get('label', name)
        desc   = f.get('description', '')
        req    = f.get('required', False)
        secret = f.get('secret', False)
        line   = f'  - "{name}" ({label})'
        if desc:   line += f': {desc}'
        if req:    line += ' [REQUIRED]'
        if secret: line += ' [sensitive]'
        field_lines.append(line)
        field_names.append(name)

    fields_template  = ', '.join(f'"{n}": null' for n in field_names)
    response_template = (
        '{"fields": {' + fields_template + '}, '
        '"summary": "plain English of what was found", '
        '"follow_up": "single clarifying question or null", '
        '"complete": false, '
        '"notes": "any other important observations or null"}'
    )

    system_prompt = (
        "You are an expert at extracting structured data from unstructured text "
        "(emails, Slack messages, notes, etc.).\n\n"
        "Fields to extract:\n"
        + '\n'.join(field_lines) + "\n\n"
        "RULES:\n"
        "  - Extract as many fields as possible from the provided text.\n"
        "  - If a required field is missing or ambiguous, ask ONE clarifying question.\n"
        "  - Ask follow-up questions ONE AT A TIME — never ask multiple at once.\n"
        "  - Set complete:true when you have enough information to fill the form "
        "(all required fields are present or can be reasonably inferred).\n"
        "  - Leave optional fields as null if not found.\n\n"
        "Respond with ONLY valid JSON — no markdown fences, no extra text:\n"
        + response_template
    )

    chat_cfg = get_project_ai_config(project_slug, 'chat')
    full_cfg = load_ai_config()

    def _global_key(p):
        return full_cfg.get('global', {}).get(p, {}).get('api_key', '').strip()

    provider = chat_cfg.get('provider', 'openai')
    api_key  = (chat_cfg.get('api_key') or '').strip() or _global_key(provider)
    model    = chat_cfg.get('model', 'gpt-4o-mini')

    if not api_key:
        fb_prov = chat_cfg.get('fallback_provider')
        api_key  = _global_key(fb_prov) if fb_prov else ''
        if api_key:
            provider = fb_prov
            model    = chat_cfg.get('fallback_model', model)

    if not api_key:
        return jsonify({
            'error': 'No AI API key configured — set up an AI provider in Settings first.'
        }), 503

    try:
        raw_response = ''

        if provider == 'openai':
            import openai as _oai
            client = _oai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{'role': 'system', 'content': system_prompt}] + conversation,
                temperature=0,
                max_tokens=800,
            )
            raw_response = resp.choices[0].message.content or ''

        elif provider == 'anthropic':
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=800,
                system=system_prompt,
                messages=conversation,
            )
            raw_response = resp.content[0].text if resp.content else ''

        else:
            return jsonify({'error': f'Unsupported AI provider: {provider}'}), 400

        stripped = raw_response.strip()
        if stripped.startswith('```'):
            stripped = stripped.split('\n', 1)[1]
            stripped = stripped.rsplit('```', 1)[0]

        parsed = _json.loads(stripped)
        return jsonify(parsed)

    except Exception as e:
        logger.error(f'ai_form_parse error: {e}')
        return jsonify({'error': str(e)}), 500
