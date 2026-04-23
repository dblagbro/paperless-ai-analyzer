import logging
from flask import Blueprint, request, jsonify, session
from flask_login import login_required

from analyzer.services.ai_config_service import load_ai_config, get_project_ai_config
from analyzer.app import safe_json_body

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

    data         = safe_json_body()
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

    from analyzer.llm.proxy_call import call_llm, LLMUnavailableError
    try:
        result = call_llm(
            messages=[{'role': 'system', 'content': system_prompt}] + conversation,
            task='extraction',
            max_tokens=800,
            temperature=0,
            project_slug=project_slug,
            operation='ai_form_parse',
            direct_provider=chat_cfg.get('provider'),
            direct_api_key=chat_cfg.get('api_key'),
            direct_model=chat_cfg.get('model'),
        )
        raw_response = result['content'] or ''
    except LLMUnavailableError as e:
        logger.warning(f"ai_form_parse: {e} attempted={e.attempted}")
        return jsonify({
            'error': 'No AI provider available — check Configuration → AI Settings.',
            'source': 'llm-pool-exhausted',
        }), 503

    try:
        stripped = raw_response.strip()
        if stripped.startswith('```'):
            stripped = stripped.split('\n', 1)[1]
            stripped = stripped.rsplit('```', 1)[0]

        parsed = _json.loads(stripped)
        return jsonify(parsed)

    except Exception as e:
        logger.error(f'ai_form_parse error: {e}')
        return jsonify({'error': str(e)}), 500
