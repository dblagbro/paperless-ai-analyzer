import logging
from flask import Blueprint, request, jsonify, session
from flask_login import login_required, current_user

from analyzer.app import admin_required, safe_json_body
from analyzer.services.ai_config_service import (
    load_ai_config, save_ai_config, get_project_ai_config,
    _AI_DEFAULTS, _AI_PROVIDER_MODELS,
)

logger = logging.getLogger(__name__)

bp = Blueprint('ai_config', __name__)


def _can_access_project_config(slug: str) -> bool:
    """Non-admin users may only access their own current project's config."""
    if current_user.is_admin:
        return True
    return session.get('current_project', 'default') == slug


# ---------------------------------------------------------------------------
# AI Config routes
# ---------------------------------------------------------------------------

@bp.route('/api/ai-config', methods=['GET'])
@login_required
def api_ai_config_get():
    """Get current AI configuration (full v2 structure)."""
    try:
        return jsonify({'success': True, 'config': load_ai_config()})
    except Exception as e:
        logger.error(f"Failed to get AI config: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ai-config', methods=['POST'])
@login_required
def api_ai_config_save():
    """Save AI configuration (accepts v1 or v2 format)."""
    try:
        data = safe_json_body()
        config = data.get('config')
        if not config:
            return jsonify({'error': 'Configuration is required'}), 400
        if 'global' not in config and 'document_analysis' not in config:
            return jsonify({'error': 'Invalid configuration structure'}), 400
        if save_ai_config(config):
            return jsonify({'success': True, 'message': 'AI configuration saved.'})
        return jsonify({'error': 'Failed to save configuration'}), 500
    except Exception as e:
        logger.error(f"Failed to save AI config: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ai-config/test', methods=['POST'])
@login_required
def api_ai_config_test():
    """Test an AI provider/model configuration."""
    try:
        data = safe_json_body()
        provider = data.get('provider')
        api_key = data.get('api_key', '').strip()
        model = data.get('model', '')

        if not provider or not api_key:
            return jsonify({'error': 'Provider and API key are required'}), 400

        if provider == 'openai':
            import openai
            client = openai.OpenAI(api_key=api_key)

            models_list = client.models.list()
            model_ids = {m.id for m in models_list.data}
            preferred = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']
            available = [m for m in preferred if m in model_ids]
            display = ', '.join(available[:4]) if available else (next(iter(model_ids), 'unknown'))
            return jsonify({
                'success': True,
                'message': f'✓ OpenAI key is valid. Available models include: {display}',
                'model': available[0] if available else ''
            })

        elif provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            models_to_try = [model] if model else [
                'claude-3-opus-20240229',
                'claude-3-5-sonnet-20241022',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307'
            ]

            for test_model in models_to_try:
                try:
                    response = client.messages.create(
                        model=test_model,
                        max_tokens=10,
                        messages=[{"role": "user", "content": "Say 'test successful'"}]
                    )
                    return jsonify({
                        'success': True,
                        'message': f'✓ Anthropic API key is valid! Using model: {test_model}',
                        'model': test_model
                    })
                except Exception as e:
                    if '404' not in str(e) and 'not_found' not in str(e):
                        raise
                    continue

            return jsonify({
                'success': False,
                'error': 'No Claude models available with this API key'
            }), 400
        else:
            return jsonify({'error': f'Unknown provider: {provider}'}), 400

    except Exception as e:
        logger.error(f"AI config test failed: {e}")
        error_msg = str(e)
        if 'authentication' in error_msg.lower() or 'invalid' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': f'✗ Authentication failed: {error_msg}'
            }), 401
        return jsonify({
            'success': False,
            'error': f'✗ Error: {error_msg}'
        }), 500


@bp.route('/api/ai-config/global', methods=['GET'])
@admin_required
def api_ai_config_global_get():
    """Return global provider API keys (admin only)."""
    try:
        cfg = load_ai_config()
        return jsonify({'success': True, 'global': cfg.get('global', {})})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ai-config/global', methods=['POST'])
@admin_required
def api_ai_config_global_save():
    """Update global API keys (admin only)."""
    try:
        data = safe_json_body()
        new_global = data.get('global', {})
        if not isinstance(new_global, dict):
            return jsonify({'error': 'global must be a dict'}), 400
        cfg = load_ai_config()
        cfg.setdefault('global', {})
        for provider, vals in new_global.items():
            cfg['global'].setdefault(provider, {}).update(vals)
        save_ai_config(cfg)
        return jsonify({'success': True, 'message': 'Global API keys saved.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ai-config/projects/<slug>', methods=['GET'])
@login_required
def api_ai_config_project_get(slug):
    """Get per-project AI config (owner or admin)."""
    if not _can_access_project_config(slug):
        return jsonify({'error': 'Access denied'}), 403
    cfg = load_ai_config()
    project_cfg = cfg.get('projects', {}).get(slug, {})
    safe_cfg = {k: v for k, v in project_cfg.items()}
    has_openai    = bool(project_cfg.get('openai_api_key', '').strip())
    has_anthropic = bool(project_cfg.get('anthropic_api_key', '').strip())
    if has_openai:    safe_cfg['openai_api_key']    = '••••••••'
    if has_anthropic: safe_cfg['anthropic_api_key'] = '••••••••'
    return jsonify({'success': True, 'slug': slug, 'config': safe_cfg,
                    'has_openai_key': has_openai, 'has_anthropic_key': has_anthropic,
                    'defaults': _AI_DEFAULTS, 'models': _AI_PROVIDER_MODELS})


@bp.route('/api/ai-config/projects/<slug>', methods=['POST'])
@login_required
def api_ai_config_project_save(slug):
    """Save per-project AI config (owner or admin)."""
    if not _can_access_project_config(slug):
        return jsonify({'error': 'Access denied'}), 403
    try:
        data = safe_json_body()
        new_proj_cfg = data.get('config', {})
        cfg = load_ai_config()
        existing = cfg.get('projects', {}).get(slug, {})
        for key_field in ('openai_api_key', 'anthropic_api_key'):
            submitted = new_proj_cfg.get(key_field, '').strip()
            if not submitted or submitted == '••••••••':
                new_proj_cfg[key_field] = existing.get(key_field, '')
        cfg.setdefault('projects', {})[slug] = new_proj_cfg
        save_ai_config(cfg)
        return jsonify({'success': True, 'message': f'AI config saved for project "{slug}".'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ai-config/projects/<slug>/copy-use-case', methods=['POST'])
@login_required
def api_ai_config_copy_use_case(slug):
    """Copy one use-case config to all three use-cases within the same project."""
    if not _can_access_project_config(slug):
        return jsonify({'error': 'Access denied'}), 403
    try:
        data = safe_json_body()
        source_use_case = data.get('use_case')
        if source_use_case not in ('document_analysis', 'chat', 'case_intelligence'):
            return jsonify({'error': 'Invalid use_case'}), 400
        cfg = load_ai_config()
        proj = cfg.setdefault('projects', {}).setdefault(slug, {})
        source_cfg = proj.get(source_use_case, _AI_DEFAULTS[source_use_case])
        import copy
        for uc in ('document_analysis', 'chat', 'case_intelligence'):
            proj[uc] = copy.deepcopy(source_cfg)
        save_ai_config(cfg)
        return jsonify({'success': True, 'message': f'Copied {source_use_case} config to all use-cases in "{slug}".'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ai-config/projects/copy', methods=['POST'])
@admin_required
def api_ai_config_copy_project():
    """Copy one project's full AI config to another project (admin only)."""
    try:
        data = safe_json_body()
        src = data.get('source_slug', '').strip()
        dst = data.get('dest_slug', '').strip()
        if not src or not dst:
            return jsonify({'error': 'source_slug and dest_slug are required'}), 400
        cfg = load_ai_config()
        projects = cfg.setdefault('projects', {})
        if src not in projects:
            return jsonify({'error': f'Source project "{src}" has no AI config'}), 404
        import copy
        projects[dst] = copy.deepcopy(projects[src])
        save_ai_config(cfg)
        return jsonify({'success': True, 'message': f'Copied AI config from "{src}" to "{dst}".'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# LLM legacy routes
# ---------------------------------------------------------------------------

@bp.route('/api/llm/status')
@login_required
def api_llm_status():
    """Get LLM configuration status."""
    import os
    enabled = os.environ.get('LLM_ENABLED', 'true').lower() == 'true'
    provider = os.environ.get('LLM_PROVIDER', 'anthropic')
    has_key = bool(os.environ.get('LLM_API_KEY'))

    return jsonify({
        'enabled': enabled,
        'provider': provider,
        'has_key': has_key,
        'setup_url': 'https://console.anthropic.com/settings/keys' if provider == 'anthropic' else 'https://platform.openai.com/api-keys'
    })


@bp.route('/api/llm/test', methods=['POST'])
@login_required
def api_llm_test():
    """Test an LLM API key."""
    data = safe_json_body()
    provider = data.get('provider', 'anthropic')
    api_key = data.get('api_key', '').strip()

    if not api_key:
        return jsonify({'success': False, 'error': 'API key is required'}), 400

    try:
        if provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            models_to_try = [
                'claude-3-5-sonnet-20241022',
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307'
            ]

            last_error = None
            for model in models_to_try:
                try:
                    response = client.messages.create(
                        model=model,
                        max_tokens=10,
                        messages=[{'role': 'user', 'content': 'Hi'}]
                    )
                    return jsonify({
                        'success': True,
                        'message': f'✓ Claude API key is valid! Using model: {response.model}',
                        'model': response.model
                    })
                except Exception as e:
                    last_error = e
                    if '404' not in str(e):
                        raise
                    continue

            raise last_error or Exception("No models available")
        elif provider == 'openai':
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model='gpt-4-turbo-preview',
                messages=[{'role': 'user', 'content': 'Hi'}],
                max_tokens=10
            )
            return jsonify({
                'success': True,
                'message': f'✓ OpenAI API key is valid! Model: {response.model}'
            })
        else:
            return jsonify({'success': False, 'error': 'Unknown provider'}), 400

    except Exception as e:
        error_msg = str(e)
        if '401' in error_msg or 'authentication' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': '✗ Invalid API key - authentication failed'
            }), 400
        else:
            return jsonify({
                'success': False,
                'error': f'✗ Error: {error_msg}'
            }), 500


@bp.route('/api/llm/save', methods=['POST'])
@login_required
def api_llm_save():
    """Save LLM configuration and restart container."""
    from pathlib import Path
    data = safe_json_body()
    provider = data.get('provider', 'anthropic')
    api_key = data.get('api_key', '').strip()

    if not api_key:
        return jsonify({'success': False, 'error': 'API key is required'}), 400

    try:
        compose_file = Path('/docker-compose.yml')

        if compose_file.exists():
            with open(compose_file, 'r') as f:
                content = f.read()

            import re
            content = re.sub(
                r'LLM_ENABLED: "[^"]*"',
                'LLM_ENABLED: "true"',
                content
            )
            content = re.sub(
                r'LLM_PROVIDER: \w+',
                f'LLM_PROVIDER: {provider}',
                content
            )
            content = re.sub(
                r'LLM_API_KEY: [^\n]+',
                f'LLM_API_KEY: {api_key}',
                content
            )

            with open(compose_file, 'w') as f:
                f.write(content)

            return jsonify({
                'success': True,
                'message': 'Configuration saved! Please restart the container:\ndocker compose up -d paperless-ai-analyzer'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not find docker-compose.yml'
            }), 500

    except Exception as e:
        logger.error(f"Failed to save LLM config: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ---------------------------------------------------------------------------
# LLM usage stats routes
# ---------------------------------------------------------------------------

@bp.route('/api/llm-usage/stats', methods=['GET'])
@login_required
def api_get_llm_usage_stats():
    """Get LLM usage statistics."""
    from flask import current_app
    try:
        days = request.args.get('days', 30, type=int)

        if not hasattr(current_app, 'document_analyzer') or not current_app.document_analyzer:
            return jsonify({'error': 'Document analyzer not available'}), 503

        if not hasattr(current_app.document_analyzer, 'usage_tracker') or not current_app.document_analyzer.usage_tracker:
            return jsonify({'error': 'Usage tracker not available'}), 503

        stats = current_app.document_analyzer.usage_tracker.get_usage_stats(days=days)
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/llm-usage/recent', methods=['GET'])
@login_required
def api_get_recent_llm_calls():
    """Get recent LLM API calls."""
    from flask import current_app
    try:
        limit = request.args.get('limit', 50, type=int)

        if not hasattr(current_app, 'document_analyzer') or not current_app.document_analyzer:
            return jsonify({'error': 'Document analyzer not available'}), 503

        if not hasattr(current_app.document_analyzer, 'usage_tracker') or not current_app.document_analyzer.usage_tracker:
            return jsonify({'error': 'Usage tracker not available'}), 503

        calls = current_app.document_analyzer.usage_tracker.get_recent_calls(limit=limit)
        return jsonify({'calls': calls})

    except Exception as e:
        logger.error(f"Failed to get recent calls: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/llm-usage/pricing', methods=['GET'])
@login_required
def api_get_llm_pricing():
    """Get current LLM pricing information."""
    from flask import current_app
    try:
        if not hasattr(current_app, 'document_analyzer') or not current_app.document_analyzer:
            return jsonify({'error': 'Document analyzer not available'}), 503

        if not hasattr(current_app.document_analyzer, 'usage_tracker') or not current_app.document_analyzer.usage_tracker:
            return jsonify({'error': 'Usage tracker not available'}), 503

        pricing = current_app.document_analyzer.usage_tracker.get_pricing()
        return jsonify({'pricing': pricing})

    except Exception as e:
        logger.error(f"Failed to get pricing: {e}")
        return jsonify({'error': str(e)}), 500
