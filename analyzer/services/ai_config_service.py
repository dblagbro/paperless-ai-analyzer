"""
AI configuration service — load, save, migrate, and resolve per-project AI config.

This module is framework-agnostic (no Flask). Import it from route handlers.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

AI_CONFIG_PATH = Path('/app/data/ai_config.json')

_AI_PROVIDER_MODELS = {
    'openai':    ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
    'anthropic': ['claude-sonnet-4-6', 'claude-opus-4-6', 'claude-haiku-4-5-20251001',
                  'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'],
}

_AI_DEFAULTS = {
    'document_analysis': {
        'provider': 'openai', 'model': 'gpt-4o',
        'fallback_provider': 'anthropic', 'fallback_model': 'claude-sonnet-4-6',
    },
    'chat': {
        'provider': 'anthropic', 'model': 'claude-sonnet-4-6',
        'fallback_provider': 'openai', 'fallback_model': 'gpt-4o',
    },
    'case_intelligence': {
        'provider': 'openai', 'model': 'gpt-4o',
        'fallback_provider': 'anthropic', 'fallback_model': 'claude-sonnet-4-6',
    },
}


def get_default_ai_config() -> dict:
    """Return empty v2 config (backward-compat; prefer load_ai_config())."""
    return _empty_new_ai_config()


def _empty_new_ai_config() -> dict:
    return {
        'global': {
            'openai':    {'api_key': os.environ.get('OPENAI_API_KEY', ''),
                          'enabled': bool(os.environ.get('OPENAI_API_KEY'))},
            'anthropic': {'api_key': os.environ.get('LLM_API_KEY', ''),
                          'enabled': bool(os.environ.get('LLM_API_KEY'))},
        },
        'projects': {},
    }


def _migrate_old_ai_config(old_cfg: dict) -> dict:
    """Convert v1 flat format → v2 per-project format."""
    import sqlite3 as _sqlite3
    import copy

    def _key(section, provider_name):
        for p in old_cfg.get(section, {}).get('providers', []):
            if p.get('name') == provider_name:
                return p.get('api_key', ''), bool(p.get('enabled', False))
        return '', False

    oai_key, oai_en = _key('document_analysis', 'openai')
    ant_key, ant_en = _key('document_analysis', 'anthropic')

    new_cfg = {
        'global': {
            'openai':    {'api_key': oai_key or os.environ.get('OPENAI_API_KEY', ''),
                          'enabled': oai_en or bool(os.environ.get('OPENAI_API_KEY'))},
            'anthropic': {'api_key': ant_key or os.environ.get('LLM_API_KEY', ''),
                          'enabled': ant_en or bool(os.environ.get('LLM_API_KEY'))},
        },
        'projects': {},
    }

    da_prim = 'openai' if oai_key else ('anthropic' if ant_key else 'openai')
    da_fallbk = 'anthropic' if da_prim == 'openai' else 'openai'
    project_cfg = {
        'document_analysis': {
            'provider': da_prim, 'model': _AI_DEFAULTS['document_analysis']['model'],
            'fallback_provider': da_fallbk,
            'fallback_model': _AI_DEFAULTS['document_analysis']['fallback_model'],
        },
        'chat': dict(_AI_DEFAULTS['chat']),
        'case_intelligence': dict(_AI_DEFAULTS['case_intelligence']),
    }

    projects_db = Path('/app/data/projects.db')
    if projects_db.exists():
        try:
            with _sqlite3.connect(str(projects_db)) as conn:
                rows = conn.execute("SELECT slug FROM projects WHERE is_archived = 0").fetchall()
                for (slug,) in rows:
                    new_cfg['projects'][slug] = copy.deepcopy(project_cfg)
        except Exception as _e:
            logger.warning(f"AI config migration: could not read projects.db — {_e}")

    logger.info(f"Migrated AI config v1→v2 ({len(new_cfg['projects'])} projects copied)")
    return new_cfg


def load_ai_config() -> dict:
    """Load AI config from file. Auto-migrates v1 format → v2 on first access."""
    try:
        if AI_CONFIG_PATH.exists():
            with open(AI_CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            if 'document_analysis' in cfg and 'global' not in cfg:
                cfg = _migrate_old_ai_config(cfg)
                save_ai_config(cfg)
            return cfg
    except Exception as e:
        logger.warning(f"Failed to load AI config: {e}")
    return _empty_new_ai_config()


def save_ai_config(config: dict) -> bool:
    """Persist AI config to file."""
    try:
        AI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AI_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Saved AI configuration")
        return True
    except Exception as e:
        logger.error(f"Failed to save AI config: {e}")
        return False


def get_project_ai_config(project_slug: str, use_case: str) -> dict:
    """Return resolved {provider, model, fallback_provider, fallback_model, api_key}
    for a given project + use_case.

    use_case: 'document_analysis' | 'chat' | 'case_intelligence'
    Key resolution order: project override → global key → env var.
    """
    cfg = load_ai_config()
    proj_cfg = cfg.get('projects', {}).get(project_slug, {})
    use_case_cfg = proj_cfg.get(use_case)
    if use_case_cfg and use_case_cfg.get('provider'):
        result = dict(use_case_cfg)
    else:
        result = dict(_AI_DEFAULTS.get(use_case, _AI_DEFAULTS['document_analysis']))
    provider = result.get('provider', 'openai')
    project_key = proj_cfg.get(f'{provider}_api_key', '').strip()
    if project_key:
        result['api_key'] = project_key
    else:
        result['api_key'] = cfg.get('global', {}).get(provider, {}).get('api_key', '')
    return result
