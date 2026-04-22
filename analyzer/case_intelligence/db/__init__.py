"""
Case Intelligence database package.

Re-exports all public symbols so that existing
`from analyzer.case_intelligence.db import X` statements continue to work
without any changes.
"""

from analyzer.case_intelligence.db.schema import init_ci_db, recover_orphaned_runs
from analyzer.case_intelligence.db.runs import (
    create_ci_run, get_ci_run, list_ci_runs, update_ci_run,
    increment_ci_run_cost, increment_ci_run_docs, delete_ci_run,
    add_ci_run_share, remove_ci_run_share, list_ci_run_shares,
    get_run_ids_shared_with, is_run_shared_with,
    add_ci_question, get_ci_questions, answer_ci_question,
)
from analyzer.case_intelligence.db.analysis import (
    upsert_ci_entity, get_ci_entities, get_ci_entities_active,
    mark_entity_merged, update_entity_aliases,
    add_ci_event, get_ci_timeline,
    add_ci_contradiction, get_ci_contradictions,
    add_ci_disputed_fact, get_ci_disputed_facts,
    add_ci_theory, update_ci_theory, get_ci_theories,
    add_ci_web_research, get_ci_web_research,
)
from analyzer.case_intelligence.db.authorities import (
    add_ci_authority, get_ci_authorities,
    upsert_authority_corpus_entry, mark_authority_embedded,
    get_unembedded_authorities, get_authority_corpus_stats,
)
from analyzer.case_intelligence.db.reports import (
    create_ci_report, update_ci_report, get_ci_report, get_ci_reports_for_run,
    upsert_manager_report, get_manager_reports,
    upsert_forensic_report, get_forensic_report,
    upsert_discovery_gaps, get_discovery_gaps,
    upsert_witness_card, get_witness_cards,
    upsert_war_room, get_war_room, update_war_room_senior_notes,
    upsert_deep_forensics, get_deep_forensics,
    upsert_trial_strategy, get_trial_strategy,
    upsert_multi_model_comparison, get_multi_model_comparison,
    upsert_settlement_valuation, get_settlement_valuation,
)

__all__ = [
    'init_ci_db', 'recover_orphaned_runs',
    'create_ci_run', 'get_ci_run', 'list_ci_runs', 'update_ci_run',
    'increment_ci_run_cost', 'increment_ci_run_docs', 'delete_ci_run',
    'add_ci_run_share', 'remove_ci_run_share', 'list_ci_run_shares',
    'get_run_ids_shared_with', 'is_run_shared_with',
    'add_ci_question', 'get_ci_questions', 'answer_ci_question',
    'upsert_ci_entity', 'get_ci_entities', 'get_ci_entities_active',
    'mark_entity_merged', 'update_entity_aliases',
    'add_ci_event', 'get_ci_timeline',
    'add_ci_contradiction', 'get_ci_contradictions',
    'add_ci_disputed_fact', 'get_ci_disputed_facts',
    'add_ci_theory', 'update_ci_theory', 'get_ci_theories',
    'add_ci_web_research', 'get_ci_web_research',
    'add_ci_authority', 'get_ci_authorities',
    'upsert_authority_corpus_entry', 'mark_authority_embedded',
    'get_unembedded_authorities', 'get_authority_corpus_stats',
    'create_ci_report', 'update_ci_report', 'get_ci_report', 'get_ci_reports_for_run',
    'upsert_manager_report', 'get_manager_reports',
    'upsert_forensic_report', 'get_forensic_report',
    'upsert_discovery_gaps', 'get_discovery_gaps',
    'upsert_witness_card', 'get_witness_cards',
    'upsert_war_room', 'get_war_room', 'update_war_room_senior_notes',
    'upsert_deep_forensics', 'get_deep_forensics',
    'upsert_trial_strategy', 'get_trial_strategy',
    'upsert_multi_model_comparison', 'get_multi_model_comparison',
    'upsert_settlement_valuation', 'get_settlement_valuation',
]
