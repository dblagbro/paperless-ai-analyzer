import logging
import os
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, session, current_app
from flask_login import login_required, current_user

from analyzer.app import admin_required, ui_state, log_buffer
from analyzer.db import get_analyzed_doc_ids

logger = logging.getLogger(__name__)

bp = Blueprint('documents', __name__)


@bp.route('/api/scan/process-unanalyzed', methods=['POST'])
@login_required
def api_process_unanalyzed():
    """Find every Paperless document not yet in processed_documents and analyze only those."""
    try:
        if not hasattr(current_app, 'document_analyzer') or not current_app.document_analyzer:
            return jsonify({'error': 'Analyzer not running'}), 503

        # Get already-analyzed IDs from DB
        analyzed_ids = get_analyzed_doc_ids()

        # Fetch all Paperless document IDs (lightweight list, no content yet)
        all_docs = []
        page = 1
        while True:
            resp = current_app.paperless_client.get_documents(ordering='-modified', page_size=100, page=page)
            page_results = resp.get('results', [])
            all_docs.extend(page_results)
            if not resp.get('next'):
                break
            page += 1

        missing_docs = [d for d in all_docs if d['id'] not in analyzed_ids]

        if not missing_docs:
            return jsonify({'success': True, 'queued': 0, 'message': 'All documents already analyzed'})

        def _run(docs):
            logger.info(f"Process-unanalyzed: starting {len(docs)} documents")
            ok = 0
            for d in docs:
                try:
                    full_doc = current_app.paperless_client.get_document(d['id'])
                    current_app.document_analyzer.analyze_document(full_doc)
                    ok += 1
                except Exception as e:
                    logger.warning(f"Process-unanalyzed: failed doc {d['id']}: {e}")
            logger.info(f"Process-unanalyzed: complete — {ok}/{len(docs)} succeeded")

        from threading import Thread
        Thread(target=_run, args=(missing_docs,), daemon=True).start()

        return jsonify({
            'success': True,
            'queued': len(missing_docs),
            'message': f'Queued {len(missing_docs)} unanalyzed documents — check logs for progress',
        })

    except Exception as e:
        logger.error(f"Failed to start process-unanalyzed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/settings/poll-interval', methods=['POST'])
@login_required
def api_settings_poll_interval():
    """Update the poll interval setting."""
    try:
        data = request.json
        interval = data.get('interval')

        if not interval or not isinstance(interval, (int, float)):
            return jsonify({'error': 'Invalid interval value'}), 400

        interval = int(interval)

        # Validate range (5 seconds to 1 hour)
        if interval < 5 or interval > 3600:
            return jsonify({'error': 'Interval must be between 5 and 3600 seconds'}), 400

        # Update docker-compose.yml
        import yaml
        compose_path = '/docker-compose.yml'

        try:
            with open(compose_path, 'r') as f:
                compose_data = yaml.safe_load(f)

            # Update the environment variable
            if 'services' in compose_data and 'paperless-ai-analyzer' in compose_data['services']:
                env_vars = compose_data['services']['paperless-ai-analyzer'].get('environment', {})
                env_vars['POLL_INTERVAL_SECONDS'] = str(interval)
                compose_data['services']['paperless-ai-analyzer']['environment'] = env_vars

                # Write back
                with open(compose_path, 'w') as f:
                    yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)

                logger.info(f"Updated poll interval to {interval} seconds")

                return jsonify({
                    'success': True,
                    'message': f'Poll interval updated to {interval} seconds. Restart container to apply.',
                    'interval': interval
                })
            else:
                return jsonify({'error': 'Could not find service in docker-compose.yml'}), 500

        except Exception as e:
            logger.error(f"Failed to update docker-compose.yml: {e}")
            return jsonify({'error': f'Failed to update config: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Failed to update poll interval: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/trigger', methods=['POST'])
@login_required
def api_trigger():
    """Manually trigger analysis of a document."""
    data = request.json
    doc_id = data.get('doc_id')

    if not doc_id:
        return jsonify({'error': 'doc_id required'}), 400

    try:
        # Verify document exists
        doc = current_app.paperless_client.get_document(doc_id)

        return jsonify({
            'success': True,
            'message': f'Document {doc_id} queued for analysis',
            'document': {
                'id': doc['id'],
                'title': doc['title']
            }
        })
    except Exception as e:
        logger.error(f"Failed to trigger analysis for doc {doc_id}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/logs')
@login_required
def api_logs():
    """Get recent log entries from the running process."""
    try:
        limit = int(request.args.get('limit', '100'))

        # Get state info for header
        state_stats = current_app.state_manager.get_stats()

        logs = []

        # Add header with status
        logs.append(f"=== Analyzer Status (Updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}) ===")
        logs.append(f"Last run: {state_stats.get('last_run', 'Never')}")
        logs.append(f"Documents processed: {state_stats.get('total_documents_processed', 0)}")
        logs.append(f"Active profiles: {len(current_app.profile_loader.profiles)}")
        logs.append(f"LLM enabled: {os.environ.get('LLM_ENABLED', 'true')}")
        logs.append("")

        # Get logs from in-memory buffer
        if log_buffer:
            logs.append("=== Recent Activity (Live Updates) ===")
            # Get last N lines from buffer
            recent_logs = list(log_buffer)[-limit:]
            logs.extend(recent_logs)
        else:
            logs.append("=== No logs available yet ===")
            logs.append("Logs will appear here once the analyzer starts processing documents")

    except Exception as e:
        logger.error(f"Failed to generate logs: {e}")
        logs = [f"Error: {str(e)}"]

    return jsonify({
        'logs': logs
    })


@bp.route('/api/reprocess', methods=['POST'])
@login_required
def api_reprocess():
    """Reset state and reprocess all documents."""
    try:
        # Delete state file to force reprocessing
        state_file = Path('/app/data/state.json')
        if state_file.exists():
            state_file.unlink()
            logger.info("State file deleted - will reprocess all documents on next poll")

        # Reset in-memory state using proper method
        current_app.state_manager.reset()

        return jsonify({
            'success': True,
            'message': 'State reset - all documents will be reprocessed on next poll cycle'
        })
    except Exception as e:
        logger.error(f"Failed to reset state: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/reprocess/<int:doc_id>', methods=['POST'])
@login_required
def api_reprocess_document(doc_id):
    """Reprocess a specific document by removing it from state."""
    try:
        # Remove document from seen_ids so it will be reprocessed
        if hasattr(current_app.state_manager.state, 'last_seen_ids'):
            if doc_id in current_app.state_manager.state.get('last_seen_ids', []):
                current_app.state_manager.state['last_seen_ids'].remove(doc_id)
                current_app.state_manager.save_state()

        return jsonify({
            'success': True,
            'message': f'Document {doc_id} will be reprocessed on next poll cycle'
        })
    except Exception as e:
        logger.error(f"Failed to reprocess document {doc_id}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/reconcile', methods=['POST'])
@login_required
@admin_required
def api_reconcile():
    """Reconcile processed_documents and vector store against current Paperless contents.
    Removes stale records for documents deleted from Paperless. Does NOT re-analyze anything."""
    try:
        if not hasattr(current_app, 'paperless_client') or not current_app.paperless_client:
            return jsonify({'error': 'Paperless client not available'}), 503

        project_slug = session.get('current_project', 'default')

        # 1. Fetch every doc ID currently in Paperless (paginated)
        paperless_ids = set()
        page = 1
        while True:
            resp = current_app.paperless_client.get_documents(ordering='id', page_size=100, page=page)
            for doc in resp.get('results', []):
                paperless_ids.add(doc['id'])
            if not resp.get('next'):
                break
            page += 1

        # 2. Get doc IDs from processed_documents for this project
        from analyzer.db import get_analyzed_doc_ids
        import sqlite3 as _sqlite3
        with _sqlite3.connect('/app/data/app.db') as _conn:
            _conn.row_factory = _sqlite3.Row
            db_rows = _conn.execute(
                "SELECT doc_id FROM processed_documents WHERE project_slug = ?", (project_slug,)
            ).fetchall()
        db_ids = {r['doc_id'] for r in db_rows}

        # 3. Get doc IDs from Chroma for this project
        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=project_slug)
        chroma_ids = set()
        if vs.enabled:
            raw = vs.collection.get(include=['metadatas'])
            for meta in (raw.get('metadatas') or []):
                if meta and meta.get('document_id'):
                    chroma_ids.add(int(meta['document_id']))

        # 4. Find orphans (in our indexes but deleted from Paperless)
        db_orphans    = db_ids    - paperless_ids
        chroma_orphans = chroma_ids - paperless_ids

        # 5. Remove orphans from processed_documents
        db_removed = 0
        if db_orphans:
            with _sqlite3.connect('/app/data/app.db') as _conn:
                for oid in db_orphans:
                    _conn.execute("DELETE FROM processed_documents WHERE doc_id = ?", (oid,))
                    db_removed += 1
            logger.info(f"Reconcile: removed {db_removed} stale records from processed_documents")

        # 6. Remove orphans from Chroma
        chroma_removed = 0
        if chroma_orphans and vs.enabled:
            for oid in chroma_orphans:
                vs.delete_document(oid)
                chroma_removed += 1
            logger.info(f"Reconcile: removed {chroma_removed} stale embeddings from Chroma")

        # 7. Compute gaps (in Paperless but missing from our indexes)
        not_in_db     = len(paperless_ids - db_ids)
        not_in_chroma = len(paperless_ids - chroma_ids)

        return jsonify({
            'success': True,
            'paperless_total': len(paperless_ids),
            'db_orphans_removed': db_removed,
            'chroma_orphans_removed': chroma_removed,
            'not_analyzed': not_in_db,
            'not_embedded': not_in_chroma,
            'message': (
                f"Removed {db_removed} stale DB record(s) and {chroma_removed} stale embedding(s). "
                f"{not_in_db} doc(s) not yet analyzed, {not_in_chroma} doc(s) not yet embedded."
            )
        })

    except Exception as e:
        logger.error(f"Reconcile error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@bp.route('/api/search')
@login_required
def api_search():
    """Search analysis results."""
    query = request.args.get('q', '').lower()
    risk_min = request.args.get('risk_min', type=int)
    risk_max = request.args.get('risk_max', type=int)
    has_anomalies = request.args.get('has_anomalies', type=bool)

    def _chroma_meta_to_result(m):
        anomalies_str = m.get('anomalies', '')
        anomalies_list = [a.strip() for a in anomalies_str.split(',') if a.strip()] if anomalies_str else []
        return {
            'doc_id': m.get('document_id'),
            'document_id': m.get('document_id'),
            'title': m.get('title', ''),
            'document_title': m.get('title', ''),
            'brief_summary': m.get('brief_summary', ''),
            'full_summary': m.get('full_summary', ''),
            'anomalies_found': anomalies_list,
            'risk_score': m.get('risk_score', 0),
            'timestamp': m.get('timestamp', ''),
        }

    # When any search criteria is provided, query Chroma directly
    if query or has_anomalies or risk_min is not None or risk_max is not None:
        try:
            from analyzer.vector_store import VectorStore
            vs = VectorStore(project_slug=session.get('current_project', 'default'))
            if vs and vs.enabled:
                results = []
                seen_ids = set()

                if query:
                    # Exact doc_id match
                    try:
                        exact = vs.collection.get(ids=[query], include=['metadatas'])
                        if exact['ids']:
                            results.append(_chroma_meta_to_result(exact['metadatas'][0]))
                            seen_ids.add(str(exact['ids'][0]))
                    except Exception:
                        pass

                    # Semantic search across all embedded docs
                    semantic = vs.search(query, n_results=50)
                    for s in semantic:
                        doc_id = str(s['document_id'])
                        if doc_id not in seen_ids:
                            try:
                                raw = vs.collection.get(ids=[doc_id], include=['metadatas'])
                                if raw['ids']:
                                    results.append(_chroma_meta_to_result(raw['metadatas'][0]))
                                    seen_ids.add(doc_id)
                            except Exception:
                                pass

                    # Also filter all docs by title/summary/anomaly text for exact substring matches
                    all_docs = vs.collection.get(include=['metadatas'])
                    for m in all_docs['metadatas']:
                        doc_id = str(m.get('document_id', ''))
                        if doc_id in seen_ids:
                            continue
                        if (query in m.get('title', '').lower() or
                                query in m.get('brief_summary', '').lower() or
                                query in m.get('full_summary', '').lower() or
                                query in m.get('anomalies', '').lower()):
                            results.append(_chroma_meta_to_result(m))
                            seen_ids.add(doc_id)
                else:
                    # No text query — fetch all and filter by metadata
                    all_docs = vs.collection.get(include=['metadatas'])
                    for m in all_docs['metadatas']:
                        results.append(_chroma_meta_to_result(m))

                # Apply risk and anomaly filters
                if has_anomalies:
                    results = [r for r in results if r.get('anomalies_found')]
                if risk_min is not None:
                    results = [r for r in results if r.get('risk_score', 0) >= risk_min]
                if risk_max is not None:
                    results = [r for r in results if r.get('risk_score', 0) <= risk_max]

                return jsonify({'results': results, 'count': len(results)})
        except Exception as _e:
            logger.warning(f"Chroma search failed, falling back to recent_analyses: {_e}")

    # Fallback: in-memory recent_analyses (last 100 from current session)
    with ui_state['lock']:
        results = ui_state['recent_analyses']

        if query:
            results = [r for r in results if
                      query in r.get('title', '').lower() or
                      query in str(r.get('doc_id', '')).lower() or
                      query in r.get('brief_summary', '').lower() or
                      query in r.get('full_summary', '').lower() or
                      any(query in a.lower() for a in r.get('anomalies_found', []))]
        if risk_min is not None:
            results = [r for r in results if r.get('risk_score', 0) >= risk_min]
        if risk_max is not None:
            results = [r for r in results if r.get('risk_score', 0) <= risk_max]
        if has_anomalies:
            results = [r for r in results if r.get('anomalies_found')]

        return jsonify({'results': results, 'count': len(results)})


@bp.route('/api/tag-evidence/<int:doc_id>')
@login_required
def api_tag_evidence(doc_id):
    """
    Get enhanced tag evidence for a specific document.
    Returns detailed information about why each tag was flagged.
    """
    import sqlite3
    import json

    # Define fallback explanations for anomaly types
    ANOMALY_EXPLANATIONS = {
        'balance_mismatch': {
            'category': 'Financial Integrity',
            'description': 'The running balance does not match the calculated balance based on debits and credits.',
            'severity': 'high'
        },
        'duplicate_lines': {
            'category': 'Data Quality',
            'description': 'Duplicate transaction entries were detected in the document.',
            'severity': 'medium'
        },
        'duplicate_transaction': {
            'category': 'Data Quality',
            'description': 'The same transaction appears multiple times with identical details.',
            'severity': 'medium'
        },
        'date_ordering': {
            'category': 'Data Quality',
            'description': 'Transaction dates are not in chronological order.',
            'severity': 'low'
        },
        'missing_data': {
            'category': 'Completeness',
            'description': 'Required fields or data are missing from the document.',
            'severity': 'medium'
        },
        'forensic_risk_high': {
            'category': 'Document Forensics',
            'description': 'High risk of document tampering detected through image analysis (risk score > 60%).',
            'severity': 'critical'
        },
        'forensic_risk_medium': {
            'category': 'Document Forensics',
            'description': 'Medium risk of document tampering detected through image analysis (risk score 30-60%).',
            'severity': 'medium'
        },
        'forensic_risk_low': {
            'category': 'Document Forensics',
            'description': 'Low risk indicators detected through image analysis (risk score < 30%).',
            'severity': 'low'
        }
    }

    # Try to find analysis in recent_analyses first for enhanced tags
    analysis = None
    with ui_state['lock']:
        for result in ui_state['recent_analyses']:
            if result.get('doc_id') == doc_id or result.get('document_id') == doc_id:
                analysis = result
                break

    # If not in recent_analyses, fetch from Paperless API
    if not analysis:
        try:
            doc = current_app.paperless_client.get_document(doc_id)

            # Get tags from Paperless
            tags = []
            for tag_id in doc.get('tags', []):
                try:
                    tag_response = current_app.paperless_client.session.get(
                        f'{current_app.paperless_client.base_url}/api/tags/{tag_id}/'
                    )
                    if tag_response.ok:
                        tags.append(tag_response.json().get('name', ''))
                except:
                    pass

            # Extract anomaly tags
            anomalies = [t.replace('anomaly:', '') for t in tags if t.startswith('anomaly:')]

            analysis = {
                'document_id': doc_id,
                'document_title': doc.get('title', 'Unknown'),
                'anomalies_found': anomalies,
                'enhanced_tags': []
            }
        except Exception as e:
            logger.error(f"Failed to fetch document {doc_id}: {e}")
            return jsonify({'error': 'Document not found'}), 404

    # Collect all tags with evidence
    all_tags = []

    # Add enhanced tags with detailed evidence (issue: type tags from LLM analysis)
    enhanced_tags = analysis.get('enhanced_tags', [])
    all_tags.extend(enhanced_tags)

    # Query anomaly-detector database for detailed evidence
    anomaly_detector_evidence = {}
    try:
        db_path = '/app/anomaly_data/anomaly_detector.db'
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                has_anomalies,
                anomaly_types,
                balance_check_status,
                balance_diff_amount,
                beginning_balance,
                ending_balance,
                calculated_balance,
                pattern_flags,
                document_type
            FROM processed_documents
            WHERE paperless_doc_id = ?
        """, (doc_id,))

        row = cursor.fetchone()
        if row:
            anomaly_types = json.loads(row['anomaly_types']) if row['anomaly_types'] else []
            pattern_flags = json.loads(row['pattern_flags']) if row['pattern_flags'] else []
            db_balance_status = row['balance_check_status']

            flags_by_type = {}
            for flag in pattern_flags:
                t = flag.get('type', '')
                flags_by_type.setdefault(t, []).append(flag)

            for anomaly_type in anomaly_types:
                matching_flags = flags_by_type.get(anomaly_type, [])
                evidence = {'db_confirmed': True, 'severity': 'medium'}

                if anomaly_type == 'balance_mismatch':
                    evidence['status'] = db_balance_status
                    if row['balance_diff_amount'] is not None:
                        evidence['difference'] = f"${abs(row['balance_diff_amount']):.2f}"
                    if row['beginning_balance'] is not None:
                        evidence['beginning_balance'] = f"${row['beginning_balance']:.2f}"
                    if row['ending_balance'] is not None:
                        evidence['ending_balance'] = f"${row['ending_balance']:.2f}"
                    if row['calculated_balance'] is not None:
                        evidence['calculated_balance'] = f"${row['calculated_balance']:.2f}"
                    issue_lines = []
                    for flag in matching_flags:
                        issue_lines.append(flag.get('description', ''))
                        for detail in flag.get('details', []):
                            issue_lines.append(f"  – {detail}")
                    if issue_lines:
                        evidence['issues'] = issue_lines

                elif anomaly_type == 'duplicate_lines':
                    duplicate_texts = []
                    for flag in matching_flags:
                        duplicate_texts.extend(flag.get('details', []))
                    evidence['duplicate_texts'] = duplicate_texts
                    evidence['count'] = len(duplicate_texts)
                    if matching_flags:
                        evidence['severity'] = matching_flags[0].get('severity', 'medium')

                elif anomaly_type == 'page_discontinuity':
                    for flag in matching_flags:
                        evidence['details'] = flag.get('details', [])
                        evidence['found_pages'] = flag.get('found_pages', [])
                        evidence['declared_max'] = flag.get('declared_max')
                        evidence['actual_count'] = flag.get('actual_count')
                        evidence['severity'] = flag.get('severity', 'medium')
                        break

                else:
                    issue_lines = []
                    for flag in matching_flags:
                        issue_lines.append(flag.get('description', ''))
                        for detail in flag.get('details', []):
                            issue_lines.append(f"  – {detail}")
                    if issue_lines:
                        evidence['issues'] = issue_lines
                    if matching_flags:
                        evidence['severity'] = matching_flags[0].get('severity', 'medium')

                anomaly_detector_evidence[anomaly_type] = evidence

            anomaly_detector_evidence['_balance_status'] = db_balance_status

        conn.close()
    except Exception as e:
        logger.error(f"Failed to query anomaly-detector database: {e}")

    def _build_description(anomaly, evidence):
        """Build a human-readable description from structured evidence."""
        if anomaly == 'balance_mismatch':
            if evidence.get('status') == 'PASS':
                return "Balance check passed — no arithmetic mismatch found in this document."
            lines = []
            if evidence.get('beginning_balance'):
                lines.append(f"Beginning Balance: {evidence['beginning_balance']}")
            if evidence.get('ending_balance'):
                lines.append(f"Ending Balance:    {evidence['ending_balance']}")
            if evidence.get('calculated_balance'):
                lines.append(f"Calculated Total:  {evidence['calculated_balance']}")
            if evidence.get('difference'):
                lines.append(f"Discrepancy:       {evidence['difference']}")
            header = "\n".join(f"• {l}" for l in lines) if lines else ""
            specific = "\n".join(f"• {i}" for i in evidence.get('issues', []))
            return (header + "\n\n" + specific).strip() if specific else (header or "Balance mismatch detected — specific amounts unavailable.")

        elif anomaly == 'duplicate_lines':
            texts = evidence.get('duplicate_texts', [])
            count = evidence.get('count', len(texts))
            if texts:
                quoted = "\n".join(f'  "{t}"' for t in texts[:10])
                return f"Found {count} duplicate line(s):\n{quoted}"
            return f"Found {count} duplicate transaction line(s)."

        elif anomaly == 'page_discontinuity':
            details = evidence.get('details', [])
            found = evidence.get('found_pages', [])
            declared = evidence.get('declared_max')
            actual = evidence.get('actual_count')
            lines = list(details)
            if found and declared:
                expected = set(range(1, declared + 1))
                missing = sorted(expected - set(found))
                if missing:
                    lines.append(f"Pages present in headers: {found}")
                    lines.append(f"Expected (1–{declared}): {missing} are missing")
            if actual and declared and actual != declared:
                lines.append(f"PDF has {actual} physical pages, headers say 1 of {declared}")
            return "\n".join(f"• {l}" for l in lines) if lines else "Page numbering inconsistencies detected."

        else:
            issues = evidence.get('issues', [])
            return "\n".join(f"• {i}" for i in issues) if issues else f"Flagged with: {anomaly.replace('_', ' ').title()}"

    # Add standard anomaly tags with real evidence from anomaly-detector
    anomalies = analysis.get('anomalies_found', [])
    db_balance_status = anomaly_detector_evidence.get('_balance_status')

    for anomaly in anomalies:
        if any(tag.get('tag', '').endswith(anomaly) for tag in enhanced_tags):
            continue

        if anomaly in anomaly_detector_evidence:
            real_evidence = anomaly_detector_evidence[anomaly]
            description = _build_description(anomaly, real_evidence)
            explanation = ANOMALY_EXPLANATIONS.get(anomaly, {
                'category': 'Anomaly Detection',
                'severity': real_evidence.get('severity', 'medium')
            })
            severity = explanation.get('severity', 'medium')
            if anomaly == 'balance_mismatch' and real_evidence.get('status') == 'PASS':
                severity = 'info'

            all_tags.append({
                'tag': f'anomaly:{anomaly}',
                'category': explanation.get('category', 'Anomaly Detection'),
                'description': description,
                'severity': severity,
                'evidence': real_evidence
            })

        elif anomaly == 'balance_mismatch' and db_balance_status == 'PASS':
            all_tags.append({
                'tag': f'anomaly:{anomaly}',
                'category': 'Financial Integrity',
                'description': "Balance check passed — no arithmetic mismatch found in this document.",
                'severity': 'info',
                'evidence': {'status': 'PASS'}
            })

        else:
            explanation = ANOMALY_EXPLANATIONS.get(anomaly, {
                'category': 'Anomaly Detection',
                'description': f'This document was flagged with: {anomaly.replace("_", " ").title()}',
                'severity': 'medium'
            })
            all_tags.append({
                'tag': f'anomaly:{anomaly}',
                'category': explanation['category'],
                'description': explanation['description'],
                'severity': explanation['severity'],
                'evidence': {}
            })

    return jsonify({
        'document_id': doc_id,
        'document_title': analysis.get('document_title') or analysis.get('title', 'Unknown'),
        'tags': all_tags,
        'integrity_summary': analysis.get('integrity_summary', ''),
        'issue_count': analysis.get('issue_count', 0) + len(anomalies),
        'critical_count': analysis.get('critical_count', 0)
    })
