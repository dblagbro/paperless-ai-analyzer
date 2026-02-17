"""
Web UI for Paperless AI Analyzer

Simple Flask-based dashboard for monitoring and control.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from flask import Flask, render_template, jsonify, request
from threading import Thread, Lock
from collections import deque

logger = logging.getLogger(__name__)

# In-memory log buffer
log_buffer = deque(maxlen=200)  # Keep last 200 log lines


class LogBufferHandler(logging.Handler):
    """Custom log handler that stores logs in memory."""

    def emit(self, record):
        try:
            msg = self.format(record)
            log_buffer.append(msg)
        except Exception:
            self.handleError(record)

# Global state
ui_state = {
    'recent_analyses': [],
    'stats': {
        'total_analyzed': 0,
        'anomalies_detected': 0,
        'profiles_matched': 0,
        'profiles_needed': 0,
        'high_risk_count': 0
    },
    'last_update': None,
    'lock': Lock()
}

app = Flask(__name__, template_folder='/app/analyzer/templates', static_folder='/app/analyzer/static')

# Configure session secret key (v1.5.0 for project tracking)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')


def create_app(state_manager, profile_loader, paperless_client,
               project_manager=None, llm_client=None, smart_uploader=None, document_analyzer=None):
    """
    Create and configure Flask app.

    Args:
        state_manager: StateManager instance
        profile_loader: ProfileLoader instance
        paperless_client: PaperlessClient instance
        project_manager: ProjectManager instance (v1.5.0)
        llm_client: LLM client instance (v1.5.0)
        smart_uploader: SmartUploader instance (v1.5.0)
        document_analyzer: DocumentAnalyzer instance (v1.5.0 - for re-analysis)

    Returns:
        Flask app
    """
    app.state_manager = state_manager
    app.profile_loader = profile_loader
    app.paperless_client = paperless_client
    app.document_analyzer = document_analyzer

    # v1.5.0: Project management
    app.project_manager = project_manager
    app.llm_client = llm_client
    app.smart_uploader = smart_uploader

    return app


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Get current analyzer status."""
    with ui_state['lock']:
        state_stats = app.state_manager.get_stats()

        # Get vector store stats
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore()
        vector_stats = vector_store.get_stats() if vector_store.enabled else {'enabled': False, 'total_documents': 0}

        return jsonify({
            'status': 'running',
            'uptime_seconds': _get_uptime(),
            'state': state_stats,
            'stats': ui_state['stats'],
            'last_update': ui_state['last_update'],
            'active_profiles': len(app.profile_loader.profiles),
            'vector_store': vector_stats
        })


@app.route('/api/recent')
def api_recent():
    """Get recent analysis results."""
    with ui_state['lock']:
        return jsonify({
            'analyses': ui_state['recent_analyses'][-50:]  # Last 50
        })


@app.route('/api/profiles')
def api_profiles():
    """Get profile information."""
    active_profiles = []
    for profile in app.profile_loader.profiles:
        active_profiles.append({
            'id': profile.profile_id,
            'name': profile.display_name,
            'version': profile.version,
            'checks': profile.checks_enabled
        })

    # Get staging profiles
    staging_dir = Path('/app/profiles/staging')
    staging_profiles = []
    if staging_dir.exists():
        for profile_file in staging_dir.glob('*.yaml'):
            staging_profiles.append({
                'filename': profile_file.name,
                'created': datetime.fromtimestamp(profile_file.stat().st_mtime).isoformat(),
                'size': profile_file.stat().st_size
            })

    return jsonify({
        'active': active_profiles,
        'staging': staging_profiles
    })


@app.route('/api/staging/<filename>')
def api_staging_profile(filename):
    """Get staging profile content."""
    staging_file = Path('/app/profiles/staging') / filename
    if not staging_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        import yaml
        with open(staging_file, 'r') as f:
            content = yaml.safe_load(f)
        return jsonify(content)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/staging/<filename>/activate', methods=['POST'])
def api_activate_staging_profile(filename):
    """Activate a staging profile by moving it to active profiles."""
    staging_file = Path('/app/profiles/staging') / filename
    if not staging_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        # Move from staging to active
        active_file = Path('/app/profiles/active') / filename
        staging_file.rename(active_file)

        logger.info(f"Activated staging profile: {filename}")

        return jsonify({
            'success': True,
            'message': f'Profile "{filename}" activated! Restart the analyzer to load it.'
        })
    except Exception as e:
        logger.error(f"Failed to activate profile {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/staging/activate-all', methods=['POST'])
def api_activate_all_staging_profiles():
    """Activate all staging profiles at once."""
    staging_dir = Path('/app/profiles/staging')
    if not staging_dir.exists():
        return jsonify({'error': 'Staging directory not found'}), 404

    results = {'success': [], 'failed': []}

    for profile_file in staging_dir.glob('*.yaml'):
        try:
            active_file = Path('/app/profiles/active') / profile_file.name
            profile_file.rename(active_file)
            results['success'].append(profile_file.name)
            logger.info(f"Activated staging profile: {profile_file.name}")
        except Exception as e:
            results['failed'].append({'filename': profile_file.name, 'error': str(e)})
            logger.error(f"Failed to activate {profile_file.name}: {e}")

    return jsonify({
        'success': True,
        'activated': len(results['success']),
        'failed': len(results['failed']),
        'details': results,
        'message': f"Activated {len(results['success'])} profiles. Restart analyzer to load them."
    })


@app.route('/api/staging/<filename>/delete', methods=['POST'])
def api_delete_staging_profile(filename):
    """Delete a staging profile."""
    staging_file = Path('/app/profiles/staging') / filename
    if not staging_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        staging_file.unlink()
        logger.info(f"Deleted staging profile: {filename}")

        return jsonify({
            'success': True,
            'message': f'Profile "{filename}" deleted'
        })
    except Exception as e:
        logger.error(f"Failed to delete profile {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active/<filename>', methods=['GET'])
def api_get_active_profile(filename):
    """Get an active profile's content."""
    active_file = Path('/app/profiles/active') / filename
    if not active_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        import yaml
        with open(active_file, 'r') as f:
            profile_data = yaml.safe_load(f)
        return jsonify(profile_data)
    except Exception as e:
        logger.error(f"Failed to read profile {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active/<filename>/rename', methods=['POST'])
def api_rename_active_profile(filename):
    """Rename an active profile (update display_name)."""
    active_file = Path('/app/profiles/active') / filename
    if not active_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        import yaml
        data = request.json
        new_name = data.get('display_name', '').strip()

        if not new_name:
            return jsonify({'error': 'Display name required'}), 400

        # Load, update, save
        with open(active_file, 'r') as f:
            profile_data = yaml.safe_load(f)

        profile_data['display_name'] = new_name

        with open(active_file, 'w') as f:
            yaml.dump(profile_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Renamed active profile {filename} to '{new_name}'")

        return jsonify({
            'success': True,
            'message': f'Profile renamed to "{new_name}"'
        })
    except Exception as e:
        logger.error(f"Failed to rename profile {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active/<filename>/delete', methods=['POST'])
def api_delete_active_profile(filename):
    """Delete an active profile."""
    active_file = Path('/app/profiles/active') / filename
    if not active_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        active_file.unlink()
        logger.info(f"Deleted active profile: {filename}")

        return jsonify({
            'success': True,
            'message': f'Profile "{filename}" deleted'
        })
    except Exception as e:
        logger.error(f"Failed to delete profile {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active/duplicates', methods=['GET'])
def api_detect_duplicates():
    """Detect duplicate profiles in active directory."""
    active_dir = Path('/app/profiles/active')
    if not active_dir.exists():
        return jsonify({'duplicates': [], 'groups': []}), 200

    try:
        import yaml
        import hashlib
        from collections import defaultdict

        profiles = []

        # Load all active profiles
        for profile_file in active_dir.glob('*.yaml'):
            if profile_file.name in ['active', 'examples']:
                continue

            try:
                with open(profile_file, 'r') as f:
                    content = f.read()
                    profile_data = yaml.safe_load(content)

                profiles.append({
                    'filename': profile_file.name,
                    'profile_id': profile_data.get('profile_id', ''),
                    'display_name': profile_data.get('display_name', ''),
                    'content': content,
                    'content_hash': hashlib.md5(content.encode()).hexdigest(),
                    'keywords': set(profile_data.get('match', {}).get('keywords', {}).get('any', [])),
                    'checks': set(profile_data.get('checks_enabled', []))
                })
            except Exception as e:
                logger.warning(f"Failed to load profile {profile_file.name}: {e}")
                continue

        # Find duplicates using Union-Find to group similar profiles
        duplicate_groups = []

        # 1. Exact content duplicates
        content_map = defaultdict(list)
        for profile in profiles:
            content_map[profile['content_hash']].append(profile)

        for content_hash, group in content_map.items():
            if len(group) > 1:
                duplicate_groups.append({
                    'type': 'exact',
                    'reason': 'Identical file content',
                    'profiles': [{'filename': p['filename'], 'display_name': p['display_name']} for p in group]
                })

        # 2. Similar profiles (high keyword overlap) - use Union-Find to group properly
        # Build similarity graph
        similar_pairs = []
        exact_hashes = {p['content_hash'] for group in content_map.values() if len(group) > 1 for p in group}

        for i, p1 in enumerate(profiles):
            # Skip profiles already in exact duplicate groups
            if p1['content_hash'] in exact_hashes:
                continue

            for p2 in profiles[i+1:]:
                # Skip if same profile or already in exact duplicate
                if p1['content_hash'] == p2['content_hash'] or p2['content_hash'] in exact_hashes:
                    continue

                # Calculate keyword similarity
                if p1['keywords'] and p2['keywords']:
                    intersection = len(p1['keywords'] & p2['keywords'])
                    union = len(p1['keywords'] | p2['keywords'])
                    similarity = intersection / union if union > 0 else 0

                    # If >70% similar keywords and same checks, flag as similar
                    if similarity > 0.7 and p1['checks'] == p2['checks']:
                        similar_pairs.append((i, profiles[i+1:].index(p2) + i + 1, similarity))

        # Use Union-Find to group similar profiles
        if similar_pairs:
            parent = {i: i for i in range(len(profiles))}

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            # Union similar profiles
            for i, j, sim in similar_pairs:
                union(i, j)

            # Group profiles by their root parent
            groups = defaultdict(list)
            for i, profile in enumerate(profiles):
                if profile['content_hash'] not in exact_hashes:  # Skip exact duplicates
                    root = find(i)
                    groups[root].append((profile, i))

            # Create duplicate groups for similar profiles
            for root, group_profiles in groups.items():
                if len(group_profiles) > 1:
                    # Calculate average similarity
                    avg_sim = sum(sim for i, j, sim in similar_pairs
                                 if find(i) == root and find(j) == root) / max(len(similar_pairs), 1)

                    duplicate_groups.append({
                        'type': 'similar',
                        'reason': f'~{int(avg_sim * 100)}% keyword overlap, same checks',
                        'profiles': [{'filename': p['filename'], 'display_name': p['display_name']}
                                   for p, _ in group_profiles]
                    })

        return jsonify({
            'total_profiles': len(profiles),
            'duplicate_groups': len(duplicate_groups),
            'groups': duplicate_groups
        })

    except Exception as e:
        logger.error(f"Failed to detect duplicates: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/active/duplicates/remove', methods=['POST'])
def api_remove_duplicates():
    """Remove specified duplicate profiles."""
    try:
        data = request.json
        filenames = data.get('filenames', [])

        if not filenames:
            return jsonify({'error': 'No filenames provided'}), 400

        active_dir = Path('/app/profiles/active')
        removed = []
        failed = []

        for filename in filenames:
            profile_file = active_dir / filename
            if not profile_file.exists():
                failed.append({'filename': filename, 'error': 'File not found'})
                continue

            try:
                profile_file.unlink()
                removed.append(filename)
                logger.info(f"Removed duplicate profile: {filename}")
            except Exception as e:
                failed.append({'filename': filename, 'error': str(e)})
                logger.error(f"Failed to remove {filename}: {e}")

        # Auto-reload profiles after removal
        if removed and hasattr(app, 'profile_loader'):
            try:
                app.profile_loader.load_profiles()
                logger.info(f"Profiles reloaded after removing {len(removed)} duplicates")
                reload_msg = " Profiles automatically reloaded."
            except Exception as e:
                logger.error(f"Failed to reload profiles: {e}")
                reload_msg = " Please use the 'Reload Profiles' button to refresh."
        else:
            reload_msg = ""

        return jsonify({
            'success': True,
            'removed': len(removed),
            'failed': len(failed),
            'removed_files': removed,
            'failed_files': failed,
            'message': f'Removed {len(removed)} duplicate profiles.{reload_msg}'
        })

    except Exception as e:
        logger.error(f"Failed to remove duplicates: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/reload-profiles', methods=['POST'])
def api_reload_profiles():
    """Reload all active profiles without restarting the container."""
    try:
        if not hasattr(app, 'profile_loader'):
            return jsonify({'error': 'Profile loader not available'}), 500

        # Get count before reload
        old_count = len(app.profile_loader.profiles)

        # Reload profiles from disk
        app.profile_loader.load_profiles()

        # Get count after reload
        new_count = len(app.profile_loader.profiles)

        logger.info(f"Profiles reloaded: {old_count} → {new_count}")

        return jsonify({
            'success': True,
            'message': f'Profiles reloaded successfully',
            'old_count': old_count,
            'new_count': new_count,
            'profiles': [p.profile_id for p in app.profile_loader.profiles]
        })

    except Exception as e:
        logger.error(f"Failed to reload profiles: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat with AI about documents using RAG (semantic search)."""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        history = data.get('history', [])
        document_type = data.get('document_type', None)  # Optional filter by document type

        if not user_message:
            return jsonify({'error': 'Message required'}), 400

        # Get stats
        with ui_state['lock']:
            stats = ui_state['stats']

        # Use semantic search with vector store
        from analyzer.vector_store import VectorStore

        vector_store = VectorStore()

        logger.info(f"Vector store enabled: {vector_store.enabled}")

        if vector_store.enabled:
            vector_stats = vector_store.get_stats()
            logger.info(f"Vector store stats: {vector_stats}")

            # Semantic search for relevant documents
            # Use more results for summary queries, fewer for specific queries
            n_results = 30 if any(word in user_message.lower() for word in ['summary', 'all', 'comprehensive', 'total', 'overview']) else 15

            # Apply document type filter if specified
            if document_type and document_type != 'all':
                logger.info(f"Filtering by document type: {document_type}")
                relevant_docs = vector_store.search(user_message, n_results=n_results, document_type=document_type)
            else:
                relevant_docs = vector_store.search(user_message, n_results=n_results)

            logger.info(f"RAG: Search returned {len(relevant_docs)} documents")

            if relevant_docs:
                recent_analyses = relevant_docs
                logger.info(f"RAG: Using {len(relevant_docs)} semantically relevant documents")
            else:
                # If search returns nothing, fetch from Paperless as fallback
                logger.warning("RAG: No documents found via semantic search, falling back to Paperless query")
                with ui_state['lock']:
                    recent_analyses = ui_state['recent_analyses']
        else:
            logger.warning("Vector store not enabled, using in-memory analyses")
            # Fallback: fetch from memory or Paperless
            with ui_state['lock']:
                recent_analyses = ui_state['recent_analyses']

        # If we don't have analyses, fetch from Paperless
        if not recent_analyses or len(recent_analyses) < 5:
            try:
                # Get documents with analyzed tags
                paperless_client = app.paperless_client
                documents = paperless_client.session.get(
                    f'{paperless_client.base_url}/api/documents/',
                    params={
                        'tags__name__icontains': 'analyzed',
                        'ordering': '-modified',
                        'page_size': 50
                    }
                ).json()

                recent_analyses = []
                for doc in documents.get('results', [])[:20]:  # Limit to 20 for performance
                    doc_id = doc['id']

                    # Get full document details including notes
                    try:
                        full_doc = paperless_client.get_document(doc_id)
                        notes = full_doc.get('notes', '')

                        # Extract AI analysis from notes (it's the section after "🤖 AI ANOMALY ANALYSIS")
                        ai_analysis = ""
                        if "🤖 AI ANOMALY ANALYSIS" in notes:
                            # Get the latest AI analysis (last occurrence)
                            parts = notes.split("🤖 AI ANOMALY ANALYSIS")
                            if len(parts) > 1:
                                ai_analysis = parts[-1].split("---")[0].strip()[:1000]  # First 1000 chars
                    except:
                        notes = ""
                        ai_analysis = ""

                    # Extract anomaly tags
                    tags = [paperless_client.session.get(
                        f'{paperless_client.base_url}/api/tags/{tag_id}/'
                    ).json().get('name', '') for tag_id in doc.get('tags', [])[:10]]

                    anomalies = [t.replace('anomaly:', '') for t in tags if t.startswith('anomaly:')]

                    # Determine risk score from tags
                    risk_score = 0
                    if 'anomaly:forensic_risk_high' in tags:
                        risk_score = 80
                    elif 'anomaly:forensic_risk_medium' in tags:
                        risk_score = 60
                    elif 'anomaly:forensic_risk_low' in tags:
                        risk_score = 30

                    recent_analyses.append({
                        'document_id': doc_id,
                        'document_title': doc['title'],
                        'anomalies_found': anomalies[:5],
                        'risk_score': risk_score,
                        'timestamp': doc['modified'],
                        'ai_analysis': ai_analysis,
                        'created': doc.get('created', ''),
                        'correspondent': doc.get('correspondent', None)
                    })

                logger.info(f"Fetched {len(recent_analyses)} analyzed documents from Paperless")
            except Exception as e:
                logger.error(f"Failed to fetch documents from Paperless: {e}")

        # Build context for AI
        context = f"""You are an AI assistant helping analyze financial documents.

Current Statistics:
- Total documents analyzed: {stats.get('total_analyzed', 0)}
- Anomalies detected: {stats.get('anomalies_detected', 0)}
- High risk documents: {stats.get('high_risk_count', 0)}

Recent Analyses:
"""
        for analysis in recent_analyses[-20:]:  # Last 20 analyses
            doc_id = analysis.get('document_id', 'Unknown')
            doc_title = analysis.get('document_title', 'Unknown')
            anomalies = analysis.get('anomalies_found', [])
            risk = analysis.get('risk_score', 0)
            context += f"\n- Doc {doc_id}: {doc_title}"
            if anomalies:
                context += f" | Anomalies: {', '.join(anomalies)}"
            context += f" | Risk: {risk}%"

        context += f"""

User's question: {user_message}

Provide a helpful, data-driven response based on the actual document content available."""

        # Build conversation history
        messages = []
        for msg in history[-10:]:  # Last 10 messages
            if msg.get('role') in ['user', 'assistant']:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

        # Add current user message
        messages.append({'role': 'user', 'content': user_message})

        # Use system parameter for context (Claude's preferred way)
        rag_mode = vector_store.enabled if 'vector_store' in locals() else False

        total_in_vector_store = vector_stats.get('total_documents', 0) if 'vector_stats' in locals() else 0

        # Build document type context and determine document count for prompt
        type_filter_text = ""
        filtered_count = total_in_vector_store
        filter_mode_text = "ALL documents"

        if document_type and document_type != 'all':
            type_breakdown = vector_stats.get('by_type', {}) if 'vector_stats' in locals() else {}
            filtered_count = type_breakdown.get(document_type, 0)
            type_filter_text = f"\n- **SEARCH IS FILTERED**: Only searching '{document_type}' documents"
            filter_mode_text = f"documents of type '{document_type}'"

        system_prompt = f"""You are an AI assistant analyzing financial documents in a document management system.

{'[RAG MODE: Semantic search active across ' + filter_mode_text + ']' if rag_mode else '[Standard mode]'}

CRITICAL - YOU HAVE ACCESS TO:
- **{filtered_count} documents** available for this query{type_filter_text}
- Below are the {len(recent_analyses)} MOST RELEVANT documents for this specific query
- Database contains {total_in_vector_store} total documents across all types

Current Statistics (for available documents):
- Documents in current scope: {filtered_count}
- Total across all types: {total_in_vector_store}
- Anomalies detected across all docs: {stats.get('anomalies_detected', 0)}
- High risk documents: {stats.get('high_risk_count', 0)}

{'Most Relevant Documents for This Query (from ' + str(filtered_count) + ' available):' if rag_mode else 'Recent Document Analyses:'}
"""
        if recent_analyses:
            for analysis in recent_analyses[-20:]:
                doc_id = analysis.get('document_id', 'Unknown')
                doc_title = analysis.get('document_title', 'Unknown')
                anomalies = analysis.get('anomalies_found', [])
                risk = analysis.get('risk_score', 0)
                timestamp = analysis.get('timestamp', 'Unknown')

                # Get content - vector store uses 'content', fallback uses 'ai_analysis'
                content = analysis.get('content', analysis.get('ai_analysis', ''))

                system_prompt += f"\n\n--- Document {doc_id} ---"
                system_prompt += f"\nTitle: {doc_title}"
                system_prompt += f"\nRisk Score: {risk}%"
                if anomalies:
                    system_prompt += f"\nAnomalies: {', '.join(anomalies)}"
                system_prompt += f"\nAnalyzed: {timestamp}"

                # Include full document content
                if content:
                    system_prompt += f"\n\nFull Document Analysis:\n{content}"
        else:
            system_prompt += "\n(No documents analyzed yet)"

        system_prompt += """

CRITICAL - NEVER HALLUCINATE DATA:
- If a document says "(Scanned image PDF with no extractable text)" or similar, DO NOT invent financial data
- NEVER make up dollar amounts, totals, sections, or line items that aren't explicitly in the content above
- If asked about specific numbers in a document with no extracted content, respond:
  "This document is a scanned image with no extracted text. I cannot analyze specific financial figures without OCR or Vision AI to read the document."
- Only report numbers and facts that are EXPLICITLY stated in the document content provided above
- If the content is missing or incomplete, acknowledge this limitation clearly

IMPORTANT INSTRUCTIONS:
When users ask for summaries or "all documents":
- You SHOULD provide comprehensive analysis based on the documents shown above
- These are the most relevant documents selected via semantic search
- Frame responses as "Based on analysis of [total] documents..."
- Provide statistics, patterns, and insights from the documents shown
- Be specific with numbers, document IDs, and findings
- DO NOT say you can't access documents - you have them above

When users ask specific questions:
- Reference the relevant documents from those shown above
- Provide data-driven insights based on actual analyses
- Be specific with document IDs, titles, risk scores, and anomalies
- ONLY use information explicitly present in the document content above

FORMATTING REQUIREMENTS:
- Use markdown formatting for better readability
- Use bullet points (-) for lists
- Use **bold** for important information
- Use tables when presenting structured data
- Use line breaks between sections
- Example table format:
  | Column 1 | Column 2 |
  |----------|----------|
  | Data 1   | Data 2   |

LEDGER AND REPORT GENERATION:
- When asked to generate ledgers, reports, or summaries, use ALL available data from documents
- Extract account numbers, balances, dates, and transaction details from document content
- Create tables with available information, clearly noting any gaps or missing data
- If some documents lack certain information, work with what's available and note limitations
- Provide the most complete analysis possible given the data you can access"""

        # Load AI configuration and try providers/models in order
        ai_config = load_ai_config()
        chat_config = ai_config.get('chat', {})
        providers = chat_config.get('providers', [])

        ai_response = None
        last_error = None
        attempted = []

        # Try each enabled provider and their models
        for provider_config in providers:
            if not provider_config.get('enabled', False):
                continue

            provider_name = provider_config.get('name')
            api_key = provider_config.get('api_key', '').strip()
            models = provider_config.get('models', [])

            if not api_key:
                logger.warning(f"Skipping {provider_name}: No API key configured")
                continue

            # Initialize provider client
            try:
                if provider_name == 'openai':
                    import openai
                    client = openai.OpenAI(api_key=api_key)

                    # Try each model for this provider
                    for model in models:
                        try:
                            logger.info(f"Trying chat: OpenAI {model}")
                            attempted.append(f"OpenAI {model}")

                            response = client.chat.completions.create(
                                model=model,
                                messages=[{"role": "system", "content": system_prompt}] + messages,
                                max_tokens=4096  # Increased for legal/court documents - comprehensive answers
                            )
                            ai_response = response.choices[0].message.content
                            logger.info(f"✓ Successfully used: OpenAI {model}")
                            break
                        except Exception as e:
                            error_str = str(e)
                            if '404' in error_str or 'model_not_found' in error_str:
                                logger.warning(f"Model {model} not available, trying next...")
                                last_error = e
                                continue
                            else:
                                logger.warning(f"Error with OpenAI {model}: {e}")
                                last_error = e
                                continue

                elif provider_name == 'anthropic':
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)

                    # Try each model for this provider
                    for model in models:
                        try:
                            logger.info(f"Trying chat: Anthropic {model}")
                            attempted.append(f"Anthropic {model}")

                            response = client.messages.create(
                                model=model,
                                max_tokens=4096,  # Increased for legal/court documents - comprehensive answers
                                system=system_prompt,
                                messages=messages
                            )
                            ai_response = response.content[0].text
                            logger.info(f"✓ Successfully used: Anthropic {model}")
                            break
                        except Exception as e:
                            error_str = str(e)
                            if '404' in error_str or 'not_found' in error_str:
                                logger.warning(f"Model {model} not available, trying next...")
                                last_error = e
                                continue
                            else:
                                logger.warning(f"Error with Anthropic {model}: {e}")
                                last_error = e
                                continue

                if ai_response:
                    break  # Success, stop trying providers

            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} client: {e}")
                last_error = e
                continue

        if ai_response is None:
            attempted_str = ", ".join(attempted) if attempted else "no models"
            raise Exception(f"No available models found. Tried: {attempted_str}. Last error: {last_error}")

        logger.info(f"Chat query: {user_message[:100]}")

        return jsonify({
            'response': ai_response,
            'success': True
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/types', methods=['GET'])
def api_vector_types():
    """Get list of all document types in vector store."""
    try:
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore()

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        document_types = vector_store.get_document_types()
        stats = vector_store.get_stats()

        return jsonify({
            'success': True,
            'document_types': document_types,
            'by_type': stats.get('by_type', {})
        })

    except Exception as e:
        logger.error(f"Failed to get document types: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/delete/<int:document_id>', methods=['POST'])
def api_vector_delete_document(document_id):
    """Delete a specific document from vector store."""
    try:
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore()

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        success = vector_store.delete_document(document_id)

        if success:
            return jsonify({
                'success': True,
                'message': f'Document {document_id} deleted from vector store'
            })
        else:
            return jsonify({'error': 'Failed to delete document'}), 500

    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/delete-document', methods=['POST'])
def api_vector_delete_document_json():
    """Delete a specific document from vector store (JSON body)."""
    try:
        data = request.json
        doc_id = data.get('doc_id')

        if not doc_id:
            return jsonify({'error': 'doc_id required'}), 400

        from analyzer.vector_store import VectorStore
        vector_store = VectorStore()

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        success = vector_store.delete_document(int(doc_id))

        if success:
            return jsonify({
                'success': True,
                'message': f'Document {doc_id} deleted from vector store'
            })
        else:
            return jsonify({'error': 'Failed to delete document'}), 500

    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/delete-by-type', methods=['POST'])
def api_vector_delete_by_type():
    """Delete all documents of a specific type from vector store."""
    try:
        data = request.json
        document_type = data.get('document_type', '').strip()

        if not document_type:
            return jsonify({'error': 'document_type required'}), 400

        from analyzer.vector_store import VectorStore
        vector_store = VectorStore()

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        count = vector_store.delete_by_type(document_type)

        return jsonify({
            'success': True,
            'message': f'Deleted {count} documents of type "{document_type}"',
            'count': count
        })

    except Exception as e:
        logger.error(f"Failed to delete documents by type: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/clear', methods=['POST'])
def api_vector_clear():
    """Clear all documents from vector store."""
    try:
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore()

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        success = vector_store.clear()

        if success:
            return jsonify({
                'success': True,
                'message': 'Vector store cleared successfully'
            })
        else:
            return jsonify({'error': 'Failed to clear vector store'}), 500

    except Exception as e:
        logger.error(f"Failed to clear vector store: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/documents', methods=['GET'])
def api_vector_documents():
    """Get all documents from vector store with details, grouped by type."""
    try:
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore()

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        # Get all documents from ChromaDB
        try:
            all_docs = vector_store.collection.get(include=['metadatas'])

            # Group by document type
            documents_by_type = {}
            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i]
                doc_type = metadata.get('document_type', 'unknown')

                if doc_type not in documents_by_type:
                    documents_by_type[doc_type] = []

                documents_by_type[doc_type].append({
                    'doc_id': metadata.get('document_id'),
                    'title': metadata.get('title', 'Unknown'),
                    'risk_score': metadata.get('risk_score', 0),
                    'timestamp': metadata.get('timestamp', '')
                })

            # Sort documents within each type by doc_id
            for doc_type in documents_by_type:
                documents_by_type[doc_type].sort(key=lambda x: x['doc_id'])

            return jsonify({
                'success': True,
                'documents': documents_by_type
            })

        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.error(f"Failed to get vector documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings/poll-interval', methods=['POST'])
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


# AI Configuration System
AI_CONFIG_PATH = Path('/app/data/ai_config.json')

def get_default_ai_config():
    """Get default AI configuration with proper model priorities."""
    return {
        'document_analysis': {
            'providers': [
                {
                    'name': 'openai',
                    'api_key': os.environ.get('OPENAI_API_KEY', ''),
                    'models': ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
                    'enabled': bool(os.environ.get('OPENAI_API_KEY'))
                },
                {
                    'name': 'anthropic',
                    'api_key': os.environ.get('LLM_API_KEY', ''),
                    'models': ['claude-3-opus-20240229', 'claude-3-5-sonnet-20241022', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                    'enabled': bool(os.environ.get('LLM_API_KEY'))
                }
            ]
        },
        'chat': {
            'providers': [
                {
                    'name': 'openai',
                    'api_key': os.environ.get('OPENAI_API_KEY', ''),
                    'models': ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
                    'enabled': bool(os.environ.get('OPENAI_API_KEY'))
                },
                {
                    'name': 'anthropic',
                    'api_key': os.environ.get('LLM_API_KEY', ''),
                    'models': ['claude-3-opus-20240229', 'claude-3-5-sonnet-20241022', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                    'enabled': bool(os.environ.get('LLM_API_KEY'))
                }
            ]
        }
    }

def load_ai_config():
    """Load AI configuration from file, or return defaults."""
    try:
        if AI_CONFIG_PATH.exists():
            with open(AI_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                logger.info("Loaded AI configuration from file")
                return config
    except Exception as e:
        logger.warning(f"Failed to load AI config: {e}")

    return get_default_ai_config()

def save_ai_config(config):
    """Save AI configuration to file."""
    try:
        AI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AI_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Saved AI configuration")
        return True
    except Exception as e:
        logger.error(f"Failed to save AI config: {e}")
        return False


@app.route('/api/ai-config', methods=['GET'])
def api_ai_config_get():
    """Get current AI configuration."""
    try:
        config = load_ai_config()
        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        logger.error(f"Failed to get AI config: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-config', methods=['POST'])
def api_ai_config_save():
    """Save AI configuration."""
    try:
        data = request.json
        config = data.get('config')

        if not config:
            return jsonify({'error': 'Configuration is required'}), 400

        # Validate structure
        if 'document_analysis' not in config or 'chat' not in config:
            return jsonify({'error': 'Invalid configuration structure'}), 400

        # Save configuration
        if save_ai_config(config):
            return jsonify({
                'success': True,
                'message': 'AI configuration saved. Changes will apply to new analysis tasks.'
            })
        else:
            return jsonify({'error': 'Failed to save configuration'}), 500

    except Exception as e:
        logger.error(f"Failed to save AI config: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-config/test', methods=['POST'])
def api_ai_config_test():
    """Test an AI provider/model configuration."""
    try:
        data = request.json
        provider = data.get('provider')
        api_key = data.get('api_key', '').strip()
        model = data.get('model', '')

        if not provider or not api_key:
            return jsonify({'error': 'Provider and API key are required'}), 400

        if provider == 'openai':
            import openai
            client = openai.OpenAI(api_key=api_key)

            # Try the specified model or default
            test_model = model if model else 'gpt-3.5-turbo'
            try:
                response = client.chat.completions.create(
                    model=test_model,
                    messages=[{"role": "user", "content": "Say 'test successful'"}],
                    max_tokens=10
                )
                return jsonify({
                    'success': True,
                    'message': f'✓ OpenAI API key is valid! Using model: {test_model}',
                    'model': test_model
                })
            except Exception as e:
                error_msg = str(e)
                if '404' in error_msg or 'model_not_found' in error_msg:
                    return jsonify({
                        'success': False,
                        'error': f'Model {test_model} not available with this API key'
                    }), 400
                raise

        elif provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            # Try models in order
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


@app.route('/api/trigger', methods=['POST'])
def api_trigger():
    """Manually trigger analysis of a document."""
    data = request.json
    doc_id = data.get('doc_id')

    if not doc_id:
        return jsonify({'error': 'doc_id required'}), 400

    try:
        # Verify document exists
        doc = app.paperless_client.get_document(doc_id)

        # Queue for analysis (in a real implementation, you'd queue this)
        # For now, just return success
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


@app.route('/api/logs')
def api_logs():
    """Get recent log entries from the running process."""
    try:
        limit = int(request.args.get('limit', '100'))

        # Get state info for header
        state_stats = app.state_manager.get_stats()

        logs = []

        # Add header with status
        logs.append(f"=== Analyzer Status (Updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}) ===")
        logs.append(f"Last run: {state_stats.get('last_run', 'Never')}")
        logs.append(f"Documents processed: {state_stats.get('total_documents_processed', 0)}")
        logs.append(f"Active profiles: {len(app.profile_loader.profiles)}")
        logs.append(f"LLM enabled: {os.environ.get('LLM_ENABLED', 'false')}")
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


@app.route('/api/reprocess', methods=['POST'])
def api_reprocess():
    """Reset state and reprocess all documents."""
    try:
        # Delete state file to force reprocessing
        state_file = Path('/app/data/state.json')
        if state_file.exists():
            state_file.unlink()
            logger.info("State file deleted - will reprocess all documents on next poll")

        # Reset in-memory state using proper method
        app.state_manager.reset()

        return jsonify({
            'success': True,
            'message': 'State reset - all documents will be reprocessed on next poll cycle'
        })
    except Exception as e:
        logger.error(f"Failed to reset state: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reprocess/<int:doc_id>', methods=['POST'])
def api_reprocess_document(doc_id):
    """Reprocess a specific document by removing it from state."""
    try:
        # Remove document from seen_ids so it will be reprocessed
        if hasattr(app.state_manager.state, 'last_seen_ids'):
            if doc_id in app.state_manager.state.get('last_seen_ids', []):
                app.state_manager.state['last_seen_ids'].remove(doc_id)
                app.state_manager.save_state()

        return jsonify({
            'success': True,
            'message': f'Document {doc_id} will be reprocessed on next poll cycle'
        })
    except Exception as e:
        logger.error(f"Failed to reprocess document {doc_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/status')
def api_llm_status():
    """Get LLM configuration status."""
    import os
    enabled = os.environ.get('LLM_ENABLED', 'false').lower() == 'true'
    provider = os.environ.get('LLM_PROVIDER', 'anthropic')
    has_key = bool(os.environ.get('LLM_API_KEY'))

    return jsonify({
        'enabled': enabled,
        'provider': provider,
        'has_key': has_key,
        'setup_url': 'https://console.anthropic.com/settings/keys' if provider == 'anthropic' else 'https://platform.openai.com/api-keys'
    })


@app.route('/api/llm/test', methods=['POST'])
def api_llm_test():
    """Test an LLM API key."""
    data = request.json
    provider = data.get('provider', 'anthropic')
    api_key = data.get('api_key', '').strip()

    if not api_key:
        return jsonify({'success': False, 'error': 'API key is required'}), 400

    try:
        if provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Try models in order of preference
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

            # If we got here, all models failed
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


@app.route('/api/llm/save', methods=['POST'])
def api_llm_save():
    """Save LLM configuration and restart container."""
    data = request.json
    provider = data.get('provider', 'anthropic')
    api_key = data.get('api_key', '').strip()

    if not api_key:
        return jsonify({'success': False, 'error': 'API key is required'}), 400

    try:
        # Update docker-compose.yml
        compose_file = Path('/docker-compose.yml')

        if compose_file.exists():
            with open(compose_file, 'r') as f:
                content = f.read()

            # Replace LLM settings
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


@app.route('/api/search')
def api_search():
    """Search analysis results."""
    query = request.args.get('q', '').lower()
    risk_min = request.args.get('risk_min', type=int)
    risk_max = request.args.get('risk_max', type=int)
    has_anomalies = request.args.get('has_anomalies', type=bool)

    with ui_state['lock']:
        results = ui_state['recent_analyses']

        # Filter by query (search title, doc_id, anomalies, and summaries)
        if query:
            results = [r for r in results if
                      query in r.get('title', '').lower() or
                      query in str(r.get('doc_id', '')).lower() or
                      query in r.get('brief_summary', '').lower() or
                      query in r.get('full_summary', '').lower() or
                      any(query in a.lower() for a in r.get('anomalies_found', []))]

        # Filter by risk score
        if risk_min is not None:
            results = [r for r in results if r.get('risk_score', 0) >= risk_min]
        if risk_max is not None:
            results = [r for r in results if r.get('risk_score', 0) <= risk_max]

        # Filter by anomalies
        if has_anomalies:
            results = [r for r in results if r.get('anomalies_found')]

        return jsonify({
            'results': results,
            'count': len(results)
        })


@app.route('/api/tag-evidence/<int:doc_id>')
def api_tag_evidence(doc_id):
    """
    Get enhanced tag evidence for a specific document.
    Returns detailed information about why each tag was flagged.
    """
    with ui_state['lock']:
        # Find the analysis for this document
        analysis = None
        for result in ui_state['recent_analyses']:
            if result.get('doc_id') == doc_id:
                analysis = result
                break

        if not analysis:
            return jsonify({'error': 'Document not found'}), 404

        # Return enhanced tags with evidence
        enhanced_tags = analysis.get('enhanced_tags', [])

        return jsonify({
            'document_id': doc_id,
            'document_title': analysis.get('title', 'Unknown'),
            'tags': enhanced_tags,
            'integrity_summary': analysis.get('integrity_summary', ''),
            'issue_count': analysis.get('issue_count', 0),
            'critical_count': analysis.get('critical_count', 0)
        })


# ==================== Project Management API (v1.5.0) ====================

@app.route('/api/projects', methods=['GET'])
def api_list_projects():
    """List all projects."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        include_archived = request.args.get('archived', type=bool, default=False)
        projects = app.project_manager.list_projects(include_archived=include_archived)

        # Add statistics to each project
        for project in projects:
            try:
                stats = app.project_manager.get_statistics(project['slug'])
                project.update(stats)
            except Exception:
                pass  # Skip stats if unavailable

        return jsonify({'projects': projects, 'count': len(projects)})

    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects', methods=['POST'])
def api_create_project():
    """Create new project."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        data = request.json

        if not data.get('name'):
            return jsonify({'error': 'Project name is required'}), 400

        # Generate slug if not provided
        slug = data.get('slug')
        if not slug:
            slug = app.project_manager.suggest_slug(data['name'])

        # Create project
        project = app.project_manager.create_project(
            slug=slug,
            name=data['name'],
            description=data.get('description', ''),
            color=data.get('color'),
            metadata=data.get('metadata')
        )

        # Create Paperless tag
        if app.paperless_client:
            app.paperless_client.get_or_create_tag(
                f"project:{slug}",
                color=project.get('color', '#3498db')
            )

        logger.info(f"Created project: {slug}")
        return jsonify(project), 201

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>', methods=['GET'])
def api_get_project(slug):
    """Get project details."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        project = app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        # Add statistics
        stats = app.project_manager.get_statistics(slug)
        project.update(stats)

        return jsonify(project)

    except Exception as e:
        logger.error(f"Failed to get project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>', methods=['PUT'])
def api_update_project(slug):
    """Update project metadata."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        data = request.json
        project = app.project_manager.update_project(slug, **data)

        logger.info(f"Updated project: {slug}")
        return jsonify(project)

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to update project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>', methods=['DELETE'])
def api_delete_project(slug):
    """Delete project."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        delete_data = request.args.get('delete_data', type=bool, default=True)

        # Delete vector store collection if requested
        if delete_data:
            from analyzer.vector_store import VectorStore
            vs = VectorStore(project_slug=slug)
            if vs.enabled:
                vs.delete_collection()

        # Delete project
        success = app.project_manager.delete_project(slug, delete_data=delete_data)

        if success:
            logger.info(f"Deleted project: {slug}")
            return jsonify({'success': True, 'message': f'Project {slug} deleted'})
        else:
            return jsonify({'error': 'Failed to delete project'}), 500

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to delete project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/archive', methods=['POST'])
def api_archive_project(slug):
    """Archive project."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        project = app.project_manager.archive_project(slug)
        logger.info(f"Archived project: {slug}")
        return jsonify(project)

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to archive project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/unarchive', methods=['POST'])
def api_unarchive_project(slug):
    """Unarchive project."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        project = app.project_manager.unarchive_project(slug)
        logger.info(f"Unarchived project: {slug}")
        return jsonify(project)

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to unarchive project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/current-project', methods=['GET'])
def api_get_current_project():
    """Get currently selected project from session."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        from flask import session
        project_slug = session.get('current_project', 'default')
        project = app.project_manager.get_project(project_slug)

        if not project:
            # Fallback to default
            project = app.project_manager.get_project('default')
            session['current_project'] = 'default'

        return jsonify(project)

    except Exception as e:
        logger.error(f"Failed to get current project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/current-project', methods=['POST'])
def api_set_current_project():
    """Set current project in session."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        from flask import session
        data = request.json
        project_slug = data.get('project_slug')

        if not project_slug:
            return jsonify({'error': 'project_slug is required'}), 400

        # Validate project exists
        project = app.project_manager.get_project(project_slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        session['current_project'] = project_slug
        logger.info(f"Switched to project: {project_slug}")

        return jsonify(project)

    except Exception as e:
        logger.error(f"Failed to set current project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/orphan-documents', methods=['GET'])
def api_list_orphan_documents():
    """List documents without project tag."""
    if not app.paperless_client:
        return jsonify({'error': 'Paperless client not available'}), 503

    try:
        orphans = app.paperless_client.get_documents_without_project()

        # Format for UI
        orphan_list = []
        for doc in orphans[:100]:  # Limit to 100 for performance
            orphan_list.append({
                'id': doc['id'],
                'title': doc['title'],
                'created': doc.get('created'),
                'correspondent': doc.get('correspondent'),
                'tags': [t['name'] for t in doc.get('tags', [])]
            })

        return jsonify({'orphans': orphan_list, 'count': len(orphan_list)})

    except Exception as e:
        logger.error(f"Failed to get orphan documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/assign-project', methods=['POST'])
def api_assign_project_to_documents():
    """Assign project to one or more documents."""
    if not app.paperless_client or not app.project_manager:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        data = request.json
        document_ids = data.get('document_ids', [])
        project_slug = data.get('project_slug')

        if not document_ids or not project_slug:
            return jsonify({'error': 'document_ids and project_slug are required'}), 400

        # Validate project
        project = app.project_manager.get_project(project_slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        # Add project tag to each document
        success_count = 0
        failed = []

        for doc_id in document_ids:
            if app.paperless_client.add_project_tag(doc_id, project_slug, color=project.get('color')):
                success_count += 1
            else:
                failed.append(doc_id)

        # Update project document count
        if success_count > 0:
            app.project_manager.increment_document_count(project_slug, delta=success_count)

        result = {
            'success': True,
            'assigned': success_count,
            'total': len(document_ids)
        }

        if failed:
            result['failed'] = failed

        logger.info(f"Assigned {success_count}/{len(document_ids)} documents to project '{project_slug}'")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Failed to assign documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/reanalyze', methods=['POST'])
def api_reanalyze_project(slug):
    """Manually trigger re-analysis of all documents in a project with full context."""
    if not app.project_manager or not app.document_analyzer:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        # Validate project exists
        project = app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        logger.info(f"Manual re-analysis requested for project '{slug}'")

        # Trigger re-analysis in background thread to avoid blocking
        from threading import Thread
        def run_reanalysis():
            try:
                app.document_analyzer.re_analyze_project(slug)
            except Exception as e:
                logger.error(f"Background re-analysis failed: {e}")

        thread = Thread(target=run_reanalysis, daemon=True)
        thread.start()

        # Return immediately
        return jsonify({
            'success': True,
            'message': f'Re-analysis started for project: {slug}',
            'note': 'This runs in background and may take several minutes. Check logs for progress.'
        })

    except Exception as e:
        logger.error(f"Failed to trigger re-analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/migrate-documents', methods=['POST'])
def api_migrate_documents():
    """Migrate documents from one project to another."""
    if not app.project_manager or not app.paperless_client:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        data = request.json
        source_slug = data.get('source_project')
        dest_slug = data.get('destination_project')
        document_ids = data.get('document_ids', [])  # Optional: specific documents, or all if empty

        if not source_slug or not dest_slug:
            return jsonify({'error': 'Both source_project and destination_project are required'}), 400

        if source_slug == dest_slug:
            return jsonify({'error': 'Source and destination projects must be different'}), 400

        # Validate both projects exist
        source_project = app.project_manager.get_project(source_slug)
        dest_project = app.project_manager.get_project(dest_slug)

        if not source_project:
            return jsonify({'error': f'Source project not found: {source_slug}'}), 404
        if not dest_project:
            return jsonify({'error': f'Destination project not found: {dest_slug}'}), 404

        logger.info(f"Migration requested: {source_slug} → {dest_slug} ({len(document_ids) if document_ids else 'all'} docs)")

        # Run migration in background thread
        from threading import Thread
        migration_result = {'status': 'running', 'migrated': 0, 'errors': 0}

        def run_migration():
            try:
                # Get or create project tags
                source_tag = f"project:{source_slug}"
                dest_tag = f"project:{dest_slug}"

                # Get source tag ID
                source_tag_id = app.paperless_client.get_or_create_tag(source_tag, color='#95a5a6')
                dest_tag_id = app.paperless_client.get_or_create_tag(dest_tag, color='#e74c3c')

                if not source_tag_id or not dest_tag_id:
                    raise Exception("Failed to get/create project tags")

                # Get documents to migrate
                if document_ids:
                    # Specific documents
                    docs_to_migrate = document_ids
                else:
                    # All documents with source project tag
                    all_docs = app.paperless_client.get_documents_by_tag(source_tag_id)
                    docs_to_migrate = [doc['id'] for doc in all_docs]

                migrated_count = 0
                error_count = 0

                for doc_id in docs_to_migrate:
                    try:
                        # Get current document
                        doc = app.paperless_client.get_document(doc_id)
                        if not doc:
                            error_count += 1
                            continue

                        current_tags = doc.get('tags', [])

                        # Remove source tag, add dest tag
                        updated_tags = [t for t in current_tags if t != source_tag_id]
                        if dest_tag_id not in updated_tags:
                            updated_tags.append(dest_tag_id)

                        # Update document tags
                        success = app.paperless_client.update_document(doc_id, {'tags': updated_tags})
                        if success:
                            migrated_count += 1
                        else:
                            error_count += 1

                    except Exception as e:
                        logger.error(f"Failed to migrate document {doc_id}: {e}")
                        error_count += 1

                # Update project counts
                app.project_manager.increment_document_count(source_slug, delta=-migrated_count)
                app.project_manager.increment_document_count(dest_slug, delta=migrated_count)

                migration_result['status'] = 'completed'
                migration_result['migrated'] = migrated_count
                migration_result['errors'] = error_count

                logger.info(f"Migration completed: {migrated_count} docs migrated, {error_count} errors")

            except Exception as e:
                logger.error(f"Migration failed: {e}")
                migration_result['status'] = 'failed'
                migration_result['error'] = str(e)

        thread = Thread(target=run_migration, daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'message': f'Migration started: {source_slug} → {dest_slug}',
            'note': 'Migration runs in background. Check logs for progress.'
        })

    except Exception as e:
        logger.error(f"Failed to start migration: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/from-url', methods=['POST'])
def api_upload_from_url():
    """Download file from URL and analyze with AI."""
    if not app.smart_uploader:
        return jsonify({'error': 'Smart upload not available (LLM disabled)'}), 503

    try:
        data = request.json
        url = data.get('url')
        auth_type = data.get('auth_type', 'none')
        username = data.get('username')
        password = data.get('password')
        token = data.get('token')
        custom_headers = data.get('custom_headers', {})

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Initialize downloader
        from analyzer.remote_downloader import RemoteFileDownloader
        downloader = RemoteFileDownloader()

        # Download file
        try:
            file_path, download_metadata = downloader.download_from_url(
                url=url,
                auth_type=auth_type,
                username=username,
                password=password,
                token=token,
                custom_headers=custom_headers
            )
        except Exception as e:
            return jsonify({'error': f'Download failed: {str(e)}'}), 400

        # Analyze document with AI
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            metadata = loop.run_until_complete(app.smart_uploader.analyze_document(file_path))
        finally:
            loop.close()

        # Add download metadata
        metadata['download_info'] = download_metadata

        # Clean up downloaded file
        downloader.cleanup(file_path)

        logger.info(f"Analyzed document from URL: {download_metadata['filename']}")
        return jsonify(metadata)

    except Exception as e:
        logger.error(f"Failed to upload from URL: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/analyze', methods=['POST'])
def api_analyze_upload():
    """Analyze uploaded file and extract metadata."""
    if not app.smart_uploader:
        return jsonify({'error': 'Smart upload not available (LLM disabled)'}), 503

    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save file temporarily
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # Analyze document
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            metadata = loop.run_until_complete(app.smart_uploader.analyze_document(file_path))
        finally:
            loop.close()

        # Clean up temp file
        os.remove(file_path)
        os.rmdir(temp_dir)

        logger.info(f"Analyzed upload: {file.filename}")
        return jsonify(metadata)

    except Exception as e:
        logger.error(f"Failed to analyze upload: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/submit', methods=['POST'])
def api_submit_upload():
    """Submit analyzed file to Paperless with metadata."""
    if not app.smart_uploader:
        return jsonify({'error': 'Smart upload not available (LLM disabled)'}), 503

    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get metadata from form data
        project_slug = request.form.get('project_slug')
        if not project_slug:
            return jsonify({'error': 'project_slug is required'}), 400

        # Parse metadata JSON from form
        import json
        metadata_json = request.form.get('metadata', '{}')
        metadata = json.loads(metadata_json)

        # Save file temporarily
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # Upload to Paperless
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                app.smart_uploader.upload_to_paperless(file_path, project_slug, metadata)
            )
        finally:
            loop.close()

        # Clean up temp file
        os.remove(file_path)
        os.rmdir(temp_dir)

        if result:
            logger.info(f"Uploaded {file.filename} to project '{project_slug}'")
            return jsonify({
                'success': True,
                'document_id': result.get('id'),
                'title': metadata.get('suggested_title')
            })
        else:
            return jsonify({'error': 'Upload failed'}), 500

    except Exception as e:
        logger.error(f"Failed to submit upload: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm-usage/stats', methods=['GET'])
def api_get_llm_usage_stats():
    """Get LLM usage statistics."""
    try:
        days = request.args.get('days', 30, type=int)

        if not hasattr(app, 'document_analyzer') or not app.document_analyzer:
            return jsonify({'error': 'Document analyzer not available'}), 503

        if not hasattr(app.document_analyzer, 'usage_tracker') or not app.document_analyzer.usage_tracker:
            return jsonify({'error': 'Usage tracker not available'}), 503

        stats = app.document_analyzer.usage_tracker.get_usage_stats(days=days)
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm-usage/recent', methods=['GET'])
def api_get_recent_llm_calls():
    """Get recent LLM API calls."""
    try:
        limit = request.args.get('limit', 50, type=int)

        if not hasattr(app, 'document_analyzer') or not app.document_analyzer:
            return jsonify({'error': 'Document analyzer not available'}), 503

        if not hasattr(app.document_analyzer, 'usage_tracker') or not app.document_analyzer.usage_tracker:
            return jsonify({'error': 'Usage tracker not available'}), 503

        calls = app.document_analyzer.usage_tracker.get_recent_calls(limit=limit)
        return jsonify({'calls': calls})

    except Exception as e:
        logger.error(f"Failed to get recent calls: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm-usage/pricing', methods=['GET'])
def api_get_llm_pricing():
    """Get current LLM pricing information."""
    try:
        if not hasattr(app, 'document_analyzer') or not app.document_analyzer:
            return jsonify({'error': 'Document analyzer not available'}), 503

        if not hasattr(app.document_analyzer, 'usage_tracker') or not app.document_analyzer.usage_tracker:
            return jsonify({'error': 'Usage tracker not available'}), 503

        pricing = app.document_analyzer.usage_tracker.get_pricing()
        return jsonify({'pricing': pricing})

    except Exception as e:
        logger.error(f"Failed to get pricing: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        # Check Paperless API
        healthy = app.paperless_client.health_check()

        if healthy:
            return jsonify({'status': 'healthy'}), 200
        else:
            return jsonify({'status': 'unhealthy', 'reason': 'paperless_api_unreachable'}), 503
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'reason': str(e)}), 503


def update_ui_stats(analysis_result: Dict[str, Any]) -> None:
    """
    Update UI statistics with analysis result.

    Args:
        analysis_result: Result from document analysis
    """
    with ui_state['lock']:
        # Add to recent analyses
        ui_state['recent_analyses'].append(analysis_result)

        # Keep only last 100
        if len(ui_state['recent_analyses']) > 100:
            ui_state['recent_analyses'] = ui_state['recent_analyses'][-100:]

        # Update stats
        ui_state['stats']['total_analyzed'] += 1

        if analysis_result.get('anomalies_found'):
            ui_state['stats']['anomalies_detected'] += len(analysis_result['anomalies_found'])

        if analysis_result.get('profile_matched'):
            ui_state['stats']['profiles_matched'] += 1
        else:
            ui_state['stats']['profiles_needed'] += 1

        if analysis_result.get('risk_score', 0) >= 70:
            ui_state['stats']['high_risk_count'] += 1

        ui_state['last_update'] = datetime.utcnow().isoformat()


def _get_uptime() -> int:
    """Get analyzer uptime in seconds."""
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
            return int(uptime_seconds)
    except:
        return 0


def run_web_server(state_manager, profile_loader, paperless_client, host='0.0.0.0', port=8051,
                  project_manager=None, llm_client=None, smart_uploader=None, document_analyzer=None):
    """
    Run the Flask web server in a separate thread using Waitress (production WSGI server).

    Args:
        state_manager: StateManager instance
        profile_loader: ProfileLoader instance
        paperless_client: PaperlessClient instance
        host: Host to bind to
        port: Port to bind to
        project_manager: ProjectManager instance (v1.5.0)
        llm_client: LLM client instance (v1.5.0)
        smart_uploader: SmartUploader instance (v1.5.0)
        document_analyzer: DocumentAnalyzer instance (v1.5.0 - for re-analysis)
    """
    create_app(state_manager, profile_loader, paperless_client,
              project_manager=project_manager, llm_client=llm_client, smart_uploader=smart_uploader,
              document_analyzer=document_analyzer)

    logger.info(f"Starting production web UI on {host}:{port}")

    # Use Waitress for production-grade serving
    from waitress import serve
    serve(
        app,
        host=host,
        port=port,
        threads=4,  # Handle 4 concurrent requests
        channel_timeout=300,  # 5 minute timeout
        cleanup_interval=10,  # Clean up connections every 10s
        _quiet=False  # Show startup message
    )


def start_web_server_thread(state_manager, profile_loader, paperless_client, host='0.0.0.0', port=8051,
                           project_manager=None, llm_client=None, smart_uploader=None, document_analyzer=None):
    """
    Start web server in background thread.

    Args:
        state_manager: StateManager instance
        profile_loader: ProfileLoader instance
        paperless_client: PaperlessClient instance
        host: Host to bind to
        port: Port to bind to
        project_manager: ProjectManager instance (v1.5.0)
        llm_client: LLM client instance (v1.5.0)
        smart_uploader: SmartUploader instance (v1.5.0)
        document_analyzer: DocumentAnalyzer instance (v1.5.0 - for re-analysis)
    """
    # Set up log buffer handler
    root_logger = logging.getLogger()
    buffer_handler = LogBufferHandler()
    buffer_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(buffer_handler)
    logger.info("Log buffer handler registered - live logs enabled")

    thread = Thread(
        target=run_web_server,
        args=(state_manager, profile_loader, paperless_client, host, port,
              project_manager, llm_client, smart_uploader, document_analyzer),
        daemon=True
    )
    thread.start()
    logger.info(f"Web UI thread started on {host}:{port}")
    return thread
