"""Chat core endpoints: /api/chat and /api/chat/compare.

Extracted from routes/chat.py during the v3.9.3 maintainability refactor.
"""
import logging
import os
from datetime import datetime
from threading import Thread
from flask import Blueprint, request, jsonify, session, make_response, render_template
from flask_login import login_required, current_user

from analyzer.app import ui_state
from analyzer.db import (
    get_session, create_session, can_access_session, get_sessions,
    get_all_sessions_by_user, get_messages, append_message, update_session_title,
    delete_session, get_session_shares, share_session, unshare_session,
    get_active_leaf, set_active_leaf, get_message_by_id, update_message_content,
    delete_messages_from,
)
from analyzer.db import get_user_by_username
from analyzer.services.ai_config_service import load_ai_config, get_project_ai_config

# Business-logic helpers extracted 2026-04-23 (v3.9.1 refactor).
# These used to live in this file as _vision_extract_doc, _load_session_web_context,
# _save_session_web_context, _ddg_search, _resolve_court_docket_url,
# _fetch_url_text, _compute_branch_data. See refactor-log Entry 005.
from analyzer.services.web_research_service import (
    SEARCH_INTENT_PHRASES as _SEARCH_INTENT_PHRASES,
    load_session_web_context as _load_session_web_context,
    save_session_web_context as _save_session_web_context,
    ddg_search as _ddg_search,
    resolve_court_docket_url as _resolve_court_docket_url,
    fetch_url_text as _fetch_url_text,
)
from analyzer.services.vision_service import vision_extract_doc as _vision_extract_doc
from analyzer.services.chat_branch_service import compute_branch_data as _compute_branch_data

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Chat routes (helpers moved to analyzer/services/ — see imports above)
# ---------------------------------------------------------------------------

from analyzer.routes.chat import bp

@bp.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    """Chat with AI about documents using RAG (semantic search)."""
    from flask import current_app
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        history = data.get('history', [])
        document_type = data.get('document_type', None)  # Optional filter by document type
        session_id = data.get('session_id', None)
        # branch_parent_id: set by the branch edit flow so we attach messages to the right node.
        # Use a sentinel so we can distinguish "explicitly null root branch" from "not provided".
        _UNSET = object()
        branch_parent_id = data.get('branch_parent_id', _UNSET)

        if not user_message:
            return jsonify({'error': 'Message required'}), 400

        # ── Web context helpers (populated after session_id resolved below) ────
        import re as _re_chat
        _fetched_urls = []     # [(key, text)] injected into prompt
        _failed_urls  = []     # [(url, reason)]
        # ─────────────────────────────────────────────────────────────────────

        # Resolve or create a chat session
        if session_id:
            sess = get_session(session_id)
            if not sess or not can_access_session(session_id, current_user.id):
                session_id = None  # Fall through to create new
        if not session_id:
            session_id = create_session(
                current_user.id,
                title='New Chat',
                document_type=document_type,
            )

        # Determine parent_id for the user message being inserted:
        # - branch edit flow:  use branch_parent_id from request (may be None for root messages)
        # - normal flow:       use the session's current active_leaf_id
        if branch_parent_id is _UNSET:
            _user_msg_parent = get_active_leaf(session_id)
        else:
            _user_msg_parent = branch_parent_id  # explicit value (possibly None)

        # ── Load persisted web context + process current message ─────────────
        _web_ctx = _load_session_web_context(session_id)
        _url_pat = _re_chat.compile(r'https?://[^\s<>"\']+', _re_chat.IGNORECASE)
        for _u in _url_pat.findall(user_message)[:3]:
            if _u not in _web_ctx:
                _text, _err = _resolve_court_docket_url(_u)
                if not _text and not _err:
                    _text, _err = _fetch_url_text(_u)
                if _text:
                    _web_ctx[_u] = _text
                    logger.info(f"Chat URL fetch OK: {_u} ({len(_text)} chars)")
                else:
                    _failed_urls.append((_u, _err))
                    logger.info(f"Chat URL fetch failed: {_u} — {_err}")

        _msg_lower = user_message.lower()
        if any(p in _msg_lower for p in _SEARCH_INTENT_PHRASES):
            _search_q = user_message[:200]
            _sr = _ddg_search(_search_q, max_results=6)
            if _sr:
                _sk = f"search:{_search_q[:60]}"
                _st = f"WEB SEARCH RESULTS for: {_search_q}\n\n"
                for _i, _r in enumerate(_sr, 1):
                    _st += f"{_i}. {_r['title']}\n   {_r['excerpt']}\n   {_r['url']}\n\n"
                _web_ctx[_sk] = _st
                logger.info(f"Chat web search: {_search_q[:60]} ({len(_sr)} results)")

        _fetched_urls = list(_web_ctx.items())   # all persisted context for this session
        # ─────────────────────────────────────────────────────────────────────

        # Get stats
        with ui_state['lock']:
            stats = ui_state['stats']

        # Use semantic search with vector store — scoped to current project
        from analyzer.vector_store import VectorStore
        _chat_project = session.get('current_project', 'default')
        vector_store = VectorStore(project_slug=_chat_project)

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

        # ── Vision AI content enrichment ────────────────────────────────────────
        # For each retrieved doc whose stored content is too short (empty OCR at
        # embed time), first try a fresh Paperless OCR fetch, then fall back to
        # Vision AI.  Cap Vision AI at 2 docs per query to limit latency.
        vision_ai_used = []
        if recent_analyses and hasattr(current_app, 'paperless_client'):
            _ai_cfg_vision = load_ai_config()
            vision_cap = 0
            enriched = []
            for _a in recent_analyses:
                _content = _a.get('content', _a.get('ai_analysis', ''))
                _doc_id  = _a.get('document_id')
                _doc_title = _a.get('document_title', '')
                if len(_content.strip()) < 500 and _doc_id:
                    try:
                        _fresh = current_app.paperless_client.get_document(int(_doc_id))
                        _fresh_text = _fresh.get('content', '').strip()
                        if len(_fresh_text) >= 200:
                            _a = dict(_a)
                            _a['content'] = f"Document Content (Paperless OCR — live fetch):\n{_fresh_text}"
                            logger.info(f"Chat enrichment: refreshed OCR for doc {_doc_id} ({len(_fresh_text)} chars)")
                        elif vision_cap < 2:
                            # OCR still short — try Vision AI
                            logger.info(f"Chat enrichment: running Vision AI on doc {_doc_id}")
                            _vtext = _vision_extract_doc(int(_doc_id), _doc_title, current_app.paperless_client, _ai_cfg_vision)
                            if _vtext and len(_vtext) > 200:
                                _a = dict(_a)
                                _a['content'] = f"Document Content (Vision AI — extracted during this chat):\n{_vtext}"
                                vision_ai_used.append(_doc_title or str(_doc_id))
                                vision_cap += 1
                                # Re-embed in background so future queries benefit
                                def _bg_reembed(_did=int(_doc_id), _dtitle=_doc_title, _dtext=_vtext,
                                                _dtype=_a.get('document_type', 'unknown'),
                                                _drisk=_a.get('risk_score', 0)):
                                    try:
                                        if current_app.document_analyzer and current_app.document_analyzer.vector_store:
                                            current_app.document_analyzer.vector_store.embed_document(
                                                _did, _dtitle, _dtext,
                                                {'risk_score': _drisk, 'anomalies': [],
                                                 'timestamp': datetime.utcnow().isoformat(),
                                                 'paperless_modified': '', 'document_type': _dtype}
                                            )
                                            logger.info(f"Chat Vision AI: re-embedded doc {_did}")
                                    except Exception as _re:
                                        logger.warning(f"Chat Vision AI: re-embed failed for doc {_did}: {_re}")
                                Thread(target=_bg_reembed, daemon=True).start()
                    except Exception as _ee:
                        logger.warning(f"Chat enrichment failed for doc {_doc_id}: {_ee}")
                enriched.append(_a)
            recent_analyses = enriched
        # ────────────────────────────────────────────────────────────────────────

        # If we don't have analyses, fetch from Paperless
        # Only fall back if Chroma returned nothing — do NOT use < 5 threshold
        # because small projects legitimately have few docs and the fallback
        # is not project-scoped (it returns all analyzed docs from Paperless).
        if not recent_analyses:
            try:
                # Get documents with analyzed tags
                paperless_client = current_app.paperless_client
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
                    except Exception:
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

                    # Generate AI-powered comparative summary if LLM is available
                    brief_summary = ""
                    full_summary = ""
                    try:
                        if current_app.document_analyzer and current_app.document_analyzer.llm_client:
                            # Query vector store for similar documents
                            similar_docs = []
                            if current_app.document_analyzer.vector_store:
                                try:
                                    # Search for similar documents using title as query
                                    results = current_app.document_analyzer.vector_store.query(
                                        query_text=doc['title'],
                                        n_results=6  # Get 6 to exclude self
                                    )
                                    if results and 'documents' in results:
                                        # Filter out the current document and format results
                                        for i, result_doc in enumerate(results.get('documents', [[]])[0]):
                                            metadata = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
                                            result_id = metadata.get('document_id', '')
                                            if result_id and str(result_id) != str(doc_id):
                                                similar_docs.append({
                                                    'id': result_id,
                                                    'title': metadata.get('title', 'Unknown'),
                                                    'created': metadata.get('created', 'unknown'),
                                                    'similarity': results.get('distances', [[]])[0][i] if i < len(results.get('distances', [[]])[0]) else 0
                                                })
                                except Exception as ve:
                                    logger.debug(f"Vector search failed for doc {doc_id}: {ve}")

                            # Generate comparative summary
                            doc_info = {
                                'id': doc_id,
                                'title': doc['title'],
                                'document_type': 'financial document',
                                'created': doc.get('created', ''),
                            }
                            summary = current_app.document_analyzer.llm_client.generate_comparative_summary(
                                doc_info,
                                content_preview=notes[:500] if notes else "",
                                similar_documents=similar_docs
                            )
                            brief_summary = summary.get('brief', '')
                            full_summary = summary.get('full', '')
                    except Exception as sum_err:
                        logger.debug(f"Failed to generate summary for doc {doc_id}: {sum_err}")
                        brief_summary = f"Financial document: {doc['title']}"
                        full_summary = brief_summary

                    recent_analyses.append({
                        'document_id': doc_id,
                        'document_title': doc['title'],
                        'anomalies_found': anomalies[:5],
                        'risk_score': risk_score,
                        'timestamp': doc['modified'],
                        'ai_analysis': ai_analysis,
                        'created': doc.get('created', ''),
                        'correspondent': doc.get('correspondent', None),
                        'brief_summary': brief_summary,
                        'full_summary': full_summary
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
            context += f"\n- [Document #{doc_id}]: {doc_title}"
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

                system_prompt += f"\n\n--- [Document #{doc_id}] ---"
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

VISION AI CAPABILITY:
- Documents whose stored OCR was empty or too short have been enriched LIVE during this chat.
- Content marked "Vision AI — extracted during this chat" was read directly from the original PDF pages using image recognition — treat it as authoritative.
- Content marked "Paperless OCR — live fetch" was retrieved fresh from Paperless at query time.
- You CAN and SHOULD analyze the content in these enriched documents fully.

CRITICAL - NEVER HALLUCINATE DATA:
- NEVER invent dollar amounts, totals, dates, names, or any data not explicitly present in the content shown above.
- Only report numbers and facts that are EXPLICITLY stated in the document content provided above.
- If content is still empty or very short after enrichment, say: "This document's content could not be extracted even with Vision AI. I cannot analyze specific figures without the source file being accessible."
- Do NOT claim you lack access to PDFs — Vision AI has already been applied where needed.

DOCUMENT REFERENCES: When mentioning any document, ALWAYS use the exact format [Document #NNN] where NNN is the document ID. This enables clickable links. Never write "Doc NNN" without the brackets and #.

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
- Provide the most complete analysis possible given the data you can access

HELP & DOCUMENTATION:
When a user asks how to do something in the interface, or asks about a feature, include a helpful link to the user manual. Use markdown link format.
Available documentation pages (prepend the app URL prefix):
- Overview & feature list: /docs/overview
- Quick start guide: /docs/getting-started
- Projects & workspaces: /docs/projects
- Smart Upload (file/URL/cloud): /docs/upload
- AI Chat usage: /docs/chat
- Search & Analysis: /docs/search
- Anomaly detection tags: /docs/anomaly-detection
- Debug & Tools (reprocess, logs): /docs/tools
- Configuration (AI keys, profiles, SMTP): /docs/configuration
- User management: /docs/users
- LLM usage & cost tracking: /docs/llm-usage
- API reference: /docs/api
Example: "You can learn more about projects in the [Projects documentation](/docs/projects)."
Only include a docs link when it is genuinely relevant to the user's question.

WEB ACCESS CAPABILITY — CRITICAL:
You CAN access URLs and search the web. This system automatically fetches URLs and runs web searches on your behalf.
- Any URLs the user pastes are fetched and their content is injected below under "WEB CONTENT".
- Any search requests (e.g. "search for...", "find online...") trigger a live DuckDuckGo search — results appear below.
- NEVER tell the user you cannot browse the internet, cannot access URLs, or cannot search online.
- NEVER apologize for not having web access — you DO have it through this system.
- If no web content appears below, either no URL/search was provided, or the fetch failed (failure reason will be shown)."""

        if vision_ai_used:
            system_prompt += f"\n\n[Note: Vision AI was used during this query to extract content from {len(vision_ai_used)} document(s) with poor OCR: {', '.join(vision_ai_used[:3])}. Their embeddings have been updated for future queries.]"

        # ── Inject web context (URLs + searches) into the prompt ─────────────
        if _fetched_urls:
            system_prompt += "\n\n" + "="*60
            system_prompt += "\nWEB CONTENT (fetched live — URLs and search results from this session):"
            system_prompt += "\nUse this content freely alongside the documents above."
            for _ck, _cv in _fetched_urls:
                system_prompt += f"\n\n--- {_ck} ---\n{_cv}"
            system_prompt += "\n" + "="*60

        if _failed_urls:
            system_prompt += "\n\nWEB FETCH FAILURES:"
            for _u, _reason in _failed_urls:
                system_prompt += f"\n- Could NOT fetch: {_u} (reason: {_reason})"
            system_prompt += (
                "\nTell the user clearly: \"I was unable to access [URL] — [brief reason]\". "
                "Do NOT attempt to answer questions about that URL's content. "
                "Do NOT hallucinate or guess what that page might contain."
            )

        # Save updated web context back to session for future turns
        if _web_ctx:
            _save_session_web_context(session_id, _web_ctx)
        # ─────────────────────────────────────────────────────────────────────

        # Load AI configuration — v2 format: per-project primary/fallback, global keys as fallback
        project_slug = session.get('current_project', 'default')
        chat_cfg = get_project_ai_config(project_slug, 'chat')
        _full_cfg = load_ai_config()

        def _global_key(provider_name):
            return _full_cfg.get('global', {}).get(provider_name, {}).get('api_key', '').strip()

        providers = []
        prov = chat_cfg.get('provider', 'openai')
        pkey = (chat_cfg.get('api_key') or '').strip() or _global_key(prov)
        if pkey:
            providers.append({'name': prov, 'enabled': True, 'api_key': pkey,
                              'models': [chat_cfg.get('model', 'gpt-4o')]})
        fb_prov = chat_cfg.get('fallback_provider')
        fb_model = chat_cfg.get('fallback_model')
        if fb_prov and fb_model and fb_prov != prov:
            fb_key = _global_key(fb_prov)
            if fb_key:
                providers.append({'name': fb_prov, 'enabled': True, 'api_key': fb_key,
                                  'models': [fb_model]})

        # Pick the first usable provider+model combo as the direct-provider fallback
        direct_provider = direct_api_key = direct_model = None
        for pc in providers:
            if pc.get('enabled') and (pc.get('api_key') or '').strip() and pc.get('models'):
                direct_provider = pc.get('name')
                direct_api_key = pc.get('api_key').strip()
                direct_model = (pc.get('models') or [None])[0]
                break

        from analyzer.llm.proxy_call import call_llm, LLMUnavailableError
        try:
            result = call_llm(
                messages=[{"role": "system", "content": system_prompt}] + messages,
                task='chat',
                max_tokens=4096,  # comprehensive answers for legal/court documents
                project_slug=session.get('current_project', 'default') if 'current_project' in session else 'default',
                operation='chat_completion',
                direct_provider=direct_provider,
                direct_api_key=direct_api_key,
                direct_model=direct_model,
            )
            ai_response = result['content']
        except LLMUnavailableError as e:
            logger.error(f"chat: LLM unavailable: {e} attempted={e.attempted}")
            raise Exception(f"No available models responded. Tried: {', '.join(e.attempted)}.")

        logger.info(f"Chat query: {user_message[:100]}")

        # Persist messages to the session (with branch parent chain)
        user_msg_id = append_message(session_id, 'user', user_message, parent_id=_user_msg_parent)
        append_message(session_id, 'assistant', ai_response, parent_id=user_msg_id)

        # Auto-title: set from first user message if still 'New Chat'
        current_sess = get_session(session_id)
        if current_sess and current_sess['title'] == 'New Chat':
            auto_title = user_message[:60].strip()
            if auto_title:
                update_session_title(session_id, auto_title)

        return jsonify({
            'response': ai_response,
            'success': True,
            'session_id': session_id,
            'user_message_id': user_msg_id,
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/chat/compare', methods=['POST'])
@login_required
def api_chat_compare():
    """Call both configured LLM providers in parallel and return both responses."""
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        data = request.json or {}
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Message required'}), 400

        # Resolve session
        session_id = data.get('session_id') or None
        if session_id:
            sess = get_session(session_id)
            if not sess or not can_access_session(session_id, current_user.id):
                session_id = None
        if not session_id:
            session_id = create_session(current_user.id, title='New Chat')

        # Build provider list (same logic as api_chat)
        project_slug = session.get('current_project', 'default')
        chat_cfg = get_project_ai_config(project_slug, 'chat')
        _full_cfg = load_ai_config()

        def _global_key(pname):
            return _full_cfg.get('global', {}).get(pname, {}).get('api_key', '').strip()

        providers = []
        prov = chat_cfg.get('provider', 'openai')
        pkey = (chat_cfg.get('api_key') or '').strip() or _global_key(prov)
        if pkey:
            providers.append({'name': prov, 'api_key': pkey,
                              'model': chat_cfg.get('model', 'gpt-4o')})
        fb_prov = chat_cfg.get('fallback_provider')
        fb_model = chat_cfg.get('fallback_model')
        if fb_prov and fb_model and fb_prov != prov:
            fb_key = _global_key(fb_prov)
            if fb_key:
                providers.append({'name': fb_prov, 'api_key': fb_key, 'model': fb_model})

        if len(providers) < 2:
            return jsonify({'error': 'Two configured AI providers are required for compare mode. Please add both a primary and a fallback provider in AI Configuration.'}), 400

        # Build messages list (last 10 from history)
        history = data.get('history', [])
        messages = [{'role': m['role'], 'content': m['content']}
                    for m in history[-10:] if m.get('role') in ('user', 'assistant')]
        messages.append({'role': 'user', 'content': user_message})

        # Build system prompt (reuse simple version without full RAG for speed)
        system_prompt = "You are an AI assistant helping analyze documents. Answer helpfully and accurately. When mentioning any document, ALWAYS use the format [Document #NNN] to enable clickable links."

        from analyzer.llm.proxy_call import call_llm as _proxy_call, LLMUnavailableError

        def _call_provider(pconf):
            """Force routing to a specific provider via LMRH fallback-chain pinning.
            Direct-provider fallback uses the configured key if proxy fails."""
            name = pconf['name']
            key = pconf['api_key']
            model = pconf['model']
            try:
                result = _proxy_call(
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    task='chat',
                    max_tokens=4096,
                    fallback_chain=name,  # pin proxy to this provider only
                    model_pref=model,
                    operation='chat_compare',
                    direct_provider=name,
                    direct_api_key=key,
                    direct_model=model,
                )
                return name, result['content'], None
            except LLMUnavailableError as e:
                return name, None, str(e)
            except Exception as e:
                return name, None, str(e)

        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futs = {executor.submit(_call_provider, p): p['name'] for p in providers[:2]}
            for fut in as_completed(futs):
                name, text, err = fut.result()
                results[name] = {'text': text, 'error': err}

        primary = providers[0]
        secondary = providers[1]
        prim_name = primary['name']
        sec_name = secondary['name']
        prim_result = results.get(prim_name, {})
        sec_result = results.get(sec_name, {})

        primary_response = prim_result.get('text') or f"Error: {prim_result.get('error', 'No response')}"
        secondary_response = sec_result.get('text') or f"Error: {sec_result.get('error', 'No response')}"
        secondary_error = bool(sec_result.get('error'))

        # Save primary response to DB for conversation continuity
        _cmp_parent = get_active_leaf(session_id)
        user_msg_id = append_message(session_id, 'user', user_message, parent_id=_cmp_parent)
        append_message(session_id, 'assistant', primary_response, parent_id=user_msg_id)

        # Auto-title
        cur_sess = get_session(session_id)
        if cur_sess and cur_sess['title'] == 'New Chat':
            update_session_title(session_id, user_message[:60].strip())

        return jsonify({
            'primary_provider': prim_name.capitalize(),
            'primary_response': primary_response,
            'secondary_provider': sec_name.capitalize(),
            'secondary_response': secondary_response,
            'secondary_error': secondary_error,
            'session_id': session_id,
            'user_message_id': user_msg_id,
        })
    except Exception as e:
        logger.error(f"Compare chat error: {e}")
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Chat Session Routes
# ---------------------------------------------------------------------------

