// ── Chat Sessions ────────────────────────────────────────────────────────

        // ── Chat session state ────────────────────────────────────────
        let currentSessionId = null;

        async function loadSessions() {
            try {
                const res = await apiFetch(apiUrl('/api/chat/sessions'));
                const data = await res.json();
                renderSessionList(data);
            } catch (e) {
                console.error('Failed to load sessions:', e);
            }
        }

        function renderSessionList(data) {
            const container = document.getElementById('session-list');
            const isAdmin = data.is_admin;
            let html = '';

            function formatDate(iso) {
                if (!iso) return '';
                const d = new Date(iso);
                const now = new Date();
                const diffMs = now - d;
                const diffH = Math.floor(diffMs / 3600000);
                if (diffH < 1) return 'Just now';
                if (diffH < 24) return `${diffH}h ago`;
                const diffD = Math.floor(diffH / 24);
                if (diffD < 7) return `${diffD}d ago`;
                return d.toLocaleDateString();
            }

            function sessionItemHtml(s) {
                const active = s.id === currentSessionId ? ' active' : '';
                const sharedBadge = s.is_shared ? '<span class="session-badge-shared">shared</span>' : '';
                return `<div class="session-item${active}" data-session-id="${s.id}" onclick="loadSession('${s.id}')">
                    <span class="session-title">topic: ${escapeHtml(s.title)}${sharedBadge}</span>
                    <span class="session-meta">${formatDate(s.updated_at)}</span>
                    <div class="session-actions" onclick="event.stopPropagation()">
                        <button class="session-action-btn" title="Rename" onclick="renameSessionPrompt('${s.id}', '${escapeHtml(s.title).replace(/'/g, "\\'")}')">✏️</button>
                        ${s.is_own !== false ? `<button class="session-action-btn" title="Delete" onclick="deleteSession('${s.id}')">🗑️</button>` : ''}
                    </div>
                </div>`;
            }

            if (isAdmin && data.sessions_by_user) {
                for (const [username, sessions] of Object.entries(data.sessions_by_user)) {
                    html += `<div class="session-group-header">${escapeHtml(username)}</div>`;
                    sessions.forEach(s => { html += sessionItemHtml(s); });
                }
            } else if (data.sessions) {
                const own = data.sessions.filter(s => s.is_own && !s.is_shared);
                const shared = data.sessions.filter(s => s.is_shared);
                if (own.length) {
                    html += '<div class="session-group-header">My Chats</div>';
                    own.forEach(s => { html += sessionItemHtml(s); });
                }
                if (shared.length) {
                    html += '<div class="session-group-header">Shared with me</div>';
                    shared.forEach(s => { html += sessionItemHtml(s); });
                }
            }

            if (!html) {
                html = '<div style="padding: 20px 14px; font-size: 12px; color: #556677;">No chats yet. Start a new chat!</div>';
            }
            container.innerHTML = html;
        }

        function escapeHtml(str) {
            return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
        }

        async function loadSession(sessionId) {
            try {
                const res = await apiFetch(apiUrl(`/api/chat/sessions/${sessionId}`));
                if (!res.ok) { alert('Failed to load session'); return; }
                const data = await res.json();

                currentSessionId = sessionId;
                chatHistory = [];
                _sessionForkPoints = data.fork_points || [];

                // Update header title
                document.getElementById('active-session-title').textContent = data.session.title;

                // Populate chat messages (active branch only)
                const messagesDiv = document.getElementById('chat-messages');
                messagesDiv.innerHTML = '';
                data.messages.forEach(m => {
                    // Find if this user message sits at a fork point
                    const fp = (m.role === 'user')
                        ? _sessionForkPoints.find(f => f.variants.some(v => v.id === m.id))
                        : null;
                    addChatMessage(m.role, m.content, m.role === 'user' ? m.id : null, fp);
                    chatHistory.push({role: m.role, content: m.content});
                });

                if (!data.messages.length) {
                    // Show welcome message if empty session
                    addWelcomeMessage();
                }

                // Switch to AI Chat tab
                switchTab('ai-chat');
                // Update sidebar highlights
                loadSessions();
            } catch (e) {
                console.error('loadSession error:', e);
            }
        }

        function newChat() {
            currentSessionId = null;
            chatHistory = [];
            document.getElementById('active-session-title').textContent = 'New Chat';
            const messagesDiv = document.getElementById('chat-messages');
            messagesDiv.innerHTML = '';
            addWelcomeMessage();
            switchTab('ai-chat');
            // Deselect active session in sidebar
            document.querySelectorAll('.session-item.active').forEach(el => el.classList.remove('active'));
        }

        function addWelcomeMessage() {
            const messagesDiv = document.getElementById('chat-messages');
            messagesDiv.innerHTML = `
                <div class="chat-message assistant">
                    <div class="message-bubble">
                        <div>👋 Hi! I'm your AI document assistant. I can help you:</div>
                        <ul style="margin: 10px 0 0 20px; line-height: 1.8;">
                            <li>Analyze patterns across all your documents</li>
                            <li>Find specific transactions or anomalies</li>
                            <li>Generate reports and ledgers</li>
                            <li>Answer questions about your financial documents</li>
                        </ul>
                        <div style="margin-top: 10px;">Try asking me something!</div>
                    </div>
                </div>
                <div class="chat-suggestions">
                    <div class="suggestion-chip" onclick="sendSuggestion('Show me all high-risk documents')">Show me all high-risk documents</div>
                    <div class="suggestion-chip" onclick="sendSuggestion('What anomalies were most common?')">What anomalies were most common?</div>
                    <div class="suggestion-chip" onclick="sendSuggestion('Generate a summary of all analyzed documents')">Generate a summary</div>
                </div>`;
        }

        async function deleteSession(sessionId) {
            if (!confirm('Delete this chat session? This cannot be undone.')) return;
            try {
                const res = await apiFetch(apiUrl(`/api/chat/sessions/${sessionId}`), {method: 'DELETE'});
                if (!res.ok) { alert('Failed to delete session'); return; }
                if (currentSessionId === sessionId) newChat();
                loadSessions();
            } catch (e) { console.error(e); }
        }

        async function renameSessionPrompt(sessionId, currentTitle) {
            const newTitle = prompt('Rename chat:', currentTitle);
            if (!newTitle || newTitle.trim() === currentTitle) return;
            try {
                const res = await apiFetch(apiUrl(`/api/chat/sessions/${sessionId}`), {
                    method: 'PATCH',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({title: newTitle.trim()}),
                });
                if (!res.ok) { alert('Failed to rename session'); return; }
                if (currentSessionId === sessionId) {
                    document.getElementById('active-session-title').textContent = newTitle.trim();
                }
                loadSessions();
            } catch (e) { console.error(e); }
        }

        // ── Share modal ────────────────────────────────────────────────
        let _shareSessionId = null;

        function shareCurrentChat() {
            if (!currentSessionId) { alert('Start a chat first, then share it.'); return; }
            openShareModal(currentSessionId);
        }

        async function openShareModal(sessionId) {
            _shareSessionId = sessionId;
            document.getElementById('share-modal').classList.add('active');
            document.getElementById('share-username-input').value = '';
            document.getElementById('share-error').style.display = 'none';
            await refreshShareList();
        }

        function closeShareModal() {
            document.getElementById('share-modal').classList.remove('active');
            _shareSessionId = null;
        }

        async function refreshShareList() {
            if (!_shareSessionId) return;
            const res = await apiFetch(apiUrl(`/api/chat/sessions/${_shareSessionId}`));
            const data = await res.json();
            const shares = data.shared_with || [];
            const list = document.getElementById('share-list');
            if (!shares.length) {
                list.innerHTML = '<div style="color: #999; font-size: 13px;">No shares yet</div>';
            } else {
                list.innerHTML = shares.map(s => `
                    <div class="share-list-item">
                        <span>👤 ${escapeHtml(s.username)}</span>
                        <button class="btn btn-danger" style="font-size: 12px; padding: 3px 10px;"
                            onclick="removeShare(${s.id})">Remove</button>
                    </div>`).join('');
            }
        }

        async function addShare() {
            const username = document.getElementById('share-username-input').value.trim();
            const errEl = document.getElementById('share-error');
            if (!username) { errEl.textContent = 'Enter a username'; errEl.style.display = 'block'; return; }
            errEl.style.display = 'none';
            try {
                const res = await apiFetch(apiUrl(`/api/chat/sessions/${_shareSessionId}/share`), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username}),
                });
                const data = await res.json();
                if (!res.ok) { errEl.textContent = data.error || 'Failed to share'; errEl.style.display = 'block'; return; }
                document.getElementById('share-username-input').value = '';
                await refreshShareList();
            } catch (e) { errEl.textContent = e.message; errEl.style.display = 'block'; }
        }

        async function removeShare(uid) {
            await apiFetch(apiUrl(`/api/chat/sessions/${_shareSessionId}/share/${uid}`), {method: 'DELETE'});
            await refreshShareList();
        }

        document.addEventListener('click', e => {
            const m = document.getElementById('share-modal');
            if (m && e.target === m) closeShareModal();
        });

        // ── Export ─────────────────────────────────────────────────────
        function exportCurrentChat() {
            if (!currentSessionId) { alert('Start a chat first, then export it.'); return; }
            window.location.href = apiUrl(`/api/chat/sessions/${currentSessionId}/export`);
        }

// ── Chat Functionality ──────────────────────────────────────────────────
        // Chat functionality
        let chatHistory = [];
        let _chatAbortController = null;
        let _compareMode = false;
        let _sessionForkPoints = [];   // fork point data from last loadSession
        let _branchParentId = null;    // set before sendChatMessage for edit-branch flow

        /**
         * Replace [Document #NNN] / Document #NNN references in rendered HTML
         * with clickable "View in Paperless" links for the current project.
         */
        function _linkifyDocRefs(html, slug) {
            // Match [Document #NNN], Document #NNN, [Doc #NNN], doc #NNN, Doc NNN (case-insensitive)
            return html.replace(/(?:\[Document #(\d+)\]|\[Doc #(\d+)\]|Document #(\d+)|Doc\.?\s+#?(\d+)(?=\b)|\bdoc(?:ument)? #(\d+))/gi, (match, id1, id2, id3, id4, id5) => {
                const did = parseInt(id1 || id2 || id3 || id4 || id5, 10);
                const url = _paperlessDocUrl(slug, did);
                if (url) {
                    return `<a href="${url}" target="_blank" style="color:#1d4ed8;text-decoration:underline;" title="Open Document #${did} in Paperless">${match} ↗</a>`;
                }
                return match;
            });
        }

        function _makeEditBtn(messageId) {
            const btn = document.createElement('button');
            btn.className = 'msg-edit-btn';
            btn.textContent = '✏️ Edit';
            btn.title = 'Edit this message and resend';
            btn.onclick = () => startEditMessage(messageId);
            return btn;
        }

        function _makeVariantNav(forkPoint) {
            const total = forkPoint.variants.length;
            const active = forkPoint.active_variant_index;
            const nav = document.createElement('div');
            nav.className = 'variant-nav';

            const prevBtn = document.createElement('button');
            prevBtn.className = 'variant-btn';
            prevBtn.textContent = '←';
            prevBtn.title = 'Previous version of this message';
            prevBtn.disabled = active === 0;
            if (active > 0) prevBtn.onclick = () => switchVariant(forkPoint.variants[active - 1].leaf_id);

            const counter = document.createElement('span');
            counter.className = 'variant-counter';
            counter.textContent = `${active + 1}/${total}`;

            const nextBtn = document.createElement('button');
            nextBtn.className = 'variant-btn';
            nextBtn.textContent = '→';
            nextBtn.title = 'Next version of this message';
            nextBtn.disabled = active === total - 1;
            if (active < total - 1) nextBtn.onclick = () => switchVariant(forkPoint.variants[active + 1].leaf_id);

            nav.appendChild(prevBtn);
            nav.appendChild(counter);
            nav.appendChild(nextBtn);
            return nav;
        }

        async function switchVariant(leafId) {
            if (!currentSessionId) return;
            try {
                const res = await apiFetch(apiUrl(`/api/chat/sessions/${currentSessionId}/set-leaf`), {
                    method: 'PATCH',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({leaf_id: leafId}),
                });
                if (!res.ok) { console.error('switchVariant failed'); return; }
                // Re-render the session with the new active branch
                await loadSession(currentSessionId);
            } catch (e) {
                console.error('switchVariant error:', e);
            }
        }

        function addChatMessage(role, content, messageId = null, forkPoint = null) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${role}`;
            if (messageId) messageDiv.dataset.msgId = messageId;

            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';

            // Render markdown for assistant messages, plain text for user
            if (role === 'assistant' && typeof marked !== 'undefined') {
                marked.setOptions({
                    breaks: true, gfm: true, headerIds: false, mangle: false
                });
                const slug = document.getElementById('project-selector')?.value || 'default';
                const _renderedHtml = _linkifyDocRefs(marked.parse(content), slug);
                // Open external links in a new tab so they don't navigate away from the session
                bubble.innerHTML = _renderedHtml.replace(
                    /<a href="(https?:\/\/[^"]+)"/g,
                    '<a href="$1" target="_blank" rel="noopener noreferrer"'
                );
            } else {
                bubble.textContent = content;
            }

            // For user messages with a tracked ID: add variant nav (if branched) + edit button
            if (role === 'user' && messageId) {
                if (forkPoint && forkPoint.variants.length > 1) {
                    messageDiv.appendChild(_makeVariantNav(forkPoint));
                }
                messageDiv.appendChild(_makeEditBtn(messageId));
            }

            messageDiv.appendChild(bubble);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            chatHistory.push({role, content});
        }

        function startEditMessage(messageId) {
            const msgDiv = document.querySelector(`[data-msg-id="${messageId}"]`);
            if (!msgDiv) return;
            const editBtn = msgDiv.querySelector('.msg-edit-btn');
            const bubble = msgDiv.querySelector('.message-bubble');
            // Original text is cleanly in bubble.textContent (no emoji mixed in)
            const originalText = bubble.textContent.trim();

            if (editBtn) editBtn.style.display = 'none';  // hide while editing
            // Switch bubble to white background so the edit area is clearly visible
            const origBg = bubble.style.background;
            bubble.style.background = '#fff';
            bubble.style.border = '1px solid #d1d5db';
            bubble.style.color = '#111827';
            bubble.innerHTML = '';
            const ta = document.createElement('textarea');
            ta.className = 'msg-edit-area';
            ta.value = originalText;
            const resendBtn = document.createElement('button');
            resendBtn.style.cssText = 'padding:4px 12px; background:#3498db; color:#fff; border:none; border-radius:4px; cursor:pointer; font-size:12px; margin-right:6px;';
            resendBtn.textContent = '↩ Resend';
            resendBtn.onclick = () => confirmEditMessage(messageId, ta.value.trim());
            const cancelBtn = document.createElement('button');
            cancelBtn.style.cssText = 'padding:4px 10px; background:#f3f4f6; color:#374151; border:1px solid #d1d5db; border-radius:4px; cursor:pointer; font-size:12px;';
            cancelBtn.textContent = '✕ Cancel';
            cancelBtn.onclick = () => {
                bubble.textContent = originalText;
                // Restore original bubble appearance
                bubble.style.background = origBg || '';
                bubble.style.border = '';
                bubble.style.color = '';
                if (editBtn) editBtn.style.display = '';  // restore edit button
            };
            bubble.appendChild(ta);
            const btnRow = document.createElement('div');
            btnRow.style.marginTop = '4px';
            btnRow.appendChild(resendBtn);
            btnRow.appendChild(cancelBtn);
            bubble.appendChild(btnRow);
            ta.focus();
            ta.select();
        }

        async function confirmEditMessage(messageId, newText) {
            if (!newText || !currentSessionId) return;

            // Ask the server for the branch parent (shared prefix root) without inserting anything
            let branchParent = null;
            let historyPath = [];
            try {
                const bRes = await apiFetch(apiUrl(`/api/chat/sessions/${currentSessionId}/branch`), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({edit_from_id: messageId}),
                });
                if (bRes.ok) {
                    const bData = await bRes.json();
                    branchParent = bData.parent_id;
                    historyPath = bData.history_path || [];
                }
            } catch (_) { /* fall through: send without branch parent */ }

            // Rebuild chatHistory from the shared prefix (history_path)
            // so the LLM gets only the common ancestor context
            chatHistory = historyPath.map(m => ({role: m.role, content: m.content}));

            // Remove DOM messages from the edited position onwards so the user
            // sees a clean continuation (the branch will show after response + reload)
            const messagesDiv = document.getElementById('chat-messages');
            const allMsgDivs = [...messagesDiv.querySelectorAll('.chat-message')];
            const editedIdx = allMsgDivs.findIndex(d => d.dataset.msgId == messageId);
            if (editedIdx >= 0) {
                allMsgDivs.slice(editedIdx).forEach(d => d.remove());
            }

            // Pass branch parent so api_chat attaches the new message to the right tree node
            _branchParentId = branchParent;

            // Send edited text; after AI response comes back, loadSession re-renders
            // the full view including the variant navigation arrows
            document.getElementById('chat-input').value = newText;
            sendChatMessage();
        }

        function toggleCompareMode() {
            _compareMode = !_compareMode;
            const btn = document.getElementById('compare-toggle-btn');
            if (_compareMode) {
                btn.classList.add('active-compare');
                btn.textContent = '⚖️ Comparing — ON';
                btn.title = 'Compare mode is ON. Next Send will query both LLMs. Click to turn off.';
            } else {
                btn.classList.remove('active-compare');
                btn.textContent = '⚖️ Compare LLMs';
                btn.title = 'Send to both LLMs and see answers side-by-side';
            }
        }

        function stopChatRequest() {
            if (_chatAbortController) _chatAbortController.abort();
            document.getElementById('chat-stop-btn').style.display = 'none';
            hideTypingIndicator();
            const msgDiv = document.createElement('div');
            msgDiv.className = 'chat-message assistant';
            msgDiv.innerHTML = '<div class="message-bubble" style="color:#d97706;">⚠️ Request stopped.</div>';
            document.getElementById('chat-messages').appendChild(msgDiv);
            const input = document.getElementById('chat-input');
            const sendBtn = document.getElementById('chat-send-btn');
            input.disabled = false;
            sendBtn.disabled = false;
            input.focus();
            // Attach edit button to the stopped user message (never got an ID from server)
            const allUserDivs = document.querySelectorAll('.chat-message.user');
            const lastUser = allUserDivs[allUserDivs.length - 1];
            if (lastUser && !lastUser.dataset.msgId && !lastUser.querySelector('.msg-edit-btn')) {
                lastUser.dataset.msgId = 'stopped-' + Date.now();
                const bubble = lastUser.querySelector('.message-bubble');
                if (bubble) lastUser.insertBefore(_makeEditBtn(lastUser.dataset.msgId), bubble);
            }
        }

        function addCompareMessage(primaryProv, primaryResp, secProv, secResp, secError) {
            const messagesDiv = document.getElementById('chat-messages');
            const slug = document.getElementById('project-selector')?.value || 'default';
            const renderMd = (txt) => {
                if (typeof marked === 'undefined') return escapeHtml(txt);
                marked.setOptions({breaks: true, gfm: true, headerIds: false, mangle: false});
                const _html = _linkifyDocRefs(marked.parse(txt), slug);
                return _html.replace(/<a href="(https?:\/\/[^"]+)"/g, '<a href="$1" target="_blank" rel="noopener noreferrer"');
            };
            const secHtml = secError
                ? `<div style="color:#dc2626; font-size:13px;">⚠️ ${escapeHtml(secResp)}</div>`
                : renderMd(secResp);
            const uid = 'cmp-' + Date.now();
            const msgDiv = document.createElement('div');
            msgDiv.className = 'chat-message assistant compare-message';
            msgDiv.innerHTML = `
                <div class="message-bubble" style="padding:0;">
                    <div style="display:flex; gap:0; border-bottom:1px solid #e5e7eb; padding:6px 10px 0;">
                        <button class="ctab active" onclick="switchCompareTab('${uid}','primary',this)">🤖 ${escapeHtml(primaryProv)}</button>
                        <button class="ctab" onclick="switchCompareTab('${uid}','secondary',this)">🤖 ${escapeHtml(secProv)}</button>
                    </div>
                    <div id="${uid}-primary" class="ctab-pane" style="padding:10px 14px;">${renderMd(primaryResp)}</div>
                    <div id="${uid}-secondary" class="ctab-pane" style="display:none; padding:10px 14px;">${secHtml}</div>
                </div>`;
            messagesDiv.appendChild(msgDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            chatHistory.push({role: 'assistant', content: primaryResp});
        }

        function switchCompareTab(uid, which, btn) {
            const box = btn.closest('.compare-message');
            box.querySelectorAll('.ctab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            box.querySelectorAll('.ctab-pane').forEach(p => p.style.display = 'none');
            document.getElementById(`${uid}-${which}`).style.display = '';
        }

        function showTypingIndicator() {
            const messagesDiv = document.getElementById('chat-messages');
            const typingDiv = document.createElement('div');
            typingDiv.className = 'chat-message assistant';
            typingDiv.id = 'typing-indicator';
            typingDiv.innerHTML = '<div class="message-bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>';
            messagesDiv.appendChild(typingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function hideTypingIndicator() {
            const typingDiv = document.getElementById('typing-indicator');
            if (typingDiv) {
                typingDiv.remove();
            }
        }

        async function sendChatMessage() {
            const input = document.getElementById('chat-input');
            const sendBtn = document.getElementById('chat-send-btn');
            const stopBtn = document.getElementById('chat-stop-btn');
            const documentTypeFilter = document.getElementById('document-type-filter');
            const message = input.value.trim();

            if (!message) return;

            // If compare mode is active, delegate to compare handler
            if (_compareMode) {
                await sendCompareMessage(message);
                return;
            }

            // Add user message (no ID yet — will get it from response)
            addChatMessage('user', message);
            input.value = '';

            // Disable input while processing
            input.disabled = true;
            sendBtn.disabled = true;
            stopBtn.style.display = '';
            showTypingIndicator();

            _chatAbortController = new AbortController();

            try {
                const requestBody = {
                    message: message,
                    history: chatHistory.slice(-10), // Last 10 messages for context
                    session_id: currentSessionId,
                };

                // Branch edit flow: pass parent node so the server links to the right tree branch
                if (_branchParentId !== null) {
                    requestBody.branch_parent_id = _branchParentId;
                    _branchParentId = null;  // consume immediately
                }

                // Add document type filter if selected
                if (documentTypeFilter && documentTypeFilter.value !== 'all') {
                    requestBody.document_type = documentTypeFilter.value;
                }

                const response = await apiFetch(apiUrl('/api/chat'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(requestBody),
                    signal: _chatAbortController.signal,
                });

                // Detect session expiry: redirected to login page
                if (response.redirected && response.url.includes('/login')) {
                    hideTypingIndicator();
                    addChatMessage('assistant', '⚠️ Your session has expired. Please [reload the page](' + window.location.href + ') and log in again.');
                    return;
                }

                if (!response.ok && response.status === 401) {
                    hideTypingIndicator();
                    window.location.href = apiUrl('/login');
                    return;
                }

                const data = await response.json();

                hideTypingIndicator();

                if (data.response) {
                    // Attach message ID to the last user bubble so it can be edited
                    if (data.user_message_id) {
                        const allUserDivs = document.querySelectorAll('.chat-message.user');
                        const lastUser = allUserDivs[allUserDivs.length - 1];
                        if (lastUser && !lastUser.dataset.msgId) {
                            lastUser.dataset.msgId = data.user_message_id;
                            const bubble = lastUser.querySelector('.message-bubble');
                            if (bubble) {
                                lastUser.insertBefore(_makeEditBtn(data.user_message_id), bubble);
                            }
                        }
                    }
                    addChatMessage('assistant', data.response);
                    // Track session and update sidebar
                    if (data.session_id) {
                        const wasNew = !currentSessionId;
                        currentSessionId = data.session_id;
                        if (wasNew) {
                            // Auto-title: update header from first message
                            const autoTitle = message.slice(0, 60);
                            document.getElementById('active-session-title').textContent = autoTitle;
                        }
                        loadSessions();
                    }
                    // If this was a branch edit, reload so variant nav arrows appear
                    if ('branch_parent_id' in requestBody && currentSessionId) {
                        setTimeout(() => loadSession(currentSessionId), 300);
                    }
                } else {
                    addChatMessage('assistant', 'Sorry, I encountered an error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                if (error.name === 'AbortError') return; // Stopped by user — stopChatRequest() handles UI
                hideTypingIndicator();
                addChatMessage('assistant', 'Sorry, I encountered an error: ' + error.message);
            } finally {
                stopBtn.style.display = 'none';
                input.disabled = false;
                sendBtn.disabled = false;
                input.focus();
            }
        }

        async function sendCompareMessage(message) {
            const input = document.getElementById('chat-input');
            const sendBtn = document.getElementById('chat-send-btn');
            const stopBtn = document.getElementById('chat-stop-btn');
            const documentTypeFilter = document.getElementById('document-type-filter');

            addChatMessage('user', message);
            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;
            stopBtn.style.display = '';
            showTypingIndicator();

            _chatAbortController = new AbortController();

            try {
                const requestBody = {
                    message,
                    history: chatHistory.slice(-10),
                    session_id: currentSessionId,
                };
                if (documentTypeFilter && documentTypeFilter.value !== 'all') {
                    requestBody.document_type = documentTypeFilter.value;
                }

                const response = await apiFetch(apiUrl('/api/chat/compare'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(requestBody),
                    signal: _chatAbortController.signal,
                });

                hideTypingIndicator();
                const data = await response.json();

                if (data.primary_response) {
                    addCompareMessage(
                        data.primary_provider || 'Primary',
                        data.primary_response,
                        data.secondary_provider || 'Secondary',
                        data.secondary_response || '(no response)',
                        data.secondary_error || false
                    );
                    if (data.session_id) {
                        const wasNew = !currentSessionId;
                        currentSessionId = data.session_id;
                        if (wasNew) document.getElementById('active-session-title').textContent = message.slice(0, 60);
                        loadSessions();
                    }
                } else {
                    addChatMessage('assistant', 'Compare error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                if (error.name === 'AbortError') return;
                hideTypingIndicator();
                addChatMessage('assistant', 'Sorry, I encountered an error: ' + error.message);
            } finally {
                stopBtn.style.display = 'none';
                input.disabled = false;
                sendBtn.disabled = false;
                input.focus();
            }
        }

        function handleChatKeypress(event) {
            if (event.key === 'Enter' && !event.ctrlKey) {
                event.preventDefault();
                sendChatMessage();
            }
            // Ctrl+Enter falls through to default textarea behaviour (inserts newline)
        }

        function sendSuggestion(text) {
            document.getElementById('chat-input').value = text;
            sendChatMessage();
        }

        function clearChat() {
            if (!confirm('Start a new chat? The current session will remain in the sidebar.')) return;
            newChat();
        }

