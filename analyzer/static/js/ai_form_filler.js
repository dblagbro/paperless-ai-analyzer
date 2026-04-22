// ── AIFormFiller Widget ─────────────────────────────────────────────────

    // ── AIFormFiller widget ──────────────────────────────────────────────────
    // Reusable class: auto-fills any form from free-form pasted text via AI.
    // Drop this class + POST /api/ai-form/parse into any Flask project for the
    // same behaviour on any form.
    class AIFormFiller {
        static _count = 0;

        constructor({ schema, fields, container, endpoint, onComplete, projectSlug } = {}) {
            this._id         = 'aff' + (++AIFormFiller._count);
            this._schema     = schema || [];
            this._fields     = fields || {};        // {apiFieldName: '#css-selector'}
            this._container  = typeof container === 'string'
                               ? document.querySelector(container) : container;
            this._endpoint   = endpoint;
            this._onComplete = onComplete || null;
            this._slug       = projectSlug;         // string or function
            this._conversation = [];
            this._els        = {};
        }

        _pfx(s) { return `${this._id}-${s}`; }

        _currentSlug() {
            if (typeof this._slug === 'function') return this._slug();
            return this._slug || (typeof window.APP_CONFIG.currentProject !== 'undefined' ? window.APP_CONFIG.currentProject : 'default');
        }

        render() {
            if (!this._container) return;
            this._container.innerHTML = `
                <p style="color:#6b7280; margin-bottom:10px; font-size:13px;">
                    Paste an email, Slack message, or any notes that contain the credentials.
                    AI will extract the fields and ask follow-up questions if needed.
                </p>
                <textarea id="${this._pfx('paste')}"
                          placeholder="Paste text here — e.g. a Slack message, email, or attorney notes\u2026"
                          style="width:100%; box-sizing:border-box; height:90px; padding:8px 10px; border:1px solid #d1d5db; border-radius:6px; font-size:13px; resize:vertical; margin-bottom:8px;"></textarea>
                <button id="${this._pfx('parse-btn')}"
                        style="padding:7px 14px; background:#2563eb; color:#fff; border:none; border-radius:5px; cursor:pointer; font-size:13px; margin-bottom:10px;">\uD83E\uDD16 Parse with AI</button>
                <div id="${this._pfx('chat')}" style="display:none; max-height:240px; overflow-y:auto; margin-bottom:8px; padding:2px;">
                    <div id="${this._pfx('msgs')}"></div>
                </div>
                <div id="${this._pfx('reply-row')}" style="display:none; gap:6px; align-items:center; margin-bottom:4px;">
                    <input type="text" id="${this._pfx('reply')}" placeholder="Type your answer\u2026"
                           style="flex:1; padding:7px 10px; border:1px solid #d1d5db; border-radius:5px; font-size:13px;">
                    <button id="${this._pfx('send-btn')}"
                            style="padding:7px 12px; background:#2563eb; color:#fff; border:none; border-radius:5px; cursor:pointer; font-size:13px; white-space:nowrap;">Send \u2192</button>
                </div>
            `;
            this._els.paste    = document.getElementById(this._pfx('paste'));
            this._els.parseBtn = document.getElementById(this._pfx('parse-btn'));
            this._els.chat     = document.getElementById(this._pfx('chat'));
            this._els.msgs     = document.getElementById(this._pfx('msgs'));
            this._els.replyRow = document.getElementById(this._pfx('reply-row'));
            this._els.reply    = document.getElementById(this._pfx('reply'));
            this._els.sendBtn  = document.getElementById(this._pfx('send-btn'));

            this._els.parseBtn.addEventListener('click', () => this._parse());
            this._els.sendBtn.addEventListener('click', () => this._sendReply());
            this._els.reply.addEventListener('keydown', e => { if (e.key === 'Enter') this._sendReply(); });
        }

        reset() {
            this._conversation = [];
            if (!this._els.paste) return;
            this._els.paste.value            = '';
            this._els.chat.style.display     = 'none';
            this._els.msgs.innerHTML         = '';
            this._els.replyRow.style.display = 'none';
            this._els.reply.value            = '';
        }

        async _parse() {
            const rawText = (this._els.paste?.value || '').trim();
            if (!rawText) { showToast('Paste some text first', 'error'); return; }
            this._conversation = [{
                role: 'user',
                content: `Please extract the required fields from this text:\n\n${rawText}`
            }];
            this._renderBubbles();
            this._els.replyRow.style.display = 'none';
            await this._call();
        }

        async _sendReply() {
            const text = (this._els.reply?.value || '').trim();
            if (!text) return;
            this._conversation.push({ role: 'user', content: text });
            this._els.reply.value            = '';
            this._els.replyRow.style.display = 'none';
            this._renderBubbles();
            await this._call();
        }

        async _call() {
            this._els.chat.style.display = 'block';
            const thinking = document.createElement('div');
            thinking.id = this._pfx('thinking');
            thinking.innerHTML = '<em style="font-size:12px; color:#9ca3af;">\uD83E\uDD16 Thinking\u2026</em>';
            this._els.msgs.appendChild(thinking);
            try {
                const r = await apiFetch(apiUrl(this._endpoint), {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify({
                        schema:       this._schema,
                        conversation: this._conversation,
                        project_slug: this._currentSlug(),
                    })
                });
                const d = await r.json();
                document.getElementById(this._pfx('thinking'))?.remove();
                if (d.error) { showToast('AI: ' + d.error, 'error'); return; }
                this._conversation.push({ role: 'assistant', content: JSON.stringify(d) });
                this._renderBubbles();
                if (d.complete) {
                    this.autofill(d.fields || d);
                } else if (d.follow_up) {
                    this._els.replyRow.style.display = 'flex';
                    this._els.reply.focus();
                }
            } catch(e) {
                document.getElementById(this._pfx('thinking'))?.remove();
                showToast('Parse error: ' + e.message, 'error');
            }
        }

        _renderBubbles() {
            const el = this._els.msgs;
            if (!el) return;
            el.innerHTML = '';
            const PREFIX = 'Please extract the required fields from this text:\n\n';
            this._conversation.forEach(turn => {
                const div = document.createElement('div');
                div.style.cssText = 'margin-bottom:8px;';
                if (turn.role === 'user') {
                    let preview = turn.content;
                    if (preview.startsWith(PREFIX)) {
                        preview = '\uD83D\uDCCB ' + preview.replace(PREFIX, '').slice(0, 80) + (preview.length > 130 ? '\u2026' : '');
                    } else {
                        preview = '\uD83D\uDC64 ' + preview;
                    }
                    div.innerHTML = `<div style="background:#f3f4f6; padding:7px 10px; border-radius:6px; font-size:12px; color:#374151;">${escapeHtml(preview)}</div>`;
                } else {
                    try {
                        const p = JSON.parse(turn.content);
                        let html = `<div style="background:#eff6ff; border:1px solid #bfdbfe; padding:9px 11px; border-radius:6px; font-size:12px; color:#1e40af;">`;
                        html += `<div style="font-weight:600; margin-bottom:4px;">\uD83E\uDD16 ${escapeHtml(p.summary || 'Parsed')}</div>`;
                        if (p.notes) html += `<div style="color:#3b82f6; margin-bottom:4px;">${escapeHtml(p.notes)}</div>`;
                        if (p.complete) {
                            html += `<div style="color:#16a34a; font-weight:600;">\u2713 Fields ready \u2014 review the form below.</div>`;
                        } else if (p.follow_up) {
                            html += `<div style="margin-top:4px; font-style:italic;">${escapeHtml(p.follow_up)}</div>`;
                        }
                        html += '</div>';
                        div.innerHTML = html;
                    } catch {
                        div.innerHTML = `<div style="font-size:12px; color:#6b7280;">${escapeHtml(turn.content)}</div>`;
                    }
                }
                el.appendChild(div);
            });
            el.scrollTop = el.scrollHeight;
        }

        autofill(extractedFields) {
            if (!extractedFields) return;
            Object.entries(this._fields).forEach(([fieldName, selector]) => {
                const val = extractedFields[fieldName];
                if (val == null) return;
                const el = document.querySelector(selector);
                if (el) el.value = String(val);
            });
            if (this._onComplete) this._onComplete(extractedFields);
        }
    }
    // ── end AIFormFiller ─────────────────────────────────────────────────────

    // ── Court Import: state ──────────────────────────────────────────────────
    let _courtCurrentSystem = 'federal';
    let _courtCurrentCaseId = null;
    let _courtCurrentCaseNumber = '';
    let _courtCurrentCaseTitle = '';
    let _courtDocketEntries = [];
    let _courtImportJobId = null;
    let _courtPollTimer = null;
    let _courtAIFiller = null;             // AIFormFiller instance for the wizard
    const _courtProjectSlug = () => {
        // Prefer the upload tab's project selector (same context), fall back to global
        const sel = document.getElementById('upload-project-select');
        return (sel && sel.value) || window.APP_CONFIG.currentProject || 'default';
    };

    // ── Wizard ───────────────────────────────────────────────────────────────
    function openCourtWizard() {
        document.getElementById('court-wizard-overlay').style.display = 'flex';
        courtWizardStep(1);
        loadCourtCredStatus();
    }
    function closeCourtWizard() {
        document.getElementById('court-wizard-overlay').style.display = 'none';
    }
    function toggleCourtPassword(inputId, btn) {
        const el = document.getElementById(inputId);
        if (!el) return;
        const showing = el.type === 'text';
        el.type = showing ? 'password' : 'text';
        btn.textContent = showing ? 'Show' : 'Hide';
    }
    function toggleNyscefPublic(isPublic) {
        // Pro Se users still need their free NYSCEF account credentials (anonymous
        // searches hit CAPTCHA). We always show the credential fields; just update
        // the hint text to match the account type.
        const desc = document.getElementById('court-wiz-nyscef-cred-desc');
        const userLbl = document.getElementById('court-wiz-nyscef-user-lbl');
        if (desc) desc.textContent = isPublic
            ? 'Enter your free NYSCEF Pro Se account username and password (create one at iapps.courts.state.ny.us/nyscef → Unrepresented Litigants → Create an Account).'
            : 'Enter your NY Attorney Registration number and e-Filing password.';
        if (userLbl) userLbl.textContent = isPublic ? 'NYSCEF Username' : 'NY Attorney Registration #';
    }
    function courtWizardStep(n) {
        [1,2,3].forEach(i => {
            document.getElementById('court-wizard-step'+i).style.display = (i===n) ? 'block' : 'none';
        });
        if (n === 2) {
            const system = document.querySelector('input[name="court-wizard-system"]:checked')?.value || 'federal';
            _courtCurrentSystem = system;
            document.getElementById('court-wiz-test-result').innerHTML = '';
            // Reset NYSCEF public access toggle
            const pubChk = document.getElementById('court-wiz-nyscef-public');
            if (pubChk) { pubChk.checked = false; toggleNyscefPublic(false); }
            // Lazily init and always reset the AI filler on each entry to step 2
            if (!_courtAIFiller) { _courtAIFiller = _initCourtAIFiller(); }
            _courtAIFiller.reset();
            courtWizEntryMode('manual');
        }
    }

    function courtWizEntryMode(mode) {
        const manBtn  = document.getElementById('court-wiz-tab-manual');
        const pasBtn  = document.getElementById('court-wiz-tab-paste');
        const manPanel = document.getElementById('court-wiz-manual-panel');
        const pasPanel = document.getElementById('court-wiz-paste-panel');
        if (!manBtn || !manPanel) return;

        [['manual', manBtn], ['paste', pasBtn]].forEach(([m, btn]) => {
            const active = m === mode;
            btn.style.color       = active ? '#2563eb' : '#6b7280';
            btn.style.fontWeight  = active ? '600'     : '400';
            btn.style.borderBottom = active ? '2px solid #2563eb' : '2px solid transparent';
        });
        manPanel.style.display = mode === 'manual' ? '' : 'none';
        if (pasPanel) pasPanel.style.display = mode === 'paste' ? '' : 'none';

        if (mode === 'manual') {
            // Re-apply system field visibility
            const fedEl = document.getElementById('court-wizard-federal-fields');
            const nycEl = document.getElementById('court-wizard-nyscef-fields');
            if (fedEl) fedEl.style.display = _courtCurrentSystem === 'federal' ? '' : 'none';
            if (nycEl) nycEl.style.display = _courtCurrentSystem === 'nyscef'  ? '' : 'none';
        }
    }

    async function testCourtCredentials() {
        const system = _courtCurrentSystem;
        const resultEl = document.getElementById('court-wiz-test-result');
        resultEl.innerHTML = '<span style="color:#aaa;">Testing…</span>';
        const body = _buildCredBody(system);
        try {
            const r = await apiFetch(apiUrl('/api/court/credentials/test'), {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify(body)
            });
            const d = await r.json();
            if (d.ok) {
                resultEl.innerHTML = `<span style="color:#4caf50;">✓ ${escapeHtml(d.account_info)}</span>`;
            } else {
                resultEl.innerHTML = `<span style="color:#e74c3c;">✗ ${escapeHtml(d.error)}</span>`;
            }
        } catch(e) {
            resultEl.innerHTML = `<span style="color:#e74c3c;">Error: ${escapeHtml(e.message)}</span>`;
        }
    }

    async function saveCourtCredentials() {
        const system = _courtCurrentSystem;
        const body = _buildCredBody(system);
        try {
            const r = await apiFetch(apiUrl('/api/court/credentials'), {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify(body)
            });
            const d = await r.json();
            if (d.ok) {
                courtWizardStep(3);
                document.getElementById('court-wiz-confirm-msg').innerHTML =
                    `✓ ${escapeHtml(system.charAt(0).toUpperCase()+system.slice(1))} credentials saved.`;
                loadCourtCredStatus();
            } else {
                showToast(d.error || 'Save failed', 'error');
            }
        } catch(e) {
            showToast('Error: '+e.message, 'error');
        }
    }

    function _buildCredBody(system) {
        const slug = _courtProjectSlug();
        if (system === 'federal') {
            return {
                court_system: 'federal',
                project_slug: slug,
                username: document.getElementById('court-wiz-federal-user').value.trim(),
                password: document.getElementById('court-wiz-federal-pass').value,
                extra_config: {
                    pacer_login_court: document.getElementById('court-wiz-federal-court').value.trim(),
                    pacer_client_code: document.getElementById('court-wiz-federal-client').value.trim(),
                    courtlistener_api_token: document.getElementById('court-wiz-cl-token').value.trim(),
                }
            };
        } else {
            const _nyscefPublic = document.getElementById('court-wiz-nyscef-public')?.checked || false;
            return {
                court_system: 'nyscef',
                project_slug: slug,
                username: _nyscefPublic ? '' : document.getElementById('court-wiz-nyscef-user').value.trim(),
                password: _nyscefPublic ? '' : document.getElementById('court-wiz-nyscef-pass').value,
                extra_config: {
                    nyscef_county: document.getElementById('court-wiz-nyscef-county').value.trim(),
                    public_only: _nyscefPublic,
                }
            };
        }
    }

    // ── AI Paste credential parsing (via AIFormFiller) ───────────────────────

    function _initCourtAIFiller() {
        const filler = new AIFormFiller({
            schema: [
                { name: 'court_system', label: 'Court System',
                  description: 'Either "federal" (PACER/CourtListener) or "nyscef" (NY state courts)', required: true },
                { name: 'username', label: 'Username',
                  description: 'PACER username (federal) or NY Attorney Registration number (nyscef)' },
                { name: 'password', label: 'Password', description: 'Login password', secret: true },
                { name: 'pacer_client_code', label: 'PACER Client Code',
                  description: 'Optional PACER billing client code (federal only)' },
                { name: 'courtlistener_api_token', label: 'CourtListener API Token',
                  description: 'Optional free token from courtlistener.com for higher rate limits', secret: true },
                { name: 'nyscef_county', label: 'Default NYSCEF County',
                  description: 'Default NY county for NYSCEF filings' },
                { name: 'public_only', label: 'Public Access Only',
                  description: 'Set true if user is a party/defendant/plaintiff (not an attorney) with no credentials — works for both federal (CourtListener) and NYSCEF (index number only)' },
            ],
            fields: {
                pacer_client_code:       '#court-wiz-federal-client',
                courtlistener_api_token: '#court-wiz-cl-token',
                nyscef_county:           '#court-wiz-nyscef-county',
            },
            container:   '#court-wiz-paste-panel',
            endpoint:    '/api/ai-form/parse',
            projectSlug: _courtProjectSlug,
            onComplete(fields) {
                // Set court system radio
                if (fields.court_system) {
                    const radio = document.querySelector(`input[name="court-wizard-system"][value="${fields.court_system}"]`);
                    if (radio) radio.checked = true;
                    _courtCurrentSystem = fields.court_system;
                }
                const sys = _courtCurrentSystem || fields.court_system || '';
                // Handle public access flag
                if (fields.public_only) {
                    if (sys === 'nyscef') {
                        const pubChk = document.getElementById('court-wiz-nyscef-public');
                        if (pubChk) { pubChk.checked = true; toggleNyscefPublic(true); }
                    }
                    // For federal public_only, no extra action needed — no PACER creds required
                }
                // Route username/password to the correct system's inputs
                if (sys === 'federal') {
                    const u = document.getElementById('court-wiz-federal-user');
                    const p = document.getElementById('court-wiz-federal-pass');
                    if (u && fields.username) u.value = fields.username;
                    if (p && fields.password) p.value = fields.password;
                }
                if (sys === 'nyscef' && !fields.public_only) {
                    const u = document.getElementById('court-wiz-nyscef-user');
                    const p = document.getElementById('court-wiz-nyscef-pass');
                    if (u && fields.username) u.value = fields.username;
                    if (p && fields.password) p.value = fields.password;
                }
                setTimeout(() => {
                    courtWizEntryMode('manual');
                    const msg = fields.public_only
                        ? 'Public access mode set \u2014 click Test Connection to verify, then Save'
                        : 'Fields auto-filled \u2014 review and click Test Connection';
                    showToast(msg, 'info');
                }, 400);
            }
        });
        filler.render();
        return filler;
    }

    function openCourtWizardAIPaste() {
        document.getElementById('court-wizard-overlay').style.display = 'flex';
        // Default to federal; user can change by going back to step 1
        const fedRadio = document.querySelector('input[name="court-wizard-system"][value="federal"]');
        if (fedRadio) fedRadio.checked = true;
        courtWizardStep(2);       // inits/resets filler, shows step 2
        courtWizEntryMode('paste'); // open on the AI Paste tab immediately
        loadCourtCredStatus();
    }

    // ── Credential status pills ───────────────────────────────────────────────

    async function loadCourtCredStatus() {
        const slug = _courtProjectSlug();
        try {
            const r = await apiFetch(apiUrl('/api/court/credentials?project_slug='+encodeURIComponent(slug)));
            const d = await r.json();
            const creds = d.credentials || [];
            ['federal','nyscef'].forEach(sys => {
                const el = document.getElementById('court-cred-pill-'+sys);
                if (!el) return;
                const c = creds.find(x=>x.court_system===sys);
                if (c) {
                    const tested = c.last_tested_at ? (c.last_test_success ? ' ✓' : ' ✗') : '';
                    el.style.color = c.last_test_success ? '#4caf50' : '#f39c12';
                    el.style.borderColor = c.last_test_success ? '#4caf50' : '#f39c12';
                    el.textContent = `${sys.charAt(0).toUpperCase()+sys.slice(1)}: ${escapeHtml(c.username||'configured')}${tested}`;
                } else {
                    el.style.color = '#888';
                    el.style.borderColor = '#333';
                    el.textContent = `${sys.charAt(0).toUpperCase()+sys.slice(1)} — not configured`;
                }
            });
        } catch(e) { /* silently fail */ }
    }

    // ── Case Search ──────────────────────────────────────────────────────────
    async function searchCourt() {
        const system = document.getElementById('court-system-select').value;
        const caseNum = document.getElementById('court-case-number').value.trim();
        const party   = document.getElementById('court-party-name').value.trim();
        const slug    = _courtProjectSlug();
        if (!caseNum && !party) { showToast('Enter a case number or party name','error'); return; }
        const resultsEl = document.getElementById('court-search-results');
        resultsEl.innerHTML = '<span style="color:#aaa;">Searching…</span>';
        try {
            const r = await apiFetch(apiUrl('/api/court/search'), {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({court_system:system, case_number:caseNum, party_name:party, project_slug:slug})
            });
            const d = await r.json();
            if (d.error) { resultsEl.innerHTML = `<span style="color:#e74c3c;">${escapeHtml(d.error)}</span>`; return; }
            const cases = d.results || [];
            if (!cases.length) { resultsEl.innerHTML = '<span style="color:#aaa;">No results found.</span>'; return; }
            resultsEl.innerHTML = cases.map(c=>`
                <div onclick="loadDocket('${escapeHtml(system)}','${escapeHtml(c.case_id)}','${escapeHtml(c.case_number)}','${escapeHtml(c.case_title.replace(/'/g,"\\'"))}')"
                     style="padding:9px 12px; margin:4px 0; background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; cursor:pointer; transition:border-color .15s;"
                     onmouseover="this.style.borderColor='#2563eb'; this.style.background='#eff6ff';" onmouseout="this.style.borderColor='#e5e7eb'; this.style.background='#f9fafb';">
                    <strong style="color:#1d4ed8; font-size:13px;">${escapeHtml(c.case_number)}</strong>
                    <span style="color:#374151; margin-left:8px; font-size:13px;">${escapeHtml(c.case_title)}</span>
                    <span style="float:right; font-size:11px; color:#9ca3af;">${escapeHtml(c.court)} · ${escapeHtml(c.source)}</span>
                </div>
            `).join('');
        } catch(e) { resultsEl.innerHTML = `<span style="color:#e74c3c;">Error: ${escapeHtml(e.message)}</span>`; }
    }

    // ── Docket ───────────────────────────────────────────────────────────────
    async function loadDocket(system, caseId, caseNumber, caseTitle) {
        _courtCurrentSystem   = system;
        _courtCurrentCaseId   = caseId;
        _courtCurrentCaseNumber = caseNumber;
        _courtCurrentCaseTitle  = caseTitle;
        const slug = _courtProjectSlug();
        const section = document.getElementById('court-docket-section');
        document.getElementById('court-docket-case-title').textContent = caseTitle || caseNumber;
        document.getElementById('court-docket-meta').textContent = 'Loading docket…';
        document.getElementById('court-docket-tbody').innerHTML = '<tr><td colspan="5" style="padding:12px; color:#aaa;">Loading…</td></tr>';
        document.getElementById('court-import-progress').style.display = 'none';
        section.style.display = 'block';

        try {
            const r = await apiFetch(apiUrl(`/api/court/docket/${encodeURIComponent(system)}/${encodeURIComponent(caseId)}?project_slug=${encodeURIComponent(slug)}`));
            const d = await r.json();
            if (d.error) {
                const isRECAPErr = d.error.includes('RECAP') || d.error.includes('401') ||
                                   d.error.includes('403') || d.error.includes('contributor');
                const isTokenErr = d.error.includes('token') || d.error.includes('Authentication') || isRECAPErr;
                document.getElementById('court-docket-meta').textContent = isTokenErr ? 'RECAP access required' : 'Error: '+d.error;
                if (isTokenErr) {
                    document.getElementById('court-docket-tbody').innerHTML =
                        `<tr><td colspan="5" style="padding:16px 12px;">
                            <div style="color:#92400e; background:#fffbeb; border:1px solid #fcd34d; border-radius:6px; padding:14px 16px; font-size:13px; line-height:1.6;">
                                <strong style="font-size:14px;">&#9888;&#65039; This case has no documents in the RECAP archive</strong><br>
                                <span style="color:#6b7280; font-size:12px;">RECAP only contains documents that were previously accessed by RECAP browser extension users. New cases are PACER-only until someone contributes them.</span>
                                <div style="margin-top:12px; border-top:1px solid #fde68a; padding-top:12px;">
                                    <strong>Option A &mdash; PACER credentials</strong> <em style="color:#6b7280;">(direct access, works immediately)</em><br>
                                    1. Register or log in at <a href="https://pacer.login.uscourts.gov" target="_blank" style="color:#d97706; font-weight:600;">pacer.login.uscourts.gov</a> &mdash; free account, instant approval.<br>
                                    2. Enter your PACER username &amp; password in <strong>&#9881;&#65039; Manage Credentials &rarr; Federal &rarr; PACER</strong>.<br>
                                    3. Re-search the case &mdash; documents will load directly from PACER.
                                </div>
                                <div style="margin-top:12px; border-top:1px solid #fde68a; padding-top:12px;">
                                    <strong>Option B &mdash; RECAP browser extension</strong> <em style="color:#6b7280;">(free, builds public archive)</em><br>
                                    1. Install <a href="https://free.law/recap/" target="_blank" style="color:#d97706; font-weight:600;">the free RECAP extension</a> in Chrome or Firefox.<br>
                                    2. Browse the case in PACER normally &mdash; RECAP automatically donates documents to the public archive.<br>
                                    3. Come back here; documents will appear once contributed.
                                </div>
                            </div>
                        </td></tr>`;
                }
                return;
            }
            _courtDocketEntries = d.entries || [];
            document.getElementById('court-docket-meta').textContent =
                `${_courtDocketEntries.length} documents · ${system} · ${caseNumber}`;
            document.getElementById('btn-court-import-all').textContent =
                `Import All (${_courtDocketEntries.length} docs)`;
            renderDocket(_courtDocketEntries);
        } catch(e) {
            document.getElementById('court-docket-meta').textContent = 'Error: '+e.message;
        }
    }

    function renderDocket(entries) {
        const tbody = document.getElementById('court-docket-tbody');
        if (!entries.length) {
            tbody.innerHTML = '<tr><td colspan="5" style="padding:12px; color:#aaa;">No docket entries found.</td></tr>';
            return;
        }
        tbody.innerHTML = entries.map(e=>`
            <tr style="border-bottom:1px solid #f3f4f6;">
                <td style="padding:5px 8px;"><input type="checkbox" class="court-doc-check" data-seq="${escapeHtml(e.seq)}"></td>
                <td style="padding:5px 8px; color:#6b7280; white-space:nowrap; font-size:11px;">${escapeHtml(e.seq)}</td>
                <td style="padding:5px 8px; color:#6b7280; white-space:nowrap; font-size:11px;">${escapeHtml(e.date||'')}</td>
                <td style="padding:5px 8px; color:#374151; font-size:12px;">${escapeHtml(e.title||'—')}</td>
                <td style="padding:5px 8px;">
                    <span style="background:${e.source==='recap'?'#dcfce7':e.source==='pacer'?'#fee2e2':'#dbeafe'}; color:${e.source==='recap'?'#15803d':e.source==='pacer'?'#dc2626':'#1d4ed8'}; border-radius:4px; padding:2px 7px; font-size:11px; font-weight:500;">${escapeHtml(e.source)}</span>
                </td>
            </tr>
        `).join('');
        // Update "Import Selected" button visibility
        document.getElementById('court-docket-tbody').querySelectorAll('.court-doc-check').forEach(cb=>{
            cb.addEventListener('change', _updateImportSelected);
        });
    }

    function filterDocket() {
        const q = document.getElementById('court-docket-filter').value.toLowerCase();
        const src = document.getElementById('court-source-filter').value;
        const filtered = _courtDocketEntries.filter(e=>{
            if (q && !e.title?.toLowerCase().includes(q) && !e.seq?.toLowerCase().includes(q)) return false;
            if (src && e.source !== src) return false;
            return true;
        });
        renderDocket(filtered);
    }

    function courtSelectAll(cb) {
        document.querySelectorAll('.court-doc-check').forEach(c=>{ c.checked = cb.checked; });
        _updateImportSelected();
    }

    function _updateImportSelected() {
        const checked = document.querySelectorAll('.court-doc-check:checked').length;
        const btn = document.getElementById('btn-court-import-selected');
        if (btn) {
            btn.style.display = checked > 0 ? 'inline-flex' : 'none';
            btn.textContent = `Import Selected (${checked})`;
        }
    }

    // ── Import ───────────────────────────────────────────────────────────────
    async function startCourtImport(importAll) {
        if (!_courtCurrentCaseId) { showToast('No case selected', 'error'); return; }
        const slug = _courtProjectSlug();
        const progressEl = document.getElementById('court-import-progress');
        progressEl.style.display = 'block';
        document.getElementById('court-import-bar').style.width = '0%';
        document.getElementById('court-import-counters').textContent = 'Starting…';
        document.getElementById('court-import-log').textContent = '';

        try {
            const r = await apiFetch(apiUrl('/api/court/import/start'), {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({
                    court_system:  _courtCurrentSystem,
                    case_id:       _courtCurrentCaseId,
                    case_number:   _courtCurrentCaseNumber,
                    case_title:    _courtCurrentCaseTitle,
                    project_slug:  slug,
                })
            });
            const d = await r.json();
            if (d.error) { showToast(d.error, 'error'); return; }
            _courtImportJobId = d.job_id;
            if (_courtPollTimer) clearInterval(_courtPollTimer);
            _courtPollTimer = setInterval(()=>pollCourtImport(_courtImportJobId), 2000);
        } catch(e) {
            showToast('Error: '+e.message, 'error');
        }
    }

    async function pollCourtImport(jobId) {
        try {
            const r = await apiFetch(apiUrl(`/api/court/import/status/${jobId}?log_lines=20`));
            const d = await r.json();
            if (d.error) { clearInterval(_courtPollTimer); return; }

            const total = d.total_docs || 1;
            const done  = (d.imported_docs||0) + (d.skipped_docs||0) + (d.failed_docs||0);
            const pct   = total > 0 ? Math.round(done/total*100) : 0;
            document.getElementById('court-import-bar').style.width = pct + '%';
            document.getElementById('court-import-counters').innerHTML =
                `${d.imported_docs||0} imported &nbsp;·&nbsp; ${d.skipped_docs||0} skipped &nbsp;·&nbsp; ${d.failed_docs||0} failed &nbsp;·&nbsp; ${pct}%`;

            const logEl = document.getElementById('court-import-log');
            if (d.log_tail && d.log_tail.length) {
                logEl.textContent = d.log_tail.join('\n');
                logEl.scrollTop = logEl.scrollHeight;
            }

            if (['completed','failed','cancelled'].includes(d.status)) {
                clearInterval(_courtPollTimer);
                _courtPollTimer = null;
                document.getElementById('btn-court-cancel').style.display = 'none';
                const color = d.status==='completed' ? '#4caf50' : d.status==='cancelled' ? '#f39c12' : '#e74c3c';
                document.getElementById('court-import-counters').innerHTML +=
                    `&nbsp;&nbsp;<strong style="color:${color};">${d.status.toUpperCase()}</strong>`;
                loadCourtHistory();
            }
        } catch(e) { /* silently continue polling */ }
    }

    async function cancelCourtImport() {
        if (!_courtImportJobId) return;
        try {
            await apiFetch(apiUrl(`/api/court/import/cancel/${_courtImportJobId}`), {method:'POST'});
            showToast('Cancel signal sent', 'info');
        } catch(e) {
            showToast('Error: '+e.message, 'error');
        }
    }

    // ── History ──────────────────────────────────────────────────────────────
    function _fmtDuration(start, end) {
        if (!start || !end) return null;
        const s = Math.round((new Date(end) - new Date(start)) / 1000);
        if (s < 1) return '< 1s';
        const m = Math.floor(s / 60), ss = s % 60;
        return m > 0 ? `${m}m ${ss}s` : `${ss}s`;
    }

    function toggleCourtJobLogs(jobId, btn) {
        const row = document.getElementById('court-log-' + jobId);
        if (!row) return;
        const hidden = row.style.display === 'none' || !row.style.display;
        row.style.display = hidden ? '' : 'none';
        btn.textContent = hidden ? '▲' : '📋';
    }

    async function loadCourtHistory() {
        const slug = _courtProjectSlug();
        const el = document.getElementById('court-history-table');
        if (!el) return;
        try {
            const r = await apiFetch(apiUrl(`/api/court/import/history?project_slug=${encodeURIComponent(slug)}&limit=20`));
            const d = await r.json();
            const jobs = d.jobs || [];
            if (!jobs.length) { el.innerHTML = '<span style="color:#555;">No import jobs yet.</span>'; return; }
            const statusColor = {completed:'#16a34a', failed:'#dc2626', cancelled:'#d97706', running:'#2563eb', queued:'#6b7280'};
            const rows = jobs.map(j => {
                const dur = _fmtDuration(j.started_at, j.completed_at);
                const logLines = (j.log_tail || []).join('\n');
                const detailRow = `<tr id="court-log-${j.id}" style="display:none; background:#f9fafb;">
                    <td colspan="6" style="padding:8px 14px; font-size:12px;">
                        ${j.error_message ? `<div style="color:#dc2626; margin-bottom:4px;">⚠️ ${escapeHtml(j.error_message)}</div>` : ''}
                        ${dur ? `<div style="color:#6b7280; margin-bottom:4px;">Duration: ${dur}</div>` : ''}
                        ${logLines ? `<pre style="margin:0; font-size:11px; color:#374151; white-space:pre-wrap; max-height:200px; overflow-y:auto; background:#f1f5f9; padding:6px 8px; border-radius:4px;">${escapeHtml(logLines)}</pre>` : '<span style="color:#9ca3af;">No log data.</span>'}
                    </td>
                </tr>`;
                const mainRow = `<tr style="border-bottom:1px solid #f3f4f6;">
                    <td style="padding:6px 8px; color:#374151; font-weight:500;" title="${escapeHtml(j.case_title||'')}">
                        ${escapeHtml(j.case_number)}</td>
                    <td style="padding:6px 8px; color:#6b7280;">${escapeHtml(j.court_system)}</td>
                    <td style="padding:6px 8px; text-align:right; white-space:nowrap;">
                        <span style="color:#16a34a;">${j.imported_docs||0}</span>&nbsp;
                        <span style="color:#d97706;">${j.skipped_docs||0}</span>&nbsp;
                        <span style="color:#dc2626;">${j.failed_docs||0}</span>
                    </td>
                    <td style="padding:6px 8px; color:#9ca3af;">
                        ${escapeHtml((j.created_at||'').split('T')[0])}</td>
                    <td style="padding:6px 8px;">
                        <span style="color:${statusColor[j.status]||'#6b7280'}; font-weight:500;">${escapeHtml(j.status)}</span>
                    </td>
                    <td style="padding:6px 8px; text-align:center; white-space:nowrap;">
                        <button style="padding:3px 6px; font-size:11px; background:#f3f4f6; color:#374151; border:1px solid #d1d5db; border-radius:4px; cursor:pointer; margin-right:4px;"
                            title="Show log" onclick="toggleCourtJobLogs(${j.id}, this)">📋</button>
                        <button style="padding:3px 10px; font-size:11px; background:#f3f4f6; color:#374151; border:1px solid #d1d5db; border-radius:4px; cursor:pointer;"
                            onclick="loadDocket('${escapeHtml(j.court_system)}','${escapeHtml(j.case_number)}','${escapeHtml(j.case_number)}','${escapeHtml((j.case_title||'').replace(/'/g,"\\'"))}')">
                            ↺ Sync Again
                        </button>
                    </td>
                </tr>`;
                return mainRow + detailRow;
            }).join('');
            el.innerHTML = `<table style="width:100%; border-collapse:collapse; font-size:12px;">
                <thead><tr style="background:#f6f8fa; color:#6b7280; text-transform:uppercase; font-size:11px; text-align:left;">
                    <th style="padding:6px 8px; border-bottom:1px solid #e5e7eb;">Case</th>
                    <th style="padding:6px 8px; border-bottom:1px solid #e5e7eb;">System</th>
                    <th style="padding:6px 8px; border-bottom:1px solid #e5e7eb; text-align:right;">↑ Skip ✗</th>
                    <th style="padding:6px 8px; border-bottom:1px solid #e5e7eb;">Date</th>
                    <th style="padding:6px 8px; border-bottom:1px solid #e5e7eb;">Status</th>
                    <th style="padding:6px 8px; border-bottom:1px solid #e5e7eb; text-align:center;">Actions</th>
                </tr></thead>
                <tbody>${rows}</tbody></table>`;
        } catch(e) { el.innerHTML = `<span style="color:#e74c3c;">Error: ${escapeHtml(e.message)}</span>`; }
    }
