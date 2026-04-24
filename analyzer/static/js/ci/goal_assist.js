    // ── Goal Assistant ────────────────────────────────────────────────────────
    let _ciGoalMessages = [];    // local conversation history
    let _ciGoalThinking = false;

    async function ciOpenGoalAssistant() {
        _ciGoalMessages = [];
        document.getElementById('ci-goal-input').value = '';
        document.getElementById('ci-goal-messages').innerHTML = '';
        document.getElementById('ci-goal-modal').style.display = 'flex';
        // Send an empty first turn so the AI greets with context
        await ciGoalSend('');
    }

    function ciCloseGoalAssistant() {
        document.getElementById('ci-goal-modal').style.display = 'none';
    }

    function _ciGoalRole() { return document.getElementById('ci-role')?.value || 'neutral'; }
    function _ciGoalJurisdiction() {
        const sel = document.getElementById('ci-jurisdiction-select');
        if (sel && sel.selectedIndex > 0) {
            return sel.options[sel.selectedIndex].text;
        }
        return 'Not specified';
    }

    function _ciGoalAppendBubble(role, text, suggestedGoal) {
        const wrap = document.getElementById('ci-goal-messages');
        const isAI = role === 'assistant';
        const div = document.createElement('div');
        div.style.cssText = `display:flex; flex-direction:column; align-items:${isAI ? 'flex-start' : 'flex-end'};`;

        // Bubble
        const bubble = document.createElement('div');
        bubble.style.cssText = `max-width:88%; padding:10px 13px; border-radius:${isAI ? '4px 12px 12px 12px' : '12px 4px 12px 12px'}; font-size:13px; line-height:1.5; white-space:pre-wrap; background:${isAI ? '#f4f0f9' : '#8e44ad'}; color:${isAI ? '#2c3e50' : '#fff'};`;
        bubble.textContent = text;
        div.appendChild(bubble);

        // "Use this goal" button on AI messages that have a suggestion
        if (isAI && suggestedGoal) {
            const btn = document.createElement('button');
            btn.textContent = '← Use this as my goal';
            btn.style.cssText = 'margin-top:5px; background:#8e44ad; color:white; border:none; border-radius:5px; padding:4px 12px; font-size:12px; font-weight:600; cursor:pointer;';
            btn.onclick = () => {
                document.getElementById('ci-goal').value = suggestedGoal;
                ciCloseGoalAssistant();
                showToast('Goal updated!', 'success');
            };
            div.appendChild(btn);
        }

        wrap.appendChild(div);
        wrap.scrollTop = wrap.scrollHeight;
        return div;
    }

    async function ciGoalSend(overrideText) {
        if (_ciGoalThinking) return;
        const inputEl = document.getElementById('ci-goal-input');
        const text = overrideText !== undefined ? overrideText : inputEl.value.trim();
        // Don't add empty user message on first greeting call (overrideText === '')
        if (text) {
            _ciGoalMessages.push({role: 'user', content: text});
            _ciGoalAppendBubble('user', text, null);
            inputEl.value = '';
        }

        // Thinking indicator
        _ciGoalThinking = true;
        const thinkingDiv = _ciGoalAppendBubble('assistant', '…', null);

        try {
            const r = await apiFetch(apiUrl('/api/ci/goal-assistant'), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    messages: _ciGoalMessages,
                    context: {
                        role: _ciGoalRole(),
                        jurisdiction: _ciGoalJurisdiction(),
                        draft_goal: document.getElementById('ci-goal')?.value?.trim() || '',
                    },
                }),
            });
            const d = await r.json();
            thinkingDiv.remove();
            if (!r.ok) {
                _ciGoalAppendBubble('assistant', '⚠️ ' + (d.error || 'Error from AI'), null);
                _ciGoalThinking = false;
                return;
            }
            _ciGoalMessages.push({role: 'assistant', content: d.response});
            _ciGoalAppendBubble('assistant', d.response, d.suggested_goal || null);
        } catch(e) {
            thinkingDiv.remove();
            _ciGoalAppendBubble('assistant', '⚠️ ' + e.message, null);
        }
        _ciGoalThinking = false;
    }

    async function ciOpenShareModal() {
        if (!ciSelectedRunId) return;
        const selector = document.getElementById('ci-run-selector');
        const label = selector?.options[selector.selectedIndex]?.text || ciSelectedRunId;
        document.getElementById('ci-share-run-label').textContent = label;
        document.getElementById('ci-share-username-input').value = '';
        document.getElementById('ci-share-error').style.display = 'none';
        document.getElementById('ci-share-modal').style.display = 'flex';
        await ciRefreshShareList();
    }

    function ciCloseShareModal() {
        document.getElementById('ci-share-modal').style.display = 'none';
    }

    async function ciRefreshShareList() {
        const listEl = document.getElementById('ci-share-list');
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + ciSelectedRunId + '/shares'));
            const d = await r.json();
            if (!r.ok) { listEl.innerHTML = `<div style="color:#e74c3c;font-size:13px;">${escapeHtml(d.error||'Error')}</div>`; return; }
            const shares = d.shares || [];
            if (!shares.length) {
                listEl.innerHTML = '<div style="color:#999;font-size:13px;">No one yet</div>';
                return;
            }
            listEl.innerHTML = shares.map(s => `
                <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 2px; border-bottom:1px solid #f0f0f0;">
                    <span style="font-size:13px;">👤 <strong>${escapeHtml(s.display_name)}</strong> <span style="color:#888;font-size:11px;">(${escapeHtml(s.username)})</span></span>
                    <button onclick="ciRemoveShare(${s.user_id})" style="background:none; border:none; color:#e74c3c; cursor:pointer; font-size:16px; padding:0 4px;" title="Remove access">×</button>
                </div>`).join('');
        } catch(e) {
            listEl.innerHTML = `<div style="color:#e74c3c;font-size:13px;">Failed to load</div>`;
        }
    }

    async function ciAddShare() {
        const username = document.getElementById('ci-share-username-input').value.trim();
        const errEl = document.getElementById('ci-share-error');
        errEl.style.display = 'none';
        if (!username) { errEl.textContent = 'Enter a username.'; errEl.style.display = 'block'; return; }
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + ciSelectedRunId + '/shares'), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username}),
            });
            const d = await r.json();
            if (!r.ok) { errEl.textContent = d.error || 'Failed'; errEl.style.display = 'block'; return; }
            document.getElementById('ci-share-username-input').value = '';
            showToast(`Run shared with ${d.shared_with}`, 'success');
            await ciRefreshShareList();
        } catch(e) {
            errEl.textContent = e.message; errEl.style.display = 'block';
        }
    }

    async function ciRemoveShare(uid) {
        if (!confirm('Remove this user\'s access?')) return;
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + ciSelectedRunId + '/shares/' + uid), { method: 'DELETE' });
            const d = await r.json();
            if (!r.ok) { showToast(d.error || 'Failed', 'error'); return; }
            showToast('Access removed', 'success');
            await ciRefreshShareList();
        } catch(e) {
            showToast(e.message, 'error');
        }
    }

    function ciRenderFindings(d, runId) {
        document.getElementById('ci-no-findings').style.display = 'none';
        document.getElementById('ci-findings-status').style.display = 'none';
        document.getElementById('ci-key-findings-section').style.display = 'block';
        if (window.APP_CONFIG.isAdvanced && document.getElementById('ci-report-builder')) {
            document.getElementById('ci-report-builder').style.display = 'block';
        }

        // ── Interrupted run banner (shown above meta header) ─────────────
        const _prevBanner = document.getElementById('ci-interrupted-banner');
        if (_prevBanner) _prevBanner.remove();
        if (d.status === 'interrupted') {
            const _banner = document.createElement('div');
            _banner.id = 'ci-interrupted-banner';
            _banner.style.cssText = 'background:#fff8e1; border:1px solid #f9a825; border-radius:8px; padding:12px 16px; margin-bottom:12px; font-size:13px; display:flex; align-items:center; gap:12px;';
            const _pct = (d.run_meta && d.run_meta.progress_pct) || 0;
            const _docs = (d.run_meta && d.run_meta.docs_processed) || 0;
            const _total = (d.run_meta && d.run_meta.docs_total) || 0;
            _banner.innerHTML = `
                <span style="font-size:20px;">⚡</span>
                <div style="flex:1;">
                    <strong style="color:#e65100;">Run Interrupted</strong> — service restart at ${_pct.toFixed(0)}% progress
                    (${_docs}/${_total} docs). Partial results below.
                </div>
                <button onclick="ciRerunInterrupted('${runId}')"
                        style="background:#e65100; color:white; border:none; border-radius:6px; padding:7px 14px; font-size:13px; font-weight:600; cursor:pointer; white-space:nowrap;">
                    🔄 Re-run Same Settings
                </button>`;
            const metaEl = document.getElementById('ci-run-meta-header');
            metaEl.parentNode.insertBefore(_banner, metaEl);
        }

        // ── Run metadata header ───────────────────────────────────────────
        const meta = d.run_meta || {};
        if (meta && Object.keys(meta).length) {
            document.getElementById('ci-run-meta-header').style.display = 'block';
            document.getElementById('ci-meta-run-by').textContent  = meta.run_by || '—';
            document.getElementById('ci-meta-role').textContent    = meta.role || '—';
            document.getElementById('ci-meta-duration').textContent = meta.duration || '—';
            document.getElementById('ci-meta-goal').textContent    = meta.goal_text || '—';
            document.getElementById('ci-meta-docs').textContent    =
                meta.docs_total ? `${meta.docs_total} documents` : '—';
            document.getElementById('ci-meta-cost').textContent    =
                meta.cost_usd != null ? `$${parseFloat(meta.cost_usd).toFixed(4)}` : '—';
            const started = meta.created_at ? meta.created_at.substring(0,16).replace('T',' ') : '—';
            document.getElementById('ci-meta-started').textContent = started;
        }

        // Key findings — from Director summary, augmented with extracted judgments/rulings
        const kf = document.getElementById('ci-key-findings-list');
        const summary = d.findings_summary;

        // Extract judgment/ruling events from timeline for prominent display
        const _judgmentTypes = /judgment|judgement|ruling|order|verdict|award|decree|sentence|conviction|acquittal|dismissal|settlement|injunction|hearing|decision|opinion/i;
        const _moneyRe = /\$[\d,]+|\d+[\s,]+(?:million|thousand|hundred)/i;
        const allEvents = d.timeline || [];
        const judgmentEvents = allEvents.filter(ev =>
            _judgmentTypes.test(ev.event_type || '') || _judgmentTypes.test(ev.description || ''));
        const moneyEvents   = allEvents.filter(ev =>
            _moneyRe.test(ev.description || ''));

        // Build prominent Judgments & Rulings block
        let judgmentHtml = '';
        if (judgmentEvents.length) {
            judgmentHtml = `<div style="margin-bottom:12px; padding:10px 14px; background:#eaf4fb; border:1px solid #aed6f1; border-radius:6px;">
                <div style="font-weight:700; color:#1a5276; font-size:12px; margin-bottom:8px;">
                    🏛️ Judgments &amp; Rulings <span style="font-weight:400; color:#666;">(${judgmentEvents.length})</span>
                </div>
                ${judgmentEvents.slice(0, 12).map(ev => `
                <div style="display:flex; gap:8px; align-items:flex-start; padding:4px 0; border-bottom:1px solid #d6eaf8; font-size:12px;">
                    <span class="severity-badge ${ev.significance || 'medium'}" style="flex-shrink:0;">${escapeHtml(ev.event_type || 'ruling')}</span>
                    <div>
                        <strong>${escapeHtml(ev.event_date || '')}</strong>
                        ${ev.event_date ? ' — ' : ''}${escapeHtml(ev.description || '')}
                    </div>
                </div>`).join('')}
            </div>`;
        }

        // Build monetary totals block from timeline events mentioning money
        let moneyHtml = '';
        const moneyDescs = [...new Set(moneyEvents.map(e => e.description).filter(Boolean))].slice(0, 8);
        if (moneyDescs.length) {
            moneyHtml = `<div style="margin-bottom:12px; padding:10px 14px; background:#eafaf1; border:1px solid #a9dfbf; border-radius:6px;">
                <div style="font-weight:700; color:#1e8449; font-size:12px; margin-bottom:8px;">
                    💰 Financial Amounts &amp; Awards
                </div>
                ${moneyDescs.map(desc => `<div style="font-size:12px; padding:3px 0; border-bottom:1px solid #d5f5e3;">
                    ${escapeHtml(desc)}
                </div>`).join('')}
            </div>`;
        }

        if (summary && summary.key_findings && summary.key_findings.length) {
            kf.innerHTML = judgmentHtml + moneyHtml + summary.key_findings.map(f => `
                <div class="ci-finding-item">
                    <span class="severity-badge ${f.severity || 'medium'}">${(f.severity || 'medium').toUpperCase()}</span>
                    <strong>${escapeHtml(f.finding || '')}</strong>
                    <p style="font-size:12px; color:#555; margin:4px 0 0 0;">${escapeHtml(f.detail || '')}</p>
                    ${f.recommended_action ? '<p style="font-size:11px; color:#2980b9; margin:4px 0 0 0;">→ ' + escapeHtml(f.recommended_action) + '</p>' : ''}
                </div>
            `).join('');
            document.getElementById('ci-key-findings-count').textContent =
                '(' + summary.key_findings.length + (judgmentEvents.length ? ` + ${judgmentEvents.length} rulings` : '') + ')';
        } else {
            // No Director summary — still show extracted judgments + prompt note
            const _runStatus = d.status || '';
            let _noSummaryMsg;
            if (_runStatus === 'cancelled') {
                _noSummaryMsg = 'This run was <strong>cancelled</strong> before the Director synthesis step completed. Start a new run to generate the full analysis.';
            } else if (_runStatus === 'budget_blocked') {
                _noSummaryMsg = 'This run was <strong>stopped by the budget ceiling</strong> before the Director synthesis step completed. Increase the budget or run with fewer documents.';
            } else if (_runStatus === 'interrupted') {
                _noSummaryMsg = 'This run was <strong>interrupted by a service restart</strong> before the Director synthesis step completed. Partial extraction results may be shown above. Click <em>Re-run Same Settings</em> to restart.';
            } else if (_runStatus === 'failed') {
                _noSummaryMsg = 'This run <strong>failed</strong> before the Director synthesis step completed. Check the run log for details.';
            } else {
                _noSummaryMsg = 'This section is synthesized by the Director LLM from entities, contradictions, and theories. '
                    + 'If this run completed but shows nothing here, the Director synthesis step may have failed or been skipped.'
                    + '<br><br>Try running again with a more specific <em>Goal</em> — e.g.: '
                    + '<em>"Identify all financial transactions made by the receiver and whether they were authorized by the court order."</em>';
            }
            kf.innerHTML = judgmentHtml + moneyHtml +
                `<div style="background:#fff8f0; border:1px solid #f0d090; border-radius:6px; padding:12px 16px; font-size:12px; color:#666;">
                    <strong style="color:#b8860b;">AI key-findings summary was not generated for this run.</strong><br>
                    ${_noSummaryMsg}
                </div>`;
            const rCount = judgmentEvents.length;
            document.getElementById('ci-key-findings-count').textContent =
                rCount ? `(${rCount} rulings extracted)` : '';
        }

        // Timeline — grouped by year, expandable, newest year open by default
        const tl = document.getElementById('ci-timeline-list');
        const events = d.timeline || [];
        document.getElementById('ci-timeline-count').textContent = '(' + events.length + ')';
        if (!events.length) {
            tl.innerHTML = '<p style="color:#888; font-size:13px;">No timeline events.</p>';
        } else {
            const byYear = {};
            for (const ev of events) {
                const yr = (ev.event_date || '').substring(0, 4) || 'Unknown';
                if (!byYear[yr]) byYear[yr] = [];
                byYear[yr].push(ev);
            }
            const years = Object.keys(byYear).sort((a, b) => b.localeCompare(a));
            let html = '';
            let isFirst = true;
            for (const yr of years) {
                const evs = byYear[yr];
                html += `<details class="ci-tl-year" ${isFirst ? 'open' : ''} style="margin-bottom:8px;">
                    <summary style="padding:7px 12px; background:#2c3e50; color:white; cursor:pointer; border-radius:5px; font-size:13px; font-weight:600; list-style:none; display:flex; justify-content:space-between; align-items:center; user-select:none;">
                        <span>📅 ${escapeHtml(yr)}</span>
                        <span style="background:rgba(255,255,255,0.2); border-radius:10px; padding:1px 8px; font-size:11px;">${evs.length} event${evs.length !== 1 ? 's' : ''}</span>
                    </summary>
                    <div style="padding:4px 0 2px 0;">
                        ${evs.map(ev => `<div class="ci-finding-item" style="margin-left:4px;">
                            <span class="severity-badge ${ev.significance || 'medium'}">${escapeHtml(ev.event_type || 'event')}</span>
                            <strong>${escapeHtml(ev.event_date || 'Date unknown')}</strong>
                            — ${escapeHtml(ev.description || '')}
                        </div>`).join('')}
                    </div>
                </details>`;
                isFirst = false;
            }
            tl.innerHTML = html;
        }

        // Entities — grouped by type with collapsible accordions
        const el = document.getElementById('ci-entities-list');
        const entities = d.entities || [];
        document.getElementById('ci-entities-count').textContent = '(' + entities.length + ')';
        if (!entities.length) {
            el.innerHTML = '<p style="color:#888; font-size:13px;">No entities extracted.</p>';
        } else {
            // Group by canonical category
            const _grp = (type, name) => {
                const t = (type || '').toLowerCase();
                const n = name || '';
                if (/person|individual|plaintiff|defendant|witness|attorney|counsel|judge|magistrate|justice|expert|trustee|receiver|beneficiary|heir|deponent|doctor|physician|officer|agent|ceo|cfo|coo|president|director|owner|partner|shareholder|principal|employee|staff|representative|guardian|administrator|notary|appraiser|broker|realtor|auditor/.test(t)) return 'People';
                if (/law.?firm|attorneys.at.law|llp$|pllc$|p\.c\./.test(t) || /llp$|pllc$|law firm|law group|& associates/.test(n)) return 'Law Firms';
                if (/court|tribunal|arbitration|supreme|appellate|district court|family court|surrogate|bankruptcy court/.test(t)) return 'Courts';
                if (/account|bank.account|financial.account|routing|iban|swift|checking|savings/.test(t)) return 'Bank Accounts';
                if (/address|property|real.estate|premises|parcel|lot\b|unit\b|suite\b|apt\b/.test(t)) return 'Addresses & Properties';
                if (/city|town|village|county|state|country|jurisdiction|district|borough|municipality/.test(t)) return 'Locations';
                if (/corp|corporation|llc|ltd|inc|company|business|organization|entity|association|institution|bank\b|credit.union|trust|fund|estate|agency|department|government|authority/.test(t)) return 'Organizations';
                if (/document|exhibit|filing|motion|order|judgment|deed|contract|agreement|note\b|loan|mortgage|lease|instrument/.test(t)) return 'Documents & Filings';
                if (/date|period|time|year|month|quarter/.test(t)) return 'Dates & Periods';
                // Fallback: check name patterns
                if (/LLP$|PLLC$|P\.C\.$|Law Office|Law Group/.test(n)) return 'Law Firms';
                if (/LLC$|Corp\.?$|Inc\.?$|Ltd\.?$|Co\.?$/.test(n)) return 'Organizations';
                return 'Other';
            };
            const _GRPORDER = ['People','Organizations','Law Firms','Courts','Bank Accounts','Addresses & Properties','Locations','Documents & Filings','Dates & Periods','Other'];
            const _GRPICONS = {'People':'👤','Organizations':'🏢','Law Firms':'⚖️','Courts':'🏛️','Bank Accounts':'🏦','Addresses & Properties':'📍','Locations':'🗺️','Documents & Filings':'📋','Dates & Periods':'📅','Other':'📌'};
            const groups = {};
            for (const e of entities) {
                const g = _grp(e.entity_type, e.name);
                if (!groups[g]) groups[g] = [];
                groups[g].push(e);
            }
            let html = '';
            let first = true;
            for (const g of _GRPORDER) {
                if (!groups[g]?.length) continue;
                const items = groups[g];
                html += `<details class="ci-entity-group" ${first ? 'open' : ''} style="margin-bottom:5px; border:1px solid #e0e4ea; border-radius:5px; overflow:hidden;">
                    <summary style="padding:7px 12px; background:#f5f7fa; cursor:pointer; font-size:12px; font-weight:600; color:#2c3e50; list-style:none; display:flex; justify-content:space-between; align-items:center; user-select:none;">
                        <span>${_GRPICONS[g] || '📌'} ${escapeHtml(g)}</span>
                        <span style="background:#2c3e50; color:white; border-radius:10px; padding:1px 8px; font-size:11px;">${items.length}</span>
                    </summary>
                    <div style="overflow-x:auto;">
                    <table style="width:100%; border-collapse:collapse; font-size:12px;">
                        <tr style="background:#ecf0f1;"><th style="padding:4px 8px; text-align:left; font-weight:600;">Type</th><th style="padding:4px 8px; text-align:left; font-weight:600;">Name</th><th style="padding:4px 8px; text-align:left; font-weight:600;">Role in Case</th></tr>
                        ${items.map(e => { let al=''; try { const aa=JSON.parse(e.aliases||'[]'); if(aa.length) al='<div style="font-size:10px;color:#bbb;margin-top:1px;">(also: '+aa.map(x=>escapeHtml(x)).join(', ')+')</div>'; } catch(err){} return `<tr style="border-top:1px solid #f0f0f0;"><td style="padding:3px 8px; color:#888; font-size:11px;">${escapeHtml(e.entity_type || '')}</td><td style="padding:3px 8px; font-weight:600;">${escapeHtml(e.name || '')}${al}</td><td style="padding:3px 8px; color:#555;">${escapeHtml(e.role_in_case || '')}</td></tr>`; }).join('')}
                    </table>
                    </div>
                </details>`;
                first = false;
            }
            el.innerHTML = html;
        }

        // Disputed facts
        const df = document.getElementById('ci-disputed-list');
        const disputed = d.disputed_facts || [];
        document.getElementById('ci-disputed-count').textContent = '(' + disputed.length + ')';
        df.innerHTML = disputed.length ? disputed.map(f => `
            <div class="ci-finding-item">
                <strong>${escapeHtml(f.fact_description || '')}</strong>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:6px; font-size:12px;">
                    <div><span style="color:#e74c3c; font-weight:600;">${escapeHtml(f.position_a_label || 'Position A')}:</span><br>${escapeHtml(f.position_a_text || '—')}</div>
                    <div><span style="color:#2980b9; font-weight:600;">${escapeHtml(f.position_b_label || 'Position B')}:</span><br>${escapeHtml(f.position_b_text || '—')}</div>
                </div>
            </div>
        `).join('') : `<div style="background:#f8f8ff; border:1px solid #c8d0e8; border-radius:6px; padding:12px 16px; font-size:12px; color:#666;">
            <strong style="color:#555;">No disputed facts were identified in this run.</strong>
            The Disputed Facts Matrix requires at least two documents with conflicting positions on the same fact.
            <br><br><em>Suggestions for next run:</em> Try a goal focused on a specific dispute — e.g.:
            <em>"What is each party's position on the amount owed under the mortgage and what documents support each position?"</em>
        </div>`;

        // Contradictions
        const cl = document.getElementById('ci-contradictions-list');
        const contradictions = d.contradictions || [];
        document.getElementById('ci-contradictions-count').textContent = '(' + contradictions.length + ')';
        cl.innerHTML = contradictions.length ? contradictions.map(c => `
            <div class="ci-finding-item">
                <span class="severity-badge ${c.severity || 'medium'}">${(c.severity || 'medium').toUpperCase()}</span>
                ${escapeHtml(c.description || '')}
                ${c.explanation ? '<p style="font-size:12px; color:#555; margin:4px 0 0 0;">' + escapeHtml(c.explanation) + '</p>' : ''}
                ${c.suggested_action ? '<p style="font-size:11px; color:#2980b9; margin:3px 0 0 0;">→ ' + escapeHtml(c.suggested_action) + '</p>' : ''}
            </div>
        `).join('') : '<p style="color:#888; font-size:13px;">No contradictions detected.</p>';

        // Theories
        const thList = document.getElementById('ci-theories-list');
        const theories = d.theories || [];
        const docMap = d.doc_map || {};
        document.getElementById('ci-theories-count').textContent = '(' + theories.length + ')';

        function ciEvidenceCards(jsonStr, docMap, isCounter) {
            let items = [];
            try { items = JSON.parse(jsonStr || '[]'); } catch(e) { return ''; }
            if (!items || !items.length) return '';
            const color  = isCounter ? '#c0392b' : '#27ae60';
            const icon   = isCounter ? '🔴' : '🟢';
            const label  = isCounter ? 'Counter Evidence' : 'Supporting Evidence';
            const cards  = items.map(ev => {
                const did  = ev.paperless_doc_id;
                const info = did ? (docMap[did] || {}) : {};
                const titleTxt = info.title || (did ? `Document #${did}` : 'Unspecified document');
                const _ciSlug = document.getElementById('project-selector')?.value || 'default';
                const _ciDocUrl = _paperlessDocUrl(_ciSlug, did);
                const titleEl  = did
                    ? (_ciDocUrl
                        ? `<a href="${_ciDocUrl}" target="_blank"
                              style="color:#2980b9; text-decoration:none; font-weight:600;"
                              title="Open in Paperless">📄 ${escapeHtml(titleTxt)}</a> <span style="color:#aaa; font-size:10px;">#${did}</span>`
                        : `<span style="font-weight:600; color:#555;">📄 ${escapeHtml(titleTxt)}</span> <span style="color:#aaa; font-size:10px;">#${did}</span>`)
                    : `<span style="font-weight:600; color:#555;">📄 ${escapeHtml(titleTxt)}</span>`;
                const summaryEl = info.summary
                    ? `<div style="font-size:10px; color:#777; margin:2px 0 4px 0; font-style:italic;">${escapeHtml(info.summary.substring(0,200))}…</div>`
                    : '';
                const excerptEl = ev.excerpt
                    ? `<div style="font-size:11px; background:#f9f9f9; border-left:3px solid ${color}; padding:4px 8px; margin:4px 0; color:#333;">"${escapeHtml(ev.excerpt)}"</div>`
                    : '';
                const reasonKey = isCounter ? 'how_it_undermines' : 'how_it_supports';
                const reasonEl  = ev[reasonKey]
                    ? `<div style="font-size:11px; color:#555; margin:2px 0;">${escapeHtml(ev[reasonKey])}</div>`
                    : '';
                const pageEl = ev.page_number ? `<span style="font-size:10px; color:#aaa;"> — p.${ev.page_number}</span>` : '';
                return `<div style="border:1px solid #e8e8e8; border-radius:5px; padding:8px 10px; margin:5px 0; background:#fff;">
                    ${titleEl}${pageEl}${summaryEl}${excerptEl}${reasonEl}
                </div>`;
            }).join('');
            return `<div style="margin-top:10px;">
                <div style="font-size:11px; font-weight:700; color:${color}; margin-bottom:4px;">${icon} ${label}</div>
                ${cards}
            </div>`;
        }

        thList.innerHTML = theories.length ? theories.map((t, idx) => {
            const statusColor = t.status === 'supported' ? '#27ae60' : t.status === 'refuted' ? '#c0392b' : '#e67e22';
            const confPct = Math.round((t.confidence || 0.5) * 100);
            const suppHtml    = ciEvidenceCards(t.supporting_evidence, docMap, false);
            const counterHtml = ciEvidenceCards(t.counter_evidence,    docMap, true);
            const panelId = `ci-theory-body-${idx}`;
            return `
            <div style="border:1px solid #dde3ec; border-radius:8px; margin-bottom:10px; overflow:hidden;">
                <div onclick="document.getElementById('${panelId}').style.display = document.getElementById('${panelId}').style.display==='none'?'block':'none'"
                     style="display:flex; align-items:flex-start; gap:10px; padding:12px 14px; cursor:pointer; background:#fafbfc;">
                    <div style="flex-shrink:0; min-width:80px; text-align:center;">
                        <div style="font-weight:700; font-size:13px; color:${statusColor};">${(t.status || 'pending').toUpperCase()}</div>
                        <div style="font-size:20px; font-weight:800; color:${statusColor};">${confPct}%</div>
                        <div style="font-size:10px; color:#888; text-transform:uppercase;">${escapeHtml(t.theory_type || '')}</div>
                    </div>
                    <div style="flex:1; font-size:13px; color:#1a2535; line-height:1.5;">${escapeHtml(t.theory_text || '')}</div>
                    <div style="flex-shrink:0; font-size:18px; color:#aaa;">›</div>
                </div>
                <div id="${panelId}" style="display:none; padding:12px 14px; border-top:1px solid #eee; background:#fff;">
                    ${suppHtml}
                    ${counterHtml}
                    ${t.falsification_report ? `<div style="margin-top:10px;">
                        <div style="font-size:11px; font-weight:700; color:#e67e22; margin-bottom:4px;">⚔️ Adversarial Test</div>
                        <div style="font-size:12px; color:#555; line-height:1.5;">${escapeHtml(t.falsification_report)}</div>
                    </div>` : ''}
                    ${t.what_would_change ? `<div style="margin-top:10px; padding:8px 10px; background:#fffbf0; border:1px solid #f0d060; border-radius:5px;">
                        <div style="font-size:11px; font-weight:700; color:#b8860b; margin-bottom:3px;">💡 What Would Strengthen This Theory</div>
                        <div style="font-size:12px; color:#555; line-height:1.5;">${escapeHtml(t.what_would_change)}</div>
                    </div>` : ''}
                </div>
            </div>`;
        }).join('') : '<p style="color:#888; font-size:13px;">No theories generated.</p>';

        // Authorities
        const authList = document.getElementById('ci-authorities-list');
        const auths = d.authorities || [];
        document.getElementById('ci-authorities-count').textContent = '(' + auths.length + ')';
        authList.innerHTML = auths.length ? `<table style="width:100%; border-collapse:collapse; font-size:12px;">
            <tr style="background:#ecf0f1;"><th style="padding:5px 8px; text-align:left;">Citation</th><th style="padding:5px 8px; text-align:left;">Type</th><th style="padding:5px 8px; text-align:left;">Relevance</th></tr>
            ${auths.map(a => `<tr style="border-top:1px solid #f0f0f0;">
                <td style="padding:4px 8px; font-weight:600;">${escapeHtml(a.citation || '')}</td>
                <td style="padding:4px 8px;">${escapeHtml(a.authority_type || '')}</td>
                <td style="padding:4px 8px; color:#666; font-size:11px;">${escapeHtml(a.relevance_note || '')}</td>
            </tr>`).join('')}
        </table>` : `<div style="background:#f8f8ff; border:1px solid #c8d0e8; border-radius:6px; padding:12px 16px; font-size:12px; color:#666;">
            <strong style="color:#555;">No legal authorities were retrieved for this run.</strong>
            Legal authority retrieval requires a Cohere API key and an indexed authority corpus.
            <br><br>
            <strong>To enable:</strong>
            <ol style="margin:6px 0 6px 18px; padding:0; line-height:1.7;">
              <li>Add <code>COHERE_API_KEY</code> to the container environment.</li>
              <li>In the <strong>Setup</strong> sub-tab, use the
                <button onclick="ciSwitchSub('setup'); setTimeout(()=>{document.getElementById('ci-authority-corpus-card')?.scrollIntoView({behavior:'smooth'}); ciLoadAuthorityStatus();},200);"
                        style="background:#2c3e50; color:white; border:none; border-radius:4px; padding:2px 9px; font-size:12px; cursor:pointer; vertical-align:middle;">
                  📚 Authority Corpus
                </button>
                panel to populate the local index.</li>
            </ol>
            <small style="color:#888;">Note: Web research (Phase W) already provides live case law lookups independently of this corpus.</small>
        </div>`;

        // ── Web Research ─────────────────────────────────────────────────────
        const wrRows = d.web_research || [];
        const wrAccordion = document.getElementById('ci-web-research-accordion');
        const wrList  = document.getElementById('ci-webresearch-list');
        const wrCount = document.getElementById('ci-webresearch-count');
        if (wrRows.length > 0 && wrAccordion && wrList) {
            wrAccordion.style.display = '';
            // Parse results from each row
            let totalItems = 0;
            let html = '';
            // Group by search_type
            const byType = {};
            for (const wr of wrRows) {
                const t = wr.search_type || 'general';
                if (!byType[t]) byType[t] = [];
                try {
                    const items = JSON.parse(wr.results_json || '[]');
                    byType[t].push({ query: wr.query, entity_name: wr.entity_name, items });
                    totalItems += Array.isArray(items) ? items.length : (items.court_history ? items.court_history.length + items.news_mentions.length : 0);
                } catch(e) {}
            }
            // Legal authority section
            if (byType.legal_authority) {
                html += `<div style="margin-bottom:12px;">
                    <div style="font-weight:700; color:#1a5276; font-size:12px; margin-bottom:6px;">
                        ⚖️ Case Law &amp; Legal Authorities
                    </div>`;
                for (const row of byType.legal_authority) {
                    for (const item of (row.items || [])) {
                        const cite = escapeHtml(item.citation || item.title || 'Unknown');
                        const court = escapeHtml(item.court || '');
                        const date  = escapeHtml(item.date || '');
                        const excerpt = escapeHtml((item.excerpt || '').substring(0, 200));
                        const url   = item.url || '';
                        const src   = escapeHtml(item.source || 'web');
                        html += `<div style="border:1px solid #d0e0f0; border-radius:5px; padding:8px 10px; margin-bottom:6px; font-size:12px;">
                            <div style="font-weight:600; color:#2c3e50;">
                                ${url ? `<a href="${escapeHtml(url)}" target="_blank" style="color:#2471a3;">${cite}</a>` : cite}
                            </div>
                            <div style="color:#666; margin-top:2px;">${court}${date ? ' · ' + date : ''} · <span style="color:#888; font-size:11px;">${src}</span></div>
                            ${excerpt ? `<div style="color:#555; margin-top:4px; font-style:italic;">"${excerpt}${item.excerpt && item.excerpt.length > 200 ? '…' : ''}"</div>` : ''}
                        </div>`;
                    }
                }
                html += `</div>`;
            }
            // Entity background section
            if (byType.entity_background) {
                html += `<div style="margin-bottom:12px;">
                    <div style="font-weight:700; color:#6c3483; font-size:12px; margin-bottom:6px;">
                        👤 Entity Background Research
                    </div>`;
                for (const row of byType.entity_background) {
                    const r = row.items;  // dict with name, court_history, news_mentions, summary
                    if (!r || typeof r !== 'object') continue;
                    const name = escapeHtml(r.name || row.entity_name || '?');
                    const summary = escapeHtml(r.summary || '');
                    const courtCases = r.court_history || [];
                    const news = r.news_mentions || [];
                    html += `<div style="border:1px solid #e0d0f0; border-radius:5px; padding:8px 10px; margin-bottom:6px; font-size:12px;">
                        <div style="font-weight:600; color:#6c3483; margin-bottom:4px;">${name}</div>`;
                    if (summary) html += `<div style="color:#555; margin-bottom:6px; font-size:11px;">${summary}</div>`;
                    if (courtCases.length > 0) {
                        html += `<div style="font-weight:600; font-size:11px; color:#444; margin-bottom:3px;">Federal Court Records:</div>`;
                        for (const c of courtCases.slice(0, 3)) {
                            const cn = escapeHtml(c.case_name || '');
                            const ct = escapeHtml(c.court || '');
                            const dt = escapeHtml((c.date || '').substring(0, 4));
                            const cu = c.url || '';
                            html += `<div style="margin-left:10px; font-size:11px; color:#666; margin-bottom:2px;">
                                ${cu ? `<a href="${escapeHtml(cu)}" target="_blank" style="color:#2471a3;">${cn}</a>` : cn}
                                ${ct ? `· ${ct}` : ''}${dt ? ` (${dt})` : ''}
                            </div>`;
                        }
                    }
                    if (news.length > 0) {
                        html += `<div style="font-weight:600; font-size:11px; color:#444; margin-top:4px; margin-bottom:3px;">Web Mentions:</div>`;
                        for (const n of news.slice(0, 2)) {
                            const nt = escapeHtml(n.title || '');
                            const nu = n.url || '';
                            const ns = escapeHtml((n.excerpt || '').substring(0, 120));
                            html += `<div style="margin-left:10px; font-size:11px; color:#666; margin-bottom:2px;">
                                ${nu ? `<a href="${escapeHtml(nu)}" target="_blank" style="color:#2471a3;">${nt}</a>` : nt}
                                ${ns ? `<span style="color:#888;"> — ${ns}</span>` : ''}
                            </div>`;
                        }
                    }
                    html += `</div>`;
                }
                html += `</div>`;
            }
            // General web search section
            if (byType.general) {
                html += `<div style="margin-bottom:12px;">
                    <div style="font-weight:700; color:#1e8449; font-size:12px; margin-bottom:6px;">
                        🌐 Web Search — Recent Developments
                    </div>`;
                for (const row of byType.general) {
                    for (const item of (row.items || [])) {
                        const t = escapeHtml(item.title || '');
                        const s = escapeHtml((item.excerpt || '').substring(0, 150));
                        const u = item.url || '';
                        html += `<div style="border:1px solid #d5f5e3; border-radius:5px; padding:7px 10px; margin-bottom:5px; font-size:12px;">
                            <div>${u ? `<a href="${escapeHtml(u)}" target="_blank" style="color:#1e8449; font-weight:600;">${t}</a>` : `<strong>${t}</strong>`}</div>
                            ${s ? `<div style="color:#666; margin-top:2px; font-size:11px;">${s}</div>` : ''}
                        </div>`;
                    }
                }
                html += `</div>`;
            }
            wrList.innerHTML = html || '<div style="color:#888; font-size:12px;">No web research results to display.</div>';
            if (wrCount) wrCount.textContent = `(${totalItems} results)`;
        } else if (wrAccordion) {
            wrAccordion.style.display = 'none';
        }
    }

