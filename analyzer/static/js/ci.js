// ── Case Intelligence JS ────────────────────────────────────────────────

    let ciCurrentRunId = null;
    let ciPollTimer = null;
    let ciSelectedRunId = null;
    let ciFindingsPollTimer = null;

    function ciSwitchSub(sub) {
        document.querySelectorAll('.ci-sub-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.ci-sub-content').forEach(c => c.classList.remove('active'));
        const btn = document.querySelector(`.ci-sub-btn[onclick="ciSwitchSub('${sub}')"]`);
        if (btn) btn.classList.add('active');
        const el = document.getElementById(`ci-sub-${sub}`);
        if (el) el.classList.add('active');

        if (sub === 'findings') {
            ciRefreshRuns();
        }
        if (sub === 'setup') {
            ciLoadAuthorityStatus();
        }
        // Update help panel to match the newly active CI sub-tab
        if (typeof _refreshHelpPanel === 'function') _refreshHelpPanel();
    }

    function ciToggleSection(id) {
        const el = document.getElementById(id);
        if (el) el.classList.toggle('open');
    }

    // ── 5-Tier Card Selector ──────────────────────────────────────────
    function ciSelectTier(tier) {
        document.getElementById('ci-max-tier').value = tier;
        document.querySelectorAll('.ci-tier-card').forEach(card => {
            card.classList.toggle('active', parseInt(card.dataset.tier) === tier);
        });
        ciUpdateCostEstimate();
    }

    // Initialize tier 3 as selected on load
    document.addEventListener('DOMContentLoaded', function() {
        ciSelectTier(3);
    });

    async function ciUpdateCostEstimate() {
        const tier = parseInt(document.getElementById('ci-max-tier').value) || 3;
        // Use paperless_total from stats (full project doc count), fall back to
        // docs_total from a previously selected run, then to total_analyzed
        let docs = 10;
        try {
            const sr = await apiFetch(apiUrl('/api/status'));
            const sd = await sr.json();
            // total_analyzed = Chroma count = docs CI actually runs against
            if (sd.stats && sd.stats.total_analyzed > 0) {
                docs = sd.stats.total_analyzed;
            } else if (sd.paperless_total > 0) {
                docs = sd.paperless_total;
            }
        } catch(e) {}
        // If we have a selected run with a known doc count, prefer that (it may differ from project total)
        try {
            if (ciSelectedRunId) {
                const r = await apiFetch(apiUrl(`/api/ci/runs/${ciSelectedRunId}/status`));
                const d = await r.json();
                if (d.run && d.run.docs_total) docs = d.run.docs_total;
            }
        } catch(e) {}
        try {
            const r = await apiFetch(apiUrl(`/api/ci/cost-estimate?docs=${docs}&tier=${tier}`));
            const d = await r.json();
            const el = document.getElementById('ci-cost-estimate');
            if (el && d.estimated_usd !== undefined) {
                el.textContent = `~$${d.estimated_usd.toFixed(2)} (${docs} docs)`;
            }
        } catch(e) {
            const el = document.getElementById('ci-cost-estimate');
            if (el) el.textContent = '—';
        }
    }

    // Copy RFP list to clipboard
    function ciCopyRFPs() {
        const rfpList = document.getElementById('ci-discovery-rfp-list');
        if (!rfpList) return;
        const items = rfpList.querySelectorAll('.ci-rfp-item');
        const text = Array.from(items).map(el => el.querySelector('.rfp-text')?.textContent || el.textContent).join('\n\n');
        navigator.clipboard.writeText(text).then(() => showToast('RFP list copied to clipboard', 'success', 2500));
    }

    // Load jurisdiction profiles on tab open
    async function ciLoadJurisdictions() {
        try {
            const r = await apiFetch(apiUrl('/api/ci/jurisdictions'));
            const d = await r.json();
            const sel = document.getElementById('ci-jurisdiction-select');
            sel.innerHTML = '<option value="">— Select template —</option>';
            (d.jurisdictions || []).forEach(j => {
                const o = document.createElement('option');
                o.value = j.jurisdiction_id;
                o.textContent = j.display_name;
                o.dataset.court = j.court;
                o.dataset.framework = j.baseline_framework;
                sel.appendChild(o);
            });
        } catch(e) {
            console.warn('CI jurisdictions load failed:', e);
        }
    }

    function ciLoadJurisdiction() {
        const sel = document.getElementById('ci-jurisdiction-select');
        const opt = sel.options[sel.selectedIndex];
        if (!opt || !opt.value) {
            document.getElementById('ci-jurisdiction-details').style.display = 'none';
            return;
        }
        document.getElementById('ci-jd-court').value = opt.dataset.court || '';
        document.getElementById('ci-jurisdiction-details').style.display = 'flex';
    }

    async function ciDetectJurisdiction() {
        const statusEl = document.getElementById('ci-jd-detect-status');
        if (!statusEl) return;
        statusEl.textContent = '🔍 Detecting jurisdiction from documents…';
        statusEl.style.color = '#888';
        try {
            const r = await apiFetch(apiUrl('/api/ci/detect-jurisdiction'), { method: 'POST' });
            const d = await r.json();
            if (!d.detected) {
                statusEl.textContent = d.reason || 'Could not detect jurisdiction automatically.';
                return;
            }
            // Pre-select the detected profile in the dropdown
            const sel = document.getElementById('ci-jurisdiction-select');
            let matched = false;
            for (let i = 0; i < sel.options.length; i++) {
                if (sel.options[i].value === d.jurisdiction_id) {
                    sel.selectedIndex = i;
                    matched = true;
                    break;
                }
            }
            if (matched) {
                ciLoadJurisdiction();
                // Populate county if detected
                if (d.county) {
                    const countyEl = document.getElementById('ci-jd-county');
                    if (countyEl && !countyEl.value) countyEl.value = d.county;
                }
                const pct = Math.round((d.confidence || 0.5) * 100);
                const reason = d.reason ? ` — ${d.reason}` : '';
                statusEl.innerHTML = `✓ Auto-detected: <strong>${escapeHtml(d.display_name)}</strong> (${pct}% confidence${reason})`;
                statusEl.style.color = '#27ae60';
            } else {
                statusEl.textContent = `Detected: ${d.display_name} (not in dropdown — select manually)`;
                statusEl.style.color = '#e67e22';
            }
        } catch(e) {
            statusEl.textContent = 'Auto-detection unavailable.';
            console.warn('CI detect-jurisdiction failed:', e);
        }
    }

    function ciGetConfig() {
        const jdSelect = document.getElementById('ci-jurisdiction-select');
        const jdOpt = jdSelect.options[jdSelect.selectedIndex];
        const jurisdiction = {
            jurisdiction_id: jdSelect.value,
            display_name: jdOpt ? jdOpt.textContent.trim() : 'Custom',
            court: document.getElementById('ci-jd-court').value,
            county: document.getElementById('ci-jd-county').value,
            judge_part: document.getElementById('ci-jd-part').value,
            baseline_framework: jdOpt ? (jdOpt.dataset.framework || '') : '',
            authority_jurisdictions: ['NYS', '2nd Circuit', 'US'],
        };
        // Build web_research_config from WEB RESEARCH section
        const wrEnabled = document.getElementById('ci-wr-enabled')?.checked || false;
        const _wrVal = (id) => (document.getElementById(id)?.value || '').trim();
        const _wrCk  = (id) => document.getElementById(id)?.checked || false;
        const _wrKey = (cbId, keyId, cfgKey) => {
            const k = _wrVal(keyId);
            return (_wrCk(cbId) && k) ? {[cfgKey]: k} : {};
        };
        const web_research_config = {
            enabled:         wrEnabled,
            // Case law (free)
            courtlistener:   _wrCk('ci-wr-courtlistener'),
            caselaw_api:     _wrCk('ci-wr-caselaw'),
            legal_search:    true,
            // Web search (free)
            general_search:  _wrCk('ci-wr-general'),
            // Public records (free toggles)
            entity_research: _wrCk('ci-wr-entity'),
            bop_search:      _wrCk('ci-wr-bop'),
            ofac_search:     _wrCk('ci-wr-ofac'),
            sec_edgar:       _wrCk('ci-wr-sec'),
            // News (free)
            gdelt_news:      _wrCk('ci-wr-gdelt'),
            // Paid keys — only included when checkbox is checked and key is non-empty
            ..._wrKey('ci-wr-docket-cb',       'ci-wr-docket-user',         'docket_alarm_user'),
            ..._wrKey('ci-wr-docket-cb',       'ci-wr-docket-pass',         'docket_alarm_pass'),
            ..._wrKey('ci-wr-unicourt-cb',     'ci-wr-unicourt-id',         'unicourt_id'),
            ..._wrKey('ci-wr-unicourt-cb',     'ci-wr-unicourt-secret',     'unicourt_secret'),
            ..._wrKey('ci-wr-brave-cb',        'ci-wr-brave-key',           'brave_key'),
            ..._wrKey('ci-wr-gcse-cb',         'ci-wr-gcse-key',            'google_cse_key'),
            ..._wrKey('ci-wr-gcse-cb',         'ci-wr-gcse-cx',             'google_cse_cx'),
            ..._wrKey('ci-wr-exa-cb',          'ci-wr-exa-key',             'exa_key'),
            ..._wrKey('ci-wr-perplexity-cb',   'ci-wr-perplexity-key',      'perplexity_key'),
            ..._wrKey('ci-wr-tavily-cb',       'ci-wr-tavily-key',          'tavily_key'),
            ..._wrKey('ci-wr-serper-cb',       'ci-wr-serper-key',          'serper_key'),
            ..._wrKey('ci-wr-fec-cb',          'ci-wr-fec-key',             'fec_key'),
            ..._wrKey('ci-wr-opensanctions-cb','ci-wr-opensanctions-key',   'opensanctions_key'),
            ..._wrKey('ci-wr-opencorp-cb',     'ci-wr-opencorp-key',        'opencorporates_key'),
            ..._wrKey('ci-wr-clear-cb',        'ci-wr-clear-key',           'clear_key'),
            ..._wrKey('ci-wr-newsapi-cb',      'ci-wr-newsapi-key',         'newsapi_key'),
            ..._wrKey('ci-wr-lexis-cb',        'ci-wr-lexis-key',           'lexisnexis_key'),
            ..._wrKey('ci-wr-vlex-cb',         'ci-wr-vlex-key',            'vlex_key'),
            ..._wrKey('ci-wr-westlaw-cb',      'ci-wr-westlaw-key',         'westlaw_key'),
        };

        // Budget overage policy: -1=unlimited, 20=20% overage, 0=hard block
        const _allowUnlimited = document.getElementById('ci-allow-unlimited')?.checked;
        const _allowOverage   = document.getElementById('ci-allow-overage')?.checked;
        const allow_overage_pct = _allowUnlimited ? -1 : (_allowOverage ? 20 : 0);

        return {
            role: document.getElementById('ci-role').value,
            goal_text: document.getElementById('ci-goal').value.trim(),
            budget_per_run_usd: parseFloat(document.getElementById('ci-budget').value) || 10.0,
            allow_overage_pct,
            max_tier: parseInt(document.getElementById('ci-max-tier').value) || 3,
            jurisdiction,
            notification_email: (document.getElementById('ci-notification-email')?.value || '').trim(),
            notify_on_complete: document.getElementById('ci-notify-complete')?.checked ? 1 : 0,
            notify_on_budget:   document.getElementById('ci-notify-budget')?.checked   ? 1 : 0,
            web_research_config,
        };
    }

    function ciOveragePolicyChanged() {
        // The two checkboxes are mutually exclusive
        const unlimited = document.getElementById('ci-allow-unlimited');
        const overage   = document.getElementById('ci-allow-overage');
        if (!unlimited || !overage) return;
        if (unlimited.checked && overage.checked) {
            // Whichever was just clicked takes precedence — disable the other
            if (document.activeElement === unlimited) overage.checked = false;
            else unlimited.checked = false;
        }
    }

    function ciToggleWebResearch(hdr) {
        const body = document.getElementById('ci-wr-body');
        const lbl  = document.getElementById('ci-wr-toggle');
        if (!body) return;
        const hidden = body.style.display === 'none';
        body.style.display = hidden ? '' : 'none';
        if (lbl) lbl.textContent = hidden ? '▲ hide' : '▼ show';
    }

    function ciUpdateWrRoleLabel() {
        const enabled = document.getElementById('ci-wr-enabled')?.checked;
        const allOpts = document.getElementById('ci-wr-all-opts');
        const note    = document.getElementById('ci-wr-note');
        if (allOpts) {
            allOpts.style.opacity      = enabled ? '1' : '0.5';
            allOpts.style.pointerEvents = enabled ? '' : 'none';
        }
        if (note) note.style.display = enabled ? '' : 'none';
        const role = document.getElementById('ci-role')?.value || 'neutral';
        const lbl  = document.getElementById('ci-wr-role-label');
        if (lbl) lbl.textContent = role || 'the selected role';
    }

    function ciToggleAuthorityCorpus(hdr) {
        const body = document.getElementById('ci-ac-body');
        const lbl  = document.getElementById('ci-ac-toggle');
        if (!body) return;
        const hidden = body.style.display === 'none';
        body.style.display = hidden ? '' : 'none';
        if (lbl) lbl.textContent = hidden ? '▲ hide' : '▼ show';
    }

    async function ciLoadAuthorityStatus() {
        const el = document.getElementById('ci-ac-status-text');
        if (!el) return;
        el.textContent = '⏳ Loading…';
        try {
            const r = await apiFetch(apiUrl('/api/ci/authority/status'));
            const d = await r.json();
            const dbCount   = d.db?.total_authorities ?? 0;
            const chromaCount = d.chroma?.count ?? 0;
            const enabled   = d.chroma?.enabled ?? false;
            if (!enabled) {
                el.innerHTML = '❌ <strong>Cohere not configured</strong> — add <code>COHERE_API_KEY</code> to the container environment to enable authority retrieval.';
                el.style.color = '#c0392b';
            } else if (chromaCount === 0 && dbCount === 0) {
                el.innerHTML = '⚠️ Corpus is <strong>empty</strong> — click "Populate / Update Corpus" to ingest authorities.';
                el.style.color = '#e67e22';
            } else {
                el.innerHTML = `✅ Corpus ready — <strong>${chromaCount.toLocaleString()}</strong> vectors embedded` +
                    (dbCount > chromaCount ? ` (${dbCount - chromaCount} pending embed)` : '') + '.';
                el.style.color = '#27ae60';
            }
        } catch (e) {
            el.textContent = '⚠️ Could not load corpus status — ' + e.message;
            el.style.color = '#c0392b';
        }
    }

    async function ciIngestAuthority() {
        const btn = document.getElementById('ci-ac-ingest-btn');
        const statusEl = document.getElementById('ci-ac-ingest-status');
        const sources = [];
        if (document.getElementById('ci-ac-nysenate')?.checked)     sources.push('nysenate');
        if (document.getElementById('ci-ac-ecfr')?.checked)         sources.push('ecfr');
        if (document.getElementById('ci-ac-courtlistener')?.checked) sources.push('courtlistener');
        if (!sources.length) { alert('Select at least one source to ingest.'); return; }

        if (btn) { btn.disabled = true; btn.textContent = '⏳ Ingesting…'; }
        if (statusEl) {
            statusEl.style.display = '';
            statusEl.style.background = '#eaf4fb';
            statusEl.style.borderColor = '#a9d4df';
            statusEl.style.color = '#1a6e8e';
            statusEl.innerHTML = `⏳ Ingestion started for sources: <strong>${sources.join(', ')}</strong>.<br>
                This runs in the background and may take several minutes.
                Reload corpus status after a few minutes to confirm completion.`;
        }
        try {
            await apiFetch(apiUrl('/api/ci/authority/ingest'), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sources}),
            });
        } catch (e) {
            if (statusEl) {
                statusEl.style.background = '#fdf0ee';
                statusEl.style.borderColor = '#f5c6c0';
                statusEl.style.color = '#c0392b';
                statusEl.innerHTML = '❌ Failed to start ingestion: ' + escapeHtml(e.message);
            }
        }
        if (btn) { btn.disabled = false; btn.textContent = '⚡ Populate / Update Corpus'; }
    }

    async function ciSaveDraft() {
        if (!window.APP_CONFIG.isAdvanced) return;
        try {
            const config = ciGetConfig();
            const r = await apiFetch(apiUrl('/api/ci/runs'), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config),
            });
            const d = await r.json();
            if (d.run_id) {
                ciCurrentRunId = d.run_id;
                showToast('Config saved (draft run: ' + d.run_id.substr(0,8) + ')', 'success');
                // Load questions
                await ciLoadQuestions(d.run_id);
            } else {
                showToast(d.error || 'Save failed', 'error');
            }
        } catch(e) {
            showToast('Error: ' + e.message, 'error');
        }
    }

    async function ciStartRun() {
        if (!window.APP_CONFIG.isAdvanced) return;
        // Save draft first if no run
        if (!ciCurrentRunId) {
            await ciSaveDraft();
        }
        if (!ciCurrentRunId) return;

        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + ciCurrentRunId + '/start'), {
                method: 'POST',
            });
            const d = await r.json();
            if (d.status === 'started') {
                document.getElementById('ci-status-bar').classList.add('visible');
                document.getElementById('ci-cancel-btn').style.display = 'inline-block';
                document.getElementById('ci-run-btn').disabled = true;
                showToast('Case Intelligence run started', 'success');
                ciStartPolling(ciCurrentRunId);
            } else {
                showToast(d.error || 'Failed to start', 'error');
            }
        } catch(e) {
            showToast('Error: ' + e.message, 'error');
        }
    }

    async function ciCancelRun() {
        if (!ciCurrentRunId) return;
        try {
            await apiFetch(apiUrl('/api/ci/runs/' + ciCurrentRunId + '/cancel'), { method: 'POST' });
            showToast('Cancellation signal sent', 'info');
        } catch(e) {
            showToast('Error: ' + e.message, 'error');
        }
    }

    function ciStartPolling(runId) {
        if (ciPollTimer) clearInterval(ciPollTimer);
        ciPollTimer = setInterval(() => ciPollStatus(runId), 3000);
        ciPollStatus(runId);
    }

    async function ciPollStatus(runId) {
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + runId + '/status'));
            const d = await r.json();
            ciUpdateStatusBar(d);
            if (['completed', 'failed', 'cancelled', 'budget_blocked'].includes(d.status)) {
                clearInterval(ciPollTimer);
                document.getElementById('ci-cancel-btn').style.display = 'none';
                document.getElementById('ci-run-btn').disabled = false;
                if (d.status === 'completed') {
                    showToast('CI run completed!', 'success');
                } else if (d.status === 'budget_blocked') {
                    showToast('Budget ceiling reached: ' + (d.budget_blocked_note || ''), 'warning');
                } else if (d.status === 'failed') {
                    showToast('Run failed: ' + (d.error_message || 'unknown error'), 'error');
                }
            }
        } catch(e) {
            console.warn('CI poll failed:', e);
        }
    }

    function ciUpdateStatusBar(d) {
        const pct = d.progress_pct || 0;
        const fill = document.getElementById('ci-progress-fill');
        if (fill) {
            fill.style.width = pct + '%';
            fill.className = 'ci-progress-fill' + (d.budget_blocked ? ' blocked' : '');
        }
        const stage = document.getElementById('ci-stage-label');
        if (stage) stage.textContent = d.current_stage || d.status || '—';
        const cost = document.getElementById('ci-cost-label');
        if (cost) cost.textContent = '$' + (d.cost_so_far_usd || 0).toFixed(2) + ' / $' + (d.budget_per_run_usd || 10).toFixed(2);
        const docs = document.getElementById('ci-docs-label');
        if (docs) docs.textContent = (d.docs_processed || 0) + ' / ' + (d.docs_total || 0) + ' docs';
        const st = document.getElementById('ci-status-label');
        if (st) st.textContent = d.status || '—';

        // Workers + managers
        const wkLabel = document.getElementById('ci-workers-label');
        if (wkLabel) {
            const mgr = d.active_managers || 0, wkr = d.active_workers || 0;
            if (mgr > 0 || wkr > 0) {
                document.getElementById('ci-workers-val').textContent = mgr + 'mgr · ' + wkr + 'wkr';
                wkLabel.style.display = 'inline';
            } else { wkLabel.style.display = 'none'; }
        }
        // Tokens
        const tokLabel = document.getElementById('ci-tokens-label');
        if (tokLabel) {
            const total = (d.tokens_in || 0) + (d.tokens_out || 0);
            if (total > 0) {
                document.getElementById('ci-tokens-val').textContent =
                    total >= 1000 ? (total / 1000).toFixed(1) + 'k' : total;
                tokLabel.style.display = 'inline';
            } else { tokLabel.style.display = 'none'; }
        }
        // Elapsed + ETA
        const elapsed = d.elapsed_seconds || 0;
        const elLabel = document.getElementById('ci-elapsed-label');
        if (elLabel && elapsed > 0) {
            document.getElementById('ci-elapsed-val').textContent =
                elapsed < 60 ? elapsed + 's' : Math.floor(elapsed / 60) + 'm ' + Math.floor(elapsed % 60) + 's';
            elLabel.style.display = 'inline';
        } else if (elLabel) { elLabel.style.display = 'none'; }
        const etaLabel = document.getElementById('ci-eta-label');
        if (etaLabel && elapsed > 0 && pct >= 10 && pct < 100) {
            const remaining = Math.max(0, (elapsed / (pct / 100)) - elapsed);
            document.getElementById('ci-eta-val').textContent =
                remaining < 60 ? Math.round(remaining) + 's' : Math.round(remaining / 60) + 'm';
            etaLabel.style.display = 'inline';
        } else if (etaLabel) { etaLabel.style.display = 'none'; }
    }

    async function ciLoadQuestions(runId) {
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + runId + '/questions'));
            const d = await r.json();
            const qs = d.questions || [];
            const section = document.getElementById('ci-questions-section');
            const list = document.getElementById('ci-questions-list');
            const count = document.getElementById('ci-q-count');
            if (qs.length === 0) {
                section.style.display = 'none';
                return;
            }
            section.style.display = 'block';
            count.textContent = '(' + qs.filter(q => q.is_required).length + ' required)';
            list.innerHTML = qs.map(q => `
                <div class="ci-q-item ${q.is_required ? 'required' : ''}">
                    <label>${q.is_required ? '★ ' : ''}${escapeHtml(q.question)}</label>
                    <input type="text" id="ci-q-${q.id}" placeholder="Your answer..." value="${escapeHtml(q.answer || '')}">
                </div>
            `).join('');
        } catch(e) {
            console.warn('CI load questions failed:', e);
        }
    }

    async function ciSaveAnswers() {
        if (!ciCurrentRunId) return;
        const qs = document.querySelectorAll('[id^="ci-q-"]');
        const answers = {};
        qs.forEach(inp => {
            const qid = inp.id.replace('ci-q-', '');
            if (inp.value.trim()) answers[qid] = inp.value.trim();
        });
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + ciCurrentRunId + '/answers'), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ answers }),
            });
            const d = await r.json();
            if (d.status === 'answers_saved') showToast('Answers saved', 'success');
        } catch(e) {
            showToast('Error saving answers: ' + e.message, 'error');
        }
    }

    async function ciProceedWithAssumptions() {
        if (!ciCurrentRunId) return;
        try {
            await apiFetch(apiUrl('/api/ci/runs/' + ciCurrentRunId + '/answers'), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ proceed_with_assumptions: true }),
            });
            showToast('Will proceed with assumptions', 'info');
            document.getElementById('ci-questions-section').style.display = 'none';
        } catch(e) {
            showToast('Error: ' + e.message, 'error');
        }
    }

    function ciStartNewRun() {
        ciCurrentRunId = null;
        document.getElementById('ci-status-bar').classList.remove('visible');
        document.getElementById('ci-questions-section').style.display = 'none';
    }

    // ── Findings sub-tab ─────────────────────────────────────────────
    async function ciRefreshRuns() {
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs'));
            const d = await r.json();
            const runs = d.runs || [];
            const sel = document.getElementById('ci-run-selector');
            const prevVal = sel.value;
            sel.innerHTML = '<option value="">— Select a run —</option>';
            runs.forEach(run => {
                const o = document.createElement('option');
                o.value = run.id;
                const date = run.created_at ? run.created_at.substring(0,16).replace('T',' ') : '';
                const ownerNote = run._shared ? ` · shared by ${run.owner_name}` :
                                  (run.owner_name && run.user_id !== window.APP_CONFIG.currentUserId ? ` · ${run.owner_name}` : '');
                const goalSnip = run.goal_text ? ` — ${run.goal_text.substring(0, 40)}${run.goal_text.length > 40 ? '…' : ''}` : '';
                const statusIcon = run.status === 'completed' ? '✓' : run.status === 'failed' ? '✗' : run.status === 'interrupted' ? '⚡' : run.status === 'running' ? '⟳' : '○';
                o.textContent = `${statusIcon} ${date}${goalSnip} (${run.status})${ownerNote}`;
                o.dataset.status = run.status;
                o.dataset.ownerId = run.user_id;
                sel.appendChild(o);
            });
            if (prevVal && runs.find(r => r.id === prevVal)) sel.value = prevVal;
        } catch(e) {
            console.warn('CI refresh runs failed:', e);
        }
    }

    async function ciLoadFindings() {
        const runId = document.getElementById('ci-run-selector').value;
        const deleteBtn = document.getElementById('ci-delete-run-btn');
        const shareBtn = document.getElementById('ci-share-run-btn');
        if (!runId) {
            document.getElementById('ci-no-findings').style.display = 'block';
            document.getElementById('ci-key-findings-section').style.display = 'none';
            document.getElementById('ci-findings-status').style.display = 'none';
            document.getElementById('ci-run-meta-header').style.display = 'none';
            document.getElementById('ci-report-builder') && (document.getElementById('ci-report-builder').style.display = 'none');
            if (deleteBtn) deleteBtn.style.display = 'none';
            if (shareBtn) shareBtn.style.display = 'none';
            return;
        }
        ciSelectedRunId = runId;
        if (deleteBtn) deleteBtn.style.display = 'inline-block';
        if (shareBtn) shareBtn.style.display = 'inline-block';
        if (ciFindingsPollTimer) clearInterval(ciFindingsPollTimer);

        // First check status
        try {
            const sr = await apiFetch(apiUrl('/api/ci/runs/' + runId + '/status'));
            const sd = await sr.json();
            if (['running', 'queued'].includes(sd.status)) {
                // Clear any stale findings from a previously-viewed run
                document.getElementById('ci-key-findings-section').style.display = 'none';
                document.getElementById('ci-run-meta-header').style.display = 'none';
                document.getElementById('ci-no-findings').style.display = 'none';
                // Hide specialist accordions too
                ['ci-forensic-accordion','ci-discovery-accordion','ci-witnesses-accordion','ci-warroom-accordion',
                 'ci-deepforensics-accordion','ci-trialstrategy-accordion','ci-multimodel-accordion','ci-settlement-accordion'].forEach(id => {
                    const el = document.getElementById(id); if (el) el.style.display = 'none';
                });
                document.getElementById('ci-findings-status').style.display = 'block';
                // Seed elapsed counter with server value immediately
                _ciElapsedBase = sd.elapsed_seconds || 0;
                _ciElapsedTick = Date.now();
                ciStartElapsedTicker();
                ciFindingsPollTimer = setInterval(() => ciFindingsPoll(runId), 3000);
                ciFindingsPoll(runId);
                return;
            }
        } catch(e) {}

        // Load full findings
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + runId + '/findings'));
            const d = await r.json();
            ciRenderFindings(d, runId);
            ciLoadSpecialistFindings(runId);
        } catch(e) {
            showToast('Error loading findings: ' + e.message, 'error');
        }
    }

    async function ciRerunInterrupted(runId) {
        try {
            const r = await apiFetch(apiUrl(`/api/ci/runs/${runId}/rerun`), {method:'POST'});
            const d = await r.json();
            if (d.error) { showToast('Re-run failed: ' + d.error, 'error'); return; }
            showToast('Re-run started! Switching to new run…', 'success');
            await ciRefreshRuns();
            const sel = document.getElementById('ci-run-selector');
            if (sel && d.run_id) { sel.value = d.run_id; await ciLoadFindings(); }
        } catch(e) {
            showToast('Re-run error: ' + e.message, 'error');
        }
    }

    // ── Elapsed timer helpers ──────────────────────────────────────────
    let _ciElapsedBase = 0;   // seconds at last server sync
    let _ciElapsedTick = 0;   // Date.now() at last server sync
    let _ciElapsedTimer = null;

    function ciStartElapsedTicker() {
        if (_ciElapsedTimer) clearInterval(_ciElapsedTimer);
        _ciElapsedTimer = setInterval(() => {
            const elapsed = _ciElapsedBase + Math.floor((Date.now() - _ciElapsedTick) / 1000);
            const h = Math.floor(elapsed / 3600);
            const m = Math.floor((elapsed % 3600) / 60);
            const s = elapsed % 60;
            const txt = h > 0
                ? `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`
                : `${m}:${String(s).padStart(2,'0')}`;
            const el = document.getElementById('ci-f-elapsed');
            if (el) el.textContent = txt;
        }, 1000);
    }

    function ciStopElapsedTicker() {
        if (_ciElapsedTimer) { clearInterval(_ciElapsedTimer); _ciElapsedTimer = null; }
    }

    async function ciFindingsPoll(runId) {
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + runId + '/status'));
            const d = await r.json();
            const stage = document.getElementById('ci-f-stage');
            if (stage) stage.textContent = d.current_stage || d.status;
            const cost = document.getElementById('ci-f-cost');
            if (cost) cost.textContent = '$' + (d.cost_so_far_usd || 0).toFixed(2) + ' / $' + (d.budget_per_run_usd || 10).toFixed(2);
            const progress = document.getElementById('ci-f-progress');
            if (progress) progress.style.width = (d.progress_pct || 0) + '%';

            // Sync elapsed timer base from server
            if (d.elapsed_seconds != null) {
                _ciElapsedBase = d.elapsed_seconds;
                _ciElapsedTick = Date.now();
            }

            // Docs display: docs_processed counts per-domain tasks (can exceed docs_total).
            // Show "N / M docs · X tasks" so the numbers make sense.
            const docs = document.getElementById('ci-f-docs');
            if (docs) {
                const total = d.docs_total || 0;
                const tasks = d.docs_processed || 0;
                if (total > 0) {
                    const uniqueDone = Math.min(tasks, total);
                    docs.textContent = `${uniqueDone} / ${total} docs · ${tasks} analysis tasks`;
                } else {
                    docs.textContent = `${tasks} analysis tasks`;
                }
            }

            if (['completed', 'failed', 'cancelled', 'budget_blocked', 'interrupted'].includes(d.status)) {
                clearInterval(ciFindingsPollTimer);
                ciStopElapsedTicker();
                document.getElementById('ci-findings-status').style.display = 'none';
                if (d.status === 'completed') {
                    ciLoadFindings();
                } else if (d.status === 'interrupted') {
                    ciLoadFindings();  // show partial results + rerun banner
                }
            }
        } catch(e) {}
    }

    async function ciCancelRunFromFindings() {
        if (!ciSelectedRunId) return;
        try {
            await apiFetch(apiUrl('/api/ci/runs/' + ciSelectedRunId + '/cancel'), { method: 'POST' });
            showToast('Cancellation sent', 'info');
        } catch(e) {}
    }

    async function ciDeleteSelectedRun() {
        if (!ciSelectedRunId) return;
        const selector = document.getElementById('ci-run-selector');
        const runLabel = selector ? selector.options[selector.selectedIndex]?.text : ciSelectedRunId;
        if (!confirm(`Permanently delete this run and all its findings?\n\n"${runLabel}"\n\nThis cannot be undone.`)) return;
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + ciSelectedRunId), { method: 'DELETE' });
            const d = await r.json();
            if (!r.ok) { showToast(d.error || 'Delete failed', 'error'); return; }
            showToast('Run deleted', 'success');
            // Remove the option from the selector and reset
            if (selector) {
                const opt = selector.querySelector(`option[value="${ciSelectedRunId}"]`);
                if (opt) opt.remove();
            }
            ciSelectedRunId = null;
            const deleteBtn = document.getElementById('ci-delete-run-btn');
            if (deleteBtn) deleteBtn.style.display = 'none';
            const shareBtn2 = document.getElementById('ci-share-run-btn');
            if (shareBtn2) shareBtn2.style.display = 'none';
            // Reset findings panel
            document.getElementById('ci-no-findings').style.display = 'block';
            document.getElementById('ci-key-findings-section').style.display = 'none';
            document.getElementById('ci-run-meta-header').style.display = 'none';
            document.getElementById('ci-report-builder') && (document.getElementById('ci-report-builder').style.display = 'none');
            if (selector) selector.value = '';
        } catch(e) {
            showToast('Delete failed: ' + e.message, 'error');
        }
    }

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

    // ── Specialist Findings: Forensic, Discovery, Witnesses, War Room ────────
    async function ciLoadSpecialistFindings(runId) {
        if (!runId) return;
        const [forensicRes, discoveryRes, witnessRes, warRoomRes,
               deepForensicsRes, trialStrategyRes, multiModelRes,
               settlementRes] = await Promise.allSettled([
            apiFetch(apiUrl('/api/ci/runs/' + runId + '/forensic-report')).then(r => r.json()),
            apiFetch(apiUrl('/api/ci/runs/' + runId + '/discovery-gaps')).then(r => r.json()),
            apiFetch(apiUrl('/api/ci/runs/' + runId + '/witness-cards')).then(r => r.json()),
            apiFetch(apiUrl('/api/ci/runs/' + runId + '/war-room')).then(r => r.json()),
            apiFetch(apiUrl('/api/ci/runs/' + runId + '/deep-forensics')).then(r => r.json()),
            apiFetch(apiUrl('/api/ci/runs/' + runId + '/trial-strategy')).then(r => r.json()),
            apiFetch(apiUrl('/api/ci/runs/' + runId + '/multi-model')).then(r => r.json()),
            apiFetch(apiUrl('/api/ci/runs/' + runId + '/settlement-valuation')).then(r => r.json()),
        ]);
        // Each API returns {present, data: <payload>} — unwrap before passing to renderers
        ciRenderForensicTab(forensicRes.status      === 'fulfilled' ? (forensicRes.value?.data      || null) : null);
        ciRenderDiscoveryTab(discoveryRes.status    === 'fulfilled' ? (discoveryRes.value?.data     || null) : null);
        ciRenderWitnessesTab(witnessRes.status      === 'fulfilled' ? (witnessRes.value?.data       || null) : null);
        ciRenderWarRoomTab(warRoomRes.status        === 'fulfilled' ? (warRoomRes.value?.data       || null) : null);
        ciRenderDeepForensicsTab(deepForensicsRes.status  === 'fulfilled' ? (deepForensicsRes.value?.data  || null) : null);
        ciRenderTrialStrategyTab(trialStrategyRes.status  === 'fulfilled' ? (trialStrategyRes.value?.data  || null) : null);
        ciRenderMultiModelTab(multiModelRes.status        === 'fulfilled' ? (multiModelRes.value?.data     || null) : null);
        ciRenderSettlementTab(settlementRes.status        === 'fulfilled' ? (settlementRes.value?.data     || null) : null);
    }

    function ciRenderForensicTab(data) {
        const acc  = document.getElementById('ci-forensic-accordion');
        const body = document.getElementById('ci-forensic-body');
        if (!acc || !body) return;
        if (!data || data.error || (!data.flagged_transactions?.length && !data.summary)) {
            acc.style.display = 'none'; return;
        }
        acc.style.display = '';
        const flagged = data.flagged_transactions || [];
        const flows   = data.cash_flow_by_party  || [];
        const chains  = data.transaction_chains  || [];
        let html = '';
        if (flows.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#1a5276; font-size:12px; margin-bottom:6px;">💵 Cash Flow by Party</div>
                <table style="width:100%; border-collapse:collapse; font-size:12px;">
                    <tr style="background:#ecf0f1;"><th style="padding:4px 8px; text-align:left;">Party</th><th style="padding:4px 8px; text-align:right;">In</th><th style="padding:4px 8px; text-align:right;">Out</th><th style="padding:4px 8px; text-align:right;">Net</th></tr>
                    ${flows.map(f => `<tr style="border-top:1px solid #f0f0f0;">
                        <td style="padding:3px 8px; font-weight:600;">${escapeHtml(f.party || '')}</td>
                        <td style="padding:3px 8px; text-align:right; color:#27ae60;">+$${(f.total_in || 0).toLocaleString()}</td>
                        <td style="padding:3px 8px; text-align:right; color:#c0392b;">-$${(f.total_out || 0).toLocaleString()}</td>
                        <td style="padding:3px 8px; text-align:right; font-weight:700; color:${(f.net || 0) >= 0 ? '#27ae60' : '#c0392b'};">$${(f.net || 0).toLocaleString()}</td>
                    </tr>`).join('')}
                </table>
            </div>`;
        }
        if (flagged.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#c0392b; font-size:12px; margin-bottom:6px;">🚩 Flagged Transactions (${flagged.length})</div>
                ${flagged.map(ft => `<div class="ci-flagged-item">
                    <span class="severity-badge ${ft.significance || 'medium'}">${escapeHtml(ft.type || 'flag')}</span>
                    <strong>$${(ft.amount_usd || 0).toLocaleString()}</strong> — ${escapeHtml(ft.significance || '')}
                    <br><span style="font-size:11px; color:#666;">${escapeHtml(ft.date || '')}${ft.parties?.length ? ' · ' + ft.parties.join(', ') : ''}</span>
                </div>`).join('')}
            </div>`;
        }
        if (chains.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#6c3483; font-size:12px; margin-bottom:6px;">🔗 Transaction Chains</div>
                ${chains.map(c => `<div style="font-size:12px; padding:6px 10px; border:1px solid #e8d5f0; border-radius:5px; margin-bottom:5px;">
                    <strong>${escapeHtml(c.from || '')} → ${escapeHtml((c.via || []).join(' → '))} → ${escapeHtml(c.to || '')}</strong>
                    <span style="color:#888; margin-left:8px;">$${(c.total_usd || 0).toLocaleString()}</span>
                </div>`).join('')}
            </div>`;
        }
        if (data.summary) {
            html += `<div style="background:#f8f4ff; border:1px solid #d5c8f0; border-radius:6px; padding:12px 14px; font-size:12px; color:#333; line-height:1.6;">
                <div style="font-weight:700; color:#6c3483; margin-bottom:6px;">📋 Forensic Accounting Memo</div>
                ${escapeHtml(data.summary).replace(/\n/g, '<br>')}
                ${data.total_documented_exposure_usd ? `<div style="margin-top:8px; font-weight:700; color:#c0392b; font-size:13px;">Total Documented Exposure: $${data.total_documented_exposure_usd.toLocaleString()}</div>` : ''}
            </div>`;
        }
        body.innerHTML = html || '<p style="color:#888; font-size:13px;">No forensic findings.</p>';
    }

    function ciRenderDiscoveryTab(data) {
        const acc  = document.getElementById('ci-discovery-accordion');
        const body = document.getElementById('ci-discovery-body');
        if (!acc || !body) return;
        if (!data || data.error || (!data.rfp_list?.length && !data.missing_document_types?.length)) {
            acc.style.display = 'none'; return;
        }
        acc.style.display = '';
        const missing    = data.missing_document_types || [];
        const custodian  = data.custodian_gaps         || [];
        const spoliation = data.spoliation_indicators  || [];
        const rfps       = data.rfp_list               || [];
        const subpoenas  = data.subpoena_targets       || [];
        let html = '';
        if (missing.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#c0392b; font-size:12px; margin-bottom:6px;">📭 Missing Documents (${missing.length})</div>
                ${missing.map(m => `<div class="ci-flagged-item">
                    <span class="severity-badge ${m.priority || 'medium'}">${(m.priority || 'medium').toUpperCase()}</span>
                    <strong>${escapeHtml(m.description || '')}</strong>
                    <p style="font-size:11px; color:#666; margin:2px 0 0 0;">${escapeHtml(m.why_expected || '')}</p>
                </div>`).join('')}
            </div>`;
        }
        if (spoliation.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#c0392b; font-size:12px; margin-bottom:6px;">⚠️ Spoliation Indicators (${spoliation.length})</div>
                ${spoliation.map(s => `<div class="ci-flagged-item">
                    <span class="severity-badge ${s.severity || 'high'}">${(s.severity || 'high').toUpperCase()}</span>
                    ${escapeHtml(s.indicator || '')}
                </div>`).join('')}
            </div>`;
        }
        if (rfps.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#1a5276; font-size:12px; margin-bottom:6px;">📝 Requests for Production (${rfps.length})</div>
                <div id="ci-discovery-rfp-list">
                    ${rfps.map((r, i) => `<div class="ci-rfp-item">
                        <span class="rfp-text"><strong>RFP #${i+1}:</strong> ${escapeHtml(typeof r === 'string' ? r : (r.item || ''))}</span>
                        ${(r.legal_basis && typeof r !== 'string') ? '<div style="font-size:11px; color:#777; margin-top:2px;">Legal basis: ' + escapeHtml(r.legal_basis) + '</div>' : ''}
                    </div>`).join('')}
                </div>
            </div>`;
        }
        if (custodian.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#6c3483; font-size:12px; margin-bottom:6px;">👤 Custodian Gaps</div>
                <table style="width:100%; border-collapse:collapse; font-size:12px;">
                    <tr style="background:#ecf0f1;"><th style="padding:4px 8px; text-align:left;">Person</th><th style="padding:4px 8px; text-align:left;">Role</th><th style="padding:4px 8px; text-align:left;">Expected</th><th style="padding:4px 8px; text-align:right;">Found</th></tr>
                    ${custodian.map(c => `<tr style="border-top:1px solid #f0f0f0;">
                        <td style="padding:3px 8px; font-weight:600;">${escapeHtml(c.person || '')}</td>
                        <td style="padding:3px 8px; color:#666;">${escapeHtml(c.role || '')}</td>
                        <td style="padding:3px 8px; color:#555; font-size:11px;">${escapeHtml(c.expected_docs || '')}</td>
                        <td style="padding:3px 8px; text-align:right;">${c.actual_doc_count ?? '?'}</td>
                    </tr>`).join('')}
                </table>
            </div>`;
        }
        if (subpoenas.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#e67e22; font-size:12px; margin-bottom:6px;">📋 Subpoena Targets</div>
                ${subpoenas.map(s => `<div style="border:1px solid #fdebd0; border-radius:5px; padding:8px 10px; margin-bottom:5px; font-size:12px;">
                    <strong>${escapeHtml(s.entity || '')}</strong>
                    <p style="margin:3px 0 0 0; color:#666;">${escapeHtml(s.reason || '')}</p>
                    ${s.likely_docs ? '<p style="margin:2px 0 0 0; font-size:11px; color:#888;">Likely docs: ' + escapeHtml(s.likely_docs) + '</p>' : ''}
                </div>`).join('')}
            </div>`;
        }
        if (data.summary) {
            html += `<div style="background:#f0f8ff; border:1px solid #aed6f1; border-radius:6px; padding:12px 14px; font-size:12px; color:#333; line-height:1.6;">
                <div style="font-weight:700; color:#1a5276; margin-bottom:6px;">📋 Discovery Strategy Memo</div>
                ${escapeHtml(data.summary).replace(/\n/g, '<br>')}
            </div>`;
        }
        body.innerHTML = html || '<p style="color:#888; font-size:13px;">No discovery gaps identified.</p>';
    }

    function ciRenderWitnessesTab(data) {
        const acc  = document.getElementById('ci-witnesses-accordion');
        const body = document.getElementById('ci-witnesses-body');
        if (!acc || !body) return;
        const witnesses = Array.isArray(data) ? data : (data?.witness_cards || data?.witnesses || []);
        if (!witnesses.length) { acc.style.display = 'none'; return; }
        acc.style.display = '';
        const html = witnesses.map((w, idx) => {
            const score      = parseFloat(w.credibility_score || 0.5);
            const scorePct   = Math.round(score * 100);
            const scoreColor = score >= 0.7 ? '#27ae60' : score >= 0.4 ? '#e67e22' : '#c0392b';
            const impeachment     = w.impeachment_points       || [];
            const inconsistencies = w.prior_inconsistencies    || [];
            const keyQs           = w.deposition_key_questions || [];
            const panelId = `ci-witness-body-${idx}`;
            return `<div class="ci-witness-card">
                <div onclick="const p=document.getElementById('${panelId}');p.style.display=p.style.display==='none'?'block':'none'"
                     style="display:flex; align-items:center; gap:12px; cursor:pointer;">
                    <div style="flex:1;">
                        <div style="font-weight:700; font-size:13px;">${escapeHtml(w.witness_name || 'Unknown')}</div>
                        ${w.vulnerability_summary ? `<div style="font-size:11px; color:#666; margin-top:2px;">${escapeHtml(w.vulnerability_summary)}</div>` : ''}
                    </div>
                    <div style="flex-shrink:0; text-align:center; min-width:70px;">
                        <div style="font-size:11px; color:#888; margin-bottom:2px;">Credibility</div>
                        <div class="ci-credibility-bar"><div class="ci-credibility-fill" style="width:${scorePct}%; background:${scoreColor};"></div></div>
                        <div style="font-size:12px; font-weight:700; color:${scoreColor};">${scorePct}%</div>
                    </div>
                    <div style="font-size:18px; color:#aaa;">›</div>
                </div>
                <div id="${panelId}" style="display:none; margin-top:10px; padding-top:10px; border-top:1px solid #eee;">
                    ${impeachment.length ? `<div style="margin-bottom:10px;">
                        <div style="font-weight:700; font-size:11px; color:#c0392b; margin-bottom:4px;">🎯 Impeachment Points</div>
                        ${impeachment.map(p => `<div class="ci-flagged-item">
                            <span class="severity-badge ${p.severity || 'medium'}">${(p.severity || 'medium').toUpperCase()}</span>
                            ${escapeHtml(typeof p === 'string' ? p : (p.point || ''))}
                        </div>`).join('')}
                    </div>` : ''}
                    ${inconsistencies.length ? `<div style="margin-bottom:10px;">
                        <div style="font-weight:700; font-size:11px; color:#e67e22; margin-bottom:4px;">⚡ Prior Inconsistent Statements</div>
                        ${inconsistencies.map(pi => `<div style="font-size:12px; padding:6px 10px; border:1px solid #fde8d0; border-radius:5px; margin-bottom:5px;">
                            <div style="color:#e67e22;">A: ${escapeHtml(pi.statement_a || '')}</div>
                            <div style="color:#c0392b; margin-top:3px;">B: ${escapeHtml(pi.statement_b || '')}</div>
                        </div>`).join('')}
                    </div>` : ''}
                    ${keyQs.length ? `<div style="margin-bottom:8px;">
                        <div style="font-weight:700; font-size:11px; color:#1a5276; margin-bottom:4px;">❓ Key Deposition Questions</div>
                        ${keyQs.map((q, qi) => `<div style="font-size:12px; padding:4px 0; border-bottom:1px solid #f0f0f0;">Q${qi+1}: ${escapeHtml(q)}</div>`).join('')}
                    </div>` : ''}
                    ${w.recommended_deposition_order ? `<div style="font-size:11px; color:#888; margin-top:6px;">Recommended deposition order: #${w.recommended_deposition_order}</div>` : ''}
                </div>
            </div>`;
        }).join('');
        body.innerHTML = html;
    }

    function ciRenderWarRoomTab(data) {
        const acc  = document.getElementById('ci-warroom-accordion');
        const body = document.getElementById('ci-warroom-body');
        if (!acc || !body) return;
        if (!data || data.error || (!data.war_room_memo && !data.opposing_case_summary)) {
            acc.style.display = 'none';
            return;
        }
        acc.style.display = '';
        const dangerous      = data.top_3_dangerous_arguments || data.top_dangerous_arguments || [];
        const vulnerabilities = data.client_vulnerabilities || [];
        const smokingGuns    = data.smoking_guns_against_client || data.smoking_guns || [];
        let html = '';
        if (data.opposing_case_summary) {
            html += `<div style="margin-bottom:14px; padding:12px 14px; background:#fff5f5; border:1px solid #f5c6cb; border-radius:6px; font-size:12px; line-height:1.6;">
                <div style="font-weight:700; color:#c0392b; margin-bottom:6px;">⚔️ Opposing Counsel's Case Theory</div>
                ${escapeHtml(data.opposing_case_summary).replace(/\n/g, '<br>')}
            </div>`;
        }
        if (data.likelihood_of_success_pct != null) {
            const pct = data.likelihood_of_success_pct;
            const lc  = pct >= 65 ? '#27ae60' : pct >= 40 ? '#e67e22' : '#c0392b';
            html += `<div style="margin-bottom:14px; display:flex; align-items:center; gap:12px;">
                <div style="font-size:28px; font-weight:800; color:${lc};">${pct}%</div>
                <div style="font-size:12px; color:#555;">Estimated likelihood of success<br><span style="color:#888; font-size:11px;">Based on document analysis and theory strength</span></div>
            </div>`;
        }
        if (dangerous.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#c0392b; font-size:12px; margin-bottom:6px;">🔥 Top ${dangerous.length} Most Dangerous Arguments (Opposing)</div>
                ${dangerous.map((arg, i) => `<div class="ci-warroom-arg">
                    <div style="font-weight:700; color:#c0392b; margin-bottom:4px;">Arg ${i+1}: ${escapeHtml(arg.argument || '')}</div>
                    ${arg.our_response ? `<div style="margin-top:6px;"><span style="font-size:11px; font-weight:700; color:#27ae60;">OUR RESPONSE:</span> <span style="font-size:12px;">${escapeHtml(arg.our_response)}</span></div>` : ''}
                    ${arg.response_strength ? `<div style="margin-top:3px;"><span class="severity-badge ${arg.response_strength}">${arg.response_strength.toUpperCase()}</span> response strength</div>` : ''}
                </div>`).join('')}
            </div>`;
        }
        if (vulnerabilities.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#e67e22; font-size:12px; margin-bottom:6px;">⚠️ Client Vulnerabilities</div>
                ${vulnerabilities.map(v => `<div class="ci-flagged-item">
                    <span class="severity-badge ${v.severity || 'medium'}">${(v.severity || 'medium').toUpperCase()}</span>
                    <strong>${escapeHtml(v.vulnerability || '')}</strong>
                    ${v.mitigation ? '<p style="font-size:11px; color:#27ae60; margin:3px 0 0 0;">Mitigation: ' + escapeHtml(v.mitigation) + '</p>' : ''}
                </div>`).join('')}
            </div>`;
        }
        if (smokingGuns.length) {
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#922b21; font-size:12px; margin-bottom:6px;">💥 Documents Dangerous to Client</div>
                ${smokingGuns.map(sg => `<div style="border:1px solid #f1948a; border-radius:5px; padding:8px 10px; margin-bottom:5px; font-size:12px; background:#fff8f8;">
                    <strong>${escapeHtml(sg.doc_title || sg.title || 'Doc #' + (sg.doc_id || sg.paperless_doc_id || '?'))}</strong>
                    <p style="margin:3px 0 0 0; color:#666;">${escapeHtml(sg.why_dangerous || '')}</p>
                </div>`).join('')}
            </div>`;
        }
        if (data.settlement_analysis) {
            const s = data.settlement_analysis;
            html += `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#1a5276; font-size:12px; margin-bottom:6px;">💼 Settlement Analysis</div>
                <table class="ci-settlement-table">
                    <tr><td>Settlement Range</td><td><strong>$${(s.range_low_usd || 0).toLocaleString()} — $${(s.range_high_usd || 0).toLocaleString()}</strong></td></tr>
                    ${s.walk_away_usd != null ? `<tr><td>Walk-Away Threshold</td><td><strong>$${s.walk_away_usd.toLocaleString()}</strong></td></tr>` : ''}
                    ${s.rationale ? `<tr><td colspan="2" style="color:#666; font-size:11px; padding-top:6px;">${escapeHtml(s.rationale)}</td></tr>` : ''}
                </table>
                ${s.leverage_points?.length ? `<div style="margin-top:8px; font-size:12px;"><strong>Leverage Points:</strong><ul style="margin:4px 0 0 0; padding-left:18px; color:#555;">${s.leverage_points.map(lp => `<li>${escapeHtml(lp)}</li>`).join('')}</ul></div>` : ''}
            </div>`;
        }
        if (data.war_room_memo) {
            html += `<div style="background:#f8f4ff; border:1px solid #d5c8f0; border-radius:6px; padding:12px 14px; font-size:12px; color:#333; line-height:1.6; margin-bottom:12px;">
                <div style="font-weight:700; color:#6c3483; margin-bottom:6px;">📋 War Room Memo</div>
                ${escapeHtml(data.war_room_memo).replace(/\n/g, '<br>')}
            </div>`;
        }
        // Opposing Counsel Checklist
        const occ = data.opposing_counsel_checklist || [];
        const occEl = document.getElementById('ci-warroom-occ');
        if (occEl) {
            if (occ.length) {
                const catColors = {discovery:'#1a5276', deposition:'#6c3483', motion:'#c0392b', trial:'#e67e22', investigation:'#27ae60'};
                occEl.style.display = '';
                occEl.innerHTML = `<div style="border-top:1px solid #eee; padding-top:12px;">
                    <div style="font-weight:700; color:#2c3e50; font-size:12px; margin-bottom:8px;">📋 Opposing Counsel Checklist — What They Will Do</div>
                    ${occ.map(item => {
                        const catColor = catColors[item.category] || '#555';
                        return `<div style="display:flex; gap:10px; padding:7px 10px; border-bottom:1px solid #f5f5f5; font-size:12px; align-items:flex-start;">
                            <span style="min-width:80px; font-size:10px; font-weight:700; color:${catColor}; text-transform:uppercase; padding-top:2px;">${escapeHtml(item.category || '')}</span>
                            <div style="flex:1;">
                                <div style="font-weight:600;">${escapeHtml(item.action || '')}</div>
                                ${item.timing ? `<div style="color:#888; font-size:11px; margin-top:1px;">⏱ ${escapeHtml(item.timing)}</div>` : ''}
                                ${item.our_preparation ? `<div style="color:#27ae60; font-size:11px; margin-top:3px;">→ Our prep: ${escapeHtml(item.our_preparation)}</div>` : ''}
                            </div>
                        </div>`;
                    }).join('')}
                </div>`;
            } else {
                occEl.style.display = 'none';
            }
        }
        // Senior partner notes — build inline (body.innerHTML overwrites the child divs)
        if (data.senior_partner_notes) {
            const _rawSPN = data.senior_partner_notes;
            let notes = {};
            if (typeof _rawSPN === 'string') {
                try { notes = JSON.parse(_rawSPN); } catch(e) { notes = {}; }
            } else {
                notes = _rawSPN || {};
            }
            // Fallback: if no structured fields parsed, render raw text
            const _hasStructured = notes.single_most_important_finding || notes.missed_issues?.length
                || notes.logical_leaps?.length || notes.theories_that_wont_survive_cross?.length
                || notes.senior_partner_notes;
            if (!_hasStructured && typeof _rawSPN === 'string') {
                notes = { senior_partner_notes: _rawSPN };
            }
            let snHtml = '';
            if (notes.single_most_important_finding) {
                snHtml += `<div style="margin-bottom:10px; padding:10px 12px; background:#fffbf0; border:1px solid #f0d060; border-radius:5px;">
                    <div style="font-weight:700; color:#b8860b; font-size:12px; margin-bottom:4px;">⭐ Most Important Finding</div>
                    <div style="font-size:12px; color:#333;">${escapeHtml(notes.single_most_important_finding)}</div>
                </div>`;
            }
            if (notes.missed_issues?.length) {
                snHtml += `<div style="margin-bottom:10px;">
                    <div style="font-weight:700; font-size:11px; color:#c0392b; margin-bottom:4px;">❌ Issues Missed or Understated</div>
                    ${notes.missed_issues.map(mi => `<div style="font-size:12px; padding:4px 8px; background:#fff5f5; border-left:3px solid #e74c3c; margin-bottom:3px;">${escapeHtml(mi)}</div>`).join('')}
                </div>`;
            }
            if (notes.logical_leaps?.length) {
                snHtml += `<div style="margin-bottom:10px;">
                    <div style="font-weight:700; font-size:11px; color:#e67e22; margin-bottom:4px;">⚡ Unsupported Conclusions</div>
                    ${notes.logical_leaps.map(ll => `<div style="font-size:12px; padding:4px 8px; background:#fffbf0; border-left:3px solid #e67e22; margin-bottom:3px;">${escapeHtml(ll)}</div>`).join('')}
                </div>`;
            }
            if (notes.theories_that_wont_survive_cross?.length) {
                snHtml += `<div style="margin-bottom:10px;">
                    <div style="font-weight:700; font-size:11px; color:#922b21; margin-bottom:4px;">🎯 Theories That Won't Survive Cross</div>
                    ${notes.theories_that_wont_survive_cross.map(t => `<div style="font-size:12px; padding:4px 8px; background:#fff0f0; border-left:3px solid #c0392b; margin-bottom:3px;">${escapeHtml(t)}</div>`).join('')}
                </div>`;
            }
            // Fallback: raw text memo (stored before structured JSON was available)
            if (!snHtml && notes.senior_partner_notes && typeof notes.senior_partner_notes === 'string') {
                snHtml += `<div style="font-size:12px; line-height:1.6; white-space:pre-wrap;">${escapeHtml(notes.senior_partner_notes)}</div>`;
            }
            if (snHtml) {
                html += `<div style="margin-top:14px; border-top:2px solid #f0d060; padding-top:12px;">
                    <div style="font-weight:700; color:#b8860b; font-size:12px; margin-bottom:8px;">🧑‍⚖️ Senior Partner Review Notes</div>
                    ${snHtml}
                </div>`;
            }
        }
        body.innerHTML = html || '<p style="color:#888; font-size:13px;">No war room data.</p>';
    }

    // ── Tier 5: Deep Financial Forensics ─────────────────────────────
    function ciRenderDeepForensicsTab(data) {
        const acc  = document.getElementById('ci-deepforensics-accordion');
        const body = document.getElementById('ci-deepforensics-body');
        if (!acc || !body) return;
        if (!data || data.error || (!data.summary && !data.risk_score)) {
            acc.style.display = 'none'; return;
        }
        acc.style.display = '';

        // Benford's Law section
        const bf = data.benford_analysis || {};
        let bfHtml = '';
        if (bf.sufficient_data) {
            const sigColor = bf.significance === 'highly_significant' ? '#c0392b'
                           : bf.significance === 'significant' ? '#e67e22'
                           : bf.significance === 'borderline' ? '#f39c12' : '#27ae60';
            bfHtml = `<div style="margin-bottom:14px;">
                <div style="font-weight:700; color:#1a5276; font-size:12px; margin-bottom:6px;">📊 Benford's Law Analysis (${bf.sample_size} transactions)</div>
                <div style="padding:6px 10px; border-left:3px solid ${sigColor}; background:#f8f9fa; margin-bottom:8px; font-size:12px;">
                    <strong style="color:${sigColor};">${(bf.significance || '').replace(/_/g, ' ').toUpperCase()}</strong> — χ²=${bf.chi2_statistic}
                    <br>${escapeHtml(bf.interpretation || '')}
                </div>
                ${bf.notable_deviations?.length ? `<div style="font-size:11px; color:#666; margin-bottom:4px;">Notable deviations: ${bf.notable_deviations.map(d => `Digit ${d.digit}: ${d.observed_pct}% obs vs ${d.expected_pct}% exp (${d.direction})`).join(' · ')}</div>` : ''}
                ${data.benford_interpretation ? `<div style="font-size:12px; color:#555; margin-top:6px; font-style:italic;">${escapeHtml(data.benford_interpretation)}</div>` : ''}
            </div>`;
        }
        document.getElementById('ci-deepforensics-benfords').innerHTML = bfHtml;

        // Round-trip transactions
        const rts = data.round_trip_transactions || [];
        let rtHtml = '';
        if (rts.length) {
            rtHtml = `<div>
                <div style="font-weight:700; color:#c0392b; font-size:12px; margin-bottom:6px;">🔄 Round-Trip Transactions (${rts.length})</div>
                ${rts.map(rt => `<div style="padding:8px 12px; border:1px solid #fad7d7; border-radius:5px; margin-bottom:6px; background:#fff8f8;">
                    <div style="font-size:12px; font-weight:600;">${escapeHtml(rt.description || '')}</div>
                    <div style="font-size:11px; color:#777; margin-top:3px;">${(rt.chain || []).join(' → ')} — $${(rt.total_usd || 0).toLocaleString()}</div>
                    <div style="font-size:11px; color:#555; margin-top:3px;">${escapeHtml(rt.actual_effect || '')}</div>
                </div>`).join('')}
            </div>`;
        }
        document.getElementById('ci-deepforensics-roundtrips').innerHTML = rtHtml;

        // Shell entity flags
        const shells = data.shell_entity_flags || [];
        let shellHtml = '';
        if (shells.length) {
            shellHtml = `<div>
                <div style="font-weight:700; color:#6c3483; font-size:12px; margin-bottom:6px;">🏚️ Shell Entity Flags (${shells.length})</div>
                ${shells.map(s => `<div style="padding:8px 12px; border:1px solid #d7bde2; border-radius:5px; margin-bottom:6px; background:#fdf2ff;">
                    <div style="font-size:12px; font-weight:600;">${escapeHtml(s.entity_name || '')} <span style="font-weight:400; color:#888;">${escapeHtml(s.jurisdiction || '')}</span></div>
                    <div style="font-size:11px; color:#666; margin-top:3px;">${(s.shell_indicators || []).join(' · ')}</div>
                    <div style="font-size:11px; color:#444; margin-top:4px; font-style:italic;">${escapeHtml(s.assessment || '')}</div>
                </div>`).join('')}
            </div>`;
        }
        document.getElementById('ci-deepforensics-shells').innerHTML = shellHtml;

        // Layering schemes
        const layers = data.layering_schemes || [];
        let layerHtml = '';
        if (layers.length) {
            layerHtml = `<div>
                <div style="font-weight:700; color:#922b21; font-size:12px; margin-bottom:6px;">🌊 Layering Schemes (${layers.length})</div>
                ${layers.map(l => `<div style="padding:8px 12px; border:1px solid #f5b7b1; border-radius:5px; margin-bottom:6px; background:#fff5f5;">
                    <div style="font-size:12px; font-weight:600;">${escapeHtml(l.description || '')}</div>
                    <div style="font-size:11px; color:#555; margin-top:3px;">Steps: ${(l.steps || []).map(s => `${s.from}→${s.to} ($${(s.amount_usd || 0).toLocaleString()})`).join(' → ')}</div>
                    <div style="font-size:11px; color:#777; margin-top:3px;">Ultimate beneficiary: <strong>${escapeHtml(l.ultimate_beneficiary || 'Unknown')}</strong> — $${(l.amount_laundered_usd || 0).toLocaleString()}</div>
                </div>`).join('')}
            </div>`;
        }
        document.getElementById('ci-deepforensics-layering').innerHTML = layerHtml;

        // Risk score + summary
        const riskScore = data.risk_score || 0;
        const riskColor = riskScore >= 70 ? '#c0392b' : riskScore >= 40 ? '#e67e22' : '#27ae60';
        let summaryHtml = `<div style="display:flex; align-items:center; gap:16px; margin-bottom:10px;">
            <div style="text-align:center; min-width:70px;">
                <div style="font-size:28px; font-weight:800; color:${riskColor};">${riskScore}</div>
                <div style="font-size:10px; color:#777; font-weight:600;">RISK SCORE</div>
            </div>
            ${data.highest_priority_investigation ? `<div style="flex:1; font-size:12px; padding:8px 12px; background:#fffbf0; border:1px solid #f0d060; border-radius:5px;"><strong>Priority Lead:</strong> ${escapeHtml(data.highest_priority_investigation)}</div>` : ''}
        </div>`;
        if (data.summary) {
            summaryHtml += `<div style="font-size:13px; color:#444; line-height:1.7; white-space:pre-wrap;">${escapeHtml(data.summary)}</div>`;
        }
        document.getElementById('ci-deepforensics-summary').innerHTML = summaryHtml;
    }

    // ── Tier 5: Trial Strategy ────────────────────────────────────────
    function ciRenderTrialStrategyTab(data) {
        const acc  = document.getElementById('ci-trialstrategy-accordion');
        const body = document.getElementById('ci-trialstrategy-body');
        if (!acc || !body) return;
        if (!data || data.error || !data.strategy_memo) {
            acc.style.display = 'none'; return;
        }
        acc.style.display = '';

        // Opening theme + narrative
        let themeHtml = '';
        if (data.opening_theme) {
            themeHtml = `<div style="padding:12px 16px; background:linear-gradient(135deg,#1a5276,#2980b9); color:#fff; border-radius:8px; margin-bottom:10px;">
                <div style="font-size:10px; font-weight:700; opacity:0.8; margin-bottom:4px;">OPENING THEME</div>
                <div style="font-size:15px; font-weight:700; line-height:1.4;">"${escapeHtml(data.opening_theme)}"</div>
            </div>`;
        }
        if (data.our_narrative || data.their_narrative) {
            themeHtml += `<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:8px;">
                ${data.our_narrative ? `<div style="padding:8px 12px; background:#e8f8f5; border-radius:5px; font-size:12px;"><strong style="color:#1e8449;">Our Story:</strong><br>${escapeHtml(data.our_narrative)}</div>` : ''}
                ${data.their_narrative ? `<div style="padding:8px 12px; background:#fdf2f2; border-radius:5px; font-size:12px;"><strong style="color:#c0392b;">Their Story:</strong><br>${escapeHtml(data.their_narrative)}</div>` : ''}
            </div>`;
        }
        document.getElementById('ci-trialstrategy-theme').innerHTML = themeHtml;

        // Witness order
        const witnesses = data.witness_order || [];
        let witnessHtml = '';
        if (witnesses.length) {
            witnessHtml = `<div>
                <div style="font-weight:700; color:#2c3e50; font-size:12px; margin-bottom:6px;">👤 Witness Order (${witnesses.length})</div>
                ${witnesses.map(w => `<div style="display:flex; gap:10px; padding:6px 10px; border-bottom:1px solid #f0f0f0; align-items:flex-start; font-size:12px;">
                    <div style="min-width:24px; height:24px; background:#2c3e50; color:#fff; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700;">${w.order}</div>
                    <div style="flex:1;">
                        <strong>${escapeHtml(w.witness_name || '')}</strong> <span style="color:#888; font-size:11px;">${escapeHtml(w.role || '')}</span>
                        <div style="color:#555; margin-top:2px;">${escapeHtml(w.purpose || '')}</div>
                        ${w.risk ? `<div style="color:#c0392b; font-size:11px; margin-top:2px;">⚠️ Risk: ${escapeHtml(w.risk)}</div>` : ''}
                    </div>
                </div>`).join('')}
            </div>`;
        }
        document.getElementById('ci-trialstrategy-witnesses').innerHTML = witnessHtml;

        // Key exhibits
        const exhibits = data.key_exhibits || [];
        let exhibitHtml = '';
        if (exhibits.length) {
            exhibitHtml = `<div>
                <div style="font-weight:700; color:#2c3e50; font-size:12px; margin-bottom:6px;">📄 Key Exhibits (Top ${Math.min(exhibits.length, 10)})</div>
                ${exhibits.slice(0,10).map(e => `<div style="display:flex; gap:8px; padding:5px 8px; border-bottom:1px solid #f0f0f0; font-size:12px; align-items:flex-start;">
                    <div style="min-width:20px; font-weight:700; color:#7f8c8d;">#${e.rank}</div>
                    <div style="flex:1;">
                        <span style="font-weight:600;">${escapeHtml(e.doc_description || '')}</span>
                        ${e.paperless_doc_id ? ` <span style="color:#888; font-size:11px;">[Doc #${e.paperless_doc_id}]</span>` : ''}
                        <div style="color:#27ae60; font-size:11px; margin-top:1px;">${escapeHtml(e.why_powerful || '')}</div>
                    </div>
                </div>`).join('')}
            </div>`;
        }
        document.getElementById('ci-trialstrategy-exhibits').innerHTML = exhibitHtml;

        // Motions in limine
        const mils = data.motions_in_limine || [];
        let milHtml = '';
        if (mils.length) {
            milHtml = `<div>
                <div style="font-weight:700; color:#2c3e50; font-size:12px; margin-bottom:6px;">⚖️ Motions In Limine (${mils.length})</div>
                ${mils.map(m => `<div style="padding:6px 10px; border:1px solid #e8e8e8; border-radius:4px; margin-bottom:5px; font-size:12px;">
                    <div style="font-weight:600;">${escapeHtml(m.motion || '')}</div>
                    <div style="color:#666; font-size:11px; margin-top:2px;">${escapeHtml(m.legal_basis || '')} — Likelihood: <strong>${escapeHtml(m.likelihood_of_success || '')}</strong></div>
                    ${m.impact_if_admitted ? `<div style="color:#c0392b; font-size:11px; margin-top:2px;">If denied: ${escapeHtml(m.impact_if_admitted)}</div>` : ''}
                </div>`).join('')}
            </div>`;
        }
        document.getElementById('ci-trialstrategy-mils').innerHTML = milHtml;

        // Strategy memo
        document.getElementById('ci-trialstrategy-memo').innerHTML = data.strategy_memo
            ? `<div style="border-top:1px solid #eee; padding-top:12px; font-size:13px; color:#444; line-height:1.7; white-space:pre-wrap;">${escapeHtml(data.strategy_memo)}</div>`
            : '';
    }

    // ── Tier 5: Multi-Model Comparison ───────────────────────────────
    function ciRenderMultiModelTab(data) {
        const acc  = document.getElementById('ci-multimodel-accordion');
        const body = document.getElementById('ci-multimodel-body');
        if (!acc || !body) return;
        if (!data || data.error || (!data.agreed_theories?.length && !data.merged_summary)) {
            acc.style.display = 'none'; return;
        }
        acc.style.display = '';

        const agreementRate = data.models_agreement_rate || 0;
        const agreeColor = agreementRate >= 0.7 ? '#27ae60' : agreementRate >= 0.4 ? '#f39c12' : '#c0392b';

        // Agreed theories
        const agreed = data.agreed_theories || [];
        let agreedHtml = '';
        if (agreed.length) {
            agreedHtml = `<div>
                <div style="font-weight:700; color:#27ae60; font-size:12px; margin-bottom:6px;">✅ Agreed Findings (${agreed.length}) — Both models independently identified</div>
                ${agreed.map(t => `<div style="padding:6px 10px; border:1px solid #d5f5e3; border-radius:4px; margin-bottom:4px; background:#f0faf5; font-size:12px;">
                    <strong>${escapeHtml(t.theory_text || '')}</strong>
                    <span style="color:#888; margin-left:8px; font-size:11px;">conf: ${((t.merged_confidence || 0) * 100).toFixed(0)}% · ${escapeHtml(t.significance || '')}</span>
                </div>`).join('')}
            </div>`;
        }
        document.getElementById('ci-multimodel-agreed').innerHTML = agreedHtml;

        // Unique findings from each model
        const aOnly = data.model_a_only || [];
        const bOnly = data.model_b_only || [];
        let uniqueHtml = '';
        if (aOnly.length || bOnly.length) {
            uniqueHtml = `<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">`;
            if (aOnly.length) {
                uniqueHtml += `<div>
                    <div style="font-weight:700; color:#8e44ad; font-size:11px; margin-bottom:4px;">🔵 Anthropic Only (${aOnly.length})</div>
                    ${aOnly.map(t => `<div style="padding:5px 8px; border:1px solid #d7bde2; border-radius:4px; margin-bottom:3px; background:#fdf2ff; font-size:11px;">
                        ${escapeHtml(t.theory_text || '')}
                        <div style="color:#888; font-size:10px; margin-top:1px;">${escapeHtml(t.assessment || '')}</div>
                    </div>`).join('')}
                </div>`;
            }
            if (bOnly.length) {
                uniqueHtml += `<div>
                    <div style="font-weight:700; color:#2980b9; font-size:11px; margin-bottom:4px;">🟢 OpenAI Only (${bOnly.length})</div>
                    ${bOnly.map(t => `<div style="padding:5px 8px; border:1px solid #aed6f1; border-radius:4px; margin-bottom:3px; background:#eaf4fd; font-size:11px;">
                        ${escapeHtml(t.theory_text || '')}
                        <div style="color:#888; font-size:10px; margin-top:1px;">${escapeHtml(t.assessment || '')}</div>
                    </div>`).join('')}
                </div>`;
            }
            uniqueHtml += `</div>`;
        }
        document.getElementById('ci-multimodel-unique').innerHTML = uniqueHtml;

        // Disagreements
        const disags = data.disagreements || [];
        let disagHtml = '';
        if (disags.length) {
            disagHtml = `<div>
                <div style="font-weight:700; color:#c0392b; font-size:12px; margin-bottom:6px;">⚡ Disagreements — High-Uncertainty Areas (${disags.length})</div>
                ${disags.map(d => `<div style="padding:8px 12px; border:1px solid #f5b7b1; border-radius:5px; margin-bottom:6px; background:#fff8f8;">
                    <div style="font-size:12px; font-weight:600; margin-bottom:4px;">${escapeHtml(d.topic || '')}</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:11px; margin-bottom:6px;">
                        <div style="padding:4px 8px; background:#f5eef8; border-radius:3px;"><strong>Anthropic:</strong> ${escapeHtml(d.model_a_position || '')}</div>
                        <div style="padding:4px 8px; background:#eaf4fd; border-radius:3px;"><strong>OpenAI:</strong> ${escapeHtml(d.model_b_position || '')}</div>
                    </div>
                    ${d.recommendation ? `<div style="font-size:11px; color:#2c3e50; font-style:italic;">→ ${escapeHtml(d.recommendation)}</div>` : ''}
                </div>`).join('')}
            </div>`;
        }
        document.getElementById('ci-multimodel-disagreements').innerHTML = disagHtml;

        // Summary + agreement meter
        document.getElementById('ci-multimodel-summary').innerHTML = `
            <div style="border-top:1px solid #eee; padding-top:12px;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                    <div style="font-size:11px; color:#555; font-weight:600;">Model Agreement:</div>
                    <div style="flex:1; background:#e8e8e8; border-radius:10px; height:10px; overflow:hidden;">
                        <div style="background:${agreeColor}; height:100%; width:${(agreementRate * 100).toFixed(0)}%; border-radius:10px;"></div>
                    </div>
                    <div style="font-size:13px; font-weight:700; color:${agreeColor};">${(agreementRate * 100).toFixed(0)}%</div>
                </div>
                ${data.merged_summary ? `<div style="font-size:13px; color:#444; line-height:1.7; white-space:pre-wrap;">${escapeHtml(data.merged_summary)}</div>` : ''}
            </div>`;
    }

    // ── Tier 5: Settlement Valuation ─────────────────────────────────
    function ciRenderSettlementTab(data) {
        const acc  = document.getElementById('ci-settlement-accordion');
        const body = document.getElementById('ci-settlement-body');
        if (!acc || !body) return;
        if (!data || data.error || (!data.summary_memo && !data.settlement_recommendation)) {
            acc.style.display = 'none'; return;
        }
        acc.style.display = '';

        // Total exposure summary
        const exp = data.total_exposure || {};
        let exposureHtml = '';
        if (exp.low_usd || exp.likely_usd || exp.high_usd) {
            const fmtUsd = v => v ? '$' + Number(v).toLocaleString() : '—';
            exposureHtml = `<div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:10px;">
                <div style="flex:1; min-width:120px; padding:12px; background:#f8f9fa; border:1px solid #dee2e6; border-radius:6px; text-align:center;">
                    <div style="font-size:10px; color:#888; font-weight:600; text-transform:uppercase; margin-bottom:4px;">Low Exposure</div>
                    <div style="font-size:18px; font-weight:700; color:#27ae60;">${fmtUsd(exp.low_usd)}</div>
                </div>
                <div style="flex:1; min-width:120px; padding:12px; background:#fff3cd; border:1px solid #ffc107; border-radius:6px; text-align:center;">
                    <div style="font-size:10px; color:#856404; font-weight:600; text-transform:uppercase; margin-bottom:4px;">Likely Exposure</div>
                    <div style="font-size:18px; font-weight:700; color:#856404;">${fmtUsd(exp.likely_usd)}</div>
                </div>
                <div style="flex:1; min-width:120px; padding:12px; background:#f8d7da; border:1px solid #f5c6cb; border-radius:6px; text-align:center;">
                    <div style="font-size:10px; color:#721c24; font-weight:600; text-transform:uppercase; margin-bottom:4px;">High Exposure</div>
                    <div style="font-size:18px; font-weight:700; color:#c0392b;">${fmtUsd(exp.high_usd)}</div>
                </div>
            </div>`;
            if (exp.notes) {
                exposureHtml += `<div style="font-size:12px; color:#666; margin-bottom:8px; font-style:italic;">${escapeHtml(exp.notes)}</div>`;
            }
        }
        document.getElementById('ci-settlement-exposure').innerHTML = exposureHtml;

        // Damages breakdown
        const damages = data.damages_breakdown || [];
        let damagesHtml = '';
        if (damages.length) {
            damagesHtml = `<div>
                <div style="font-weight:700; color:#2c3e50; font-size:12px; margin-bottom:6px;">📊 Damages Breakdown by Category</div>
                <table style="width:100%; border-collapse:collapse; font-size:12px;">
                    <thead><tr style="background:#f8f9fa; font-size:11px; color:#555;">
                        <th style="padding:6px 8px; text-align:left; border-bottom:2px solid #dee2e6;">Category</th>
                        <th style="padding:6px 8px; text-align:right; border-bottom:2px solid #dee2e6;">Low</th>
                        <th style="padding:6px 8px; text-align:right; border-bottom:2px solid #dee2e6;">Likely</th>
                        <th style="padding:6px 8px; text-align:right; border-bottom:2px solid #dee2e6;">High</th>
                    </tr></thead>
                    <tbody>
                        ${damages.map(d => `<tr style="border-bottom:1px solid #f0f0f0;">
                            <td style="padding:6px 8px;">${escapeHtml(d.category || '')}</td>
                            <td style="padding:6px 8px; text-align:right; color:#27ae60;">${d.low_usd ? '$'+Number(d.low_usd).toLocaleString() : '—'}</td>
                            <td style="padding:6px 8px; text-align:right; color:#856404; font-weight:600;">${d.likely_usd ? '$'+Number(d.likely_usd).toLocaleString() : '—'}</td>
                            <td style="padding:6px 8px; text-align:right; color:#c0392b;">${d.high_usd ? '$'+Number(d.high_usd).toLocaleString() : '—'}</td>
                        </tr>`).join('')}
                    </tbody>
                </table>
            </div>`;
        }
        document.getElementById('ci-settlement-damages').innerHTML = damagesHtml;

        // Settlement recommendation
        const rec = data.settlement_recommendation || {};
        let recHtml = '';
        if (rec.open_at_usd || rec.target_usd || rec.walk_away_usd) {
            const fmtUsd = v => v ? '$' + Number(v).toLocaleString() : '—';
            recHtml = `<div>
                <div style="font-weight:700; color:#2c3e50; font-size:12px; margin-bottom:8px;">🤝 Settlement Recommendation</div>
                <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:8px;">
                    <div style="flex:1; min-width:100px; padding:10px; background:#e8f5e9; border:1px solid #c8e6c9; border-radius:5px; text-align:center;">
                        <div style="font-size:10px; color:#2e7d32; font-weight:600; text-transform:uppercase; margin-bottom:3px;">Open At</div>
                        <div style="font-size:16px; font-weight:700; color:#1b5e20;">${fmtUsd(rec.open_at_usd)}</div>
                    </div>
                    <div style="flex:1; min-width:100px; padding:10px; background:#fff8e1; border:1px solid #ffe082; border-radius:5px; text-align:center;">
                        <div style="font-size:10px; color:#f57f17; font-weight:600; text-transform:uppercase; margin-bottom:3px;">Target</div>
                        <div style="font-size:16px; font-weight:700; color:#e65100;">${fmtUsd(rec.target_usd)}</div>
                    </div>
                    <div style="flex:1; min-width:100px; padding:10px; background:#ffebee; border:1px solid #ef9a9a; border-radius:5px; text-align:center;">
                        <div style="font-size:10px; color:#b71c1c; font-weight:600; text-transform:uppercase; margin-bottom:3px;">Walk Away</div>
                        <div style="font-size:16px; font-weight:700; color:#b71c1c;">${fmtUsd(rec.walk_away_usd)}</div>
                    </div>
                </div>
                ${rec.rationale ? `<div style="font-size:12px; color:#444; font-style:italic; padding:8px; background:#f8f9fa; border-radius:4px;">${escapeHtml(rec.rationale)}</div>` : ''}
            </div>`;
        }
        document.getElementById('ci-settlement-recommendation').innerHTML = recHtml;

        // Leverage timeline
        const leverage = data.leverage_timeline || [];
        let leverageHtml = '';
        if (leverage.length) {
            leverageHtml = `<div>
                <div style="font-weight:700; color:#2c3e50; font-size:12px; margin-bottom:6px;">⏱️ Leverage Timeline</div>
                ${leverage.map(l => `<div style="display:flex; gap:10px; padding:6px 10px; border-left:3px solid #3498db; margin-bottom:5px; background:#f8f9ff; font-size:12px;">
                    <div style="font-weight:600; color:#2980b9; min-width:80px;">${escapeHtml(l.milestone || l.phase || '')}</div>
                    <div style="flex:1;">${escapeHtml(l.leverage_shift || l.description || '')}</div>
                </div>`).join('')}
            </div>`;
        }
        document.getElementById('ci-settlement-leverage').innerHTML = leverageHtml;

        // Mediation strategy
        const med = data.mediation_strategy || {};
        let mediationHtml = '';
        if (med.recommended_timing || med.mediator_profile || med.opening_position) {
            mediationHtml = `<div>
                <div style="font-weight:700; color:#2c3e50; font-size:12px; margin-bottom:6px;">🕊️ Mediation Strategy</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px;">
                    ${med.recommended_timing ? `<div style="padding:6px 10px; background:#f0f4ff; border-radius:4px;"><strong>Best Timing:</strong> ${escapeHtml(med.recommended_timing)}</div>` : ''}
                    ${med.mediator_profile ? `<div style="padding:6px 10px; background:#f0f4ff; border-radius:4px;"><strong>Mediator Profile:</strong> ${escapeHtml(med.mediator_profile)}</div>` : ''}
                    ${med.opening_position ? `<div style="padding:6px 10px; background:#f0f4ff; border-radius:4px;"><strong>Opening Position:</strong> ${escapeHtml(med.opening_position)}</div>` : ''}
                    ${med.fallback_strategy ? `<div style="padding:6px 10px; background:#f0f4ff; border-radius:4px;"><strong>Fallback:</strong> ${escapeHtml(med.fallback_strategy)}</div>` : ''}
                </div>
            </div>`;
        }
        document.getElementById('ci-settlement-mediation').innerHTML = mediationHtml;

        // Summary memo
        document.getElementById('ci-settlement-memo').innerHTML = data.summary_memo
            ? `<div style="border-top:1px solid #eee; padding-top:12px; font-size:13px; color:#444; line-height:1.7; white-space:pre-wrap;">${escapeHtml(data.summary_memo)}</div>`
            : '';
    }

    // ── Report builder ───────────────────────────────────────────────
    let ciCurrentReportId = null;
    let ciReportPollTimer = null;

    async function ciGenerateReport() {
        if (!window.APP_CONFIG.isAdvanced || !ciSelectedRunId) return;
        const instructions = document.getElementById('ci-report-instructions').value.trim();
        if (!instructions) { showToast('Please enter report instructions', 'warning'); return; }
        const template = document.getElementById('ci-report-template').value;

        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + ciSelectedRunId + '/reports'), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ instructions, template }),
            });
            const d = await r.json();
            if (d.report_id) {
                ciCurrentReportId = d.report_id;
                document.getElementById('ci-report-status').style.display = 'block';
                document.getElementById('ci-report-status').textContent = 'Generating report...';
                document.getElementById('ci-report-preview-section').style.display = 'none';
                if (ciReportPollTimer) clearInterval(ciReportPollTimer);
                ciReportPollTimer = setInterval(() => ciPollReport(ciSelectedRunId, ciCurrentReportId), 3000);
            } else {
                showToast(d.error || 'Failed to start report generation', 'error');
            }
        } catch(e) {
            showToast('Error: ' + e.message, 'error');
        }
    }

    async function ciPollReport(runId, reportId) {
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + runId + '/reports/' + reportId));
            const d = await r.json();
            if (d.status === 'complete' && d.content) {
                clearInterval(ciReportPollTimer);
                document.getElementById('ci-report-status').textContent = 'Report ready!';
                document.getElementById('ci-report-preview-section').style.display = 'block';
                document.getElementById('ci-report-preview').textContent = d.content.substring(0, 2000) + (d.content.length > 2000 ? '\n\n[... truncated for preview ...]' : '');
                const pdfLink = document.getElementById('ci-report-pdf-link');
                pdfLink.style.display = 'inline';
                pdfLink.href = apiUrl('/api/ci/runs/' + runId + '/reports/' + reportId + '/pdf');
            } else if (d.status === 'failed') {
                clearInterval(ciReportPollTimer);
                document.getElementById('ci-report-status').textContent = 'Report generation failed.';
            }
        } catch(e) {}
    }

    // ── Tab open hook ────────────────────────────────────────────────
    const _origSwitchTab = window.switchTab;
    if (_origSwitchTab) {
        window.switchTab = function(tab) {
            _origSwitchTab(tab);
            if (tab === 'case-intelligence') {
                ciLoadJurisdictions().then(() => ciDetectJurisdiction());
                ciRefreshRuns();
            }
        };
    }

    function escapeHtml(s) {
        if (!s) return '';
        return String(s)
            .replace(/&/g,'&amp;')
            .replace(/</g,'&lt;')
            .replace(/>/g,'&gt;')
            .replace(/"/g,'&quot;');
    }

