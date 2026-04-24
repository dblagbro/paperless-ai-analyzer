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

