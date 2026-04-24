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

