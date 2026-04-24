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

