// ── Overview Tab — Stats & Recent Documents ─────────────────────────────


        // Auto-refresh data every 10 seconds
        let refreshInterval;

        async function fetchStatus() {
            try {
                const response = await apiFetch(apiUrl('/api/status'));
                const data = await response.json();

                // Update stats
                document.getElementById('status').textContent = data.status === 'running' ? 'Running' : 'Stopped';
                document.getElementById('total-analyzed').textContent = data.stats.total_analyzed;
                document.getElementById('anomalies-detected').textContent = data.stats.anomalies_detected;
                document.getElementById('high-risk-count').textContent = data.stats.high_risk_count;
                document.getElementById('active-profiles').textContent = data.active_profiles;

                // Update embedded docs count
                if (data.vector_store && data.vector_store.enabled) {
                    document.getElementById('embedded-docs').textContent = data.vector_store.total_documents + ' ✓';
                } else {
                    document.getElementById('embedded-docs').textContent = 'Disabled';
                }

                // Update awaiting OCR count
                const awaitingOcr = data.awaiting_ocr || 0;
                document.getElementById('awaiting-ocr-count').textContent = awaitingOcr;
                document.getElementById('awaiting-ocr-card').style.display = awaitingOcr > 0 ? '' : 'none';

                // Update awaiting AI count
                const awaitingAi = data.awaiting_ai || 0;
                document.getElementById('awaiting-ai-count').textContent = awaitingAi;
                document.getElementById('awaiting-ai-card').style.display = awaitingAi > 0 ? '' : 'none';

                // Update court imported count
                const courtCount = data.court_doc_count || 0;
                if (courtCount > 0) {
                    document.getElementById('court-imported-count').textContent = courtCount;
                    document.getElementById('court-imported-card').style.display = '';
                } else {
                    document.getElementById('court-imported-card').style.display = 'none';
                }

                // Format last update time
                if (data.last_update) {
                    const date = new Date(data.last_update);
                    document.getElementById('last-update').textContent = date.toLocaleTimeString();
                } else {
                    document.getElementById('last-update').textContent = 'Never';
                }
            } catch (error) {
                console.error('Failed to fetch status:', error);
            }
        }

        async function fetchRecent() {
            try {
                const response = await apiFetch(apiUrl('/api/recent'));
                const data = await response.json();

                const overviewContainer = document.getElementById('recent-analyses');

                if (data.analyses.length === 0) {
                    if (overviewContainer) overviewContainer.innerHTML = '<div class="empty-state">No analyses yet</div>';
                    return;
                }

                const analysesHtml = data.analyses.slice(-20).reverse().map(analysis => {
                    const hasAnomalies = analysis.anomalies_found && analysis.anomalies_found.length > 0;
                    const isHighRisk = analysis.risk_score >= 70;
                    const classes = ['recent-item'];
                    if (hasAnomalies) classes.push('has-anomalies');
                    if (isHighRisk) classes.push('high-risk');

                    let tags = '';
                    if (analysis.anomalies_found) {
                        // Check if this document has enhanced tags (issue count > 0 indicates detailed evidence)
                        const hasEvidence = analysis.issue_count > 0 || analysis.enhanced_tags;

                        tags = analysis.anomalies_found.map(a => {
                            // All anomaly tags are now clickable to show explanations
                            return `<span class="tag anomaly clickable" onclick="event.stopPropagation(); showTagEvidence(${analysis.document_id || analysis.doc_id}, '${(analysis.document_title || analysis.title).replace(/'/g, "\\'")}')" title="Click to see why this was flagged">🔍 ${a}</span>`;
                        }).join('');

                        // Add "View Evidence" button if document has integrity issues
                        if (hasEvidence) {
                            tags += ` <span class="tag clickable" style="background: #3498db; color: white;" onclick="event.stopPropagation(); showTagEvidence(${analysis.document_id || analysis.doc_id}, '${(analysis.document_title || analysis.title).replace(/'/g, "\\'")}');" title="Click to view all evidence">📋 View Evidence</span>`;
                        }
                    }

                    // Format summaries — always fully visible, no collapse
                    const briefSummary = analysis.brief_summary || '';
                    const fullSummary = analysis.full_summary || '';
                    const summaryHtml = briefSummary ? `
                        <div style="margin-top: 8px; padding: 8px 10px; background: rgba(52, 152, 219, 0.08); border-left: 3px solid #3498db; border-radius: 4px; font-size: 0.9em;">
                            <div style="color: #444; line-height: 1.5;"><strong>Brief Summary:</strong> ${briefSummary}</div>
                            ${fullSummary ? `
                                <div style="margin-top: 6px; color: #555; line-height: 1.5; border-top: 1px solid rgba(52,152,219,0.2); padding-top: 6px;"><strong>Full Summary:</strong> ${fullSummary}</div>
                            ` : ''}
                        </div>
                    ` : '';

                    return `
                        <div class="${classes.join(' ')}">
                            <div class="title">${analysis.document_title || analysis.title}</div>
                            <div class="meta">
                                <span>📄 ${(() => { const _did = analysis.document_id || analysis.doc_id; const _lnk = analysis.paperless_link || (_globalPaperlessBase ? `${_globalPaperlessBase}/documents/${_did}/details` : null); return _lnk ? `<a href="${_lnk}" target="_blank" style="color:#3498db;text-decoration:none;">Doc #${_did}</a>` : `<span>Doc #${_did}</span>`; })()}</span>
                                <span>⚠️ Risk: ${analysis.risk_score}%</span>
                                ${analysis.profile_matched ? `<span>📋 ${analysis.profile_matched}</span>` : '<span>❓ No profile</span>'}
                                <span>⏱️ ${new Date(analysis.timestamp).toLocaleString()}</span>
                            </div>
                            ${summaryHtml}
                            ${tags ? `<div style="margin-top: 8px;">${tags}</div>` : ''}
                        </div>
                    `;
                }).join('');

                // Always update overview recent-analyses panel
                if (overviewContainer) overviewContainer.innerHTML = analysesHtml;
            } catch (error) {
                console.error('Failed to fetch recent analyses:', error);
                const overviewContainer = document.getElementById('recent-analyses');
                if (overviewContainer) overviewContainer.innerHTML = '<div class="empty-state">Failed to load</div>';
            }
        }

        async function fetchProfiles() {
            try {
                const response = await apiFetch(apiUrl('/api/profiles'));
                const data = await response.json();

                // Active profiles
                const activeList = document.getElementById('active-profiles-list');
                if (data.active.length === 0) {
                    activeList.innerHTML = '<li class="empty-state">No active profiles</li>';
                } else {
                    activeList.innerHTML = data.active.map(profile => `
                        <li style="display: flex; justify-content: space-between; align-items: center; padding: 12px;">
                            <div>
                                <span class="profile-name">${profile.name}</span>
                                <span class="profile-version">v${profile.version}</span>
                                <div style="font-size: 11px; color: #999; margin-top: 4px;">${profile.filename}</div>
                            </div>
                            <div style="display: flex; gap: 8px;">
                                <button class="btn btn-small" onclick="viewActiveProfile('${profile.filename}')">👁️ View</button>
                                <button class="btn btn-small" onclick="renameActiveProfile('${profile.filename}', '${profile.name}')">✏️ Rename</button>
                                <button class="btn btn-small btn-danger" onclick="deleteActiveProfile('${profile.filename}')">✗ Delete</button>
                            </div>
                        </li>
                    `).join('');
                }

                // Staging profiles
                const stagingSection = document.getElementById('staging-section');
                const stagingContainer = document.getElementById('staging-profiles');
                if (data.staging.length > 0) {
                    if (stagingSection) stagingSection.style.display = 'block';
                    if (stagingContainer) {
                        stagingContainer.innerHTML = data.staging.map(profile => `
                            <div class="staging-profile">
                                <div class="filename">${profile.filename}</div>
                                <div class="actions" style="margin-top: 8px;">
                                    <span style="font-size: 12px; color: #666;">
                                        Created: ${new Date(profile.created).toLocaleString()}
                                    </span>
                                    <div style="margin-top: 8px; display: flex; gap: 10px;">
                                        <button class="btn btn-small" onclick="viewStagingProfile('${profile.filename}')">👁️ View</button>
                                        <button class="btn btn-small btn-success" onclick="activateStagingProfile('${profile.filename}')">✓ Activate</button>
                                        <button class="btn btn-small btn-danger" onclick="deleteStagingProfile('${profile.filename}')">✗ Delete</button>
                                    </div>
                                </div>
                            </div>
                        `).join('');
                    }
                } else {
                    if (stagingSection) stagingSection.style.display = 'none';
                }
            } catch (error) {
                console.error('Failed to fetch profiles:', error);
            }
        }

// ── System Health & Container Manager ──────────────────────────────────
        // ── System Health & Container Manager (v2.5.0) ───────────────────────

        // Silent background poll that updates only the header dots
        async function pollHeaderHealth() {
            try {
                const res = await apiFetch(apiUrl('/api/system-health'));
                if (!res.ok) return;
                const data = await res.json();
                _updateHeaderDots(data.components || {}, data.overall || 'warning');
            } catch (e) {
                // Silent — don't alert on background poll failures
            }
        }

        // Full health load for the Health sub-tab
        // manualRefresh=true → user clicked Refresh; reset the countdown timer
        async function loadSystemHealth(manualRefresh = false) {
            if (manualRefresh) _resetToolsCountdown('health');
            const container = document.getElementById('health-cards');
            if (!container) return;
            container.innerHTML = '<div class="loading">Checking components…</div>';
            try {
                const res = await apiFetch(apiUrl('/api/system-health'));
                const data = await res.json();
                _updateHeaderDots(data.components || {}, data.overall || 'warning');
                const labels = {
                    paperless_api: 'Paperless API',
                    chromadb: 'ChromaDB',
                    llm: 'LLM',
                    analyzer_loop: 'Analyzer Loop',
                    postgres: 'PostgreSQL',
                    redis: 'Redis',
                    projects_containers: 'Project Instances',
                };
                container.innerHTML = '';
                for (const [id, label] of Object.entries(labels)) {
                    const result = (data.components || {})[id] || {status: 'warning', latency_ms: 0, detail: 'No data'};
                    container.appendChild(_renderHealthCard(id, label, result));
                }
                // Overall status line
                const checkedAt = data.checked_at ? new Date(data.checked_at).toLocaleTimeString() : '';
                const overallDiv = document.createElement('div');
                overallDiv.style.cssText = 'grid-column:1/-1;font-size:12px;color:#888;margin-top:4px;';
                overallDiv.textContent = `Overall: ${data.overall || '?'} — checked at ${checkedAt}`;
                container.appendChild(overallDiv);
            } catch (e) {
                container.innerHTML = `<div style="color:#e74c3c;padding:12px;">Failed to load health data: ${e.message}</div>`;
            }
        }

        function _renderHealthCard(id, label, result) {
            const colors = {ok: '#27ae60', warning: '#f39c12', error: '#e74c3c'};
            const badgeBg = {ok: 'rgba(39,174,96,0.15)', warning: 'rgba(243,156,18,0.15)', error: 'rgba(231,76,60,0.15)'};
            const color = colors[result.status] || '#888';
            const card = document.createElement('div');
            card.style.cssText = `background:#fff;border-radius:8px;padding:14px 16px;border-left:4px solid ${color};box-shadow:0 1px 4px rgba(0,0,0,0.08);`;
            const latencyStr = result.latency_ms > 0 ? ` <span style="color:#999;font-size:11px;">${result.latency_ms}ms</span>` : '';
            card.innerHTML = `
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                    <span style="font-weight:600;font-size:14px;color:#333;">${label}</span>
                    <span style="background:${badgeBg[result.status]};color:${color};border:1px solid ${color};padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">${result.status}${latencyStr}</span>
                </div>
                <div style="font-size:13px;color:#555;">${result.detail || ''}</div>`;
            return card;
        }

        function _updateHeaderDots(components, overall) {
            for (const [comp, result] of Object.entries(components)) {
                const dot = document.querySelector(`[data-comp="${comp}"]`);
                if (!dot) continue;
                dot.className = 'hh-dot hh-' + (result.status || 'loading');
                const labels = {
                    paperless_api: 'Paperless API', chromadb: 'ChromaDB', llm: 'LLM',
                    analyzer_loop: 'Analyzer Loop', postgres: 'PostgreSQL', redis: 'Redis',
                    projects_containers: 'Project Instances',
                };
                dot.title = `${labels[comp] || comp}: ${result.status} — ${result.detail || ''}`;
            }
            const overallEl = document.getElementById('hh-overall');
            if (overallEl) {
                const overallColors = {ok: '#27ae60', warning: '#f39c12', error: '#e74c3c'};
                overallEl.textContent = overall === 'ok' ? 'All systems OK' : overall === 'error' ? 'System error' : 'Warning';
                overallEl.style.color = overallColors[overall] || '#aabbcc';
            }
        }

        // ── Container Manager ─────────────────────────────────────────────────

        // manualRefresh=true → user clicked Refresh; reset the countdown timer
        async function loadContainers(manualRefresh = false) {
            if (manualRefresh) _resetToolsCountdown('containers');
            const cardsEl = document.getElementById('container-cards');
            const unavailEl = document.getElementById('container-unavailable');
            if (!cardsEl) return;
            cardsEl.innerHTML = '<div class="loading">Loading containers…</div>';
            try {
                const res = await apiFetch(apiUrl('/api/containers'));
                const data = await res.json();
                if (!data.available) {
                    unavailEl && (unavailEl.style.display = 'block');
                    cardsEl.innerHTML = '';
                    return;
                }
                unavailEl && (unavailEl.style.display = 'none');
                cardsEl.innerHTML = '';
                (data.containers || []).forEach(c => cardsEl.appendChild(_renderContainerCard(c)));
            } catch (e) {
                cardsEl.innerHTML = `<div style="color:#e74c3c;padding:12px;">Failed to load containers: ${e.message}</div>`;
            }
        }

        function _renderContainerCard(c) {
            const statusColors = {running: '#27ae60', exited: '#e74c3c', not_found: '#888', error: '#e74c3c'};
            const dotColor = statusColors[c.status] || '#f39c12';
            const uptimeStr = c.uptime_seconds != null ? _fmtUptime(c.uptime_seconds) : '—';
            const row = document.createElement('div');
            row.id = `container-row-${c.name}`;
            row.style.cssText = 'background:#fff;border-radius:6px;padding:12px 16px;display:flex;align-items:center;gap:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08);flex-wrap:wrap;';
            const dot = `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${dotColor};flex-shrink:0;"></span>`;
            const imgStr = c.image ? `<span style="color:#888;font-size:12px;font-family:monospace;">${c.image}</span>` : '';
            const uptimePart = `<span style="color:#888;font-size:12px;">Up: ${uptimeStr}</span>`;
            let adminBtns = '';
            if (window.APP_CONFIG.isAdmin && c.status !== 'not_found') {
                adminBtns = `
                    <button class="btn btn-small" onclick="restartContainer('${c.name}')" style="font-size:12px;padding:4px 10px;">⟳ Restart</button>
                    <button class="btn btn-small" onclick="viewContainerLogs('${c.name}')" style="font-size:12px;padding:4px 10px;">📋 Logs</button>`;
            }
            row.innerHTML = `
                ${dot}
                <span style="font-weight:600;font-size:14px;color:#333;min-width:240px;">${c.name}</span>
                <span style="font-size:12px;padding:2px 8px;border-radius:10px;background:${dotColor}22;color:${dotColor};font-weight:600;">${c.status}</span>
                ${imgStr}
                ${uptimePart}
                <div style="margin-left:auto;display:flex;gap:6px;align-items:center;">${adminBtns}</div>
                <div id="container-msg-${c.name}" style="width:100%;font-size:12px;display:none;padding:4px 0;"></div>`;
            return row;
        }

        function _fmtUptime(s) {
            if (s < 60) return `${s}s`;
            const m = Math.floor(s / 60) % 60;
            const h = Math.floor(s / 3600) % 24;
            const d = Math.floor(s / 86400);
            if (d > 0) return `${d}d ${h}h`;
            if (h > 0) return `${h}h ${m}m`;
            return `${m}m`;
        }

        async function restartContainer(name) {
            if (!confirm(`Restart container "${name}"? It will be briefly unavailable.`)) return;
            const msgEl = document.getElementById(`container-msg-${name}`);
            if (msgEl) { msgEl.style.display = 'block'; msgEl.style.color = '#888'; msgEl.textContent = 'Restarting…'; }
            try {
                const res = await apiFetch(apiUrl(`/api/containers/${name}/restart`), {method: 'POST'});
                const data = await res.json();
                if (res.ok) {
                    if (msgEl) { msgEl.style.color = '#27ae60'; msgEl.textContent = `✓ ${data.message || 'Restarted'}`; }
                    setTimeout(loadContainers, 3000);
                } else {
                    if (msgEl) { msgEl.style.color = '#e74c3c'; msgEl.textContent = `✗ ${data.error || 'Failed'}`; }
                }
            } catch (e) {
                if (msgEl) { msgEl.style.color = '#e74c3c'; msgEl.textContent = `✗ ${e.message}`; }
            }
        }

        async function viewContainerLogs(name) {
            // Remove any existing log drawer for this container
            const existingDrawer = document.getElementById(`log-drawer-${name}`);
            if (existingDrawer) { existingDrawer.remove(); return; }

            const cardsEl = document.getElementById('container-cards');
            const drawer = document.createElement('div');
            drawer.id = `log-drawer-${name}`;
            drawer.className = 'logs-viewer';
            drawer.style.cssText = 'margin-top:8px;margin-bottom:8px;';
            drawer.innerHTML = `
                <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 12px;border-bottom:1px solid #e0e0e0;">
                    <strong style="font-size:13px;">Logs: ${name}</strong>
                    <button class="btn btn-small" onclick="document.getElementById('log-drawer-${name}').remove()" style="font-size:11px;">✕ Close</button>
                </div>
                <div id="log-drawer-lines-${name}" style="padding:10px 12px;font-size:12px;color:#aabbcc;"><em>Loading…</em></div>`;
            if (cardsEl) cardsEl.appendChild(drawer);

            try {
                const res = await apiFetch(apiUrl(`/api/containers/${name}/logs?lines=100`));
                const data = await res.json();
                const linesEl = document.getElementById(`log-drawer-lines-${name}`);
                if (!linesEl) return;
                if (!res.ok) { linesEl.innerHTML = `<span style="color:#e74c3c;">${data.error || 'Error fetching logs'}</span>`; return; }
                if (!data.lines || data.lines.length === 0) { linesEl.innerHTML = '<em style="color:#888;">No log lines returned.</em>'; return; }
                linesEl.innerHTML = data.lines.map(l =>
                    `<div style="white-space:pre-wrap;word-break:break-all;padding:1px 0;">${l.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`
                ).join('');
                linesEl.scrollTop = linesEl.scrollHeight;
            } catch (e) {
                const linesEl = document.getElementById(`log-drawer-lines-${name}`);
                if (linesEl) linesEl.innerHTML = `<span style="color:#e74c3c;">${e.message}</span>`;
            }
        }
