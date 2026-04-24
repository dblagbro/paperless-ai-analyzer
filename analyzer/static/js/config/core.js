// ── Configuration Sub-tabs & Vector Store ───────────────────────────────


        // ── Configuration sub-tab state ──────────────────────────────
        let vsAllDocs = [];       // flat array: {title, doc_id, type}
        let vsFiltered = [];
        let vsPage = 0;
        const VS_PAGE_SIZE = 25;
        let vsLoaded = false;

        function switchConfigTab(name) {
            document.querySelectorAll('.config-sub-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.config-sub-content').forEach(c => c.classList.remove('active'));
            const btn = document.getElementById('cfg-btn-' + name);
            const pane = document.getElementById('cfg-' + name);
            if (btn) btn.classList.add('active');
            if (pane) pane.classList.add('active');
            // Reset scroll — each active config pane is its own scroll container
            if (pane) pane.scrollTop = 0;
            // Lazy-load usage stats on each visit
            if (name === 'usage') { loadLlmUsage(); }
            // Lazy-load vector store on first visit
            if (name === 'vectorstore' && !vsLoaded) {
                loadVectorStoreDocuments();
            }
            // Lazy-load users on first visit (admin only)
            if (name === 'users' && typeof loadUsers === 'function') {
                loadUsers();
            }
            // Lazy-load LLM proxy endpoints on each visit (admin only)
            if (name === 'llm-proxy' && typeof loadLLMProxy === 'function') {
                loadLLMProxy();
            }
            // Update help panel to match the newly active config sub-tab
            _refreshHelpPanel();
        }

        // ── Tools sub-tab switcher ────────────────────────────────────
        // ── Tools sub-tab auto-refresh state ─────────────────────────
        let _toolsAutoRefreshTimer = null;
        const _toolsAutoRefreshIntervals = { health: 30, containers: 60 };

        function _startToolsAutoRefresh(subTab) {
            _stopToolsAutoRefresh();
            const secs = _toolsAutoRefreshIntervals[subTab];
            if (!secs) return;
            let remaining = secs;
            _setToolsCountdown(subTab, remaining);
            _toolsAutoRefreshTimer = setInterval(() => {
                remaining--;
                _setToolsCountdown(subTab, remaining);
                if (remaining <= 0) {
                    remaining = secs;
                    if (subTab === 'health') loadSystemHealth(false);
                    else if (subTab === 'containers') loadContainers(false);
                }
            }, 1000);
        }

        function _stopToolsAutoRefresh() {
            if (_toolsAutoRefreshTimer) { clearInterval(_toolsAutoRefreshTimer); _toolsAutoRefreshTimer = null; }
            document.querySelectorAll('[id^="tools-countdown-"]').forEach(el => el.textContent = '');
        }

        function _setToolsCountdown(subTab, secs) {
            const el = document.getElementById('tools-countdown-' + subTab);
            if (el) el.textContent = secs > 0 ? `Next refresh in ${secs}s` : 'Refreshing…';
        }

        function _resetToolsCountdown(subTab) {
            // Reset the visible countdown when user manually refreshes;
            // the interval keeps running from its current tick position but
            // the countdown display resets to the full interval.
            const secs = _toolsAutoRefreshIntervals[subTab];
            if (!secs) return;
            // Restart the whole interval so timing aligns with the fresh data
            _startToolsAutoRefresh(subTab);
        }

        function switchToolsTab(name) {
            document.querySelectorAll('.tools-sub-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tools-sub-content').forEach(c => c.classList.remove('active'));
            const btn = document.getElementById('tools-btn-' + name);
            const pane = document.getElementById('tools-' + name);
            if (btn) btn.classList.add('active');
            if (pane) { pane.classList.add('active'); pane.scrollTop = 0; }
            // Lazy-load on each visit
            if (name === 'health') { loadSystemHealth(false); }
            if (name === 'containers') { loadContainers(false); }
            if (name === 'logs') { refreshLogs(); }
            // Start auto-refresh for tabs that support it; stop for others
            if (_toolsAutoRefreshIntervals[name]) {
                _startToolsAutoRefresh(name);
            } else {
                _stopToolsAutoRefresh();
            }
            // Update help panel for this sub-tab
            _refreshHelpPanel();
        }

        // ── AI Usage / Cost ───────────────────────────────────────────
        async function loadLlmUsage() {
            const days = parseInt(document.getElementById('usage-days-select')?.value || '30');
            document.getElementById('usage-summary').innerHTML = '<div class="loading">Loading…</div>';

            try {
                const res = await apiFetch(apiUrl('/api/llm-usage/stats?days=' + days));
                const data = await res.json();
                if (data.error) { document.getElementById('usage-summary').innerHTML = '<div style="color:#e74c3c;">Error: ' + data.error + '</div>'; return; }

                const o = data.overall || {};
                const fmt$ = v => v != null ? '$' + (parseFloat(v)||0).toFixed(4) : '—';
                const fmtN = v => v != null ? (parseInt(v)||0).toLocaleString() : '—';

                // Summary cards
                document.getElementById('usage-summary').innerHTML = `
                    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:14px 16px;">
                        <div style="font-size:11px;color:#166534;font-weight:600;text-transform:uppercase;margin-bottom:4px;">Total Cost</div>
                        <div style="font-size:22px;font-weight:700;color:#15803d;">${fmt$(o.total_cost)}</div>
                        <div style="font-size:11px;color:#6b7280;margin-top:2px;">Last ${days} days</div>
                    </div>
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px 16px;">
                        <div style="font-size:11px;color:#1e40af;font-weight:600;text-transform:uppercase;margin-bottom:4px;">API Calls</div>
                        <div style="font-size:22px;font-weight:700;color:#1d4ed8;">${fmtN(o.total_calls)}</div>
                        <div style="font-size:11px;color:#6b7280;margin-top:2px;">${fmtN(o.successful_calls)} ok / ${fmtN(o.failed_calls)} failed</div>
                    </div>
                    <div style="background:#fefce8;border:1px solid #fde68a;border-radius:8px;padding:14px 16px;">
                        <div style="font-size:11px;color:#92400e;font-weight:600;text-transform:uppercase;margin-bottom:4px;">Total Tokens</div>
                        <div style="font-size:22px;font-weight:700;color:#b45309;">${fmtN(o.total_tokens)}</div>
                        <div style="font-size:11px;color:#6b7280;margin-top:2px;">in: ${fmtN(o.total_input_tokens)} / out: ${fmtN(o.total_output_tokens)}</div>
                    </div>`;

                // By model table
                const tblStyle = 'width:100%;border-collapse:collapse;font-size:13px;';
                const thStyle = 'text-align:left;padding:6px 10px;border-bottom:2px solid #e5e7eb;font-size:12px;color:#6b7280;font-weight:600;';
                const tdStyle = 'padding:6px 10px;border-bottom:1px solid #f3f4f6;';
                const byModel = data.per_model || [];
                if (byModel.length) {
                    document.getElementById('usage-by-model').innerHTML = `<table style="${tblStyle}">
                        <thead><tr><th style="${thStyle}">Model</th><th style="${thStyle}">Calls</th><th style="${thStyle}">Tokens</th><th style="${thStyle}">Cost</th></tr></thead>
                        <tbody>${byModel.map(m => `<tr>
                            <td style="${tdStyle}">${m.model||'—'}</td>
                            <td style="${tdStyle}">${fmtN(m.calls)}</td>
                            <td style="${tdStyle}">${fmtN(m.total_tokens)}</td>
                            <td style="${tdStyle};font-weight:600;">${fmt$(m.cost)}</td>
                        </tr>`).join('')}</tbody></table>`;
                } else {
                    document.getElementById('usage-by-model').innerHTML = '<div style="color:#888;font-size:13px;">No data for this period.</div>';
                }

                // By operation table
                const byOp = data.per_operation || [];
                if (byOp.length) {
                    document.getElementById('usage-by-op').innerHTML = `<table style="${tblStyle}">
                        <thead><tr><th style="${thStyle}">Operation</th><th style="${thStyle}">Calls</th><th style="${thStyle}">Tokens</th><th style="${thStyle}">Cost</th></tr></thead>
                        <tbody>${byOp.map(op => `<tr>
                            <td style="${tdStyle}">${op.operation||'—'}</td>
                            <td style="${tdStyle}">${fmtN(op.calls)}</td>
                            <td style="${tdStyle}">${fmtN(op.total_tokens)}</td>
                            <td style="${tdStyle};font-weight:600;">${fmt$(op.cost)}</td>
                        </tr>`).join('')}</tbody></table>`;
                } else {
                    document.getElementById('usage-by-op').innerHTML = '<div style="color:#888;font-size:13px;">No data for this period.</div>';
                }

                // Daily usage chart + table
                const daily = data.daily_usage || [];
                renderUsageChart(daily);
                if (daily.length) {
                    document.getElementById('usage-daily').innerHTML = `<table style="${tblStyle}">
                        <thead><tr><th style="${thStyle}">Date</th><th style="${thStyle}">Calls</th><th style="${thStyle}">Tokens</th><th style="${thStyle}">Cost</th></tr></thead>
                        <tbody>${daily.map(d => `<tr>
                            <td style="${tdStyle}">${d.date||'—'}</td>
                            <td style="${tdStyle}">${fmtN(d.calls)}</td>
                            <td style="${tdStyle}">${fmtN(d.tokens)}</td>
                            <td style="${tdStyle};font-weight:600;">${fmt$(d.cost)}</td>
                        </tr>`).join('')}</tbody></table>`;
                } else {
                    document.getElementById('usage-daily').innerHTML = '<div style="color:#888;font-size:13px;">No data for this period.</div>';
                }

                // Pricing reference
                const pricingRes = await apiFetch(apiUrl('/api/llm-usage/pricing'));
                const pricingData = await pricingRes.json();
                const pricing = pricingData.pricing || {};
                const pricingModels = Object.keys(pricing);
                if (pricingModels.length) {
                    document.getElementById('usage-pricing').innerHTML = `<table style="${tblStyle}">
                        <thead><tr><th style="${thStyle}">Model</th><th style="${thStyle}">Input / 1M tokens</th><th style="${thStyle}">Output / 1M tokens</th></tr></thead>
                        <tbody>${pricingModels.map(m => `<tr>
                            <td style="${tdStyle}">${m}</td>
                            <td style="${tdStyle}">$${(pricing[m].input||0).toFixed(2)}</td>
                            <td style="${tdStyle}">$${(pricing[m].output||0).toFixed(2)}</td>
                        </tr>`).join('')}</tbody></table>`;
                } else {
                    document.getElementById('usage-pricing').innerHTML = '<div style="color:#888;font-size:13px;">No pricing data available.</div>';
                }

            } catch(e) {
                document.getElementById('usage-summary').innerHTML = '<div style="color:#e74c3c;">Failed to load usage data: ' + e.message + '</div>';
            }
        }

        // ── Daily usage bar chart (Canvas) ────────────────────────────────
        function renderUsageChart(dailyData) {
            const canvas = document.getElementById('usage-chart');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            if (!rect.width) return;
            canvas.width  = rect.width  * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            const W = rect.width, H = rect.height;

            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, W, H);

            if (!dailyData || dailyData.length === 0) {
                ctx.fillStyle = '#9ca3af';
                ctx.font = '13px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('No data for this period', W / 2, H / 2);
                return;
            }

            const sorted = [...dailyData].sort((a, b) => a.date.localeCompare(b.date));
            const maxCost  = Math.max(...sorted.map(d => parseFloat(d.cost)  || 0), 0.00001);
            const maxCalls = Math.max(...sorted.map(d => parseInt(d.calls)   || 0), 1);

            const PAD_L = 58, PAD_R = 16, PAD_T = 22, PAD_B = 38;
            const cW = W - PAD_L - PAD_R, cH = H - PAD_T - PAD_B;
            const n  = sorted.length;

            // Grid + Y-axis (cost)
            const ySteps = 4;
            for (let i = 0; i <= ySteps; i++) {
                const y   = PAD_T + cH - (i / ySteps) * cH;
                const val = maxCost * i / ySteps;
                ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(PAD_L, y); ctx.lineTo(PAD_L + cW, y); ctx.stroke();
                ctx.fillStyle = '#6b7280'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
                ctx.fillText('$' + (val < 0.01 ? val.toFixed(4) : val < 0.1 ? val.toFixed(3) : val.toFixed(2)), PAD_L - 4, y + 4);
            }

            // X axis line
            ctx.strokeStyle = '#d1d5db'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(PAD_L, PAD_T + cH); ctx.lineTo(PAD_L + cW, PAD_T + cH); ctx.stroke();

            // Bars + labels
            const slotW = cW / n;
            const barW  = Math.max(slotW * 0.65, 3);
            const barOff = (slotW - barW) / 2;

            sorted.forEach((d, i) => {
                const cost  = parseFloat(d.cost)  || 0;
                const barH  = (cost / maxCost) * cH;
                const x     = PAD_L + i * slotW + barOff;
                const y     = PAD_T + cH - barH;

                // Bar gradient feel
                ctx.fillStyle = '#3b82f6';
                ctx.fillRect(x, y, barW, barH);
                if (barH > 4) { ctx.fillStyle = 'rgba(255,255,255,0.18)'; ctx.fillRect(x, y, barW, 4); }

                // Cost label above tall bars
                if (barH > 18 && cost > 0) {
                    ctx.fillStyle = '#1e40af'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
                    ctx.fillText('$' + cost.toFixed(3), x + barW / 2, y - 2);
                }

                // X-axis date label
                const label = d.date ? d.date.substring(5) : '';  // "MM-DD"
                ctx.fillStyle = '#6b7280'; ctx.textAlign = 'center';
                if (n > 20) {
                    ctx.font = '9px sans-serif';
                    ctx.save(); ctx.translate(x + barW/2, PAD_T + cH + 4); ctx.rotate(Math.PI/4);
                    ctx.fillText(label, 0, 0); ctx.restore();
                } else {
                    ctx.font = '10px sans-serif';
                    ctx.fillText(label, x + barW / 2, PAD_T + cH + 13);
                }
            });

            // Calls overlay line (secondary axis, dashed amber)
            ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 2]);
            ctx.beginPath();
            sorted.forEach((d, i) => {
                const calls = parseInt(d.calls) || 0;
                const cx = PAD_L + i * slotW + barOff + barW / 2;
                const cy = PAD_T + cH - (calls / maxCalls) * cH;
                i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
            });
            ctx.stroke(); ctx.setLineDash([]);

            // Legend
            ctx.fillStyle = '#3b82f6'; ctx.fillRect(PAD_L, 4, 10, 8);
            ctx.fillStyle = '#374151'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('Daily Cost', PAD_L + 13, 12);
            ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 2]);
            ctx.beginPath(); ctx.moveTo(PAD_L + 72, 8); ctx.lineTo(PAD_L + 84, 8); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#374151'; ctx.fillText('API Calls', PAD_L + 87, 12);
        }

        function filterVsDocs() {
            const q = (document.getElementById('vs-search')?.value || '').toLowerCase().trim();
            vsFiltered = q
                ? vsAllDocs.filter(d =>
                    (d.title || '').toLowerCase().includes(q) ||
                    String(d.doc_id).includes(q) ||
                    (d.type || '').toLowerCase().includes(q))
                : [...vsAllDocs];
            vsPage = 0;
            renderVsPage();
        }

        function renderVsPage() {
            const container = document.getElementById('vector-store-manager');
            const pagDiv = document.getElementById('vs-pagination');
            const countLabel = document.getElementById('vs-count-label');
            if (!container) return;

            const total = vsFiltered.length;
            const totalPages = Math.ceil(total / VS_PAGE_SIZE) || 1;
            if (countLabel) countLabel.textContent = total + ' document' + (total !== 1 ? 's' : '');

            if (total === 0) {
                container.innerHTML = '<p style="color:#999;">No documents match.</p>';
                if (pagDiv) pagDiv.style.display = 'none';
                return;
            }

            const start = vsPage * VS_PAGE_SIZE;
            const slice = vsFiltered.slice(start, start + VS_PAGE_SIZE);

            // Group current page slice by type for display
            const byType = {};
            slice.forEach(d => {
                if (!byType[d.type]) byType[d.type] = [];
                byType[d.type].push(d);
            });

            let html = '';
            Object.keys(byType).sort().forEach(type => {
                const docs = byType[type];
                html += `<div class="vector-type-section">
                    <div class="vector-type-header">
                        <h3>${escapeHtml(type)} <span class="vector-type-count">(${docs.length})</span></h3>
                        <button class="btn btn-small btn-warning" onclick="confirmDeleteType('${escapeHtml(type)}')">Delete All ${escapeHtml(type)}</button>
                    </div>
                    <div class="vector-documents-list">`;
                docs.forEach(doc => {
                    html += `<div class="vector-doc-item">
                        <div class="vector-doc-info">
                            <div class="vector-doc-title">${escapeHtml(doc.title || 'Untitled')}</div>
                            <div class="vector-doc-id">ID: ${doc.doc_id}</div>
                        </div>
                        <button class="btn btn-small btn-warning" onclick="deleteIndividualDocument(${doc.doc_id}, '${escapeHtml(doc.title || 'this document')}')">Delete</button>
                    </div>`;
                });
                html += `</div></div>`;
            });
            container.innerHTML = html;

            // Pagination controls
            if (pagDiv) {
                if (totalPages <= 1) {
                    pagDiv.style.display = 'none';
                } else {
                    pagDiv.style.display = 'flex';
                    let pagHtml = `<button class="btn btn-small" onclick="vsGoPage(${vsPage-1})" ${vsPage===0?'disabled':''}>◀ Prev</button>`;
                    // Show up to 7 page buttons
                    const maxBtns = 7, half = Math.floor(maxBtns/2);
                    let pStart = Math.max(0, vsPage - half);
                    let pEnd = Math.min(totalPages - 1, pStart + maxBtns - 1);
                    if (pEnd - pStart < maxBtns - 1) pStart = Math.max(0, pEnd - maxBtns + 1);
                    for (let p = pStart; p <= pEnd; p++) {
                        pagHtml += `<button class="btn btn-small${p===vsPage?' active':''}" onclick="vsGoPage(${p})">${p+1}</button>`;
                    }
                    pagHtml += `<button class="btn btn-small" onclick="vsGoPage(${vsPage+1})" ${vsPage>=totalPages-1?'disabled':''}>Next ▶</button>`;
                    pagHtml += `<span style="font-size:13px;color:#666;margin-left:6px;">Page ${vsPage+1} of ${totalPages}</span>`;
                    pagDiv.innerHTML = pagHtml;
                }
            }
        }

        function vsGoPage(p) {
            const totalPages = Math.ceil(vsFiltered.length / VS_PAGE_SIZE) || 1;
            vsPage = Math.max(0, Math.min(p, totalPages - 1));
            renderVsPage();
            // Scroll to top of the vector store pane when changing pages
            const vsPane = document.getElementById('cfg-vectorstore');
            if (vsPane) vsPane.scrollTop = 0;
        }



// ── SMTP (moved here from legacy config.js tail) ──────────────────────
// ── SMTP Settings ────────────────────────────────────────────────────────
        // ── SMTP Settings (admin) ─────────────────────────────────────────────
        function _ensureSmtpLoaded(subtab) {
            if (subtab === 'smtp' && !_smtpLoaded) {
                _smtpLoaded = true;
                loadSmtpSettings();
            }
        }
        // Wrap switchConfigTab to lazy-load SMTP settings
        (function() {
            var _orig = switchConfigTab;
            switchConfigTab = function(name) { _orig(name); _ensureSmtpLoaded(name); };
        })();
        function loadSmtpSettings() {
            apiFetch(apiUrl('/api/smtp-settings')).then(function(r) { return r.json(); }).then(function(d) {
                if (d.error) return;
                document.getElementById('smtp-host').value = d.host || '';
                document.getElementById('smtp-port').value = d.port || 587;
                document.getElementById('smtp-user').value = d.user || '';
                document.getElementById('smtp-pass').value = d.pass || '';
                document.getElementById('smtp-from').value = d.from || '';
                document.getElementById('smtp-helo').value = d.helo || '';
                document.getElementById('smtp-bug-report-to').value = d.bug_report_to || '';
                document.getElementById('smtp-starttls').checked = !!d.starttls;
            }).catch(function() {});
        }
        function _showSmtpResult(msg, ok) {
            var el = document.getElementById('smtp-result');
            el.style.display = 'block';
            el.style.background = ok ? 'rgba(39,174,96,0.15)' : 'rgba(231,76,60,0.15)';
            el.style.border = '1px solid ' + (ok ? 'rgba(39,174,96,0.4)' : 'rgba(231,76,60,0.4)');
            el.style.color = ok ? '#27ae60' : '#e74c3c';
            el.textContent = msg;
        }
        async function saveSmtpSettings() {
            var payload = {
                host: document.getElementById('smtp-host').value.trim(),
                port: parseInt(document.getElementById('smtp-port').value) || 587,
                user: document.getElementById('smtp-user').value.trim(),
                pass: document.getElementById('smtp-pass').value,
                from: document.getElementById('smtp-from').value.trim(),
                helo: document.getElementById('smtp-helo').value.trim(),
                bug_report_to: document.getElementById('smtp-bug-report-to').value.trim(),
                starttls: document.getElementById('smtp-starttls').checked,
            };
            try {
                var r = await apiFetch(apiUrl('/api/smtp-settings'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                var d = await r.json();
                _showSmtpResult(d.ok ? d.message : (d.error || 'Failed'), !!d.ok);
            } catch (e) {
                _showSmtpResult('Error: ' + e.message, false);
            }
        }
        async function testSmtpSettings() {
            _showSmtpResult('Sending test email…', true);
            try {
                var r = await apiFetch(apiUrl('/api/smtp-settings/test'), { method: 'POST' });
                var d = await r.json();
                _showSmtpResult(d.ok ? d.message : (d.error || 'Failed'), !!d.ok);
            } catch (e) {
                _showSmtpResult('Error: ' + e.message, false);
            }
        }
