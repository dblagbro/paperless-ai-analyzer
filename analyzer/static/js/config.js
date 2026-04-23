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


// ── Manage Projects ──────────────────────────────────────────────────────

        // ── Manage Projects ───────────────────────────────────────────────
        const PROJ_COLORS = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316','#64748b'];
        let _projEditSlug = null;   // null = new project, slug string = edit
        let _projCache = [];           // last fetched project list
        let _globalPaperlessBase = ''; // PAPERLESS_PUBLIC_BASE_URL from server

        /**
         * Build a public Paperless document URL for a given project slug + doc ID.
         * Uses paperless_doc_base_url from _projCache if available.
         * Returns null if no base URL is configured for the project.
         */
        function _paperlessDocUrl(slug, docId) {
            if (!docId) return null;
            const proj = _projCache.find(p => p.slug === slug);
            // Use per-project URL if configured, else fall back to global base
            const base = ((proj && proj.paperless_doc_base_url) ? proj.paperless_doc_base_url : _globalPaperlessBase).replace(/\/$/, '');
            return base ? `${base}/documents/${docId}/details` : null;
        }

        function _renderProjectCard(p) {
            const color    = p.color || '#3b82f6';
            const archived = p.is_archived;
            const cardId   = `proj-docs-${p.slug}`;
            return `
            <div style="background:#fff; border:1px solid #e5e7eb; border-left:4px solid ${color}; border-radius:8px; margin-bottom:12px; ${archived ? 'opacity:.6;' : ''}">
                <div id="provision-banner-${p.slug}" style="display:none; padding:8px 20px; font-size:12px; border-bottom:1px solid; border-radius:8px 8px 0 0;"></div>
                <div style="padding:18px 20px; display:flex; align-items:center; gap:16px;">
                    <div style="flex:1; min-width:0;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:3px;">
                            <span style="font-weight:700; font-size:15px; color:#111; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${p.name}</span>
                        </div>
                        <div style="font-size:13px; color:#6b7280; margin-bottom:6px;">${p.description || '<em style="color:#aaa;">No description</em>'}</div>
                        <div style="font-size:12px; color:#9ca3af;">
                            📄 <strong style="color:#374151;">${p.document_count}</strong> analyzed &nbsp;·&nbsp;
                            ${p.court_doc_count > 0 ? `🏛️ <strong style="color:#374151;">${p.court_doc_count}</strong> court &nbsp;·&nbsp; ` : ''}
                            💾 ${(p.storage_size_mb||0).toFixed(2)} MB &nbsp;·&nbsp;
                            📅 ${new Date(p.created_at).toLocaleDateString()} &nbsp;·&nbsp;
                            <span style="font-family:monospace; font-size:11px; background:#f3f4f6; padding:1px 6px; border-radius:4px;">${p.slug}</span>
                        </div>
                    </div>
                    <div style="display:flex; gap:6px; flex-shrink:0; align-items:center;">
                        ${!archived ? `<button onclick="openEditProjectModal('${p.slug}')" title="Edit"
                            style="padding:6px 11px; border:1px solid #d1d5db; border-radius:6px; background:#fff; cursor:pointer; font-size:13px;">✏️</button>
                        <button onclick="openMoveModal('${p.slug}')" title="Move documents"
                            style="padding:6px 11px; border:1px solid #d1d5db; border-radius:6px; background:#fff; cursor:pointer; font-size:13px;">📦 Move</button>` : ''}
                        ${archived
                            ? `<button onclick="toggleArchiveProject('${p.slug}', false)" title="Restore"
                                style="padding:6px 11px; border:1px solid #6b7280; border-radius:6px; background:#fff; cursor:pointer; font-size:13px;">📂 Restore</button>`
                            : `<button onclick="toggleArchiveProject('${p.slug}', true)" title="Archive"
                                style="padding:6px 11px; border:1px solid #d1d5db; border-radius:6px; background:#fff; cursor:pointer; font-size:13px;">🗄️ Archive</button>`}
                        <button onclick="analyzeProjectMissing('${p.slug}')" title="Scan for unanalyzed docs"
                            style="padding:6px 11px; border:1px solid #93c5fd; border-radius:6px; background:#fff; cursor:pointer; font-size:13px; color:#1d4ed8;">🔍 Analyze Missing</button>
                        <button onclick="openPaperlessModal('${p.slug}')" title="Configure Paperless instance"
                            style="padding:6px 11px; border:1px solid ${p.paperless_configured ? '#bbf7d0' : '#d1d5db'}; border-radius:6px; background:${p.paperless_configured ? '#f0fdf4' : '#fff'}; cursor:pointer; font-size:13px; color:${p.paperless_configured ? '#15803d' : '#374151'};">⚙️ Paperless</button>
                        ${p.slug !== 'default'
                            ? `<button onclick="openDelProjModal('${p.slug}', '${p.name.replace(/'/g,"\\'")}', ${p.document_count})" title="Delete project"
                                style="padding:6px 11px; border:1px solid #fca5a5; border-radius:6px; background:#fff; cursor:pointer; font-size:13px; color:#dc2626;">🗑️</button>`
                            : ''}
                        <button onclick="toggleProjDocs('${p.slug}')" id="proj-toggle-${p.slug}"
                            title="Show/hide documents"
                            style="padding:6px 11px; border:1px solid #d1d5db; border-radius:6px; background:#f9fafb; cursor:pointer; font-size:13px;">
                            📋 Docs ▼
                        </button>
                    </div>
                </div>
                <div id="${cardId}" style="display:none; border-top:1px solid #e5e7eb; padding:0 20px 16px 20px;">
                    <div id="${cardId}-inner" style="padding-top:12px; font-size:13px; color:#374151;">
                        <em style="color:#9ca3af;">Loading…</em>
                    </div>
                </div>
            </div>`;
        }

        async function loadProjectsList() {
            const list = document.getElementById('projects-list');
            try {
                const res = await apiFetch(apiUrl('/api/projects'));
                if (!res.ok) throw new Error('HTTP ' + res.status);
                const data = await res.json();
                _projCache = data.projects || [];
                _globalPaperlessBase = (data.global_paperless_base_url || '').replace(/\/$/, '');

                const active   = _projCache.filter(p => !p.is_archived);
                const archived = _projCache.filter(p =>  p.is_archived);

                if (!_projCache.length) {
                    list.innerHTML = '<div style="text-align:center; padding:40px; color:#999;">No projects found. Click <strong>+ New Project</strong> to get started.</div>';
                    return;
                }

                let html = active.map(_renderProjectCard).join('');

                if (archived.length) {
                    html += `
                    <div style="margin-top:32px; margin-bottom:12px; display:flex; align-items:center; gap:10px;">
                        <span style="font-weight:700; font-size:14px; color:#6b7280; text-transform:uppercase; letter-spacing:.05em;">🗄️ Archived (${archived.length})</span>
                        <div style="flex:1; height:1px; background:#e5e7eb;"></div>
                    </div>
                    ${archived.map(_renderProjectCard).join('')}`;
                }

                list.innerHTML = html;

                // Restore card banners for any actively-polling slugs
                Object.keys(_cardProvisionTimers).forEach(slug => {
                    apiFetch(apiUrl(`/api/projects/${slug}/provision-status`))
                        .then(r => r.json())
                        .then(d => _updateCardBanner(slug, d))
                        .catch(() => {});
                });
            } catch (e) {
                list.innerHTML = `<div style="color:#e74c3c; padding:20px; text-align:center;"><strong>Error loading projects:</strong><br>${e.message}</div>`;
            }
        }

        // ── New / Edit modal ───────────────────────────────────────────

        function _buildColorSwatches(selectedColor) {
            const wrap = document.getElementById('proj-color-swatches');
            wrap.innerHTML = PROJ_COLORS.map(c =>
                `<div onclick="selectSwatch('${c}')" style="width:26px; height:26px; border-radius:50%; background:${c}; cursor:pointer; outline:${c === selectedColor ? '3px solid #1d4ed8' : '2px solid transparent'}; outline-offset:2px; transition:outline .1s;" data-color="${c}"></div>`
            ).join('');
        }

        function selectSwatch(color) {
            document.getElementById('proj-color-val').value = color;
            document.querySelectorAll('#proj-color-swatches div').forEach(el => {
                el.style.outline = el.dataset.color === color ? '3px solid #1d4ed8' : '2px solid transparent';
            });
        }

        function selectCustomColor(color) {
            document.getElementById('proj-color-val').value = color;
            document.querySelectorAll('#proj-color-swatches div').forEach(el => {
                el.style.outline = '2px solid transparent';
            });
        }

        function autoSlug() {
            if (_projEditSlug) return;  // don't overwrite slug when editing
            const name = document.getElementById('proj-name').value;
            const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
            document.getElementById('proj-slug').value = slug;
        }

        function openNewProjectModal() {
            _projEditSlug = null;
            document.getElementById('proj-modal-title').textContent = 'New Project';
            document.getElementById('proj-name').value = '';
            document.getElementById('proj-slug').value = '';
            document.getElementById('proj-slug').disabled = false;
            document.getElementById('proj-desc').value = '';
            document.getElementById('proj-modal-error').style.display = 'none';
            const defaultColor = PROJ_COLORS[Math.floor(Math.random() * PROJ_COLORS.length)];
            document.getElementById('proj-color-val').value = defaultColor;
            _buildColorSwatches(defaultColor);
            document.getElementById('proj-modal').style.display = 'flex';
            document.getElementById('proj-name').focus();
        }

        function openEditProjectModal(slug) {
            const p = _projCache.find(x => x.slug === slug);
            if (!p) return;
            _projEditSlug = slug;
            document.getElementById('proj-modal-title').textContent = 'Edit Project';
            document.getElementById('proj-name').value = p.name;
            document.getElementById('proj-slug').value = p.slug;
            document.getElementById('proj-slug').disabled = true;
            document.getElementById('proj-desc').value = p.description || '';
            document.getElementById('proj-modal-error').style.display = 'none';
            const color = p.color || PROJ_COLORS[0];
            document.getElementById('proj-color-val').value = color;
            _buildColorSwatches(color);
            document.getElementById('proj-modal').style.display = 'flex';
            document.getElementById('proj-name').focus();
        }

        function closeProjModal() {
            document.getElementById('proj-modal').style.display = 'none';
        }

        async function saveProject() {
            const name = document.getElementById('proj-name').value.trim();
            const slug = document.getElementById('proj-slug').value.trim();
            const description = document.getElementById('proj-desc').value.trim();
            const color = document.getElementById('proj-color-val').value;
            const errDiv = document.getElementById('proj-modal-error');
            const btn = document.getElementById('proj-save-btn');

            if (!name) { errDiv.textContent = 'Name is required.'; errDiv.style.display = ''; return; }
            if (!_projEditSlug && !slug) { errDiv.textContent = 'Slug is required.'; errDiv.style.display = ''; return; }
            if (!_projEditSlug && !/^[a-z0-9-]+$/.test(slug)) {
                errDiv.textContent = 'Slug must contain only lowercase letters, numbers, and hyphens.';
                errDiv.style.display = ''; return;
            }

            btn.disabled = true; errDiv.style.display = 'none';

            try {
                let res;
                if (_projEditSlug) {
                    res = await apiFetch(apiUrl(`/api/projects/${_projEditSlug}`), {
                        method: 'PUT',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({name, description, color})
                    });
                } else {
                    res = await apiFetch(apiUrl('/api/projects'), {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({name, slug, description, color})
                    });
                }
                if (!res.ok) {
                    const data = await res.json();
                    throw new Error(data.error || 'Request failed');
                }
                closeProjModal();
                const data = await res.json();
                await loadProjectsList();
                loadProjectSelector();
                // If new project was created, start background card-level provision polling
                if (!_projEditSlug && data.slug) {
                    showToast('Project created — provisioning Paperless instance in the background…', 'info', 6000);
                    _startCardProvisionPoll(data.slug);
                }
            } catch (e) {
                errDiv.textContent = e.message;
                errDiv.style.display = '';
            } finally {
                btn.disabled = false;
            }
        }

        // ── Archive / Unarchive ────────────────────────────────────────

        async function toggleArchiveProject(slug, archive) {
            const action = archive ? 'archive' : 'unarchive';
            try {
                const res = await apiFetch(apiUrl(`/api/projects/${slug}/${action}`), {method: 'POST'});
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Request failed');
                loadProjectsList();
                loadProjectSelector();
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        // ── Analyze Missing ────────────────────────────────────────────

        async function analyzeProjectMissing(slug) {
            try {
                const res = await apiFetch(apiUrl(`/api/projects/${slug}/analyze-missing`), {method: 'POST'});
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Request failed');
                showToast('Scanning for unanalyzed documents… check logs for progress.', 'info');
            } catch (e) {
                showToast('Error: ' + e.message, 'error');
            }
        }

        // ── Configure Paperless modal (v3.6.0) ────────────────────────────

        let _plCurrentSlug = null;
        let _migratePollerTimer = null;

        function showPaperlessTab(tab) {
            ['provision', 'connect', 'migrate'].forEach(t => {
                document.getElementById('pl-pane-' + t).style.display = t === tab ? 'block' : 'none';
                const btn = document.getElementById('pl-tab-' + t);
                btn.style.borderBottomColor = t === tab ? '#3b82f6' : 'transparent';
                btn.style.color = t === tab ? '#3b82f6' : '#6b7280';
                btn.style.fontWeight = t === tab ? '600' : '500';
            });
        }

        async function openPaperlessModal(slug) {
            _plCurrentSlug = slug;
            document.getElementById('paperless-modal-slug').textContent = slug;
            document.querySelectorAll('.pl-slug-inline').forEach(el => el.textContent = slug);
            document.getElementById('pl-connect-status').style.display = 'none';
            showPaperlessTab('provision');

            // Load provision snippets
            try {
                const r = await apiFetch(apiUrl(`/api/projects/${slug}/provision-snippets`));
                const d = await r.json();
                document.getElementById('pl-compose-snippet').textContent = d.compose || '';
                document.getElementById('pl-nginx-snippet').textContent = d.nginx || '';
                document.getElementById('pl-postgres-snippet').textContent = d.postgres_sql || '';
                document.getElementById('pl-web-svc-name').textContent = d.web_service || '';
                document.getElementById('pl-consumer-svc-name').textContent = d.consumer_service || '';
            } catch(e) {
                document.getElementById('pl-compose-snippet').textContent = 'Error loading snippets: ' + e.message;
            }

            // Load existing connect config
            try {
                const r2 = await apiFetch(apiUrl(`/api/projects/${slug}/paperless-config`));
                const d2 = await r2.json();
                document.getElementById('pl-url-input').value = d2.url || '';
                document.getElementById('pl-token-input').value = '';
                document.getElementById('pl-token-input').placeholder = d2.token_set ? '(token saved — leave blank to keep)' : 'Paste API token here';
                document.getElementById('pl-base-url-input').value = d2.doc_base_url || '';
            } catch(e) { /* ignore */ }

            document.getElementById('paperless-modal').style.display = 'flex';

            // Resume polling if provisioning is in progress
            try {
                const rs = await apiFetch(apiUrl(`/api/projects/${slug}/provision-status`));
                const ds = await rs.json();
                const st = ds.status || 'idle';
                if (st === 'running' || st === 'queued') {
                    _showProvisionStatus('⏳ ' + (ds.phase || 'Provisioning…'), 'running');
                    const btn = document.getElementById('pl-autoprovision-btn');
                    if (btn) { btn.disabled = true; btn.textContent = '⏳ Provisioning…'; }
                    startProvisionPoll(slug, () => { loadProjectsList(); loadProjectSelector(); });
                } else if (st === 'complete') {
                    _showProvisionStatus('✅ Provisioned — Paperless is running. <a href="' + (ds.doc_base_url || '#') + '" target="_blank" style="color:inherit;font-weight:600;">Open →</a>', 'success');
                } else if (st === 'error') {
                    _showProvisionStatus('❌ Last provision failed: ' + (ds.error || ''), 'error');
                }
            } catch(e) { /* ignore */ }

            // Resume migration progress bar if a migration is already running
            try {
                const rm = await apiFetch(apiUrl(`/api/projects/${slug}/migration-status`));
                const dm = await rm.json();
                if (dm.status === 'running') {
                    const statusWrap = document.getElementById('pl-migrate-status-wrap');
                    const resultEl   = document.getElementById('pl-migrate-result');
                    const btn        = document.getElementById('pl-migrate-btn');
                    const total    = dm.total || 0;
                    const migrated = dm.migrated || 0;
                    const pct      = total > 0 ? Math.round(migrated / total * 100) : 0;
                    statusWrap.style.display = 'block';
                    resultEl.style.display = 'none';
                    btn.disabled = true;
                    document.getElementById('pl-migrate-bar').style.width = pct + '%';
                    document.getElementById('pl-migrate-status-text').textContent =
                        `Migrating… ${migrated}/${total} docs (${pct}%)`;
                    document.getElementById('pl-migrate-count').textContent = dm.phase || '';
                    if (!_migratePollerTimer) {
                        _migratePollerTimer = setInterval(() => _pollMigrationStatus(slug), 2000);
                    }
                } else if (dm.status === 'done') {
                    document.getElementById('pl-migrate-status-wrap').style.display = 'block';
                    document.getElementById('pl-migrate-bar').style.width = '100%';
                    document.getElementById('pl-migrate-status-text').textContent = '✅ Migration complete';
                    const m = dm.migrated || 0, f = dm.failed || 0;
                    document.getElementById('pl-migrate-count').textContent =
                        `${m} docs moved${f > 0 ? `, ${f} failed` : ''}`;
                }
            } catch(e) { /* ignore */ }
        }

        function closePaperlessModal() {
            document.getElementById('paperless-modal').style.display = 'none';
            if (_migratePollerTimer) { clearInterval(_migratePollerTimer); _migratePollerTimer = null; }
            if (_provisionPollerTimer) { clearInterval(_provisionPollerTimer); _provisionPollerTimer = null; }
        }

        // ── Auto-provisioning (v3.6.1) ─────────────────────────────────────────────
        let _provisionPollerTimer = null;
        const _cardProvisionTimers = {};  // slug -> intervalId

        // Update the provision-status banner on a project card (outside any modal)
        function _updateCardBanner(slug, d) {
            const el = document.getElementById(`provision-banner-${slug}`);
            if (!el) return;
            const st = d.status || 'idle';
            if (st === 'idle') { el.style.display = 'none'; return; }
            el.style.display = 'block';
            if (st === 'queued' || st === 'running') {
                el.style.background = '#fefce8'; el.style.borderColor = '#fde047'; el.style.color = '#92400e';
                el.innerHTML = `<span style="animation:spin 1s linear infinite;display:inline-block;">⏳</span> Provisioning Paperless instance… <em style="font-weight:500;">${d.phase || ''}</em>`;
            } else if (st === 'complete') {
                el.style.background = '#f0fdf4'; el.style.borderColor = '#86efac'; el.style.color = '#15803d';
                el.innerHTML = `✅ Paperless instance ready — <a href="${d.doc_base_url || '#'}" target="_blank" style="color:inherit;font-weight:600;">Open Paperless →</a>`;
            } else if (st === 'error') {
                el.style.background = '#fef2f2'; el.style.borderColor = '#fca5a5'; el.style.color = '#dc2626';
                el.innerHTML = `❌ Provisioning failed: ${d.error || 'Unknown error'} — <button onclick="openPaperlessModal('${slug}')" style="background:none;border:none;color:inherit;text-decoration:underline;cursor:pointer;font-size:inherit;">Retry in modal</button>`;
            }
        }

        function _startCardProvisionPoll(slug) {
            if (_cardProvisionTimers[slug]) clearInterval(_cardProvisionTimers[slug]);
            _cardProvisionTimers[slug] = setInterval(async () => {
                try {
                    const r = await apiFetch(apiUrl(`/api/projects/${slug}/provision-status`));
                    const d = await r.json();
                    _updateCardBanner(slug, d);
                    const st = d.status || 'idle';
                    if (st === 'complete' || st === 'error' || st === 'idle') {
                        clearInterval(_cardProvisionTimers[slug]);
                        delete _cardProvisionTimers[slug];
                        if (st === 'complete') {
                            // Refresh card after short delay so doc counts etc update
                            setTimeout(() => { loadProjectsList(); loadProjectSelector(); }, 1500);
                        }
                    }
                } catch(e) { /* ignore transient */ }
            }, 3000);
        }

        function _showProvisionStatus(msg, type) {
            const el = document.getElementById('pl-provision-status');
            if (!el) return;
            el.style.display = 'block';
            const colors = {
                info:    {bg:'#eff6ff', border:'#bfdbfe', text:'#1d4ed8'},
                success: {bg:'#f0fdf4', border:'#bbf7d0', text:'#15803d'},
                error:   {bg:'#fef2f2', border:'#fecaca', text:'#dc2626'},
                running: {bg:'#fefce8', border:'#fef08a', text:'#92400e'},
            };
            const c = colors[type] || colors.info;
            el.style.background = c.bg;
            el.style.border = `1px solid ${c.border}`;
            el.style.color = c.text;
            el.innerHTML = msg;
        }

        function startProvisionPoll(slug, onComplete) {
            if (_provisionPollerTimer) clearInterval(_provisionPollerTimer);
            _provisionPollerTimer = setInterval(async () => {
                try {
                    const r = await apiFetch(apiUrl(`/api/projects/${slug}/provision-status`));
                    const d = await r.json();
                    const st = d.status || 'idle';
                    if (st === 'idle') {
                        clearInterval(_provisionPollerTimer);
                        _provisionPollerTimer = null;
                        return;
                    }
                    if (st === 'complete') {
                        clearInterval(_provisionPollerTimer);
                        _provisionPollerTimer = null;
                        _showProvisionStatus('✅ Provisioning complete! Paperless instance is ready. <a href="' + (d.doc_base_url || '#') + '" target="_blank" style="color:inherit;font-weight:600;">Open Paperless →</a>', 'success');
                        const btn = document.getElementById('pl-autoprovision-btn');
                        if (btn) { btn.disabled = false; btn.textContent = '⚡ Re-Provision'; }
                        if (onComplete) onComplete(d);
                    } else if (st === 'error') {
                        clearInterval(_provisionPollerTimer);
                        _provisionPollerTimer = null;
                        _showProvisionStatus('❌ Provisioning failed: ' + (d.error || 'Unknown error'), 'error');
                        const btn = document.getElementById('pl-autoprovision-btn');
                        if (btn) { btn.disabled = false; btn.textContent = '⚡ Retry Provision'; }
                    } else {
                        _showProvisionStatus('⏳ ' + (d.phase || 'Working…'), 'running');
                    }
                } catch(e) { /* ignore transient errors */ }
            }, 3000);
        }

        async function triggerReprovision() {
            const slug = _plCurrentSlug;
            if (!slug) return;
            const btn = document.getElementById('pl-autoprovision-btn');
            if (btn) { btn.disabled = true; btn.textContent = '⏳ Starting…'; }
            _showProvisionStatus('⏳ Queuing provisioning…', 'running');
            try {
                const r = await apiFetch(apiUrl(`/api/projects/${slug}/reprovision`), {method: 'POST'});
                const d = await r.json();
                if (!r.ok) throw new Error(d.error || 'Request failed');
                // Poll inside the modal
                startProvisionPoll(slug, () => {
                    loadProjectsList();
                    loadProjectSelector();
                });
                // Also poll on the card (visible after modal close)
                _startCardProvisionPoll(slug);
            } catch(e) {
                _showProvisionStatus('❌ ' + e.message, 'error');
                if (btn) { btn.disabled = false; btn.textContent = '⚡ Auto-Provision Now'; }
            }
        }

        function copySnippet(elemId) {
            const text = document.getElementById(elemId).textContent;
            navigator.clipboard.writeText(text).then(() => showToast('Copied to clipboard', 'success', 2000));
        }

        async function savePaperlessConfig() {
            const slug = _plCurrentSlug;
            const url  = document.getElementById('pl-url-input').value.trim();
            const tok  = document.getElementById('pl-token-input').value.trim();
            const base = document.getElementById('pl-base-url-input').value.trim();
            const statusEl = document.getElementById('pl-connect-status');
            statusEl.style.display = 'block';
            statusEl.style.color = '#6b7280';
            statusEl.textContent = 'Saving…';
            try {
                const body = { url, doc_base_url: base };
                if (tok) body.token = tok;
                const r = await apiFetch(apiUrl(`/api/projects/${slug}/paperless-config`), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                const d = await r.json();
                if (!r.ok) throw new Error(d.error || 'Save failed');
                statusEl.style.color = '#15803d';
                statusEl.textContent = '✅ Saved! Polling thread will pick up new credentials within 5 minutes.';
                // Refresh project list so button colour updates
                loadProjectsList();
            } catch(e) {
                statusEl.style.color = '#dc2626';
                statusEl.textContent = '❌ ' + e.message;
            }
        }

        async function testPaperlessConnection() {
            const url = document.getElementById('pl-url-input').value.trim();
            const tok = document.getElementById('pl-token-input').value.trim();
            // token_set: true means a token is already saved server-side — allow testing without re-entering
            const tokenSaved = document.getElementById('pl-token-input').placeholder.includes('token saved');
            const statusEl = document.getElementById('pl-connect-status');
            if (!url || (!tok && !tokenSaved)) {
                statusEl.style.display = 'block';
                statusEl.style.color = '#d97706';
                statusEl.textContent = '⚠️ Enter the URL (and token if not already saved) to test.';
                return;
            }
            statusEl.style.display = 'block';
            statusEl.style.color = '#6b7280';
            statusEl.textContent = 'Testing connection…';
            try {
                const body = {url};
                if (tok) body.token = tok;  // only send if user entered a new one; server uses saved token otherwise
                const r = await apiFetch(apiUrl(`/api/projects/${_plCurrentSlug}/paperless-health-check`), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                const d = await r.json();
                if (d.ok) {
                    statusEl.style.color = '#15803d';
                    statusEl.textContent = '✅ Connection OK — ' + (d.message || 'Paperless responded');
                } else {
                    statusEl.style.color = '#dc2626';
                    statusEl.textContent = '❌ ' + (d.error || 'Connection failed');
                }
            } catch(e) {
                statusEl.style.color = '#dc2626';
                statusEl.textContent = '❌ ' + e.message;
            }
        }

        async function startMigration() {
            const slug = _plCurrentSlug;
            if (!slug) return;

            const btn = document.getElementById('pl-migrate-btn');
            const statusWrap = document.getElementById('pl-migrate-status-wrap');
            const resultEl   = document.getElementById('pl-migrate-result');
            const prereqEl   = document.getElementById('pl-migrate-prereq');

            btn.disabled = true;
            prereqEl.style.display = 'none';
            resultEl.style.display = 'none';
            statusWrap.style.display = 'block';
            document.getElementById('pl-migrate-status-text').textContent = 'Starting migration…';
            document.getElementById('pl-migrate-count').textContent = '';
            document.getElementById('pl-migrate-bar').style.width = '0%';

            try {
                const r = await apiFetch(apiUrl(`/api/projects/${slug}/migrate-to-own-paperless`), { method: 'POST' });
                const d = await r.json();
                if (!r.ok) {
                    if (r.status === 409) {
                        document.getElementById('pl-migrate-status-text').textContent = 'Migration already running';
                    } else if (d.error && d.error.includes('No dedicated')) {
                        statusWrap.style.display = 'none';
                        prereqEl.style.display = 'block';
                        btn.disabled = false;
                        return;
                    } else {
                        throw new Error(d.error || 'Start failed');
                    }
                }
                // Start polling for status
                _migratePollerTimer = setInterval(() => _pollMigrationStatus(slug), 2000);
            } catch(e) {
                statusWrap.style.display = 'none';
                resultEl.style.display = 'block';
                resultEl.style.color = '#dc2626';
                resultEl.textContent = '❌ ' + e.message;
                btn.disabled = false;
            }
        }

        async function _pollMigrationStatus(slug) {
            try {
                const r = await apiFetch(apiUrl(`/api/projects/${slug}/migration-status`));
                const d = await r.json();
                const statusWrap = document.getElementById('pl-migrate-status-wrap');
                const resultEl   = document.getElementById('pl-migrate-result');
                const btn = document.getElementById('pl-migrate-btn');

                const total    = d.total || 0;
                const migrated = d.migrated || 0;
                const failed   = d.failed || 0;
                const pct      = total > 0 ? Math.round(migrated / total * 100) : 0;

                document.getElementById('pl-migrate-bar').style.width = pct + '%';
                document.getElementById('pl-migrate-status-text').textContent =
                    d.status === 'done' ? '✅ Migration complete' :
                    d.status === 'error' ? '❌ Migration failed' :
                    `Migrating… (${d.phase || ''})`;
                document.getElementById('pl-migrate-count').textContent =
                    total > 0 ? `${migrated}/${total} docs${failed > 0 ? `, ${failed} failed` : ''}` : '';

                if (d.status === 'done' || d.status === 'error') {
                    if (_migratePollerTimer) { clearInterval(_migratePollerTimer); _migratePollerTimer = null; }
                    btn.disabled = false;
                    if (d.status === 'done') {
                        resultEl.style.display = 'block';
                        resultEl.style.color = '#15803d';
                        resultEl.textContent =
                            `✅ Migration complete: ${migrated} document(s) moved to dedicated instance.` +
                            (failed > 0 ? ` (${failed} failed — check logs)` : '');
                        loadProjectsList();
                    } else {
                        resultEl.style.display = 'block';
                        resultEl.style.color = '#dc2626';
                        resultEl.textContent = '❌ ' + (d.error || 'Migration error — check logs');
                    }
                }
            } catch(e) {
                // Ignore transient errors during polling
            }
        }

        // ── Delete modal ───────────────────────────────────────────────

        function openDelProjModal(slug, name, docCount) {
            document.getElementById('del-proj-slug').value = slug;
            document.getElementById('del-proj-name').textContent = name;
            document.getElementById('del-proj-count').textContent = docCount;
            document.getElementById('del-proj-data').checked = true;
            document.getElementById('del-proj-error').style.display = 'none';
            document.getElementById('del-proj-modal').style.display = 'flex';
        }

        function closeDelProjModal() {
            document.getElementById('del-proj-modal').style.display = 'none';
        }

        async function confirmDeleteProject() {
            const slug = document.getElementById('del-proj-slug').value;
            const deleteData = document.getElementById('del-proj-data').checked;
            const errDiv = document.getElementById('del-proj-error');
            try {
                const res = await apiFetch(apiUrl(`/api/projects/${slug}?delete_data=${deleteData}`), {method: 'DELETE'});
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Request failed');
                closeDelProjModal();
                loadProjectsList();
                loadProjectSelector();
            } catch (e) {
                errDiv.textContent = e.message;
                errDiv.style.display = '';
            }
        }

        // ── Move Documents modal ────────────────────────────────────────

        function _populateMoveSelects(sourceSlug) {
            const active = _projCache.filter(p => !p.is_archived);
            const opts = active.map(p => `<option value="${p.slug}">${p.name} (${p.document_count} docs)</option>`).join('');
            const src = document.getElementById('move-source');
            const dst = document.getElementById('move-dest');
            src.innerHTML = opts;
            dst.innerHTML = opts;
            if (sourceSlug) src.value = sourceSlug;
            // Default dest to something different
            const destSlug = active.find(p => p.slug !== src.value)?.slug;
            if (destSlug) dst.value = destSlug;
        }

        function openMoveModal(sourceSlug) {
            _populateMoveSelects(sourceSlug);
            document.querySelector('input[name="move-scope"][value="all"]').checked = true;
            document.getElementById('move-ids-wrap').style.display = 'none';
            document.getElementById('move-doc-ids').value = '';
            document.getElementById('move-modal-error').style.display = 'none';
            document.getElementById('move-modal-status').style.display = 'none';
            document.getElementById('move-confirm-btn').disabled = false;
            updateMovePreview();
            document.getElementById('move-modal').style.display = 'flex';
        }

        function closeMoveModal() {
            document.getElementById('move-modal').style.display = 'none';
        }

        function toggleMoveIds() {
            const specific = document.querySelector('input[name="move-scope"][value="specific"]').checked;
            document.getElementById('move-ids-wrap').style.display = specific ? '' : 'none';
        }

        function updateMovePreview() {
            const srcSlug = document.getElementById('move-source').value;
            const srcProj = _projCache.find(p => p.slug === srcSlug);
            const preview = document.getElementById('move-preview');
            if (srcProj) {
                document.getElementById('move-all-count').textContent = srcProj.document_count;
                preview.textContent = `📦 Up to ${srcProj.document_count} document(s) will be re-tagged in Paperless. This runs in the background.`;
                preview.style.display = '';
            } else {
                preview.style.display = 'none';
            }
        }

        async function confirmMove() {
            const srcSlug = document.getElementById('move-source').value;
            const dstSlug = document.getElementById('move-dest').value;
            const specific = document.querySelector('input[name="move-scope"][value="specific"]').checked;
            const errDiv = document.getElementById('move-modal-error');
            const statusDiv = document.getElementById('move-modal-status');
            const btn = document.getElementById('move-confirm-btn');

            if (srcSlug === dstSlug) {
                errDiv.textContent = 'Source and destination must be different projects.';
                errDiv.style.display = ''; return;
            }

            let docIds = [];
            if (specific) {
                const raw = document.getElementById('move-doc-ids').value.trim();
                if (!raw) { errDiv.textContent = 'Enter at least one document ID.'; errDiv.style.display = ''; return; }
                docIds = raw.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
                if (!docIds.length) { errDiv.textContent = 'No valid document IDs found.'; errDiv.style.display = ''; return; }
            }

            btn.disabled = true;
            errDiv.style.display = 'none';
            statusDiv.style.display = 'none';

            try {
                const res = await apiFetch(apiUrl('/api/projects/migrate-documents'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        source_project: srcSlug,
                        destination_project: dstSlug,
                        document_ids: docIds
                    })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Request failed');
                // Close dialog immediately and show a toast so the user gets clear feedback
                closeMoveModal();
                btn.textContent = 'Move Documents';
                btn.onclick = confirmMove;
                btn.disabled = false;
                loadProjectsList();
                loadProjectSelector();
                const msg = (data.message || 'Migration started.') + (data.note ? ' — ' + data.note : '');
                showToast('✅ ' + msg, 'success', 5000);
            } catch (e) {
                errDiv.textContent = e.message;
                errDiv.style.display = '';
                btn.disabled = false;
            }
        }

        // ── Project document expand panel ─────────────────────────────────
        const _projDocsLoaded = {};

        async function toggleProjDocs(slug) {
            const panel = document.getElementById(`proj-docs-${slug}`);
            const btn   = document.getElementById(`proj-toggle-${slug}`);
            const inner = document.getElementById(`proj-docs-${slug}-inner`);
            if (panel.style.display === 'none') {
                panel.style.display = '';
                btn.textContent = '📋 Docs ▲';
                if (!_projDocsLoaded[slug]) {
                    inner.innerHTML = '<em style="color:#9ca3af;">Loading…</em>';
                    await _loadProjDocs(slug);
                }
            } else {
                panel.style.display = 'none';
                btn.textContent = '📋 Docs ▼';
            }
        }

        async function _loadProjDocs(slug) {
            const inner = document.getElementById(`proj-docs-${slug}-inner`);
            try {
                const res  = await apiFetch(apiUrl(`/api/projects/${slug}/documents`));
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Request failed');
                const docs = data.documents || [];
                _projDocsLoaded[slug] = true;
                if (!docs.length) {
                    inner.innerHTML = '<p style="color:#9ca3af;text-align:center;padding:8px 0;">No analyzed documents in this project yet.</p>';
                    return;
                }
                const rows = docs.map((d, idx) => {
                    const brief    = _escHtml(d.brief_summary || '—');
                    const full     = d.full_summary || '';
                    const fullShort = _escHtml(full.length > 120 ? full.slice(0, 120) + '…' : full) || '—';
                    const fullFull  = _escHtml(full) || '—';
                    const date     = d.timestamp ? new Date(d.timestamp).toLocaleDateString() : '—';
                    const rowId    = `doc-row-${slug}-${d.doc_id}`;
                    const fullId   = `doc-full-${slug}-${d.doc_id}`;
                    return `
                    <tr id="${rowId}" style="border-bottom:1px solid #f3f4f6;">
                        <td style="padding:6px 8px;white-space:nowrap;color:#9ca3af;">${idx + 1}</td>
                        <td style="padding:6px 8px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${_escHtml(d.title)}">${_escHtml(d.title)}</td>
                        <td style="padding:6px 8px;white-space:nowrap;color:#6b7280;">${date}</td>
                        <td style="padding:6px 8px;max-width:180px;color:#374151;">${brief}</td>
                        <td style="padding:6px 8px;max-width:220px;color:#374151;">
                            <span id="${fullId}-short">${fullShort}</span>
                            <span id="${fullId}-full" style="display:none;">${fullFull}</span>
                            ${full.length > 120 ? `<button onclick="toggleDocFull('${fullId}')" id="${fullId}-btn"
                                style="margin-left:4px;font-size:11px;color:#3b82f6;background:none;border:none;cursor:pointer;padding:0;">more</button>` : ''}
                        </td>
                        <td style="padding:6px 8px;white-space:nowrap;">
                            ${d.paperless_link ? `<a href="${d.paperless_link}" target="_blank" title="Open in Paperless"
                                style="display:inline-block;padding:3px 7px;border:1px solid #bfdbfe;border-radius:5px;background:#eff6ff;color:#1d4ed8;font-size:12px;text-decoration:none;margin-right:4px;">↗ View</a>` : ''}
                            <button onclick="confirmDeleteDoc('${slug}', ${d.doc_id}, ${_escHtml(JSON.stringify(d.title))})"
                                style="padding:3px 8px;border:1px solid #fca5a5;border-radius:5px;background:#fff;cursor:pointer;font-size:12px;color:#dc2626;">
                                🗑️ Delete
                            </button>
                        </td>
                    </tr>`;
                }).join('');
                inner.innerHTML = `
                <div style="overflow-x:auto;">
                    <table style="width:100%;border-collapse:collapse;font-size:13px;">
                        <thead>
                            <tr style="background:#f9fafb;text-align:left;">
                                <th style="padding:5px 8px;font-weight:600;color:#374151;">#</th>
                                <th style="padding:5px 8px;font-weight:600;color:#374151;">Title</th>
                                <th style="padding:5px 8px;font-weight:600;color:#374151;">Date</th>
                                <th style="padding:5px 8px;font-weight:600;color:#374151;">Brief Summary</th>
                                <th style="padding:5px 8px;font-weight:600;color:#374151;">Full Summary</th>
                                <th style="padding:5px 8px;font-weight:600;color:#374151;">Actions</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                </div>`;
            } catch (e) {
                inner.innerHTML = `<p style="color:#e74c3c;padding:8px 0;">Error: ${e.message}</p>`;
            }
        }

        function toggleDocFull(fullId) {
            const s   = document.getElementById(fullId + '-short');
            const f   = document.getElementById(fullId + '-full');
            const btn = document.getElementById(fullId + '-btn');
            const showing = f.style.display === '';
            s.style.display = showing ? '' : 'none';
            f.style.display = showing ? 'none' : '';
            btn.textContent = showing ? 'more' : 'less';
        }

        async function confirmDeleteDoc(slug, docId, title) {
            if (!confirm(`Delete "${title}" (Doc #${docId})?\n\nThis permanently removes the document from:\n• Paperless-ngx\n• The AI vector store\n• The analyzer database\n\nThis cannot be undone.`)) return;
            try {
                const res  = await apiFetch(apiUrl(`/api/projects/${slug}/documents/${docId}`), {method: 'DELETE'});
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Request failed');
                const row = document.getElementById(`doc-row-${slug}-${docId}`);
                if (row) row.remove();
                delete _projDocsLoaded[slug];
                if (data.warnings && data.warnings.length) {
                    showToast('⚠️ Partial delete: ' + data.warnings.join('; '), 'warning', 7000);
                } else {
                    showToast(`✅ Document #${docId} deleted`, 'success', 3000);
                }
                loadProjectsList();
            } catch (e) {
                alert('Delete failed: ' + e.message);
            }
        }

        function _escHtml(str) {
            if (!str) return '';
            return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
        }

// ── Search & Analysis Tab ───────────────────────────────────────────────

        // ── Search & Analysis tab — state ────────────────────────────────────────
        var _saDocs = [];
        var _saSortField = 'doc_id';
        var _saSortDir = 'asc';
        var _saLoaded = false;
        var _saFiltered = [];   // current filtered+sorted result set
        var _saPage = 0;        // current page (0-based)
        var _saPerPage = 25;    // items per page; 0 = all

        // Load (or reload) all docs for current project into the table
        async function loadSearchTab() {
            if (_saLoaded) return;
            const tbody = document.getElementById('sa-tbody');
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:30px;color:#9ca3af;">Loading…</td></tr>';
            try {
                const slug = document.getElementById('project-selector').value || 'default';
                const resp = await apiFetch(apiUrl(`/api/projects/${encodeURIComponent(slug)}/documents`));
                const data = await resp.json();
                _saDocs = data.documents || [];
                _saLoaded = true;
                saApplyFilters();
            } catch (err) {
                console.error('loadSearchTab error:', err);
                tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:30px;color:#e74c3c;">Failed to load documents</td></tr>';
            }
        }

        // Apply all active filters + sort, reset to page 0, then render
        function saApplyFilters() {
            const global  = (document.getElementById('sa-global')?.value || '').toLowerCase();
            const fDocId  = (document.getElementById('sa-filter-doc_id')?.value || '').toLowerCase();
            const fTitle  = (document.getElementById('sa-filter-title')?.value || '').toLowerCase();
            const fDate   = (document.getElementById('sa-filter-date')?.value || '').toLowerCase();
            const rMin    = parseFloat(document.getElementById('sa-filter-risk-min')?.value) || null;
            const rMax    = parseFloat(document.getElementById('sa-filter-risk-max')?.value) || null;
            const fAnom   = (document.getElementById('sa-filter-anomalies')?.value || '').toLowerCase();
            const fSum    = (document.getElementById('sa-filter-summary')?.value || '').toLowerCase();
            const anomOnly= document.getElementById('sa-anomalies-only')?.checked || false;

            // Read sort from dropdown
            const sortSel = document.getElementById('sa-sort-select');
            if (sortSel) _saSortField = sortSel.value;

            let docs = _saDocs.filter(d => {
                const title    = (d.title || '').toLowerCase();
                const docId    = String(d.doc_id || '');
                const dateStr  = d.timestamp ? new Date(d.timestamp).toLocaleString().toLowerCase() : '';
                const risk     = d.risk_score || 0;
                const anomStr  = (d.anomalies || []).join(' ').toLowerCase();
                const brief    = (d.brief_summary || '').toLowerCase();
                const full     = (d.full_summary || '').toLowerCase();
                const summaryAll = brief + ' ' + full;

                if (anomOnly && (!d.anomalies || d.anomalies.length === 0)) return false;
                if (rMin !== null && risk < rMin) return false;
                if (rMax !== null && risk > rMax) return false;
                if (fDocId  && !docId.includes(fDocId))     return false;
                if (fTitle  && !title.includes(fTitle))     return false;
                if (fDate   && !dateStr.includes(fDate))    return false;
                if (fAnom   && !anomStr.includes(fAnom))    return false;
                if (fSum    && !summaryAll.includes(fSum))  return false;
                if (global  && !(title + ' ' + docId + ' ' + anomStr + ' ' + summaryAll).includes(global)) return false;
                return true;
            });

            // Sort
            docs.sort((a, b) => {
                let av, bv;
                if (_saSortField === 'doc_id')         { av = a.doc_id || 0;               bv = b.doc_id || 0; }
                else if (_saSortField === 'title')      { av = (a.title||'').toLowerCase(); bv = (b.title||'').toLowerCase(); }
                else if (_saSortField === 'timestamp')  { av = a.timestamp || '';           bv = b.timestamp || ''; }
                else if (_saSortField === 'risk_score') { av = a.risk_score || 0;           bv = b.risk_score || 0; }
                else if (_saSortField === 'anomalies')  { av = (a.anomalies||[]).length;    bv = (b.anomalies||[]).length; }
                else { av = 0; bv = 0; }
                if (av < bv) return _saSortDir === 'asc' ? -1 : 1;
                if (av > bv) return _saSortDir === 'asc' ?  1 : -1;
                return 0;
            });

            updateSaSortIndicators();
            _saFiltered = docs;
            _saPage = 0;
            saRenderPage();
        }

        // Render the current page from _saFiltered
        function saRenderPage() {
            const total = _saFiltered.length;
            let pageDocs;
            if (_saPerPage === 0) {
                pageDocs = _saFiltered;
            } else {
                const start = _saPage * _saPerPage;
                pageDocs = _saFiltered.slice(start, start + _saPerPage);
            }
            renderSearchTable(pageDocs);
            saRenderPagination(total);

            const cnt = document.getElementById('sa-count');
            if (cnt) {
                if (total === 0) {
                    cnt.textContent = `0 of ${_saDocs.length} doc${_saDocs.length !== 1 ? 's' : ''}`;
                } else if (_saPerPage === 0) {
                    cnt.textContent = `${total} of ${_saDocs.length} doc${_saDocs.length !== 1 ? 's' : ''}`;
                } else {
                    const s = _saPage * _saPerPage + 1;
                    const e = Math.min((_saPage + 1) * _saPerPage, total);
                    cnt.textContent = `${s}–${e} of ${total} (${_saDocs.length} total)`;
                }
            }
        }

        // Render pagination bar
        function saRenderPagination(total) {
            const pag = document.getElementById('sa-pagination');
            const info = document.getElementById('sa-page-info');
            const controls = document.getElementById('sa-page-controls');
            if (!pag) return;

            if (total === 0 || _saPerPage === 0) {
                pag.style.display = 'none';
                return;
            }

            const totalPages = Math.ceil(total / _saPerPage);
            pag.style.display = 'flex';
            const s = _saPage * _saPerPage + 1;
            const e = Math.min((_saPage + 1) * _saPerPage, total);
            if (info) info.textContent = `Showing ${s}–${e} of ${total} documents`;

            // Build page buttons
            const btnBase = 'padding:4px 9px; border:1px solid #d1d5db; border-radius:4px; cursor:pointer; font-size:12px; background:#fff; color:#374151;';
            const btnActive = 'padding:4px 9px; border:1px solid #3b82f6; border-radius:4px; cursor:pointer; font-size:12px; background:#3b82f6; color:#fff; font-weight:600;';
            const btnDis = 'padding:4px 9px; border:1px solid #e5e7eb; border-radius:4px; font-size:12px; background:#f9fafb; color:#9ca3af; cursor:default;';

            let pages = [];
            if (totalPages <= 9) {
                for (let i = 0; i < totalPages; i++) pages.push(i);
            } else {
                pages.push(0);
                if (_saPage > 3) pages.push('…');
                for (let i = Math.max(1, _saPage - 2); i <= Math.min(totalPages - 2, _saPage + 2); i++) pages.push(i);
                if (_saPage < totalPages - 4) pages.push('…');
                pages.push(totalPages - 1);
            }

            let html = `<button onclick="saGoPage(${_saPage - 1})" ${_saPage === 0 ? 'disabled' : ''} style="${_saPage === 0 ? btnDis : btnBase}">‹ Prev</button>`;
            pages.forEach(p => {
                if (p === '…') {
                    html += `<span style="padding:4px 6px; font-size:12px; color:#9ca3af; align-self:center;">…</span>`;
                } else {
                    html += `<button onclick="saGoPage(${p})" style="${p === _saPage ? btnActive : btnBase}">${p + 1}</button>`;
                }
            });
            html += `<button onclick="saGoPage(${_saPage + 1})" ${_saPage >= totalPages - 1 ? 'disabled' : ''} style="${_saPage >= totalPages - 1 ? btnDis : btnBase}">Next ›</button>`;
            if (controls) controls.innerHTML = html;
        }

        // Navigate to a specific page
        function saGoPage(p) {
            const totalPages = _saPerPage === 0 ? 1 : Math.ceil(_saFiltered.length / _saPerPage);
            if (p < 0 || p >= totalPages) return;
            _saPage = p;
            saRenderPage();
            // Scroll table back to top
            const wrap = document.querySelector('#tab-search [style*="overflow-x"]');
            if (wrap) wrap.scrollTop = 0;
        }

        // Change items per page
        function saSetPerPage(val) {
            _saPerPage = parseInt(val) || 0;
            _saPage = 0;
            saRenderPage();
        }

        // Change sort field; toggle direction if same field clicked again
        function saSortBy(field) {
            if (_saSortField === field) {
                _saSortDir = _saSortDir === 'asc' ? 'desc' : 'asc';
            } else {
                _saSortField = field;
                _saSortDir = 'asc';
            }
            // Sync dropdown
            const sel = document.getElementById('sa-sort-select');
            if (sel && ['doc_id','title','timestamp','risk_score'].includes(field)) sel.value = field;
            saApplyFilters();
        }

        // Toggle asc/desc via the direction button
        function saToggleSortDir() {
            _saSortDir = _saSortDir === 'asc' ? 'desc' : 'asc';
            saApplyFilters();
        }

        // Update ▲/▼ indicators on column headers
        function updateSaSortIndicators() {
            const fields = ['doc_id','title','timestamp','risk_score','anomalies'];
            fields.forEach(f => {
                const el = document.getElementById('sa-sort-ind-' + f);
                if (!el) return;
                if (f === _saSortField) {
                    el.textContent = _saSortDir === 'asc' ? ' ▲' : ' ▼';
                } else {
                    el.textContent = '';
                }
            });
            // Sync dir button label
            const btn = document.getElementById('sa-sort-dir-btn');
            if (btn) btn.textContent = _saSortDir === 'asc' ? '▲ Asc' : '▼ Desc';
        }

        // Render filtered+sorted docs into the table body
        function renderSearchTable(docs) {
            const tbody = document.getElementById('sa-tbody');
            if (!docs.length) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:30px;color:#9ca3af;">' +
                    (_saLoaded ? 'No documents match the current filters.' : 'Loading…') + '</td></tr>';
                return;
            }
            tbody.innerHTML = docs.map((d, idx) => {
                const risk = d.risk_score || 0;
                const riskColor = risk >= 70 ? '#dc2626' : risk >= 40 ? '#d97706' : '#16a34a';
                const anomList = (d.anomalies || []);
                const anomHtml = anomList.length
                    ? anomList.map(a => `<span style="display:inline-block;background:#fef2f2;color:#991b1b;border:1px solid #fecaca;border-radius:3px;padding:1px 5px;margin:1px;font-size:11px;">${_escHtml(a)}</span>`).join('')
                    : '<span style="color:#9ca3af;font-size:12px;">—</span>';

                const brief = d.brief_summary || '';
                const full  = d.full_summary  || '';
                const fullId = 'sa-full-' + idx;
                let summaryHtml;
                if (!brief) {
                    summaryHtml = '<span style="color:#9ca3af;font-size:12px;">—</span>';
                } else if (!full) {
                    summaryHtml = `<span style="font-size:12px;">${_escHtml(brief)}</span>`;
                } else {
                    const briefTrunc = brief.length > 80 ? brief.slice(0, 80) + '…' : brief;
                    summaryHtml = `<span style="font-size:12px;">${_escHtml(briefTrunc)}</span>` +
                        ` <span id="${fullId}" style="display:none;font-size:12px;color:#374151;">${_escHtml(full)}</span>` +
                        ` <button onclick="toggleSaFull('${fullId}')" style="font-size:11px;padding:1px 5px;border:1px solid #d1d5db;border-radius:3px;background:#f9fafb;cursor:pointer;margin-left:3px;" id="${fullId}-btn">more</button>`;
                }

                const dateStr = d.timestamp ? new Date(d.timestamp).toLocaleDateString() : '—';
                const rowBg = anomList.length ? '#fff8f8' : (idx % 2 === 0 ? '#fff' : '#f9fafb');
                return `<tr style="background:${rowBg}; border-bottom:1px solid #e5e7eb;">
                    <td style="padding:7px 8px; white-space:nowrap; font-size:12px; text-align:center;">
                        ${d.paperless_link
                            ? `<a href="${d.paperless_link}" target="_blank" style="color:#3b82f6;text-decoration:none;">#${d.doc_id}</a>`
                            : `<span style="color:#6b7280;">#${d.doc_id}</span>`}
                    </td>
                    <td style="padding:7px 10px; font-size:13px; word-wrap:break-word; overflow-wrap:break-word;">${_escHtml(d.title || '—')}</td>
                    <td style="padding:7px 8px; white-space:nowrap; font-size:12px; color:#6b7280;">${dateStr}</td>
                    <td style="padding:7px 8px; white-space:nowrap; font-weight:600; color:${riskColor}; font-size:13px; text-align:center;">${risk}%</td>
                    <td style="padding:7px 10px; word-wrap:break-word; overflow-wrap:break-word;">${anomHtml}</td>
                    <td style="padding:7px 10px;">${summaryHtml}</td>
                </tr>`;
            }).join('');
        }

        // Toggle full summary visibility
        function toggleSaFull(fullId) {
            const el  = document.getElementById(fullId);
            const btn = document.getElementById(fullId + '-btn');
            if (!el) return;
            const hidden = el.style.display === 'none';
            el.style.display  = hidden ? 'inline' : 'none';
            if (btn) btn.textContent = hidden ? 'less' : 'more';
        }

        // Clear all filters and reload
        function saClearAll() {
            ['sa-global','sa-filter-doc_id','sa-filter-title','sa-filter-date',
             'sa-filter-risk-min','sa-filter-risk-max','sa-filter-anomalies','sa-filter-summary'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = '';
            });
            const anom = document.getElementById('sa-anomalies-only');
            if (anom) anom.checked = false;
            _saPage = 0;
            saApplyFilters();
        }

// ── LLM Settings & Profiles ─────────────────────────────────────────────

        // Check LLM status and show setup panel if needed
        async function checkLLMStatus() {
            try {
                const response = await apiFetch(apiUrl('/api/llm/status'));
                const data = await response.json();

                if (!data.enabled || !data.has_key) {
                    document.getElementById('llm-setup-panel').style.display = 'block';
                    document.getElementById('llm-setup-link').href = data.setup_url;
                }

                // Update provider dropdown change handler (element only exists on Config tab)
                const llmProviderEl = document.getElementById('llm-provider');
                if (llmProviderEl) {
                    llmProviderEl.value = data.provider;
                    llmProviderEl.addEventListener('change', function(e) {
                        const links = {
                            'anthropic': 'https://console.anthropic.com/settings/keys',
                            'openai': 'https://platform.openai.com/api-keys'
                        };
                        const setupLink = document.getElementById('llm-setup-link');
                        if (setupLink) setupLink.href = links[e.target.value];
                    });
                }
            } catch (error) {
                console.error('Failed to check LLM status:', error);
            }
        }

        // Test LLM API key
        async function testLLMKey() {
            const apiKey = document.getElementById('llm-api-key').value.trim();
            const provider = document.getElementById('llm-provider').value;
            const resultDiv = document.getElementById('llm-test-result');
            const saveBtn = document.getElementById('save-llm-btn');

            if (!apiKey) {
                resultDiv.style.display = 'block';
                resultDiv.style.background = '#fff3cd';
                resultDiv.style.color = '#856404';
                resultDiv.textContent = '⚠️ Please paste your API key first';
                saveBtn.style.display = 'none';
                return;
            }

            resultDiv.style.display = 'block';
            resultDiv.style.background = '#e7f3ff';
            resultDiv.style.color = '#004085';
            resultDiv.textContent = '🔄 Testing API key...';
            saveBtn.style.display = 'none';

            try {
                const response = await apiFetch(apiUrl('/api/llm/test'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({provider, api_key: apiKey})
                });

                const data = await response.json();

                if (data.success) {
                    resultDiv.style.background = '#d4edda';
                    resultDiv.style.color = '#155724';
                    resultDiv.textContent = data.message;
                    saveBtn.style.display = 'inline-block';
                } else {
                    resultDiv.style.background = '#f8d7da';
                    resultDiv.style.color = '#721c24';
                    resultDiv.textContent = data.error;
                    saveBtn.style.display = 'none';
                }
            } catch (error) {
                resultDiv.style.background = '#f8d7da';
                resultDiv.style.color = '#721c24';
                resultDiv.textContent = '✗ Failed to test key: ' + error.message;
                saveBtn.style.display = 'none';
            }
        }

        // View staging profile
        async function viewStagingProfile(filename) {
            try {
                const response = await apiFetch(apiUrl(`/api/staging/${filename}`));
                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Format YAML nicely
                const yaml = JSON.stringify(data, null, 2);
                const formatted = yaml.replace(/[{}"]/g, '').replace(/,\n/g, '\n');

                alert(`Profile: ${filename}\n\n${formatted}\n\nTo activate, click the ✓ Activate button.`);
            } catch (error) {
                alert('Failed to load profile: ' + error.message);
            }
        }

        // Activate staging profile
        async function activateStagingProfile(filename) {
            if (!confirm(`Activate profile "${filename}"?\n\nThis will move it to active profiles and restart the analyzer.`)) {
                return;
            }

            try {
                const response = await fetch(`api/staging/${filename}/activate`, {
                    method: 'POST'
                });
                const data = await response.json();

                if (data.success) {
                    alert('✅ ' + data.message);
                    setTimeout(() => location.reload(), 2000);
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed: ' + error.message);
            }
        }

        // Activate all staging profiles
        async function activateAllStagingProfiles() {
            if (!confirm('Activate ALL staging profiles?\n\nThis will move all staging profiles to active and require an analyzer restart.')) {
                return;
            }

            try {
                const response = await apiFetch(apiUrl('/api/staging/activate-all'), {
                    method: 'POST'
                });
                const data = await response.json();

                if (data.success) {
                    const msg = `✅ ${data.message}\n\nActivated: ${data.activated}\nFailed: ${data.failed}`;
                    alert(msg);
                    setTimeout(() => location.reload(), 2000);
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed: ' + error.message);
            }
        }

        // Delete staging profile
        async function deleteStagingProfile(filename) {
            if (!confirm(`Delete profile "${filename}"?\n\nThis cannot be undone.`)) {
                return;
            }

            try {
                const response = await fetch(`api/staging/${filename}/delete`, {
                    method: 'POST'
                });
                const data = await response.json();

                if (data.success) {
                    alert('✅ Profile deleted');
                    fetchProfiles();
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed: ' + error.message);
            }
        }

        // Active profile management
        async function viewActiveProfile(filename) {
            try {
                const response = await apiFetch(apiUrl(`/api/active/${filename}`));
                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                const yaml = JSON.stringify(data, null, 2);
                const formatted = yaml.replace(/[{}"]/g, '').replace(/,\n/g, '\n');
                alert(`Active Profile: ${filename}\n\n${formatted}`);
            } catch (error) {
                alert('❌ Failed: ' + error.message);
            }
        }

        async function renameActiveProfile(filename, currentName) {
            const newName = prompt(`Rename profile "${currentName}":\n\nEnter new display name:`, currentName);
            if (!newName || newName === currentName) {
                return;
            }

            try {
                const response = await fetch(`api/active/${filename}/rename`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({display_name: newName})
                });
                const data = await response.json();

                if (data.success) {
                    alert('✅ Profile renamed! Reloading profiles...');
                    fetchProfiles();
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed: ' + error.message);
            }
        }

        async function deleteActiveProfile(filename) {
            if (!confirm(`Delete active profile "${filename}"?\n\nThis will stop using this profile for matching. This cannot be undone.`)) {
                return;
            }

            try {
                const response = await fetch(`api/active/${filename}/delete`, {
                    method: 'POST'
                });
                const data = await response.json();

                if (data.success) {
                    alert('✅ Profile deleted! Restart the analyzer to reload profiles.');
                    fetchProfiles();
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed: ' + error.message);
            }
        }

        // Detect duplicate profiles
        async function detectDuplicates() {
            const resultDiv = document.getElementById('duplicates-result');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<div class="loading">Analyzing profiles for duplicates...</div>';

            try {
                const response = await apiFetch(apiUrl('/api/active/duplicates'));
                const data = await response.json();

                if (data.duplicate_groups === 0) {
                    resultDiv.innerHTML = '<div style="padding: 15px; background: #d4edda; color: #155724; border-radius: 4px;">✅ No duplicates found! All profiles are unique.</div>';
                    return;
                }

                // Build duplicate groups display
                let html = `<div style="padding: 15px; background: #fff3cd; color: #856404; border-radius: 4px; margin-bottom: 15px;">
                    ⚠️ Found ${data.duplicate_groups} duplicate group(s) affecting ${data.groups.reduce((sum, g) => sum + g.profiles.length, 0)} profiles
                </div>`;

                data.groups.forEach((group, idx) => {
                    html += `<div style="border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 15px; background: #f8f9fa;">
                        <strong>Duplicate Group ${idx + 1}</strong> (${group.type})
                        <div style="font-size: 12px; color: #666; margin: 5px 0;">${group.reason}</div>
                        <ul style="margin: 10px 0; padding-left: 20px;">`;

                    group.profiles.forEach(profile => {
                        html += `<li>
                            <input type="checkbox" class="duplicate-checkbox" value="${profile.filename}" style="margin-right: 8px;">
                            <strong>${profile.filename}</strong> - ${profile.display_name}
                        </li>`;
                    });

                    html += `</ul></div>`;
                });

                html += `<div style="margin-top: 15px; display: flex; gap: 10px;">
                    <button class="btn btn-danger" onclick="removeDuplicates()">🗑️ Remove Selected</button>
                    <button class="btn btn-secondary" onclick="selectAllDuplicates()">Select All</button>
                    <button class="btn btn-secondary" onclick="deselectAllDuplicates()">Deselect All</button>
                </div>`;

                resultDiv.innerHTML = html;
            } catch (error) {
                resultDiv.innerHTML = `<div style="padding: 15px; background: #f8d7da; color: #721c24; border-radius: 4px;">❌ Failed to detect duplicates: ${error.message}</div>`;
            }
        }

        function selectAllDuplicates() {
            document.querySelectorAll('.duplicate-checkbox').forEach(cb => cb.checked = true);
        }

        function deselectAllDuplicates() {
            document.querySelectorAll('.duplicate-checkbox').forEach(cb => cb.checked = false);
        }

        async function removeDuplicates() {
            const checkboxes = document.querySelectorAll('.duplicate-checkbox:checked');
            if (checkboxes.length === 0) {
                alert('Please select at least one profile to remove');
                return;
            }

            const filenames = Array.from(checkboxes).map(cb => cb.value);
            if (!confirm(`Remove ${filenames.length} selected profile(s)?\n\nFiles:\n${filenames.join('\n')}\n\nThis cannot be undone.`)) {
                return;
            }

            try {
                const response = await apiFetch(apiUrl('/api/active/duplicates/remove'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({filenames})
                });
                const data = await response.json();

                if (data.success) {
                    alert(`✅ ${data.message}\n\nRemoved: ${data.removed}\nFailed: ${data.failed}`);
                    fetchProfiles();
                    document.getElementById('duplicates-result').style.display = 'none';
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed: ' + error.message);
            }
        }

        // Save LLM API key
        async function saveLLMKey() {
            const apiKey = document.getElementById('llm-api-key').value.trim();
            const provider = document.getElementById('llm-provider').value;
            const resultDiv = document.getElementById('llm-test-result');

            if (!confirm('Save this API key and enable AI analysis?')) {
                return;
            }

            resultDiv.style.display = 'block';
            resultDiv.style.background = '#e7f3ff';
            resultDiv.style.color = '#004085';
            resultDiv.textContent = '💾 Saving configuration...';

            try {
                const response = await apiFetch(apiUrl('/api/llm/save'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({provider, api_key: apiKey})
                });

                const data = await response.json();

                if (data.success) {
                    resultDiv.style.background = '#d4edda';
                    resultDiv.style.color = '#155724';
                    resultDiv.innerHTML = '✅ ' + data.message.replace(/\n/g, '<br>');

                    // Offer to restart
                    setTimeout(() => {
                        if (confirm('Restart the analyzer container now to enable AI analysis?')) {
                            resultDiv.textContent = '🔄 Restarting container... (page will reload in 10 seconds)';
                            setTimeout(() => location.reload(), 10000);
                        }
                    }, 2000);
                } else {
                    resultDiv.style.background = '#f8d7da';
                    resultDiv.style.color = '#721c24';
                    resultDiv.textContent = '✗ ' + data.error;
                }
            } catch (error) {
                resultDiv.style.background = '#f8d7da';
                resultDiv.style.color = '#721c24';
                resultDiv.textContent = '✗ Failed to save: ' + error.message;
            }
        }


// ── AI Configuration Management (v2 per-project) ────────────────────────
        // ── AI Configuration Management (v2 per-project) ──────────────

        function _showAIResult(divId, success, msg) {
            const d = document.getElementById(divId);
            if (!d) return;
            d.style.display = 'block';
            d.style.padding = '8px 12px';
            d.style.borderRadius = '4px';
            if (success) {
                d.style.background = '#d4edda'; d.style.color = '#155724';
                d.textContent = '✅ ' + msg;
                setTimeout(() => { d.style.display = 'none'; }, 5000);
            } else {
                d.style.background = '#f8d7da'; d.style.color = '#721c24';
                d.textContent = '✗ ' + msg;
            }
        }

        async function loadAIConfig() {
            // Load global keys (admin only)
            if (window.APP_CONFIG.isAdmin) {
                try {
                    const r = await apiFetch(apiUrl('/api/ai-config/global'));
                    const d = await r.json();
                    if (d.success && d.global) {
                        const g = d.global;
                        if (document.getElementById('openai-api-key'))
                            document.getElementById('openai-api-key').value = g.openai?.api_key || '';
                        if (document.getElementById('anthropic-api-key'))
                            document.getElementById('anthropic-api-key').value = g.anthropic?.api_key || '';
                    }
                } catch(e) { console.error('Failed to load global AI keys:', e); }

                // Populate admin project selectors (only once — refresh() calls us repeatedly)
                const sel = document.getElementById('ai-project-selector');
                if (sel && sel.options.length <= 1) {
                    try {
                        const r2 = await apiFetch(apiUrl('/api/projects'));
                        const d2 = await r2.json();
                        const projects = Array.isArray(d2) ? d2 : (d2.projects || []);
                        const cpDst = document.getElementById('ai-copy-dest-project');
                        projects.forEach(p => {
                            const slug = p.slug || p.name || String(p);
                            const opt = document.createElement('option');
                            opt.value = slug; opt.textContent = slug;
                            sel.appendChild(opt);
                            if (cpDst) {
                                const opt2 = document.createElement('option');
                                opt2.value = slug; opt2.textContent = slug;
                                cpDst.appendChild(opt2);
                            }
                        });
                    } catch(e) { console.error('Failed to load project list:', e); }
                }
            } else {
                // Non-admin: auto-load current project
                loadProjectAIConfig(window.APP_CONFIG.currentProject);
            }
        }

        async function saveGlobalAIKeys() {
            const oKey = document.getElementById('openai-api-key')?.value.trim() || '';
            const aKey = document.getElementById('anthropic-api-key')?.value.trim() || '';
            try {
                const r = await apiFetch(apiUrl('/api/ai-config/global'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ global: {
                        openai:    { api_key: oKey,  enabled: !!oKey },
                        anthropic: { api_key: aKey, enabled: !!aKey }
                    }})
                });
                const d = await r.json();
                _showAIResult('global-keys-result', d.success, d.message || d.error || 'Saved.');
            } catch(e) {
                _showAIResult('global-keys-result', false, e.message);
            }
        }

        const _AI_PROVIDERS = ['openai', 'anthropic'];
        const _AI_PROVIDER_MODELS = {
            openai:    ['gpt-4o','gpt-4-turbo','gpt-4','gpt-3.5-turbo'],
            anthropic: ['claude-sonnet-4-6','claude-opus-4-6','claude-haiku-4-5-20251001',
                        'claude-3-5-sonnet-20241022','claude-3-opus-20240229']
        };
        let _aiCurrentSlug = null;

        function _makeProviderSelect(id, selectedVal) {
            let html = `<select id="${id}" style="font-size:12px; padding:3px 6px; border:1px solid #ccc; border-radius:4px;">`;
            _AI_PROVIDERS.forEach(p => {
                html += `<option value="${p}" ${p === selectedVal ? 'selected' : ''}>${p}</option>`;
            });
            html += '</select>';
            return html;
        }

        function _makeModelSelect(id, provider, selectedVal) {
            const models = _AI_PROVIDER_MODELS[provider] || _AI_PROVIDER_MODELS.openai;
            let html = `<select id="${id}" style="font-size:12px; padding:3px 6px; border:1px solid #ccc; border-radius:4px;">`;
            models.forEach(m => {
                html += `<option value="${m}" ${m === selectedVal ? 'selected' : ''}>${m}</option>`;
            });
            html += '</select>';
            return html;
        }

        function _renderProjectAITable(slug, config, defaults) {
            _aiCurrentSlug = slug;
            const tbody = document.getElementById('project-ai-tbody');
            const useCases = [
                ['document_analysis', 'Document Analysis'],
                ['chat', 'AI Chat'],
                ['case_intelligence', 'Case Intelligence']
            ];
            let rows = '';
            useCases.forEach(([uc, label]) => {
                const cfg = config[uc] || defaults[uc] || {};
                const pProv = cfg.provider || 'openai';
                const pMod  = cfg.model || _AI_PROVIDER_MODELS.openai[0];
                const fProv = cfg.fallback_provider || 'anthropic';
                const fMod  = cfg.fallback_model || _AI_PROVIDER_MODELS.anthropic[0];
                rows += `<tr style="border-bottom:1px solid #f0f0f0;">
                  <td style="padding:8px 10px; font-weight:500;">${label}</td>
                  <td style="padding:8px 10px;">
                    <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
                      ${_makeProviderSelect(`ai-${uc}-prov`, pProv)}
                      ${_makeModelSelect(`ai-${uc}-mod`, pProv, pMod)}
                    </div>
                  </td>
                  <td style="padding:8px 10px;">
                    <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
                      ${_makeProviderSelect(`ai-${uc}-fprov`, fProv)}
                      ${_makeModelSelect(`ai-${uc}-fmod`, fProv, fMod)}
                    </div>
                  </td>
                  <td style="padding:8px 10px; text-align:center;">
                    <button class="btn btn-sm" onclick="copyUseCaseToAll('${uc}')" style="font-size:11px; padding:3px 8px;" title="Copy this row to all use-cases">⇉ All</button>
                  </td>
                </tr>`;
            });
            tbody.innerHTML = rows;
            // Wire up provider selects to refresh model selects
            useCases.forEach(([uc]) => {
                ['', 'f'].forEach(prefix => {
                    const pSel = document.getElementById(`ai-${uc}-${prefix}prov`);
                    const mSel = document.getElementById(`ai-${uc}-${prefix}mod`);
                    if (pSel && mSel) {
                        pSel.onchange = () => {
                            const newProv = pSel.value;
                            const models = _AI_PROVIDER_MODELS[newProv] || _AI_PROVIDER_MODELS.openai;
                            mSel.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
                        };
                    }
                });
            });
            document.getElementById('project-ai-placeholder').style.display = 'none';
            document.getElementById('project-ai-table').style.display = 'block';
        }

        async function loadProjectAIConfig(slug) {
            if (!slug) return;
            try {
                const r = await apiFetch(apiUrl(`/api/ai-config/projects/${slug}`));
                const d = await r.json();
                if (d.success !== false) {
                    _renderProjectAITable(slug, d.config || {}, d.defaults || {});
                    // Populate per-project key inputs
                    const oKey = document.getElementById('proj-openai-key');
                    const aKey = document.getElementById('proj-anthropic-key');
                    const oSt  = document.getElementById('proj-openai-key-status');
                    const aSt  = document.getElementById('proj-anthropic-key-status');
                    if (oKey) oKey.value = d.has_openai_key    ? '••••••••' : '';
                    if (aKey) aKey.value = d.has_anthropic_key ? '••••••••' : '';
                    if (oSt)  oSt.textContent  = d.has_openai_key    ? '✓ key saved' : '(using global)';
                    if (aSt)  aSt.textContent  = d.has_anthropic_key ? '✓ key saved' : '(using global)';
                    if (oSt)  oSt.style.color  = d.has_openai_key    ? '#16a34a' : '#9ca3af';
                    if (aSt)  aSt.style.color  = d.has_anthropic_key ? '#16a34a' : '#9ca3af';
                    const wrap = document.getElementById('project-api-keys-wrap');
                    if (wrap) wrap.style.display = 'block';
                } else {
                    console.error('loadProjectAIConfig error:', d.error);
                }
            } catch(e) { console.error('Failed to load project AI config:', e); }
        }

        function clearProjectKey(provider) {
            const inp = document.getElementById(`proj-${provider}-key`);
            const st  = document.getElementById(`proj-${provider}-key-status`);
            if (inp) inp.value = '';
            if (st)  { st.textContent = '(will use global on save)'; st.style.color = '#9ca3af'; }
        }

        async function testProjectKey(provider) {
            const inp = document.getElementById(`proj-${provider}-key`);
            const st  = document.getElementById(`proj-${provider}-key-status`);
            const key = inp?.value.trim();
            if (!key || key === '••••••••') {
                if (st) { st.textContent = 'Enter a new key to test'; st.style.color = '#d97706'; }
                return;
            }
            if (st) { st.textContent = '🧪 Testing…'; st.style.color = '#2563eb'; }
            try {
                const r = await apiFetch(apiUrl('/api/ai-config/test'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({provider, api_key: key})
                });
                const d = await r.json();
                if (st) { st.textContent = d.success ? ('✓ ' + d.message) : ('✗ ' + d.error); st.style.color = d.success ? '#16a34a' : '#dc2626'; }
            } catch(e) {
                if (st) { st.textContent = '✗ ' + e.message; st.style.color = '#dc2626'; }
            }
        }

        async function saveProjectAIConfig() {
            if (!_aiCurrentSlug) return;
            const useCases = ['document_analysis', 'chat', 'case_intelligence'];
            const config = {};
            useCases.forEach(uc => {
                config[uc] = {
                    provider:          document.getElementById(`ai-${uc}-prov`)?.value || 'openai',
                    model:             document.getElementById(`ai-${uc}-mod`)?.value || 'gpt-4o',
                    fallback_provider: document.getElementById(`ai-${uc}-fprov`)?.value || 'anthropic',
                    fallback_model:    document.getElementById(`ai-${uc}-fmod`)?.value || 'claude-sonnet-4-6',
                };
            });
            // Include per-project API key overrides (backend preserves existing if masked)
            config.openai_api_key    = document.getElementById('proj-openai-key')?.value.trim()    || '';
            config.anthropic_api_key = document.getElementById('proj-anthropic-key')?.value.trim() || '';
            try {
                const r = await apiFetch(apiUrl(`/api/ai-config/projects/${_aiCurrentSlug}`), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({config})
                });
                const d = await r.json();
                _showAIResult('project-ai-result', d.success, d.message || d.error || 'Saved.');
                if (d.success) loadProjectAIConfig(_aiCurrentSlug); // refresh key status badges
            } catch(e) {
                _showAIResult('project-ai-result', false, e.message);
            }
        }

        async function copyUseCaseToAll(useCase) {
            if (!_aiCurrentSlug) return;
            try {
                const r = await apiFetch(apiUrl(`/api/ai-config/projects/${_aiCurrentSlug}/copy-use-case`), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({use_case: useCase})
                });
                const d = await r.json();
                if (d.success) {
                    await loadProjectAIConfig(_aiCurrentSlug);
                    _showAIResult('project-ai-result', true, d.message);
                } else {
                    _showAIResult('project-ai-result', false, d.error);
                }
            } catch(e) {
                _showAIResult('project-ai-result', false, e.message);
            }
        }

        async function copyProjectAIConfig() {
            if (!_aiCurrentSlug) return;
            const destSlug = document.getElementById('ai-copy-dest-project')?.value;
            if (!destSlug) { alert('Select a destination project first.'); return; }
            try {
                const r = await apiFetch(apiUrl('/api/ai-config/projects/copy'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({source_slug: _aiCurrentSlug, dest_slug: destSlug})
                });
                const d = await r.json();
                _showAIResult('project-ai-result', d.success, d.message || d.error || 'Done.');
            } catch(e) {
                _showAIResult('project-ai-result', false, e.message);
            }
        }

        // Legacy alias kept for testAIProvider compatibility
        async function saveAIConfig() { await saveGlobalAIKeys(); }

        async function testAIProvider(provider) {
            const apiKeyInput = document.getElementById(`${provider}-api-key`);
            const resultDiv = document.getElementById(`${provider}-test-result`);
            const apiKey = apiKeyInput.value.trim();

            if (!apiKey) {
                resultDiv.style.display = 'block';
                resultDiv.style.background = '#f8d7da';
                resultDiv.style.padding = '10px';
                resultDiv.style.borderRadius = '4px';
                resultDiv.style.color = '#721c24';
                resultDiv.textContent = '✗ Please enter an API key';
                return;
            }

            resultDiv.style.display = 'block';
            resultDiv.style.background = '#e7f3ff';
            resultDiv.style.padding = '10px';
            resultDiv.style.borderRadius = '4px';
            resultDiv.style.color = '#004085';
            resultDiv.textContent = '🧪 Testing API key...';

            try {
                const response = await apiFetch(apiUrl('/api/ai-config/test'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({provider, api_key: apiKey})
                });

                const data = await response.json();

                if (data.success) {
                    resultDiv.style.background = '#d4edda';
                    resultDiv.style.color = '#155724';
                    resultDiv.textContent = data.message;
                } else {
                    resultDiv.style.background = '#f8d7da';
                    resultDiv.style.color = '#721c24';
                    resultDiv.textContent = data.error;
                }
            } catch (error) {
                resultDiv.style.background = '#f8d7da';
                resultDiv.style.color = '#721c24';
                resultDiv.textContent = '✗ Test failed: ' + error.message;
            }
        }

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
