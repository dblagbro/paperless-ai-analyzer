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

