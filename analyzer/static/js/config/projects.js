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
                    const prov = data.provision || {};
                    if (prov.eta_seconds && prov.eta_seconds > 0) {
                        const mins = Math.ceil(prov.eta_seconds / 60);
                        const pos = prov.queue_position || 1;
                        showToast(`Project created — queued for Paperless provisioning (#${pos} in line, starts in ~${mins} min). Host is throttled to one stack at a time.`, 'info', 10000);
                    } else {
                        showToast('Project created — provisioning Paperless instance in the background…', 'info', 6000);
                    }
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
            if (st === 'queued_waiting') {
                const eta = d.eta_seconds || 0;
                const pos = d.queue_position || 1;
                const mins = Math.ceil(eta / 60);
                el.style.background = '#f1f5f9'; el.style.borderColor = '#cbd5e1'; el.style.color = '#334155';
                el.innerHTML = `<span style="display:inline-block;">⏸️</span> Waiting in provisioning queue (#${pos}). Starting in ~${mins} min${mins !== 1 ? 's' : ''} <em style="font-weight:500;">— host throttled to one Paperless stack at a time.</em>`;
            } else if (st === 'queued' || st === 'running') {
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
                    // queued_waiting is a transient state — keep polling until it
                    // transitions to running/complete/error.
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

