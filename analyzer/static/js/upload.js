// ── Upload Tab ───────────────────────────────────────────────────────────


        // ── Upload tab state ──────────────────────────────────────────────
        let _uploadMode = 'file';          // 'file' | 'url' | 'cloud'
        let _smartPendingPayload = null;   // holds {type, file|url, metadata, source} during preview
        let _fileListFiles = [];           // files returned by scan-url directory scan
        let _fileListAuthPayload = {};     // auth/source payload shared by all batch uploads

        function switchUploadMode(mode) {
            _uploadMode = mode;
            ['file', 'url', 'cloud', 'court'].forEach(m => {
                const btn = document.getElementById('umode-' + m);
                const pnl = document.getElementById('uinput-' + m);
                if (!btn || !pnl) return;   // court mode elements absent when feature disabled
                const active = m === mode;
                btn.style.background = active ? '#fff' : 'transparent';
                btn.style.borderBottom = active ? '3px solid #2563eb' : '3px solid transparent';
                btn.style.fontWeight = active ? '600' : '500';
                btn.style.color = active ? '#2563eb' : '#555';
                pnl.style.display = active ? '' : 'none';
            });
            // Hide the shared upload controls (project/smart-meta/submit) in court mode
            const shared = document.getElementById('upload-shared-controls');
            if (shared) shared.style.display = mode === 'court' ? 'none' : '';
            document.getElementById('upload-status').innerHTML = '';
            cancelSmartPreview();
            // Initialise court sub-tab on first switch
            if (mode === 'court') {
                if (typeof loadCourtCredStatus === 'function') loadCourtCredStatus();
                if (typeof loadCourtHistory === 'function') loadCourtHistory();
            }
        }

        function updateAuthFields() {
            const t = document.getElementById('upload-auth-type').value;
            document.getElementById('upload-auth-basic').style.display = t === 'basic' ? '' : 'none';
            document.getElementById('upload-auth-token').style.display = t === 'token' ? '' : 'none';
        }

        function handleDrop(event) {
            event.preventDefault();
            const files = event.dataTransfer.files;
            if (!files.length) return;
            const input = document.getElementById('upload-file-input');
            const dt = new DataTransfer();
            dt.items.add(files[0]);
            input.files = dt.files;
            document.getElementById('upload-file-name').textContent = files[0].name;
            const dz = document.getElementById('upload-dropzone');
            dz.style.background = '';
            dz.style.borderColor = '#2563eb';
        }

        function setUploadStatus(html) {
            document.getElementById('upload-status').innerHTML = html;
        }

        function cancelSmartPreview() {
            _smartPendingPayload = null;
            document.getElementById('upload-meta-preview').style.display = 'none';
            document.getElementById('upload-meta-content').innerHTML = '';
        }

        async function detectAndTransform(url) {
            const res = await apiFetch(apiUrl('/api/upload/transform-url'), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url})
            });
            return await res.json();
        }

        async function _scanUrl(url, authPayload) {
            try {
                const res = await apiFetch(apiUrl('/api/upload/scan-url'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(Object.assign({url}, authPayload)),
                });
                return await res.json();
            } catch(e) {
                return {error: e.message};
            }
        }

        function _fmtSize(bytes) {
            if (!bytes) return '';
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1048576) return Math.round(bytes / 1024) + ' KB';
            return (bytes / 1048576).toFixed(1) + ' MB';
        }

        function _showFileList(files, authPayload) {
            _fileListFiles = files;
            _fileListAuthPayload = authPayload;

            const extColor = {
                '.pdf':'#dc2626','.docx':'#2563eb','.doc':'#2563eb','.odt':'#2563eb',
                '.png':'#16a34a','.jpg':'#16a34a','.jpeg':'#16a34a','.gif':'#16a34a',
                '.tiff':'#16a34a','.tif':'#16a34a','.webp':'#16a34a','.heic':'#16a34a',
                '.xlsx':'#15803d','.xls':'#15803d','.ods':'#15803d','.csv':'#15803d',
                '.pptx':'#d97706','.ppt':'#d97706','.txt':'#6b7280','.eml':'#9333ea',
            };
            const rows = files.map(function(f, i) {
                const col = extColor[f.ext] || '#6b7280';
                const badge = f.ext.slice(1).toUpperCase();
                const sz = _fmtSize(f.size_bytes);
                return '<div style="display:flex;align-items:center;gap:8px;padding:7px 12px;border-bottom:1px solid #f0f0f0;" id="flist-row-' + i + '">'
                    + '<input type="checkbox" class="flist-cb" data-idx="' + i + '" checked style="width:13px;height:13px;flex-shrink:0;">'
                    + '<span style="background:' + col + ';color:#fff;font-size:10px;font-weight:700;padding:1px 5px;border-radius:3px;min-width:34px;text-align:center;flex-shrink:0;">' + badge + '</span>'
                    + '<span style="flex:1;font-size:13px;color:#111;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="' + f.filename + '">' + f.filename + '</span>'
                    + (sz ? '<span style="font-size:11px;color:#9ca3af;white-space:nowrap;">' + sz + '</span>' : '')
                    + '<span id="flist-st-' + i + '" style="font-size:12px;width:20px;text-align:center;flex-shrink:0;"></span>'
                    + '</div>';
            }).join('');

            document.getElementById('upload-filelist-table').innerHTML = rows;
            document.getElementById('upload-filelist-count').textContent = files.length;
            document.getElementById('upload-filelist-selectall').checked = true;
            document.getElementById('upload-filelist').style.display = '';
            document.getElementById('upload-filelist-progress').textContent = '';
            setUploadStatus('');
            _updateFileListBtn();
        }

        function _updateFileListBtn() {
            const n = document.querySelectorAll('.flist-cb:checked').length;
            const btn = document.getElementById('upload-filelist-btn');
            btn.textContent = 'Upload ' + n + ' file' + (n !== 1 ? 's' : '');
            btn.disabled = n === 0;
        }

        function toggleSelectAllFiles(checked) {
            document.querySelectorAll('.flist-cb').forEach(function(cb) { cb.checked = checked; });
            _updateFileListBtn();
        }

        function cancelFileList() {
            _fileListFiles = [];
            _fileListAuthPayload = {};
            document.getElementById('upload-filelist').style.display = 'none';
            setUploadStatus('');
        }

        async function uploadSelectedFiles() {
            const indices = [];
            document.querySelectorAll('.flist-cb:checked').forEach(function(cb) {
                indices.push(parseInt(cb.getAttribute('data-idx')));
            });
            if (!indices.length) return;

            const btn = document.getElementById('upload-filelist-btn');
            const prog = document.getElementById('upload-filelist-progress');
            btn.disabled = true;

            let done = 0, failed = 0;
            for (let i = 0; i < indices.length; i++) {
                const idx = indices[i];
                const f = _fileListFiles[idx];
                const stEl = document.getElementById('flist-st-' + idx);
                stEl.textContent = '⏳';
                prog.textContent = (i + 1) + '/' + indices.length + ' uploading…';
                try {
                    const payload = Object.assign({}, _fileListAuthPayload, {url: f.url});
                    const res = await apiFetch(apiUrl('/api/upload/from-url'), {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(payload),
                    });
                    const data = await res.json();
                    if (data.success) { stEl.textContent = '✅'; done++; }
                    else { stEl.title = data.error || 'failed'; stEl.textContent = '❌'; failed++; }
                } catch(e) {
                    stEl.title = e.message; stEl.textContent = '❌'; failed++;
                }
            }
            prog.textContent = done + ' uploaded' + (failed ? ', ' + failed + ' failed' : '') + ' — ⏳ analyzer picks up in ~30 s';
            btn.disabled = false;
            if (done > 0) loadImportHistory();
        }

        async function doUpload() {
            const btn = document.getElementById('upload-submit-btn');
            const smartMeta = document.getElementById('upload-smart-meta').checked;
            btn.disabled = true;
            cancelSmartPreview();

            try {
                if (_uploadMode === 'file') {
                    await _doFileUpload(smartMeta);
                } else {
                    await _doUrlUpload(smartMeta);
                }
            } finally {
                btn.disabled = false;
            }
        }

        async function _doFileUpload(smartMeta) {
            const fileInput = document.getElementById('upload-file-input');
            if (!fileInput.files || !fileInput.files.length) {
                setUploadStatus('<div style="color:#e74c3c;">Please select a file first.</div>');
                return;
            }
            const file = fileInput.files[0];

            if (smartMeta) {
                // Step 1: analyze
                setUploadStatus('<div style="color:#2563eb;">🔍 Analyzing document with AI…</div>');
                const fd = new FormData();
                fd.append('file', file);
                const res = await apiFetch(apiUrl('/api/upload/analyze'), {method: 'POST', body: fd});
                const meta = await res.json();
                if (meta.error) {
                    setUploadStatus('<div style="color:#e74c3c;">❌ Analysis failed: ' + meta.error + '</div>');
                    return;
                }
                _smartPendingPayload = {type: 'file', file, metadata: meta};
                showSmartPreview(meta);
                setUploadStatus('');
            } else {
                setUploadStatus('<div style="color:#2563eb;">⏳ Uploading…</div>');
                const fd = new FormData();
                fd.append('file', file);
                const chosenProject = document.getElementById('upload-project-select')?.value;
                if (chosenProject) fd.append('project_slug', chosenProject);
                const res = await apiFetch(apiUrl('/api/upload/submit'), {method: 'POST', body: fd});
                const data = await res.json();
                if (data.success) {
                    setUploadStatus('<div style="color:#16a34a; padding:10px; background:#dcfce7; border-radius:6px;">✅ Uploaded: <strong>' + (data.title || file.name) + '</strong><br><span style="font-size:12px; color:#6b7280;">⏳ Analyzer will process it on the next poll (~30 s)</span></div>');
                    fileInput.value = '';
                    document.getElementById('upload-file-name').textContent = '';
                    loadImportHistory();
                } else {
                    setUploadStatus('<div style="color:#e74c3c;">❌ ' + (data.error || 'Upload failed') + '</div>');
                }
            }
        }

        async function _doUrlUpload(smartMeta) {
            const rawUrl = (_uploadMode === 'cloud'
                ? document.getElementById('upload-cloud-input')
                : document.getElementById('upload-url-input')).value.trim();

            if (!rawUrl) {
                setUploadStatus('<div style="color:#e74c3c;">Please enter a URL.</div>');
                return;
            }

            // Step 1: transform cloud / share links to direct download URLs
            let directUrl = rawUrl;
            let service = 'url';
            try {
                setUploadStatus('<div style="color:#2563eb;">🔗 Resolving URL…</div>');
                const tx = await detectAndTransform(rawUrl);
                if (tx.error) throw new Error(tx.error);
                directUrl = tx.direct_url;
                service = tx.service || 'url';
                if (_uploadMode === 'cloud') {
                    const det = document.getElementById('upload-cloud-detected');
                    det.textContent = 'Detected: ' + service.replace('_', ' ');
                }
            } catch (e) {
                setUploadStatus('<div style="color:#e74c3c;">❌ URL transform error: ' + e.message + '</div>');
                return;
            }

            const authType = _uploadMode === 'url' ? document.getElementById('upload-auth-type').value : 'none';
            const chosenProject = document.getElementById('upload-project-select')?.value || undefined;
            const urlPayload = {
                url: directUrl,
                source: service,
                auth_type: authType,
                username: authType === 'basic' ? document.getElementById('upload-auth-user').value : undefined,
                password: authType === 'basic' ? document.getElementById('upload-auth-pass').value : undefined,
                token: authType === 'token' ? document.getElementById('upload-auth-token-val').value : undefined,
                project_slug: chosenProject,
            };

            // Step 2: scan — single file or a directory listing?
            setUploadStatus('<div style="color:#2563eb;">🔍 Scanning URL…</div>');
            const scan = await _scanUrl(directUrl, urlPayload);
            if (scan.error) {
                setUploadStatus('<div style="color:#e74c3c;">❌ ' + scan.error + '</div>');
                return;
            }

            if (scan.type === 'directory') {
                // Show file-picker panel; user chooses which files to upload
                _showFileList(scan.files, urlPayload);
                return;
            }

            // Single file — proceed to direct upload
            if (smartMeta) {
                setUploadStatus('<div style="color:#f59e0b; padding:8px 12px; background:#fef3c7; border-radius:6px;">⚠️ Smart Metadata is not available for URL uploads — uploading directly.</div>');
                await new Promise(r => setTimeout(r, 1500));
            }

            setUploadStatus('<div style="color:#2563eb;">⏳ Downloading and uploading…</div>');
            const res = await apiFetch(apiUrl('/api/upload/from-url'), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(urlPayload)
            });
            const data = await res.json();
            if (data.success) {
                setUploadStatus('<div style="color:#16a34a; padding:10px; background:#dcfce7; border-radius:6px;">✅ Uploaded: <strong>' + (data.title || directUrl) + '</strong><br><span style="font-size:12px; color:#6b7280;">⏳ Analyzer will process it on the next poll (~30 s)</span></div>');
                if (_uploadMode === 'cloud') document.getElementById('upload-cloud-input').value = '';
                if (_uploadMode === 'url') document.getElementById('upload-url-input').value = '';
                loadImportHistory();
            } else {
                setUploadStatus('<div style="color:#e74c3c;">❌ ' + (data.error || 'Upload failed') + '</div>');
            }
        }

        function showSmartPreview(meta) {
            const top = meta.project_suggestions?.[0];
            const html = `
                <table style="width:100%; border-collapse:collapse; font-size:13px;">
                    <tr><td style="padding:4px 8px; color:#6b7280; white-space:nowrap; width:130px;">Title</td><td style="padding:4px 8px; font-weight:600;">${meta.suggested_title || '—'}</td></tr>
                    <tr><td style="padding:4px 8px; color:#6b7280;">Type</td><td style="padding:4px 8px;">${meta.document_type || '—'}</td></tr>
                    <tr><td style="padding:4px 8px; color:#6b7280;">Tags</td><td style="padding:4px 8px;">${(meta.suggested_tags || []).join(', ') || '—'}</td></tr>
                    <tr><td style="padding:4px 8px; color:#6b7280;">Project</td><td style="padding:4px 8px;">${top ? top.name + ' (' + Math.round((top.confidence||0)*100) + '%)' : '—'}</td></tr>
                </table>
                ${top ? '<input type="hidden" id="smart-project-slug" value="' + top.slug + '">' : ''}
            `;
            document.getElementById('upload-meta-content').innerHTML = html;
            document.getElementById('upload-meta-preview').style.display = '';
        }

        async function confirmSmartUpload() {
            if (!_smartPendingPayload) return;
            const {type, file, metadata} = _smartPendingPayload;
            // User's explicit dropdown choice beats AI suggestion
            const dropdownSlug = document.getElementById('upload-project-select')?.value;
            const aiSlug = document.getElementById('smart-project-slug')?.value;
            const projectSlug = dropdownSlug || aiSlug;
            cancelSmartPreview();
            setUploadStatus('<div style="color:#2563eb;">⏳ Uploading with AI metadata…</div>');

            const fd = new FormData();
            fd.append('file', file);
            if (projectSlug) fd.append('project_slug', projectSlug);
            fd.append('metadata', JSON.stringify(metadata));
            const res = await apiFetch(apiUrl('/api/upload/submit'), {method: 'POST', body: fd});
            const data = await res.json();
            if (data.success) {
                setUploadStatus('<div style="color:#16a34a; padding:10px; background:#dcfce7; border-radius:6px;">✅ Uploaded: <strong>' + (data.title || file.name) + '</strong><br><span style="font-size:12px; color:#6b7280;">⏳ Analyzer will process it on the next poll (~30 s)</span></div>');
                document.getElementById('upload-file-input').value = '';
                document.getElementById('upload-file-name').textContent = '';
                loadImportHistory();
            } else {
                setUploadStatus('<div style="color:#e74c3c;">❌ ' + (data.error || 'Upload failed') + '</div>');
            }
        }

        async function loadUploadProjects() {
            const sel = document.getElementById('upload-project-select');
            if (!sel) return;
            try {
                const res = await apiFetch(apiUrl('/api/projects'));
                const data = await res.json();
                const projects = data.projects || [];
                // Keep first option (None), rebuild the rest
                sel.innerHTML = '<option value="">None (auto-assign by analyzer)</option>';
                projects.forEach(p => {
                    const opt = document.createElement('option');
                    opt.value = p.slug;
                    opt.textContent = p.name + (p.document_count != null ? ' (' + p.document_count + ' docs)' : '');
                    sel.appendChild(opt);
                });
            } catch (e) {
                // Non-critical — silently ignore if projects can't be loaded
            }
        }

        async function loadImportHistory() {
            const body = document.getElementById('upload-history-body');
            if (!body) return;
            try {
                const res = await apiFetch(apiUrl('/api/upload/history'));
                const data = await res.json();
                const rows = data.history || [];
                if (!rows.length) {
                    body.innerHTML = '<div style="padding:20px; color:#9ca3af; font-size:13px; text-align:center;">No imports yet.</div>';
                    return;
                }
                const badgeColors = {
                    file: '#6366f1', url: '#0284c7',
                    google_drive: '#16a34a', dropbox: '#2563eb', onedrive: '#0ea5e9'
                };
                const slug = document.getElementById('project-selector')?.value || 'default';
                const rows_html = rows.map(r => {
                    const color = badgeColors[r.source] || '#6b7280';
                    const badge = `<span style="background:${color}; color:#fff; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600;">${(r.source || 'unknown').replace('_', '\u00a0')}</span>`;
                    const ts = r.created_at ? r.created_at.replace('T', ' ').slice(0, 16) : '';
                    // Filename cell — link if original_url present
                    const fname = r.filename || r.original_url || '—';
                    const fnameCell = r.original_url
                        ? `<a href="${escapeHtml(r.original_url)}" target="_blank" title="${escapeHtml(r.original_url)}" style="color:#2563eb; text-decoration:none; font-size:13px;">${escapeHtml(fname)}</a>`
                        : `<span style="font-size:13px;">${escapeHtml(fname)}</span>`;
                    // Status cell — tooltip on error
                    let statusCell;
                    if (r.status === 'uploaded') {
                        statusCell = '✅ uploaded';
                    } else {
                        const errTip = r.error_msg ? ` title="${escapeHtml(r.error_msg)}"` : '';
                        statusCell = `<span${errTip} style="cursor:${r.error_msg ? 'help' : 'default'}; color:#dc2626;">❌ ${escapeHtml(r.status || 'error')}</span>`;
                    }
                    // Paperless link cell
                    let linkCell = '—';
                    if (r.paperless_doc_id) {
                        const docUrl = _paperlessDocUrl(slug, r.paperless_doc_id);
                        linkCell = docUrl
                            ? `<a href="${docUrl}" target="_blank" style="color:#2563eb; text-decoration:none;" title="View in Paperless">🔗 View</a>`
                            : `<span style="color:#9ca3af;">Doc #${r.paperless_doc_id}</span>`;
                    }
                    return `<tr style="border-top:1px solid #f3f4f6;">
                        <td style="padding:8px 12px; max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${fnameCell}</td>
                        <td style="padding:8px 12px;">${badge}</td>
                        <td style="padding:8px 12px; font-size:12px; color:#6b7280; white-space:nowrap;">${ts}</td>
                        <td style="padding:8px 12px; font-size:13px;">${statusCell}</td>
                        <td style="padding:8px 12px; font-size:13px;">${linkCell}</td>
                    </tr>`;
                }).join('');
                body.innerHTML = `<table style="width:100%; border-collapse:collapse;">
                    <thead><tr style="background:#f9fafb;">
                        <th style="padding:8px 12px; text-align:left; font-size:12px; color:#6b7280; font-weight:600;">Filename / URL</th>
                        <th style="padding:8px 12px; text-align:left; font-size:12px; color:#6b7280; font-weight:600;">Source</th>
                        <th style="padding:8px 12px; text-align:left; font-size:12px; color:#6b7280; font-weight:600;">Uploaded</th>
                        <th style="padding:8px 12px; text-align:left; font-size:12px; color:#6b7280; font-weight:600;">Status</th>
                        <th style="padding:8px 12px; text-align:left; font-size:12px; color:#6b7280; font-weight:600;">Link</th>
                    </tr></thead>
                    <tbody>${rows_html}</tbody>
                </table>`;
            } catch (e) {
                body.innerHTML = '<div style="padding:12px; color:#e74c3c; font-size:13px;">Failed to load history.</div>';
            }
        }

        function refresh() {
            fetchStatus();
            fetchRecent();
            fetchProfiles();
            refreshVectorTypes();
            loadAIConfig();
            // Refresh import history only when the upload tab is visible
            if (document.getElementById('tab-upload')?.classList.contains('active')) {
                loadImportHistory();
            }
            // Vector store documents are loaded lazily when the sub-tab is opened;
            // no need to reload 700+ doc rows on every 10-second poll
        }

        // Reprocess all documents
        async function reprocessAll() {
            if (!confirm('Reset state and reprocess ALL documents? This will analyze every document on the next poll cycle.')) {
                return;
            }

            try {
                const response = await apiFetch(apiUrl('/api/reprocess'), { method: 'POST' });
                const data = await response.json();

                if (data.success) {
                    alert('✅ ' + data.message);
                    refresh();
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed to reprocess: ' + error.message);
            }
        }

        // Reprocess specific document
        async function reprocessDocument() {
            const docId = document.getElementById('reprocess-doc-id').value;
            if (!docId) {
                alert('Please enter a document ID');
                return;
            }

            try {
                const response = await apiFetch(apiUrl(`/api/reprocess/${docId}`), { method: 'POST' });
                const data = await response.json();

                if (data.success) {
                    alert('✅ ' + data.message);
                    document.getElementById('reprocess-doc-id').value = '';
                    refresh();
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed to reprocess: ' + error.message);
            }
        }

        async function runReconcile() {
            const btn = document.getElementById('reconcile-btn');
            const result = document.getElementById('reconcile-result');
            btn.disabled = true;
            btn.textContent = '⏳ Reconciling...';
            result.style.display = 'none';
            try {
                const r = await apiFetch(apiUrl('/api/reconcile'), { method: 'POST' });
                const d = await r.json();
                if (!r.ok) throw new Error(d.error || 'Failed');
                const cleaned = d.db_orphans_removed + d.chroma_orphans_removed > 0;
                result.style.background = cleaned ? '#f0fdf4' : '#f8fafc';
                result.style.border = `1px solid ${cleaned ? '#86efac' : '#e2e8f0'}`;
                result.style.color = cleaned ? '#166534' : '#475569';
                result.innerHTML = `
                    <strong>${cleaned ? '✅ Reconcile complete' : 'ℹ️ Everything looks clean'}</strong><br>
                    📄 Paperless documents: <strong>${d.paperless_total}</strong><br>
                    🗑️ Stale DB records removed: <strong>${d.db_orphans_removed}</strong><br>
                    🗑️ Stale embeddings removed: <strong>${d.chroma_orphans_removed}</strong><br>
                    ⏳ Not yet analyzed: <strong>${d.not_analyzed}</strong><br>
                    ⏳ Not yet embedded: <strong>${d.not_embedded}</strong>
                `;
                result.style.display = 'block';
            } catch(e) {
                result.style.background = '#fef2f2';
                result.style.border = '1px solid #fca5a5';
                result.style.color = '#991b1b';
                result.innerHTML = `<strong>❌ Error:</strong> ${e.message}`;
                result.style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.textContent = '🔁 Reconcile Now';
            }
        }

        // Vector store management functions
        async function refreshVectorTypes() {
            try {
                const response = await apiFetch(apiUrl('/api/vector/types'));
                const data = await response.json();

                if (!data.success) {
                    console.error('Failed to load document types');
                    return;
                }

                // Update dropdown
                const dropdown = document.getElementById('document-type-filter');
                const currentSelection = dropdown.value;

                dropdown.innerHTML = '<option value="all">All Documents</option>';
                data.document_types.forEach(type => {
                    const option = document.createElement('option');
                    option.value = type;
                    const count = data.by_type[type] || 0;
                    option.textContent = `${type} (${count})`;
                    dropdown.appendChild(option);
                });

                // Restore selection if it still exists
                if (data.document_types.includes(currentSelection)) {
                    dropdown.value = currentSelection;
                } else {
                    dropdown.value = 'all';
                }

                // Update breakdown text
                const breakdownSpan = document.getElementById('type-breakdown');
                const totalDocs = Object.values(data.by_type).reduce((sum, count) => sum + count, 0);
                breakdownSpan.textContent = `${totalDocs} total documents`;

            } catch (error) {
                console.error('Failed to refresh document types:', error);
            }
        }

        // Load all vector store documents grouped by type
        async function loadVectorStoreDocuments() {
            vsLoaded = true;
            const container = document.getElementById('vector-store-manager');
            if (container) container.innerHTML = '<div class="loading">Loading vector store documents…</div>';
            try {
                const response = await apiFetch(apiUrl('/api/vector/documents'));
                const data = await response.json();

                if (!data.success) {
                    if (container) container.innerHTML = '<p style="color:#e74c3c;">Failed to load vector store documents</p>';
                    return;
                }

                if (!data.documents || Object.keys(data.documents).length === 0) {
                    if (container) container.innerHTML = '<p style="color:#999;">No documents in vector store yet.</p>';
                    vsAllDocs = [];
                    vsFiltered = [];
                    renderVsPage();
                    return;
                }

                // Flatten into [{title, doc_id, type}, …]
                vsAllDocs = [];
                Object.entries(data.documents).forEach(([type, docs]) => {
                    docs.forEach(d => vsAllDocs.push({title: d.title, doc_id: d.doc_id, type}));
                });
                // Preserve any active search filter
                filterVsDocs();

            } catch (error) {
                console.error('Failed to load vector store documents:', error);
                if (container) container.innerHTML = '<p style="color:#e74c3c;">Failed to load vector store documents</p>';
            }
        }

        // Delete individual document from vector store
        async function deleteIndividualDocument(docId, docTitle) {
            if (!confirm(`Delete "${docTitle}" from vector store?\n\nThis will remove the embedding but NOT delete the original document from Paperless.`)) {
                return;
            }

            try {
                const response = await apiFetch(apiUrl('/api/vector/delete-document'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ doc_id: docId })
                });
                const data = await response.json();

                if (data.success) {
                    alert(`✅ ${data.message}`);
                    vsLoaded = false;
                    loadVectorStoreDocuments();
                    fetchStatus();
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed to delete document: ' + error.message);
            }
        }

        async function confirmDeleteType(documentType) {
            if (!confirm(`Delete all "${documentType}" documents from vector store?\n\nThis will remove embeddings but NOT delete the original documents from Paperless.`)) {
                return;
            }

            try {
                const response = await apiFetch(apiUrl('/api/vector/delete-by-type'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ document_type: documentType })
                });
                const data = await response.json();

                if (data.success) {
                    alert(`✅ ${data.message}`);
                    refreshVectorTypes();
                    vsLoaded = false;
                    loadVectorStoreDocuments();
                    fetchStatus();
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed to delete documents: ' + error.message);
            }
        }

        async function confirmClearVectorStore() {
            if (!confirm('Clear ALL embeddings from vector store?\n\nThis will NOT delete the original documents from Paperless, only the vector embeddings. You can re-embed documents by reprocessing them.')) {
                return;
            }

            try {
                const response = await apiFetch(apiUrl('/api/vector/clear'), { method: 'POST' });
                const data = await response.json();

                if (data.success) {
                    alert('✅ ' + data.message);
                    refreshVectorTypes();
                    vsLoaded = false;
                    loadVectorStoreDocuments();
                    fetchStatus();
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed to clear vector store: ' + error.message);
            }
        }

        async function triggerReembedStale() {
            const btn = document.getElementById('btn-reembed-stale');
            const orig = btn ? btn.textContent : '';
            if (btn) { btn.disabled = true; btn.textContent = '⏳ Checking…'; }
            try {
                const response = await apiFetch(apiUrl('/api/vector/reembed-stale'), { method: 'POST' });
                const data = await response.json();
                if (data.success) {
                    alert('✅ ' + data.message);
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed to start stale check: ' + error.message);
            } finally {
                if (btn) { btn.disabled = false; btn.textContent = orig; }
            }
        }

        async function processUnanalyzed() {
            const btn = document.getElementById('btn-process-unanalyzed');
            const orig = btn ? btn.textContent : '';
            if (btn) { btn.disabled = true; btn.textContent = '⏳ Scanning…'; }
            try {
                const response = await apiFetch(apiUrl('/api/scan/process-unanalyzed'), { method: 'POST' });
                const data = await response.json();
                if (data.success) {
                    alert('✅ ' + data.message);
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('❌ Failed: ' + error.message);
            } finally {
                if (btn) { btn.disabled = false; btn.textContent = orig; }
            }
        }

        // Update poll interval setting
        async function updatePollInterval() {
            const interval = document.getElementById('poll-interval').value;
            const resultDiv = document.getElementById('settings-result');

            if (!interval || interval < 5 || interval > 3600) {
                resultDiv.style.display = 'block';
                resultDiv.style.background = '#fff3cd';
                resultDiv.style.color = '#856404';
                resultDiv.textContent = '⚠️ Poll interval must be between 5 and 3600 seconds';
                return;
            }

            if (!confirm(`Update poll interval to ${interval} seconds?\n\nThis will require restarting the analyzer container to take effect.`)) {
                return;
            }

            resultDiv.style.display = 'block';
            resultDiv.style.background = '#e7f3ff';
            resultDiv.style.color = '#004085';
            resultDiv.textContent = '🔄 Updating poll interval...';

            try {
                const response = await apiFetch(apiUrl('/api/settings/poll-interval'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ interval: parseInt(interval) })
                });

                const data = await response.json();

                if (data.success) {
                    resultDiv.style.background = '#d4edda';
                    resultDiv.style.color = '#155724';
                    resultDiv.innerHTML = '✅ ' + data.message.replace(/\n/g, '<br>');
                } else {
                    resultDiv.style.background = '#f8d7da';
                    resultDiv.style.color = '#721c24';
                    resultDiv.textContent = '✗ ' + (data.error || 'Unknown error');
                }
            } catch (error) {
                resultDiv.style.background = '#f8d7da';
                resultDiv.style.color = '#721c24';
                resultDiv.textContent = '✗ Failed to update: ' + error.message;
            }
        }

        // Refresh logs
        async function refreshLogs() {
            try {
                const response = await apiFetch(apiUrl('/api/logs?limit=100'));
                const data = await response.json();

                const logsViewer = document.getElementById('logs-viewer');

                if (data.logs.length === 0) {
                    logsViewer.innerHTML = '<div class="empty-state">No logs available</div>';
                    return;
                }

                logsViewer.innerHTML = data.logs.filter(line => line.trim()).map(line => {
                    let className = 'log-line';
                    if (line.includes('ERROR')) className += ' error';
                    else if (line.includes('WARNING') || line.includes('WARN')) className += ' warning';
                    else if (line.includes('INFO')) className += ' info';

                    return `<div class="${className}">${escapeHtml(line)}</div>`;
                }).join('');

                // Auto-scroll to bottom
                logsViewer.scrollTop = logsViewer.scrollHeight;
            } catch (error) {
                console.error('Failed to fetch logs:', error);
                document.getElementById('logs-viewer').innerHTML = '<div class="empty-state">Failed to load logs</div>';
            }
        }
