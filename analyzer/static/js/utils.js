// ── Global Utilities ────────────────────────────────────────────────────

        // Helper function to build API URLs
        function apiUrl(path) {
            return window.APP_CONFIG.basePath + path;
        }

        // Simple toast notification (fixed banner, auto-dismisses)
        function showToast(message, type, duration) {
            const existing = document.getElementById('app-toast');
            if (existing) existing.remove();
            const bg = type === 'success' ? '#16a34a' : type === 'error' ? '#dc2626' : '#2563eb';
            const toast = document.createElement('div');
            toast.id = 'app-toast';
            toast.style.cssText = `position:fixed;top:20px;left:50%;transform:translateX(-50%);background:${bg};color:#fff;padding:12px 24px;border-radius:8px;font-size:14px;font-weight:500;z-index:99999;box-shadow:0 4px 16px rgba(0,0,0,0.3);max-width:600px;text-align:center;opacity:1;transition:opacity 0.4s;`;
            toast.textContent = message;
            document.body.appendChild(toast);
            setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 400); }, duration || 4000);
        }

        // Wrapper around fetch that detects session expiry and redirects to login
        async function apiFetch(url, options = {}) {
            const response = await fetch(url, options);
            if (response.redirected && response.url.includes('/login')) {
                window.location.href = apiUrl('/login');
                throw new Error('Session expired');
            }
            if (response.status === 401) {
                window.location.href = apiUrl('/login');
                throw new Error('Session expired');
            }
            return response;
        }

// ── Check AI Key Status ──────────────────────────────────────────────────
        async function checkAiKeyStatus() {
            try {
                const res = await apiFetch(apiUrl('/api/llm/status'));
                const data = await res.json();
                // Check v2 config: any global key present = chat is usable
                const configRes = await apiFetch(apiUrl('/api/ai-config'));
                const configData = await configRes.json();
                const globalCfg = configData.config?.global || {};
                const hasAnyKey = Object.values(globalCfg).some(p => p.api_key && p.api_key.trim());
                document.getElementById('ai-key-warning').style.display = hasAnyKey ? 'none' : 'block';
            } catch (e) {
                // If we can't check, hide the warning (don't block usage)
            }
        }

// ── Profile Modal ────────────────────────────────────────────────────────

        async function openProfileModal() {
            // Clear state
            ['profile-error','profile-success','cp-error','cp-success'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = 'none';
            });
            ['cp-current','cp-new','cp-confirm'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = '';
            });
            document.getElementById('profile-modal').style.display = 'flex';

            // Load current profile
            try {
                const r = await apiFetch(apiUrl('/api/me'));
                const d = await r.json();
                document.getElementById('profile-username-display').textContent =
                    '@' + (d.username || '') + '  ·  ' + (d.role || '').charAt(0).toUpperCase() + (d.role || '').slice(1);
                document.getElementById('profile-display-name').value = d.display_name || '';
                document.getElementById('profile-job-title').value   = d.job_title || '';
                document.getElementById('profile-email').value       = d.email || '';
                document.getElementById('profile-phone').value       = d.phone || '';
                document.getElementById('profile-address').value     = d.address || '';
            } catch(e) {
                const err = document.getElementById('profile-error');
                err.textContent = 'Could not load profile: ' + e.message;
                err.style.display = 'block';
            }
        }

        function closeProfileModal() {
            document.getElementById('profile-modal').style.display = 'none';
        }

        async function submitProfileUpdate() {
            const errEl = document.getElementById('profile-error');
            const okEl  = document.getElementById('profile-success');
            errEl.style.display = 'none';
            okEl.style.display  = 'none';
            const payload = {
                display_name: document.getElementById('profile-display-name').value.trim(),
                job_title:    document.getElementById('profile-job-title').value.trim(),
                email:        document.getElementById('profile-email').value.trim(),
                phone:        document.getElementById('profile-phone').value.trim(),
                address:      document.getElementById('profile-address').value.trim(),
            };
            if (!payload.display_name) {
                errEl.textContent = 'Full name cannot be empty.';
                errEl.style.display = 'block';
                return;
            }
            try {
                const r = await apiFetch(apiUrl('/api/me'), {
                    method: 'PATCH',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload),
                });
                const d = await r.json();
                if (d.success) {
                    okEl.textContent = 'Profile saved.';
                    okEl.style.display = 'block';
                    setTimeout(() => { if (okEl) okEl.style.display = 'none'; }, 3000);
                } else {
                    errEl.textContent = d.error || 'Save failed.';
                    errEl.style.display = 'block';
                }
            } catch(e) {
                errEl.textContent = 'Network error: ' + e.message;
                errEl.style.display = 'block';
            }
        }

        async function submitChangePassword() {
            const current = document.getElementById('cp-current').value;
            const newPw = document.getElementById('cp-new').value;
            const confirm = document.getElementById('cp-confirm').value;
            const errEl = document.getElementById('cp-error');
            const okEl = document.getElementById('cp-success');
            errEl.style.display = 'none';
            okEl.style.display = 'none';
            if (!current || !newPw) { errEl.textContent = 'All fields are required.'; errEl.style.display = 'block'; return; }
            if (newPw !== confirm) { errEl.textContent = 'New passwords do not match.'; errEl.style.display = 'block'; return; }
            if (newPw.length < 6) { errEl.textContent = 'New password must be at least 6 characters.'; errEl.style.display = 'block'; return; }
            try {
                const res = await apiFetch(apiUrl('/api/change-password'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({current_password: current, new_password: newPw})
                });
                const data = await res.json();
                if (data.success) {
                    okEl.textContent = 'Password updated successfully!';
                    okEl.style.display = 'block';
                    document.getElementById('cp-current').value = '';
                    document.getElementById('cp-new').value = '';
                    document.getElementById('cp-confirm').value = '';
                    setTimeout(() => { if (okEl) okEl.style.display = 'none'; }, 3000);
                } else {
                    errEl.textContent = data.error || 'Failed to update password.';
                    errEl.style.display = 'block';
                }
            } catch (e) {
                errEl.textContent = 'Network error: ' + e.message;
                errEl.style.display = 'block';
            }
        }

// ── HTML Escape Utility ─────────────────────────────────────────────────
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

// ── Tag Evidence Modal ──────────────────────────────────────────────────
        // Tag Evidence Modal Functions
        async function showTagEvidence(docId, docTitle) {
            const modal = document.getElementById('tag-evidence-modal');
            const modalTitle = document.getElementById('modal-doc-title');
            const modalBody = document.getElementById('modal-evidence-body');

            // Show modal and loading state
            modal.classList.add('active');
            modalTitle.textContent = `Document #${docId}: ${docTitle}`;
            modalBody.innerHTML = '<div class="loading">Loading evidence...</div>';

            try {
                const response = await apiFetch(apiUrl(`/api/tag-evidence/${docId}`));
                const data = await response.json();

                if (data.error) {
                    modalBody.innerHTML = `<div class="no-evidence">${data.error}</div>`;
                    return;
                }

                if (!data.tags || data.tags.length === 0) {
                    modalBody.innerHTML = '<div class="no-evidence">No detailed evidence available for this document.</div>';
                    return;
                }

                // Render evidence items
                let html = '';
                if (data.integrity_summary) {
                    html += `<div style="margin-bottom: 20px; padding: 12px; background: #e8f4f8; border-radius: 6px;">
                        <strong>Integrity Summary:</strong> ${data.integrity_summary}
                        ${data.critical_count > 0 ? `<span style="color: #e74c3c; margin-left: 12px;">⚠️ ${data.critical_count} critical issue(s)</span>` : ''}
                    </div>`;
                }

                for (const tag of data.tags) {
                    const sev = tag.severity || 'low';
                    const severityClass = `severity-${sev}`;
                    const evidence = tag.evidence || {};

                    // Convert newlines in description to <br> for HTML rendering
                    const descHtml = (tag.description || 'No description')
                        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                        .replace(/\n/g, '<br>');

                    // Style for info-level (false positive / pass)
                    const itemStyle = sev === 'info'
                        ? 'style="background: #f0f9ff; border-left: 4px solid #3498db; opacity: 0.85;"'
                        : '';

                    html += `
                        <div class="evidence-item ${severityClass}" ${itemStyle}>
                            <div class="evidence-tag-name">
                                <span>${tag.tag}</span>
                                <span class="evidence-severity ${sev}">${sev}</span>
                            </div>
                            <div class="evidence-description">
                                <strong>${tag.category || 'Issue'}:</strong><br>${descHtml}
                            </div>
                    `;

                    // Duplicate text lines — show as quoted code blocks
                    if (evidence.duplicate_texts && evidence.duplicate_texts.length > 0) {
                        html += '<div class="evidence-quotes"><h4>🔁 Duplicated text:</h4>';
                        for (const line of evidence.duplicate_texts.slice(0, 10)) {
                            const safe = line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                            html += `<div class="evidence-quote" style="font-family:monospace; font-size:0.85em;">${safe}</div>`;
                        }
                        html += '</div>';
                    }

                    // Details list (page_discontinuity, etc.)
                    if (evidence.details && evidence.details.length > 0) {
                        html += '<div class="evidence-quotes"><h4>📋 Details:</h4>';
                        for (const detail of evidence.details) {
                            const safe = detail.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                            html += `<div class="evidence-quote">• ${safe}</div>`;
                        }
                        html += '</div>';
                    }

                    // Quotes if available (LLM-detected issues)
                    if (evidence.quotes && evidence.quotes.length > 0) {
                        html += '<div class="evidence-quotes"><h4>📄 Evidence from document:</h4>';
                        for (const quote of evidence.quotes.slice(0, 3)) {
                            const safe = quote.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                            html += `<div class="evidence-quote">"${safe}"</div>`;
                        }
                        html += '</div>';
                    }

                    // Conflicting values if available
                    if (evidence.conflicting_values && evidence.conflicting_values.length > 0) {
                        html += '<div class="evidence-quotes"><h4>⚠️ Conflicting values:</h4>';
                        for (const conflict of evidence.conflicting_values) {
                            const safe = conflict.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                            html += `<div class="evidence-quote">${safe}</div>`;
                        }
                        html += '</div>';
                    }

                    // Location if available
                    if (evidence.location) {
                        html += `<div class="evidence-location">📍 Found: ${evidence.location}</div>`;
                    }

                    // Impact
                    if (tag.impact) {
                        html += `<div class="evidence-impact"><strong>Impact:</strong> ${tag.impact}</div>`;
                    }

                    // Suggested action
                    if (tag.suggested_action) {
                        html += `<div class="evidence-impact" style="background: #e8f8f5;"><strong>Suggested Action:</strong> ${tag.suggested_action}</div>`;
                    }

                    html += '</div>';
                }

                modalBody.innerHTML = html;
            } catch (error) {
                modalBody.innerHTML = `<div class="no-evidence">Error loading evidence: ${error.message}</div>`;
            }
        }

        function closeTagModal() {
            document.getElementById('tag-evidence-modal').classList.remove('active');
        }

        // Close modal when clicking outside
        document.addEventListener('click', (e) => {
            const modal = document.getElementById('tag-evidence-modal');
            if (e.target === modal) {
                closeTagModal();
            }
        });

// ── Project Selector ────────────────────────────────────────────────────
        // Project Selector Functions
        async function loadProjectSelector() {
            try {
                const response = await apiFetch(apiUrl('/api/projects'));
                const data = await response.json();

                const selector = document.getElementById('project-selector');
                if (!selector) return;

                // Get current project
                const currentResponse = await apiFetch(apiUrl('/api/current-project'));
                const currentProject = await currentResponse.json();

                // Populate dropdown
                selector.innerHTML = data.projects
                    .filter(p => !p.is_archived)
                    .map(p => `<option value="${p.slug}" ${p.slug === currentProject.slug ? 'selected' : ''}>${p.name}</option>`)
                    .join('');

            } catch (error) {
                console.error('Failed to load project selector:', error);
                const selector = document.getElementById('project-selector');
                if (selector) {
                    selector.innerHTML = '<option value="">Error loading projects</option>';
                }
            }
        }

        async function switchProject() {
            const selector = document.getElementById('project-selector');
            const projectSlug = selector.value;

            try {
                const response = await apiFetch(apiUrl('/api/current-project'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ project_slug: projectSlug })
                });

                if (response.ok) {
                    // Reload page to refresh all data for new project
                    window.location.reload();
                } else {
                    alert('Failed to switch project');
                }
            } catch (error) {
                console.error('Failed to switch project:', error);
                alert('Failed to switch project');
            }
        }
