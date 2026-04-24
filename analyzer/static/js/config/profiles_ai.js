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

