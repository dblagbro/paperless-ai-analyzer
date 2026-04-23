// LLM Proxy admin sub-tab — matches users.js pattern (list / add / edit inline / delete / test).
// URLs are wrapped with apiUrl() to prepend the app's sub-path prefix.

async function loadLLMProxy() {
    try {
        const resp = await apiFetch(apiUrl('/api/llm-proxy/endpoints'));
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        renderLLMProxyTable(data.endpoints || []);
    } catch (e) {
        console.error('Failed to load LLM proxy endpoints:', e);
        const tbody = document.getElementById('llm-proxy-table-body');
        if (tbody) tbody.innerHTML = `<tr><td colspan="8" style="color:#c00; padding:12px;">Failed to load: ${e}</td></tr>`;
    }
}

function renderLLMProxyTable(eps) {
    const tbody = document.getElementById('llm-proxy-table-body');
    if (!tbody) return;
    if (!eps.length) {
        tbody.innerHTML = `<tr><td colspan="8" style="color:#888; padding:12px;">No endpoints configured yet. Click "+ Add Endpoint".</td></tr>`;
        return;
    }
    tbody.innerHTML = eps.map(e => {
        const breaker = e.breaker || {};
        const breakerLabel = breaker.tripped
            ? `<span style="color:#c00;" title="Cooling down for ${breaker.cooldown_remaining_sec}s">⚠ tripped (${breaker.cooldown_remaining_sec}s)</span>`
            : `<span style="color:#3a7;">✓ ok</span>`;
        const enabledToggle = `<input type="checkbox" ${e.enabled?'checked':''} onchange="toggleLLMProxyEnabled('${e.id}', this.checked)">`;
        return `
            <tr data-ep-id="${e.id}">
                <td><input type="text" value="${escapeHTML(e.label)}" onblur="patchLLMProxyField('${e.id}','label',this.value)" style="width:98%;padding:4px;"></td>
                <td><input type="text" value="${escapeHTML(e.url)}" onblur="patchLLMProxyField('${e.id}','url',this.value)" style="width:98%;padding:4px;"></td>
                <td><select onchange="patchLLMProxyField('${e.id}','version',parseInt(this.value))" style="padding:4px;">
                    <option value="1" ${e.version===1?'selected':''}>v1</option>
                    <option value="2" ${e.version===2?'selected':''}>v2</option>
                </select></td>
                <td><input type="number" value="${e.priority}" onblur="patchLLMProxyField('${e.id}','priority',parseInt(this.value))" style="width:60px;padding:4px;"></td>
                <td style="text-align:center;">${enabledToggle}</td>
                <td style="font-family:monospace; font-size:12px;">${escapeHTML(e.api_key_masked || '—')}</td>
                <td>${breakerLabel}</td>
                <td style="display:flex; gap:6px;">
                    <button class="btn" style="padding:4px 10px;" onclick="testLLMProxy('${e.id}', this)">Test</button>
                    <button class="btn" style="padding:4px 10px; color:#c00;" onclick="deleteLLMProxy('${e.id}')">Delete</button>
                </td>
            </tr>
        `;
    }).join('');
}

function escapeHTML(s) {
    return String(s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function showAddLLMProxyForm() {
    document.getElementById('add-llm-proxy-form').style.display = 'block';
    document.getElementById('llm-proxy-add-label').focus();
}

function hideAddLLMProxyForm() {
    document.getElementById('add-llm-proxy-form').style.display = 'none';
    ['llm-proxy-add-label','llm-proxy-add-url','llm-proxy-add-key'].forEach(id => document.getElementById(id).value = '');
    document.getElementById('llm-proxy-add-priority').value = '10';
}

async function createLLMProxy() {
    const payload = {
        label: document.getElementById('llm-proxy-add-label').value.trim(),
        url:   document.getElementById('llm-proxy-add-url').value.trim(),
        api_key: document.getElementById('llm-proxy-add-key').value.trim(),
        version: parseInt(document.getElementById('llm-proxy-add-version').value),
        priority: parseInt(document.getElementById('llm-proxy-add-priority').value) || 10,
        enabled: document.getElementById('llm-proxy-add-enabled').value === 'true',
    };
    if (!payload.label || !payload.url || !payload.api_key) {
        alert('Label, URL, and API key are required.');
        return;
    }
    try {
        const resp = await apiFetch(apiUrl('/api/llm-proxy/endpoints'), {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        hideAddLLMProxyForm();
        loadLLMProxy();
    } catch (e) {
        alert(`Failed to create endpoint: ${e}`);
    }
}

async function patchLLMProxyField(id, field, value) {
    const body = {};
    body[field] = value;
    try {
        const resp = await apiFetch(apiUrl(`/api/llm-proxy/endpoints/${id}`), {
            method: 'PATCH',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({error: 'save failed'}));
            alert(`Save failed: ${data.error}`);
            loadLLMProxy();  // revert UI
        }
    } catch (e) {
        alert(`Save failed: ${e}`);
        loadLLMProxy();
    }
}

async function toggleLLMProxyEnabled(id, enabled) {
    await patchLLMProxyField(id, 'enabled', enabled);
}

async function deleteLLMProxy(id) {
    if (!confirm('Delete this endpoint?')) return;
    try {
        const resp = await apiFetch(apiUrl(`/api/llm-proxy/endpoints/${id}`), {method:'DELETE'});
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        loadLLMProxy();
    } catch (e) {
        alert(`Delete failed: ${e}`);
    }
}

async function testLLMProxy(id, btn) {
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Testing…';
    try {
        const resp = await apiFetch(apiUrl(`/api/llm-proxy/endpoints/${id}/test`), {method: 'POST'});
        const data = await resp.json();
        if (data.ok) {
            btn.style.background = '#3a7';
            btn.style.color = 'white';
            btn.textContent = `✓ ${data.model || 'ok'}`;
        } else {
            btn.style.background = '#c33';
            btn.style.color = 'white';
            btn.textContent = `✗ failed`;
            btn.title = data.error || '';
        }
    } catch (e) {
        btn.style.background = '#c33';
        btn.style.color = 'white';
        btn.textContent = `✗ error`;
        btn.title = String(e);
    } finally {
        setTimeout(() => {
            btn.disabled = false;
            btn.textContent = originalText;
            btn.style.background = '';
            btn.style.color = '';
            btn.title = '';
            loadLLMProxy();  // refresh to show breaker state
        }, 3000);
    }
}
