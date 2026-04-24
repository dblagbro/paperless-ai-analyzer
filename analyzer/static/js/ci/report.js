    // ── Report builder ───────────────────────────────────────────────
    let ciCurrentReportId = null;
    let ciReportPollTimer = null;

    async function ciGenerateReport() {
        if (!window.APP_CONFIG.isAdvanced || !ciSelectedRunId) return;
        const instructions = document.getElementById('ci-report-instructions').value.trim();
        if (!instructions) { showToast('Please enter report instructions', 'warning'); return; }
        const template = document.getElementById('ci-report-template').value;

        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + ciSelectedRunId + '/reports'), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ instructions, template }),
            });
            const d = await r.json();
            if (d.report_id) {
                ciCurrentReportId = d.report_id;
                document.getElementById('ci-report-status').style.display = 'block';
                document.getElementById('ci-report-status').textContent = 'Generating report...';
                document.getElementById('ci-report-preview-section').style.display = 'none';
                if (ciReportPollTimer) clearInterval(ciReportPollTimer);
                ciReportPollTimer = setInterval(() => ciPollReport(ciSelectedRunId, ciCurrentReportId), 3000);
            } else {
                showToast(d.error || 'Failed to start report generation', 'error');
            }
        } catch(e) {
            showToast('Error: ' + e.message, 'error');
        }
    }

    async function ciPollReport(runId, reportId) {
        try {
            const r = await apiFetch(apiUrl('/api/ci/runs/' + runId + '/reports/' + reportId));
            const d = await r.json();
            if (d.status === 'complete' && d.content) {
                clearInterval(ciReportPollTimer);
                document.getElementById('ci-report-status').textContent = 'Report ready!';
                document.getElementById('ci-report-preview-section').style.display = 'block';
                document.getElementById('ci-report-preview').textContent = d.content.substring(0, 2000) + (d.content.length > 2000 ? '\n\n[... truncated for preview ...]' : '');
                const pdfLink = document.getElementById('ci-report-pdf-link');
                pdfLink.style.display = 'inline';
                pdfLink.href = apiUrl('/api/ci/runs/' + runId + '/reports/' + reportId + '/pdf');
            } else if (d.status === 'failed') {
                clearInterval(ciReportPollTimer);
                document.getElementById('ci-report-status').textContent = 'Report generation failed.';
            }
        } catch(e) {}
    }

    // ── Tab open hook ────────────────────────────────────────────────
    const _origSwitchTab = window.switchTab;
    if (_origSwitchTab) {
        window.switchTab = function(tab) {
            _origSwitchTab(tab);
            if (tab === 'case-intelligence') {
                ciLoadJurisdictions().then(() => ciDetectJurisdiction());
                ciRefreshRuns();
            }
        };
    }

    function escapeHtml(s) {
        if (!s) return '';
        return String(s)
            .replace(/&/g,'&amp;')
            .replace(/</g,'&lt;')
            .replace(/>/g,'&gt;')
            .replace(/"/g,'&quot;');
    }

