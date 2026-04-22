// ── Tab Switching ────────────────────────────────────────────────────────

        // ── Tab switching function ────────────────────────────────────
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });

            // Deactivate all tab buttons
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });

            // Show selected tab and reset its scroll position
            const selectedTab = document.getElementById('tab-' + tabName);
            if (selectedTab) {
                selectedTab.classList.add('active');
                selectedTab.scrollTop = 0;
            }

            // Activate selected tab button (match by onclick attribute, not text content)
            document.querySelectorAll('.tab-button').forEach(btn => {
                const m = btn.getAttribute('onclick') && btn.getAttribute('onclick').match(/'([^']+)'/);
                if (m && m[1] === tabName) {
                    btn.classList.add('active');
                }
            });

            // Load projects when projects tab is opened
            if (tabName === 'projects') {
                console.log('Projects tab selected, typeof loadProjectsList:', typeof loadProjectsList);
                if (typeof loadProjectsList === 'function') {
                    loadProjectsList();
                } else {
                    console.error('loadProjectsList is not a function!');
                }
            }
            // Config tab: reset to AI Settings sub-tab on each visit
            if (tabName === 'config') {
                switchConfigTab('ai');
            }
            // Tools tab: show Health sub-tab on each visit
            if (tabName === 'tools') {
                switchToolsTab('health');
            }
            // Check API key when AI chat tab is opened
            if (tabName === 'ai-chat') {
                checkAiKeyStatus();
            }
            // Load import history and project list when upload tab is opened
            if (tabName === 'upload') {
                loadImportHistory();
                loadUploadProjects();
            }
            // Load document table when search tab is opened
            if (tabName === 'search') {
                loadSearchTab();
            }
        }


// ── Initial Load Calls ──────────────────────────────────────────────────
        // Initial load
        refresh();
        refreshLogs();
        checkLLMStatus();

        // Auto-refresh every 10 seconds
        refreshInterval = setInterval(refresh, 10000);

        // Auto-refresh logs every 15 seconds
        setInterval(refreshLogs, 15000);

        // Header health widget — initial poll + 30 s auto-poll
        pollHeaderHealth();
        setInterval(pollHeaderHealth, 30000);

        // Load project selector on page load
        loadProjectSelector();
        // Load chat sessions on page load
        loadSessions();

// ── About / Help / Bug Report — state and functions ─────────────────────

        // ── About / Help / Bug Report / SMTP — initialise state first ──────────
        // Variables must be initialised before any code that could throw, so
        // that function bodies that reference them don't hit the TDZ.
        var _helpVisible = false;
        var _helpPanelMap = {
            'overview': 'help-overview',
            'projects': 'help-projects',
            'upload':   'help-upload',
            'ai-chat':  'help-ai-chat',
            'search':   'help-search',
            'config':   'help-config',
            // tools sub-tabs are handled dynamically in _refreshHelpPanel
        };
        var _smtpLoaded = false;

        // ── About Modal ──────────────────────────────────────────────────────
        function openAboutModal() {
            var m = document.getElementById('about-modal');
            if (!m) return;
            m.style.display = 'flex';
            document.getElementById('about-version-line').textContent = 'Loading…';
            document.getElementById('about-components').innerHTML = '<span style="color:#8899aa;">Loading…</span>';
            apiFetch(apiUrl('/api/about')).then(function(r) { return r.json(); }).then(function(d) {
                document.getElementById('about-version-line').textContent = 'Version ' + d.version;
                var comp = d.components || {};
                var rows = '';
                Object.keys(comp).forEach(function(k) {
                    rows += '<span style="color:#aabbcc;">' + escapeHtml(k) + '</span>' +
                            '<span style="color:#7db8f7; font-weight:600;">' + escapeHtml(comp[k]) + '</span>';
                });
                document.getElementById('about-components').innerHTML = rows || '<span style="color:#aabbcc;">—</span>';
            }).catch(function() {
                document.getElementById('about-version-line').textContent = 'Version unknown';
            });
        }
        function closeAboutModal() {
            var m = document.getElementById('about-modal');
            if (m) m.style.display = 'none';
        }

        // ── Tab Help Panels ──────────────────────────────────────────────────
        function toggleTabHelp() {
            _helpVisible = !_helpVisible;
            var btn = document.getElementById('help-toggle-btn');
            if (btn) {
                btn.style.background = _helpVisible
                    ? 'rgba(52,152,219,0.35)'
                    : 'rgba(255,255,255,0.12)';
                btn.textContent = _helpVisible ? '? Help: On' : '? Help: Off';
            }
            _refreshHelpPanel();
        }
        function _refreshHelpPanel() {
            // Hide all help panels (both mapped and sub-tab panels)
            document.querySelectorAll('.tab-help-panel').forEach(function(el) {
                el.style.display = 'none';
            });
            if (!_helpVisible) return;
            var activeTab = document.querySelector('.tab-button.active');
            if (!activeTab) return;
            var m = activeTab.getAttribute('onclick').match(/'([^']+)'/);
            var tabName = m ? m[1] : null;
            if (tabName === 'tools') {
                // Show help for the active tools sub-tab
                var activeSubBtn = document.querySelector('.tools-sub-btn.active');
                if (activeSubBtn) {
                    var sm = activeSubBtn.getAttribute('onclick').match(/'([^']+)'/);
                    var subName = sm ? sm[1] : null;
                    if (subName) {
                        var subEl = document.getElementById('help-tools-' + subName);
                        if (subEl) subEl.style.display = 'block';
                    }
                }
            } else if (tabName === 'config') {
                // Show help for the active config sub-tab; fall back to main config panel
                var activeSubBtn = document.querySelector('.config-sub-btn.active');
                var subName = null;
                if (activeSubBtn) {
                    var sm = activeSubBtn.getAttribute('onclick').match(/'([^']+)'/);
                    subName = sm ? sm[1] : null;
                }
                var subEl = subName ? document.getElementById('help-config-' + subName) : null;
                var fallback = document.getElementById('help-config');
                var panel = subEl || fallback;
                if (panel) panel.style.display = 'block';
            } else if (tabName === 'case-intelligence') {
                // Show help for the active CI sub-tab; fall back to main CI panel
                var activeSubBtn = document.querySelector('.ci-sub-btn.active');
                var subName = null;
                if (activeSubBtn) {
                    var sm = activeSubBtn.getAttribute('onclick').match(/'([^']+)'/);
                    subName = sm ? sm[1] : null;
                }
                var subEl = subName ? document.getElementById('help-ci-' + subName) : null;
                var fallback = document.getElementById('help-ci');
                var panel = subEl || fallback;
                if (panel) panel.style.display = 'block';
            } else {
                var panelId = tabName ? _helpPanelMap[tabName] : null;
                if (panelId) {
                    var el = document.getElementById(panelId);
                    if (el) el.style.display = 'block';
                }
            }
        }
        // Wrap switchTab so the help panel updates, config sub-tabs are reset,
        // and the tools auto-refresh stops when leaving the tools tab.
        (function() {
            var _orig = switchTab;
            switchTab = function(name) {
                _orig(name);
                // Explicitly clear config sub-tab active state when leaving config.
                if (name !== 'config') {
                    document.querySelectorAll('.config-sub-content').forEach(function(c) {
                        c.classList.remove('active');
                    });
                }
                // Stop tools auto-refresh when navigating away from the tools tab
                if (name !== 'tools') {
                    _stopToolsAutoRefresh();
                }
                _refreshHelpPanel();
            };
        })();

// ── Bug Report Modal ─────────────────────────────────────────────────────
        // ── Bug Report Modal ─────────────────────────────────────────────────
        function openBugReportModal() {
            var m = document.getElementById('bug-report-modal');
            if (!m) return;
            m.style.display = 'flex';
            document.getElementById('bug-report-result').style.display = 'none';
            document.getElementById('bug-description').value = '';
            document.getElementById('bug-contact-email').value = '';
            document.getElementById('bug-har-file').value = '';
            document.getElementById('bug-severity').value = 'Medium';
            document.getElementById('bug-include-logs').checked = true;
        }
        function closeBugReportModal() {
            var m = document.getElementById('bug-report-modal');
            if (m) m.style.display = 'none';
        }
        async function submitBugReport() {
            var desc = document.getElementById('bug-description').value.trim();
            var resEl = document.getElementById('bug-report-result');
            function showErr(msg) {
                resEl.style.display = 'block';
                resEl.style.background = 'rgba(231,76,60,0.15)';
                resEl.style.border = '1px solid rgba(231,76,60,0.4)';
                resEl.style.color = '#f08080';
                resEl.textContent = msg;
            }
            if (!desc) { showErr('Please describe the problem before sending.'); return; }
            var email = document.getElementById('bug-contact-email').value.trim();
            if (!email) { showErr('Please enter your email address so we can follow up.'); return; }
            var btn = document.querySelector('#bug-report-modal button[onclick="submitBugReport()"]');
            if (btn) { btn.disabled = true; btn.textContent = 'Sending…'; }
            var fd = new FormData();
            fd.append('description', desc);
            fd.append('severity', document.getElementById('bug-severity').value);
            fd.append('contact_email', document.getElementById('bug-contact-email').value);
            fd.append('include_logs', document.getElementById('bug-include-logs').checked ? 'true' : 'false');
            var harInput = document.getElementById('bug-har-file');
            if (harInput.files && harInput.files[0]) fd.append('har_file', harInput.files[0]);
            try {
                var r = await apiFetch(apiUrl('/api/bug-report'), { method: 'POST', body: fd });
                var d = await r.json();
                resEl.style.display = 'block';
                if (d.ok) {
                    resEl.style.background = 'rgba(39,174,96,0.15)';
                    resEl.style.border = '1px solid rgba(39,174,96,0.4)';
                    resEl.style.color = '#7dcea0';
                    resEl.textContent = d.message;
                    if (btn) { btn.disabled = false; btn.textContent = '📨 Send Report'; }
                    setTimeout(closeBugReportModal, 3000);
                } else {
                    showErr(d.error || 'Failed to send report.');
                    if (btn) { btn.disabled = false; btn.textContent = '📨 Send Report'; }
                }
            } catch (e) {
                showErr('Network error: ' + e.message);
                if (btn) { btn.disabled = false; btn.textContent = '📨 Send Report'; }
            }
        }


// ─────────────────────────────────────────────────────────────────────────
// ── DOMContentLoaded Init ────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────

// ─── API Key Guide modal — conversational agent ─────────────────────────────

const API_KEY_INFO = {
    brave:          {name:'Brave Search API',            tier:'$5 / 1,000 queries',    url:'https://brave.com/search/api/',        desc:'Independent web search index — great for news, entity mentions, and court references without Google bias.'},
    gcse:           {name:'Google Custom Search API',    tier:'100 free / day',         url:'https://programmablesearchengine.google.com/controlpanel/create', desc:'Google search results via API. Requires two credentials: an API Key and a Custom Search Engine ID (CX).'},
    exa:            {name:'Exa AI Search',               tier:'~$7 / 1,000 queries',    url:'https://dashboard.exa.ai/',            desc:'Neural semantic search engine. Excellent for finding similar cases, related articles, and conceptual matches.'},
    perplexity:     {name:'Perplexity Sonar API',        tier:'~$5 / 1,000 queries',    url:'https://www.perplexity.ai/settings/api', desc:'AI-powered search with cited sources. Returns synthesized answers with source links.'},
    tavily:         {name:'Tavily Search API',           tier:'1,000 searches/mo free', url:'https://app.tavily.com/home',          desc:'Research-optimized search API — designed for AI agents. Returns clean, structured results.'},
    serper:         {name:'Serper.dev',                  tier:'2,500 free queries',     url:'https://serper.dev/',                  desc:'Google SERP API. Fast, reliable access to Google search results programmatically.'},
    docket:         {name:'Docket Alarm',                tier:'$99 / month',            url:'https://www.docketalarm.com/accounts/register/', desc:'675M+ federal and state court dockets. Best for searching by party name, attorney, or case type.'},
    unicourt:       {name:'UniCourt API',                tier:'$49–299 / month',        url:'https://unicourt.com/api',             desc:'Normalized federal and state court case data. Uses OAuth2 client credentials (client ID + secret).'},
    fec:            {name:'FEC API (OpenFEC)',            tier:'Free — email signup',    url:'https://api.data.gov/signup/',         desc:'Federal Election Commission campaign finance data. Free API key from api.data.gov — just provide your email.'},
    opensanctions:  {name:'OpenSanctions API',           tier:'~€0.10 per call',        url:'https://www.opensanctions.org/api/',   desc:'Sanctions lists, PEP databases, and watchlists from 100+ sources. Pay-per-use — no subscription required.'},
    opencorp:       {name:'OpenCorporates API',          tier:'Paid subscription',      url:'https://opencorporates.com/api_accounts/new', desc:'200M+ global business entity registrations. Useful for finding related companies and shell corporations.'},
    clear:          {name:'CLEAR (Thomson Reuters)',     tier:'Enterprise contract',    url:'https://legal.thomsonreuters.com/en/products/clear-investigation-software', desc:'Comprehensive investigative database — people, assets, court records, news. Requires a Thomson Reuters enterprise contract.'},
    newsapi:        {name:'NewsAPI',                     tier:'$449/mo (commercial)',   url:'https://newsapi.org/register',         desc:'150,000+ news sources. Note: the free developer plan is non-commercial only and limited to 100 requests/day.'},
    lexis:          {name:'LexisNexis API',              tier:'Enterprise license',     url:'https://developer.lexisnexis.com/',    desc:'Comprehensive US and international legal research. Requires an existing LexisNexis subscription + API access approval.'},
    vlex:           {name:'vLex API',                    tier:'Subscription required',  url:'https://vlex.com/api',                 desc:'100+ countries of case law and legal content. Contact vLex for API access as part of a subscription plan.'},
    westlaw:        {name:'Westlaw Edge API',            tier:'Enterprise license',     url:'https://legal.thomsonreuters.com/en/products/westlaw', desc:'Premier US legal research database. API access requires a Thomson Reuters Westlaw contract and developer enrollment.'},
    nysenate:       {name:'NY Senate Open Legislation',  tier:'Free — email signup',    url:'https://legislation.nysenate.gov/register', desc:'Access to all NYS statutes (CPLR, DRL, EPTL, etc.). Free API token — just register with your email address.'},
    courtlistener:  {name:'CourtListener API',           tier:'Free account',           url:'https://www.courtlistener.com/sign-in/', desc:'RECAP archive: 6M+ court opinions and PACER documents. Free account gives higher rate limits. Unauthenticated access also works but is throttled.'},
    cohere:         {name:'Cohere Embed API',            tier:'Free trial / pay-per-use', url:'https://dashboard.cohere.com/register', desc:'Used to embed legal authorities into ChromaDB for semantic search. Free trial includes 1,000 calls/month; paid is $0.10 per 1M tokens.'},
};

// Maps service key → which form fields to auto-fill when credentials arrive
const AKGM_FIELD_MAP = {
    brave:          [{input:'ci-wr-brave-key',         cred:'key'},                                           {cb:'ci-wr-brave-cb'}],
    gcse:           [{input:'ci-wr-gcse-key',          cred:'key'}, {input:'ci-wr-gcse-cx', cred:'cx'},       {cb:'ci-wr-gcse-cb'}],
    exa:            [{input:'ci-wr-exa-key',           cred:'key'},                                           {cb:'ci-wr-exa-cb'}],
    perplexity:     [{input:'ci-wr-perplexity-key',    cred:'key'},                                           {cb:'ci-wr-perplexity-cb'}],
    tavily:         [{input:'ci-wr-tavily-key',        cred:'key'},                                           {cb:'ci-wr-tavily-cb'}],
    serper:         [{input:'ci-wr-serper-key',        cred:'key'},                                           {cb:'ci-wr-serper-cb'}],
    fec:            [{input:'ci-wr-fec-key',           cred:'key'},                                           {cb:'ci-wr-fec-cb'}],
    opensanctions:  [{input:'ci-wr-opensanctions-key', cred:'key'},                                           {cb:'ci-wr-opensanctions-cb'}],
    opencorp:       [{input:'ci-wr-opencorporates-key',cred:'key'},                                           {cb:'ci-wr-opencorporates-cb'}],
    docket:         [{input:'ci-wr-docket-user',       cred:'user'},{input:'ci-wr-docket-pass', cred:'pass'}, {cb:'ci-wr-docket-cb'}],
    unicourt:       [{input:'ci-wr-unicourt-id',       cred:'client_id'},{input:'ci-wr-unicourt-secret', cred:'client_secret'}, {cb:'ci-wr-unicourt-cb'}],
    newsapi:        [{input:'ci-wr-newsapi-key',       cred:'key'},                                           {cb:'ci-wr-newsapi-cb'}],
    courtlistener:  [{input:'ci-wr-courtlistener-key', cred:'key'},                                           {cb:'ci-wr-courtlistener-cb'}],
};

let _akgmCurrentService = null;
let _akgmMessages = [];   // [{role:'user'|'assistant', content:'...'}]
let _akgmDone = false;

function openApiKeyGuide(svc) {
    const info = API_KEY_INFO[svc];
    if (!info) return;
    _akgmCurrentService = svc;
    _akgmMessages = [];
    _akgmDone = false;

    document.getElementById('akgm-title').textContent = info.name;
    document.getElementById('akgm-tier').textContent = info.tier;
    const link = document.getElementById('akgm-reglink');
    link.href = info.url;
    document.getElementById('akgm-chat').innerHTML = '';
    document.getElementById('akgm-success-bar').style.display = 'none';
    document.getElementById('akgm-input-area').style.display = 'flex';
    document.getElementById('akgm-input').value = '';
    document.getElementById('akgm-send-btn').disabled = false;

    // Show modal as flex
    document.getElementById('akgm-overlay').style.display = 'block';
    const modal = document.getElementById('akgm-modal');
    modal.style.display = 'flex';

    // Kick off the conversation with an empty user turn so AI greets first
    akgmCallAI('');
}

function closeApiKeyGuide() {
    document.getElementById('akgm-overlay').style.display = 'none';
    document.getElementById('akgm-modal').style.display = 'none';
    _akgmCurrentService = null;
}

function _akgmLinkify(text) {
    // Escape HTML, then convert URLs to clickable links
    const escaped = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    return escaped.replace(/(https?:\/\/[^\s"<>]+)/g,
        '<a href="$1" target="_blank" rel="noopener" style="color:#1565c0;text-decoration:underline;word-break:break-all;">$1</a>');
}

function _akgmAppendBubble(role, text) {
    const chat = document.getElementById('akgm-chat');
    const isUser = role === 'user';
    const div = document.createElement('div');
    div.style.cssText = `display:flex; justify-content:${isUser ? 'flex-end' : 'flex-start'};`;
    const bubble = document.createElement('div');
    bubble.style.cssText = `
        max-width:85%; padding:9px 13px; border-radius:${isUser ? '14px 14px 4px 14px' : '14px 14px 14px 4px'};
        background:${isUser ? '#2c3e50' : '#f0f4ff'}; color:${isUser ? '#fff' : '#2c3e50'};
        font-size:13px; line-height:1.55; white-space:pre-wrap; word-break:break-word;
    `;
    bubble.innerHTML = _akgmLinkify(text);
    div.appendChild(bubble);
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

async function akgmSend() {
    if (_akgmDone) return;
    const input = document.getElementById('akgm-input');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    _akgmAppendBubble('user', text);
    _akgmMessages.push({role: 'user', content: text});
    await akgmCallAI(null);
}

async function akgmCallAI(initialPrompt) {
    document.getElementById('akgm-send-btn').disabled = true;
    document.getElementById('akgm-typing').style.display = 'block';

    const msgs = initialPrompt === '' ? [] : _akgmMessages;

    try {
        const r = await apiFetch(apiUrl('/api/ci/key-guide'), {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                service: _akgmCurrentService,
                service_name: (API_KEY_INFO[_akgmCurrentService] || {}).name || _akgmCurrentService,
                messages: msgs,
            }),
        });
        const d = await r.json();
        document.getElementById('akgm-typing').style.display = 'none';

        if (d.error) {
            _akgmAppendBubble('assistant', '⚠️ ' + d.error);
        } else {
            const reply = d.response || '';
            _akgmAppendBubble('assistant', reply);
            _akgmMessages.push({role: 'assistant', content: reply});

            if (d.credentials && Object.keys(d.credentials).length > 0) {
                akgmApplyCredentials(d.credentials);
            }
        }
    } catch(e) {
        document.getElementById('akgm-typing').style.display = 'none';
        _akgmAppendBubble('assistant', '⚠️ Network error — please try again.');
    }

    if (!_akgmDone) {
        document.getElementById('akgm-send-btn').disabled = false;
        document.getElementById('akgm-input').focus();
    }
}

function akgmApplyCredentials(creds) {
    const svc = _akgmCurrentService;
    const fields = AKGM_FIELD_MAP[svc];
    if (!fields) return;

    let filled = 0;
    for (const f of fields) {
        if (f.input && f.cred && creds[f.cred]) {
            const el = document.getElementById(f.input);
            if (el) { el.value = creds[f.cred]; filled++; }
        }
        if (f.cb) {
            const cb = document.getElementById(f.cb);
            if (cb) cb.checked = true;
        }
    }

    if (filled > 0) {
        _akgmDone = true;
        document.getElementById('akgm-input-area').style.display = 'none';
        document.getElementById('akgm-success-bar').style.display = 'block';
        // Enable the WR section if not already
        const wrEnabled = document.getElementById('ci-wr-enabled');
        if (wrEnabled) wrEnabled.checked = true;
        // Auto-close after 3 seconds
        setTimeout(() => closeApiKeyGuide(), 3500);
        if (typeof showToast === 'function') {
            showToast('API key saved and enabled for ' + (API_KEY_INFO[svc]||{}).name, 'success');
        }
    }
}

// close on Escape
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeApiKeyGuide(); });
