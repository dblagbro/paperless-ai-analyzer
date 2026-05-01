#!/usr/bin/env python3
"""
Paperless AI Analyzer — Multi-Instance Medium-Depth Test
Checks ~40 endpoints per instance across Auth, Status, Projects, Users,
Chat, Search, Reconcile, LLM, CI, Upload, Court, Docs.
"""
import json, sys, time
from datetime import datetime
from playwright.sync_api import sync_playwright

INSTANCES = [
    {"name": "PROD",     "base": "http://localhost:8051/paperless-ai-analyzer"},
    {"name": "DEV",      "base": "http://localhost:8052/paperless-ai-analyzer-dev"},
    {"name": "JACOB/QA", "base": "http://localhost:8053/paperless-ai-analyzer-jacob"},
]

ADMIN_USER = "dblagbro"
ADMIN_PASS = "Super*120120"

RESET=GREEN=RED=YELLOW=CYAN=BOLD="\033[0m"
RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"

all_results = {}

def check(results, label, condition, detail=""):
    if condition:
        results.append({"label": label, "pass": True, "detail": detail})
        print(f"  {GREEN}✓{RESET} {label}" + (f"  ({detail})" if detail else ""))
    else:
        results.append({"label": label, "pass": False, "detail": detail})
        print(f"  {RED}✗{RESET} {label}" + (f"  — {detail}" if detail else ""))

def warn(results, label, detail=""):
    results.append({"label": label, "pass": None, "detail": detail})
    print(f"  {YELLOW}⚠{RESET} {label}" + (f"  — {detail}" if detail else ""))

def section(name):
    print(f"\n  {CYAN}── {name} ──{RESET}")

def run_instance(browser, inst):
    base = inst["base"]
    name = inst["name"]
    print(f"\n{'═'*68}")
    print(f"  {BOLD}{name}{RESET}  —  {base}")
    print(f"{'═'*68}")

    results = []
    ctx = browser.new_context()
    page = ctx.new_page()

    def api(method, path, timeout_ms=20000, **kw):
        return ctx.request.fetch(f"{base}{path}", method=method, timeout=timeout_ms, **kw)

    # ── 0 Reachable ─────────────────────────────────────────────────
    section("Reachability")
    try:
        r = api("GET", "/health")
        check(results, "GET /health", r.status == 200, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /health", False, str(e)[:90])
        all_results[name] = results
        ctx.close()
        return

    # ── 1 Auth ───────────────────────────────────────────────────────
    section("Authentication")
    try:
        page.goto(f"{base}/login", timeout=12000, wait_until="domcontentloaded")
        page.fill("input[name=username]", ADMIN_USER)
        page.fill("input[name=password]", ADMIN_PASS)
        with page.expect_navigation(timeout=12000):
            page.click("button[type=submit]")
        check(results, "Admin login", "login" not in page.url, f"url={page.url[-50:]}")
    except Exception as e:
        check(results, "Admin login", False, str(e)[:90])
        all_results[name] = results
        ctx.close()
        return

    try:
        r = api("GET", "/api/me")
        check(results, "GET /api/me → 200", r.status == 200)
        if r.status == 200:
            me = r.json()
            check(results, "  /api/me: role=admin", me.get("role") == "admin", f"role={me.get('role')}")
            check(results, "  /api/me: username=dblagbro", me.get("username") == "dblagbro")
    except Exception as e:
        check(results, "GET /api/me", False, str(e)[:90])

    # ── 2 Status / About / System Health ─────────────────────────────
    section("Status & Health")
    try:
        r = api("GET", "/api/status")
        if r.status == 200:
            s = r.json()
            check(results, "GET /api/status → 200", True, f"svc={s.get('status')}")
            check(results, "  status has paperless_total",  "paperless_total" in s)
            check(results, "  status has vector_store",    "vector_store" in s)
            check(results, "  status has stats",           "stats" in s)
        else:
            check(results, "GET /api/status → 200", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/status", False, str(e)[:90])

    try:
        r = api("GET", "/api/about")
        if r.status == 200:
            v = r.json().get("version", "?")
            check(results, "GET /api/about → 200", True, f"version={v}")
            check(results, "  about version = 3.8.1", v == "3.8.1", f"got {v}")
        else:
            check(results, "GET /api/about → 200", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/about", False, str(e)[:90])

    try:
        r = api("GET", "/api/system-health")
        if r.status == 200:
            sh = r.json()
            comps = sh.get("components", sh)
            check(results, "GET /api/system-health → 200", True)
            for key in ("paperless_api", "chromadb", "llm", "postgres", "redis"):
                if key in comps:
                    c = comps[key]
                    is_healthy = c.get("is_healthy", c.get("healthy"))
                    status_str = c.get("status", "")
                    ok = is_healthy in (True, "ok", "healthy") or status_str in ("ok", "healthy")
                    check(results, f"  component[{key}]", ok, f"{status_str or is_healthy}")
        else:
            check(results, "GET /api/system-health → 200", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/system-health", False, str(e)[:90])

    # ── 3 Projects ───────────────────────────────────────────────────
    section("Projects")
    try:
        r = api("GET", "/api/projects")
        if r.status == 200:
            pd = r.json()
            projects = pd.get("projects", pd) if isinstance(pd, dict) else pd
            active = [p for p in projects if not p.get("is_archived", False)]
            check(results, "GET /api/projects → 200", True, f"{len(active)} active")
            junk = [p["slug"] for p in active if p["slug"].startswith(("pw-", "xss-", "this-is-a-test"))]
            check(results, "  no leftover test projects", not junk, f"junk: {junk}" if junk else "clean")
        else:
            check(results, "GET /api/projects → 200", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/projects", False, str(e)[:90])

    try:
        r = api("GET", "/api/current-project")
        if r.status == 200:
            cp = r.json()
            check(results, "GET /api/current-project → 200", True, f"{cp.get('slug')} ({cp.get('document_count','?')} docs)")
        else:
            check(results, "GET /api/current-project → 200", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/current-project", False, str(e)[:90])

    # ── 4 Users / RBAC ───────────────────────────────────────────────
    section("Users & RBAC")
    try:
        r = api("GET", "/api/users")
        if r.status == 200:
            u = r.json()
            users = u.get("users", u) if isinstance(u, dict) else u
            check(results, "GET /api/users → 200 (admin)", True, f"{len(users)} users")
        else:
            check(results, "GET /api/users → 200 (admin)", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/users", False, str(e)[:90])

    # ── 5 Search ─────────────────────────────────────────────────────
    section("Search")
    try:
        r = api("GET", "/api/search?q=invoice")
        if r.status == 200:
            d = r.json()
            results_list = d.get("results", d) if isinstance(d, dict) else d
            check(results, "GET /api/search?q=invoice → 200", True, f"{len(results_list) if isinstance(results_list, list) else '?'} hits")
        else:
            check(results, "GET /api/search → 200", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/search", False, str(e)[:90])

    # ── 6 LLM / AI Config ────────────────────────────────────────────
    section("LLM & AI Config")
    try:
        r = api("GET", "/api/ai-config")
        if r.status == 200:
            c = r.json()
            provider = c.get("provider", c.get("chat_provider", "?"))
            check(results, "GET /api/ai-config → 200", True, f"provider={provider}")
        else:
            check(results, "GET /api/ai-config → 200", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/ai-config", False, str(e)[:90])

    try:
        r = api("GET", "/api/llm-usage/stats")
        if r.status == 200:
            d = r.json()
            tc = d.get("total_cost", "?")
            check(results, "GET /api/llm-usage/stats → 200", True, f"total_cost={tc}")
        else:
            check(results, "GET /api/llm-usage/stats → 200", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/llm-usage/stats", False, str(e)[:90])

    # ── 7 Reconcile (the big one) ────────────────────────────────────
    section("Reconcile")
    try:
        t0 = time.time()
        r = api("POST", "/api/reconcile", timeout_ms=90000)
        dt = time.time() - t0
        if r.status == 200:
            d = r.json()
            check(results, f"POST /api/reconcile → 200", True, f"{dt:.1f}s, chroma={d.get('chroma_count','?')} paperless={d.get('paperless_count','?')}")
            check(results, "  reconcile completes under 60s", dt < 60, f"{dt:.1f}s")
        else:
            check(results, "POST /api/reconcile → 200", False, f"HTTP {r.status} after {dt:.1f}s")
    except Exception as e:
        check(results, "POST /api/reconcile", False, str(e)[:90])

    # ── 8 Chat ───────────────────────────────────────────────────────
    section("Chat")
    try:
        r = api("GET", "/api/chat/sessions")
        check(results, "GET /api/chat/sessions → 200", r.status == 200, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/chat/sessions", False, str(e)[:90])

    session_id = None
    try:
        r = api("POST", "/api/chat/sessions",
                data=json.dumps({"title": f"medium-test-{name}-{int(time.time())}"}),
                headers={"Content-Type": "application/json"})
        if r.status in (200, 201):
            d = r.json()
            session_id = d.get("session_id", d.get("id"))
            check(results, "POST /api/chat/sessions → 200/201", True, f"id={session_id}")
        else:
            check(results, "POST /api/chat/sessions → 200/201", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "POST /api/chat/sessions", False, str(e)[:90])

    # Send a tiny message to verify LLM end-to-end
    if session_id:
        try:
            r = api("POST", "/api/chat",
                    data=json.dumps({"session_id": session_id, "message": "reply with just the word 'pong'"}),
                    headers={"Content-Type": "application/json"},
                    timeout_ms=60000)
            if r.status == 200:
                d = r.json()
                content = str(d.get("response", d.get("message", d.get("content", ""))))[:80]
                check(results, "POST /api/chat (LLM reply) → 200", True, f"reply={content!r}")
            else:
                check(results, "POST /api/chat → 200", False, f"HTTP {r.status}: {r.text()[:80]}")
        except Exception as e:
            check(results, "POST /api/chat", False, str(e)[:90])

        # Clean up
        try:
            api("DELETE", f"/api/chat/sessions/{session_id}")
        except Exception:
            pass

    # ── 9 Case Intelligence ──────────────────────────────────────────
    section("Case Intelligence")
    try:
        r = api("GET", "/api/ci/runs")
        if r.status == 200:
            d = r.json()
            runs = d.get("runs", d) if isinstance(d, dict) else d
            check(results, "GET /api/ci/runs → 200", True, f"{len(runs) if isinstance(runs, list) else '?'} runs")
        else:
            check(results, "GET /api/ci/runs → 200", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/ci/runs", False, str(e)[:90])

    try:
        r = api("GET", "/api/ci/status")
        check(results, "GET /api/ci/status → 200", r.status == 200, f"HTTP {r.status}")
    except Exception as e:
        warn(results, "GET /api/ci/status", str(e)[:80])

    # ── 10 Upload / Court ────────────────────────────────────────────
    section("Upload & Court")
    try:
        r = api("GET", "/api/upload/history")
        check(results, "GET /api/upload/history → 200", r.status == 200, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/upload/history", False, str(e)[:90])

    try:
        r = api("GET", "/api/court/credentials")
        check(results, "GET /api/court/credentials → 200", r.status == 200, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/court/credentials", False, str(e)[:90])

    # ── 11 Profiles / Vector ─────────────────────────────────────────
    section("Profiles & Vector")
    try:
        r = api("GET", "/api/profiles")
        check(results, "GET /api/profiles → 200", r.status == 200, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/profiles", False, str(e)[:90])

    try:
        r = api("GET", "/api/vector/documents")
        check(results, "GET /api/vector/documents → 200", r.status == 200, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /api/vector/documents", False, str(e)[:90])

    # ── 12 Docs / AI Form ────────────────────────────────────────────
    section("Docs & AI Form")
    try:
        r = api("GET", "/docs/")
        check(results, "GET /docs/ → 200", r.status == 200, f"HTTP {r.status}")
    except Exception as e:
        check(results, "GET /docs/", False, str(e)[:90])

    try:
        r = api("POST", "/api/docs/ask",
                data=json.dumps({"question": "What features are in v3.8.1?"}),
                headers={"Content-Type": "application/json"}, timeout_ms=45000)
        if r.status == 200:
            d = r.json()
            a = str(d.get("answer", ""))[:100]
            check(results, "POST /api/docs/ask → 200", True, f"answer: {a!r}")
        else:
            check(results, "POST /api/docs/ask → 200", False, f"HTTP {r.status}")
    except Exception as e:
        check(results, "POST /api/docs/ask", False, str(e)[:90])

    # ── 13 UI tab clicks ─────────────────────────────────────────────
    section("UI Tab Interaction")
    console_errs = []
    page.on("console", lambda m: console_errs.append(m.text) if m.type == "error" else None)
    try:
        page.goto(f"{base}/", timeout=15000, wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        for tab in ["Overview", "Config", "AI Chat", "Smart Upload", "Case Intelligence"]:
            try:
                btn = page.locator(f"button.tab-btn:has-text('{tab}')").first
                if btn.is_visible(timeout=1200):
                    btn.click()
                    page.wait_for_timeout(600)
                    check(results, f"Tab '{tab}' clickable", True)
                else:
                    warn(results, f"Tab '{tab}' not visible")
            except Exception as e:
                check(results, f"Tab '{tab}' clickable", False, str(e)[:80])
        check(results, "Zero JS console errors across tabs", not console_errs,
              f"{len(console_errs)} errors: {console_errs[:2]}" if console_errs else "")
    except Exception as e:
        check(results, "UI tab interaction", False, str(e)[:90])

    all_results[name] = results
    ctx.close()


def main():
    print(f"\n{'═'*68}")
    print(f"  {BOLD}Paperless AI Analyzer — Medium-Depth Test{RESET}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Instances: PROD, DEV, JACOB/QA  —  ~35 checks each")
    print(f"{'═'*68}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for inst in INSTANCES:
            run_instance(browser, inst)
        browser.close()

    # ── Roll-up ───────────────────────────────────────────────────────
    print(f"\n\n{'═'*68}")
    print(f"  {BOLD}SUMMARY{RESET}")
    print(f"{'═'*68}")
    for inst_name, r_list in all_results.items():
        passed  = sum(1 for r in r_list if r["pass"] is True)
        failed  = sum(1 for r in r_list if r["pass"] is False)
        warnings = sum(1 for r in r_list if r["pass"] is None)
        total = len(r_list)
        color = GREEN if failed == 0 else (YELLOW if failed <= 3 else RED)
        print(f"\n  {color}{BOLD}{inst_name}{RESET}  —  {passed}/{total} passed  |  {failed} failed  |  {warnings} warnings")
        if failed > 0:
            for r in r_list:
                if r["pass"] is False:
                    print(f"    {RED}✗{RESET} {r['label']}  {r['detail']}")

    # JSON report
    path = f"/tmp/instance_medium_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full report: {path}\n")

    total_failed = sum(1 for rl in all_results.values() for r in rl if r["pass"] is False)
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
