"""
Report Generator for Case Intelligence AI.

Generates attorney-ready memos and reports from CI findings.
Supports Markdown output and PDF download via weasyprint.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional
import os

from analyzer.case_intelligence.db import (
    get_ci_run, get_ci_entities, get_ci_timeline, get_ci_contradictions,
    get_ci_theories, get_ci_authorities, get_ci_disputed_facts,
    update_ci_report,
)
from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

REPORT_GENERATION_PROMPT = """You are a senior litigation attorney drafting a legal document.

Generate a {template_label} based on the CI findings below. Follow the user's instructions exactly.

USER INSTRUCTIONS:
{instructions}

CASE METADATA:
- Role: {role}
- Goal: {goal_text}
- Jurisdiction: {jurisdiction}
- Documents analyzed: {docs_total}
- Run date: {run_date}

ENTITIES (Cast of Characters):
{entities_summary}

TIMELINE (Key Events):
{timeline_summary}

FINANCIAL FACTS:
{financial_summary}

CONTRADICTIONS FOUND:
{contradictions_summary}

THEORIES (with confidence):
{theories_summary}

LEGAL AUTHORITIES:
{authorities_summary}

DISPUTED FACTS:
{disputed_facts_summary}

REQUIREMENTS:
1. Every claim MUST cite a document (Doc #ID, Page N).
2. Use proper headings and professional legal writing style.
3. Do NOT include speculative statements without clearly marking them as "NOTE: unconfirmed."
4. Format as Markdown with clear section headers.
5. Include a "Document Citations" section at the end listing all cited docs.
6. Keep to the user's instructions — don't add sections they didn't ask for.
"""

TEMPLATE_LABELS = {
    'attorney_memo': 'Attorney Working Memo',
    'chronology': 'Case Chronology',
    'financial': 'Financial Analysis Report',
    'deposition': 'Deposition Preparation Guide',
    'custom': 'Custom Report',
}


class ReportGenerator:
    """
    Generates CI reports using gpt-4o (escalates to Claude).
    Optionally converts to PDF using weasyprint.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('report_generation')

    def generate(self, run_id: str, report_id: str, instructions: str,
                 template: str = 'custom') -> bool:
        """
        Generate a report for a CI run.
        Updates ci_reports table with content when complete.

        Returns True on success.
        """
        run = get_ci_run(run_id)
        if not run:
            logger.error(f"ReportGenerator: run {run_id} not found")
            update_ci_report(report_id, content='Run not found.', status='failed')
            return False

        try:
            # Gather findings
            entities = get_ci_entities(run_id)
            timeline = get_ci_timeline(run_id)
            contradictions = get_ci_contradictions(run_id)
            theories = get_ci_theories(run_id)
            authorities = get_ci_authorities(run_id)
            disputed_facts = get_ci_disputed_facts(run_id)

            # Build summaries
            entities_summary = self._summarize_entities(entities)
            timeline_summary = self._summarize_timeline(timeline)
            contradictions_summary = self._summarize_contradictions(contradictions)
            theories_summary = self._summarize_theories(theories)
            authorities_summary = self._summarize_authorities(authorities)
            disputed_facts_summary = self._summarize_disputed_facts(disputed_facts)
            financial_summary = run['findings_summary'] or 'No financial summary available.'

            jurisdiction_json = run['jurisdiction_json'] or '{}'
            try:
                jd = json.loads(jurisdiction_json)
                jurisdiction = jd.get('display_name', 'Not specified')
            except Exception:
                jurisdiction = 'Not specified'

            prompt = REPORT_GENERATION_PROMPT.format(
                template_label=TEMPLATE_LABELS.get(template, 'Report'),
                instructions=instructions,
                role=run['role'],
                goal_text=run['goal_text'] or 'Not specified',
                jurisdiction=jurisdiction,
                docs_total=run['docs_total'] or 0,
                run_date=run['created_at'],
                entities_summary=entities_summary[:2000],
                timeline_summary=timeline_summary[:2000],
                financial_summary=financial_summary[:1000],
                contradictions_summary=contradictions_summary[:2000],
                theories_summary=theories_summary[:2000],
                authorities_summary=authorities_summary[:1500],
                disputed_facts_summary=disputed_facts_summary[:1500],
            )

            content = self._call_with_escalation(prompt)
            if not content:
                update_ci_report(report_id, content='Report generation failed.', status='failed')
                return False

            update_ci_report(report_id, content=content, status='complete')
            logger.info(f"Report {report_id} generated successfully ({len(content)} chars)")
            return True

        except Exception as e:
            logger.error(f"ReportGenerator.generate failed: {e}", exc_info=True)
            update_ci_report(report_id, content=f'Error: {e}', status='failed')
            return False

    def generate_pdf(self, markdown_content: str, title: str = 'CI Report') -> Optional[bytes]:
        """
        Convert Markdown report to PDF using weasyprint.
        Returns PDF bytes or None if weasyprint not available.
        """
        try:
            from weasyprint import HTML, CSS
            import markdown as md_lib

            # Convert Markdown to HTML
            html_content = md_lib.markdown(
                markdown_content,
                extensions=['tables', 'fenced_code', 'toc'],
            )

            full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<style>
  body {{ font-family: Georgia, serif; font-size: 11pt; margin: 1.5in; color: #222; }}
  h1 {{ font-size: 18pt; border-bottom: 2px solid #2c3e50; padding-bottom: 6px; }}
  h2 {{ font-size: 14pt; color: #2c3e50; margin-top: 24px; }}
  h3 {{ font-size: 12pt; color: #34495e; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #bdc3c7; padding: 6px 10px; text-align: left; }}
  th {{ background: #ecf0f1; }}
  code {{ background: #f8f9fa; padding: 2px 4px; font-family: monospace; font-size: 9pt; }}
  blockquote {{ border-left: 4px solid #3498db; padding: 4px 12px; color: #555; margin: 12px 0; }}
  @page {{ margin: 1in; }}
</style>
</head><body>
{html_content}
</body></html>"""

            pdf_bytes = HTML(string=full_html).write_pdf()
            return pdf_bytes

        except ImportError:
            logger.warning("weasyprint not available — PDF generation disabled")
            return None
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None

    def _call_with_escalation(self, prompt: str) -> Optional[str]:
        for model_key in ('primary', 'escalate'):
            model = (self.task_def.primary_model if model_key == 'primary'
                     else self.task_def.escalate_model)
            provider = (self.task_def.primary_provider if model_key == 'primary'
                        else self.task_def.escalate_provider)

            client = self.llm_clients.get(provider)
            if not client:
                continue

            try:
                result = self._call_llm(client, model, prompt, provider)
                if result:
                    return result
                if model_key == 'primary':
                    logger.debug("ReportGenerator: escalating")
            except Exception as e:
                logger.error(f"ReportGenerator: {provider}/{model} failed: {e}")

        return None

    def _call_llm(self, client, model: str, prompt: str,
                  provider: str) -> Optional[str]:
        try:
            if provider == 'anthropic':
                response = client.client.messages.create(
                    model=model,
                    max_tokens=self.task_def.max_output_tokens,
                    messages=[{'role': 'user', 'content': prompt}],
                )
                text = response.content[0].text
                usage = response.usage
                if self.usage_tracker:
                    self.usage_tracker.log_usage(
                        provider=provider, model=model,
                        operation='ci:report_generation',
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )
                return text
            elif provider == 'openai':
                response = client.client.chat.completions.create(
                    model=model,
                    max_tokens=self.task_def.max_output_tokens,
                    messages=[{'role': 'user', 'content': prompt}],
                )
                text = response.choices[0].message.content
                usage = response.usage
                if self.usage_tracker:
                    self.usage_tracker.log_usage(
                        provider=provider, model=model,
                        operation='ci:report_generation',
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                    )
                return text
        except Exception as e:
            logger.error(f"ReportGenerator LLM call failed: {e}")
            return None

    # ---------- Summary helpers ----------

    def _summarize_entities(self, entities) -> str:
        if not entities:
            return 'No entities extracted.'
        lines = []
        for e in entities[:30]:
            lines.append(f"- [{e['entity_type'].upper()}] {e['name']}: {e['role_in_case'] or 'role unknown'}")
        return '\n'.join(lines)

    def _summarize_timeline(self, events) -> str:
        if not events:
            return 'No timeline events extracted.'
        lines = []
        for ev in events[:40]:
            date = ev['event_date'] or 'Date unknown'
            sig = ev['significance'] or 'medium'
            lines.append(f"- {date} [{sig.upper()}]: {ev['description']}")
        return '\n'.join(lines)

    def _summarize_contradictions(self, contradictions) -> str:
        if not contradictions:
            return 'No contradictions detected.'
        lines = []
        for c in contradictions[:20]:
            lines.append(f"- [{c['severity'].upper()}] {c['description']}")
        return '\n'.join(lines)

    def _summarize_theories(self, theories) -> str:
        if not theories:
            return 'No theories generated.'
        lines = []
        for t in theories[:10]:
            conf = int((t['confidence'] or 0.5) * 100)
            lines.append(f"- [{t['status'].upper()} {conf}%] {t['theory_text'][:200]}")
        return '\n'.join(lines)

    def _summarize_authorities(self, authorities) -> str:
        if not authorities:
            return 'No legal authorities cited.'
        lines = []
        for a in authorities[:20]:
            lines.append(f"- [{a['authority_type'].upper()}] {a['citation']}: {a['relevance_note'] or ''}")
        return '\n'.join(lines)

    def _summarize_disputed_facts(self, facts) -> str:
        if not facts:
            return 'No disputed facts identified.'
        lines = []
        for f in facts[:15]:
            lines.append(
                f"- {f['fact_description']}: "
                f"{f['position_a_label'] or 'A'} vs {f['position_b_label'] or 'B'}"
            )
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Standalone scientific-paper report generator (used by Director D2)
# ---------------------------------------------------------------------------

def _render_exec_summary(manager_reports: list) -> str:
    total = sum(len(r.get('findings', [])) for r in manager_reports)
    critical = sum(1 for r in manager_reports
                   for f in r.get('findings', []) if f.get('confidence') == 'high')
    domains = [r.get('domain', '?') for r in manager_reports if r.get('findings')]
    cost = sum(r.get('cost_usd', 0) for r in manager_reports)
    return '\n'.join([
        "This report presents the results of an automated Case Intelligence analysis.",
        "",
        f"- **Total findings:** {total} ({critical} high-confidence)",
        f"- **Domains analyzed:** {', '.join(domains)}",
        f"- **Analysis cost:** ${cost:.4f} USD",
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    ])


def _render_key_findings(manager_reports: list) -> str:
    high_conf = []
    for r in manager_reports:
        for f in r.get('findings', []):
            if f.get('confidence') in ('high', 'medium'):
                high_conf.append((f.get('confidence', 'medium'), r.get('domain', '?'), f))
    high_conf.sort(key=lambda x: (0 if x[0] == 'high' else 1, x[1]))
    if not high_conf:
        return "_No high-confidence findings identified._"
    lines = []
    for i, (conf, domain, f) in enumerate(high_conf[:20], 1):
        badge = "🔴 CRITICAL" if conf == 'high' else "🟡 NOTABLE"
        content = f.get('content', '').replace('\n', ' ').strip()
        source = f.get('source', '')
        lines.append(f"{i}. **[{badge}]** [{domain.upper()}] {content}")
        if source:
            lines.append(f"   > *Source: {source}*")
        lines.append("")
    return '\n'.join(lines)


def _render_cast_of_characters(manager_reports: list) -> str:
    er = next((r for r in manager_reports if r.get('domain') == 'entities'), None)
    if not er or not er.get('findings'):
        return "_Entity extraction not available._"
    lines = ["| Name / Entity | Role / Description | Source |", "|---|---|---|"]
    for f in er['findings'][:30]:
        content = f.get('content', '').replace('|', '\\|').replace('\n', ' ')
        source = f.get('source', '').replace('|', '\\|')
        lines.append(f"| — | {content} | {source} |")
    return '\n'.join(lines)


def _render_chronology(manager_reports: list) -> str:
    tr = next((r for r in manager_reports if r.get('domain') == 'timeline'), None)
    if not tr or not tr.get('findings'):
        return "_Timeline extraction not available._"
    lines = ["| Date / Period | Event | Source |", "|---|---|---|"]
    for f in sorted(tr['findings'][:40], key=lambda x: x.get('content', '')[:10]):
        content = f.get('content', '').replace('|', '\\|').replace('\n', ' ')
        source = f.get('source', '').replace('|', '\\|')
        lines.append(f"| — | {content} | {source} |")
    return '\n'.join(lines)


def _render_financial(manager_reports: list) -> str:
    fr = next((r for r in manager_reports if r.get('domain') == 'financial'), None)
    if not fr or not fr.get('findings'):
        return "_Financial analysis not available._"
    lines = []
    for f in fr['findings'][:25]:
        lines.append(f"- {f.get('content', '')}")
        if f.get('source'):
            lines.append(f"  > *{f['source']}*")
    return '\n'.join(lines) if lines else "_No financial findings._"


def _render_contested_facts(manager_reports: list) -> str:
    cr = next((r for r in manager_reports if r.get('domain') == 'contradictions'), None)
    if not cr or not cr.get('findings'):
        return "_No contradictions identified._"
    lines = ["| Contested Issue | Position A | Position B | Source |", "|---|---|---|---|"]
    for f in cr['findings'][:20]:
        content = f.get('content', '').replace('|', '\\|').replace('\n', ' ')
        source = f.get('source', '').replace('|', '\\|')
        lines.append(f"| {content} | — | — | {source} |")
    return '\n'.join(lines)


def _render_theories(manager_reports: list) -> str:
    tr = next((r for r in manager_reports if r.get('domain') == 'theories'), None)
    if not tr or not tr.get('findings'):
        return "_No theories identified._"
    lines = []
    for i, f in enumerate(tr['findings'][:10], 1):
        lines.append(f"### Theory {i}")
        lines.append(f.get('content', ''))
        if f.get('source'):
            lines.append(f"> *Source: {f['source']}*")
        lines.append("")
    return '\n'.join(lines)


def _render_authorities(manager_reports: list) -> str:
    ar = next((r for r in manager_reports if r.get('domain') == 'authorities'), None)
    if not ar or not ar.get('findings'):
        return "_No legal authorities identified._"
    binding = [f for f in ar['findings'] if 'binding' in str(f.get('type', '')).lower()]
    persuasive = [f for f in ar['findings']
                  if 'persuasive' in str(f.get('type', '')).lower() or f not in binding]
    lines = [f"**Binding authorities ({len(binding)}):**\n"]
    for f in binding[:15]:
        lines.append(f"- {f.get('content', '')} — *{f.get('source', '')}*")
    lines.append(f"\n**Persuasive authorities ({len(persuasive)}):**\n")
    for f in persuasive[:15]:
        lines.append(f"- {f.get('content', '')} — *{f.get('source', '')}*")
    return '\n'.join(lines)


def _render_discovery_gaps(manager_reports: list) -> str:
    gaps = []
    for domain in ('entities', 'timeline', 'financial', 'contradictions', 'theories', 'authorities'):
        r = next((x for x in manager_reports if x.get('domain') == domain), None)
        if not r or not r.get('findings'):
            gaps.append(f"- **Incomplete domain:** {domain}")
    all_findings = [f for r in manager_reports for f in r.get('findings', [])]
    low_conf = [f for f in all_findings if f.get('confidence') == 'low']
    if low_conf:
        gaps.append(f"- **{len(low_conf)} low-confidence findings** require human review")
    gaps += [
        "- Review source documents for completeness",
        "- Verify entity identifications against external sources",
        "- Cross-reference financial figures with original statements",
    ]
    return '\n'.join(gaps)


def _render_doc_index(doc_ids: list) -> str:
    if not doc_ids:
        return "_No documents indexed._"
    lines = ["| Doc ID | Bates Ref |", "|---|---|"]
    for i, doc_id in enumerate(doc_ids, 1):
        lines.append(f"| {doc_id} | BATES-{i:05d} |")
    return '\n'.join(lines)


def _render_contradictions_catalog(manager_reports: list) -> str:
    cr = next((r for r in manager_reports if r.get('domain') == 'contradictions'), None)
    if not cr or not cr.get('findings'):
        return "_No contradictions catalogued._"
    lines = []
    for i, f in enumerate(cr['findings'], 1):
        lines.append(f"**C-{i:03d}:** {f.get('content', '')}  ")
        lines.append(f"*Source: {f.get('source', 'unknown')}*\n")
    return '\n'.join(lines)


def _render_authority_citations(manager_reports: list) -> str:
    ar = next((r for r in manager_reports if r.get('domain') == 'authorities'), None)
    if not ar:
        return "_No authorities cited._"
    lines = []
    for i, f in enumerate(ar.get('findings', []), 1):
        lines.append(f"{i}. {f.get('content', '')}  ")
        lines.append(f"   *{f.get('source', '')}*\n")
    return '\n'.join(lines)


def generate_report(manager_reports: list,
                    case_name: str = 'Case',
                    doc_ids: Optional[list] = None,
                    run_id: str = '') -> str:
    """
    Generate a scientific-paper CI report from hierarchical manager findings.

    Args:
        manager_reports: list of manager report dicts (domain, findings, cost_usd)
        case_name: display name for the case
        doc_ids: list of document IDs in scope
        run_id: CI run ID for reference

    Returns:
        Markdown string of the full report.
    """
    doc_ids = doc_ids or []
    total = sum(len(r.get('findings', [])) for r in manager_reports)
    critical = sum(1 for r in manager_reports
                   for f in r.get('findings', []) if f.get('confidence') == 'high')

    sections = [
        "# CASE INTELLIGENCE ANALYSIS REPORT",
        f"**Case:** {case_name}  ",
        f"**Run ID:** {run_id}  ",
        f"**Documents:** {len(doc_ids)}  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## I. Executive Summary",
        _render_exec_summary(manager_reports),
        "",
        "---",
        "",
        f"## II. Key Findings ({total} findings, {critical} critical)",
        _render_key_findings(manager_reports),
        "",
        "---",
        "",
        "## III. Cast of Characters",
        _render_cast_of_characters(manager_reports),
        "",
        "---",
        "",
        "## IV. Chronology",
        _render_chronology(manager_reports),
        "",
        "---",
        "",
        "## V. Financial Analysis",
        _render_financial(manager_reports),
        "",
        "---",
        "",
        "## VI. Contested Facts Matrix",
        _render_contested_facts(manager_reports),
        "",
        "---",
        "",
        "## VII. Factual & Legal Theories",
        _render_theories(manager_reports),
        "",
        "---",
        "",
        "## VIII. Legal Framework & Authorities",
        _render_authorities(manager_reports),
        "",
        "---",
        "",
        "## IX. Discovery Gaps & Next Steps",
        _render_discovery_gaps(manager_reports),
        "",
        "---",
        "",
        "## Appendix A: Document Index & Bates References",
        _render_doc_index(doc_ids),
        "",
        "## Appendix B: Contradictions Catalog",
        _render_contradictions_catalog(manager_reports),
        "",
        "## Appendix C: Full Authority Citations",
        _render_authority_citations(manager_reports),
        "",
    ]
    return '\n'.join(sections)
