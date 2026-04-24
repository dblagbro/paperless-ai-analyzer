"""Re-export shim — actual implementation moved to `web_researchers/`
package during the v3.9.8 refactor. Keeps existing imports working:

    from analyzer.case_intelligence.web_researcher import WebResearcher
"""
from analyzer.case_intelligence.web_researchers import WebResearcher  # noqa: F401
