#!/usr/bin/env python3
"""
Fix all API fetch calls to work with nginx subpath proxying.
This adds BASE_PATH detection and wraps all /api/ calls with apiUrl().
"""

import re

def fix_dashboard_html(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the first <script> tag
    script_start = content.find('    <script>')
    if script_start == -1:
        print("ERROR: Could not find <script> tag")
        return False

    # Insert BASE_PATH and apiUrl helper right after <script>
    base_path_code = '''    <script>
        // Detect base path for API calls (handles nginx subpath proxying)
        const BASE_PATH = (() => {
            const path = window.location.pathname;
            // If we're under /paperless-ai-analyzer/, use that as base
            if (path.startsWith('/paperless-ai-analyzer')) {
                return '/paperless-ai-analyzer';
            }
            // Otherwise, no base path (direct access)
            return '';
        })();

        // Helper function to build API URLs
        function apiUrl(path) {
            return BASE_PATH + path;
        }

'''

    # Replace the <script> tag with the new version
    content = content.replace('    <script>', base_path_code, 1)

    # Fix all fetch calls with single quotes:
    # fetch('api/...') or fetch('/api/...') -> fetch(apiUrl('/api/...'))
    content = re.sub(
        r"fetch\('/?api/([^']+)'\)",
        r"fetch(apiUrl('/api/\1'))",
        content
    )

    # Fix all fetch calls with template literals:
    # fetch(`api/${var}`) or fetch(`/api/${var}`) -> fetch(apiUrl(`/api/${var}`))
    content = re.sub(
        r"fetch\(`/?api/([^`]+)`\)",
        r"fetch(apiUrl(`/api/\1`))",
        content
    )

    # Write back
    with open(filepath, 'w') as f:
        f.write(content)

    print("✓ Fixed all API fetch calls in dashboard.html")
    print(f"  - Added BASE_PATH detection")
    print(f"  - Added apiUrl() helper function")

    # Count how many apiUrl calls we have now
    apiurl_count = content.count('apiUrl(')
    print(f"  - Updated {apiurl_count - 1} fetch calls (excluding helper definition)")

    return True

if __name__ == '__main__':
    filepath = '/home/dblagbro/docker/paperless-ai-analyzer/analyzer/templates/dashboard.html'
    success = fix_dashboard_html(filepath)
    exit(0 if success else 1)
