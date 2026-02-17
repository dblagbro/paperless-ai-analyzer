"""
Smart Uploader

AI-powered document upload with automatic metadata extraction.
Analyzes documents and suggests project assignment, title, tags, etc.
"""

import logging
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class SmartUploader:
    """AI-powered document upload with metadata extraction."""

    def __init__(self, llm_client, paperless_client, project_manager):
        """
        Initialize smart uploader.

        Args:
            llm_client: LLM client for metadata extraction
            paperless_client: Paperless API client
            project_manager: Project manager instance
        """
        self.llm_client = llm_client
        self.paperless_client = paperless_client
        self.project_manager = project_manager

    async def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze document and extract metadata using AI.

        Args:
            file_path: Path to uploaded file

        Returns:
            Dict with extracted metadata and project suggestions
        """
        try:
            # Read document content (first 2000 chars for analysis)
            content_preview = self._read_document(file_path)

            # Extract metadata using LLM
            extraction_prompt = self._build_extraction_prompt(content_preview)
            response = self.llm_client._call_llm(extraction_prompt)

            # Parse response
            metadata = self.llm_client._safe_json_parse(response)

            if not metadata:
                logger.warning("Failed to extract metadata, using defaults")
                metadata = self._default_metadata(file_path)

            # Add file info
            metadata['file_id'] = str(uuid.uuid4())
            metadata['file_name'] = os.path.basename(file_path)
            metadata['file_size'] = os.path.getsize(file_path)

            # Suggest projects
            existing_projects = self.project_manager.list_projects()
            project_suggestions = self.suggest_project(metadata, existing_projects)
            metadata['project_suggestions'] = project_suggestions

            logger.info(f"Analyzed document: {metadata.get('suggested_title', 'Unknown')}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to analyze document: {e}")
            return self._default_metadata(file_path)

    def suggest_project(self, metadata: Dict, existing_projects: List[Dict]) -> List[Dict]:
        """
        Suggest projects based on extracted metadata.

        Args:
            metadata: Extracted document metadata
            existing_projects: List of existing projects

        Returns:
            List of project suggestions with confidence scores
        """
        suggestions = []

        case_number = metadata.get('case_number')
        parties = metadata.get('parties', [])
        suggested_slug = metadata.get('suggested_project_slug')
        suggested_name = metadata.get('suggested_project_name')

        # Check for matching existing projects
        for project in existing_projects:
            confidence = 0.0
            reasons = []

            # Match by case number in metadata
            if case_number and project.get('metadata', {}).get('case_number') == case_number:
                confidence += 0.5
                reasons.append(f"Case number matches: {case_number}")

            # Match by slug similarity
            if suggested_slug and suggested_slug in project['slug']:
                confidence += 0.3
                reasons.append(f"Slug similarity: {suggested_slug}")

            # Match by name similarity
            if suggested_name:
                name_similarity = self._calculate_similarity(suggested_name.lower(), project['name'].lower())
                if name_similarity > 0.5:
                    confidence += 0.2 * name_similarity
                    reasons.append(f"Name similarity: {int(name_similarity * 100)}%")

            # Match by parties mentioned
            if parties:
                project_name_lower = project['name'].lower()
                for party in parties:
                    if party.lower() in project_name_lower:
                        confidence += 0.1
                        reasons.append(f"Party mentioned: {party}")
                        break

            if confidence > 0.3:  # Only suggest if reasonable confidence
                suggestions.append({
                    'slug': project['slug'],
                    'name': project['name'],
                    'confidence': min(confidence, 1.0),
                    'reasons': reasons,
                    'existing': True
                })

        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)

        # Add option to create new project
        if suggested_slug and suggested_name:
            suggestions.append({
                'slug': suggested_slug,
                'name': suggested_name,
                'confidence': metadata.get('confidence', 0.8),
                'reasons': [metadata.get('reasoning', 'Based on document analysis')],
                'existing': False,
                'create_new': True
            })

        return suggestions[:5]  # Return top 5 suggestions

    async def upload_to_paperless(self, file_path: str, project_slug: str,
                                  metadata: Dict) -> Optional[Dict]:
        """
        Upload document to Paperless with metadata.

        Args:
            file_path: Path to file
            project_slug: Target project
            metadata: Extracted metadata

        Returns:
            Paperless document dict or None if failed
        """
        try:
            # Ensure project exists
            project = self.project_manager.get_project(project_slug)
            if not project:
                logger.error(f"Project '{project_slug}' not found")
                return None

            # Get/create project tag
            tag_ids = []
            project_tag_id = self.paperless_client.get_or_create_tag(
                f"project:{project_slug}",
                color=project.get('color', '#3498db')
            )
            if project_tag_id:
                tag_ids.append(project_tag_id)

            # Add additional tags
            additional_tags = metadata.get('suggested_tags', [])
            for tag_name in additional_tags:
                tag_id = self.paperless_client.get_or_create_tag(tag_name)
                if tag_id:
                    tag_ids.append(tag_id)

            # Upload document
            result = self.paperless_client.upload_document(
                file_path=file_path,
                title=metadata.get('suggested_title'),
                tags=tag_ids,
                created=metadata.get('extracted_date')
            )

            if result:
                logger.info(f"Uploaded document to project '{project_slug}': {metadata.get('suggested_title')}")

                # Update project document count
                self.project_manager.increment_document_count(project_slug, delta=1)

            return result

        except Exception as e:
            logger.error(f"Failed to upload to Paperless: {e}")
            return None

    def _read_document(self, file_path: str) -> str:
        """
        Read document content for analysis.

        Args:
            file_path: Path to file

        Returns:
            Document content (first 2000 chars)
        """
        try:
            # For PDFs, we'd need to extract text (using PyPDF2 or similar)
            # For now, just return file info
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            # TODO: Add PDF text extraction
            # For MVP, use filename and basic info
            content = f"Filename: {file_name}\nSize: {file_size} bytes\n"

            return content[:2000]

        except Exception as e:
            logger.error(f"Failed to read document: {e}")
            return ""

    def _build_extraction_prompt(self, content_preview: str) -> str:
        """
        Build LLM prompt for metadata extraction.

        Args:
            content_preview: Document content preview

        Returns:
            Prompt string
        """
        return f"""Analyze this document and extract key metadata.

Document content (first 2000 chars):
{content_preview}

Extract the following information:
1. **Document Type**: motion, order, letter, invoice, bank statement, etc.
2. **Case/Matter Number**: Court case number, docket number, or matter ID
3. **Parties Involved**: Names of people, companies, or entities
4. **Key Dates**: Filing dates, event dates, deadlines
5. **Court/Institution**: Name of court, bank, or institution
6. **Document Title**: Suggested title (max 100 chars)

Based on the extracted information, suggest a project identifier:
- Format: lowercase-with-dashes
- Include case number or key identifier
- Max 50 characters
- Example: "case-2024-123-fairbridge" or "acme-corp-2024"

Return JSON:
{{
  "document_type": "motion|order|correspondence|invoice|statement|other",
  "case_number": "extracted case/matter number or null",
  "parties": ["party 1", "party 2", ...],
  "dates": ["2024-01-15", ...],
  "court_or_institution": "name or null",
  "suggested_title": "concise title",
  "suggested_project_slug": "lowercase-slug",
  "suggested_project_name": "Human Readable Name",
  "suggested_tags": ["tag1", "tag2"],
  "extracted_date": "2024-01-15 or null",
  "confidence": 0.0-1.0,
  "reasoning": "why this project suggestion"
}}

IMPORTANT: Return ONLY valid JSON, no additional text."""

    def _default_metadata(self, file_path: str) -> Dict:
        """
        Generate default metadata when extraction fails.

        Args:
            file_path: Path to file

        Returns:
            Default metadata dict
        """
        file_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(file_name)[0]

        return {
            'document_type': 'other',
            'case_number': None,
            'parties': [],
            'dates': [],
            'court_or_institution': None,
            'suggested_title': name_without_ext,
            'suggested_project_slug': self._slugify(name_without_ext),
            'suggested_project_name': name_without_ext,
            'suggested_tags': [],
            'extracted_date': None,
            'confidence': 0.5,
            'reasoning': 'Based on filename (AI extraction failed)',
            'project_suggestions': []
        }

    def _slugify(self, text: str) -> str:
        """
        Convert text to URL-safe slug.

        Args:
            text: Input text

        Returns:
            Slugified string
        """
        # Lowercase
        slug = text.lower()

        # Remove special chars
        slug = re.sub(r'[^\w\s-]', '', slug)

        # Replace spaces/underscores with dashes
        slug = re.sub(r'[-\s]+', '-', slug)

        # Trim dashes
        slug = slug.strip('-')

        # Limit length
        return slug[:50]

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate simple similarity score between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple word overlap similarity
        words1 = set(str1.split())
        words2 = set(str2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0
