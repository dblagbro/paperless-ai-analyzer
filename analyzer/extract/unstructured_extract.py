"""
Unstructured-based PDF Extraction

Extracts tables and structured data from PDFs using the Unstructured library.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class UnstructuredExtractor:
    """Extracts structured data from PDFs using Unstructured."""

    def __init__(self):
        """Initialize the extractor."""
        self.partition_pdf = None
        self.use_fallback = False
        self.use_simple_extractor = False

        # Try to import partition_pdf (may fail due to onnxruntime issues)
        try:
            # Test if the import actually works by doing a minimal check
            from unstructured.partition.pdf import partition_pdf as _test_pdf
            # If we get here, import succeeded
            self.partition_pdf = _test_pdf
            logger.info("Unstructured partition_pdf loaded successfully")
        except (ImportError, OSError, Exception) as e:
            logger.warning(f"Cannot import partition_pdf (likely onnxruntime issue): {e}")
            # Use simple PyPDF2/pdfplumber-based extraction as fallback
            try:
                import PyPDF2
                from pdf2image import convert_from_path
                self.use_simple_extractor = True
                logger.info("Using simple PyPDF2-based extraction as fallback")
            except ImportError as e2:
                logger.error(f"Cannot import PyPDF2 fallback: {e2}")
                self.partition_pdf = None

    def extract(self,
                pdf_path: str,
                profile: Any) -> Dict[str, Any]:
        """
        Extract structured data from PDF.

        Args:
            pdf_path: Path to PDF file
            profile: Document type profile with extraction config

        Returns:
            Dict with extracted transactions and metadata
        """
        # Use simple extraction if unstructured not available
        if self.use_simple_extractor:
            return self._simple_extract(pdf_path, profile)

        if not self.partition_pdf:
            logger.error("Cannot extract: no extraction method available")
            return {'transactions': [], 'page_totals': {}, 'error': 'no_extractor_available'}

        try:
            extraction_config = profile.extraction if profile else {}
            mode = extraction_config.get('mode', 'fast')  # Default to 'fast' to avoid onnxruntime

            logger.info(f"Extracting from {pdf_path} using mode={mode}, fallback={self.use_fallback}")

            # Partition PDF
            if self.use_fallback:
                # Using auto partition - simpler signature
                elements = self.partition_pdf(
                    filename=pdf_path,
                    strategy='fast',  # Force fast mode for fallback
                    include_page_breaks=True
                )
            else:
                # Using standard partition_pdf
                # Force 'fast' mode if hi_res or elements is requested (avoids onnxruntime)
                if mode in ('hi_res', 'elements'):
                    logger.warning(f"Forcing mode='fast' to avoid onnxruntime issues")
                    mode = 'fast'

                elements = self.partition_pdf(
                    filename=pdf_path,
                    strategy=mode,
                    infer_table_structure=True,
                    include_page_breaks=True
                )

            # Extract transactions from tables
            transactions = self._extract_transactions(elements, extraction_config)

            # Extract page totals if present
            page_totals = self._extract_page_totals(elements)

            return {
                'transactions': transactions,
                'page_totals': page_totals,
                'total_transactions': len(transactions),
                'pages_processed': self._count_pages(elements)
            }

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            return {
                'transactions': [],
                'page_totals': {},
                'error': str(e)
            }

    def _extract_transactions(self,
                             elements: List,
                             config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract transactions from Unstructured elements.

        Args:
            elements: Unstructured document elements
            config: Extraction configuration

        Returns:
            List of transaction dicts
        """
        transactions = []
        table_hints = config.get('table_hints', {})
        column_keywords = [kw.lower() for kw in table_hints.get('column_keywords', [])]
        date_regex = table_hints.get('date_regex', r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')

        current_page = 1

        for element in elements:
            # Track page numbers
            if str(type(element).__name__) == 'PageBreak':
                current_page += 1
                continue

            # Look for tables
            if str(type(element).__name__) == 'Table':
                try:
                    # Get table data
                    table_data = element.metadata.text_as_html if hasattr(element.metadata, 'text_as_html') else None
                    if not table_data:
                        # Fallback to plain text parsing
                        table_data = str(element)

                    # Parse table rows
                    rows = self._parse_table_text(table_data, date_regex, column_keywords)

                    for row in rows:
                        row['page_num'] = current_page
                        transactions.append(row)

                except Exception as e:
                    logger.debug(f"Failed to parse table on page {current_page}: {e}")
                    continue

        logger.info(f"Extracted {len(transactions)} transactions from {current_page} pages")
        return transactions

    def _parse_table_text(self,
                         table_text: str,
                         date_regex: str,
                         column_keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Parse table text to extract transaction rows.

        Args:
            table_text: Table as text (HTML or plain)
            date_regex: Regex for date detection
            column_keywords: Keywords to identify columns

        Returns:
            List of transaction dicts
        """
        rows = []

        # Split into lines
        lines = table_text.split('\n')

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check if this line contains a date (likely a transaction row)
            if not re.search(date_regex, line):
                continue

            # Extract date
            date_match = re.search(date_regex, line)
            date_str = date_match.group(0) if date_match else ''

            # Remove date from line to parse remaining fields
            remaining = line.replace(date_str, '', 1).strip()

            # Try to extract numeric values (debits, credits, balances)
            numbers = re.findall(r'\$?[\d,]+\.\d{2}', remaining)
            numbers = [self._clean_number(n) for n in numbers]

            # Extract description (text between date and numbers)
            desc_match = re.search(r'^(.*?)(?=\$?[\d,]+\.\d{2}|$)', remaining)
            description = desc_match.group(1).strip() if desc_match else remaining

            # Heuristic: determine which numbers are debit, credit, balance
            # This is simplified - real-world needs column detection
            debit = None
            credit = None
            balance = None

            if len(numbers) == 1:
                balance = numbers[0]
            elif len(numbers) == 2:
                # Could be (amount, balance) or (debit, credit)
                # Assume last is balance
                balance = numbers[-1]
                credit = numbers[0]
            elif len(numbers) >= 3:
                # Likely (debit, credit, balance) or similar
                debit = numbers[0] if numbers[0] else None
                credit = numbers[1] if numbers[1] else None
                balance = numbers[-1]

            rows.append({
                'date': date_str,
                'description': description[:100],  # Limit length
                'debit': debit,
                'credit': credit,
                'balance': balance,
                'row_index': line_idx,
                'confidence': 0.7  # Placeholder confidence
            })

        return rows

    def _extract_page_totals(self, elements: List) -> Dict[int, Dict[str, float]]:
        """
        Extract page-level totals if present.

        Args:
            elements: Unstructured elements

        Returns:
            Dict mapping page number to totals
        """
        page_totals = {}
        current_page = 1

        for element in elements:
            if str(type(element).__name__) == 'PageBreak':
                current_page += 1
                continue

            # Look for "total" or "subtotal" patterns
            text = str(element).lower()
            if 'total' in text or 'subtotal' in text:
                # Extract numbers from this element
                numbers = re.findall(r'\$?[\d,]+\.\d{2}', str(element))
                if numbers:
                    # Assume last number is the total
                    total_value = self._clean_number(numbers[-1])
                    if current_page not in page_totals:
                        page_totals[current_page] = {}

                    if 'debit' in text or 'withdrawal' in text:
                        page_totals[current_page]['debit'] = total_value
                    elif 'credit' in text or 'deposit' in text:
                        page_totals[current_page]['credit'] = total_value

        return page_totals

    def _count_pages(self, elements: List) -> int:
        """Count pages in elements."""
        return sum(1 for e in elements if str(type(e).__name__) == 'PageBreak') + 1

    def _clean_number(self, num_str: str) -> Optional[float]:
        """Clean and convert number string to float."""
        try:
            cleaned = num_str.replace('$', '').replace(',', '').strip()
            if cleaned and cleaned != '-':
                return float(cleaned)
        except ValueError:
            pass
        return None

    def _simple_extract(self, pdf_path: str, profile: Any) -> Dict[str, Any]:
        """
        Simple extraction using PyPDF2 (fallback when unstructured fails).

        Args:
            pdf_path: Path to PDF file
            profile: Document type profile

        Returns:
            Dict with extracted transactions
        """
        import PyPDF2

        logger.info(f"Using simple extraction for {pdf_path}")

        try:
            extraction_config = profile.extraction if profile else {}
            table_hints = extraction_config.get('table_hints', {})
            date_regex = table_hints.get('date_regex', r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')

            transactions = []
            page_totals = {}

            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)

                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    text = page.extract_text()

                    # Split into lines and parse
                    lines = text.split('\n')
                    for line_idx, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue

                        # Check if this line contains a date (likely a transaction row)
                        if not re.search(date_regex, line):
                            continue

                        # Extract date
                        date_match = re.search(date_regex, line)
                        date_str = date_match.group(0) if date_match else ''

                        # Remove date from line to parse remaining fields
                        remaining = line.replace(date_str, '', 1).strip()

                        # Try to extract numeric values
                        numbers = re.findall(r'\$?[\d,]+\.\d{2}', remaining)
                        numbers = [self._clean_number(n) for n in numbers]

                        # Extract description
                        desc_match = re.search(r'^(.*?)(?=\$?[\d,]+\.\d{2}|$)', remaining)
                        description = desc_match.group(1).strip() if desc_match else remaining

                        # Heuristic: determine which numbers are debit, credit, balance
                        debit = None
                        credit = None
                        balance = None

                        if len(numbers) == 1:
                            balance = numbers[0]
                        elif len(numbers) == 2:
                            balance = numbers[-1]
                            credit = numbers[0]
                        elif len(numbers) >= 3:
                            debit = numbers[0] if numbers[0] else None
                            credit = numbers[1] if numbers[1] else None
                            balance = numbers[-1]

                        if balance is not None or credit is not None or debit is not None:
                            transactions.append({
                                'date': date_str,
                                'description': description[:100],
                                'debit': debit,
                                'credit': credit,
                                'balance': balance,
                                'page_num': page_num + 1,
                                'row_index': line_idx,
                                'confidence': 0.6  # Lower confidence for simple extraction
                            })

                logger.info(f"Simple extraction found {len(transactions)} transactions from {num_pages} pages")

                return {
                    'transactions': transactions,
                    'page_totals': page_totals,
                    'total_transactions': len(transactions),
                    'pages_processed': num_pages
                }

        except Exception as e:
            logger.error(f"Simple extraction failed: {e}", exc_info=True)
            return {
                'transactions': [],
                'page_totals': {},
                'error': str(e)
            }
