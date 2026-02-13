"""
Deterministic Anomaly Checks

Implements mathematical and rule-based anomaly detection.
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


class DeterministicChecker:
    """Performs deterministic anomaly checks on extracted data."""

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize checker.

        Args:
            tolerance: Balance mismatch tolerance (e.g., 0.01 for $0.01)
        """
        self.tolerance = Decimal(str(tolerance))

    def check_all(self,
                  extracted_data: Dict[str, Any],
                  profile: Any) -> Dict[str, Any]:
        """
        Run all enabled checks.

        Args:
            extracted_data: Extracted transaction data
            profile: Document type profile

        Returns:
            Dict with anomaly findings
        """
        results = {
            'anomalies_found': [],
            'checks_passed': [],
            'warnings': [],
            'evidence': {}
        }

        transactions = extracted_data.get('transactions', [])
        if not transactions:
            results['warnings'].append("No transactions extracted")
            return results

        checks_enabled = profile.checks_enabled if profile else []

        # Running balance check
        if 'running_balance' in checks_enabled:
            balance_result = self.check_running_balance(
                transactions,
                profile.validation.get('running_balance_tolerance', 0.01)
            )
            if balance_result['mismatches']:
                results['anomalies_found'].append('balance_mismatch')
                results['evidence']['balance_mismatch'] = balance_result
            else:
                results['checks_passed'].append('running_balance')

        # Page totals check
        if 'page_totals' in checks_enabled:
            totals_result = self.check_page_totals(extracted_data)
            if totals_result['mismatches']:
                results['anomalies_found'].append('page_total_mismatch')
                results['evidence']['page_total_mismatch'] = totals_result
            else:
                results['checks_passed'].append('page_totals')

        # Continuity check (between pages)
        if 'continuity' in checks_enabled:
            continuity_result = self.check_continuity(extracted_data)
            if continuity_result['gaps_found']:
                results['anomalies_found'].append('continuity_mismatch')
                results['evidence']['continuity_mismatch'] = continuity_result
            else:
                results['checks_passed'].append('continuity')

        # Duplicate detection
        if 'duplicates' in checks_enabled:
            dup_result = self.check_duplicates(transactions)
            if dup_result['duplicates']:
                results['anomalies_found'].append('duplicate_transactions')
                results['evidence']['duplicate_transactions'] = dup_result
            else:
                results['checks_passed'].append('duplicates')

        # Date order check
        if 'date_order' in checks_enabled:
            date_result = self.check_date_order(transactions)
            if date_result['violations']:
                results['anomalies_found'].append('date_order_violation')
                results['evidence']['date_order_violation'] = date_result
            else:
                results['checks_passed'].append('date_order')

        return results

    def check_running_balance(self,
                             transactions: List[Dict],
                             tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Verify running balance arithmetic.

        Args:
            transactions: List of transaction dicts with date, debit, credit, balance
            tolerance: Acceptable difference

        Returns:
            Dict with mismatch details
        """
        mismatches = []
        tolerance_decimal = Decimal(str(tolerance))

        for i, txn in enumerate(transactions):
            if i == 0:
                continue  # Skip first transaction (no prior balance)

            try:
                # Get prior balance
                prior_balance = self._to_decimal(transactions[i-1].get('balance'))
                if prior_balance is None:
                    continue

                # Get current transaction amounts
                debit = self._to_decimal(txn.get('debit', 0))
                credit = self._to_decimal(txn.get('credit', 0))
                actual_balance = self._to_decimal(txn.get('balance'))

                if actual_balance is None:
                    continue

                # Calculate expected balance
                expected_balance = prior_balance + credit - debit

                # Check for mismatch
                diff = abs(actual_balance - expected_balance)
                if diff > tolerance_decimal:
                    mismatches.append({
                        'row_index': i,
                        'date': txn.get('date'),
                        'description': txn.get('description', '')[:50],
                        'prior_balance': float(prior_balance),
                        'debit': float(debit),
                        'credit': float(credit),
                        'expected_balance': float(expected_balance),
                        'actual_balance': float(actual_balance),
                        'difference': float(diff)
                    })

            except (InvalidOperation, TypeError, ValueError) as e:
                logger.debug(f"Skipping transaction {i} due to parse error: {e}")
                continue

        return {
            'mismatches': mismatches,
            'total_transactions': len(transactions),
            'mismatch_count': len(mismatches)
        }

    def check_page_totals(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify page-level totals match transaction sums.

        Args:
            extracted_data: Full extracted data with page_totals

        Returns:
            Dict with total mismatch details
        """
        mismatches = []
        page_totals = extracted_data.get('page_totals', {})
        transactions = extracted_data.get('transactions', [])

        if not page_totals:
            return {'mismatches': [], 'message': 'No page totals found'}

        # Group transactions by page
        txns_by_page = {}
        for txn in transactions:
            page = txn.get('page_num', 1)
            if page not in txns_by_page:
                txns_by_page[page] = []
            txns_by_page[page].append(txn)

        # Check each page's totals
        for page_num, stated_totals in page_totals.items():
            page_txns = txns_by_page.get(page_num, [])

            computed_debit = sum(self._to_decimal(t.get('debit', 0)) or Decimal('0')
                                for t in page_txns)
            computed_credit = sum(self._to_decimal(t.get('credit', 0)) or Decimal('0')
                                 for t in page_txns)

            stated_debit = self._to_decimal(stated_totals.get('debit'))
            stated_credit = self._to_decimal(stated_totals.get('credit'))

            if stated_debit and abs(computed_debit - stated_debit) > self.tolerance:
                mismatches.append({
                    'page': page_num,
                    'type': 'debit',
                    'computed': float(computed_debit),
                    'stated': float(stated_debit),
                    'difference': float(abs(computed_debit - stated_debit))
                })

            if stated_credit and abs(computed_credit - stated_credit) > self.tolerance:
                mismatches.append({
                    'page': page_num,
                    'type': 'credit',
                    'computed': float(computed_credit),
                    'stated': float(stated_credit),
                    'difference': float(abs(computed_credit - stated_credit))
                })

        return {
            'mismatches': mismatches,
            'pages_checked': len(page_totals),
            'mismatch_count': len(mismatches)
        }

    def check_continuity(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for balance continuity between pages.

        Args:
            extracted_data: Full extracted data

        Returns:
            Dict with continuity gaps
        """
        gaps = []
        transactions = extracted_data.get('transactions', [])

        # Group by page
        pages = {}
        for txn in transactions:
            page_num = txn.get('page_num', 1)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(txn)

        # Sort pages
        sorted_pages = sorted(pages.keys())

        for i in range(len(sorted_pages) - 1):
            current_page = sorted_pages[i]
            next_page = sorted_pages[i + 1]

            # Get last balance of current page
            last_txn = pages[current_page][-1]
            last_balance = self._to_decimal(last_txn.get('balance'))

            # Get first balance of next page
            first_txn = pages[next_page][0]
            first_balance = self._to_decimal(first_txn.get('balance'))

            if last_balance is None or first_balance is None:
                continue

            # Check if they match (allowing for one transaction in between)
            # In practice, pages often overlap or have carry-forward differences
            diff = abs(last_balance - first_balance)

            # Allow for reasonable carry-forward (adjust threshold as needed)
            if diff > Decimal('0.10'):  # More than 10 cents difference
                gaps.append({
                    'between_pages': f"{current_page}-{next_page}",
                    'page_end_balance': float(last_balance),
                    'next_page_start_balance': float(first_balance),
                    'difference': float(diff)
                })

        return {
            'gaps_found': gaps,
            'pages_checked': len(sorted_pages) - 1 if len(sorted_pages) > 1 else 0
        }

    def check_duplicates(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Detect duplicate transactions.

        Args:
            transactions: List of transactions

        Returns:
            Dict with duplicate groups
        """
        duplicates = []
        seen = {}

        for i, txn in enumerate(transactions):
            # Create signature from key fields
            date = txn.get('date', '')
            desc = txn.get('description', '')[:50]  # First 50 chars
            debit = self._to_decimal(txn.get('debit', 0))
            credit = self._to_decimal(txn.get('credit', 0))

            signature = (date, desc, debit, credit)

            if signature in seen:
                duplicates.append({
                    'rows': [seen[signature], i],
                    'date': date,
                    'description': desc,
                    'amount': float(debit or credit or 0)
                })
            else:
                seen[signature] = i

        return {
            'duplicates': duplicates,
            'duplicate_count': len(duplicates)
        }

    def check_date_order(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Check if transactions are in chronological order.

        Args:
            transactions: List of transactions

        Returns:
            Dict with date order violations
        """
        violations = []
        prev_date = None

        for i, txn in enumerate(transactions):
            date_str = txn.get('date')
            if not date_str:
                continue

            try:
                # Try to parse date (supports multiple formats)
                current_date = self._parse_date(date_str)
                if current_date is None:
                    continue

                if prev_date and current_date < prev_date:
                    violations.append({
                        'row_index': i,
                        'current_date': date_str,
                        'previous_date': prev_date.strftime('%Y-%m-%d'),
                        'description': txn.get('description', '')[:50]
                    })

                prev_date = current_date

            except Exception as e:
                logger.debug(f"Date parse error at row {i}: {e}")
                continue

        return {
            'violations': violations,
            'violation_count': len(violations)
        }

    def _to_decimal(self, value: Any) -> Decimal:
        """Convert value to Decimal, handling various formats."""
        if value is None or value == '':
            return None

        if isinstance(value, Decimal):
            return value

        try:
            # Remove currency symbols and commas
            if isinstance(value, str):
                value = value.replace('$', '').replace(',', '').strip()
                if value == '' or value == '-':
                    return None
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string in various formats."""
        formats = ['%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d', '%m-%d-%Y', '%d/%m/%Y']

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None
