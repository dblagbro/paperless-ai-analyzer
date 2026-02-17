"""
LLM Usage Tracker

Tracks token usage and costs for LLM API calls.
"""

import logging
import sqlite3
import threading
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LLMUsageTracker:
    """Track LLM token usage and costs."""

    def __init__(self, db_path: str = '/app/data/llm_usage.db'):
        """
        Initialize usage tracker.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.lock = threading.RLock()

        # Pricing per 1M tokens (as of 2026-02-16)
        self.pricing = {
            # Anthropic Claude 4.x/4.6
            'claude-opus-4-6': {'input': 15.00, 'output': 75.00},
            'claude-sonnet-4-5-20250929': {'input': 3.00, 'output': 15.00},
            'claude-haiku-4-5-20251001': {'input': 0.80, 'output': 4.00},

            # Anthropic Claude 3.x (legacy)
            'claude-3-opus-20240229': {'input': 15.00, 'output': 75.00},
            'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
            'claude-3-5-sonnet-20240620': {'input': 3.00, 'output': 15.00},
            'claude-3-sonnet-20240229': {'input': 3.00, 'output': 15.00},
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},

            # OpenAI GPT-4
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-4': {'input': 30.00, 'output': 60.00},
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
        }

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Create usage_log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    input_cost_usd REAL DEFAULT 0.0,
                    output_cost_usd REAL DEFAULT 0.0,
                    total_cost_usd REAL DEFAULT 0.0,
                    document_id INTEGER,
                    success INTEGER DEFAULT 1,
                    error TEXT
                )
            ''')

            # Create index on timestamp for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp
                ON usage_log(timestamp)
            ''')

            # Create index on model for aggregations
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_usage_model
                ON usage_log(model)
            ''')

            conn.commit()
            conn.close()

            logger.info(f"LLM usage tracker initialized at {self.db_path}")

    def log_usage(self, provider: str, model: str, operation: str,
                  input_tokens: int, output_tokens: int,
                  document_id: int = None, success: bool = True,
                  error: str = None):
        """
        Log an LLM API call.

        Args:
            provider: Provider name (anthropic, openai, etc.)
            model: Model name
            operation: Operation type (analysis, integrity_check, metadata_extraction, etc.)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            document_id: Associated document ID
            success: Whether the call succeeded
            error: Error message if failed
        """
        with self.lock:
            # Calculate costs
            total_tokens = input_tokens + output_tokens
            pricing = self.pricing.get(model, {'input': 0, 'output': 0})

            input_cost = (input_tokens / 1_000_000) * pricing['input']
            output_cost = (output_tokens / 1_000_000) * pricing['output']
            total_cost = input_cost + output_cost

            # Insert into database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO usage_log
                (provider, model, operation, input_tokens, output_tokens, total_tokens,
                 input_cost_usd, output_cost_usd, total_cost_usd, document_id, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (provider, model, operation, input_tokens, output_tokens, total_tokens,
                  input_cost, output_cost, total_cost, document_id, 1 if success else 0, error))

            conn.commit()
            conn.close()

            logger.debug(f"Logged LLM usage: {model} - {total_tokens} tokens (${total_cost:.4f})")

    def get_usage_stats(self, days: int = 30) -> Dict:
        """
        Get usage statistics for the last N days.

        Args:
            days: Number of days to look back

        Returns:
            Dict with usage statistics
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Calculate date threshold
            since_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

            # Overall stats
            cursor.execute('''
                SELECT
                    COUNT(*) as total_calls,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(total_cost_usd) as total_cost,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls
                FROM usage_log
                WHERE timestamp >= ?
            ''', (since_date,))

            overall = dict(cursor.fetchone())

            # Per-model breakdown
            cursor.execute('''
                SELECT
                    model,
                    COUNT(*) as calls,
                    SUM(input_tokens) as input_tokens,
                    SUM(output_tokens) as output_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(total_cost_usd) as cost
                FROM usage_log
                WHERE timestamp >= ?
                GROUP BY model
                ORDER BY cost DESC
            ''', (since_date,))

            per_model = [dict(row) for row in cursor.fetchall()]

            # Per-operation breakdown
            cursor.execute('''
                SELECT
                    operation,
                    COUNT(*) as calls,
                    SUM(total_tokens) as total_tokens,
                    SUM(total_cost_usd) as cost
                FROM usage_log
                WHERE timestamp >= ?
                GROUP BY operation
                ORDER BY cost DESC
            ''', (since_date,))

            per_operation = [dict(row) for row in cursor.fetchall()]

            # Daily usage (last 7 days)
            cursor.execute('''
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as calls,
                    SUM(total_tokens) as tokens,
                    SUM(total_cost_usd) as cost
                FROM usage_log
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 7
            ''', (since_date,))

            daily_usage = [dict(row) for row in cursor.fetchall()]

            conn.close()

            return {
                'period_days': days,
                'overall': overall,
                'per_model': per_model,
                'per_operation': per_operation,
                'daily_usage': daily_usage
            }

    def get_recent_calls(self, limit: int = 50) -> List[Dict]:
        """
        Get recent LLM API calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of recent calls
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT
                    id, timestamp, provider, model, operation,
                    input_tokens, output_tokens, total_tokens,
                    total_cost_usd, document_id, success, error
                FROM usage_log
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            calls = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return calls

    def get_cost_by_period(self, period: str = 'day', limit: int = 30) -> List[Dict]:
        """
        Get cost aggregated by time period.

        Args:
            period: 'day', 'week', or 'month'
            limit: Number of periods to return

        Returns:
            List of period costs
        """
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Determine grouping
            if period == 'day':
                group_by = "DATE(timestamp)"
            elif period == 'week':
                group_by = "strftime('%Y-W%W', timestamp)"
            elif period == 'month':
                group_by = "strftime('%Y-%m', timestamp)"
            else:
                group_by = "DATE(timestamp)"

            cursor.execute(f'''
                SELECT
                    {group_by} as period,
                    COUNT(*) as calls,
                    SUM(input_tokens) as input_tokens,
                    SUM(output_tokens) as output_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(total_cost_usd) as cost
                FROM usage_log
                GROUP BY {group_by}
                ORDER BY period DESC
                LIMIT ?
            ''', (limit,))

            results = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return results

    def clear_old_data(self, days: int = 90):
        """
        Delete usage data older than N days.

        Args:
            days: Keep data from last N days
        """
        with self.lock:
            delete_before = (datetime.utcnow() - timedelta(days=days)).isoformat()

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                DELETE FROM usage_log
                WHERE timestamp < ?
            ''', (delete_before,))

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Deleted {deleted} usage records older than {days} days")

    def get_pricing(self) -> Dict[str, Dict]:
        """
        Get current pricing information.

        Returns:
            Dict of model pricing (per 1M tokens)
        """
        return self.pricing.copy()

    def update_pricing(self, model: str, input_price: float, output_price: float):
        """
        Update pricing for a model.

        Args:
            model: Model name
            input_price: Input price per 1M tokens
            output_price: Output price per 1M tokens
        """
        self.pricing[model] = {
            'input': input_price,
            'output': output_price
        }
        logger.info(f"Updated pricing for {model}: ${input_price}/{output_price} per 1M tokens")
