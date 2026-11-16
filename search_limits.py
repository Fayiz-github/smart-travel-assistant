"""Persistent, process-safe SerpApi request budget.

SQLite serializes quota acquisition across limiter instances and processes
sharing the same database. Minute limits use a rolling 60-second window so
requests cannot evade the limit at a wall-clock minute boundary.
For multi-host deployments, use Redis or a central quota service.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import sqlite3
import threading


class SearchBudgetExceeded(RuntimeError):
    """Raised before a provider call when a configured quota has been reached."""


class ProviderRateLimiter:
    def __init__(
        self,
        per_minute: int = 20,
        per_day: int = 100,
        per_month: int = 200,
        db_path: str | os.PathLike[str] | None = None,
    ):
        self.per_minute = max(1, int(per_minute))
        self.per_day = max(1, int(per_day))
        self.per_month = max(1, int(per_month))
        configured = Path(
            db_path or os.getenv("SERPAPI_LIMIT_DB_PATH", "search_limits.db")
        ).expanduser()
        if not configured.is_absolute():
            configured = Path(__file__).resolve().parent / configured
        self.db_path = str(configured.resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # This lock protects this instance; BEGIN IMMEDIATE protects all
        # instances/processes that share the SQLite database.
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_usage (
                    period_type TEXT NOT NULL,
                    period_key TEXT NOT NULL,
                    calls INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(period_type, period_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_request_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    requested_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_request_events_time
                ON provider_request_events(requested_at)
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_provider_usage_updated "
                "ON provider_usage(updated_at)"
            )
        finally:
            conn.close()

    def acquire(self, provider: str = "serpapi") -> None:
        # `provider` is retained for API compatibility. Flight and hotel calls
        # share the same SerpApi account quota and therefore the same counters.
        with self._lock:
            conn = self._connect()
            try:
                # Acquire the write reservation before reading any counters.
                # SQLite makes this transaction mutually exclusive across
                # independent ProviderRateLimiter objects and processes.
                conn.execute("BEGIN IMMEDIATE")

                now = datetime.now(timezone.utc)
                now_text = now.isoformat(timespec="microseconds")
                minute_cutoff = (now - timedelta(seconds=60)).isoformat(
                    timespec="microseconds"
                )
                day_key = now.strftime("%Y-%m-%d")
                month_key = now.strftime("%Y-%m")

                # Rolling 60-second window, rather than a wall-clock minute
                # bucket, prevents a boundary burst from doubling the quota.
                conn.execute(
                    "DELETE FROM provider_request_events WHERE requested_at <= ?",
                    (minute_cutoff,),
                )
                minute_count = conn.execute(
                    "SELECT COUNT(*) FROM provider_request_events "
                    "WHERE requested_at > ?",
                    (minute_cutoff,),
                ).fetchone()[0]
                if int(minute_count) >= self.per_minute:
                    raise SearchBudgetExceeded(
                        f"SerpApi minute search budget reached "
                        f"({self.per_minute} requests/rolling 60 seconds). "
                        "Please retry after the limit resets or adjust the "
                        "configured limit to match your provider plan."
                    )

                for period_type, period_key, maximum in (
                    ("day", day_key, self.per_day),
                    ("month", month_key, self.per_month),
                ):
                    row = conn.execute(
                        "SELECT calls FROM provider_usage "
                        "WHERE period_type=? AND period_key=?",
                        (period_type, period_key),
                    ).fetchone()
                    count = int(row[0]) if row else 0
                    if count >= maximum:
                        raise SearchBudgetExceeded(
                            f"SerpApi {period_type} search budget reached "
                            f"({maximum} requests/{period_type}). Please retry "
                            "after the period resets or adjust the configured "
                            "limit to match your provider plan."
                        )

                conn.execute(
                    "INSERT INTO provider_request_events(requested_at) VALUES (?)",
                    (now_text,),
                )
                for period_type, period_key in (
                    ("day", day_key),
                    ("month", month_key),
                ):
                    conn.execute(
                        """
                        INSERT INTO provider_usage
                            (period_type, period_key, calls, updated_at)
                        VALUES (?, ?, 1, ?)
                        ON CONFLICT(period_type, period_key)
                        DO UPDATE SET
                            calls=provider_usage.calls+1,
                            updated_at=excluded.updated_at
                        """,
                        (period_type, period_key, now_text),
                    )

                # Bound persistent history.
                conn.execute(
                    "DELETE FROM provider_request_events WHERE requested_at < ?",
                    ((now - timedelta(hours=2)).isoformat(timespec="microseconds"),),
                )
                conn.execute(
                    "DELETE FROM provider_usage "
                    "WHERE period_type='day' AND period_key < ?",
                    ((now - timedelta(days=40)).strftime("%Y-%m-%d"),),
                )
                conn.execute(
                    "DELETE FROM provider_usage "
                    "WHERE period_type='month' AND period_key < ?",
                    ((now - timedelta(days=400)).strftime("%Y-%m"),),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()


def _safe_positive_env_int(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return min(max(1, value), maximum)


MAX_CALLS_PER_MINUTE = _safe_positive_env_int(
    "SERPAPI_MAX_CALLS_PER_MINUTE", 20, 1000
)
MAX_CALLS_PER_DAY = _safe_positive_env_int(
    "SERPAPI_MAX_CALLS_PER_DAY", 100, 100000
)
# Conservative default; override to the actual monthly allowance in .env.
MAX_CALLS_PER_MONTH = _safe_positive_env_int(
    "SERPAPI_MAX_CALLS_PER_MONTH", 200, 1000000
)
SERPAPI_LIMITER = ProviderRateLimiter(
    MAX_CALLS_PER_MINUTE,
    MAX_CALLS_PER_DAY,
    MAX_CALLS_PER_MONTH,
)
