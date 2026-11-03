"""Process-wide SerpApi request budget. For multi-user deployments, replace with Redis-backed limits."""
from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
import os
import threading
import time

class SearchBudgetExceeded(RuntimeError):
    pass

class ProviderRateLimiter:
    def __init__(self, per_minute: int = 20, per_day: int = 200):
        self.per_minute = max(1, int(per_minute))
        self.per_day = max(1, int(per_day))
        self._lock = threading.Lock()
        self._recent: deque[float] = deque()
        self._day = datetime.now(timezone.utc).date()
        self._daily_count = 0

    def acquire(self, provider: str = "serpapi") -> None:
        now = time.monotonic()
        today = datetime.now(timezone.utc).date()
        with self._lock:
            if today != self._day:
                self._day = today
                self._daily_count = 0
                self._recent.clear()
            while self._recent and now - self._recent[0] >= 60:
                self._recent.popleft()
            if len(self._recent) >= self.per_minute:
                raise SearchBudgetExceeded(
                    f"Search rate limit reached ({self.per_minute} requests/minute). Please retry shortly."
                )
            if self._daily_count >= self.per_day:
                raise SearchBudgetExceeded(
                    f"Daily search budget reached ({self.per_day} requests/day). Try again tomorrow."
                )
            self._recent.append(now)
            self._daily_count += 1

def _safe_positive_env_int(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return min(max(1, value), maximum)


MAX_CALLS_PER_MINUTE = _safe_positive_env_int("SERPAPI_MAX_CALLS_PER_MINUTE", 20, 1000)
MAX_CALLS_PER_DAY = _safe_positive_env_int("SERPAPI_MAX_CALLS_PER_DAY", 100, 100000)
SERPAPI_LIMITER = ProviderRateLimiter(MAX_CALLS_PER_MINUTE, MAX_CALLS_PER_DAY)
