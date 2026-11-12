"""Scheduled runner for proactive travel-wishlist risk checks.

Run once from Windows Task Scheduler, or run continuously in a second terminal:
    python travel_risk_scheduler.py --once
    python travel_risk_scheduler.py

Only destinations whose next_check_at is due are searched. The monitor writes
checks and alerts to the same SQLite database used by the main Travel AI app.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_risk_monitor import make_monitor_from_environment


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _append_notification(path: str, item: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_once(owner_id: str | None = None, max_destinations: int = 5) -> int:
    monitor = make_monitor_from_environment()
    owner = owner_id or monitor.owner_id
    results = monitor.check_all_due(owner_id=owner, max_destinations=max_destinations)
    if not results:
        print(f"[{datetime.now(timezone.utc).isoformat()}] No wishlist destinations are due for a fresh check.")
        return 0

    notification_file = os.getenv("RISK_NOTIFICATION_LOG", "risk_notifications.jsonl")
    for result in results:
        label = result.get("city") or result.get("country") or f"destination_id={result.get('destination_id')}"
        level = str(result.get("risk_level") or "unknown").upper()
        print(f"[{result.get('checked_at', 'now')}] {label}: {level} ({result.get('status')})")
        if result.get("new_alert"):
            alert_record = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "destination_id": result.get("destination_id"),
                "destination": label,
                "risk_level": level,
                "summary": result.get("summary", ""),
                "sources": result.get("sources", []),
            }
            _append_notification(notification_file, alert_record)
            print(f"  NEW {level} ALERT: saved to {notification_file}")
    return len(results)


def main() -> int:
    _load_environment()
    parser = argparse.ArgumentParser(description="Check due travel-wishlist destinations for new travel risks.")
    parser.add_argument("--once", action="store_true", help="Run one due-check cycle and exit.")
    parser.add_argument("--owner", default=None, help="Wishlist owner id; defaults to TRAVEL_USER_ID in .env.")
    parser.add_argument("--max-destinations", type=int, default=5, help="Maximum destinations per cycle (1-20).")
    args = parser.parse_args()
    max_destinations = max(1, min(args.max_destinations, 20))

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.once:
        try:
            run_once(args.owner, max_destinations)
            return 0
        except Exception as exc:
            logging.exception("Risk-monitor cycle failed")
            print(f"Risk monitor could not start: {type(exc).__name__}: {exc}")
            return 1

    interval_minutes = _env_int("RISK_SCHEDULER_INTERVAL_MINUTES", 60, 5, 1440)
    print(f"Travel wishlist risk scheduler running. Polling every {interval_minutes} minutes. Press Ctrl+C to stop.")
    while True:
        try:
            run_once(args.owner, max_destinations)
        except KeyboardInterrupt:
            print("Risk scheduler stopped.")
            return 0
        except Exception:
            logging.exception("Risk-monitor cycle failed; the scheduler will retry at the next interval.")
        try:
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("Risk scheduler stopped.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
