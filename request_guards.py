"""Deterministic guardrails for search-tool confirmation.

This module intentionally has no LangChain dependency so its logic can be tested
without constructing a model or graph.
"""
from __future__ import annotations
import json
import re
from typing import Any


def _normalise_confirmation_text(text: str) -> str:
    """Normalize punctuation while preserving words so compound requests stay distinct."""
    value = (text or "").lower().replace("’", "'").strip()
    value = re.sub(r"[^a-z0-9']+", " ", value)
    return " ".join(value.split())


def is_strict_affirmative(text: str) -> bool:
    """Accept clear standalone approvals, including natural variants like 'yes absolutely'.

    A message such as 'yes, but change the date' remains false because it is not
    an exact approval phrase after normalization.
    """
    value = _normalise_confirmation_text(text)
    return value in {
        "yes", "yes please", "yes absolutely", "absolutely", "yes definitely",
        "definitely", "yep", "yeah", "sure", "sure thing", "sure go ahead",
        "correct", "that's correct", "that is correct", "confirm", "confirmed",
        "proceed", "go ahead", "okay", "ok", "yes proceed", "please proceed",
        "please do", "go ahead please", "yes go ahead", "yes please proceed",
    }


def is_search_confirmation_prompt(text: str) -> bool:
    """Recognize either an explicit search-confirmation question or a search promise.

    A later standalone `search` is safe to accept only when this function confirms
    that the immediately preceding assistant turn had already requested approval
    or told the user the approved search was about to run.
    """
    value = _normalise_confirmation_text(text)
    patterns = (
        r"\bshould i (?:search|run|check)\b",
        r"\bshall i (?:search|run|check)\b",
        r"\bwould you like me to (?:search|run|check)\b",
        r"\bdo you want me to (?:search|run|check)\b",
        r"\bmay i proceed with (?:the )?(?:flight |hotel )?search\b",
        r"\b(?:please )?confirm (?:these )?(?:(?:flight|hotel) )?search(?: details)?\b",
        r"\bconfirm the (?:flight|hotel) search details\b",
        r"\bconfirm (?:these|the) (?:flight |hotel )?details\b",
        r"\bconfirm.*\b(?:i will|i can|i'll|so i can|and i will|and i'll)\s+(?:now\s+)?(?:search|run|check)\b",
        r"\bbefore i search\b",
        # Some specialist turns state that the user-approved search is starting
        # without using a question mark. A subsequent standalone 'search' is
        # then an explicit instruction to carry out that already-discussed task.
        r"\b(?:i|we)\s+will\s+(?:now\s+)?search\b",
        r"\bi'll\s+(?:now\s+)?search\b",
        r"\b(?:i am|i'm)\s+(?:now\s+)?searching\b",
        r"\bplease hold\b.{0,120}\b(?:search|gather|retrieve)\b",
        r"\bsearch(?:ing)?\s+(?:will|is about to)\s+start\b",
    )
    return any(re.search(pattern, value) for pattern in patterns)


def has_explicit_search_now(text: str) -> bool:
    value = _normalise_confirmation_text(text)
    return value in {
        "search", "please search", "search now", "run search", "run the search",
        "start search", "start the search", "go ahead", "go ahead and search",
        "proceed", "proceed with search", "proceed with the search", "please proceed",
        "do the search", "yes search", "yes please search",
    } or bool(re.search(
        r"\b(?:search now|run the search|go ahead and search|proceed with (?:the )?search)\b",
        value,
    ))


def canonical_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize relevant fields/defaults so a confirmed search cannot silently mutate."""
    raw = args if isinstance(args, dict) else {}
    def get_str(key: str, default: str = "") -> str:
        value = raw.get(key, default)
        return str(value if value is not None else default).strip()
    def get_int(key: str, default: int) -> int:
        value = raw.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    def get_bool(key: str, default: bool = False) -> bool:
        value = raw.get(key, default)
        return value if isinstance(value, bool) else default
    def get_float(key: str, default: float = 0.0) -> float:
        value = raw.get(key, default)
        try:
            number = float(value)
            return number if number == number and abs(number) != float("inf") else default
        except (TypeError, ValueError):
            return default
    def get_int_list(key: str) -> list[int]:
        value = raw.get(key, [])
        if not isinstance(value, list):
            return []
        output: list[int] = []
        for item in value:
            try:
                output.append(int(item))
            except (TypeError, ValueError):
                output.append(-999999)
        return output

    if tool_name == "search_flights_tool":
        return {
            "dep_id": get_str("dep_id").upper(),
            "arr_id": get_str("arr_id").upper(),
            "target_date": get_str("target_date"),
            "trip_type": get_str("trip_type", "2"),
            "return_date": get_str("return_date", "NONE").upper(),
            "adults": get_int("adults", 1),
            "children": get_int("children", 0),
            "currency": get_str("currency", "INR").upper(),
            "stops": get_str("stops", "0"),
            "travel_class": get_str("travel_class", "1"),
            "force_refresh": get_bool("force_refresh"),
        }
    if tool_name == "compare_flight_dates_tool":
        dates = raw.get("target_dates", [])
        dates = [str(x).strip() for x in dates] if isinstance(dates, list) else []
        return {
            "dep_id": get_str("dep_id").upper(),
            "arr_id": get_str("arr_id").upper(),
            "target_dates": dates,
            "trip_type": get_str("trip_type", "2"),
            "return_date": get_str("return_date", "NONE").upper(),
            "adults": get_int("adults", 1),
            "children": get_int("children", 0),
            "currency": get_str("currency", "INR").upper(),
            "stops": get_str("stops", "0"),
            "travel_class": get_str("travel_class", "1"),
            "force_refresh": get_bool("force_refresh"),
        }
    if tool_name == "explore_destinations_tool":
        airports = raw.get("candidate_airports")
        if not isinstance(airports, list):
            airports = ["BLR", "DEL", "BOM", "MAA", "GOI", "DXB", "SIN", "KUL", "BKK"]
        return {
            "dep_id": get_str("dep_id").upper(),
            "candidate_airports": [str(x).strip().upper() for x in airports],
            "date_str": get_str("date_str"),
            "budget": round(get_float("budget"), 2),
            "trip_type": get_str("trip_type", "2"),
            "return_date": get_str("return_date", "NONE").upper(),
            "adults": get_int("adults", 1),
            "children": get_int("children", 0),
            "currency": get_str("currency", "INR").upper(),
            "stops": get_str("stops", "0"),
            "travel_class": get_str("travel_class", "1"),
        }
    if tool_name == "search_hotels_tool":
        ages = get_int_list("children_ages")
        return {
            "location_query": get_str("location_query").casefold(),
            "check_in_date": get_str("check_in_date"),
            "check_out_date": get_str("check_out_date"),
            "adults": get_int("adults", 2),
            "children": get_int("children", 0),
            "children_ages": ages,
            "rooms": get_int("rooms", 1),
            "currency": get_str("currency", "INR").upper(),
            "min_star": raw.get("min_star"),
            "only_exact_star": get_bool("only_exact_star"),
            "food_preference": get_str("food_preference").casefold(),
            "force_refresh": get_bool("force_refresh"),
        }
    return {str(k): raw[k] for k in sorted(raw)}


def tool_call_record(tool_call: dict[str, Any]) -> dict[str, Any]:
    name = str(tool_call.get("name", ""))
    normalized = canonical_tool_args(name, tool_call.get("args", {}))
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {"tool_name": name, "fingerprint": serialized, "args": normalized}


def confirmation_decision(
    tool_calls: list[dict[str, Any]],
    pending_record: dict[str, Any] | None,
    latest_user_text: str,
    confirmation_was_requested: bool,
) -> tuple[bool, dict[str, Any] | None, str]:
    """Return (allowed, pending_record_to_store, reason)."""
    candidate_calls = [tool_call_record(call) for call in tool_calls]
    candidate = {"calls": candidate_calls}
    # An imperative such as 'search' is sufficient only after the assistant
    # has already presented/confirmed a search or explicitly promised to run it.
    # It must never bypass the first confirmation step for a new search.
    explicit_now = has_explicit_search_now(latest_user_text) and confirmation_was_requested
    affirmative = is_strict_affirmative(latest_user_text) and confirmation_was_requested

    if pending_record is not None and pending_record.get("calls") != candidate_calls:
        return False, candidate, "Search parameters differ from the previously presented details. Ask the user to confirm the updated details."
    if explicit_now:
        return True, pending_record, "explicit search request"
    if affirmative:
        return True, pending_record, "user confirmed the presented search details"
    return False, pending_record or candidate, "user confirmation is missing or is not a standalone confirmation"
