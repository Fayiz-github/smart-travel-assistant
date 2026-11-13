"""Deterministic extraction of explicit travel preferences and constraints.

This module deliberately does not use an LLM and never turns an inference into
an asserted user preference. It updates only fields supported by the current
message, while retaining previously confirmed values.
"""
from __future__ import annotations

import re
from datetime import datetime
from copy import deepcopy
from typing import Any


DEFAULT_PREFERENCE_PROFILE: dict[str, Any] = {
    "budget_status": "unspecified",  # unspecified|provided|flexible|undecided|declined
    "budget_amount": None,
    "budget_currency": None,
    "budget_scope": None,             # total|per_person|per_day|unspecified
    "price_sensitivity": "unknown", # unknown|budget|value|premium|flexible
    "cabin_class": None,              # economy|premium_economy|business|first
    "traveler_count": None,           # explicitly stated party size
    "passport_country": None,        # explicitly stated passport nationality; never inferred
    "travel_purpose": None,           # tourism|business|transit|study|work|other when explicit
    "departure_date": None,           # ISO date when explicitly stated
    "return_date": None,              # ISO date when explicitly stated
    "accommodation_style": None,      # budget|mid_range|luxury|hostel|apartment|resort|hotel
    "interests": [],
    "constraints": [],
    "service_search_preferences": {
        "flight_search": "not_stated",  # not_stated|declined|deferred|requested
        "hotel_search": "not_stated",
    },
    "booking_preference": "confirmation_required",
}

_CURRENCY_PATTERNS = (
    ("INR", r"₹|\bINR\b|\brupees?\b"),
    ("USD", r"\$|\bUSD\b|\bdollars?\b"),
    ("EUR", r"€|\bEUR\b|\beuros?\b"),
    ("GBP", r"£|\bGBP\b|\bpounds?\b"),
    ("MYR", r"\bMYR\b|\bringgit\b"),
    ("AED", r"\bAED\b|\bdirhams?\b"),
    ("SGD", r"\bSGD\b|\bSingapore dollars?\b"),
    ("AUD", r"\bAUD\b|\bAustralian dollars?\b"),
    ("CAD", r"\bCAD\b|\bCanadian dollars?\b"),
    ("JPY", r"\bJPY\b|\byen\b"),
)

_AMOUNT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<amount>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,3}(?:,\d{2})+,\d{3}(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<suffix>[kKmM])?"
)


def _normalise(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _has_any(value: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)


def _extract_budget(text: str) -> tuple[float | None, str | None]:
    value = text or ""
    lower = _normalise(value)
    # Support common South Asian amount units without interpreting a bare
    # year/date as a budget.
    local_unit = re.search(
        r"\b(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>lakhs?|lacs?|crores?)\b",
        lower,
        re.IGNORECASE,
    )
    if local_unit:
        amount = float(local_unit.group("number"))
        unit = local_unit.group("unit").lower()
        amount *= 10_000_000 if unit.startswith("crore") else 100_000
        currency = "INR"
        return amount, currency

    has_budget_context = _has_any(lower, (
        r"\bbudget\b", r"\bspend(?:ing)?\b", r"\bafford\b", r"\bprice\s+limit\b",
        r"\bmaximum\s+(?:amount|spend|price)\b", r"\bup\s+to\b",
    ))
    currency = None
    for code, pattern in _CURRENCY_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            currency = code
            break

    # A currency-marked amount is explicit even if the word "budget" is absent.
    for match in _AMOUNT_RE.finditer(value):
        raw = match.group("amount").replace(",", "")
        try:
            amount = float(raw)
        except ValueError:
            continue
        suffix = (match.group("suffix") or "").lower()
        if suffix == "k":
            amount *= 1_000
        elif suffix == "m":
            amount *= 1_000_000
        has_currency_nearby = bool(currency) and (
            abs(match.start() - value.lower().find(currency.lower())) < 35
            or any(re.search(p, value[max(0, match.start()-18):match.end()+18], re.IGNORECASE) for _, p in _CURRENCY_PATTERNS)
        )
        if has_budget_context or has_currency_nearby:
            return amount, currency
    return None, currency


def _update_service_search_preferences(profile: dict[str, Any], value: str) -> None:
    services = profile.setdefault("service_search_preferences", {})
    for service, nouns in (
        ("flight_search", r"(?:flights?|airfare|air\s*tickets?)"),
        ("hotel_search", r"(?:hotels?|accommodation|hostels?|resorts?|lodging)"),
    ):
        mentioned = re.search(rf"\b{nouns}\b", value, re.IGNORECASE)
        if not mentioned:
            continue
        before = value[max(0, mentioned.start()-75):mentioned.start()]
        after = value[mentioned.end():mentioned.end()+35]
        negated = _has_any(before, (
            r"\b(?:don't|do not|dont|never|no|not interested in|not looking for|avoid|without)\b",
            r"\b(?:i\s+)?(?:don't|do not|dont)\s+(?:want|need)\b",
            r"\bno\s+need\s+for\b",
        ))
        deferred = negated and _has_any(after, (r"\byet\b", r"\bfor now\b", r"\bat this stage\b", r"\blater\b", r"\bright now\b", r"\bnot now\b"))
        if deferred:
            services[service] = "deferred"
        elif negated:
            services[service] = "declined"
        elif _has_any(value, (
            rf"\b(?:search|find|compare|look for|show me|check availability of)\b.{{0,70}}\b{nouns}\b",
            rf"\b{nouns}\b.{{0,50}}\b(?:search|find|compare|look for)\b",
        )):
            services[service] = "requested"


_MONTH_DATE_RE = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(?P<date>\d{4}-\d{2}-\d{2})\b")


def _extract_cued_date(value: str, cue_pattern: str) -> str | None:
    """Extract an explicit date only when attached to a departure/return cue."""
    cue = re.search(cue_pattern, value, re.IGNORECASE)
    if not cue:
        return None
    tail = value[cue.end():cue.end() + 90]
    iso = _ISO_DATE_RE.search(tail)
    if iso:
        try:
            return datetime.strptime(iso.group("date"), "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None
    month_match = _MONTH_DATE_RE.search(tail)
    if month_match:
        month_token = month_match.group("month")
        month_number = datetime.strptime(month_token[:3].title(), "%b").month
        try:
            parsed = datetime(int(month_match.group("year")), month_number, int(month_match.group("day")))
            return parsed.date().isoformat()
        except ValueError:
            return None
    return None


def is_trip_summary_request(user_message: str) -> bool:
    """Recognize a request to recap the active trip rather than start a search."""
    value = _normalise(user_message)
    return bool(re.search(
        r"\b(?:summari[sz]e|recap|repeat|list|show)\b.{0,70}\b(?:my|our|the)?\s*(?:trip|travel)\s*(?:details|information|plan|preferences)?\b"
        r"|\bwhat\s+(?:trip|travel)\s+(?:details|information)\s+(?:do\s+you\s+have|have\s+i\s+given|do\s+you\s+know)\b"
        r"|\bwhat\s+do\s+you\s+have\s+(?:about|for)\s+(?:my|our)\s+trip\b"
        r"|\bwhat\s+details\s+do\s+you\s+have\s+(?:about|for)\s+(?:my|our)\s+trip\b",
        value,
        re.IGNORECASE,
    ))


def is_trip_detail_message(user_message: str) -> bool:
    """Recognize a partial answer or preference during general trip intake.

    This is intentionally broader than a single pending slot: travellers often
    volunteer dates, party size, cabin class, interests, constraints, or budget
    preferences in any order. It does not authorize a search.
    """
    value = _normalise(user_message)
    if not value:
        return False
    patterns = (
        # Dates and duration
        r"\b(?:today|tomorrow|this weekend|next week|next month|\d{4}-\d{2}-\d{2}|"
        r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|"
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2})\b",
        r"\b\d+\s*(?:days?|nights?|weeks?)\b|\b(?:for|about)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\b",
        # Party size
        r"\b(?:there\s+are|we\s+are|party\s+of|family\s+of|travell?ing\s+with)\s+"
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
        r"(?:travell?ers?|people|persons?|passengers?|adults?|guests?)\b",
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:travell?ers?|people|persons?|passengers?|adults?|guests?)\b",
        # Class and accommodation preferences
        r"\b(?:economy|premium economy|business class|first class)\b",
        r"\b(?:hostel|hotel|resort|apartment|luxury stay|budget stay|mid[- ]range stay)\b",
        # Interests and travel style
        r"\b(?:sightseeing|food|foodie|shopping|nature|hiking|beaches?|culture|heritage|"
        r"nightlife|family[- ]friendly|relaxation|adventure|local cuisine|museums?)\b",
        # Budget preferences, including a request not to be asked
        r"\b(?:budget|price|cheapest|cheap|affordable|premium|luxury|best value|"
        r"don't ask me|do not ask me|not decided|not sure|prefer not to share)\b",
        # Common explicit constraints
        r"\b(?:direct flights?|non[- ]stop|no layovers?|no stopovers?|window seat|aisle seat|"
        r"wheelchair access|non[- ]smoking|don't book|do not book|without asking me)\b",
        r"\b(?:flexible dates?|dates? are flexible|any dates?|no fixed dates?)\b",
    )
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)



def update_preference_profile(existing: dict[str, Any] | None, user_message: str) -> dict[str, Any]:
    """Return a merged profile using only explicit evidence in user_message."""
    profile = deepcopy(DEFAULT_PREFERENCE_PROFILE)
    if isinstance(existing, dict):
        for key, value in existing.items():
            if key in profile:
                profile[key] = deepcopy(value)
    return profile


__all__ = ["DEFAULT_PREFERENCE_PROFILE", "is_trip_detail_message", "is_trip_summary_request", "update_preference_profile"]
