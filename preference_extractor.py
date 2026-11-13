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
    """Return a merged profile using only explicit evidence in ``user_message``."""
    profile = deepcopy(DEFAULT_PREFERENCE_PROFILE)
    if isinstance(existing, dict):
        for key, value in existing.items():
            if key not in profile:
                continue
            default_value = profile[key]
            # None is an intentional empty/default value for optional fields
            # (e.g. cabin_class, traveler_count, budget_amount). Accept a
            # previously extracted non-None value so it survives later turns.
            if default_value is None or isinstance(value, type(default_value)):
                profile[key] = deepcopy(value)
        # Keep forward-compatible fields that may be stored by newer releases.
        for key, value in existing.items():
            if key not in profile:
                profile[key] = deepcopy(value)

    value = _normalise(user_message)
    if not value:
        return profile

    # Explicitly cued trip dates are stored separately from generic dates so a
    # year in another context (or a date mentioned in a side question) is not
    # silently assigned to the itinerary.
    departure = _extract_cued_date(value, r"\b(?:depart(?:ure)?|leave|leaving|outbound|start(?:ing)?)\b(?:\s+on)?")
    if departure:
        profile["departure_date"] = departure
    returning = _extract_cued_date(value, r"\b(?:return(?:ing)?|come\s+back|coming\s+back|back\s+on)\b(?:\s+on)?")
    if returning:
        profile["return_date"] = returning

    # Budget: optional, and the state distinguishes not mentioned from a refusal.
    if _has_any(value, (
        r"\bno\s+(?:fixed|specific|set)\s+budget\b", r"\bbudget\s+is\s+flexible\b",
        r"\bflexible\s+(?:on|with|about)\s+(?:the\s+)?budget\b", r"\bprice\s+isn't\s+an\s+issue\b",
        r"\bdon't\s+worry\s+about\s+(?:the\s+)?price\b", r"\bany\s+price\s+is\s+fine\b",
        r"\bregardless\s+of\s+(?:the\s+)?price\b",
    )):
        profile["budget_status"] = "flexible"
        profile["budget_amount"] = None
        profile["budget_currency"] = None
        profile["budget_scope"] = None
        profile["price_sensitivity"] = "flexible"
    elif _has_any(value, (
        r"\bnot\s+decided\s+(?:on\s+)?(?:my\s+)?budget\b", r"\bhaven't\s+decided\s+(?:on\s+)?(?:my\s+)?budget\b",
        r"\bnot\s+sure\s+(?:about|of)\s+(?:my\s+)?budget\b", r"\bdon't\s+know\s+my\s+budget\b",
    )):
        profile["budget_status"] = "undecided"
        profile["budget_amount"] = None
        profile["budget_currency"] = None
        profile["budget_scope"] = None
    elif _has_any(value, (
        r"\bprefer\s+not\s+to\s+(?:say|share|disclose)\s+(?:my\s+)?budget\b",
        r"\bdon't\s+want\s+to\s+(?:share|disclose)\s+(?:my\s+)?budget\b",
        r"\bnot\s+comfortable\s+sharing\s+(?:my\s+)?budget\b",
        r"\b(?:don't|do not|dont)\s+ask\s+me\s+(?:(?:about|for)\s+)?(?:a\s+)?budget\b",
        r"\bno\s+budget\s+questions?\b",
    )):
        profile["budget_status"] = "declined"
        profile["budget_amount"] = None
        profile["budget_currency"] = None
        profile["budget_scope"] = None
    else:
        amount, currency = _extract_budget(user_message)
        if amount is not None:
            profile["budget_status"] = "provided"
            profile["budget_amount"] = amount
            profile["budget_currency"] = currency
            if _has_any(value, (r"\bper\s+person\b", r"\beach\b", r"\beach\s+travell?er\b")):
                profile["budget_scope"] = "per_person"
            elif _has_any(value, (r"\bper\s+day\b", r"\bdaily\b")):
                profile["budget_scope"] = "per_day"
            elif _has_any(value, (r"\btotal\b", r"\bfor\s+the\s+whole\s+trip\b", r"\ball\s+in\b")):
                profile["budget_scope"] = "total"
            else:
                profile["budget_scope"] = "unspecified"

    # Price sensitivity is distinct from the numeric budget.
    if _has_any(value, (r"\bcheapest\b", r"\blowest\s+price\b", r"\bbudget[- ]friendly\b", r"\bcheap(?:est)?\b", r"\bkeep\s+costs?\s+low\b")):
        profile["price_sensitivity"] = "budget"
    elif _has_any(value, (r"\bbest\s+value\b", r"\breasonable\s+price\b", r"\bvalue\s+for\s+money\b", r"\bmid[- ]range\b")):
        profile["price_sensitivity"] = "value"
    elif _has_any(value, (r"\bluxury\b", r"\bpremium\b", r"\bhigh[- ]end\b", r"\btop[- ]class\b")):
        profile["price_sensitivity"] = "premium"

    # Passport nationality is personal information and must only be stored when
    # the user explicitly ties a country/demonym to a passport. Never infer it
    # from origin, residence, language, or government-source preferences.
    passport_patterns = (
        ("India", r"\b(?:indian\s+passport(?:s)?|passport\s+holders?\s+(?:are\s+)?indian|passport\s+from\s+india)\b"),
        ("United Kingdom", r"\b(?:british\s+passport(?:s)?|uk\s+passport(?:s)?|passport\s+from\s+(?:the\s+)?(?:uk|united kingdom))\b"),
        ("United States", r"\b(?:american\s+passport(?:s)?|us\s+passport(?:s)?|passport\s+from\s+(?:the\s+)?(?:us|united states))\b"),
        ("Ireland", r"\b(?:irish\s+passport(?:s)?|passport\s+from\s+ireland)\b"),
        ("Australia", r"\b(?:australian\s+passport(?:s)?|passport\s+from\s+australia)\b"),
        ("Canada", r"\b(?:canadian\s+passport(?:s)?|passport\s+from\s+canada)\b"),
        ("Germany", r"\b(?:german\s+passport(?:s)?|passport\s+from\s+germany)\b"),
        ("France", r"\b(?:french\s+passport(?:s)?|passport\s+from\s+france)\b"),
        ("United Arab Emirates", r"\b(?:emirati\s+passport(?:s)?|uae\s+passport(?:s)?|passport\s+from\s+(?:the\s+)?uae)\b"),
        ("Singapore", r"\b(?:singaporean\s+passport(?:s)?|passport\s+from\s+singapore)\b"),
        ("Malaysia", r"\b(?:malaysian\s+passport(?:s)?|passport\s+from\s+malaysia)\b"),
    )
    for country, pattern in passport_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            profile["passport_country"] = country
            break

    # Travel purpose is optional. If not supplied, downstream research may
    # provide clearly labelled general/tourism guidance without assuming it.
    purpose_patterns = (
        ("tourism", r"\b(?:tourism|tourist|holiday|vacation|leisure)\b"),
        ("business", r"\b(?:business\s+trip|business\s+travel|attend(?:ing)?\s+(?:a\s+)?conference|work\s+meeting)\b"),
        ("transit", r"\btransit(?:ing)?\b"),
        ("study", r"\b(?:study|student|education)\b"),
        ("work", r"\b(?:employment|work\s+visa|taking\s+up\s+work)\b"),
    )
    for purpose, pattern in purpose_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            profile["travel_purpose"] = purpose
            break

    # Traveller count: only record a count when the message explicitly labels
    # it as a party/traveller/person count (never mistake dates or money for it).
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    party_match = re.search(
        r"\b(?:there\s+are|we\s+are|party\s+of|family\s+of|travell?ing\s+with)?\s*"
        r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
        r"(?:travell?ers?|people|persons?|passengers?|adults?|guests?)\b",
        value,
        re.IGNORECASE,
    )
    if party_match:
        raw_count = party_match.group("count").lower()
        try:
            count = int(raw_count)
        except ValueError:
            count = number_words.get(raw_count)
        if count is not None and 1 <= count <= 50:
            profile["traveler_count"] = count

    # Cabin class: the latest explicit class mention wins.
    cabin_matches = list(re.finditer(r"\b(first\s+class|business\s+class|premium\s+economy|economy(?:\s+class)?)\b", value))
    if cabin_matches:
        cabin = cabin_matches[-1].group(1).replace(" ", "_")
        profile["cabin_class"] = {
            "first_class": "first", "business_class": "business",
            "premium_economy": "premium_economy", "economy": "economy", "economy_class": "economy",
        }[cabin]

    # Accommodation preference.
    accommodation_patterns = (
        ("hotel", r"\bhotels?\b"),
        ("hostel", r"\bhostels?\b"),
        ("apartment", r"\bapartments?\b|\bserviced\s+apartments?\b"),
        ("resort", r"\bresorts?\b"),
        ("luxury", r"\bluxury\s+(?:hotels?|accommodation|stay)\b"),
        ("budget", r"\bbudget\s+(?:hotels?|accommodation|stay)\b"),
        ("mid_range", r"\bmid[- ]range\s+(?:hotels?|accommodation|stay)\b"),
    )
    for style, pattern in accommodation_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            profile["accommodation_style"] = style

    # Interests are additive; only explicit topic words are recorded.
    interest_patterns = {
        "sightseeing": r"\b(?:sightseeing|landmarks?|attractions?|museums?)\b",
        "food": r"\b(?:food|foodie|local cuisine|restaurants?|street food)\b",
        "shopping": r"\bshopping\b|\bmarkets?\b",
        "nature": r"\b(?:nature|hiking|mountains?|waterfalls?|wildlife|parks?)\b",
        "beaches": r"\b(?:beaches?|islands?|snorkell?ing|diving)\b",
        "culture": r"\b(?:culture|cultural|heritage|temples?|history)\b",
        "nightlife": r"\bnightlife\b|\bclubs?\b|\bbars?\b",
        "family": r"\b(?:family[- ]friendly|kids?|children)\b",
        "relaxation": r"\b(?:relax|relaxation|spa|wellness)\b",
        "adventure": r"\b(?:adventure|adrenaline|zipline|rafting)\b",
    }
    interests = set(profile.get("interests") or [])
    for interest, pattern in interest_patterns.items():
        if re.search(pattern, value, re.IGNORECASE):
            interests.add(interest)
    profile["interests"] = sorted(interests)

    # Explicit constraints/preferences; avoid interpreting ordinary mentions as constraints.
    constraints = set(profile.get("constraints") or [])
    if _has_any(value, (r"\b(?:layovers?|stopovers?|connections?)\s+(?:are\s+)?fine\b", r"\bi\s+don't\s+mind\s+(?:a\s+)?(?:layover|stopover|connection)s?\b")):
        constraints.discard("no_layovers")
        constraints.discard("direct_flights_only")
    if _has_any(value, (r"\bwindow\s+seat\s+instead\b", r"\bprefer\s+a?\s*window\s+seat\b")):
        constraints.discard("aisle_seat")
    if _has_any(value, (r"\baisle\s+seat\s+instead\b", r"\bprefer\s+an?\s*aisle\s+seat\b")):
        constraints.discard("window_seat")
    constraint_patterns = {
        "direct_flights_only": r"\b(?:direct|non[- ]stop)\s+flights?\s+only\b|\bonly\s+direct\s+flights?\b",
        "no_layovers": r"\bno\s+(?:layovers?|stopovers?|connections?)\b",
        "avoid_red_eye": r"\bno\s+red[- ]eye\b|\bavoid\s+red[- ]eye\b",
        "aisle_seat": r"\baisle\s+seat\b",
        "window_seat": r"\bwindow\s+seat\b",
        "wheelchair_access": r"\bwheelchair\s+access(?:ible)?\b|\baccessibility\s+needs?\b",
        "non_smoking": r"\bnon[- ]smoking\b|\bno\s+smoking\b",
        "family_friendly": r"\bfamily[- ]friendly\b",
        "no_booking_without_confirmation": r"\b(?:don't|do not|never)\s+book\b.{0,50}\b(?:without|unless)\b.{0,30}\b(?:ask|confirm|approval)\b",
    }
    for constraint, pattern in constraint_patterns.items():
        if re.search(pattern, value, re.IGNORECASE):
            constraints.add(constraint)
    profile["constraints"] = sorted(constraints)

    if _has_any(value, (
        r"\b(?:don't|do not|never)\s+book\b.{0,60}\bwithout\s+(?:asking|my\s+)?(?:me\s+)?(?:first|confirmation|approval)\b",
        r"\bask\s+me\s+before\s+booking\b", r"\bconfirm\s+with\s+me\s+before\s+booking\b",
    )):
        profile["booking_preference"] = "explicit_confirmation_required"

    _update_service_search_preferences(profile, value)
    return profile


__all__ = ["DEFAULT_PREFERENCE_PROFILE", "is_trip_detail_message", "is_trip_summary_request", "update_preference_profile"]
