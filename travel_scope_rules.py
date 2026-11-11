"""Deterministic travel-only scope policy for the Travel AI application.

This module has no LangChain dependency so the scope checks can be tested
without calling an LLM or any travel provider.
"""
from __future__ import annotations

import re

from intent_router import is_destination_food_request


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("â€™", "'").split()).strip()


_RECIPE_WORDS = r"(?:recipe|recipes|recipie|recipies|recipee|recipees)"

_COOKING_PATTERNS = (
    rf"\b{_RECIPE_WORDS}\b",
    r"\bhow\s+(?:do\s+i|can\s+i|to)\s+(?:make|cook|bake|prepare)\b",
    r"\b(?:show|give|write|share|create)\s+(?:me\s+)?(?:the\s+)?(?:a\s+)?recipe\b",
    r"\b(?:make|cook|bake|prepare)\b.{0,55}\b(?:pizza|chocolate|cake|cookies?|biscuits?|bread|pasta|noodles?|soup|meal|food|dessert)\b",
)


def is_general_cooking_request(text: str) -> bool:
    """True for recipe/cooking instructions, not travel food recommendations.

    Destination-specific local cuisine remains in scope (e.g. local dishes in
    Malaysia). Ready-to-eat food recommendations for a flight also remain in
    scope because they do not ask the assistant to teach cooking.
    """
    value = _norm(text)
    if not value:
        return False

    # Respect the existing destination-cuisine exception.
    if is_destination_food_request(value):
        return False

    return any(re.search(pattern, value, re.IGNORECASE) for pattern in _COOKING_PATTERNS)


_TRAVEL_IMPACT_TERMS = (
    r"\b(?:travel|travelling|traveling|trip|journey|tourist|tourism|itinerary|"
    r"flight|flights|airline|airlines|airport|airspace|route|routes|visa|visas|"
    r"border|borders|travel advisory|travel advisories|travel risk|travel safety|"
    r"safe to visit|safe to travel|entry requirements|transit|evacuation|booking)\b"
)

_GENERAL_POLITICAL_HISTORY_TERMS = (
    r"\b(?:history|historical|historical background|historical context|"
    r"relationship|relations|geopolitics|geopolitical|politics|political|"
    r"foreign policy|diplomacy|diplomatic|sanctions?|nuclear deal|"
    r"military conflict|war|conflict|conflicts|coup|revolution|hostage crisis)\b"
)


def is_nontravel_political_history_request(text: str) -> bool:
    """Reject general political/history questions unless they concern travel impact."""
    value = _norm(text)
    if not value:
        return False

    # A direct travel-impact question is allowed, even if it mentions a war,
    # conflict, sanctions or foreign-policy issue.
    if re.search(_TRAVEL_IMPACT_TERMS, value, re.IGNORECASE):
        return False

    return bool(re.search(_GENERAL_POLITICAL_HISTORY_TERMS, value, re.IGNORECASE))


def is_out_of_travel_scope(text: str) -> bool:
    return is_general_cooking_request(text) or is_nontravel_political_history_request(text)


def travel_scope_reply(text: str) -> str:
    """Return a concise deterministic scope response without invoking an LLM."""
    value = _norm(text)

    if is_general_cooking_request(value):
        return (
            "I'm a Travel AI Assistant, so I don't provide general cooking recipes or cooking instructions. "
            "I can help with travel-related food questions, such as ready-to-eat snacks for a flight, "
            "food to pack, airport meal options, airline food policies, or local cuisine at a destination."
        )

    if is_nontravel_political_history_request(value):
        return (
            "I'm a Travel AI Assistant. I can help explain how political tensions or conflicts may affect "
            "travel, including travel advisories, borders, airspace, routes and flight disruptions. "
            "I don't provide general political analysis or historical explanations. "
            "I also don't have a live news or official travel-advisory feed, so I won't guess about the current situation."
        )

    return (
        "I'm a Travel AI Assistant. I can help with flights, hotels, destinations, itineraries, "
        "travel documents, transportation, packing and travel safety. I can't assist with unrelated general topics."
    )

# Services a traveler may need at or near a destination. These are not
# booking/search integrations: the app currently has no live local-business
# locator, so the supervisor asks for a more precise area and then states
# that exact nearby availability cannot be verified.
_NEARBY_SERVICE_TERMS = (
    r"\b(?:atm|cash machine|cashpoint|bank|pharmacy|chemist|drugstore|"
    r"hospital|clinic|medical centre|medical center|urgent care|"
    r"police station|embassy|consulate|money exchange|currency exchange|"
    r"supermarket|grocery store|laundromat|laundry|bus stop|train station)\b"
)
_NEARBY_CUES = (
    r"\b(?:nearby|near me|near here|close by|closest|nearest|around here|"
    r"in the area|around the area)\b"
)
_DESTINATION_TRAVEL_CUES = (
    r"\b(?:travel(?:ling|ing)?|trip|holiday|vacation|visit(?:ing)?|"
    r"stay(?:ing)?|destination|when i arrive|when we arrive|once i arrive|"
    r"once we arrive)\b"
)


def is_destination_local_service_request(
    text: str, *, has_destination_context: bool = False
) -> bool:
    """Recognize a nearby local-service question tied to a travel destination.

    Examples:
        'I am travelling to Malaysia. Is there an ATM nearby?'
        'Are there pharmacies near me?' (when a destination is already stored)

    This only identifies intent. It does not claim that a live ATM/place search
    is available or that a particular business is open.
    """
    value = _norm(text)
    if not value:
        return False

    has_service = bool(re.search(_NEARBY_SERVICE_TERMS, value, re.IGNORECASE))
    has_nearby_cue = bool(re.search(_NEARBY_CUES, value, re.IGNORECASE))
    has_travel_context = has_destination_context or bool(
        re.search(_DESTINATION_TRAVEL_CUES, value, re.IGNORECASE)
        or re.search(
            r"\b(?:malaysia|kuala lumpur|penang|langkawi|singapore|"
            r"dubai|uae|thailand|bangkok|sri lanka|colombo|maldives|"
            r"india|kochi|cochin|calicut|kozhikode|bangalore|bengaluru|"
            r"delhi|mumbai|japan|tokyo|indonesia|bali|java|goa|"
            r"united kingdom|london|france|paris|italy|europe|usa|"
            r"united states)\b",
            value,
            re.IGNORECASE,
        )
    )
    return has_service and has_nearby_cue and has_travel_context


def canonicalize_supervisor_route(route: str) -> str:
    """Convert legacy intent labels to actual LangGraph node names.

    `general_travel` is an intent category returned by the classifier/override
    layer, not a graph node. It must resolve to `general_assistant`.
    """
    aliases = {"general_travel": "general_assistant"}
    return aliases.get(route, route)


# ---------------------------------------------------------------------------
# Destination information and conversation context
# ---------------------------------------------------------------------------
# This is a small offline destination alias catalogue, not a live geographic
# database. It covers destinations used by the app's regression tests and can
# be expanded without changing the LangGraph supervisor.
_DESTINATION_CATALOG = (
    {"name": "Yogyakarta", "aliases": ("yogyakarta", "jogja"), "country": "Indonesia", "region": "Java"},
    {"name": "Bali", "aliases": ("bali",), "country": "Indonesia", "region": "Bali"},
    {"name": "Jakarta", "aliases": ("jakarta",), "country": "Indonesia", "region": "Java"},
    {"name": "Surabaya", "aliases": ("surabaya",), "country": "Indonesia", "region": "Java"},
    {"name": "Bandung", "aliases": ("bandung",), "country": "Indonesia", "region": "Java"},
    {"name": "Java Island", "aliases": ("java island", "island of java"), "country": "Indonesia", "region": "Java"},
    {"name": "Turkey", "aliases": ("turkey", "tÃ¼rkiye", "turkiye"), "country": "Turkey", "region": None},
    {"name": "Jordan", "aliases": ("jordan",), "country": "Jordan", "region": None},
    {"name": "India", "aliases": ("india",), "country": "India", "region": None},
    {"name": "Malaysia", "aliases": ("malaysia",), "country": "Malaysia", "region": None},
    {"name": "Indonesia", "aliases": ("indonesia",), "country": "Indonesia", "region": None},
    {"name": "Kuala Lumpur", "aliases": ("kuala lumpur", "kl"), "country": "Malaysia", "region": None},
    {"name": "Penang", "aliases": ("penang",), "country": "Malaysia", "region": None},
    {"name": "Langkawi", "aliases": ("langkawi",), "country": "Malaysia", "region": None},
    {"name": "Singapore", "aliases": ("singapore",), "country": "Singapore", "region": None},
    {"name": "Dubai", "aliases": ("dubai",), "country": "United Arab Emirates", "region": None},
    {"name": "Thailand", "aliases": ("thailand",), "country": "Thailand", "region": None},
    {"name": "Bangkok", "aliases": ("bangkok",), "country": "Thailand", "region": None},
    {"name": "Japan", "aliases": ("japan",), "country": "Japan", "region": None},
    {"name": "Tokyo", "aliases": ("tokyo",), "country": "Japan", "region": None},
    {"name": "Iran", "aliases": ("iran",), "country": "Iran", "region": None},
    {"name": "United States", "aliases": ("united states", "united states of america", "usa", "u.s.a.", "america"), "country": "United States", "region": None},
    {"name": "New York", "aliases": ("new york",), "country": "United States", "region": None},
    {"name": "Atlanta", "aliases": ("atlanta",), "country": "United States", "region": "Georgia"},
    {"name": "Savannah", "aliases": ("savannah",), "country": "United States", "region": "Georgia"},
    {"name": "Istanbul", "aliases": ("istanbul",), "country": "Turkey", "region": None},
    {"name": "Cappadocia", "aliases": ("cappadocia",), "country": "Turkey", "region": None},
    {"name": "Antalya", "aliases": ("antalya",), "country": "Turkey", "region": None},
    {"name": "Tbilisi", "aliases": ("tbilisi",), "country": "Georgia", "region": None},
    {"name": "Batumi", "aliases": ("batumi",), "country": "Georgia", "region": None},
    {"name": "Petra", "aliases": ("petra",), "country": "Jordan", "region": None},
    {"name": "Amman", "aliases": ("amman",), "country": "Jordan", "region": None},
    {"name": "Wadi Rum", "aliases": ("wadi rum",), "country": "Jordan", "region": None},
    {"name": "Kochi", "aliases": ("kochi", "cochin"), "country": "India", "region": "Kerala"},
    {"name": "Kozhikode", "aliases": ("kozhikode", "calicut"), "country": "India", "region": "Kerala"},
    {"name": "Bengaluru", "aliases": ("bengaluru", "bangalore"), "country": "India", "region": "Karnataka"},
    {"name": "Delhi", "aliases": ("delhi", "new delhi"), "country": "India", "region": None},
    {"name": "Mumbai", "aliases": ("mumbai", "bombay"), "country": "India", "region": "Maharashtra"},
    {"name": "Goa", "aliases": ("goa",), "country": "India", "region": None},
    {"name": "Paris", "aliases": ("paris",), "country": "France", "region": None},
    {"name": "Nice", "aliases": ("nice",), "country": "France", "region": "French Riviera"},
)


def find_destination_mentions(text: str) -> list[dict[str, str | None]]:
    """Extract known destination names in the order they appear in a message.

    More specific place names win over parent regions/countries. The result is
    deliberately small and suitable for storing in the conversation state.
    Ambiguous names (Java and Georgia) are handled separately below.
    """
    value = _norm(text)
    if not value:
        return []

    mentions: list[tuple[int, int, dict[str, str | None]]] = []
    for item in _DESTINATION_CATALOG:
        for alias in item["aliases"]:
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            for match in re.finditer(pattern, value, re.IGNORECASE):
                mentions.append((match.start(), -(match.end() - match.start()), {
                    "name": item["name"],
                    "country": item["country"],
                    "region": item["region"],
                }))

    # Java is only recorded as a place when the wording clearly signals travel
    # or the user explicitly identifies Java Island / Indonesia.
    java_explicit = bool(re.search(r"\bjava\s+island\b|\bjava\s*,?\s*indonesia\b|\bindonesia\b.{0,35}\bjava\b", value))
    java_travel = bool(re.search(r"\b(?:go|going|travel|travelling|traveling|visit|visiting|trip|holiday|vacation)\s+(?:to|in|around)\s+java\b", value))
    if java_explicit or java_travel:
        match = re.search(r"\bjava(?:\s+island)?\b", value)
        if match:
            mentions.append((match.start(), -len(match.group(0)), {
                "name": "Java Island", "country": "Indonesia", "region": "Java"
            }))

    # Georgia needs a qualifier. Do not automatically decide whether it means
    # the country or the U.S. state.
    if re.search(r"\bgeorgia\b", value):
        if re.search(r"\bgeorgia\b.{0,25}\b(?:country|caucasus|tbilisi)\b|\b(?:country|caucasus)\b.{0,25}\bgeorgia\b", value):
            m = re.search(r"\bgeorgia\b", value)
            mentions.append((m.start(), -7, {"name": "Georgia (country)", "country": "Georgia", "region": None}))
        elif re.search(r"\bgeorgia\b.{0,35}\b(?:united states|u\.s\.|usa|us state|american state)\b|\b(?:united states|u\.s\.|usa|us state|american state)\b.{0,35}\bgeorgia\b", value):
            m = re.search(r"\bgeorgia\b", value)
            mentions.append((m.start(), -7, {"name": "Georgia (U.S. state)", "country": "United States", "region": "Georgia"}))

    # Keep the most specific match at a position and deduplicate canonical names.
    ordered: list[dict[str, str | None]] = []
    for _, _, item in sorted(mentions, key=lambda part: (part[0], part[1])):
        if item["name"] not in [entry["name"] for entry in ordered]:
            ordered.append(item)
    # If a city and its parent country were both named, keep both; that is
    # useful for storing explicit country context. No aliases are emitted twice.
    return ordered


