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


