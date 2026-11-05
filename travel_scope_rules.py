"""Deterministic travel-only scope policy for the Travel AI application.

This module has no LangChain dependency so the scope checks can be tested
without calling an LLM or any travel provider.
"""
from __future__ import annotations

import re

from intent_router import is_destination_food_request


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("’", "'").split()).strip()


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
