"""Deterministic safety and intent overrides for the Travel AI assistant.

This module deliberately has no LangChain/OpenAI dependencies.  It handles
high-confidence cases before any model call and is safe to unit-test offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("’", "'").split()).strip()


@dataclass(frozen=True)
class RouteOverride:
    route: str
    reason: str


_SECRET_PATTERNS = (
    r"\b(?:show|display|reveal|print|dump|read|give|tell)\b.{0,50}\b(?:api\s*keys?|secret\s*keys?|passwords?|credentials?|access\s*tokens?|refresh\s*tokens?|\.env|environment\s+secrets?)\b",
    r"\b(?:what\s+is|where\s+is|extract|expose|leak)\b.{0,50}\b(?:api\s*keys?|passwords?|credentials?|access\s*tokens?|secret\s*keys?)\b",
    r"\b(?:dump|print|cat|read)\s+(?:the\s+)?\.env\b",
)

_DANGEROUS_CHEMICAL_PATTERNS = (
    r"\b(?:make|create|synthesize|produce|manufacture|mix|generate)\b.{0,80}\b(?:mustard\s+gas|nerve\s+agent|sarin|vx\s+gas|chlorine\s+gas|phosgene|ricin|botulinum\s+toxin)\b",
    r"\bhow\s+to\s+(?:make|create|synthesize|produce)\b.{0,80}\b(?:toxic\s+gas|chemical\s+weapon|poison\s+gas)\b",
)

_CONCEALMENT_PATTERNS = (
    r"\b(?:hide|conceal|smuggle|sneak|illegally\s+transport)\b.{0,80}\b(?:person|people|child|animal|horse|dog|cat|weapon|explosive|contraband)\b",
    r"\b(?:person|people|child|animal|horse|dog|cat|weapon|explosive|contraband)\b.{0,80}\b(?:hide|conceal|smuggle|sneak)\b",
    r"\b(?:show|give|create|build|explain)\b.{0,50}\b(?:backdoor|security\s+bypass|authentication\s+bypass|unauthorized\s+access)\b",
)

_EXTRA_PASSENGER_PATTERNS = (
    r"\b(?:carry|take|bring|add)\b.{0,60}\b(?:extra|additional|more)\s+(?:people|passengers|persons?)\b.{0,80}\b(?:pay\s+less|without\s+(?:a\s+)?ticket|avoid\s+paying|for\s+free|one\s+ticket)\b",
    r"\b(?:extra|additional)\s+(?:people|passengers|persons?)\b.{0,80}\b(?:pay\s+less|without\s+(?:a\s+)?ticket|avoid\s+paying|for\s+free)\b",
)

_BAGGAGE_OVER_LIMIT_PATTERNS = (
    r"\b(?:overweight|over\s+the\s+limit|exceed(?:ing)?\s+the\s+limit|extra\s+weight|over\s+limit)\b.{0,50}\b(?:luggage|baggage|suitcase|bag)\b",
    r"\bhow\s+(?:can|do)\s+i\s+(?:carry|take|bring)\b.{0,80}\b(?:more|extra|50\s*kg|overweight)\b.{0,40}\b(?:luggage|baggage|weight|bags?)\b",
    r"\b(?:carry|take|bring)\b.{0,60}\b(?:luggage|baggage|suitcase|bag)\b.{0,60}\b(?:over|above|beyond)\s+(?:the\s+)?(?:flight\s+|weight\s+)?(?:weight\s+)?(?:limit|allowance)\b",
    r"\b(?:luggage|baggage|suitcase|bag)\b.{0,60}\b(?:over|above|beyond)\s+(?:the\s+)?(?:flight\s+|weight\s+)?(?:limit|allowance)\b",
)

_HOT_TUB_CLARIFICATION = (
    r"\b(?:don't|do\s+not|dont)\s+want\s+(?:any\s+)?hotels?\b.{0,60}\b(?:want|need|looking\s+for)\b.{0,30}\b(?:hot\s+tubs?|jacuzzis?|whirlpools?)\b"
)

_DESTINATION_FOOD_PATTERNS = (
    r"\b(?:local|traditional|regional|authentic|famous|popular|native)\b.{0,80}\b(?:food|foods|cuisine|recipe|recipes|dish|dishes|cookies?|snacks?)\b",
    r"\b(?:food|cuisine|recipe|recipes|dish|dishes|cookies?|snacks?)\b.{0,80}\b(?:in|of|from|there|around)\s+[a-z][a-z .'-]{1,35}\b",
)

_PASSPORT_EMERGENCY = (
    r"\b(?:lost|stolen|missing|can't\s+find|cannot\s+find)\b.{0,35}\bpassport\b",
    r"\bpassport\b.{0,35}\b(?:lost|stolen|missing)\b",
)

_LEGAL_CURRENT_INFO = (
    r"\b(?:is|are|was|were)\b.{0,70}\b(?:invasive\s+species|wildlife|animal|plant|species)\b.{0,70}\b(?:legal|allowed|permitted|prohibited|illegal)\b",
    r"\b(?:legal|law|regulation|prohibited|permitted|import\s+rules?)\b.{0,100}\b(?:malaysia|malaysian|invasive\s+species|wildlife|animal|plant|pet|dog|cat|bird|fish|horse)\b",
    r"\b(?:bring|carry|import|take)\b.{0,70}\b(?:animal|plant|species|wildlife|pet|dog|cat|bird|fish|horse|livestock)\b.{0,70}\b(?:malaysia|country|border|customs)\b",
)

_UNSUPPORTED_INTERESTS = (
    r"\b(?:anime|manga|k[- ]?pop|celebrity\s+gossip|homework|school\s+assignment)\b",
    r"\b(?:give|write|share|make)\s+me\s+(?:a\s+)?recipe\b",
)


def is_secret_exfiltration_request(text: str) -> bool:
    value = _norm(text)
    return bool(value) and any(re.search(p, value, re.IGNORECASE) for p in _SECRET_PATTERNS)


def is_safety_refusal_request(text: str) -> bool:
    value = _norm(text)
    return bool(value) and any(
        re.search(p, value, re.IGNORECASE)
        for p in (*_DANGEROUS_CHEMICAL_PATTERNS, *_CONCEALMENT_PATTERNS)
    )


def is_emergency_request(text: str) -> bool:
    value = _norm(text)
    return bool(value) and any(re.search(p, value, re.IGNORECASE) for p in _PASSPORT_EMERGENCY)


def is_destination_food_request(text: str) -> bool:
    value = _norm(text)
    has_food = bool(re.search(r"\b(?:food|foods|cuisine|recipe|recipes|dish|dishes|cookies?|snacks?)\b", value))
    has_local_modifier = bool(re.search(r"\b(?:local|traditional|regional|authentic|famous|popular|native)\b", value))
    has_place_reference = bool(re.search(
        r"\b(?:in|of|from|there|around|to)\s+[a-z][a-z .'-]{1,35}\b", value
    ) or re.search(
        r"\b(?:malaysia|malaysian|singapore|indonesia|indonesian|kerala|india|japan|japanese|thailand|thai|dubai|uae|sri lanka|maldives|bali|java)\b",
        value,
    ))
    return has_food and has_place_reference and (has_local_modifier or any(
        re.search(p, value, re.IGNORECASE) for p in _DESTINATION_FOOD_PATTERNS
    ))


def detect_route_override(text: str) -> RouteOverride | None:
    """Return an unambiguous route that must run before ordinary travel routing."""
    value = _norm(text)
    if not value:
        return None

    if is_secret_exfiltration_request(value):
        return RouteOverride("secret_refusal", "request to reveal or extract credentials/secrets")

    if is_safety_refusal_request(value):
        return RouteOverride("safety_refusal", "request to bypass security or create a dangerous substance or conceal/smuggle a person or item")

    if is_emergency_request(value):
        return RouteOverride("emergency_assistance", "lost, stolen or missing passport")

    if any(re.search(p, value, re.IGNORECASE) for p in _LEGAL_CURRENT_INFO):
        return RouteOverride("legal_info", "current legal/import restriction requires official verification")

    if any(re.search(p, value, re.IGNORECASE) for p in _EXTRA_PASSENGER_PATTERNS):
        return RouteOverride("travel_policy", "request to evade passenger/ticket requirements")

    if any(re.search(p, value, re.IGNORECASE) for p in _BAGGAGE_OVER_LIMIT_PATTERNS):
        return RouteOverride("travel_policy", "question about baggage above allowance")

    if re.search(_HOT_TUB_CLARIFICATION, value, re.IGNORECASE):
        return RouteOverride("hotel_amenity_clarify", "ambiguous request distinguishing hotels from hot-tub facilities")

    if is_destination_food_request(value):
        return RouteOverride("general_travel", "destination-specific food or cuisine question")

    if any(re.search(p, value, re.IGNORECASE) for p in _UNSUPPORTED_INTERESTS):
        return RouteOverride("unsupported_query", "unrelated entertainment, homework or general recipe request")

    return None


def is_result_follow_up(text: str) -> bool:
    """Conservative phrase patterns for clear follow-ups to prior search results."""
    value = _norm(text)
    if not value:
        return False
    exact = {
        "more", "more details", "tell me more", "show more", "details", "compare them",
        "which is cheapest", "which one is cheapest", "which is best", "what about this one",
        "the first one", "the second one", "the last one", "that one", "this one",
        "what is the address", "exact address", "give me the address", "what time is it",
    }
    if value in exact:
        return True
    patterns = (
        r"\b(?:which|what|tell me about|details of|more about)\b.{0,65}\b(?:one|option|hotel|flight|fare|price|address|room|property|airline)\b",
        r"\b(?:cheapest|cheaper|best|lowest|highest|shortest|fastest)\b.{0,55}\b(?:one|option|flight|hotel|fare|price|date)\b",
        r"\b(?:address|check[- ]?in|check[- ]?out|room type|sea[- ]view|hot\s+tub|jacuzzi|amenities|baggage|layover|departure time|arrival time)\b",
        r"\b(?:compare|recommend|choose|select)\b.{0,60}\b(?:these|those|them|options|results|one)\b",
    )
    return any(re.search(p, value, re.IGNORECASE) for p in patterns)
