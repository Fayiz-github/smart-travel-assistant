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


def destination_disambiguation_kind(text: str) -> str | None:
    """Return an ambiguity key for destinations with common non-place meanings."""
    value = _norm(text)
    travel_framing = bool(re.search(
        r"\b(?:go|going|travel|travelling|traveling|visit|visiting|trip|holiday|vacation|destination|tourist|tourism)\b",
        value,
    )) or bool(re.search(r"\b(?:tell me about|what about|places to visit|best time to visit)\b", value))
    if not travel_framing:
        return None

    if re.search(r"\bjava\b", value) and not re.search(r"\bjava\s+island\b|\bjava\s*,?\s*indonesia\b|\bindonesia\b.{0,35}\bjava\b", value):
        return "java"

    if re.search(r"\bgeorgia\b", value) and not re.search(
        r"\bgeorgia\b.{0,35}\b(?:country|caucasus|tbilisi|united states|u\.s\.|usa|us state|american state)\b|"
        r"\b(?:country|caucasus|united states|u\.s\.|usa|us state|american state)\b.{0,35}\bgeorgia\b",
        value,
    ):
        return "georgia"
    return None


def destination_disambiguation_question(kind: str) -> str:
    if kind == "java":
        return (
            "Do you mean **Java Island in Indonesia**? If so, I can help with destinations such as "
            "Yogyakarta, Jakarta, Surabaya or Bandung. I want to confirm the place before planning routes."
        )
    if kind == "georgia":
        return (
            "Do you mean **Georgia the country** in the Caucasus, or **Georgia the U.S. state**? "
            "I'll use the correct destination once you clarify."
        )
    return "Which destination do you mean? Please specify the country or city."


def resolve_destination_disambiguation(text: str, kind: str) -> dict[str, str | None] | None:
    """Resolve a reply to a pending Java/Georgia clarification."""
    value = _norm(text)
    if kind == "java" and re.search(r"\bjava\b", value) and re.search(r"\b(?:island|indonesia)\b", value):
        return {"name": "Java Island", "country": "Indonesia", "region": "Java"}
    if kind == "georgia":
        if re.search(r"\b(?:country|caucasus|tbilisi)\b", value):
            return {"name": "Georgia (country)", "country": "Georgia", "region": None}
        if re.search(r"\b(?:united states|u\.s\.|usa|us state|american state|state)\b", value):
            return {"name": "Georgia (U.S. state)", "country": "United States", "region": "Georgia"}
    return None



def is_ambiguous_location_sequence(text: str) -> bool:
    """Detect comma-separated locations whose roles are not specified.

    Example: "Kochi, Bangalore, Malaysia." could mean two alternative origins,
    a multi-city route, or three stops. It should be clarified rather than
    assigning roles based on list order.
    """
    value = _norm(text)
    mentions = find_destination_mentions(value)
    if len(mentions) < 3:
        return False
    if len(re.findall(r",", value)) < 2:
        return False
    explicit_route_markers = (
        r"\b(?:from|to|via|through|between|then|after|before|stopover|and back|round[- ]trip)\b",
    )
    if any(re.search(pattern, value, re.IGNORECASE) for pattern in explicit_route_markers):
        return False
    return True


def is_via_stopover_request(text: str) -> bool:
    """Detect a destination plus a via-point whose stay/transfer status is unclear."""
    value = _norm(text)
    return bool(
        re.search(r"\bvia\b", value)
        and len(find_destination_mentions(value)) >= 2
        and re.search(r"\b(?:go|going|travel|travelling|traveling|fly|flying|visit|visiting|trip|plan|planning)\b", value)
    )


def via_route_destinations(text: str) -> list[str]:
    """Return destination mentions in order for a via-route clarification."""
    return [str(item["name"]) for item in find_destination_mentions(text)]

def is_destination_information_request(
    text: str,
    *,
    has_destination_context: bool = False,
    destination_count: int = 0,
) -> bool:
    """Recognize destination discovery, overview and contextual follow-up turns."""
    value = _norm(text)
    if not value:
        return False

    mentions = find_destination_mentions(value)
    explicit_info_patterns = (
        r"\b(?:tell me about|information about|info about|overview of|describe)\b",
        r"\b(?:as|for)\s+(?:a\s+)?(?:tourist|travel)\s+destination\b",
        r"\b(?:best places|places|things|attractions)\s+to\s+visit\b",
        r"\b(?:is|are)\s+[a-z][a-z .'-]{1,35}\s+(?:worth visiting|good for tourists|a good destination)\b",
        r"\bcompare\b.{0,90}\b(?:and|with|versus|vs\.?)\b",
        r"\b(?:what about|how about)\s+[a-z][a-z .'-]{1,35}\??$",
        r"\b(?:best time|when is the best time)\s+to visit\b",
    )
    if mentions and any(re.search(pattern, value, re.IGNORECASE) for pattern in explicit_info_patterns):
        return True
    if mentions and re.search(
        r"\b(?:is|are)\b.{0,45}\b(?:safe|safety|expensive|cheap|worth it|good for solo|good for tourists)\b",
        value,
        re.IGNORECASE,
    ):
        return True

    if mentions and has_destination_context and re.match(r"^(?:i\s+)?(?:want|would like)\s+to\s+(?:visit|explore)\b", value):
        # A specific city on a previously selected island/region is a destination
        # refinement, not a new service-selection menu. A first-time trip request
        # still follows the normal trip-onboarding path.
        return True

    if has_destination_context or destination_count > 0:
        contextual_patterns = (
            r"^(?:is it|are they|is that|how much|how expensive|is .* expensive|which one|which is|what about|how about|compare|"
            r"what are|where are|best time|when is|is .* safe|are .* safe|for solo|solo tourists|solo travelers?|"
            r"i want to visit both|i'd like to visit both|i would like to visit both|i don't want an itinerary|i do not want an itinerary|"
            r"no itinerary)\b",
        )
        if any(re.search(pattern, value, re.IGNORECASE) for pattern in contextual_patterns):
            return True

    # A destination name on its own is accepted only when an earlier assistant
    # question/context already established that the user is discussing a place.
    return bool(mentions and has_destination_context and len(value.split()) <= 5)


def update_destination_context(text: str, context: dict) -> dict:
    """Update destination memory without carrying unrelated old destinations.

    Explicitly named destination sets in the current message replace the old
    set. A single additive follow-up such as "What about Bali?" appends to the
    existing set. A fresh request such as "I want to visit Goa" replaces it.
    """
    updated = dict(context or {})
    current = dict(updated.get("destination_context") or {})
    existing = list(current.get("destinations") or updated.get("trip_destinations") or [])
    mentions = find_destination_mentions(text)
    names = [str(item["name"]) for item in mentions]

    if names:
        value = _norm(text)
        explicit_comparison = bool(re.search(r"\b(?:compare|versus|vs\.?)\b", value))
        additive_single_destination = bool(
            len(names) == 1
            and existing
            and re.search(r"\b(?:what about|how about|also|add|include)\b", value)
        )

        # Two or more place names explicitly stated together form the new
        # destination set. Never append stale destinations from earlier turns.
        if len(names) >= 2:
            destinations = names[:]
        elif additive_single_destination:
            destinations = existing[:]
            if names[0] not in destinations:
                destinations.append(names[0])
        else:
            # A direct new destination request replaces the previous selection.
            destinations = names[:]

        metadata_by_name = {str(item["name"]): item for item in mentions}
        # Preserve metadata for a previously selected destination only when
        # the current turn explicitly adds that destination to a comparison.
        if additive_single_destination or (explicit_comparison and len(names) == 1):
            for old_item in current.get("destination_metadata", []):
                if old_item.get("name") in destinations:
                    metadata_by_name.setdefault(old_item["name"], old_item)

        parent_country_names = {
            "Indonesia", "United States", "Turkey", "Jordan", "India", "Malaysia",
            "Singapore", "Thailand", "Japan", "Iran", "France", "United Arab Emirates",
        }
        destinations = [
            name for name in destinations
            if not (
                name in parent_country_names
                and any(
                    metadata_by_name.get(other, {}).get("country") == name
                    for other in destinations if other != name
                )
            )
        ]
        specific_names = destinations or names
        active_name = next((name for name in reversed(names) if name in specific_names), specific_names[-1])
        active_meta = metadata_by_name.get(active_name, {})
        updated["destination_context"] = {
            "destinations": destinations,
            "active_destination": active_name,
            "country": active_meta.get("country"),
            "region": active_meta.get("region"),
            "destination_metadata": [metadata_by_name[name] for name in destinations if name in metadata_by_name],
        }
        updated["trip_destinations"] = destinations
        updated["active_destination"] = active_name
        updated["destination"] = active_name if len(destinations) == 1 else " and ".join(destinations)

    if re.search(r"\b(?:i want to visit both|i'd like to visit both|i would like to visit both)\b", _norm(text)) and len(existing) >= 2 and not names:
        updated["trip_destinations"] = existing
        updated["multi_destination_trip"] = True
        if not updated.get("destination_context"):
            updated["destination_context"] = {"destinations": existing, "active_destination": existing[-1]}

    if len(names) >= 2 and re.search(r"\b(?:both|visit|travel|go|trip)\b", _norm(text)):
        updated["multi_destination_trip"] = True
    if re.search(r"\b(?:i don't want an itinerary|i do not want an itinerary|no itinerary)\b", _norm(text)):
        updated["itinerary_preference"] = "not_requested"

    return updated

