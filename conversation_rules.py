"""Deterministic conversation-level routing guards for the Travel AI app.

No LangChain/OpenAI dependency. Keep high-confidence conversational cases out
of the LLM router so short replies, budget values and farewells are not guessed.
"""
from __future__ import annotations

import re
from typing import Iterable


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("’", "'").split()).strip()


_FAREWELLS = {
    "bye", "bye bye", "goodbye", "good bye", "ok bye", "okay bye",
    "alright bye", "all right bye", "see you", "see ya", "take care",
    "that's all", "that is all", "that's all thanks", "that is all thanks",
    "thanks bye", "thank you bye", "okay goodbye", "ok goodbye",
}


def is_farewell_request(text: str) -> bool:
    value = _norm(text).strip(".! ")
    if value in _FAREWELLS:
        return True
    return bool(re.fullmatch(
        r"(?:(?:ok|okay|alright|all right|thanks|thank you)\s+)?"
        r"(?:bye(?:\s+bye)?|goodbye|good\s+bye|see\s+you|see\s+ya|take\s+care)"
        r"(?:\s+(?:for\s+now|then))?",
        value,
    ))


_HISTORY_RECALL_PATTERNS = (
    r"\bwhat\s+did\s+i\s+(?:ask|say|request)\s+(?:you\s+)?(?:before|earlier|previously|last)\b",
    r"\bwhat\s+was\s+my\s+(?:previous|last|earlier)\s+(?:question|request|message)\b",
    r"\b(?:repeat|recall|remind\s+me\s+of)\s+(?:my\s+)?(?:previous|last|earlier)\s+(?:question|request|message)\b",
    r"\bwhat\s+was\s+i\s+asking\s+(?:before|earlier)\b",
)


def is_history_recall_request(text: str) -> bool:
    value = _norm(text)
    return any(re.search(pattern, value) for pattern in _HISTORY_RECALL_PATTERNS)


_CONTEXTUAL_DETAIL_EXACT = {
    "detailed", "detail", "details", "more detail", "more details",
    "more detailed", "explain more", "tell me more", "elaborate",
    "go on", "continue", "expand", "expand on that", "explain further",
    "give more details", "provide more details",
}

_CONTEXTUAL_DETAIL_PATTERNS = (
    r"^(?:give|provide|show)\s+me\s+(?:a\s+)?more\s+details?\b",
    r"^(?:detailed|more detailed)\s+(?:info|information|explanation)\b",
    r"\b(?:about|on)\s+(?:the\s+)?(?:previous|previos|prior|last)\s+(?:question|answer|topic|response)\b",
    r"\b(?:elaborate|expand)\s+(?:on\s+)?(?:that|this|your previous answer|the previous answer)\b",
)


def is_contextual_detail_followup(text: str) -> bool:
    value = _norm(text).strip(".!? ")
    return value in _CONTEXTUAL_DETAIL_EXACT or any(
        re.search(pattern, value) for pattern in _CONTEXTUAL_DETAIL_PATTERNS
    )


# Canonical country names with common conversational aliases. This is used
# only for ambiguity detection, never to infer a route direction.
_COUNTRY_ALIASES = {
    "iran": ("iran",),
    "united states": ("united states", "united states of america", "usa", "u.s.a.", "u.s.", "america", "us"),
    "india": ("india",),
    "united kingdom": ("united kingdom", "uk", "u.k.", "britain", "great britain"),
    "uae": ("uae", "united arab emirates", "emirates"),
    "saudi arabia": ("saudi arabia", "saudi"),
    "canada": ("canada",), "australia": ("australia",),
    "new zealand": ("new zealand",), "pakistan": ("pakistan",),
    "afghanistan": ("afghanistan",), "iraq": ("iraq",),
    "israel": ("israel",), "palestine": ("palestine",),
    "lebanon": ("lebanon",), "syria": ("syria",), "turkey": ("turkey", "türkiye"),
    "qatar": ("qatar",), "kuwait": ("kuwait",), "oman": ("oman",),
    "bahrain": ("bahrain",), "yemen": ("yemen",), "jordan": ("jordan",),
    "egypt": ("egypt",), "morocco": ("morocco",), "kenya": ("kenya",),
    "south africa": ("south africa",), "nigeria": ("nigeria",),
    "china": ("china",), "japan": ("japan",), "south korea": ("south korea", "korea"),
    "north korea": ("north korea",), "singapore": ("singapore",),
    "malaysia": ("malaysia",), "indonesia": ("indonesia",),
    "thailand": ("thailand",), "vietnam": ("vietnam",),
    "philippines": ("philippines",), "sri lanka": ("sri lanka",),
    "nepal": ("nepal",), "bangladesh": ("bangladesh",),
    "myanmar": ("myanmar", "burma"), "russia": ("russia",),
    "ukraine": ("ukraine",), "france": ("france",), "germany": ("germany",),
    "italy": ("italy",), "spain": ("spain",), "portugal": ("portugal",),
    "netherlands": ("netherlands", "holland"), "belgium": ("belgium",),
    "switzerland": ("switzerland",), "sweden": ("sweden",),
    "norway": ("norway",), "denmark": ("denmark",), "finland": ("finland",),
    "poland": ("poland",), "greece": ("greece",), "ireland": ("ireland",),
    "austria": ("austria",), "czechia": ("czechia", "czech republic"),
    "romania": ("romania",), "hungary": ("hungary",), "brazil": ("brazil",),
    "argentina": ("argentina",), "mexico": ("mexico",), "cuba": ("cuba",),
    "colombia": ("colombia",), "chile": ("chile",), "peru": ("peru",),
}

_COUNTRY_LOOKUP = {
    alias: canonical
    for canonical, aliases in _COUNTRY_ALIASES.items()
    for alias in aliases
}


def country_mentions(text: str) -> set[str]:
    value = _norm(text)
    found: set[str] = set()
    # Match longest aliases first so "United States" is not split/duplicated.
    for alias in sorted(_COUNTRY_LOOKUP, key=len, reverse=True):
        pattern = rf"(?<![a-z]){re.escape(alias)}(?![a-z])"
        if re.search(pattern, value):
            found.add(_COUNTRY_LOOKUP[alias])
    return found


def country_pair_names(text: str) -> tuple[str, str] | None:
    """Return the first two distinct country names in the order the user wrote them."""
    value = _norm(text)
    occurrences: list[tuple[int, str]] = []
    for alias, canonical in _COUNTRY_LOOKUP.items():
        pattern = rf"(?<![a-z]){re.escape(alias)}(?![a-z])"
        for match in re.finditer(pattern, value):
            occurrences.append((match.start(), canonical))
    ordered: list[str] = []
    for _, canonical in sorted(occurrences, key=lambda item: item[0]):
        if canonical not in ordered:
            ordered.append(canonical)
        if len(ordered) == 2:
            return ordered[0], ordered[1]
    return None


_CONFLICT_WORDS = re.compile(
    r"\b(?:war|conflict|conflicts|crisis|tension|tensions|attack|attacks|"
    r"military|bombing|airstrike|airstrikes|missile|missiles|hostilities|"
    r"ceasefire|invasion|strike|strikes)\b",
    re.IGNORECASE,
)


def is_conflict_request(
    text: str,
    recent_user_texts: Iterable[str] = (),
    pending_kind: str | None = None,
) -> bool:
    value = _norm(text)
    conflict_or_relation = bool(_CONFLICT_WORDS.search(value)) or bool(
        re.search(r"\b(?:relationship|relations|relation|historical|history|geopolitics|political)\b", value)
    )
    # While asking the numbered clarification, option 1 means conflict/travel
    # security even though the reply itself contains only the number.
    if pending_kind == "country_pair_clarification" and value in {
        "1", "1 conflict", "conflict", "war", "historical background", "history", "relationship", "relations"
    }:
        return True
    if not conflict_or_relation:
        return False
    if len(country_mentions(value)) >= 2:
        return True
    # Only the immediately preceding user message may supply the country pair.
    # A distant destination mention must not turn an itinerary interest such as
    # "war" into a geopolitical question.
    recent_list = list(recent_user_texts)
    return bool(recent_list and len(country_mentions(recent_list[-1])) >= 2)


def is_ambiguous_country_pair(text: str) -> bool:
    """Detect two countries without a clear travel direction or stated purpose."""
    value = _norm(text)
    if len(country_mentions(value)) < 2:
        return False
    if _CONFLICT_WORDS.search(value):
        return False
    # Explicit route/direction or a clearly phrased trip plan is not ambiguous.
    directional_patterns = (
        r"\bfrom\b.{1,80}\bto\b",
        r"\b(?:travel|travelling|traveling|fly|flying|trip|flight|flights|"
        r"itinerary|holiday|vacation|visit|visiting|plan|plans|planning)\b.{0,80}\b(?:to|from|between)\b",
        r"\bbetween\b.{1,80}\band\b",
    )
    return not any(re.search(pattern, value) for pattern in directional_patterns)


def is_space_destination_request(text: str) -> bool:
    value = _norm(text)
    planet_or_space = r"\b(?:mars|moon|lunar|venus|jupiter|saturn|mercury|space)\b"
    travel_intent = r"\b(?:trip|travel|travelling|traveling|holiday|vacation|tour|itinerary|plan|planning|go|visit|journey|budget)\b"
    return bool(re.search(planet_or_space, value) and re.search(travel_intent, value))


def extract_currency_amount(text: str) -> float | None:
    """Extract an explicitly rupee-denominated amount from a user message."""
    value = _norm(text).replace(",", "")
    patterns = (
        r"₹\s*(\d+(?:\.\d+)?)",
        r"\b(?:inr|rs\.?|rupees?)\s*(\d+(?:\.\d+)?)\b",
        r"\b(\d+(?:\.\d+)?)\s*(?:inr|rs\.?|rupees?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _duration_days(texts: Iterable[str]) -> int | None:
    joined = " ".join(_norm(text) for text in texts)
    match = re.search(r"\b(\d{1,3})\s*(?:day|days)\b", joined)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def needs_budget_clarification(
    current_text: str,
    prior_user_texts: Iterable[str] = (),
    prior_assistant_texts: Iterable[str] = (),
) -> bool:
    """Catch an obviously unclear/implausibly tiny quoted budget before planning.

    The guard is deliberately narrow: it requires a rupee amount <= ₹1,000,
    a multi-day trip, and a recent assistant prompt that explicitly requested
    the travel budget. Explicit per-day amounts are left to normal reasoning.
    """
    value = _norm(current_text)
    amount = extract_currency_amount(value)
    if amount is None or amount > 1000:
        return False
    if re.search(r"\b(?:per\s+day|a\s+day|daily|each\s+day)\b", value):
        return False

    assistant_context = " ".join(_norm(text) for text in prior_assistant_texts[-4:])
    if not re.search(r"\bbudget\b", assistant_context):
        return False

    user_context = list(prior_user_texts[-8:]) + [value]
    days = _duration_days(user_context)
    if days is None or days < 2:
        return False

    return True


def is_budget_clarification_answer(text: str) -> bool:
    value = _norm(text)
    return bool(
        extract_currency_amount(value) is not None
        or re.search(r"\b(?:total|overall|per\s+day|daily|per\s+person|per\s+traveller|per\s+traveler|excluding flights|including flights|excluding airfare|including airfare|excluding accommodation|including accommodation|i meant|i mean|larger budget|more budget)\b", value)
    )


def budget_scope_kind(text: str) -> str | None:
    """Classify the scope the user assigns to a previously supplied budget."""
    value = _norm(text)
    if re.search(r"\b(?:per\s+day|daily|each\s+day|a\s+day)\b", value):
        return "per_day"
    if re.search(r"\b(?:total|overall|entire\s+trip|whole\s+trip|for\s+the\s+trip)\b", value):
        return "total"
    if re.search(r"\b(?:per\s+person|per\s+traveller|per\s+traveler|each\s+person)\b", value):
        return "per_person"
    if re.search(r"\b(?:excluding|not\s+including|without)\s+(?:the\s+)?(?:flight|flights|airfare|accommodation|hotel|hotels)\b", value):
        return "exclusions"
    return None


def is_space_hypothetical_confirmation(text: str) -> bool:
    value = _norm(text).strip(".!? ")
    return value in {"yes", "yes please", "yes absolutely", "sure", "okay", "ok", "go ahead", "please do", "hypothetical", "fictional", "fiction"}
