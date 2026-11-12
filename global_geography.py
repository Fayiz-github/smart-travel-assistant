"""Offline global country and airport resolution helpers.

Uses offline airport, city-gazetteer and ISO country/currency data when installed.
No network requests are made by this module. City-to-airport matching is exact;
ambiguous cities are never silently guessed, and a city without an IATA airport
is never assigned an invented airport code.
"""
from __future__ import annotations

from functools import lru_cache
import re
import unicodedata
from typing import Any


class AmbiguousLocationError(ValueError):
    """Raised when a place name maps to more than one country."""


def normalize_geo_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


@lru_cache(maxsize=1)
def _load_iata_airports() -> dict[str, dict[str, Any]]:
    try:
        import airportsdata
        data = airportsdata.load("IATA")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_geonames_cities() -> dict[str, dict[str, Any]]:
    """Load a broad offline city gazetteer; airport data remains the fallback."""
    try:
        import geonamescache
        cache = geonamescache.GeonamesCache(min_city_population=5000)
        data = cache.get_cities()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_geonames_countries() -> dict[str, dict[str, Any]]:
    try:
        import geonamescache
        cache = geonamescache.GeonamesCache(min_city_population=5000)
        data = cache.get_countries()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_pycountry():
    try:
        import pycountry
        return pycountry
    except Exception:
        return None


def country_name_from_code(code: str) -> str:
    code = str(code or "").strip().upper()
    pycountry = _load_pycountry()
    if pycountry is not None:
        country = pycountry.countries.get(alpha_2=code)
        if country:
            return str(getattr(country, "common_name", None) or country.name)
    return code


def _country_alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    pycountry = _load_pycountry()
    if pycountry is not None:
        for country in pycountry.countries:
            canonical = str(getattr(country, "common_name", None) or country.name)
            for field in ("name", "official_name", "common_name"):
                value = getattr(country, field, None)
                if value:
                    aliases[normalize_geo_text(value)] = canonical
    manual = {
        "usa": "United States",
        "u s a": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u k": "United Kingdom",
        "britain": "United Kingdom",
        "great britain": "United Kingdom",
        "uae": "United Arab Emirates",
        "emirates": "United Arab Emirates",
        "south korea": "South Korea",
        "republic of korea": "South Korea",
        "north korea": "North Korea",
        "dpr korea": "North Korea",
        "russia": "Russia",
        "turkey": "Türkiye",
        "türkiye": "Türkiye",
        "czech republic": "Czechia",
        "holland": "Netherlands",
        "ivory coast": "Côte d'Ivoire",
        "dr congo": "Democratic Republic of the Congo",
        "democratic republic of congo": "Democratic Republic of the Congo",
        "congo kinshasa": "Democratic Republic of the Congo",
        "republic of the congo": "Republic of the Congo",
        "congo brazzaville": "Republic of the Congo",
        "myanmar": "Myanmar",
        "burma": "Myanmar",
        "cape verde": "Cabo Verde",
        "swaziland": "Eswatini",
        "macedonia": "North Macedonia",
        "east timor": "Timor-Leste",
        "vatican": "Holy See",
        "palestine": "Palestine",
        "taiwan": "Taiwan",
        "kosovo": "Kosovo",
    }
    aliases.update({normalize_geo_text(k): v for k, v in manual.items()})
    return aliases


def country_aliases() -> dict[str, str]:
    """Return normalized country aliases -> canonical display name."""
    return dict(_country_alias_map())


def canonical_country(value: str) -> str | None:
    return _country_alias_map().get(normalize_geo_text(value))


def supported_currency_codes() -> set[str]:
    pycountry = _load_pycountry()
    if pycountry is None:
        return set()
    codes: set[str] = set()
    for currency in pycountry.currencies:
        code = getattr(currency, "alpha_3", None)
        if not code:
            continue
        # Exclude withdrawn historical currencies when pycountry marks them.
        if getattr(currency, "withdrawal_date", None):
            continue
        codes.add(str(code).upper())
    return codes


def is_valid_iata_code(code: str) -> bool | None:
    """True/False with airportsdata installed; None when the dataset is unavailable."""
    clean = str(code or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", clean):
        return False
    airports = _load_iata_airports()
    if not airports:
        return None
    return clean in airports


def airport_candidates(location: str, country: str | None = None) -> list[dict[str, str]]:
    """Find exact city/airport-name matches; return all airports in that city."""
    query = normalize_geo_text(location)
    if not query:
        return []
    airports = _load_iata_airports()
    if not airports:
        return []

    country_key = normalize_geo_text(country or "")
    canonical = canonical_country(country or "") if country else None
    allowed_country_codes: set[str] = set()
    if country:
        pycountry = _load_pycountry()
        if pycountry is not None:
            for c in pycountry.countries:
                c_name = normalize_geo_text(getattr(c, "common_name", None) or c.name)
                if query == normalize_geo_text(country) or country_key == c_name or (canonical and canonical == (getattr(c, "common_name", None) or c.name)):
                    allowed_country_codes.add(str(c.alpha_2).upper())
            country_obj = pycountry.countries.get(alpha_2=str(country).upper())
            if country_obj:
                allowed_country_codes.add(str(country_obj.alpha_2).upper())
        if len(str(country).strip()) == 2:
            allowed_country_codes.add(str(country).strip().upper())

    matches: list[dict[str, str]] = []
    for code, airport in airports.items():
        if not isinstance(airport, dict):
            continue
        airport_city = normalize_geo_text(airport.get("city", ""))
        airport_name = normalize_geo_text(airport.get("name", ""))
        if query not in {airport_city, airport_name, normalize_geo_text(code)}:
            continue
        country_code = str(airport.get("country") or "").upper()
        if allowed_country_codes and country_code not in allowed_country_codes:
            continue
        matches.append({
            "iata": str(airport.get("iata") or code).upper(),
            "name": str(airport.get("name") or ""),
            "city": str(airport.get("city") or ""),
            "country": country_name_from_code(country_code),
            "country_code": country_code,
        })
    # Stable deterministic order and unique IATA codes.
    unique: dict[str, dict[str, str]] = {}
    for item in sorted(matches, key=lambda x: (x["city"].casefold(), x["iata"])):
        unique.setdefault(item["iata"], item)
    return list(unique.values())


def resolve_iata_location(location: str, country: str | None = None) -> str | None:
    """Resolve an IATA code or unambiguous city/airport name to one or more IATA codes.

    Multiple airports in the same city/country are returned comma-separated,
    which Google Flights/SerpApi supports. Ambiguous same-name cities raise.
    """
    clean = str(location or "").strip()
    if re.fullmatch(r"[A-Za-z]{3}", clean):
        valid = is_valid_iata_code(clean)
        if valid is True:
            return clean.upper()
        if valid is None:
            return None

    # Allow "City, Country", "City, State, Country", and "City Country".
    if not country:
        parts = [part.strip() for part in clean.split(",") if part.strip()]
        if len(parts) >= 2:
            right = parts[-1]
            is_iso_country = False
            pycountry = _load_pycountry()
            if pycountry is not None and len(right) == 2:
                is_iso_country = pycountry.countries.get(alpha_2=right.upper()) is not None
            if canonical_country(right) or is_iso_country:
                country = right
                clean = parts[0]
        if not country:
            words = clean.split()
            for cut in range(1, len(words)):
                possible_country = " ".join(words[cut:])
                if canonical_country(possible_country):
                    clean = " ".join(words[:cut])
                    country = possible_country
                    break

    matches = airport_candidates(clean, country)
    if not matches:
        return None
    countries = sorted({item["country"] for item in matches if item["country"]})
    if not country and len(countries) > 1:
        raise AmbiguousLocationError(
            f"{clean!r} matches airports in multiple countries: {', '.join(countries[:8])}. "
            "Please specify the country."
        )
    return ",".join(item["iata"] for item in matches)


@lru_cache(maxsize=1)
def _city_index() -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}

    def add_city(name: str, country_code: str, region: str | None = None) -> None:
        city = str(name or "").strip()
        key = normalize_geo_text(city)
        if len(key) < 3:
            return
        item = {
            "name": city,
            "country": country_name_from_code(country_code),
            "region": str(region or "") or None,
        }
        bucket = index.setdefault(key, [])
        if not any(x["country"] == item["country"] for x in bucket):
            bucket.append(item)

    # Airport city names support flight routing and destination research.
    for airport in _load_iata_airports().values():
        if isinstance(airport, dict):
            add_city(airport.get("city", ""), str(airport.get("country") or ""), airport.get("subd"))

    # GeoNames extends destination recognition to cities without an IATA airport.
    # Alternate names help with common local/English spellings; airport resolution
    # remains a separate operation and never invents an IATA code.
    for city in _load_geonames_cities().values():
        if not isinstance(city, dict):
            continue
        name = str(city.get("name") or "").strip()
        country_code = str(city.get("countrycode") or "").upper()
        # GeoNames admin1 codes are not necessarily human-readable subdivision
        # names; omit them rather than presenting raw numeric codes to users.
        add_city(name, country_code, None)
        aliases = city.get("alternatenames") or []
        if isinstance(aliases, list):
            for alias in aliases[:8]:
                alias = str(alias or "").strip()
                if alias and normalize_geo_text(alias) != normalize_geo_text(name):
                    add_city(alias, country_code, None)
    return index


def global_destination_airport_codes(limit: int = 15) -> list[str]:
    """Return a geographically diverse set of major global destination airports.

    This is a default exploration universe, not a popularity ranking. It uses
    city population and continent metadata where available, chooses one airport
    per city, and falls back to a globally distributed airport list.
    """
    limit = max(1, min(int(limit), 25))
    fallback = [
        "LHR", "CDG", "FRA", "IST", "DXB", "SIN", "HND", "SYD",
        "JFK", "LAX", "GRU", "JNB", "NBO", "DOH", "MEX",
    ]
    airports = _load_iata_airports()
    cities = _load_geonames_cities()
    countries = _load_geonames_countries()
    if not airports or not cities:
        return fallback[:limit]

    city_population: dict[tuple[str, str], int] = {}
    for city in cities.values():
        if not isinstance(city, dict):
            continue
        key = (normalize_geo_text(city.get("name", "")), str(city.get("countrycode") or "").upper())
        try:
            population = int(city.get("population") or 0)
        except (TypeError, ValueError):
            population = 0
        if key[0] and key[1]:
            city_population[key] = max(city_population.get(key, 0), population)

    if not city_population:
        return fallback[:limit]

    # One best airport per city/country. Airport name is only a tie-breaker;
    # population drives the global candidate ranking.
    city_airports: dict[tuple[str, str], tuple[int, int, str]] = {}
    for code, airport in airports.items():
        if not isinstance(airport, dict):
            continue
        city = normalize_geo_text(airport.get("city", ""))
        country_code = str(airport.get("country") or "").upper()
        if not city or not country_code:
            continue
        population = city_population.get((city, country_code), 0)
        airport_name = str(airport.get("name") or "")
        airport_rank = 1 if "international" in airport_name.casefold() else 0
        key = (city, country_code)
        candidate = (population, airport_rank, str(airport.get("iata") or code).upper())
        previous = city_airports.get(key)
        if previous is None or (candidate[0], candidate[1], candidate[2]) > (
            previous[0], previous[1], previous[2]
        ):
            city_airports[key] = candidate

    by_continent: dict[str, list[tuple[int, str]]] = {}
    all_cities: list[tuple[int, str]] = []
    for (city, country_code), (population, _airport_rank, iata) in city_airports.items():
        country_data = countries.get(country_code) or {}
        continent = str(country_data.get("continentcode") or "XX").upper()
        by_continent.setdefault(continent, []).append((population, iata))
        all_cities.append((population, iata))

    selected: list[str] = []
    for continent in ("AF", "AS", "EU", "NA", "SA", "OC"):
        regional = sorted(by_continent.get(continent, []), reverse=True)
        for _population, code in regional[:2]:
            if code not in selected:
                selected.append(code)
            if len(selected) >= limit:
                return selected[:limit]

    for _population, code in sorted(all_cities, reverse=True):
        if code not in selected:
            selected.append(code)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for code in fallback:
            if code in airports and code not in selected:
                selected.append(code)
            if len(selected) >= limit:
                break
    return selected[:limit] or fallback[:limit]


@lru_cache(maxsize=1)
def _max_city_phrase_words() -> int:
    index = _city_index()
    return min(5, max((len(key.split()) for key in index), default=1))


def find_city_mentions(text: str) -> list[tuple[int, dict[str, Any]]]:
    """Find globally indexed cities in travel-framed text or qualified locations.

    Ambiguous same-name cities are returned with candidate countries so the
    caller can ask the traveler to disambiguate instead of guessing.
    """
    if not _load_iata_airports() and not _load_geonames_cities():
        return []
    normalized = normalize_geo_text(text)
    if not normalized:
        return []
    explicit_countries = set()
    for alias, canonical in _country_alias_map().items():
        if alias and re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", normalized):
            explicit_countries.add(canonical)
    # The abbreviation "US" is accepted as a country only when it is a
    # location suffix/qualifier, never as an ordinary pronoun.
    if re.search(r",\s*(?:us|u\.s\.|usa|u\.s\.a\.)\s*$", str(text or ""), re.IGNORECASE):
        explicit_countries.add(country_name_from_code("US"))

    travel_frame = re.search(
        r"\b(?:travel|travelling|traveling|trip|holiday|vacation|visit|visiting|"
        r"tourist|tourism|destination|flight|flights|airport|airline|safe|safety|"
        r"advisory|disruption|disruptions|news|conditions|about|where|in|at|to|from|near)\b",
        normalized,
    )
    if not travel_frame and not explicit_countries:
        return []

    tokens = list(re.finditer(r"[a-z0-9]+", normalized))
    index = _city_index()
    # GeoNames alternate names and airport-city fields can overlap with
    # ordinary English words (for example, "The" or "Tell"). Never treat
    # conversational stopwords as destination names in free-form questions.
    non_place_words = {
        "a", "an", "the", "tell", "ask", "me", "about", "please", "what",
        "where", "when", "why", "how", "is", "are", "was", "were",
        "do", "does", "did", "can", "could", "would", "should",
        "i", "we", "you", "they", "he", "she", "it", "my", "our",
        "your", "their", "us", "to", "from", "in", "on", "at",
        "for", "and", "or", "but", "with", "by", "as",
        "airline", "delay", "flight", "flights", "travel", "travelling",
        "traveling", "trip", "tourist", "tourists", "safe", "safety",
    }
    max_words = _max_city_phrase_words()

    found: list[tuple[int, dict[str, Any]]] = []
    seen: set[tuple[int, str]] = set()
    for start in range(len(tokens)):
        for size in range(max_words, 0, -1):
            end = start + size
            if end > len(tokens):
                continue
            words = [token.group(0) for token in tokens[start:end]]
            # Ignore a candidate if any component is a high-frequency prose
            # word; real multi-word place names remain matchable otherwise.
            if any(word in non_place_words for word in words):
                continue
            phrase = " ".join(words)
            candidates = index.get(phrase)
            if not candidates:
                continue
            if explicit_countries:
                filtered = [c for c in candidates if c["country"] in explicit_countries]
                if filtered:
                    candidates = filtered
            countries = sorted({c["country"] for c in candidates})
            if len(countries) == 1:
                item = dict(candidates[0])
            else:
                item = {
                    "name": candidates[0]["name"],
                    "country": None,
                    "region": None,
                    "ambiguous_countries": countries[:8],
                }
            key = (tokens[start].start(), str(item["name"]))
            if key not in seen:
                found.append((tokens[start].start(), item))
                seen.add(key)
            break
    return found
