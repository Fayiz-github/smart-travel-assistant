import copy
import logging
import math
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any
from dotenv import load_dotenv

try:
    import serpapi
except ImportError:
    serpapi = None

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Date-only travel inputs are interpreted in the configured local timezone.
# Default is India because this project is configured for an India-based user.
APP_TIMEZONE_NAME = os.getenv("APP_TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata"
try:
    APP_TIMEZONE = ZoneInfo(APP_TIMEZONE_NAME)
except Exception:
    if APP_TIMEZONE_NAME == "Asia/Kolkata":
        APP_TIMEZONE = timezone(timedelta(hours=5, minutes=30))
    else:
        logging.getLogger(__name__).warning("Timezone %r is unavailable; falling back to UTC.", APP_TIMEZONE_NAME)
        APP_TIMEZONE = timezone.utc

logger = logging.getLogger("travel_system.hotel_agent")
from search_limits import SERPAPI_LIMITER, SearchBudgetExceeded

SUPPORTED_CURRENCIES = {
    "INR", "USD", "EUR", "GBP", "MYR", "SGD", "THB", "AED", "CAD", "AUD",
    "JPY", "CHF", "CNY", "HKD", "KRW", "IDR", "PHP", "VND", "LKR", "NPR",
    "BDT", "PKR", "SAR", "QAR", "KWD", "BHD", "OMR", "ZAR", "NZD", "TRY",
    "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "ILS", "EGP", "TWD", "BRL",
    "MXN", "RUB", "UAH", "MAD", "KES", "NGN", "GHS", "TZS", "UGX", "MUR",
}

def is_supported_currency(value: str) -> bool:
    return isinstance(value, str) and value.strip().upper() in SUPPORTED_CURRENCIES



# ---------------------------------------------------------------------------
# Config helpers — with bounds validation at startup (F-23)
# ---------------------------------------------------------------------------

def _env_int(key: str, default: int, min_val: int | None = None, max_val: int | None = None) -> int:
    try:
        val = int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        logger.warning("Config '%s' invalid; using default %d.", key, default)
        return default
    if min_val is not None and val < min_val:
        logger.warning("Config '%s'=%d below minimum %d; using minimum.", key, val, min_val)
        return min_val
    if max_val is not None and val > max_val:
        logger.warning("Config '%s'=%d above maximum %d; using maximum.", key, val, max_val)
        return max_val
    return val


def _env_float(key: str, default: float, min_val: float | None = None, max_val: float | None = None) -> float:
    try:
        val = float(os.getenv(key, str(default)))
        if not math.isfinite(val):
            raise ValueError("non-finite")
    except (ValueError, TypeError):
        logger.warning("Config '%s' invalid; using default %f.", key, default)
        return default
    if min_val is not None and val < min_val:
        logger.warning("Config '%s'=%f below minimum %f; using minimum.", key, val, min_val)
        return min_val
    if max_val is not None and val > max_val:
        logger.warning("Config '%s'=%f above maximum %f; using maximum.", key, val, max_val)
        return max_val
    return val


# --- DYNAMIC CONFIGURATION (with semantic bounds) ---
MIN_SAFE_RATING           = _env_float("MIN_SAFE_RATING", 3.5, min_val=0.0, max_val=5.0)
HIGH_RATING_THRESHOLD     = _env_float("HIGH_RATING_THRESHOLD", 4.3, min_val=MIN_SAFE_RATING, max_val=5.0)
SIGNIFICANT_SAVINGS_RATIO = _env_float("SIGNIFICANT_SAVINGS_RATIO", 0.85, min_val=0.1, max_val=0.99)
CHEAP_MEDIAN_FACTOR       = _env_float("HOTEL_CHEAP_MEDIAN_FACTOR", 0.75, min_val=0.1, max_val=0.99)
BUDGET_MEDIAN_FACTOR      = _env_float("HOTEL_BUDGET_MEDIAN_FACTOR", 1.30, min_val=1.0, max_val=10.0)
MAX_DISPLAY_PER_TIER      = _env_int("MAX_DISPLAY_PER_TIER", 3, min_val=1, max_val=20)
MAX_AMENITIES_SHOWN       = _env_int("MAX_AMENITIES_SHOWN", 6, min_val=1, max_val=30)
MAX_ADVANCE_DAYS          = _env_int("MAX_ADVANCE_DAYS", 330, min_val=1, max_val=730)
MAX_STAY_NIGHTS           = _env_int("MAX_STAY_NIGHTS", 30, min_val=1, max_val=90)
FLEXIBLE_DATE_RANGE_DAYS  = _env_int("HOTEL_FLEXIBLE_DATE_RANGE_DAYS", 0, min_val=0, max_val=14)
CACHE_TTL_SECONDS         = _env_int("HOTEL_CACHE_TTL_SECONDS", 120, min_val=15, max_val=1800)
CACHE_MAX_ENTRIES         = _env_int("CACHE_MAX_ENTRIES", 500, min_val=50, max_val=5000)


# ---------------------------------------------------------------------------
# Thread-safe Bounded TTL Cache (HT-08, FL-05)
# ---------------------------------------------------------------------------

class BoundedTTLCache:
    """Thread-safe bounded in-memory cache with TTL expiry and LRU eviction."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 900):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Any | None:
        with self._lock:
            if key not in self._cache:
                return None
            ts, val = self._cache[key]
            if time.time() - ts > self.ttl_seconds:
                del self._cache[key]
                return None
            return copy.deepcopy(val)

    def get_with_metadata(self, key: tuple) -> tuple[Any | None, float | None]:
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None, None
            ts, val = item
            age = max(0.0, time.time() - ts)
            if age > self.ttl_seconds:
                del self._cache[key]
                return None, None
            return copy.deepcopy(val), age

    def set(self, key: tuple, val: Any) -> None:
        with self._lock:
            now = time.time()
            expired = [k for k, (ts, _) in self._cache.items() if now - ts > self.ttl_seconds]
            for k in expired:
                del self._cache[k]
            while len(self._cache) >= self.max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (now, copy.deepcopy(val))

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_HOTEL_CACHE = BoundedTTLCache(max_size=CACHE_MAX_ENTRIES, ttl_seconds=CACHE_TTL_SECONDS)


# ---------------------------------------------------------------------------
# SerpApi wrapper with safety & timeout (HT-12)
# ---------------------------------------------------------------------------

def execute_serpapi_search(params: dict, timeout: int = 15) -> dict:
    """Execute one bounded SerpApi request using the official client timeout."""
    if serpapi is None or not hasattr(serpapi, "Client"):
        raise RuntimeError("Install the official 'serpapi>=1.2.0' package to run live searches.")
    api_key = (os.getenv("SECRET_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("SECRET_API_KEY environment variable is missing or empty.")
    SERPAPI_LIMITER.acquire("hotel")
    client = serpapi.Client(api_key=api_key, timeout=max(1, int(timeout)))
    try:
        result = client.search(params)
    except Exception as exc:
        if "timeout" in type(exc).__name__.lower():
            raise TimeoutError("SerpApi hotel search timed out.") from exc
        raise
    data = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError("SerpApi hotel search returned an API error.")
    return data


# ---------------------------------------------------------------------------
# Hotel detail evaluation & sanitization (HT-10, HT-11, V-04)
# ---------------------------------------------------------------------------

def _clean_str(text: Any, max_len: int = 200) -> str:
    """Bound and neutralize provider text before it is formatted for an LLM/user."""
    if not isinstance(text, str):
        return ""
    value = " ".join("".join(ch for ch in text if ch.isprintable()).split())
    # Provider strings are data, never markup or instructions. Remove tag-shaped
    # fragments and common instruction-injection lines; this is defence-in-depth.
    value = re.sub(r"<[^>]{0,300}>", " ", value)
    value = re.sub(
        r"(?i)\b(ignore (?:all )?(?:previous|prior) instructions?|system prompt|developer message|assistant message|call (?:the )?tool|execute (?:this )?instruction)\b[^.]{0,180}",
        "[untrusted provider text omitted]",
        value,
    )
    value = re.sub(r"(?i)\b(?:SYSTEM|DEVELOPER|ASSISTANT|TOOL)\s*:\s*[^.]{0,220}", "[untrusted provider text omitted]", value)
    value = value.replace("</", " ").replace("<", " ").replace(">", " ")
    value = " ".join(value.split())
    return value[:max_len]



def _safe_rating(value: Any) -> float | None:
    """Safely coerce rating to float in [0, 5], or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
        if math.isfinite(f) and 0.0 <= f <= 5.0:
            return round(f, 2)
    except (ValueError, TypeError):
        pass
    return None


def _safe_reviews(value: Any) -> int | None:
    """Safely coerce reviews to non-negative int, or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        i = int(float(value))
        return i if i >= 0 else None
    except (ValueError, TypeError):
        return None


def _safe_amenities(value: Any) -> list[str]:
    """Normalize amenities to a sanitized list of non-empty strings (V-04, HT-11)."""
    result: list[str] = []
    if isinstance(value, list):
        for a in value:
            cs = _clean_str(a, max_len=60)
            if cs:
                result.append(cs)
    elif isinstance(value, str) and value.strip():
        for s in value.split(","):
            cs = _clean_str(s, max_len=60)
            if cs:
                result.append(cs)
    return result


def evaluate_hotel_details(hotel: dict) -> list[str]:
    notes = []

    rating  = _safe_rating(hotel.get("overall_rating"))
    reviews = _safe_reviews(hotel.get("reviews", 0))
    reviews_str = f"{reviews:,}" if reviews is not None else "unknown number of"

    if rating is None:
        notes.append("⚠️ UNRATED PROPERTY: No guest rating available yet — book with caution.")
    elif rating < MIN_SAFE_RATING:
        notes.append(f"⚠️ LOW GUEST RATING: {rating}/5 ({reviews_str} reviews).")
    elif rating >= HIGH_RATING_THRESHOLD:
        notes.append(f"⭐ TOP-RATED STAY: {rating}/5 ({reviews_str} reviews).")
    else:
        notes.append(f"✅ Decent Rating: {rating}/5 ({reviews_str} reviews).")

    deal = _clean_str(hotel.get("deal"), max_len=80)
    if deal:
        notes.append(f"🏷️ SPECIAL DEAL: {deal}")

    amenities = _safe_amenities(hotel.get("amenities"))
    if amenities:
        shown = amenities[:MAX_AMENITIES_SHOWN]
        extra = f" (+{len(amenities) - len(shown)} more)" if len(amenities) > len(shown) else ""
        notes.append(f"🛎️ Amenities: {', '.join(shown)}{extra}")

    if hotel.get("food_match_note"):
        notes.append(_clean_str(hotel["food_match_note"], max_len=120))

    return notes


# ---------------------------------------------------------------------------
# Price extraction (V-05)
# ---------------------------------------------------------------------------

def _extract_price_value(rate_obj: Any, *keys: str) -> float | None:
    if not isinstance(rate_obj, dict):
        return None
    for key in keys:
        value = rate_obj.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(value) and value > 0:
            return float(value)
    return None

def extract_hotel_price_details(property_item: dict, num_nights: int = 1) -> dict:
    """Extract separate nightly, full-stay, and after-tax amounts from SerpApi."""
    nightly_obj = property_item.get("rate_per_night") or {}
    total_obj = property_item.get("total_rate") or {}
    nightly_after_tax = _extract_price_value(nightly_obj, "extracted_after_taxes_fees")
    nightly_display = _extract_price_value(nightly_obj, "extracted_display_price", "extracted_lowest")
    total_after_tax = _extract_price_value(total_obj, "extracted_after_taxes_fees")
    total_display = _extract_price_value(total_obj, "extracted_display_price", "extracted_lowest")
    nightly = nightly_after_tax or nightly_display
    if nightly is None and total_after_tax is not None:
        nightly = round(total_after_tax / max(1, num_nights), 2)
    if nightly is None and total_display is not None:
        nightly = round(total_display / max(1, num_nights), 2)
    return {
        "nightly_price": nightly,
        "nightly_after_taxes_fees": nightly_after_tax,
        "total_stay_after_taxes_fees": total_after_tax,
        "total_stay_after_taxes_fees_display": total_obj.get("after_taxes_fees") if isinstance(total_obj, dict) else None,
        "total_stay_display_price": total_display,
        "total_stay_display_text": (total_obj.get("display_price") or total_obj.get("lowest")) if isinstance(total_obj, dict) else None,
        "price_basis": (
            "total_after_taxes_fees" if total_after_tax is not None else
            "total_display_price" if total_display is not None else
            "nightly_after_taxes_fees" if nightly_after_tax is not None else
            "nightly_display_price" if nightly_display is not None else "unknown"
        ),
    }

def extract_nightly_price(property_item: dict, num_nights: int = 1) -> float | None:
    """Compatibility helper returning a nightly estimate; prefer extract_hotel_price_details."""
    return extract_hotel_price_details(property_item, num_nights).get("nightly_price")

def _safe_provider_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    from urllib.parse import urlparse
    parsed = urlparse(value.strip())
    if parsed.scheme == "https" and parsed.hostname and parsed.username is None and parsed.password is None:
        return value.strip()[:1000]
    return None


# ---------------------------------------------------------------------------
# Structured fetch result (F-02, HT-05, HT-06)
# ---------------------------------------------------------------------------

class HotelFetchResult:
    """Typed container so callers can tell success from failure from empty."""

    __slots__ = ("status", "hotels", "error_code", "error_message", "retryable")

    def __init__(
        self,
        status: str,                      # "success" | "empty" | "error"
        hotels: list[dict] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        retryable: bool = False,
    ):
        self.status        = status
        self.hotels        = hotels or []
        self.error_code    = error_code
        self.error_message = error_message
        self.retryable     = retryable

    def is_ok(self) -> bool:
        return self.status in ("success", "empty")


def _validate_hotel_request(
    location_query: str,
    check_in_date: str,
    check_out_date: str,
    adults: int,
    currency: str,
    children: int = 0,
    children_ages: list[int] | None = None,
    rooms: int = 1,
) -> str | None:
    """Direct validation function (HT-05, HT-06). Returns error string if invalid, else None."""
    if not isinstance(location_query, str) or not location_query.strip() or len(location_query.strip()) < 2:
        return "Location query cannot be blank, empty, or shorter than 2 characters."
    if len(location_query.strip()) > 100:
        return "Location query is too long (maximum 100 characters)."

    if not isinstance(adults, int) or isinstance(adults, bool) or adults < 1 or adults > 20:
        return f"Number of adults ({adults}) must be an integer between 1 and 20."
    if not isinstance(children, int) or isinstance(children, bool) or not 0 <= children <= 10:
        return "Children count must be an integer between 0 and 10."
    if not isinstance(rooms, int) or isinstance(rooms, bool) or not 1 <= rooms <= 8:
        return "Room count must be an integer between 1 and 8."
    child_ages = children_ages or []
    if len(child_ages) != children:
        return "Provide one age for every child so the hotel provider can calculate the correct occupancy rate."
    if any(not isinstance(age, int) or isinstance(age, bool) or age < 0 or age > 17 for age in child_ages):
        return "Each child age must be an integer between 0 and 17."

    try:
        in_dt = datetime.strptime(check_in_date.strip(), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return f"Invalid check_in_date format '{check_in_date}'. Expected YYYY-MM-DD."

    try:
        out_dt = datetime.strptime(check_out_date.strip(), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return f"Invalid check_out_date format '{check_out_date}'. Expected YYYY-MM-DD."

    today_local = datetime.now(APP_TIMEZONE).date()
    if in_dt < today_local:
        return f"check_in_date ({check_in_date}) cannot be in the past. Today is {today_local}."
    if out_dt <= in_dt:
        return "check_out_date must be strictly after check_in_date."
    if (out_dt - in_dt).days > MAX_STAY_NIGHTS:
        return f"Hotel stay cannot exceed {MAX_STAY_NIGHTS} nights."
    if (out_dt - today_local).days > MAX_ADVANCE_DAYS:
        return f"Booking dates cannot be more than {MAX_ADVANCE_DAYS} days in the future."

    clean_curr = currency.strip().upper() if isinstance(currency, str) else ""
    if not is_supported_currency(clean_curr):
        return f"Unsupported currency code '{currency}'. Choose INR, USD, MYR, SGD, EUR, GBP, or another supported currency."

    return None


def fetch_hotels(
    location_query: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    currency: str = "INR",
    children: int = 0,
    children_ages: list[int] | None = None,
    rooms: int = 1,
    force_refresh: bool = False,
) -> HotelFetchResult:
    """Fetch hotels with direct validation and structured outcomes (HT-05, HT-06, F-02)."""
    # HT-05, HT-06: Direct validation
    val_err = _validate_hotel_request(location_query, check_in_date, check_out_date, adults, currency, children, children_ages, rooms)
    if val_err:
        return HotelFetchResult(
            status="error",
            error_code="VALIDATION_ERROR",
            error_message=val_err,
            retryable=False,
        )

    clean_loc = location_query.strip().lower()
    clean_curr = currency.strip().upper()
    cache_key = (clean_loc, check_in_date.strip(), check_out_date.strip(), adults, children, tuple(children_ages or []), rooms, clean_curr)

    cached_hotels, cache_age = _HOTEL_CACHE.get_with_metadata(cache_key)
    if cached_hotels is not None and not force_refresh:
        for item in cached_hotels:
            item["_quote_source_status"] = "cached"
            item["_quote_cache_age_seconds"] = round(cache_age or 0.0, 1)
        return HotelFetchResult(status="success", hotels=cached_hotels)

    params: dict[str, Any] = {
        "engine": "google_hotels",
        "q": location_query.strip(),
        "check_in_date": check_in_date.strip(),
        "check_out_date": check_out_date.strip(),
        "adults": str(adults),
        "rooms": str(rooms),
        "currency": clean_curr,
    }
    if children:
        params["children"] = str(children)
        params["children_ages"] = ",".join(str(age) for age in (children_ages or []))

    try:
        results = execute_serpapi_search(params)
    except SearchBudgetExceeded as exc:
        return HotelFetchResult(
            status="error", error_code="SEARCH_BUDGET_EXCEEDED",
            error_message=str(exc), retryable=True,
        )
    except RuntimeError as exc:
        logger.error("Configuration error fetching hotels for %s (%s)", location_query, type(exc).__name__)
        return HotelFetchResult(
            status="error",
            error_code="CONFIG_ERROR",
            error_message="Hotel search configuration error. Please contact support.",
            retryable=False,
        )
    except Exception as exc:
        logger.error("Network/API error fetching hotels for %s (%s)", location_query, type(exc).__name__)
        return HotelFetchResult(
            status="error",
            error_code="NETWORK_ERROR",
            error_message="Hotel search temporarily unavailable. Please try again.",
            retryable=True,
        )

    if not isinstance(results, dict):
        logger.error("SerpApi returned non-dict payload for %s", location_query)
        return HotelFetchResult(
            status="error",
            error_code="PROVIDER_PAYLOAD_ERROR",
            error_message="Unexpected response from hotel data provider.",
            retryable=True,
        )

    if "error" in results:
        logger.error("SerpApi error for %s: %s", location_query, results.get("error"))
        return HotelFetchResult(
            status="error",
            error_code="PROVIDER_API_ERROR",
            error_message="Hotel data provider returned an error.",
            retryable=False,
        )

    try:
        in_date    = datetime.strptime(check_in_date.strip(),  "%Y-%m-%d")
        out_date   = datetime.strptime(check_out_date.strip(), "%Y-%m-%d")
        num_nights = max(1, (out_date - in_date).days)
    except ValueError:
        num_nights = 1

    raw_properties = results.get("properties")
    if not isinstance(raw_properties, list):
        raw_properties = []

    valid_hotels: list[dict] = []
    for prop in raw_properties:
        if not isinstance(prop, dict):
            logger.debug("Skipping non-dict hotel record: %r", type(prop))
            continue
        price_details = extract_hotel_price_details(prop, num_nights)
        nightly_price = price_details.get("nightly_price")
        if nightly_price is not None:
            prop_copy = copy.deepcopy(prop)
            prop_copy.update(price_details)
            prop_copy["price"] = nightly_price  # legacy nightly field used by the existing tier formatter
            prop_copy["name"] = _clean_str(prop_copy.get("name"), max_len=100)
            prop_copy["description"] = _clean_str(prop_copy.get("description"), max_len=300)
            prop_copy["deal"] = _clean_str(prop_copy.get("deal"), max_len=100)
            prop_copy["amenities"] = _safe_amenities(prop_copy.get("amenities"))
            prop_copy["provider_link"] = _safe_provider_url(prop_copy.get("link"))
            safe_prices = []
            for price_item in prop_copy.get("prices", []) if isinstance(prop_copy.get("prices"), list) else []:
                if isinstance(price_item, dict):
                    safe_prices.append({
                        "source": _clean_str(price_item.get("source"), 60),
                        "link": _safe_provider_url(price_item.get("link")),
                    })
            prop_copy["price_sources"] = safe_prices[:5]
            prop_copy["_quote_source_status"] = "fresh"
            prop_copy["_quote_retrieved_at"] = datetime.now(APP_TIMEZONE).isoformat()
            valid_hotels.append(prop_copy)

    if valid_hotels:
        _HOTEL_CACHE.set(cache_key, valid_hotels)
        return HotelFetchResult(status="success", hotels=valid_hotels)

    return HotelFetchResult(status="empty", hotels=[])


# ---------------------------------------------------------------------------
# Star-rating filter (F-08, HT-07)
# ---------------------------------------------------------------------------

def _extract_star_class(hotel: dict) -> int | None:
    star_class = hotel.get("extracted_hotel_class")
    if star_class is not None and not isinstance(star_class, bool):
        try:
            val = int(star_class)
            if 1 <= val <= 5:
                return val
        except (ValueError, TypeError):
            pass

    hc_str = str(hotel.get("hotel_class", ""))
    matches = re.findall(r"\b([1-5])\b", hc_str)
    if matches:
        return int(matches[0])

    return None


def filter_by_star_rating(
    hotels: list[dict],
    min_star: int | None,
    only_exact: bool = False,
    allow_fallback: bool = True,
) -> tuple[list[dict], bool, str | None]:
    if not min_star or not hotels:
        return hotels, False, None

    # F-08: hard constraint must never silently return non-matching properties
    if only_exact:
        allow_fallback = False

    filtered = []
    for hotel in hotels:
        star_class = _extract_star_class(hotel)
        if star_class is None:
            continue
        if only_exact and star_class == min_star:
            filtered.append(hotel)
        elif not only_exact and star_class >= min_star:
            filtered.append(hotel)

    if not filtered:
        if allow_fallback:
            warn = (
                f"⚠️ No hotels with {min_star}+ stars found — showing all available options. "
                f"Results below do NOT meet the star requirement.\n"
            )
            return hotels, False, warn
        return [], False, None

    return filtered, True, None


# ---------------------------------------------------------------------------
# Robust Dietary Filter with Negation Detection (HT-01, HT-02, HT-03, HT-04)
# ---------------------------------------------------------------------------

_STRICT_DIETARY_PREFS = {
    "halal", "vegan", "vegetarian", "jain", "kosher",
    "gluten-free", "gluten free", "nut-free", "nut free",
    "dairy-free", "dairy free", "allergy",
}

_DINING_PATTERN = re.compile(
    r"\b(restaurant|dining|breakfast|buffet|cafe|cafeteria|bistro)\b",
    re.IGNORECASE,
)


def _has_negative_dietary_statement(text: str, preference: str) -> bool:
    """Conservative detection: ambiguous or negative listing text is never positive evidence."""
    clean_text = " ".join((text or "").lower().split())
    pref = preference.lower().strip()
    terms = [pref]
    if pref in {"nut-free", "nut free"}: terms += ["nuts", "nut"]
    if pref in {"gluten-free", "gluten free"}: terms += ["gluten"]
    if pref in {"dairy-free", "dairy free"}: terms += ["dairy", "milk"]
    for term in terms:
        escaped = re.escape(term)
        if re.search(rf"\b(?:not|no|non|without|never|cannot|can't|may not|does not|doesn't|not guaranteed|not certified|not verified)\b.{{0,70}}\b{escaped}\b", clean_text):
            return True
        if re.search(rf"\b{escaped}\b.{{0,70}}\b(?:unavailable|not available|not served|not guaranteed|not certified|not verified|excluded|may contain)\b", clean_text):
            return True
    if pref == "halal" and re.search(r"\b(?:pork|bacon|alcohol)\b", clean_text) and not re.search(r"\b(?:no pork|pork-free|alcohol-free|halal certified)\b", clean_text):
        return True
    if pref in {"nut-free", "nut free"} and re.search(r"\b(?:may contain|shared kitchen|cross[- ]contamination|processed in a facility)\b.{0,80}\b(?:nut|nuts)\b", clean_text):
        return True
    return False


def filter_by_food_preference(
    hotels: list[dict],
    preference: str | None,
    allow_fallback: bool = True,
) -> tuple[list[dict], bool, str | None]:
    """Filter by explicit listing evidence without claiming certification or allergy safety."""
    if not preference or not hotels:
        return hotels, False, None
    pref_clean = " ".join(preference.lower().strip().split())
    allergy_request = any(token in pref_clean for token in ("allergy", "allergic", "anaphylaxis", "cross contamination"))
    if allergy_request:
        return [], False, (
            "No hotel can be verified as allergy-safe from scraped listing text. "
            "I will not recommend a property as safe based on keyword matches; obtain direct written confirmation from the property about ingredients and cross-contamination controls."
        )

    is_strict = any(strict in pref_clean for strict in _STRICT_DIETARY_PREFS)
    evidence_terms = {
        "gluten-free": r"gluten[- ]free", "gluten free": r"gluten[- ]free",
        "nut-free": r"nut[- ]free", "nut free": r"nut[- ]free",
        "dairy-free": r"dairy[- ]free", "dairy free": r"dairy[- ]free",
    }
    term_pattern = re.compile(
        rf"\b(?:{evidence_terms.get(pref_clean, re.escape(pref_clean))})\b",
        re.IGNORECASE,
    )
    matched: list[dict] = []
    for hotel in hotels:
        amenities = _safe_amenities(hotel.get("amenities"))
        description = _clean_str(hotel.get("description"), 300)
        evidence = " ".join(amenities + ([description] if description else []))
        if not evidence or _has_negative_dietary_statement(evidence, pref_clean):
            continue
        if term_pattern.search(evidence):
            copy_item = copy.deepcopy(hotel)
            copy_item["food_match_note"] = (
                f"Listing text mentions '{preference}'. This is not independent certification or a safety guarantee; confirm details with the property."
            )
            matched.append(copy_item)
        elif not is_strict and _DINING_PATTERN.search(evidence):
            copy_item = copy.deepcopy(hotel)
            copy_item["food_match_note"] = (
                f"On-site dining is mentioned, but '{preference}' availability is not verified."
            )
            matched.append(copy_item)

    if matched:
        return matched, True, None
    return [], False, (
        f"No listing contained clear, non-negative evidence for '{preference}'. "
        "I have not substituted unverified hotels."
    )


# ---------------------------------------------------------------------------
# Tier grouping & formatting (HT-10, HT-11, F-19)
# ---------------------------------------------------------------------------

def group_into_tiers(hotels: list[dict]) -> dict[str, list[dict]]:
    if not hotels:
        return {}

    sorted_hotels = sorted(hotels, key=lambda x: x.get("total_stay_after_taxes_fees") or x.get("total_stay_display_price") or x["price"])
    unique_prices = sorted({h.get("total_stay_after_taxes_fees") or h.get("total_stay_display_price") or h["price"] for h in sorted_hotels})

    if len(unique_prices) <= 2:
        return {"Available Hotels": sorted_hotels}

    median_price  = (sorted_hotels[len(sorted_hotels) // 2].get("total_stay_after_taxes_fees") or sorted_hotels[len(sorted_hotels) // 2].get("total_stay_display_price") or sorted_hotels[len(sorted_hotels) // 2]["price"])
    cheap_cutoff  = median_price * CHEAP_MEDIAN_FACTOR
    budget_cutoff = median_price * BUDGET_MEDIAN_FACTOR

    tiers: dict[str, list[dict]] = {"Cheap Tier": [], "Budget Tier": [], "Premium Tier": []}
    for h in sorted_hotels:
        price = h.get("total_stay_after_taxes_fees") or h.get("total_stay_display_price") or h["price"]
        if price <= cheap_cutoff:
            tiers["Cheap Tier"].append(h)
        elif price <= budget_cutoff:
            tiers["Budget Tier"].append(h)
        else:
            tiers["Premium Tier"].append(h)

    return tiers


def format_hotel_summary(hotel: dict, currency: str = "INR", num_nights: int = 1) -> str:
    name = _clean_str(hotel.get("name", "Unknown Hotel"), max_len=70)
    nightly = hotel.get("nightly_after_taxes_fees") or hotel.get("price")
    nightly_str = f"{nightly:,.2f}" if isinstance(nightly, (int, float)) and not isinstance(nightly, bool) else "N/A"
    nightly_label = ("Nightly price including listed taxes/fees" if hotel.get("nightly_after_taxes_fees") is not None
                     else "Provider nightly display price (tax treatment may vary)")
    total_after_tax = hotel.get("total_stay_after_taxes_fees")
    total_display = hotel.get("total_stay_display_price")
    if isinstance(total_after_tax, (int, float)):
        total_line = f"Total stay including listed taxes/fees: {currency} {total_after_tax:,.2f}"
    elif isinstance(total_display, (int, float)):
        total_line = f"Provider-displayed total stay: {currency} {total_display:,.2f} (tax treatment may vary by market)"
    elif hotel.get("total_stay_after_taxes_fees_display"):
        total_line = f"Provider total including taxes/fees: {_clean_str(hotel.get('total_stay_after_taxes_fees_display'), 80)}"
    elif hotel.get("total_stay_display_text"):
        total_line = f"Provider-displayed total stay: {_clean_str(hotel.get('total_stay_display_text'), 80)} (tax treatment may vary by market)"
    elif isinstance(nightly, (int, float)) and num_nights > 0:
        total_line = f"Estimated stay total: {currency} {nightly * num_nights:,.2f} ({num_nights} nights; not a checkout quote)"
    else:
        total_line = "Full-stay total unavailable in provider response."

    hotel_class = _clean_str(hotel.get("hotel_class", "Unclassified"), max_len=30)
    property_type = _clean_str(hotel.get("type", "hotel"), max_len=30).title()
    quote_status = "cached" if hotel.get("_quote_source_status") == "cached" else "fresh"
    quote_time = hotel.get("_quote_retrieved_at") or "timestamp unavailable"
    if quote_status == "cached":
        quote_note = f"Quote source: cached; cache age approximately {hotel.get('_quote_cache_age_seconds', 'unknown')} seconds."
    else:
        quote_note = f"Quote source: freshly retrieved at {quote_time}."

    links = []
    if hotel.get("provider_link"):
        links.append(f"Property link: {hotel['provider_link']}")
    for source in hotel.get("price_sources", []):
        if source.get("source") and source.get("link"):
            links.append(f"Offer: {_clean_str(source['source'], 50)} — {source['link']}")

    notes = evaluate_hotel_details(hotel)
    notes_str = "\n      ".join(notes) if notes else "No additional notes."
    link_str = "\n  ".join(links[:4]) if links else "Provider offer link not supplied in this result."
    return (
        f"• {name} ({hotel_class} {property_type})\n"
        f"  {nightly_label}: {currency} {nightly_str}\n"
        f"  {total_line}\n"
        f"  {quote_note}\n"
        f"  Booking sources: {link_str}\n"
        "  Final checkout amount must be verified on the provider page; this search result is not a reservation or guaranteed price.\n"
        f"  Hotel Notes:\n      {notes_str}"
    )


# ---------------------------------------------------------------------------
# High-level run_hotel_agent (HT-05, HT-06, HT-02, Test 5)
# ---------------------------------------------------------------------------

def run_hotel_agent(
    location_query: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    currency: str = "INR",
    min_star: int | None = None,
    only_exact_star: bool = False,
    food_preference: str | None = None,
    star_rating: list[int] | None = None,
    echo: bool = False,
    children: int = 0,
    children_ages: list[int] | None = None,
    rooms: int = 1,
    force_refresh: bool = False,
) -> str:
    # HT-07: Handle exact_star without min_star
    if min_star is None and isinstance(star_rating, list):
        valid_ratings = [value for value in star_rating if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5]
        if valid_ratings:
            min_star = valid_ratings[0]
    if min_star is None:
        only_exact_star = False

    # HT-05, HT-06: Direct validation check upfront
    val_err = _validate_hotel_request(location_query, check_in_date, check_out_date, adults, currency, children, children_ages, rooms)
    if val_err:
        report = f"⚠️ Validation Error: {val_err}"
        if echo:
            print(report)
        return report

    clean_loc = location_query.strip()
    clean_curr = currency.strip().upper()

    try:
        in_date_obj  = datetime.strptime(check_in_date.strip(),  "%Y-%m-%d").date()
        out_date_obj = datetime.strptime(check_out_date.strip(), "%Y-%m-%d").date()
        stay_duration = (out_date_obj - in_date_obj).days
    except ValueError:
        stay_duration = 1

    lines: list[str] = [
        f"🏨 Searching hotels in '{clean_loc}' ({check_in_date} to {check_out_date}) for "
        f"{adults} adult(s), {children} child(ren), {rooms} room(s)...\n"
    ]

    fetch_result = fetch_hotels(
        clean_loc, check_in_date, check_out_date, adults, clean_curr,
        children=children, children_ages=children_ages, rooms=rooms, force_refresh=force_refresh,
    )

    if fetch_result.status == "error":
        lines.append(f"⚠️ Hotel search failed: {fetch_result.error_message}")
        if fetch_result.retryable:
            lines.append("   Please try again in a moment.")
        report = "\n".join(lines)
        if echo:
            print(report)
        return report

    if fetch_result.status == "empty" or not fetch_result.hotels:
        lines.append("😔 No hotels found for your selected dates.")
        report = "\n".join(lines)
        if echo:
            print(report)
        return report

    main_hotels = fetch_result.hotels
    star_matched = False
    food_matched = False

    if min_star:
        star_allow_fb = not only_exact_star
        main_hotels, star_matched, star_warn = filter_by_star_rating(
            main_hotels, min_star, only_exact=only_exact_star, allow_fallback=star_allow_fb
        )
        if star_warn:
            lines.append(star_warn)
        if only_exact_star and not main_hotels:
            lines.append(
                f"😔 No hotels with exactly {min_star} stars found for these dates. "
                f"Try relaxing the star requirement."
            )
            report = "\n".join(lines)
            if echo:
                print(report)
            return report

    if food_preference:
        # HT-02 / Test 5: For strict dietary requirements, NEVER fall back to unverified hotels!
        is_strict_diet = any(strict in food_preference.lower() for strict in _STRICT_DIETARY_PREFS)
        diet_allow_fb = not is_strict_diet
        main_hotels, food_matched, food_warn = filter_by_food_preference(
            main_hotels, food_preference, allow_fallback=diet_allow_fb
        )
        if food_warn:
            lines.append(food_warn)
        if not food_matched and is_strict_diet:
            lines.append(
                f"😔 No verified '{food_preference}' hotels found. "
                f"Unverified hotels are not shown for strict dietary safety."
            )
            report = "\n".join(lines)
            if echo:
                print(report)
            return report

    if not main_hotels:
        lines.append("😔 No hotels matched your combined filters for these dates.")
        report = "\n".join(lines)
        if echo:
            print(report)
        return report

    today_date = datetime.now(APP_TIMEZONE).date()
    max_date   = today_date + timedelta(days=MAX_ADVANCE_DAYS)
    min_main_price = min((h.get("total_stay_after_taxes_fees") or h.get("total_stay_display_price") or h["price"]) for h in main_hotels)

    best_alternative = None

    for offset in range(-FLEXIBLE_DATE_RANGE_DAYS, FLEXIBLE_DATE_RANGE_DAYS + 1):
        if offset == 0:
            continue
        alt_in  = in_date_obj + timedelta(days=offset)
        alt_out = alt_in + timedelta(days=stay_duration)

        if alt_in < today_date or alt_out > max_date:
            continue

        alt_in_str  = alt_in.strftime("%Y-%m-%d")
        alt_out_str = alt_out.strftime("%Y-%m-%d")

        alt_result = fetch_hotels(clean_loc, alt_in_str, alt_out_str, adults, clean_curr, children=children, children_ages=children_ages, rooms=rooms)
        if alt_result.status != "success" or not alt_result.hotels:
            continue

        alt_hotels = alt_result.hotels
        if star_matched:
            alt_hotels, _, _ = filter_by_star_rating(
                alt_hotels, min_star, only_exact=only_exact_star, allow_fallback=False
            )
        if food_matched:
            alt_hotels, _, _ = filter_by_food_preference(alt_hotels, food_preference, allow_fallback=False)

        if alt_hotels:
            alt_min_price = min((h.get("total_stay_after_taxes_fees") or h.get("total_stay_display_price") or h["price"]) for h in alt_hotels)
            if alt_min_price <= min_main_price * SIGNIFICANT_SAVINGS_RATIO:
                if not best_alternative or alt_min_price < best_alternative[3]:
                    best_alternative = (offset, alt_in_str, alt_out_str, alt_min_price)

    if best_alternative:
        offset, alt_in_str, alt_out_str, alt_min_price = best_alternative
        direction = f"{abs(offset)} day(s) earlier" if offset < 0 else f"{abs(offset)} day(s) later"
        savings   = min_main_price - alt_min_price
        lines.append(
            f"💡 FLEXIBLE DATE TIP (indicative — ±{FLEXIBLE_DATE_RANGE_DAYS} days range): "
            f"Shifting your stay by {direction} ({alt_in_str} to {alt_out_str}) starts from "
            f"{clean_curr} {alt_min_price:,.0f} total stay "
            f"(potential saving of {clean_curr} {savings:,.0f} vs your chosen dates). "
            f"Actual availability and pricing may differ — verify before booking.\n"
        )

    price_tiers = group_into_tiers(main_hotels)

    for tier_name, hotel_list in price_tiers.items():
        total_in_tier = len(hotel_list)
        shown_count   = min(MAX_DISPLAY_PER_TIER, total_in_tier)

        lines.append(
            f"=== {tier_name.upper()} (Showing {shown_count} of {total_in_tier} options) "
            f"[tiers are relative to this search's results, not market-wide] ==="
        )
        if not hotel_list:
            lines.append("  No hotels available in this price range.\n")
            continue

        for hotel in hotel_list[:MAX_DISPLAY_PER_TIER]:
            lines.append(format_hotel_summary(hotel, clean_curr, num_nights=stay_duration))
            lines.append("-" * 55)
        lines.append("")

    report = "\n".join(lines).strip()
    if echo:
        print(report)
    return report
