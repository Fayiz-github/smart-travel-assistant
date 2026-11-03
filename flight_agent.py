import copy
import logging
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

logger = logging.getLogger("travel_system.flight_agent")
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
MIN_SAFE_LAYOVER_MINS     = _env_int("MIN_SAFE_LAYOVER_MINS", 60, min_val=0, max_val=300)
MAX_COMFORT_LAYOVER_MINS  = _env_int("MAX_COMFORT_LAYOVER_MINS", 360, min_val=MIN_SAFE_LAYOVER_MINS + 1)
SIGNIFICANT_SAVINGS_RATIO = _env_float("SIGNIFICANT_SAVINGS_RATIO", 0.85, min_val=0.1, max_val=0.99)
CHEAP_TIER_MULTIPLIER     = _env_float("FLIGHT_CHEAP_MULTIPLIER", 1.20, min_val=1.01, max_val=5.0)
BUDGET_TIER_MULTIPLIER    = _env_float("FLIGHT_BUDGET_MULTIPLIER", 1.40, min_val=CHEAP_TIER_MULTIPLIER, max_val=10.0)
MAX_DISPLAY_PER_TIER      = _env_int("MAX_DISPLAY_PER_TIER", 3, min_val=1, max_val=20)
MAX_ADVANCE_DAYS          = _env_int("MAX_ADVANCE_DAYS", 330, min_val=1, max_val=730)
MAX_EXPLORE_AIRPORTS      = _env_int("MAX_EXPLORE_AIRPORTS", 5, min_val=1, max_val=25)
FLEXIBLE_DATE_RANGE_DAYS  = _env_int("FLEXIBLE_DATE_RANGE_DAYS", 0, min_val=0, max_val=14)
CACHE_TTL_SECONDS         = _env_int("FLIGHT_CACHE_TTL_SECONDS", 180, min_val=15, max_val=3600)
CACHE_MAX_ENTRIES         = _env_int("CACHE_MAX_ENTRIES", 500, min_val=50, max_val=5000)

# Transit visa lookup — non-exhaustive; always advise users to verify (F-18)
TRANSIT_VISA_COUNTRIES = {
    "DXB": "UAE (Dubai)", "AUH": "UAE (Abu Dhabi)",
    "LHR": "United Kingdom", "LGW": "United Kingdom",
    "JFK": "United States", "EWR": "United States", "SFO": "United States",
    "YYZ": "Canada", "YVR": "Canada",
    "SYD": "Australia", "MEL": "Australia",
}

# ---------------------------------------------------------------------------
# Authoritative IATA code reference data (FL-01, FL-12)
# ---------------------------------------------------------------------------
VALID_IATA_CODES = {
    # India (Major & Regional)
    "DEL", "BOM", "BLR", "MAA", "CCJ", "COK", "TRV", "CNN", "HYD", "CCU",
    "GOI", "GOX", "AMD", "PNQ", "JAI", "LKO", "GAU", "IXC", "SXR", "ATQ",
    "BBI", "IXB", "PAT", "VNS", "NAG", "IDR", "VTZ", "CJB", "IXE", "TRZ",
    "IXM", "UDR", "BDQ", "RPR", "DED", "IXR", "IXJ", "IXA", "IXL", "IMF",
    "HBX", "IXU", "TIR", "STV", "BUP", "GWL", "BHO", "JGA", "IXZ", "AGT",
    # Middle East
    "DXB", "AUH", "SHJ", "DWC", "DOH", "MCT", "RUH", "JED", "DMM", "MED",
    "KWI", "BAH", "AMM", "BEY", "TLV", "BGW", "EBL", "MHD", "IKA",
    # Asia & Pacific
    "SIN", "KUL", "BKK", "DMK", "HKT", "CNX", "HKG", "NRT", "HND", "KIX",
    "FUK", "NGO", "CTS", "ICN", "GMP", "PUS", "TPE", "KHH", "MNL", "CEB",
    "CGK", "DPS", "SUB", "HAN", "SGN", "DAD", "RGN", "PNH", "REP", "VTE",
    "SYD", "MEL", "BNE", "PER", "ADL", "AKL", "CHC", "WLG", "POM", "NAN",
    "PEK", "PKX", "PVG", "SHA", "CAN", "SZX", "CTU", "XIY", "KMG", "WUH",
    "CMB", "MLE", "KTM", "DAC", "ISB", "KHI", "LHE", "TAS", "ALA",
    # Europe & UK
    "LHR", "LGW", "STN", "LTN", "MAN", "EDI", "BHX", "GLA", "BFS", "DUB",
    "CDG", "ORY", "NCE", "LYS", "MRS", "AMS", "FRA", "MUC", "BER", "HAM",
    "DUS", "STR", "MAD", "BCN", "AGP", "PMI", "VLC", "FCO", "MXP", "VCE",
    "BLQ", "NAP", "ZRH", "GVA", "VIE", "BRU", "CPH", "ARN", "OSL", "HEL",
    "LIS", "OPO", "ATH", "SKG", "IST", "SAW", "AYT", "WAW", "KRK", "PRG",
    "BUD", "OTP", "SOF", "BEG", "ZAG", "SVO", "DME", "LED", "KBP",
    # North America
    "JFK", "EWR", "LGA", "BOS", "IAD", "DCA", "BWI", "ATL", "MIA", "MCO",
    "TPA", "FLL", "ORD", "MDW", "DFW", "IAH", "DEN", "MSP", "DTW", "PHL",
    "CLT", "PHX", "LAS", "LAX", "SFO", "SAN", "SEA", "PDX", "SLC", "SMF",
    "AUS", "SAT", "BNA", "RDU", "CLE", "PIT", "IND", "CVG", "MKE", "HNL",
    "ANC", "YYZ", "YVR", "YUL", "YYC", "YEG", "YOW", "YHZ", "YWG", "MEX",
    "CUN", "GDL", "MTY", "TIJ", "SJD", "PVR",
    # South & Central America, Caribbean
    "GRU", "GIG", "BSB", "EZE", "AEP", "SCL", "BOG", "MDE", "LIM", "UIO",
    "PTY", "SJO", "GUA", "SAL", "SDQ", "PUJ", "SJU", "KIN", "HAV", "NAS",
    # Africa
    "JNB", "CPT", "DUR", "CAI", "HRG", "SSH", "CMN", "RAK", "NBO", "MBA",
    "ADD", "LOS", "ABV", "ACC", "DAR", "ZNZ", "EBB", "KGL", "MRU", "SEZ",
}


def is_valid_code(code: str) -> bool:
    """Validate 3-letter IATA code against maintained real-world airport directory."""
    if not isinstance(code, str):
        return False
    clean = code.strip().upper()
    if len(clean) != 3 or not clean.isascii() or not clean.isalpha():
        return False
    return clean in VALID_IATA_CODES


def _safe_provider_text(value: Any, max_len: int = 100) -> str:
    """Neutralize markup/control strings in untrusted provider text."""
    if not isinstance(value, str):
        return "N/A"
    clean = " ".join("".join(ch for ch in value if ch.isprintable()).split())
    clean = re.sub(r"<[^>]{0,300}>", " ", clean)
    clean = re.sub(
        r"(?i)\b(ignore (?:all )?(?:previous|prior) instructions?|system prompt|developer message|assistant message|call (?:the )?tool|execute (?:this )?instruction)\b[^.]{0,180}",
        "[untrusted provider text omitted]", clean,
    )
    clean = re.sub(r"(?i)\b(?:SYSTEM|DEVELOPER|ASSISTANT|TOOL)\s*:", "provider text:", clean)
    return " ".join(clean.replace("<", " ").replace(">", " ").split())[:max_len] or "N/A"


# ---------------------------------------------------------------------------
# Thread-safe Bounded TTL Cache (FL-05, FL-07)
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


_FLIGHT_CACHE = BoundedTTLCache(max_size=CACHE_MAX_ENTRIES, ttl_seconds=CACHE_TTL_SECONDS)


# ---------------------------------------------------------------------------
# SerpApi wrapper with timeout & error safety (FL-08)
# ---------------------------------------------------------------------------

def execute_serpapi_search(params: dict, timeout: int = 15) -> dict:
    """Execute one bounded SerpApi request using the official client timeout."""
    if serpapi is None or not hasattr(serpapi, "Client"):
        raise RuntimeError("Install the official 'serpapi>=1.2.0' package to run live searches.")
    api_key = (os.getenv("SECRET_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("SECRET_API_KEY environment variable is missing or empty.")
    SERPAPI_LIMITER.acquire("flight")
    client = serpapi.Client(api_key=api_key, timeout=max(1, int(timeout)))
    try:
        result = client.search(params)
    except Exception as exc:
        error_name = type(exc).__name__.lower()
        if "timeout" in error_name:
            raise TimeoutError("SerpApi flight search timed out.") from exc
        raise
    data = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    if isinstance(data, dict) and data.get("error"):
        # Do not copy arbitrary provider text into logs or model context.
        raise RuntimeError("SerpApi flight search returned an API error.")
    return data


# ---------------------------------------------------------------------------
# Layover evaluation (FL-03: guard layover id and duration types)
# ---------------------------------------------------------------------------

def evaluate_layovers(itinerary: dict) -> list[str]:
    """Describe connections without inventing a direct flight or zero-minute layover."""
    warnings: list[str] = []
    layovers = itinerary.get("layovers")
    flights = itinerary.get("flights")
    flights = flights if isinstance(flights, list) else []
    layovers = layovers if isinstance(layovers, list) else []

    if not layovers:
        if len(flights) > 1:
            warnings.append(
                f"Connection detected ({len(flights)} flight segments); layover details were not supplied by the provider."
            )
            warnings.append(
                "Verify the connection airport, transfer type, and minimum connection time before booking."
            )
            return warnings
        if len(flights) == 1:
            return ["Provider reports one flight segment; no connection is shown in the itinerary."]
        return ["Flight segment data is unavailable; direct/connection status cannot be determined."]

    for layover in layovers:
        if not isinstance(layover, dict):
            warnings.append("A connection is listed, but its layover details are incomplete.")
            continue
        raw_duration = layover.get("duration")
        valid_duration = (
            isinstance(raw_duration, (int, float))
            and not isinstance(raw_duration, bool)
            and math.isfinite(raw_duration)
            and raw_duration >= 0
        )
        raw_code = layover.get("id")
        airport_code = _safe_provider_text(raw_code, 5).upper() if isinstance(raw_code, str) else "N/A"
        if airport_code == "N/A" or not re.fullmatch(r"[A-Z0-9]{3,5}", airport_code):
            airport_code = "N/A"
        raw_name = layover.get("name")
        airport_name = _safe_provider_text(raw_name, 60) if isinstance(raw_name, str) else "Unknown Airport"
        if airport_name == "N/A":
            airport_name = "Unknown Airport"

        if not valid_duration:
            warnings.append(f"Connection at {airport_name} ({airport_code}); layover duration unavailable from provider.")
        else:
            duration = int(raw_duration)
            hours, mins = divmod(duration, 60)
            time_str = f"{hours}h {mins}m" if hours else f"{mins}m"
            if duration < MIN_SAFE_LAYOVER_MINS:
                warnings.append(f"TIGHT CONNECTION at {airport_name} ({airport_code}): {time_str} to change planes.")
            elif duration > MAX_COMFORT_LAYOVER_MINS:
                warnings.append(f"LONG LAYOVER at {airport_name} ({airport_code}): {time_str} wait.")
            else:
                warnings.append(f"Layover at {airport_name} ({airport_code}): {time_str}.")

        if bool(layover.get("overnight")):
            warnings.append(f"Overnight layover at {airport_code}; confirm whether an overnight transit visa or hotel is needed.")
        if airport_code in TRANSIT_VISA_COUNTRIES:
            warnings.append(
                f"Transit-visa reminder: connection in {TRANSIT_VISA_COUNTRIES[airport_code]} ({airport_code}); this list is not exhaustive. Verify official entry/transit rules."
            )

    if len(flights) > 1 and not any("Transit-visa reminder" in w for w in warnings):
        warnings.append("Connection note: verify transit visa, baggage transfer, and self-transfer requirements.")
    return warnings


# ---------------------------------------------------------------------------
# Structured fetch result (F-02, FL-04)
# ---------------------------------------------------------------------------

class FlightFetchResult:
    """Typed container so callers can tell success from failure from empty."""

    __slots__ = ("status", "flights", "error_code", "error_message", "retryable")

    def __init__(
        self,
        status: str,                    # "success" | "empty" | "error"
        flights: list[dict] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        retryable: bool = False,
    ):
        self.status        = status
        self.flights       = flights or []
        self.error_code    = error_code
        self.error_message = error_message
        self.retryable     = retryable

    def is_ok(self) -> bool:
        return self.status in ("success", "empty")


def _validate_flight_request(
    dep_id: str,
    arr_id: str,
    date_str: str,
    currency: str,
    trip_type: str,
    stops: str,
    return_date: str | None,
    adults: int = 1,
    children: int = 0,
    travel_class: str = "1",
) -> str | None:
    """Direct validation function (FL-04). Returns error message if invalid, else None."""
    if not is_valid_code(dep_id):
        return f"Invalid departure airport code '{dep_id}'. Must be a recognized 3-letter IATA code."
    if not is_valid_code(arr_id):
        return f"Invalid arrival airport code '{arr_id}'. Must be a recognized 3-letter IATA code."
    if dep_id.strip().upper() == arr_id.strip().upper():
        return "Departure and arrival airport codes cannot be identical."

    if not isinstance(adults, int) or isinstance(adults, bool) or adults < 1 or adults > 9:
        return f"Number of adults ({adults}) must be an integer between 1 and 9."
    if not isinstance(children, int) or isinstance(children, bool) or children < 0 or children > 9:
        return f"Number of children ({children}) must be an integer between 0 and 9."
    if adults + children > 9:
        return f"Total passengers (adults + children = {adults + children}) cannot exceed 9."

    try:
        out_dt = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return f"Invalid target_date format '{date_str}'. Expected YYYY-MM-DD."

    today_utc = datetime.now(APP_TIMEZONE).date()
    if out_dt < today_utc:
        return f"Outbound travel date ({date_str}) cannot be in the past. Today is {today_utc}."
    if (out_dt - today_utc).days > MAX_ADVANCE_DAYS:
        return f"Outbound travel date cannot be more than {MAX_ADVANCE_DAYS} days in the future."

    if trip_type not in ("1", "2"):
        return f"Invalid trip_type '{trip_type}'. Must be '1' (Round Trip) or '2' (One-Way)."

    if trip_type == "1":
        if not return_date or not return_date.strip():
            return "return_date is required when trip_type is '1' (Round Trip)."
        try:
            ret_dt = datetime.strptime(return_date.strip(), "%Y-%m-%d").date()
        except ValueError:
            return f"Invalid return_date format '{return_date}'. Expected YYYY-MM-DD."
        if ret_dt <= out_dt:
            return "return_date must be strictly after outbound travel date."
        if (ret_dt - today_utc).days > MAX_ADVANCE_DAYS:
            return f"return_date cannot be more than {MAX_ADVANCE_DAYS} days in the future."

    clean_curr = currency.strip().upper() if isinstance(currency, str) else ""
    if not is_supported_currency(clean_curr):
        return f"Unsupported currency code '{currency}'. Choose a supported currency such as INR, USD, MYR, SGD, EUR, or GBP."

    if stops not in ("0", "1", "2"):
        return f"Invalid stops value '{stops}'. Must be '0', '1', or '2'."

    if str(travel_class) not in ("1", "2", "3", "4"):
        return f"Invalid travel_class '{travel_class}'. Use '1' Economy, '2' Premium economy, '3' Business, or '4' First."

    return None


def fetch_flights(
    dep_id: str,
    arr_id: str,
    date_str: str,
    currency: str = "INR",
    trip_type: str = "2",
    stops: str = "0",
    return_date: str | None = None,
    adults: int = 1,
    children: int = 0,
    force_refresh: bool = False,
    travel_class: str = "1",
) -> FlightFetchResult:
    """Fetch flights with direct input validation and structured outcomes (FL-04, F-02)."""
    val_err = _validate_flight_request(dep_id, arr_id, date_str, currency, trip_type, stops, return_date, adults, children, travel_class)
    if val_err:
        return FlightFetchResult(
            status="error",
            error_code="VALIDATION_ERROR",
            error_message=val_err,
            retryable=False,
        )

    norm_dep = dep_id.strip().upper()
    norm_arr = arr_id.strip().upper()
    norm_curr = currency.strip().upper()
    cache_key = (
        norm_dep, norm_arr, date_str.strip(), norm_curr,
        trip_type, stops, return_date.strip() if return_date else None,
        adults, children, str(travel_class),
    )

    cached_flights, cache_age = _FLIGHT_CACHE.get_with_metadata(cache_key)
    if cached_flights is not None and not force_refresh:
        for item in cached_flights:
            item["_quote_source_status"] = "cached"
            item["_quote_cache_age_seconds"] = round(cache_age or 0.0, 1)
        return FlightFetchResult(status="success", flights=cached_flights)

    params: dict[str, Any] = {
        "engine": "google_flights",
        "departure_id": norm_dep,
        "arrival_id": norm_arr,
        "currency": norm_curr,
        "gl": "in",
        "hl": "en",
        "no_cache": True,  # Bypasses backend caching to fetch live real-time pricing
        "type": trip_type,
        "stops": stops,
        "outbound_date": date_str.strip(),
        "adults": str(adults),
        "travel_class": str(travel_class),
    }
    if children > 0:
        params["children"] = str(children)
    if trip_type == "1" and return_date:
        params["return_date"] = return_date.strip()

    try:
        results = execute_serpapi_search(params)
    except SearchBudgetExceeded as exc:
        return FlightFetchResult(
            status="error", error_code="SEARCH_BUDGET_EXCEEDED",
            error_message=str(exc), retryable=True,
        )
    except RuntimeError as exc:
        logger.error("Configuration error fetching flights %s→%s on %s (%s)", norm_dep, norm_arr, date_str, type(exc).__name__)
        return FlightFetchResult(
            status="error",
            error_code="CONFIG_ERROR",
            error_message="Flight search configuration error. Please contact support.",
            retryable=False,
        )
    except Exception as exc:
        logger.error("Network/API error fetching flights %s→%s on %s (%s)", norm_dep, norm_arr, date_str, type(exc).__name__)
        return FlightFetchResult(
            status="error",
            error_code="NETWORK_ERROR",
            error_message="Flight search temporarily unavailable. Please try again.",
            retryable=True,
        )

    if not isinstance(results, dict):
        logger.error("SerpApi returned non-dict payload for %s→%s: %r", norm_dep, norm_arr, type(results))
        return FlightFetchResult(
            status="error",
            error_code="PROVIDER_PAYLOAD_ERROR",
            error_message="Unexpected response from flight data provider.",
            retryable=True,
        )

    if "error" in results:
        logger.error("SerpApi error for %s→%s: %s", norm_dep, norm_arr, results.get("error"))
        return FlightFetchResult(
            status="error",
            error_code="PROVIDER_API_ERROR",
            error_message="Flight data provider returned an error.",
            retryable=False,
        )

    best   = results.get("best_flights")
    other  = results.get("other_flights")
    best   = best  if isinstance(best,  list) else []
    other  = other if isinstance(other, list) else []
    all_flights_raw = best + other

    valid_flights: list[dict] = []
    for f in all_flights_raw:
        if not isinstance(f, dict):
            logger.debug("Skipping non-dict flight record: %r", type(f))
            continue
        price = f.get("price")
        if (
            isinstance(price, (int, float))
            and not isinstance(price, bool)
            and math.isfinite(price)
            and price > 0
        ):
            valid_flights.append(copy.deepcopy(f))

    # STRICTLY SORT BY PRICE SO THE CHEAPEST IS ALWAYS FIRST
    valid_flights.sort(key=lambda x: x["price"])

    if valid_flights:
        retrieved_at = datetime.now(APP_TIMEZONE).isoformat()
        for item in valid_flights:
            item["_quote_source_status"] = "fresh"
            item["_quote_retrieved_at"] = retrieved_at
        _FLIGHT_CACHE.set(cache_key, valid_flights)
        return FlightFetchResult(status="success", flights=valid_flights)

    return FlightFetchResult(status="empty", flights=[])


# ---------------------------------------------------------------------------
# Tier grouping & formatting (FL-02, FL-15, V-03)
# ---------------------------------------------------------------------------

def group_into_tiers(flights: list[dict]) -> dict[str, list[dict]]:
    if not flights:
        return {}

    sorted_flights = sorted(flights, key=lambda x: x["price"])
    unique_prices  = sorted({f["price"] for f in sorted_flights})

    if len(unique_prices) <= 2:
        return {"Available Flights": sorted_flights}

    min_price     = sorted_flights[0]["price"]
    cheap_cutoff  = min_price * CHEAP_TIER_MULTIPLIER
    budget_cutoff = min_price * BUDGET_TIER_MULTIPLIER

    tiers: dict[str, list[dict]] = {"Cheap Tier": [], "Budget Tier": [], "Premium Tier": []}
    for f in sorted_flights:
        price = f["price"]
        if price <= cheap_cutoff:
            tiers["Cheap Tier"].append(f)
        elif price <= budget_cutoff:
            tiers["Budget Tier"].append(f)
        else:
            tiers["Premium Tier"].append(f)

    return tiers


def format_flight_summary(
    itinerary: dict,
    currency: str = "INR",
    total_passengers: int = 1,
    leg_note: str = "One-Way Leg Fare",
) -> str:
    """Format flight summary, handling malformed provider structures safely (FL-02, FL-15)."""
    price     = itinerary.get("price", 0)
    price_str = f"{price:,.0f}" if isinstance(price, (int, float)) and not isinstance(price, bool) else "N/A"
    pax_label = f" ({leg_note} for {total_passengers} passengers)" if total_passengers > 1 else f" ({leg_note} for 1 passenger)"

    raw_dur = itinerary.get("total_duration")
    if isinstance(raw_dur, (int, float)) and not isinstance(raw_dur, bool) and math.isfinite(raw_dur):
        hours, mins = divmod(int(raw_dur), 60)
        dur_str = f"{hours}h {mins}m"
    else:
        dur_str = "N/A"

    connecting_flights = itinerary.get("flights") or []
    if not isinstance(connecting_flights, list) or not connecting_flights:
        return f"Price: {currency} {price_str}{pax_label} | Total Time: {dur_str}"

    first_flight = connecting_flights[0] if isinstance(connecting_flights[0], dict) else {}
    last_flight  = connecting_flights[-1] if isinstance(connecting_flights[-1], dict) else {}

    airline_names = set()
    for f in connecting_flights:
        if isinstance(f, dict):
            al = f.get("airline")
            if isinstance(al, str) and al.strip():
                clean_al = _safe_provider_text(al, 50)
                if clean_al:
                    airline_names.add(clean_al)
            elif isinstance(al, (list, tuple)):
                for item in al:
                    if isinstance(item, str) and item.strip():
                        clean_item = _safe_provider_text(item, 50)
                        if clean_item:
                            airline_names.add(clean_item)
    airlines = ", ".join(sorted(airline_names)) if airline_names else "Unknown Airline"

    dep_airport = first_flight.get("departure_airport") or {}
    arr_airport = last_flight.get("arrival_airport") or {}
    dep_time = _safe_provider_text(dep_airport.get("time"), 60) if isinstance(dep_airport, dict) else "N/A"
    arr_time = _safe_provider_text(arr_airport.get("time"), 60) if isinstance(arr_airport, dict) else "N/A"

    layover_notes = evaluate_layovers(itinerary)
    notes_str     = "\n      ".join(layover_notes)
    quote_status = itinerary.get("_quote_source_status")
    quote_note = "Price source: cached result" if quote_status == "cached" else "Price source: freshly retrieved"
    if itinerary.get("_quote_retrieved_at"):
        quote_note += f" at {itinerary['_quote_retrieved_at']}"

    return (
        f"• {airlines} | {currency} {price_str}{pax_label} | Total Time: {dur_str}\n"
        f"  Departs: {dep_time} → Arrives: {arr_time}\n"
        f"  Layover Notes:\n      {notes_str}\n"
        f"  {quote_note}"
    )


# ---------------------------------------------------------------------------
# Multi-date comparison used by the LangGraph supervisor tool.
# ---------------------------------------------------------------------------

def compare_flight_dates(
    dep_id: str,
    arr_id: str,
    target_dates: list[str],
    currency: str = "INR",
    trip_type: str = "2",
    stops: str = "0",
    return_date: str | None = None,
    adults: int = 1,
    children: int = 0,
    max_options_per_date: int = 3,
    force_refresh: bool = False,
    travel_class: str = "1",
) -> str:
    """Compare multiple outbound dates using provider results, with deterministic sorting.

    For a round trip this intentionally prices the outbound and return legs as
    separate one-way searches. The combined amount is therefore an indicative
    sum of independently cheapest legs, not a guaranteed single-ticket fare.
    """
    if not isinstance(target_dates, list):
        return "Validation error: target_dates must be a list of YYYY-MM-DD dates."

    dates: list[str] = []
    for value in target_dates:
        if not isinstance(value, str):
            return "Validation error: every target date must be a string in YYYY-MM-DD format."
        clean = value.strip()
        if clean and clean not in dates:
            dates.append(clean)

    if not 2 <= len(dates) <= 7:
        return "Please compare between 2 and 7 distinct outbound dates."

    if max_options_per_date < 1 or max_options_per_date > 10:
        max_options_per_date = 3

    errors: list[str] = []
    for date_value in dates:
        validation = _validate_flight_request(
            dep_id, arr_id, date_value, currency, trip_type, stops,
            return_date, adults, children, travel_class,
        )
        if validation:
            errors.append(f"{date_value}: {validation}")
    if errors:
        return "Some dates could not be searched:\n" + "\n".join(f"- {item}" for item in errors)

    dep = dep_id.strip().upper()
    arr = arr_id.strip().upper()
    curr = currency.strip().upper()
    total_pax = adults + children
    comparisons: list[dict[str, Any]] = []

    # The same return-date result is reusable across outbound dates.
    return_result = None
    if trip_type == "1":
        return_result = fetch_flights(
            arr, dep, return_date, curr, "2", stops, None, adults, children,
            force_refresh=force_refresh, travel_class=travel_class,
        )

    for date_value in dates:
        outbound = fetch_flights(
            dep, arr, date_value, curr, "2", stops, None, adults, children,
            force_refresh=force_refresh, travel_class=travel_class,
        )
        if outbound.status != "success" or not outbound.flights:
            comparisons.append({
                "date": date_value,
                "status": outbound.status,
                "error": outbound.error_message,
                "total": None,
                "outbound": [],
                "return": None,
            })
            continue

        out_flights = sorted(outbound.flights, key=lambda item: item.get("price", float("inf")))
        cheapest_out = out_flights[0]
        cheapest_return = None
        combined_total = cheapest_out.get("price")
        status = "success"
        error = None

        if trip_type == "1":
            if return_result is None or return_result.status != "success" or not return_result.flights:
                status = "return_unavailable"
                error = (return_result.error_message if return_result else None) or "No return flights were found."
                combined_total = None
            else:
                ret_flights = sorted(return_result.flights, key=lambda item: item.get("price", float("inf")))
                cheapest_return = ret_flights[0]
                combined_total = cheapest_out["price"] + cheapest_return["price"]

        comparisons.append({
            "date": date_value,
            "status": status,
            "error": error,
            "total": combined_total,
            "outbound": out_flights[:max_options_per_date],
            "return": cheapest_return,
        })

    successful = [item for item in comparisons if item["total"] is not None]
    successful.sort(key=lambda item: (item["total"], item["date"]))

    cabin_label = {"1": "Economy", "2": "Premium Economy", "3": "Business", "4": "First"}[str(travel_class)]
    lines = [
        f"DATE COMPARISON: {dep} → {arr}",
        f"Trip: {'Round trip' if trip_type == '1' else 'One way'} | Cabin: {cabin_label} | Passengers: {total_pax} | Currency: {curr}",
        f"Dates searched: {', '.join(dates)}",
        "",
    ]

    if not successful:
        lines.append("No complete priced option was found for the requested dates.")
        for item in comparisons:
            detail = item.get("error") or item.get("status") or "No flights found."
            lines.append(f"- {item['date']}: {detail}")
        return "\n".join(lines)

    lines.append("CHEAPEST DATES (sorted by total price):")
    for rank, item in enumerate(successful, start=1):
        lines.append(f"{rank}. {item['date']} — {curr} {item['total']:,.0f}")

    failed = [item for item in comparisons if item["total"] is None]
    if failed:
        lines.extend(["", "DATES WITHOUT A COMPLETE PRICE:"])
        for item in failed:
            lines.append(f"- {item['date']}: {item.get('error') or item.get('status') or 'No flights found.'}")

    lines.extend(["", "FLIGHT DETAILS BY DATE:"])
    ordered = successful + failed
    for item in ordered:
        lines.append("")
        lines.append(f"DATE {item['date']} | STATUS: {item['status']}")
        if item["total"] is None:
            lines.append(item.get("error") or "No priced itinerary was returned.")
            continue

        if trip_type == "1":
            lines.append(
                f"Indicative combined total: {curr} {item['total']:,.0f} "
                "(separately searched cheapest outbound + cheapest return legs; not a guaranteed single-ticket fare)."
            )
            if item.get("return"):
                lines.append("Cheapest return leg:")
                lines.append(format_flight_summary(
                    item["return"], curr, total_passengers=total_pax,
                    leg_note="Return Leg Fare",
                ))
        else:
            lines.append(f"Cheapest one-way fare: {curr} {item['total']:,.0f}")

        lines.append("Outbound options:")
        for option_number, itinerary in enumerate(item["outbound"], start=1):
            lines.append(f"Option {option_number}: " + format_flight_summary(
                itinerary, curr, total_passengers=total_pax,
                leg_note="Outbound Leg Fare" if trip_type == "1" else "One-Way Fare",
            ))

    lines.extend([
        "",
        "Comparison note: provider-reported prices can change and must be rechecked before booking.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# High-level run_flight_agent (FL-04, F-30)
# ---------------------------------------------------------------------------

def run_flight_agent(
    dep_id: str,
    arr_id: str,
    target_date: str,
    currency: str = "INR",
    trip_type: str = "2",
    stops: str = "0",
    return_date: str | None = None,
    adults: int = 1,
    children: int = 0,
    echo: bool = False,
    force_refresh: bool = False,
    travel_class: str = "1",
) -> str:
    """Execute flight search agent with input validation and accurate per-leg pricing."""
    val_err = _validate_flight_request(dep_id, arr_id, target_date, currency, trip_type, stops, return_date, adults, children, travel_class)
    if val_err:
        report = f"⚠️ Validation Error: {val_err}"
        if echo:
            print(report)
        return report

    norm_dep = dep_id.strip().upper()
    norm_arr = arr_id.strip().upper()
    norm_curr = currency.strip().upper()
    total_pax = adults + children

    lines: list[str] = []
    trip_label  = "Round Trip" if trip_type == "1" else "One-Way"
    stops_label = {"1": "Nonstop", "2": "Up to 1 stop"}.get(stops, "Any stops")
    date_range  = f"{target_date} → {return_date}" if trip_type == "1" and return_date else target_date
    pax_desc    = f"{adults} adult(s)" + (f", {children} child(ren)" if children > 0 else "")
    cabin_label = {"1": "Economy", "2": "Premium Economy", "3": "Business", "4": "First"}[str(travel_class)]
    lines.append(
        f"🔎 Searching {trip_label} ({stops_label}) {cabin_label} flights from {norm_dep} to {norm_arr} "
        f"on {date_range} for {pax_desc}...\n"
    )

    fetch_result = fetch_flights(
        norm_dep, norm_arr, target_date, norm_curr, "2", stops, None, adults, children,
        force_refresh=force_refresh, travel_class=travel_class,
    )

    if fetch_result.status == "error":
        lines.append(f"⚠️ Flight search failed: {fetch_result.error_message}")
        if fetch_result.retryable:
            lines.append("   Please try again in a moment.")
        report = "\n".join(lines)
        if echo:
            print(report)
        return report

    if fetch_result.status == "empty" or not fetch_result.flights:
        lines.append(
            "😔 No flights found for your selected date and preferences. "
            "Please try different dates or options."
        )
        report = "\n".join(lines)
        if echo:
            print(report)
        return report

    main_flights = fetch_result.flights
    cheapest_outbound = min(main_flights, key=lambda x: x["price"])

    if trip_type == "2":
        try:
            date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            date_obj = None

        if date_obj:
            today_date = datetime.now(APP_TIMEZONE).date()
            max_date   = today_date + timedelta(days=MAX_ADVANCE_DAYS)
            min_main_price = cheapest_outbound["price"]
            best_alternative = None

            for offset in range(-FLEXIBLE_DATE_RANGE_DAYS, FLEXIBLE_DATE_RANGE_DAYS + 1):
                if offset == 0:
                    continue
                alt_dt = date_obj + timedelta(days=offset)
                if alt_dt < today_date or alt_dt > max_date:
                    continue

                alt_date_str = alt_dt.strftime("%Y-%m-%d")
                alt_result   = fetch_flights(
                    norm_dep, norm_arr, alt_date_str, norm_curr, "2", stops, None, adults, children,
                    travel_class=travel_class,
                )
                if alt_result.status == "success" and alt_result.flights:
                    alt_min_price = min(f["price"] for f in alt_result.flights)
                    if alt_min_price <= min_main_price * SIGNIFICANT_SAVINGS_RATIO:
                        if not best_alternative or alt_min_price < best_alternative[2]:
                            best_alternative = (offset, alt_date_str, alt_min_price)

            if best_alternative:
                offset, alt_date_str, alt_min_price = best_alternative
                direction = f"{abs(offset)} day(s) earlier" if offset < 0 else f"{abs(offset)} day(s) later"
                savings = min_main_price - alt_min_price
                lines.append(
                    f"💡 FLEXIBLE DATE TIP (indicative — ±{FLEXIBLE_DATE_RANGE_DAYS} days range): "
                    f"Shifting your trip by {direction} ({alt_date_str}) starts from "
                    f"{norm_curr} {alt_min_price:,.0f} total (potential saving of {norm_curr} {savings:,.0f} vs {target_date}). "
                    f"Actual availability and pricing may differ — verify before booking.\n"
                )

    leg_header = (
        f"🛫 OUTBOUND FLIGHTS ({norm_dep} → {norm_arr} on {target_date})"
        if trip_type == "1"
        else f"✈️ AVAILABLE FLIGHTS ({norm_dep} → {norm_arr} on {target_date})"
    )
    lines.append(f"*** {leg_header} ***")
    lines.append(f"Requested cabin class: {cabin_label}")
    if trip_type == "1":
        lines.append("ℹ️ Note: Prices below are for the standalone Outbound leg only.\n")

    price_tiers = group_into_tiers(main_flights)
    out_label = "Outbound Leg Fare" if trip_type == "1" else "One-Way Fare"

    for tier_name, flight_list in price_tiers.items():
        total_in_tier = len(flight_list)
        shown_count   = min(MAX_DISPLAY_PER_TIER, total_in_tier)

        lines.append(f"=== {tier_name.upper()} (Showing {shown_count} of {total_in_tier} options) ===")
        if not flight_list:
            lines.append("  No flights available in this price range.\n")
            continue

        for itinerary in flight_list[:MAX_DISPLAY_PER_TIER]:
            lines.append(format_flight_summary(itinerary, norm_curr, total_passengers=total_pax, leg_note=out_label))
            lines.append("-" * 50)
        lines.append("")

    if trip_type == "1" and return_date:
        lines.append(f"\n*** 🛬 RETURN FLIGHTS ({norm_arr} → {norm_dep} on {return_date}) ***")
        lines.append("ℹ️ Note: Prices below are for the standalone Return leg only.\n")
        return_result = fetch_flights(
            dep_id=norm_arr,
            arr_id=norm_dep,
            date_str=return_date,
            currency=norm_curr,
            trip_type="2",
            stops=stops,
            return_date=None,
            adults=adults,
            children=children,
            force_refresh=force_refresh,
            travel_class=travel_class,
        )
        if return_result.status == "success" and return_result.flights:
            cheapest_return = min(return_result.flights, key=lambda x: x["price"])
            return_tiers = group_into_tiers(return_result.flights)
            for r_tier_name, r_flight_list in return_tiers.items():
                if not r_flight_list:
                    continue
                r_shown = min(MAX_DISPLAY_PER_TIER, len(r_flight_list))
                lines.append(f"=== RETURN {r_tier_name.upper()} (Showing {r_shown} of {len(r_flight_list)} options) ===")
                for r_itinerary in r_flight_list[:MAX_DISPLAY_PER_TIER]:
                    lines.append(format_flight_summary(r_itinerary, norm_curr, total_passengers=total_pax, leg_note="Return Leg Fare"))
                    lines.append("-" * 50)
                lines.append("")

            combined_min = cheapest_outbound["price"] + cheapest_return["price"]
            lines.append(
                f"💰 CHEAPEST COMBINED ROUND-TRIP SUMMARY ({target_date} to {return_date}):\n"
                f"  - Cheapest Outbound Leg ({target_date}): {norm_curr} {cheapest_outbound['price']:,.0f}\n"
                f"  - Cheapest Return Leg ({return_date}): {norm_curr} {cheapest_return['price']:,.0f}\n"
                f"  - Indicative Combined Round-Trip Cost: {norm_curr} {combined_min:,.0f}\n"
                "  - Pricing note: this is the sum of separately searched cheapest outbound and return legs; it is not a guaranteed single-ticket fare."
            )
        else:
            lines.append(f"⚠️ Could not retrieve return flight schedules for {return_date}.\n")

    report = "\n".join(lines).strip()
    if echo:
        print(report)
    return report


# ---------------------------------------------------------------------------
# Destination exploration (FL-11, FL-12, F-22)
# ---------------------------------------------------------------------------

def explore_destinations(
    dep_id: str,
    candidate_airports: list[str],
    date_str: str,
    budget: float,
    currency: str = "INR",
    trip_type: str = "2",
    stops: str = "0",
    return_date: str | None = None,
    adults: int = 1,
    children: int = 0,
    echo: bool = False,
    travel_class: str = "1",
) -> str:
    if not is_valid_code(dep_id):
        msg = f"⚠️ Invalid departure airport '{dep_id}'."
        if echo:
            print(msg)
        return msg

    norm_dep = dep_id.strip().upper()
    norm_curr = currency.strip().upper()

    if not isinstance(candidate_airports, list):
        candidate_airports = []

    candidates = [
        code.strip().upper()
        for code in candidate_airports
        if isinstance(code, str) and is_valid_code(code.strip().upper()) and code.strip().upper() != norm_dep
    ][:MAX_EXPLORE_AIRPORTS]

    if not candidates:
        msg = "⚠️ No valid destination airports were provided to explore."
        if echo:
            print(msg)
        return msg

    pax_desc = f"{adults} adult(s)" + (f", {children} child(ren)" if children > 0 else "")
    lines = [
        f"🔍 Searching {len(candidates)} destination(s) for flights within {norm_curr} {budget:,.0f} total ({pax_desc})...\n"
    ]
    matches: list[tuple[str, str, float]] = []
    failed_airports: list[str] = []

    def _check_airport(arr_code: str) -> tuple[str, str, float] | None:
        result = fetch_flights(
            norm_dep, arr_code, date_str, norm_curr, trip_type, stops, return_date, adults, children,
            travel_class=travel_class,
        )
        if result.status == "error":
            raise RuntimeError(f"{arr_code}: {result.error_message}")
        if not result.flights:
            return None
        affordable = [f for f in result.flights if f["price"] <= budget]
        if not affordable:
            return None
        cheapest = min(affordable, key=lambda f: f["price"])
        legs = cheapest.get("flights") or []
        if isinstance(legs, list) and legs and isinstance(legs[-1], dict):
            arr_airport = legs[-1].get("arrival_airport") or {}
            arr_name = arr_airport.get("name", arr_code) if isinstance(arr_airport, dict) else arr_code
        else:
            arr_name = arr_code
        clean_name = _safe_provider_text(str(arr_name), 40)
        return (arr_code, clean_name, cheapest["price"])

    max_workers = min(5, len(candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_check_airport, code): code for code in candidates}
        for future in as_completed(futures):
            airport_code = futures[future]
            try:
                res = future.result()
                if res:
                    matches.append(res)
            except Exception as exc:
                logger.warning("Destination exploration failed for %s: %s", airport_code, exc)
                failed_airports.append(airport_code)

    if failed_airports:
        lines.append(f"⚠️ Could not retrieve data for: {', '.join(failed_airports)} (search error).\n")

    if not matches:
        lines.append(f"😔 No destinations found within {norm_curr} {budget:,.0f} from {norm_dep} on {date_str}.")
        report = "\n".join(lines)
        if echo:
            print(report)
        return report

    matches.sort(key=lambda x: x[2])
    lines.append(f"✅ Found {len(matches)} destination(s) within your budget!\n")
    lines.append("=" * 68)
    for i, (code, name, price) in enumerate(matches, 1):
        display_name = (name[:32] + "..") if len(name) > 34 else name
        lines.append(f"  {i:2}. ✈️  {display_name:<34} ({code})   {norm_curr} {price:>8,.0f}")
    lines.append("=" * 68)

    report = "\n".join(lines).strip()
    if echo:
        print(report)
    return report