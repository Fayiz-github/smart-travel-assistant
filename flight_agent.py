import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import serpapi
from dotenv import load_dotenv

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        val = float(os.getenv(key, str(default)))
        return val if math.isfinite(val) else default
    except (ValueError, TypeError):
        return default


# --- DYNAMIC CONFIGURATION (Safely parsed from .env) ---
MIN_SAFE_LAYOVER_MINS     = _env_int("MIN_SAFE_LAYOVER_MINS", 60)
MAX_COMFORT_LAYOVER_MINS  = _env_int("MAX_COMFORT_LAYOVER_MINS", 360)
SIGNIFICANT_SAVINGS_RATIO = _env_float("SIGNIFICANT_SAVINGS_RATIO", 0.85)
CHEAP_TIER_MULTIPLIER     = _env_float("FLIGHT_CHEAP_MULTIPLIER", 1.20)
BUDGET_TIER_MULTIPLIER    = _env_float("FLIGHT_BUDGET_MULTIPLIER", 1.40)
MAX_DISPLAY_PER_TIER      = _env_int("MAX_DISPLAY_PER_TIER", 3)
MAX_ADVANCE_DAYS          = _env_int("MAX_ADVANCE_DAYS", 330)
MAX_EXPLORE_AIRPORTS      = _env_int("MAX_EXPLORE_AIRPORTS", 10)

_client = None


def is_valid_code(code: str) -> bool:
    """Validates standard 3-letter ASCII IATA/ISO codes (A-Z only)."""
    return len(code) == 3 and code.isascii() and code.isalpha()


def get_serpapi_client() -> serpapi.Client:
    """Lazily initializes and returns the SerpApi client without import-time side effects."""
    global _client
    if _client is None:
        api_key = (os.getenv("SECRET_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("SECRET_API_KEY environment variable is missing or empty.")
        _client = serpapi.Client(api_key=api_key)
    return _client


def evaluate_layovers(itinerary: dict) -> list[str]:
    """Dynamically evaluates layovers using SerpApi response fields."""
    warnings = []
    layovers = itinerary.get("layovers", [])
    flights  = itinerary.get("flights", [])

    if not layovers:
        return ["Direct flight (No layovers)"]

    for layover in layovers:
        duration     = layover.get("duration", 0)
        airport_name = layover.get("name", "Unknown Airport")
        airport_code = layover.get("id", "N/A")

        hours, mins = divmod(duration, 60)
        time_str = f"{hours}h {mins}m" if hours else f"{mins}m"

        if duration < MIN_SAFE_LAYOVER_MINS:
            warnings.append(
                f"⚠️ TIGHT CONNECTION at {airport_name} ({airport_code}): Only {time_str} to change planes."
            )
        elif duration > MAX_COMFORT_LAYOVER_MINS:
            warnings.append(
                f"⏳ LONG LAYOVER at {airport_name} ({airport_code}): {time_str} wait."
            )
        else:
            warnings.append(
                f"✅ Comfortable layover at {airport_name} ({airport_code}): {time_str}."
            )

        if layover.get("overnight"):
            warnings.append(f"🌙 OVERNIGHT layover at {airport_code}.")

    if len(flights) > 1:
        warnings.append(
            "🛂 CONNECTION NOTE: Kindly verify if your transit airport(s) require a transit visa or self-transfer."
        )

    for ext in itinerary.get("extensions", []):
        if isinstance(ext, str) and any(k in ext.lower() for k in ("visa", "transfer", "change", "baggage")):
            warnings.append(f"ℹ️ {ext}")

    return warnings


def fetch_flights(
    dep_id: str,
    arr_id: str,
    date_str: str,
    currency: str = "INR",
    trip_type: str = "2",
    stops: str = "0",
    return_date: str | None = None
) -> list[dict]:
    """Fetches and combines 'best_flights' and 'other_flights' from SerpApi safely."""
    params = {
        "engine": "google_flights",
        "departure_id": dep_id,
        "arrival_id": arr_id,
        "currency": currency,
        "type": trip_type,
        "stops": stops,
        "outbound_date": date_str
    }
    if trip_type == "1" and return_date:
        params["return_date"] = return_date

    try:
        client  = get_serpapi_client()
        results = client.search(params)
    except Exception as exc:
        print(f"⚠️ Network/API Error while fetching flights for {arr_id} on {date_str}: {exc}")
        return []

    if results is None or not hasattr(results, "get"):
        return []

    if "error" in results:
        print(f"⚠️ API Error for {arr_id} ({date_str}): {results['error']}")
        return []

    all_flights = results.get("best_flights", []) + results.get("other_flights", [])
    return [
        f for f in all_flights
        if isinstance(f.get("price"), (int, float))
        and not isinstance(f.get("price"), bool)
        and math.isfinite(f["price"])
        and f["price"] > 0
    ]


def group_into_tiers(flights: list[dict]) -> dict[str, list[dict]]:
    """Groups flights into price tiers relative to the lowest price, with a percentile fallback."""
    if not flights:
        return {}

    sorted_flights = sorted(flights, key=lambda x: x["price"])
    unique_prices  = sorted({f["price"] for f in sorted_flights})

    if len(unique_prices) <= 2:
        return {"Available Flights": sorted_flights}

    min_price     = sorted_flights[0]["price"]
    cheap_cutoff  = min_price * CHEAP_TIER_MULTIPLIER
    budget_cutoff = min_price * BUDGET_TIER_MULTIPLIER

    tiers = {"Cheap Tier": [], "Budget Tier": [], "Premium Tier": []}
    for f in sorted_flights:
        price = f["price"]
        if price <= cheap_cutoff:
            tiers["Cheap Tier"].append(f)
        elif price <= budget_cutoff:
            tiers["Budget Tier"].append(f)
        else:
            tiers["Premium Tier"].append(f)

    if not all(tiers.values()):
        u_len         = len(unique_prices)
        cheap_cutoff  = unique_prices[max(0, (u_len // 3) - 1)]
        budget_cutoff = unique_prices[max(1, ((u_len * 2) // 3) - 1)]

        tiers = {"Cheap Tier": [], "Budget Tier": [], "Premium Tier": []}
        for f in sorted_flights:
            price = f["price"]
            if price <= cheap_cutoff:
                tiers["Cheap Tier"].append(f)
            elif price <= budget_cutoff:
                tiers["Budget Tier"].append(f)
            else:
                tiers["Premium Tier"].append(f)

    return tiers


def format_flight_summary(itinerary: dict, currency: str = "INR") -> str:
    """Formats a single flight itinerary for display."""
    price          = itinerary.get("price", 0)
    price_str      = f"{price:,.0f}" if isinstance(price, (int, float)) else "N/A"
    total_duration = itinerary.get("total_duration", 0)
    hours, mins    = divmod(total_duration, 60)

    connecting_flights = itinerary.get("flights", [])
    if not connecting_flights:
        return f"Price: {currency} {price_str}"

    first_flight = connecting_flights[0]
    last_flight  = connecting_flights[-1]

    airlines = ", ".join(sorted({f.get("airline", "Unknown") for f in connecting_flights if f.get("airline")}))
    dep_time = first_flight.get("departure_airport", {}).get("time", "N/A")
    arr_time = last_flight.get("arrival_airport", {}).get("time", "N/A")

    layover_notes = evaluate_layovers(itinerary)
    notes_str     = "\n      ".join(layover_notes)

    return (
        f"• {airlines} | {currency} {price_str} | Total Time: {hours}h {mins}m\n"
        f"  Departs: {dep_time} -> Arrives: {arr_time}\n"
        f"  Layover Notes:\n      {notes_str}"
    )


def run_flight_agent(
    dep_id: str,
    arr_id: str,
    target_date: str,
    currency: str = "INR",
    trip_type: str = "2",
    stops: str = "0",
    return_date: str | None = None
):
    trip_label  = "Round Trip" if trip_type == "1" else "One-Way"
    stops_label = {"1": "Nonstop", "2": "Up to 1 stop"}.get(stops, "Any stops")
    date_range  = f"{target_date} → {return_date}" if trip_type == "1" and return_date else target_date
    print(f"🔎 Searching {trip_label} ({stops_label}) flights from {dep_id} to {arr_id} on {date_range}...\n")

    main_flights = fetch_flights(dep_id, arr_id, target_date, currency, trip_type, stops, return_date)
    if not main_flights:
        print("😔 We couldn't find any flights for your selected date and preferences. Please try different dates or options.")
        return

    if trip_type == "2":
        date_obj       = datetime.strptime(target_date, "%Y-%m-%d")
        today_date     = datetime.today().date()
        max_date       = today_date + timedelta(days=MAX_ADVANCE_DAYS)
        min_main_price = min(f["price"] for f in main_flights)

        adjacent_candidates = []
        prev_dt = date_obj - timedelta(days=1)
        if prev_dt.date() >= today_date:
            adjacent_candidates.append(("day before", prev_dt.strftime("%Y-%m-%d")))

        next_dt = date_obj + timedelta(days=1)
        if next_dt.date() <= max_date:
            adjacent_candidates.append(("day after", next_dt.strftime("%Y-%m-%d")))

        for adj_label, adj_date in adjacent_candidates:
            adj_flights = fetch_flights(dep_id, arr_id, adj_date, currency, trip_type, stops)
            if adj_flights:
                min_adj_price = min(f["price"] for f in adj_flights)
                if min_adj_price <= min_main_price * SIGNIFICANT_SAVINGS_RATIO:
                    savings = min_main_price - min_adj_price
                    print(
                        f"💡 FLEXIBLE DATE TIP: Leaving the {adj_label} ({adj_date}) starts at "
                        f"{currency} {min_adj_price:,.0f} (Save {currency} {savings:,.0f} vs {target_date})!\n"
                    )

    price_tiers = group_into_tiers(main_flights)

    for tier_name, flight_list in price_tiers.items():
        total_in_tier = len(flight_list)
        shown_count   = min(MAX_DISPLAY_PER_TIER, total_in_tier)

        print(f"=== {tier_name.upper()} (Showing {shown_count} of {total_in_tier} options) ===")
        if not flight_list:
            print("  No flights available in this price range.\n")
            continue

        for itinerary in flight_list[:MAX_DISPLAY_PER_TIER]:
            print(format_flight_summary(itinerary, currency))
            print("-" * 50)
        print()


def explore_destinations(
    dep_id: str,
    candidate_airports: list[str],
    date_str: str,
    budget: float,
    currency: str = "INR",
    trip_type: str = "2",
    stops: str = "0",
    return_date: str | None = None
):
    """Searches user-specified or env-configured destinations concurrently within budget."""
    candidates = [code for code in candidate_airports if code != dep_id][:MAX_EXPLORE_AIRPORTS]
    if not candidates:
        print("⚠️ No valid destination airports were provided to explore. Please try again with different airport codes.")
        return

    print(f"\n🔍 Kindly hold on while we search {len(candidates)} destination(s) for flights within {currency} {budget:,.0f}...\n")

    matches = []

    def _check_airport(arr_code: str):
        flights = fetch_flights(dep_id, arr_code, date_str, currency, trip_type, stops, return_date)
        if not flights:
            return None
        affordable = [f for f in flights if f["price"] <= budget]
        if not affordable:
            return None
        cheapest = min(affordable, key=lambda f: f["price"])
        legs = cheapest.get("flights", [])
        arr_name = legs[-1].get("arrival_airport", {}).get("name", arr_code) if legs else arr_code
        return (arr_code, arr_name, cheapest["price"])

    max_workers = min(5, len(candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_check_airport, code): code for code in candidates}
        for future in as_completed(futures):
            result = future.result()
            if result:
                matches.append(result)

    if not matches:
        print(f"😔 We're sorry, no destinations were found within {currency} {budget:,.0f} from {dep_id} on {date_str}.")
        print("   Please consider trying a slightly higher budget or another travel date. 😊")
        return

    matches.sort(key=lambda x: x[2])
    print(f"✅ Wonderful! We found {len(matches)} destination(s) within your budget!\n")
    print("=" * 68)
    for i, (code, name, price) in enumerate(matches, 1):
        display_name = (name[:32] + "..") if len(name) > 34 else name
        print(f"  {i:2}. ✈️  {display_name:<34} ({code})   {currency} {price:>8,.0f}")
    print("=" * 68)
    print("\n💡 Tip: Feel free to run the agent again and choose Option 1 to view full details for your preferred destination! 😊")


if __name__ == "__main__":
    run_flight_agent(
        dep_id="MYQ",
        arr_id="MAA",
        target_date="2026-10-29",
        currency="INR"
    )


