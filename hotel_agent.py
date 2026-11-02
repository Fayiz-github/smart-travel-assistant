import math
import os
import sys
import re
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
MIN_SAFE_RATING           = _env_float("MIN_SAFE_RATING", 3.5)
HIGH_RATING_THRESHOLD     = _env_float("HIGH_RATING_THRESHOLD", 4.3)
SIGNIFICANT_SAVINGS_RATIO = _env_float("SIGNIFICANT_SAVINGS_RATIO", 0.85)
CHEAP_MEDIAN_FACTOR       = _env_float("HOTEL_CHEAP_MEDIAN_FACTOR", 0.75)
BUDGET_MEDIAN_FACTOR      = _env_float("HOTEL_BUDGET_MEDIAN_FACTOR", 1.30)
MAX_DISPLAY_PER_TIER      = _env_int("MAX_DISPLAY_PER_TIER", 3)
MAX_AMENITIES_SHOWN       = _env_int("MAX_AMENITIES_SHOWN", 6)
MAX_ADVANCE_DAYS          = _env_int("MAX_ADVANCE_DAYS", 330)
MAX_STAY_NIGHTS           = _env_int("MAX_STAY_NIGHTS", 30)
MAX_ADULTS_LIMIT          = _env_int("MAX_ADULTS_LIMIT", 20)

_client = None


def is_valid_code(code: str) -> bool:
    """Validates standard 3-letter ASCII ISO currency codes (A-Z only)."""
    return len(code) == 3 and code.isascii() and code.isalpha()


def is_valid_location(text: str) -> bool:
    """Requires at least 2 letters and allows letters, digits, spaces, and common city punctuation."""
    if not text or len(text) > 100:
        return False
    letter_count = sum(1 for c in text if c.isalpha())
    return letter_count >= 2 and bool(re.match(r"^[\w\s\-,.'()&/]+$", text))


def is_valid_dietary_pref(text: str) -> bool:
    """Requires at least 2 letters and only letters, spaces, or hyphens for food preferences."""
    if not text or len(text) > 40:
        return False
    letter_count = sum(1 for c in text if c.isalpha())
    return letter_count >= 2 and bool(re.match(r"^[A-Za-z\s\-]+$", text))


def get_serpapi_client() -> serpapi.Client:
    """Lazily initializes and returns the SerpApi client without import-time side effects."""
    global _client
    if _client is None:
        api_key = (os.getenv("SECRET_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("SECRET_API_KEY environment variable is missing or empty.")
        _client = serpapi.Client(api_key=api_key)
    return _client


def evaluate_hotel_details(hotel: dict) -> list[str]:
    """Dynamically evaluates a hotel for ratings, deals, amenities, dietary match, and timings."""
    notes = []

    rating  = hotel.get("overall_rating")
    reviews = hotel.get("reviews", 0)

    if rating is None:
        notes.append("⚠️ UNRATED PROPERTY: No guest rating available yet — book with caution.")
    elif rating < MIN_SAFE_RATING:
        notes.append(f"⚠️ LOW GUEST RATING: {rating}/5 ({reviews:,} reviews) — check recent guest complaints.")
    elif rating >= HIGH_RATING_THRESHOLD:
        notes.append(f"⭐ TOP-RATED STAY: {rating}/5 ({reviews:,} reviews).")
    else:
        notes.append(f"✅ Decent Rating: {rating}/5 ({reviews:,} reviews).")

    deal = hotel.get("deal")
    if deal:
        notes.append(f"🏷️ SPECIAL DEAL: {deal}")

    amenities = [a for a in hotel.get("amenities", []) if isinstance(a, str) and a.strip()]
    if amenities:
        shown = amenities[:MAX_AMENITIES_SHOWN]
        extra = f" (+{len(amenities) - len(shown)} more)" if len(amenities) > len(shown) else ""
        notes.append(f"🛎️ Amenities: {', '.join(shown)}{extra}")
    else:
        notes.append("ℹ️ No specific amenities listed.")

    if hotel.get("food_match_note"):
        notes.append(hotel["food_match_note"])

    check_in_time  = hotel.get("check_in_time")
    check_out_time = hotel.get("check_out_time")
    if check_in_time or check_out_time:
        notes.append(f"🕒 Timings: Check-in {check_in_time or 'N/A'} | Check-out {check_out_time or 'N/A'}")

    return notes


def extract_nightly_price(property_item: dict, num_nights: int = 1) -> float | None:
    """Extracts a positive finite nightly rate or divides total rate by nights."""
    rate_info = property_item.get("rate_per_night", {})
    price = rate_info.get("extracted_lowest")
    if isinstance(price, (int, float)) and not isinstance(price, bool) and math.isfinite(price) and price > 0:
        return float(price)

    total_info  = property_item.get("total_rate", {})
    total_price = total_info.get("extracted_lowest")
    if isinstance(total_price, (int, float)) and not isinstance(total_price, bool) and math.isfinite(total_price) and total_price > 0:
        return round(float(total_price) / max(1, num_nights), 2)

    return None


def fetch_hotels(
    location_query: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    currency: str = "INR"
) -> list[dict]:
    """Fetches hotels from SerpApi safely and attaches a clean 'price' key."""
    params = {
        "engine": "google_hotels",
        "q": location_query,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": str(adults),
        "currency": currency
    }

    try:
        client  = get_serpapi_client()
        results = client.search(params)
    except Exception as exc:
        print(f"⚠️ Network/API Error while fetching hotels for {check_in_date}: {exc}")
        return []

    if results is None or not hasattr(results, "get"):
        return []

    if "error" in results:
        print(f"⚠️ API Error for {check_in_date}: {results['error']}")
        return []

    in_date    = datetime.strptime(check_in_date,  "%Y-%m-%d")
    out_date   = datetime.strptime(check_out_date, "%Y-%m-%d")
    num_nights = max(1, (out_date - in_date).days)

    valid_hotels = []
    for prop in results.get("properties", []):
        price = extract_nightly_price(prop, num_nights)
        if price is not None:
            prop_copy = dict(prop)
            prop_copy["price"] = price
            valid_hotels.append(prop_copy)

    return valid_hotels


def filter_by_star_rating(
    hotels: list[dict],
    min_star: int | None,
    only_exact: bool = False,
    allow_fallback: bool = True
) -> tuple[list[dict], bool]:
    """Dynamically filters hotels >= min_star (or == min_star if only_exact is True)."""
    if not min_star or not hotels:
        return hotels, False

    filtered = []
    for hotel in hotels:
        star_class = hotel.get("extracted_hotel_class")
        if star_class is not None:
            try:
                star_class = int(star_class)
            except (ValueError, TypeError):
                star_class = None

        if star_class is None:
            hc_str = str(hotel.get("hotel_class", ""))
            match  = re.search(r"\b([1-5])\b", hc_str)
            if match:
                star_class = int(match.group(1))

        if star_class is not None:
            if (only_exact and star_class == min_star) or (not only_exact and star_class >= min_star):
                filtered.append(hotel)

    if not filtered:
        if allow_fallback:
            print(f"⚠️ We couldn't find hotels matching {min_star}+ stars, so we are kindly showing all available options instead.\n")
            return hotels, False
        return [], False

    return filtered, True


def filter_by_food_preference(
    hotels: list[dict],
    preference: str | None,
    allow_fallback: bool = True
) -> tuple[list[dict], bool]:
    """Uses word-boundary regex matching for any user-provided dietary keyword."""
    if not preference or not hotels:
        return hotels, False

    pref_clean = preference.lower().strip()
    filtered   = []

    escaped_pref   = re.escape(pref_clean)
    exact_pattern  = re.compile(rf"\b{escaped_pref}\b", re.IGNORECASE)
    dining_pattern = re.compile(r"\b(restaurant|dining|breakfast|buffet|cafe)\b", re.IGNORECASE)

    for hotel in hotels:
        amenities     = [str(a) for a in hotel.get("amenities", [])]
        description   = str(hotel.get("description", ""))
        property_name = str(hotel.get("name", ""))
        full_text     = f"{' '.join(amenities)} {description} {property_name}"

        if exact_pattern.search(full_text):
            hotel["food_match_note"] = f"🍽️ Explicitly mentions '{preference}' options."
            filtered.append(hotel)
        elif dining_pattern.search(full_text):
            hotel["food_match_note"] = f"🍴 On-site dining/breakfast available (kindly verify '{preference}' options with the property)."
            filtered.append(hotel)

    if not filtered:
        if allow_fallback:
            print(f"⚠️ No hotels explicitly matched '{preference}', so we are kindly showing all available options.\n")
            return hotels, False
        return [], False

    return filtered, True


def group_into_tiers(hotels: list[dict]) -> dict[str, list[dict]]:
    """Groups hotels into price tiers relative to the median nightly price, with a percentile fallback."""
    if not hotels:
        return {}

    sorted_hotels = sorted(hotels, key=lambda x: x["price"])
    unique_prices = sorted({h["price"] for h in sorted_hotels})

    if len(unique_prices) <= 2:
        return {"Available Hotels": sorted_hotels}

    median_price  = sorted_hotels[len(sorted_hotels) // 2]["price"]
    cheap_cutoff  = median_price * CHEAP_MEDIAN_FACTOR
    budget_cutoff = median_price * BUDGET_MEDIAN_FACTOR

    tiers = {"Cheap Tier": [], "Budget Tier": [], "Premium Tier": []}
    for h in sorted_hotels:
        price = h["price"]
        if price <= cheap_cutoff:
            tiers["Cheap Tier"].append(h)
        elif price <= budget_cutoff:
            tiers["Budget Tier"].append(h)
        else:
            tiers["Premium Tier"].append(h)

    if not all(tiers.values()):
        u_len         = len(unique_prices)
        cheap_cutoff  = unique_prices[max(0, (u_len // 3) - 1)]
        budget_cutoff = unique_prices[max(1, ((u_len * 2) // 3) - 1)]

        tiers = {"Cheap Tier": [], "Budget Tier": [], "Premium Tier": []}
        for h in sorted_hotels:
            price = h["price"]
            if price <= cheap_cutoff:
                tiers["Cheap Tier"].append(h)
            elif price <= budget_cutoff:
                tiers["Budget Tier"].append(h)
            else:
                tiers["Premium Tier"].append(h)

    return tiers


def format_hotel_summary(hotel: dict, currency: str = "INR") -> str:
    """Formats a single hotel property for display."""
    name           = hotel.get("name", "Unknown Hotel")
    price          = hotel.get("price")
    price_str      = f"{price:,.0f}" if isinstance(price, (int, float)) else "N/A"
    hotel_class    = hotel.get("hotel_class", "Unclassified")
    property_type  = str(hotel.get("type", "hotel")).title()
    total_rate_str = hotel.get("total_rate", {}).get("lowest", "N/A")

    hotel_notes = evaluate_hotel_details(hotel)
    notes_str   = "\n      ".join(hotel_notes) if hotel_notes else "No additional notes."

    return (
        f"• {name} ({hotel_class} {property_type}) | {currency} {price_str}/night (Total: {total_rate_str})\n"
        f"  Hotel Notes:\n      {notes_str}"
    )


def run_hotel_agent(
    location_query: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    currency: str = "INR",
    min_star: int | None = None,
    only_exact_star: bool = False,
    food_preference: str | None = None
):
    in_date_obj  = datetime.strptime(check_in_date,  "%Y-%m-%d")
    out_date_obj = datetime.strptime(check_out_date, "%Y-%m-%d")
    if out_date_obj <= in_date_obj:
        print("Error: check_out_date must be after check_in_date.")
        return

    star_desc = f"{min_star}★ only" if (min_star and only_exact_star) else (f"{min_star}★+" if min_star else "")
    print(
        f"🏨 Searching hotels in '{location_query}' "
        f"({check_in_date} to {check_out_date}) for {adults} adult(s)"
        f"{f' [Stars: {star_desc}]' if star_desc else ''}"
        f"{f' [Food: {food_preference}]' if food_preference else ''}...\n"
    )

    main_hotels = fetch_hotels(location_query, check_in_date, check_out_date, adults, currency)
    if not main_hotels:
        print("No hotels found for the selected dates.")
        return

    star_matched = False
    food_matched = False

    if min_star:
        main_hotels, star_matched = filter_by_star_rating(
            main_hotels, min_star, only_exact=only_exact_star, allow_fallback=True
        )

    if food_preference:
        main_hotels, food_matched = filter_by_food_preference(main_hotels, food_preference, allow_fallback=True)

    if not main_hotels:
        print("No hotels matched your filters for these dates.")
        return

    today_date = datetime.today().date()
    max_date   = today_date + timedelta(days=MAX_ADVANCE_DAYS)
    adjacent_shifts = []

    prev_in  = in_date_obj - timedelta(days=1)
    prev_out = out_date_obj - timedelta(days=1)
    if prev_in.date() >= today_date:
        adjacent_shifts.append(("1 day earlier", prev_in.strftime("%Y-%m-%d"), prev_out.strftime("%Y-%m-%d")))

    next_in  = in_date_obj + timedelta(days=1)
    next_out = out_date_obj + timedelta(days=1)
    if next_out.date() <= max_date:
        adjacent_shifts.append(("1 day later", next_in.strftime("%Y-%m-%d"), next_out.strftime("%Y-%m-%d")))

    min_main_price = min(h["price"] for h in main_hotels)

    for shift_label, adj_in, adj_out in adjacent_shifts:
        adj_hotels = fetch_hotels(location_query, adj_in, adj_out, adults, currency)

        if star_matched:
            adj_hotels, _ = filter_by_star_rating(
                adj_hotels, min_star, only_exact=only_exact_star, allow_fallback=False
            )
        if food_matched:
            adj_hotels, _ = filter_by_food_preference(adj_hotels, food_preference, allow_fallback=False)

        if adj_hotels:
            min_adj_price = min(h["price"] for h in adj_hotels)
            if min_adj_price <= min_main_price * SIGNIFICANT_SAVINGS_RATIO:
                savings = min_main_price - min_adj_price
                print(
                    f"💡 FLEXIBLE DATE TIP: Shifting your stay {shift_label} ({adj_in} to {adj_out}) "
                    f"starts at {currency} {min_adj_price:,.0f}/night "
                    f"(Save {currency} {savings:,.0f}/night vs your chosen dates)!\n"
                )

    price_tiers = group_into_tiers(main_hotels)

    for tier_name, hotel_list in price_tiers.items():
        total_in_tier = len(hotel_list)
        shown_count   = min(MAX_DISPLAY_PER_TIER, total_in_tier)

        print(f"=== {tier_name.upper()} (Showing {shown_count} of {total_in_tier} options) ===")
        if not hotel_list:
            print("  No hotels in this price range.\n")
            continue

        for hotel in hotel_list[:MAX_DISPLAY_PER_TIER]:
            print(format_hotel_summary(hotel, currency))
            print("-" * 55)
        print()


if __name__ == "__main__":
    run_hotel_agent(
        location_query="Bengaluru",
        check_in_date="2026-11-15",
        check_out_date="2026-11-17",
        adults=2,
        currency="INR",
        star_rating=[4, 5],
        food_preference="veg"
    )
