import os
import sys
import re
from datetime import datetime, timedelta
import serpapi
from dotenv import load_dotenv

# Ensure UTF-8 output encoding on Windows consoles for emoji and currency symbols
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

secret_api_key = os.getenv("SECRET_API_KEY")
if not secret_api_key:
    print("Error: SECRET_API_KEY is not set.")
    exit()

client = serpapi.Client(api_key=secret_api_key)

# --- CONFIGURATION ---
MIN_SAFE_RATING       = 3.5    # Warn if guest rating is below 3.5 / 5.0
HIGH_RATING_THRESHOLD = 4.3   # Highlight if guest rating is 4.3 or higher
SIGNIFICANT_SAVINGS_RATIO = 0.85  # Alert if shifting dates is >= 15% cheaper

# Important amenities to look out for in the hotel data
KEY_AMENITIES = ["Free Wi-Fi", "Free breakfast", "Free parking", "Pool", "Air conditioning", "Room Service", "Fitness center"]


def evaluate_hotel_details(hotel: dict) -> list[str]:
    """Checks a hotel for low ratings, deals, key amenities, and check-in/out times."""
    notes = []

    # 1. Check Guest Rating & Reviews
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

    # 2. Check for Special Deals
    deal = hotel.get("deal")
    if deal:
        notes.append(f"🏷️ SPECIAL DEAL: {deal}")

    # 3. Check Key Amenities
    amenities     = hotel.get("amenities", [])
    matched_perks = [perk for perk in KEY_AMENITIES if perk in amenities]
    if matched_perks:
        notes.append(f"🛎️ Perks Included: {', '.join(matched_perks)}")
    else:
        notes.append("ℹ️ No free breakfast or major perks listed.")

    # 4. Check-in / Check-out timings
    check_in_time  = hotel.get("check_in_time")
    check_out_time = hotel.get("check_out_time")
    if check_in_time or check_out_time:
        notes.append(f"🕒 Timings: Check-in {check_in_time or 'N/A'} | Check-out {check_out_time or 'N/A'}")

    return notes



def extract_nightly_price(property_item: dict, num_nights: int = 1) -> float | None:
    """
    Extracts a positive numerical nightly rate from a hotel property dictionary.
    Falls back to dividing the total rate by the number of nights if rate_per_night is missing.
    """
    rate_info = property_item.get("rate_per_night", {})
    price = rate_info.get("extracted_lowest")
    if isinstance(price, (int, float)) and price > 0:
        return float(price)

    # Fallback: divide total rate by number of nights
    total_info  = property_item.get("total_rate", {})
    total_price = total_info.get("extracted_lowest")
    if isinstance(total_price, (int, float)) and total_price > 0:
        return round(float(total_price) / max(1, num_nights), 2)

    return None


def fetch_hotels(
    location_query: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    currency: str = "INR"
) -> list[dict]:
    """Fetches hotels from SerpApi and attaches a clean 'price' key to each valid hotel."""
    params = {
        "engine": "google_hotels",
        "q": location_query,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": str(adults),
        "currency": currency
    }
    results = client.search(params)

    if "error" in results:
        print(f"API Error for {check_in_date}: {results['error']}")
        return []

    in_date    = datetime.strptime(check_in_date,  "%Y-%m-%d")
    out_date   = datetime.strptime(check_out_date, "%Y-%m-%d")
    num_nights = max(1, (out_date - in_date).days)

    raw_properties = results.get("properties", [])
    valid_hotels   = []

    for prop in raw_properties:
        price = extract_nightly_price(prop, num_nights)
        if price is not None:
            prop["price"] = price
            valid_hotels.append(prop)

    return valid_hotels


def filter_by_star_rating(
    hotels: list[dict],
    target_stars: int | list[int],
    allow_fallback: bool = True
) -> tuple[list[dict], bool]:
    """
    Filters the hotel list to match specific star classifications.
    Returns (filtered_hotels, matched_flag).
    """
    if not target_stars or not hotels:
        return hotels, False

    if isinstance(target_stars, int):
        target_stars = [target_stars]
    elif isinstance(target_stars, str):
        digits = [int(d) for d in re.findall(r'\d+', target_stars)]
        target_stars = digits if digits else []

    if not target_stars:
        return hotels, False

    filtered = []
    for hotel in hotels:
        star_class = hotel.get("extracted_hotel_class")

        # Normalize to int — SerpApi may return a string like "4" or an int
        if star_class is not None:
            try:
                star_class = int(star_class)
            except (ValueError, TypeError):
                star_class = None

        # Fallback: parse star digit from hotel_class string e.g. "4-star hotel"
        if star_class is None:
            hc_str = str(hotel.get("hotel_class", ""))
            match = re.search(r'(\d)', hc_str)
            if match:
                star_class = int(match.group(1))

        if star_class in target_stars:
            filtered.append(hotel)


    if not filtered:
        if allow_fallback:
            print(f"⚠️ No hotels found matching star rating {target_stars}. Showing all available options instead.\n")
            return hotels, False
        return [], False

    return filtered, True


def filter_by_food_preference(
    hotels: list[dict],
    preference: str,
    allow_fallback: bool = True
) -> tuple[list[dict], bool]:
    """
    Scans hotel descriptions, names, and amenities to filter and
    tag properties according to dietary preferences (veg, non-veg, halal).
    Returns (filtered_hotels, matched_flag).
    """
    if not preference or not hotels:
        return hotels, False

    pref_lower = preference.lower().strip()
    filtered   = []

    for hotel in hotels:
        amenities     = [a.lower() for a in hotel.get("amenities", [])]
        description   = hotel.get("description", "").lower()
        property_name = hotel.get("name", "").lower()
        full_text     = " ".join(amenities) + " " + description + " " + property_name

        # Check 'non-veg' BEFORE 'veg' so "non-veg" does not trigger the "veg" substring check
        if "non" in pref_lower:
            if any("restaurant" in a for a in amenities) or "dining" in full_text or "breakfast" in full_text:
                hotel["food_match_note"] = "🍗 Non-vegetarian / Multi-cuisine restaurant on-site."
                filtered.append(hotel)

        elif "veg" in pref_lower:
            if "vegetarian" in full_text or "pure veg" in full_text or "veg options" in full_text:
                hotel["food_match_note"] = "🌱 Vegetarian-friendly options indicated."
                filtered.append(hotel)
            elif any("restaurant" in a for a in amenities) or "breakfast" in full_text:
                hotel["food_match_note"] = "🍽️ Has on-site dining/breakfast."
                filtered.append(hotel)

        elif "halal" in pref_lower:
            if "halal" in full_text:
                hotel["food_match_note"] = "☪️ Halal food options/certification mentioned."
                filtered.append(hotel)
            elif any("restaurant" in a for a in amenities):
                hotel["food_match_note"] = "ℹ️ Has restaurant (Verify halal availability with property)."
                filtered.append(hotel)

        else:
            if pref_lower in full_text:
                hotel["food_match_note"] = f"🍽️ Matches '{preference}' preference."
                filtered.append(hotel)

    if not filtered:
        if allow_fallback:
            print(f"⚠️ No hotels explicitly matched '{preference}'. Showing all available options.\n")
            return hotels, False
        return [], False

    return filtered, True


def group_into_tiers(hotels: list[dict]) -> dict[str, list[dict]]:
    """
    Groups hotels into price tiers relative to the lowest nightly price:
    - Cheap Tier:   Up to 20% more than the lowest price
    - Budget Tier:  Between 75% and 130% of the median price
    - Premium Tier: More than 130% of the median price
    """
    if not hotels:
        return {}

    sorted_hotels = sorted(hotels, key=lambda x: x["price"])
    unique_prices = sorted({h["price"] for h in sorted_hotels})

    # If fewer than 3 distinct prices, 3 tiers cannot meaningfully exist
    if len(unique_prices) <= 2:
        return {"Available Hotels": sorted_hotels}

    middle_index = len(sorted_hotels) // 2
    median_price = sorted_hotels[middle_index]["price"]

    cheap_cutoff  = median_price * 0.75
    budget_cutoff = median_price * 1.30

    tiers = {"Cheap Tier": [], "Budget Tier": [], "Premium Tier": []}
    for h in sorted_hotels:
        price = h["price"]
        if price <= cheap_cutoff:
            tiers["Cheap Tier"].append(h)
        elif price <= budget_cutoff:
            tiers["Budget Tier"].append(h)
        else:
            tiers["Premium Tier"].append(h)

    # Fallback: if any tier is empty, split by unique price thirds instead
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
    name          = hotel.get("name", "Unknown Hotel")
    price         = hotel.get("price")
    price_str     = f"{price:,.0f}" if isinstance(price, (int, float)) else "N/A"
    hotel_class   = hotel.get("hotel_class", "Unclassified")
    property_type = hotel.get("type", "hotel").title()
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
    star_rating: int | list[int] = None,
    food_preference: str = None
):
    in_date_obj  = datetime.strptime(check_in_date,  "%Y-%m-%d")
    out_date_obj = datetime.strptime(check_out_date, "%Y-%m-%d")
    if out_date_obj <= in_date_obj:
        print("Error: check_out_date must be after check_in_date.")
        return

    print(
        f"🏨 Searching hotels in '{location_query}' "
        f"({check_in_date} to {check_out_date}) for {adults} adults"
        f"{f' [Stars: {star_rating}]' if star_rating else ''}"
        f"{f' [Food: {food_preference}]' if food_preference else ''}...\n"
    )

    main_hotels = fetch_hotels(location_query, check_in_date, check_out_date, adults, currency)
    if not main_hotels:
        print("No hotels found for the selected dates.")
        return

    # Apply star rating and food preference filters (tracking which filters matched)
    star_matched = False
    food_matched = False

    if star_rating:
        main_hotels, star_matched = filter_by_star_rating(main_hotels, star_rating, allow_fallback=True)

    if food_preference:
        main_hotels, food_matched = filter_by_food_preference(main_hotels, food_preference, allow_fallback=True)

    # 🛑 CRITICAL SAFETY CHECK: Stop if filters wiped out all results
    if not main_hotels:
        print("❌ No hotels match your combined star rating and food preference filters for these dates.")
        return

    # Check adjacent dates (+/- 1 day shift) for cheaper nightly rates
    adjacent_shifts = [
        ("1 day earlier",
         (in_date_obj  - timedelta(days=1)).strftime("%Y-%m-%d"),
         (out_date_obj - timedelta(days=1)).strftime("%Y-%m-%d")),
        ("1 day later",
         (in_date_obj  + timedelta(days=1)).strftime("%Y-%m-%d"),
         (out_date_obj + timedelta(days=1)).strftime("%Y-%m-%d"))
    ]

    min_main_price = min(h["price"] for h in main_hotels)

    for shift_label, adj_in, adj_out in adjacent_shifts:
        adj_hotels = fetch_hotels(location_query, adj_in, adj_out, adults, currency)

        # Apply strict filtering ONLY for filters that actually matched on the main date
        if star_matched:
            adj_hotels, _ = filter_by_star_rating(adj_hotels, star_rating, allow_fallback=False)
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
        shown_count   = min(3, total_in_tier)

        print(f"=== {tier_name.upper()} (Showing {shown_count} of {total_in_tier} options) ===")
        if not hotel_list:
            print("  No hotels in this price range.\n")
            continue

        for hotel in hotel_list[:3]:
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
