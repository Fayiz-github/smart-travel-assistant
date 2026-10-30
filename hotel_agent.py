import os
from datetime import datetime, timedelta
import serpapi
from dotenv import load_dotenv

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


def group_into_tiers(hotels: list[dict]) -> dict[str, list[dict]]:
    """
    Groups hotels into price tiers relative to the lowest nightly price:
    - Cheap Tier:   Up to 20% more than the lowest price
    - Budget Tier:  More than 20% up to 40% higher than the lowest price
    - Premium Tier: More than 40% higher than the lowest price
    """
    if not hotels:
        return {}

    sorted_hotels = sorted(hotels, key=lambda x: x["price"])

    if len(sorted_hotels) <= 2:
        return {"Available Hotels": sorted_hotels}

    min_price = sorted_hotels[0]["price"]
    max_price = sorted_hotels[-1]["price"]

    if min_price == max_price:
        return {"Available Hotels": sorted_hotels}

    cheap_cutoff  = min_price * 1.20
    budget_cutoff = min_price * 1.40

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
    price         = hotel.get("price", "N/A")
    hotel_class   = hotel.get("hotel_class", "Unclassified")
    property_type = hotel.get("type", "hotel").title()
    total_rate_str = hotel.get("total_rate", {}).get("lowest", "N/A")

    hotel_notes = evaluate_hotel_details(hotel)
    notes_str   = "\n      ".join(hotel_notes)

    return (
        f"• {name} ({hotel_class} {property_type}) | {currency} {price:,}/night (Total: {total_rate_str})\n"
        f"  Hotel Notes:\n      {notes_str}"
    )


def run_hotel_agent(
    location_query: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    currency: str = "INR"
):
    print(f"🏨 Searching hotels in '{location_query}' ({check_in_date} to {check_out_date}) for {adults} adults...\n")

    main_hotels = fetch_hotels(location_query, check_in_date, check_out_date, adults, currency)
    if not main_hotels:
        print("No hotels found for the selected dates.")
        return

    # Check adjacent dates (+/- 1 day shift) for cheaper nightly rates
    in_date_obj  = datetime.strptime(check_in_date,  "%Y-%m-%d")
    out_date_obj = datetime.strptime(check_out_date, "%Y-%m-%d")

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
        if adj_hotels:
            min_adj_price = min(h["price"] for h in adj_hotels)
            if min_adj_price <= min_main_price * SIGNIFICANT_SAVINGS_RATIO:
                savings = min_main_price - min_adj_price
                print(
                    f"💡 FLEXIBLE DATE TIP: Shifting your stay {shift_label} ({adj_in} to {adj_out}) "
                    f"starts at {currency} {min_adj_price:,}/night "
                    f"(Save {currency} {savings:,}/night vs your chosen dates)!\n"
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
        currency="INR"
    )

