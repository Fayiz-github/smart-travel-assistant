import os
import serpapi
from dotenv import load_dotenv

load_dotenv()

secret_api_key = os.getenv("SECRET_API_KEY")
if not secret_api_key:
    print("Error: SECRET_API_KEY is not set.")
    exit()

client = serpapi.Client(api_key=secret_api_key)

# --- CONFIGURATION ---
MIN_SAFE_RATING      = 3.5   # Warn if guest rating is below 3.5 / 5.0
HIGH_RATING_THRESHOLD = 4.3  # Highlight if guest rating is 4.3 or higher

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



def extract_nightly_price(property_item: dict) -> float | None:
    """Safely extracts the numerical nightly price from a SerpApi hotel property."""
    rate_info = property_item.get("rate_per_night", {})

    # SerpApi provides 'extracted_lowest' as a clean number (int or float)
    price = rate_info.get("extracted_lowest")
    if isinstance(price, (int, float)):
        return price

    # Fallback to total_rate if rate_per_night is missing
    total_info  = property_item.get("total_rate", {})
    total_price = total_info.get("extracted_lowest")
    if isinstance(total_price, (int, float)):
        return total_price

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

    raw_properties = results.get("properties", [])
    valid_hotels   = []

    for prop in raw_properties:
        price = extract_nightly_price(prop)
        if price is not None:
            prop["price"] = price  # Store clean price at top level
            valid_hotels.append(prop)

    return valid_hotels


if __name__ == "__main__":
    hotels = fetch_hotels("Bengaluru", "2026-11-15", "2026-11-17", 2, "INR")
    for h in hotels:
        print(f"{h.get('name')} | INR {h.get('price'):,}/night")
