import os
import serpapi
from dotenv import load_dotenv

load_dotenv()

secret_api_key = os.getenv("SECRET_API_KEY")
if not secret_api_key:
    print("Error: SECRET_API_KEY is not set.")
    exit()

client = serpapi.Client(api_key=secret_api_key)


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
