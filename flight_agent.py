import os
import serpapi
from dotenv import load_dotenv

load_dotenv()

secret_api_key = os.getenv("SECRET_API_KEY")
if not secret_api_key:
    print("Error: SECRET_API_KEY is not set.")
    exit()

client = serpapi.Client(api_key=secret_api_key)


def fetch_flights(dep_id: str, arr_id: str, date_str: str, currency: str = "INR"):
    """Fetches and combines both 'best_flights' and 'other_flights' from SerpApi."""
    params = {
        "engine": "google_flights",
        "departure_id": dep_id,
        "arrival_id": arr_id,
        "currency": currency,
        "type": "2",  # One-way
        "outbound_date": date_str
    }
    results = client.search(params)

    if "error" in results:
        print(f"API Error for {date_str}: {results['error']}")
        return []

    # Combine best_flights and other_flights so we don't miss cheap options
    all_flights = results.get("best_flights", []) + results.get("other_flights", [])

    # Filter out any entries missing a valid price
    valid_flights = [f for f in all_flights if isinstance(f.get("price"), (int, float))]
    return valid_flights


def group_into_tiers(flights: list[dict]) -> dict[str, list[dict]]:
    """
    Groups flights into price tiers relative to the lowest price:
    - Cheap Tier:   Up to 20% more than the lowest price
    - Budget Tier:  More than 20% up to 40% higher than the lowest price
    - Premium Tier: More than 40% higher than the lowest price
    """
    if not flights:
        return {}

    sorted_flights = sorted(flights, key=lambda x: x["price"])

    if len(sorted_flights) <= 2:
        return {"Available Flights": sorted_flights}

    min_price = sorted_flights[0]["price"]
    max_price = sorted_flights[-1]["price"]

    if min_price == max_price:
        return {"Available Flights": sorted_flights}

    cheap_cutoff  = min_price * 1.20
    budget_cutoff = min_price * 1.40

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


if __name__ == "__main__":
    flights = fetch_flights("MYQ", "MAA", "2026-10-29", "INR")
    tiers = group_into_tiers(flights)
    for tier, items in tiers.items():
        print(f"\n{tier}: {len(items)} flights")
        for f in items:
            print(f"  Price: INR {f.get('price'):,}")

