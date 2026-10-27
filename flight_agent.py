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


def format_flight_summary(itinerary: dict, currency: str = "INR") -> str:
    """Formats a single flight itinerary for display."""
    price          = itinerary.get("price", "N/A")
    total_duration = itinerary.get("total_duration", 0)
    hours, mins    = divmod(total_duration, 60)

    connecting_flights = itinerary.get("flights", [])
    if not connecting_flights:
        return f"Price: {currency} {price}"

    first_flight = connecting_flights[0]
    last_flight  = connecting_flights[-1]

    airlines = ", ".join(sorted(set(flight.get("airline", "") for flight in connecting_flights)))
    dep_time = first_flight.get("departure_airport", {}).get("time", "N/A")
    arr_time = last_flight.get("arrival_airport", {}).get("time", "N/A")

    return (
        f"• {airlines} | {currency} {price:,} | Total Time: {hours}h {mins}m\n"
        f"  Departs: {dep_time} -> Arrives: {arr_time}"
    )


def run_flight_agent(dep_id: str, arr_id: str, target_date: str, currency: str = "INR"):
    print(f"🔎 Searching flights from {dep_id} to {arr_id} on {target_date}...\n")

    main_flights = fetch_flights(dep_id, arr_id, target_date, currency)
    if not main_flights:
        print("No flights found for the selected date.")
        return

    price_tiers = group_into_tiers(main_flights)

    for tier_name, flight_list in price_tiers.items():
        total_in_tier = len(flight_list)
        shown_count   = min(3, total_in_tier)

        print(f"=== {tier_name.upper()} (Showing {shown_count} of {total_in_tier} options) ===")
        if not flight_list:
            print("  No flights in this price range.\n")
            continue

        for itinerary in flight_list[:3]:
            print(format_flight_summary(itinerary, currency))
            print("-" * 50)
        print()


if __name__ == "__main__":
    run_flight_agent(
        dep_id="MYQ",
        arr_id="MAA",
        target_date="2026-10-29",
        currency="INR"
    )


