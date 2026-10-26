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


if __name__ == "__main__":
    flights = fetch_flights("MYQ", "MAA", "2026-10-29", "INR")
    for f in flights:
        print(f"Price: {f.get('price')} | Duration: {f.get('total_duration')} mins")
