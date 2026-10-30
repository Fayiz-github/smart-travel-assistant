# Smart Travel Assistant 🛫🏨

An AI-powered travel assistant for searching flights and hotels using Google Flights and Google Hotels via SerpApi. It groups results by price tier, checks for better deals on adjacent dates, analyzes layover safety, and evaluates hotel quality — all from a simple Python script.

## Features

### ✈️ Flight Search
- Fetches live flights via SerpApi Google Flights
- Groups results into **Cheap / Budget / Premium** price tiers
- 💡 Suggests cheaper fares on adjacent days (±1 day)
- ⚠️ Flags tight connections (< 1 hour layover)
- ⏳ Warns about excessively long layovers (> 6 hours)
- 🌙 Detects overnight layovers
- 🛂 Alerts for transit visa requirements at major international hubs

### 🏨 Hotel Search
- Fetches live hotels via SerpApi Google Hotels
- Groups results into **Cheap / Budget / Premium** tiers (median-anchored pricing)
- 💡 Suggests cheaper rates by shifting stay ±1 day
- ⭐ Highlights top-rated properties (≥ 4.3 / 5.0)
- ⚠️ Flags low-rated or unrated properties
- 🛎️ Lists included perks (Wi-Fi, breakfast, pool, parking, etc.)
- 🕒 Shows check-in / check-out times
- ⭐ Filter by star rating (e.g. 4-star and 5-star only)
- 🍽️ Filter by food preference (veg, non-veg, halal)
- 🛡️ Date validation to prevent invalid search ranges

## Setup

### 1. Install dependencies
```bash
pip install serpapi python-dotenv
```

### 2. Configure API key
Create a `.env` file in the project root:
```
SECRET_API_KEY=your_serpapi_key_here
```
Get your key at [serpapi.com](https://serpapi.com).

### 3. Run

**Flight search:**
```bash
python flight_agent.py
```

**Hotel search:**
```bash
python hotel_agent.py
```

## Configuration

You can tune thresholds at the top of each file:

| Constant | Default | Description |
|---|---|---|
| `MIN_SAFE_LAYOVER_MINS` | 60 | Minimum safe connection time (minutes) |
| `MAX_COMFORT_LAYOVER_MINS` | 360 | Maximum comfortable layover (minutes) |
| `SIGNIFICANT_SAVINGS_RATIO` | 0.85 | Threshold to flag adjacent date savings (15%+) |
| `MIN_SAFE_RATING` | 3.5 | Minimum acceptable hotel rating |
| `HIGH_RATING_THRESHOLD` | 4.3 | Rating to flag a hotel as top-rated |

## Hotel Search Options

```python
run_hotel_agent(
    location_query="Bengaluru",
    check_in_date="2026-11-15",
    check_out_date="2026-11-17",
    adults=2,
    currency="INR",
    star_rating=[4, 5],       # Optional: filter by star class
    food_preference="veg"     # Optional: veg / non-veg / halal
)
```
