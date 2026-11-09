# Smart Travel Assistant 🛫🏨

An AI-powered travel assistant for searching flights and hotels using Google Flights and Google Hotels via SerpApi. It provides an interactive command-line interface, groups results into price tiers, discovers cheaper deals on adjacent dates, analyzes layover safety, and filters hotels by quality, star class, and dietary preferences.

---

## 🌟 Key Features

### ✈️ Flight Search Agent (`flight_agent.py`)
- **Interactive CLI Wizard**: Step-by-step interactive CLI with real-time input validation and friendly prompts.
- **Two Search Modes**:
  - **Specific Destination**: Search one-way or round-trip flights between any two airports.
  - **Explore Mode**: Compare multiple candidate destinations concurrently (multi-threaded) and filter to those within your budget.
- **Flight Preferences**:
  - Filter by stops: Nonstop only, up to 1 stop, or all flight options.
  - Round-trip support with departure and return date tracking.
- **Price Tier Grouping**: Automatically clusters results into **Cheap / Budget / Premium** tiers using configurable tier multipliers.
- **💡 Flexible Date Deals**: Suggests cheaper alternatives by checking adjacent dates (±1 day) when savings exceed 15%.
- **Connection & Layover Analysis**:
  - ⚠️ Flags tight connections (< 1 hour).
  - ⏳ Warns about excessively long layovers (> 6 hours).
  - 🌙 Detects overnight layovers.
  - 🛂 Identifies transit visa notices and airport changes between connecting legs.

### 🏨 Hotel Search Agent (`hotel_agent.py`)
- **Interactive CLI Wizard**: Guides destination, dates, guests, currency, star rating, and food preferences.
- **Price Tier Grouping**: Groups properties into **Cheap / Budget / Premium** tiers using median-anchored thresholds.
- **⭐ Flexible Star Rating Filter**: Filter for a minimum star rating (e.g., 4★ and above) or exact 5-star luxury stays.
- **🍽️ Dietary & Food Preference Filter**: Smart word-boundary regex matching across hotel names, descriptions, reviews, and amenities for preferences like vegetarian, non-vegetarian, halal, vegan, jain, kosher, and gluten-free.
- **💡 Flexible Stay Savings**: Checks shifting stay dates by ±1 day to discover cheaper nightly rates.
- **Quality & Amenity Highlights**:
  - ⭐ Flags top-rated hotels (≥ 4.3 / 5.0) and warns against low-rated or unrated options.
  - 🛎️ Displays amenities with clean overflow indicators (`+ N more amenities`).
  - 🕒 Lists check-in and check-out policies.

### 🛡️ Built-in Input Validation & Safety
- **Airport Codes**: Strict 3-letter IATA alphabetic validation (`is_valid_code`). Prevents identical departure and arrival codes.
- **Date Guardrails**: Validates `YYYY-MM-DD` calendar dates, prevents past dates, ensures return date > departure date, and respects the 330-day airline/hotel booking window (`MAX_ADVANCE_DAYS`).
- **Hotel Stays**: Validates check-out > check-in and caps bookings to reasonable stay durations (1 to 30 nights).
- **Guest & Budget Constraints**: Enforces positive guest counts (1–20 adults) and positive finite numbers for budgets.
- **Lazy API Initialization**: Checks `SECRET_API_KEY` when executing searches rather than failing abruptly on import.
- **Graceful Exit**: Cleanly handles `Ctrl+C` (`KeyboardInterrupt`) and `EOFError` with friendly farewells without stack traces.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install serpapi python-dotenv
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
# Required: SerpApi API key (get one at serpapi.com)
SECRET_API_KEY=your_serpapi_key_here

# Optional defaults
DEFAULT_CURRENCY=INR
EXPLORE_AIRPORTS=BLR,DEL,DXB,SIN
```

### 3. Run Interactively

Launch either assistant to start the interactive CLI:

**Flight Search:**
```bash
python flight_agent.py
```

**Hotel Search:**
```bash
python hotel_agent.py
```

---

## 💻 Programmatic Usage

Both agents can also be imported and run directly inside your own Python scripts:

### Flight Search

```python
from flight_agent import run_flight_agent, explore_destinations

# 1. Search specific route (Round-trip, Nonstop)
run_flight_agent(
    dep_id="CCJ",
    arr_id="DXB",
    target_date="2026-11-15",
    currency="INR",
    trip_type="1",          # "1" = Round Trip, "2" = One-Way
    stops="1",              # "1" = Nonstop, "2" = Up to 1 stop, "0" = Any stops
    return_date="2026-11-20"
)

# 2. Explore multiple destinations within budget
explore_destinations(
    dep_id="CCJ",
    candidate_airports=["BLR", "DEL", "DXB", "SIN"],
    date_str="2026-11-15",
    budget=25000,
    currency="INR",
    trip_type="2",          # One-Way
    stops="0"               # Any stops
)
```

### Hotel Search

```python
from hotel_agent import run_hotel_agent

run_hotel_agent(
    location_query="Bengaluru",
    check_in_date="2026-11-15",
    check_out_date="2026-11-17",
    adults=2,
    currency="INR",
    min_star=4,                   # Optional: minimum star rating (e.g. 4 = 4★ and above)
    only_exact_star=False,        # Set True for exact 5★ only
    food_preference="vegetarian"  # Optional: dietary keyword (vegetarian, halal, vegan, etc.)
)
```

---

## ⚙️ Configuration & Environment Variables

All parameters can be customized via `.env` or overridden as constants:

| Variable | Default | Description |
|---|---|---|
| `SECRET_API_KEY` | *(None)* | SerpApi secret key (required) |
| `DEFAULT_CURRENCY` | `INR` | Default 3-letter currency code |
| `MAX_ADVANCE_DAYS` | `330` | Maximum advance booking window in days |
| `SIGNIFICANT_SAVINGS_RATIO` | `0.85` | Ratio below main price to trigger flexible date deal (15%+ savings) |
| `MIN_SAFE_LAYOVER_MINS` | `60` | Minimum connection time to avoid tight connection warning |
| `MAX_COMFORT_LAYOVER_MINS` | `360` | Layover duration after which a long layover warning is shown |
| `FLIGHT_CHEAP_MULTIPLIER` | `1.20` | Multiplier above lowest fare for flight Cheap tier |
| `FLIGHT_BUDGET_MULTIPLIER` | `1.40` | Multiplier above lowest fare for flight Budget tier |
| `MAX_EXPLORE_AIRPORTS` | `10` | Maximum destination airports queried in explore mode |
| `EXPLORE_AIRPORTS` | `""` | Comma-separated fallback airport codes for explore mode |
| `MIN_SAFE_RATING` | `3.5` | Minimum acceptable hotel guest rating |
| `HIGH_RATING_THRESHOLD` | `4.3` | Rating threshold to highlight top-rated hotels |
| `HOTEL_CHEAP_MEDIAN_FACTOR` | `0.75` | Median multiplier for hotel Cheap tier |
| `HOTEL_BUDGET_MEDIAN_FACTOR` | `1.30` | Median multiplier for hotel Budget tier |
| `MAX_DISPLAY_PER_TIER` | `3` | Maximum results displayed per price tier |
| `MAX_AMENITIES_SHOWN` | `6` | Maximum amenities shown before overflow count |
| `DEFAULT_ADULTS` | `2` | Default number of adults for hotel searches |
| `MAX_ADULTS_LIMIT` | `20` | Maximum supported adults in hotel search |
| `MAX_STAY_NIGHTS` | `30` | Maximum supported consecutive hotel stay nights |


---

### 🛡️ Process-Wide Rate Limiting & Provider Safety (`search_limits.py`)
- **Sliding-Window Rate Limiter**: Thread-safe `ProviderRateLimiter` managing calls per minute (`SERPAPI_MAX_CALLS_PER_MINUTE`) and daily call quotas (`SERPAPI_MAX_CALLS_PER_DAY`).
- **Quota Protection**: Throws `SearchBudgetExceeded` when rate thresholds are approached to avoid unexpected subscription billing.
- **Flight Agent Resilience**: Integrated with 350+ valid commercial airport IATA codes (`VALID_IATA_CODES`) and an in-memory `BoundedTTLCache` (500 items, 900s TTL).


---

### 🥗 Context-Aware Dietary Negation & Quality Engine (`hotel_agent.py`)
- **Negation Detection**: Advanced regex `_has_negative_dietary_statement` detects negation prefixes (`not`, `no`, `non-`, `without`) and suffixes (`unavailable`, `not served`, `pork`) across hotel amenities.
- **Strict Non-Fallback Policy**: Completely disables falling back to unverified hotels when strict dietary preferences (halal, kosher, jain, vegan, vegetarian) are specified.
- **Name Exclusion**: Prevents hotel names from being mistaken for dietary proof to avoid false positives.
- **Memory LRU Cache**: Protected by thread-safe `BoundedTTLCache` to reduce duplicate requests.


---

### 🔒 Deterministic Intent Router & Safety Overrides (`intent_router.py`, `travel_scope_rules.py`)
- **Secret Exfiltration Defense**: Intercepts requests attempting to expose `.env` secrets, API keys, or access tokens.
- **Safety Refusals**: Blocks inquiries regarding hazardous chemicals, weapons, and concealment/smuggling requests.
- **Airline Compliance Guidance**: Provides policy-aligned answers for baggage weight limits and ticket evasion.
- **Emergency Travel Assistance**: Identifies lost passport/visa emergencies and routes to official diplomatic assistance.
- **Scope Boundary Policy**: Separates generic cooking recipes from local destination food culture.


---

### 🚦 Deterministic Routing & Conversational Guardrails (`routing_rules.py`, `request_guards.py`, `conversation_rules.py`)
- **Terminal Command Interception**: Blocks pasted terminal commands (`pip`, `python -m`, `npm`, `git`, `docker`).
- **Code & SQL Injection Blocker**: Detects pasted source code blocks and raw SQL statements (`SELECT`, `INSERT`, `DROP`).
- **Language vs Destination Disambiguation**: Differentiates programming languages from travel destinations (e.g. Java island vs Java language; Goa vs Go).
- **Search Confirmation Gates**: Requires affirmative user consent before triggering billable provider searches.
- **Conversational Rules**: Handles session farewells, detail expansion requests, and conversational history recall.


---

### 🤖 Enterprise LangGraph Multi-Agent Supervisor (`travel_langgraph_agent.py`)
- **LangGraph StateGraph Engine**: Coordinates Master Supervisor, Flight Specialist, Hotel Specialist, and General Assistant nodes.
- **Dynamic Calendar Grounding**: Local timezone calendar context (`Asia/Kolkata`) with a 14-day upcoming reference table to eliminate relative date hallucinations (*"coming friday"*, *"next week"*).
- **Mandatory Clarification Checklists**:
  - Flight checklist: Mandates explicit confirmation of departure, destination, date, and **trip type (one-way vs round-trip + return date)** before search execution.
  - Hotel checklist: Confirms destination, check-in, check-out, and guest counts before invoking tools.
- **Atomic Tool History Trimming**: Preserves complete `AIMessage(tool_calls)` + `ToolMessage` cycles, eliminating orphaned message errors in LangGraph.
- **External Data Boundaries**: Untrusted provider results are enclosed in `[EXTERNAL SEARCH RESULT - INFORMATIONAL USE ONLY]` delimiters.
- **Streaming CLI**: Dedicated speaker labels (`✈️ Flight Specialist`, `🏨 Hotel Specialist`, `🤖 Travel Assistant`).
