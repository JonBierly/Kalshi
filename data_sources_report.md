# NBA Data Sources Report

## 1. Historical Data Acquisition
### Strategy
The recommended strategy for acquiring historical NBA play-by-play (PBP) data is to use the `nba_api` Python package, which wraps the official NBA Stats API.

### Tools
- **Primary Tool**: `nba_api` (Python package)
- **Endpoints**: 
    - `PlayByPlayV2` (Primary, detailed)
    - `PlayByPlay` (Legacy/Fallback)
    - `LeagueGameFinder` (For discovering game IDs)

### Implementation Details
The provided `data_acquisition.py` module implements a `HistoricalDataClient` that:
1.  Fetches game IDs for a season.
2.  Iterates through games to fetch PBP data.
3.  Includes retry logic and fallback mechanisms (V2 -> V1) to handle API instability.
4.  Standardizes the raw JSON response into a clean pandas DataFrame.

**Note**: During development, we encountered API timeouts/blocks from `stats.nba.com`. A `MockHistoricalDataClient` was implemented to simulate data flow and unblock the pipeline development. For production, you may need to use a commercial proxy or a paid data provider to ensure reliable access to `stats.nba.com`.

## 2. Live Data API
### Requirements
- **Latency**: < 1 second ideally.
- **Reliability**: High uptime during games.
- **Granularity**: Play-by-play events with timestamps.

### Recommendations
1.  **Sportradar (Official)**:
    - **Pros**: Fastest, official source, extremely detailed.
    - **Cons**: Expensive enterprise pricing.
    - **Verdict**: Best for professional market making.

2.  **SportsDataIO**:
    - **Pros**: Developer-friendly, good latency, tiered pricing.
    - **Cons**: Slightly slower than Sportradar.
    - **Verdict**: Good balance for serious algorithmic trading.

3.  **RapidAPI Options (e.g., API-NBA)**:
    - **Pros**: Cheap/Free tiers.
    - **Cons**: High latency, often just scrapes official sites, unreliable for HFT.
    - **Verdict**: Not recommended for market making.

### Simulation Strategy
To facilitate development without an expensive subscription, we implemented a `ReplayDataClient` in `data_acquisition.py`. This class takes historical game data and "streams" it event-by-event, simulating a live feed. This allows for end-to-end testing of the feature engineering and prediction pipeline.
