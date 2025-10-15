# CS2 Match Data Scraper Project

This project scrapes CS2 (Counter-Strike 2) competitive match data from BO3.gg API to create a dataset for machine learning analysis.

## Files Overview

### Core Files
- **`map_picks_last6m_top50.csv`** - Final dataset (2,430 records from 1,032 matches)
- **`cs2_match_data_scraper.py`** - Main data scraper script
- **`current_top_50_teams.json`** - Top 50 teams ranked by earnings (last 6 months)
- **`requirements.txt`** - Python dependencies

### Supporting Files
- **`cs2api/`** - Patched CS2 API library for BO3.gg integration
- **`Phase#1-Guidelines.pdf`** - Original project requirements

## Dataset Details

- **Timeframe**: Last 6 months (March 2025 - September 2025)
- **Teams**: Top 50 teams by earnings-based rankings
- **Records**: 2,412 played maps from 1,032 unique matches
- **Filters**: 
  - Only finished tournaments (no ongoing events)
  - Only matches with complete veto/pick-ban data
  - Only actually played maps (removes unplayed maps from BO5 early endings)
  - Automatic deduplication to prevent duplicate matches
- **Format**: CSV with 10 columns including team names, map picks/bans, results, and metadata

### Match Format Distribution
- **BO1 matches**: 91 matches (1 map each)
- **BO3 matches**: 929 matches (2-3 maps each)  
- **BO5 matches**: 12 matches (3-5 maps each)

### Dataset Columns
1. `match_id` - Unique match identifier
2. `team1` - First team name
3. `team2` - Second team name  
4. `map` - Map name (Mirage, Nuke, Dust2, etc.)
5. `pick_ban_action` - Action taken (pick, ban, decider, leftover)
6. `team_action` - Which team took the action (team1/team2)
7. `winner` - Match winner (team1/team2)
8. `map_winner` - Map winner (team1/team2)
9. `date` - Match date (YYYY-MM-DD)
10. `map_was_played` - Boolean indicating if map was actually played

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Scraper
```bash
python cs2_match_data_scraper.py
```

This will:
1. Load current top 50 team rankings
2. Fetch matches for each team from the last 6 months
3. Filter for finished tournaments only
4. Extract map pick/ban data with deduplication
5. Save results to `map_picks_last6m_top50.csv`

### Load the Dataset
```python
import pandas as pd
df = pd.read_csv('map_picks_last6m_top50.csv')
print(f"Dataset contains {len(df)} records from {df['match_id'].nunique()} unique matches")
```

## ML Dataset Transformation

After scraping the raw match data, use `create_ml_dataset.py` to transform it into a machine learning-ready format with engineered features and proper train/validation/test splits.

### Create ML-Ready Dataset
```bash
python create_ml_dataset.py
```

This transformation script performs several critical preprocessing steps:

**Feature Engineering:**
- **Team normalization**: Standardizes team names for consistency across the dataset
- **Chronological winrates**: Calculates map-specific and overall winrates using only historical data to prevent leakage
- **Recent form**: Computes team performance over the last 10 matches chronologically
- **Pick/ban context**: Creates binary indicators for map picking team and decider maps
- **Rank differences**: Adds team ranking differentials and absolute differences for relative strength assessment

**Data Leakage Prevention:**
- All features are calculated chronologically using only past information
- Map winrates and recent form use strict temporal ordering
- Train/validation/test splits respect match-level boundaries to prevent information leakage

**Output Format:**
- **Input file**: `map_picks_last6m_top50.csv` (raw scraped data)
- **Output file**: `map_picks_last6m_top50_ml_ready.csv` (ML-ready format)
- **Features added**: ~12 engineered columns including winrates, form metrics, and contextual indicators
- **Target variable**: `winner_is_A` (binary: 1 if team_A wins the map, 0 otherwise)
- **Data splits**: Both row-level (`split_row`) and match-level (`split_match`) partitions for different evaluation strategies

The resulting ML-ready dataset maintains the same number of records while adding rich contextual features that capture team strengths, map preferences, and temporal dynamics essential for accurate win probability prediction.

## Dataset Extension and Updates

Use `extend_dataset.py` to automatically detect new finished tournaments and extend both raw and ML-ready datasets incrementally.

### Extend Existing Datasets
```bash
python extend_dataset.py
```

This extension script provides automated dataset maintenance:

**New Tournament Detection:**
- Automatically checks for new finished tournaments since the last dataset update
- Cross-references existing match IDs to avoid duplicates
- Filters for tournaments with complete pick/ban data and verified results

**Incremental Processing:**
- Extends the raw dataset with new match records using the same format and quality standards
- Automatically regenerates the ML-ready dataset with updated chronological features
- Maintains proper temporal ordering for feature calculation (winrates, recent form)
- Preserves existing train/validation/test splits while adding new data appropriately

**Data Safety:**
- Creates backup copies of original datasets before modification
- Comprehensive logging and error handling for reliable operation
- Validation checks to ensure data consistency after extension

**Smart Updates:**
- Only processes truly new matches to minimize API calls and processing time
- Maintains the same feature engineering pipeline for consistency
- Updates both `map_picks_last6m_top50.csv` and `map_picks_last6m_top50_ml_ready.csv` files

Run this script periodically (weekly or monthly) to keep your dataset current with the latest professional CS2 match results while maintaining all the quality controls and feature engineering of the original dataset creation process.

## Data Quality Features

- **Earnings-based rankings**: Uses live API data for current team rankings
- **Tournament filtering**: Only includes completed tournaments for stable results
- **Deduplication**: Prevents same match from being processed multiple times
- **Data validation**: Skips matches without complete veto information
- **Rate limiting**: Includes delays to respect API limits

## Technical Notes

- Built with `cs2api` library (locally patched version included)
- Async processing for efficient API calls
- Comprehensive logging and error handling
- Handles various date formats and edge cases
- Supports both BO1 and BO3 match formats