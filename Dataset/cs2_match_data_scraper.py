"""
Counter-Strike 2 Match Data Scraper - Enhanced ML-Ready Version

This script collects map pick/ban and match result data from the last 6 months
for the top 50 teams using the cs2api library (BO3.gg data) and creates
a feature-rich dataset ready for machine learning.

Author: GitHub Copilot  
Date: September 20, 2025
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import time
import logging
import asyncio
import json
from typing import List, Dict, Optional
from cs2api import CS2
from collections import defaultdict
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cs2_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CS2MatchDataScraper:
    def __init__(self):
        """Initialize the enhanced scraper with CS2 API client."""
        self.client = None  # Will be initialized in async context
        self.cutoff_date = dt.datetime.now() - timedelta(days=180)  # 6 months ago
        self.data_rows = []
        self.team_rankings = {}
        self.team_match_history = defaultdict(list)  # For calculating winrates
        self.team_map_history = defaultdict(lambda: defaultdict(list))  # For map winrates
        
        # Current top teams based on API rankings (fallback list)
        self.top_teams_names = [
            "Spirit", "Vitality", "The MongolZ", "MOUZ", "Falcons", "Aurora", "NAVI", "3DMAX", 
            "TYLOO", "FURIA", "Astralis", "FaZe", "Lynn Vision", "HEROIC", "G2", "GamerLegion",
            "paiN", "Virtus.pro", "Legacy", "BetBoom", "fnatic", "SAW", "B8", "NIP",
            "Liquid", "MIBR", "ENCE", "OG", "ECSTATIC", "Passion UA", "Imperial", "PARIVISION",
            "Rare Atom", "M80", "Nemiga", "Monte", "BIG", "Fluxo", "9z", "Sashi",
            "BESTIA", "Zero Tenacity", "Wildcard", "Sharks", "SINNERS", "RED Canids", 
            "Sangal", "Metizport", "Eternal Fire", "GUN5"
        ]
    
    def normalize_team_name(self, team_name: str) -> str:
        """Normalize team names for consistency."""
        if not team_name:
            return ""
        
        # Remove extra spaces and convert to title case
        normalized = team_name.strip()
        
        # Handle special cases for consistent naming
        name_mapping = {
            "natus vincere": "Natus Vincere",
            "ninjas in pyjamas": "Ninjas in Pyjamas",
            "the mongolz": "The MongolZ", 
            "mongolz": "The MongolZ",
            "lynn vision gaming": "Lynn Vision",
            "red canids": "RED Canids",
            "eternal fire": "Eternal Fire",
            "zero tenacity": "Zero Tenacity",
            "heroic": "HEROIC",
            "tyloo": "TYLOO",
            "mouz": "MOUZ",
            "furia esports": "FURIA",
            "furia": "FURIA",
            "betboom team": "BetBoom",
            "betboom": "BetBoom",
            "g2 esports": "G2",
            "faze clan": "FaZe",
            "team spirit": "Spirit",
            "team vitality": "Vitality",
            "team liquid": "Liquid",
            "mibr": "MIBR",
            "ence": "ENCE",
            "big": "BIG",
            "saw": "SAW",
            "og": "OG",
            "b8": "B8"
        }
        
        # Check if we have a specific mapping
        lower_name = normalized.lower()
        if lower_name in name_mapping:
            return name_mapping[lower_name]
        
        # Default to title case
        return normalized.title()
        
    async def get_top_50_teams(self) -> List[Dict]:
        """
        Get the top 50 teams using current rankings from the API.
        
        Returns:
            List of team dictionaries with id, name, and rank
        """
        logger.info("Fetching top 50 teams with current rankings...")
        
        # Try to load current rankings from file first
        try:
            import json
            import os
            
            rankings_file = "current_top_50_teams.json"
            if os.path.exists(rankings_file):
                with open(rankings_file, 'r') as f:
                    current_teams = json.load(f)
                
                logger.info(f"Loaded {len(current_teams)} teams from saved rankings file")
                
                # Convert to the expected format and populate team_rankings
                top_teams = []
                for team in current_teams:
                    team_info = {
                        'id': team.get('id'),
                        'name': team.get('name'),
                        'slug': team.get('slug'),
                        'rank': team.get('rank')
                    }
                    top_teams.append(team_info)
                    self.team_rankings[team.get('id')] = team.get('rank')
                    logger.info(f"Added team: {team_info['name']} (Rank {team_info['rank']}, ID: {team_info['id']})")
                
                logger.info(f"Successfully loaded {len(top_teams)} teams with current rankings")
                return top_teams
                
        except Exception as e:
            logger.warning(f"Could not load saved rankings: {e}")
        
        # Fallback: search for teams individually (slower but works)
        logger.info("Fallback: Searching for teams individually...")
        
        # Current known top teams based on recent rankings
        known_top_teams = [
            "Spirit", "Vitality", "The MongolZ", "MOUZ", "Falcons", "Aurora", "NAVI", "3DMAX", 
            "TYLOO", "FURIA", "Astralis", "FaZe", "Lynn Vision", "HEROIC", "G2", "GamerLegion",
            "paiN", "Virtus.pro", "Legacy", "BetBoom", "fnatic", "SAW", "B8", "NIP",
            "Liquid", "MIBR", "ENCE", "OG", "ECSTATIC", "Passion UA", "Imperial", "PARIVISION",
            "Rare Atom", "M80", "Nemiga", "Monte", "BIG", "Fluxo", "9z", "Sashi",
            "BESTIA", "Zero Tenacity", "Wildcard", "Sharks", "SINNERS", "RED Canids", 
            "Sangal", "Metizport", "Eternal Fire", "GUN5"
        ]
        
        top_teams = []
        
        for i, team_name in enumerate(known_top_teams, 1):
            try:
                search_response = await self.client.search_teams(team_name)
                
                if isinstance(search_response, dict) and 'results' in search_response:
                    results = search_response['results']
                    if results:
                        team = results[0]  # Most relevant result
                        rank = team.get('rank')
                        
                        if rank and isinstance(rank, int) and rank > 0:
                            team_info = {
                                'id': team.get('id'),
                                'name': team.get('name', team_name),
                                'slug': team.get('slug'),
                                'rank': rank
                            }
                            top_teams.append(team_info)
                            self.team_rankings[team.get('id')] = rank
                            logger.info(f"Found team: {team_info['name']} (Rank {rank}, ID: {team_info['id']})")
                
                await asyncio.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error searching for team {team_name}: {e}")
                continue
        
        # Sort by rank and take top 50
        top_teams.sort(key=lambda x: x['rank'])
        top_50 = top_teams[:50]
        
        logger.info(f"Successfully retrieved {len(top_50)} teams with current rankings")
        return top_50
    
    def is_match_within_timeframe(self, match_date: str) -> bool:
        """
        Check if match date is within the last 6 months.
        
        Args:
            match_date: Date string from match data
            
        Returns:
            True if within timeframe, False otherwise
        """
        try:
            # Parse different possible date formats
            for date_format in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']:
                try:
                    parsed_date = dt.datetime.strptime(match_date.split('T')[0], '%Y-%m-%d')
                    return parsed_date >= self.cutoff_date
                except ValueError:
                    continue
            return False
        except Exception:
            return False
    
    def is_tournament_finished(self, match: Dict) -> bool:
        """
        Check if the tournament/series this match belongs to is finished.
        
        Args:
            match: Match dictionary from API
            
        Returns:
            True if tournament is finished, False otherwise
        """
        try:
            # Check if match has a tournament/series context
            series = match.get('series')
            tournament = match.get('tournament')
            
            # Check series status if available
            if series:
                series_status = series.get('status', '').lower()
                # Series is finished if status indicates completion
                if series_status in ['finished', 'completed', 'ended']:
                    return True
                # If series is ongoing, check if enough time has passed
                series_end = series.get('end_date')
                if series_end:
                    try:
                        end_date = dt.datetime.strptime(series_end.split('T')[0], '%Y-%m-%d')
                        # Tournament finished if end date is in the past
                        return end_date < dt.datetime.now()
                    except:
                        pass
            
            # Check tournament status if available
            if tournament:
                tournament_status = tournament.get('status', '').lower()
                if tournament_status in ['finished', 'completed', 'ended']:
                    return True
                tournament_end = tournament.get('end_date')
                if tournament_end:
                    try:
                        end_date = dt.datetime.strptime(tournament_end.split('T')[0], '%Y-%m-%d')
                        return end_date < dt.datetime.now()
                    except:
                        pass
            
            # Fallback: check if match is old enough (assume tournaments finish within 1 week)
            match_date = match.get('start_date', '')
            if match_date:
                try:
                    parsed_date = dt.datetime.strptime(match_date.split('T')[0], '%Y-%m-%d')
                    # Consider finished if match is older than 7 days
                    return (dt.datetime.now() - parsed_date).days > 7
                except:
                    pass
            
            # If we can't determine status, exclude for safety
            return False
            
        except Exception as e:
            logger.debug(f"Error checking tournament status: {e}")
            return False
    
    async def get_team_matches(self, team_id: int, team_name: str) -> List[Dict]:
        """
        Get matches for a specific team within the last 12 months.
        
        Args:
            team_id: Team ID
            team_name: Team name for logging
            
        Returns:
            List of match dictionaries
        """
        logger.info(f"Fetching matches for {team_name} (ID: {team_id})")
        try:
            matches_response = await self.client.get_team_matches(team_id)
            recent_matches = []
            
            if isinstance(matches_response, dict) and 'results' in matches_response:
                matches = matches_response['results']
                total_matches = len(matches)
                recent_matches = []
                finished_tournament_matches = []
                
                for match in matches:
                    match_date = match.get('start_date', '')
                    if self.is_match_within_timeframe(match_date):
                        recent_matches.append(match)
                        if self.is_tournament_finished(match):
                            finished_tournament_matches.append(match)
                
                logger.info(f"Found {len(recent_matches)} recent matches for {team_name}")
                logger.info(f"Filtered to {len(finished_tournament_matches)} matches from finished tournaments")
                return finished_tournament_matches
            else:
                logger.warning(f"Unexpected response format for {team_name} matches")
                return []
            
        except Exception as e:
            logger.error(f"Error fetching matches for {team_name}: {e}")
            return []
    
    async def process_match_details(self, match_slug: str) -> Optional[Dict]:
        """
        Get detailed match information including veto data.
        
        Args:
            match_slug: Match slug/identifier
            
        Returns:
            Match details dictionary or None if error
        """
        try:
            match_details = await self.client.get_match_details(match_slug)
            return match_details
        except Exception as e:
            logger.error(f"Error fetching match details for match {match_slug}: {e}")
            return None
    
    def extract_map_data(self, match_details: Dict) -> None:
        """
        Extract map pick/ban data from match details and add to data_rows.
        
        Args:
            match_details: Match details dictionary
        """
        try:
            match_id = match_details.get('id')
            match_date = match_details.get('start_date', '').split('T')[0]
            
            # Get team information
            team1 = match_details.get('team1', {})
            team2 = match_details.get('team2', {})
            
            # Normalize team names for consistency
            team_a_name = self.normalize_team_name(team1.get('name', ''))
            team_b_name = self.normalize_team_name(team2.get('name', ''))
            team_a_id = team1.get('id')
            team_b_id = team2.get('id')
            
            # Get team rankings (default to 51 if not in top 50) - FIXED: use actual team rankings
            team_a_rank = self.team_rankings.get(team_a_id, 51)
            team_b_rank = self.team_rankings.get(team_b_id, 51)
            
            # Get veto information (match_maps) and games data
            match_maps_data = match_details.get('match_maps', [])
            games_data = match_details.get('games', [])
            
            if not match_maps_data:
                logger.warning(f"Match {match_id} has no veto data")
                return
            
            # Count actual games played to determine if decider was played
            games_played = len(games_data)
            maps_with_results = set()
            for game in games_data:
                map_name = game.get('map_name', '').replace('de_', '')
                maps_with_results.add(map_name.lower())
            
            # Process veto/pick data
            for map_item in match_maps_data:
                choice_type = map_item.get('choice_type')  # 1=pick, 2=ban, 3=decider
                
                # Only process picks and deciders, skip bans
                if choice_type in [1, 3]:  # pick or decider
                    map_order = map_item.get('order', 0)
                    map_info = map_item.get('maps', {})
                    map_name = map_info.get('name', '')
                    picked_by_team_id = map_item.get('team_id')
                    
                    # Determine who picked the map
                    if choice_type == 3:  # decider
                        picked_by = "decider"
                    elif picked_by_team_id == team_a_id:
                        picked_by = team_a_name
                    elif picked_by_team_id == team_b_id:
                        picked_by = team_b_name
                    else:
                        picked_by = "unknown"
                    
                    # Find corresponding game result and extract scores
                    winner = None
                    score_A = None
                    score_B = None
                    map_slug = map_info.get('map_name', '').replace('de_', '')
                    map_was_played = False
                    
                    for game in games_data:
                        game_map = game.get('map_name', '').replace('de_', '')
                        if game_map == map_slug or map_name.lower() in game_map.lower():
                            map_was_played = True
                            winner_id = game.get('winner_team_clan', {}).get('team', {}).get('id')
                            
                            # Extract scores
                            team1_score = game.get('team1_score', 0)
                            team2_score = game.get('team2_score', 0)
                            
                            # Assign scores to team_A and team_B based on team order in match
                            if team_a_id == team1.get('id'):
                                score_A = team1_score
                                score_B = team2_score
                            else:
                                score_A = team2_score
                                score_B = team1_score
                            
                            # Determine winner
                            if winner_id == team_a_id:
                                winner = team_a_name
                            elif winner_id == team_b_id:
                                winner = team_b_name
                            break
                    
                    # FIXED: Skip decider maps that weren't actually played (2-0 results)
                    if choice_type == 3 and not map_was_played:
                        logger.debug(f"Skipping unplayed decider map {map_name} in match {match_id}")
                        continue
                    
                    # FIXED: Skip any map that wasn't actually played (BO5 early endings, etc.)
                    if not map_was_played or winner is None:
                        logger.debug(f"Skipping unplayed map {map_name} in match {match_id} (no winner data)")
                        continue
                    
                    # Calculate derived features
                    rank_diff = team_a_rank - team_b_rank if not (np.isnan(team_a_rank) or np.isnan(team_b_rank)) else np.nan
                    abs_rank_diff = abs(rank_diff) if not np.isnan(rank_diff) else np.nan
                    
                    # Determine picked_by_is_A
                    if choice_type == 3:  # decider
                        picked_by_is_A = 0  # Neutral for decider
                        is_decider = 1
                    elif picked_by == team_a_name:
                        picked_by_is_A = 1
                        is_decider = 0
                    elif picked_by == team_b_name:
                        picked_by_is_A = 0
                        is_decider = 0
                    else:
                        picked_by_is_A = 0  # Default for unknown
                        is_decider = 0
                    
                    # Create target variable
                    winner_is_A = 1 if winner == team_a_name else 0
                    
                    # Add enhanced row to dataset with all features
                    row = {
                        # Base columns (reference only)
                        'match_id': match_id,
                        'date': match_date,
                        'map': map_name,
                        'map_number': map_order,  # Renamed from map_order
                        'team_A': team_a_name,
                        'team_B': team_b_name,
                        'winner': winner,  # Team name of winner
                        'score_A': score_A,  # Team A's score on this map
                        'score_B': score_B,  # Team B's score on this map
                        
                        # Derived features (main training features)
                        'team_A_rank': team_a_rank,
                        'team_B_rank': team_b_rank,
                        'rank_diff': rank_diff,
                        'abs_rank_diff': abs_rank_diff,
                        'picked_by_is_A': picked_by_is_A,
                        'is_decider': is_decider,
                        'map_winrate_A': np.nan,  # Will be calculated later
                        'map_winrate_B': np.nan,  # Will be calculated later
                        'recent_form_A': np.nan,  # Will be calculated later
                        'recent_form_B': np.nan,  # Will be calculated later
                        
                        # Target column
                        'winner_is_A': winner_is_A,
                        
                        # Reference column (optional)
                        'winner_binary': f"team_{'A' if winner_is_A else 'B'}"
                    }
                    self.data_rows.append(row)
                    logger.debug(f"Added enhanced row for match {match_id}, map {map_name} (played: {map_was_played})")
                        
        except Exception as e:
            logger.error(f"Error extracting map data from match {match_details.get('id', 'unknown')}: {e}")
    
    def recalculate_winrates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recalculate winrates using chronological order for realistic features."""
        logger.info("Recalculating winrates chronologically...")
        
        # Sort by date for chronological processing
        df_sorted = df.sort_values('date').copy()
        
        # Initialize tracking dictionaries
        team_match_wins = defaultdict(int)
        team_match_total = defaultdict(int)
        team_map_wins = defaultdict(lambda: defaultdict(int))
        team_map_total = defaultdict(lambda: defaultdict(int))
        
        updated_rows = []
        
        for idx, row in df_sorted.iterrows():
            team_a = row['team_A']
            team_b = row['team_B']
            map_name = row['map']
            winner_is_a = row['winner_is_A']
            
            # Calculate current winrates (before this match)
            # Recent form (based on last matches, simple approximation)
            if team_match_total[team_a] >= 3:
                recent_form_a = team_match_wins[team_a] / min(team_match_total[team_a], 10)
            else:
                recent_form_a = np.nan
                
            if team_match_total[team_b] >= 3:
                recent_form_b = team_match_wins[team_b] / min(team_match_total[team_b], 10)
            else:
                recent_form_b = np.nan
            
            # Map-specific winrates
            if team_map_total[team_a][map_name] >= 3:
                map_winrate_a = team_map_wins[team_a][map_name] / team_map_total[team_a][map_name]
            else:
                map_winrate_a = np.nan
                
            if team_map_total[team_b][map_name] >= 3:
                map_winrate_b = team_map_wins[team_b][map_name] / team_map_total[team_b][map_name]
            else:
                map_winrate_b = np.nan
            
            # Update row with calculated winrates
            row_copy = row.copy()
            row_copy['recent_form_A'] = recent_form_a
            row_copy['recent_form_B'] = recent_form_b
            row_copy['map_winrate_A'] = map_winrate_a
            row_copy['map_winrate_B'] = map_winrate_b
            
            updated_rows.append(row_copy)
            
            # Update counters for next iteration
            if winner_is_a:
                team_match_wins[team_a] += 1
            else:
                team_match_wins[team_b] += 1
                
            team_match_total[team_a] += 1
            team_match_total[team_b] += 1
            
            if winner_is_a:
                team_map_wins[team_a][map_name] += 1
            else:
                team_map_wins[team_b][map_name] += 1
                
            team_map_total[team_a][map_name] += 1
            team_map_total[team_b][map_name] += 1
        
        return pd.DataFrame(updated_rows)
    
    def add_train_test_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add both row-based and match-based train/test splits."""
        logger.info("Adding train/test splits...")
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Row-based split (treat each map as independent)
        n_rows = len(df)
        indices = list(range(n_rows))
        random.shuffle(indices)
        
        train_end = int(0.7 * n_rows)
        val_end = int(0.85 * n_rows)
        
        df['split_row'] = 'test'  # Default
        df.iloc[indices[:train_end], df.columns.get_loc('split_row')] = 'train'
        df.iloc[indices[train_end:val_end], df.columns.get_loc('split_row')] = 'val'
        
        # Match-based split (keep all maps from same match together)
        unique_matches = df['match_id'].unique()
        random.shuffle(unique_matches)
        
        n_matches = len(unique_matches)
        train_matches_end = int(0.7 * n_matches)
        val_matches_end = int(0.85 * n_matches)
        
        train_matches = set(unique_matches[:train_matches_end])
        val_matches = set(unique_matches[train_matches_end:val_matches_end])
        
        def assign_match_split(match_id):
            if match_id in train_matches:
                return 'train'
            elif match_id in val_matches:
                return 'val'
            else:
                return 'test'
        
        df['split_match'] = df['match_id'].apply(assign_match_split)
        
        return df
    
    def generate_quality_report(self, df: pd.DataFrame) -> None:
        """Generate comprehensive dataset quality report."""
        logger.info("Generating dataset quality report...")
        
        print("="*80)
        print("ENHANCED CS2 DATASET QUALITY REPORT")
        print("="*80)
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS:")
        print(f"Total records: {len(df):,}")
        print(f"Unique matches: {df['match_id'].nunique():,}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique teams: {len(set(df['team_A'].unique()) | set(df['team_B'].unique()))}")
        print(f"Unique maps: {len(df['map'].unique())}")
        
        # Class distribution
        print(f"\n🎯 TARGET DISTRIBUTION:")
        winner_dist = df['winner_is_A'].value_counts()
        print(f"Team A wins: {winner_dist.get(1, 0)} ({winner_dist.get(1, 0)/len(df)*100:.1f}%)")
        print(f"Team B wins: {winner_dist.get(0, 0)} ({winner_dist.get(0, 0)/len(df)*100:.1f}%)")
        
        # Split distributions
        for split_type in ['split_row', 'split_match']:
            print(f"\n📋 {split_type.upper()} DISTRIBUTION:")
            split_dist = df[split_type].value_counts()
            for split, count in split_dist.items():
                pct = count / len(df) * 100
                winner_pct = df[df[split_type] == split]['winner_is_A'].mean() * 100
                print(f"{split}: {count:,} records ({pct:.1f}%) - {winner_pct:.1f}% Team A wins")
        
        # Map distribution
        print(f"\n🗺️ MAP DISTRIBUTION:")
        map_dist = df['map'].value_counts()
        for map_name, count in map_dist.head(10).items():
            pct = count / len(df) * 100
            print(f"{map_name}: {count:,} ({pct:.1f}%)")
        
        # Feature statistics
        print(f"\n🔢 FEATURE STATISTICS:")
        print(f"picked_by_is_A = 1: {df['picked_by_is_A'].sum()} ({df['picked_by_is_A'].mean()*100:.1f}%)")
        print(f"is_decider = 1: {df['is_decider'].sum()} ({df['is_decider'].mean()*100:.1f}%)")
        
        # Missing values
        print(f"\n❓ MISSING VALUES:")
        missing = df.isnull().sum()
        for col in missing[missing > 0].index:
            count = missing[col]
            pct = count / len(df) * 100
            print(f"{col}: {count:,} ({pct:.1f}%)")
        
        # Rank statistics
        if not df['rank_diff'].isna().all():
            print(f"\n📈 RANK STATISTICS:")
            print(f"Mean rank difference: {df['rank_diff'].mean():.2f}")
            print(f"Mean absolute rank difference: {df['abs_rank_diff'].mean():.2f}")
            print(f"Max rank difference: {df['abs_rank_diff'].max():.0f}")
        
        print("="*80)

    async def scrape_data(self) -> pd.DataFrame:
        """
        Main method to scrape all data and return DataFrame.
        
        Returns:
            pandas DataFrame with scraped data
        """
        logger.info("Starting CS2 match data scraping...")
        
        # Initialize the CS2 client
        self.client = CS2()
        
        try:
            # Get top 50 teams
            top_teams = await self.get_top_50_teams()
            if not top_teams:
                logger.error("Failed to get team data")
                return pd.DataFrame()
            
            processed_matches = set()  # To avoid duplicate matches
            
            # Process each team
            for team in top_teams:
                team_id = team['id']
                team_name = team['name']
                
                # Get team matches
                matches = await self.get_team_matches(team_id, team_name)
                
                for match in matches:
                    match_id = match.get('id')
                    
                    # Skip if already processed (to avoid duplicates)
                    if match_id in processed_matches:
                        continue
                        
                    processed_matches.add(match_id)
                    
                    # Extract map data directly from match (no need for additional API call)
                    self.extract_map_data(match)
                    
                    # Add small delay to avoid rate limiting
                    await asyncio.sleep(0.1)
            
            # Create DataFrame
            df = pd.DataFrame(self.data_rows)
            
            if df.empty:
                logger.error("No data collected")
                return df
            
            logger.info(f"Created initial DataFrame with {len(df)} rows")
            
            # Recalculate winrates chronologically
            df = self.recalculate_winrates(df)
            logger.info("Recalculated winrates chronologically")
            
            # Add train/test splits
            df = self.add_train_test_splits(df)
            logger.info("Added train/test splits")
            
            return df
            
        finally:
            # Always close the client
            if self.client:
                await self.client.close()
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = "map_picks_last6m_top50_enriched.csv") -> None:
        """
        Save enhanced DataFrame to CSV file with quality report.
        
        Args:
            df: pandas DataFrame to save
            filename: Output filename
        """
        try:
            # Define the correct column order based on user's expected format
            column_order = [
                'match_id', 'date', 'map', 'map_number', 'team_A', 'team_B', 'winner', 'score_A', 'score_B',
                'team_A_rank', 'team_B_rank', 'rank_diff', 'abs_rank_diff', 'picked_by_is_A', 'is_decider',
                'map_winrate_A', 'map_winrate_B', 'recent_form_A', 'recent_form_B', 'winner_is_A',
                'winner_binary', 'split_row', 'split_match'
            ]
            
            # Reorder columns and save
            df_ordered = df[column_order]
            df_ordered.to_csv(filename, index=False)
            logger.info(f"Successfully saved {len(df_ordered)} rows to {filename}")
            
            # Generate quality report
            self.generate_quality_report(df_ordered)
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise


async def main():
    """Main function to run the enhanced scraper."""
    logger.info("Starting CS2 enhanced match data scraping...")
    scraper = CS2MatchDataScraper()
    
    try:
        # Scrape data
        df = await scraper.scrape_data()
        
        if not df.empty:
            # Save to CSV with quality report
            scraper.save_to_csv(df)
        else:
            logger.error("No data was scraped. Please check the logs for errors.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Script failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())