#!/usr/bin/env python3
"""
Dataset Extension Script for CS2 Match Data
Automatically detects new finished tournaments and extends both raw and ML-ready datasets.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
import sys
import os
from collections import defaultdict

# Add the cs2api directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cs2api'))

from cs2api import CS2API

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetExtender:
    def __init__(self, 
                 raw_dataset_path='map_picks_last6m_top50.csv',
                 ml_dataset_path='map_picks_last6m_top50_ml_ready.csv',
                 teams_file='current_top_50_teams.json'):
        """Initialize the dataset extender with file paths."""
        self.raw_dataset_path = Path(raw_dataset_path)
        self.ml_dataset_path = Path(ml_dataset_path)
        self.teams_file = Path(teams_file)
        self.api = CS2API()
        
        # Load existing datasets
        self.raw_df = None
        self.ml_df = None
        self.existing_match_ids = set()
        self.teams_data = {}
        
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing datasets and extract metadata."""
        logger.info("Loading existing datasets...")
        
        # Load raw dataset
        if self.raw_dataset_path.exists():
            self.raw_df = pd.read_csv(self.raw_dataset_path)
            self.existing_match_ids = set(self.raw_df['match_id'].unique())
            logger.info(f"Loaded raw dataset: {len(self.raw_df)} records from {len(self.existing_match_ids)} matches")
        else:
            logger.warning(f"Raw dataset not found: {self.raw_dataset_path}")
            self.raw_df = pd.DataFrame()
        
        # Load ML-ready dataset
        if self.ml_dataset_path.exists():
            self.ml_df = pd.read_csv(self.ml_dataset_path)
            logger.info(f"Loaded ML dataset: {len(self.ml_df)} records")
        else:
            logger.warning(f"ML dataset not found: {self.ml_dataset_path}")
            self.ml_df = pd.DataFrame()
        
        # Load teams data
        if self.teams_file.exists():
            with open(self.teams_file, 'r') as f:
                self.teams_data = json.load(f)
            logger.info(f"Loaded teams data for {len(self.teams_data)} teams")
        else:
            logger.error(f"Teams file not found: {self.teams_file}")
            raise FileNotFoundError(f"Required teams file not found: {self.teams_file}")
    
    def _get_date_range_for_extension(self):
        """Determine the date range for checking new tournaments."""
        if len(self.raw_df) == 0:
            # If no existing data, check last 6 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
        else:
            # Get the latest date from existing data and check from there
            latest_date = pd.to_datetime(self.raw_df['date']).max()
            start_date = latest_date - timedelta(days=7)  # Overlap by 1 week to catch late updates
            end_date = datetime.now()
        
        logger.info(f"Checking for new tournaments from {start_date.date()} to {end_date.date()}")
        return start_date, end_date
    
    async def _get_team_matches(self, team_name, team_id, start_date, end_date):
        """Fetch matches for a specific team within the date range."""
        try:
            logger.info(f"Fetching matches for {team_name} (ID: {team_id})")
            
            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            matches = await self.api.get_team_matches(
                team_id, 
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp
            )
            
            logger.info(f"Found {len(matches)} matches for {team_name}")
            return matches
            
        except Exception as e:
            logger.error(f"Error fetching matches for {team_name}: {e}")
            return []
    
    def _filter_new_matches(self, all_matches):
        """Filter out matches that already exist in the dataset."""
        new_matches = []
        
        for match in all_matches:
            match_id = match.get('id')
            if match_id not in self.existing_match_ids:
                # Additional filters
                if (match.get('tournament_status') == 'finished' and 
                    match.get('maps') and 
                    len(match.get('maps', [])) > 0):
                    new_matches.append(match)
        
        logger.info(f"Found {len(new_matches)} new matches after filtering")
        return new_matches
    
    def _process_match_to_dataframe_rows(self, match):
        """Convert a match object to dataframe rows (same logic as original scraper)."""
        rows = []
        match_id = match['id']
        team1 = match['team1']['name']
        team2 = match['team2']['name']
        winner = match['winner']
        date = datetime.fromtimestamp(match['date']).strftime('%Y-%m-%d')
        
        # Convert winner to team1/team2 format
        if winner == match['team1']['id']:
            winner_name = 'team1'
        elif winner == match['team2']['id']:
            winner_name = 'team2'
        else:
            winner_name = None
        
        maps_data = match.get('maps', [])
        
        for map_data in maps_data:
            map_name = map_data.get('map_name', '')
            pick_ban_sequence = map_data.get('pick_ban_sequence', [])
            map_winner = map_data.get('winner')
            map_was_played = map_data.get('played', True)
            
            # Convert map winner to team1/team2 format
            if map_winner == match['team1']['id']:
                map_winner_name = 'team1'
            elif map_winner == match['team2']['id']:
                map_winner_name = 'team2'
            else:
                map_winner_name = None
            
            # Process pick/ban sequence
            if pick_ban_sequence:
                for action in pick_ban_sequence:
                    if action.get('map_name') == map_name:
                        team_action = 'team1' if action.get('team_id') == match['team1']['id'] else 'team2'
                        
                        row = {
                            'match_id': match_id,
                            'team1': team1,
                            'team2': team2,
                            'map': map_name,
                            'pick_ban_action': action.get('action', 'unknown'),
                            'team_action': team_action,
                            'winner': winner_name,
                            'map_winner': map_winner_name,
                            'date': date,
                            'map_was_played': map_was_played
                        }
                        rows.append(row)
            else:
                # If no pick/ban data, create a basic row
                row = {
                    'match_id': match_id,
                    'team1': team1,
                    'team2': team2,
                    'map': map_name,
                    'pick_ban_action': 'unknown',
                    'team_action': 'unknown',
                    'winner': winner_name,
                    'map_winner': map_winner_name,
                    'date': date,
                    'map_was_played': map_was_played
                }
                rows.append(row)
        
        return rows
    
    async def fetch_new_matches(self):
        """Fetch new matches from all teams."""
        logger.info("Starting to fetch new matches...")
        
        start_date, end_date = self._get_date_range_for_extension()
        all_new_matches = []
        
        # Fetch matches for each team
        for team_name, team_data in self.teams_data.items():
            team_id = team_data['id']
            await asyncio.sleep(1)  # Rate limiting
            
            matches = await self._get_team_matches(team_name, team_id, start_date, end_date)
            all_new_matches.extend(matches)
        
        # Remove duplicates and filter
        unique_matches = {match['id']: match for match in all_new_matches}
        new_matches = self._filter_new_matches(list(unique_matches.values()))
        
        return new_matches
    
    def process_new_matches_to_raw_format(self, new_matches):
        """Convert new matches to raw dataset format."""
        logger.info(f"Processing {len(new_matches)} new matches to raw format...")
        
        all_rows = []
        for match in new_matches:
            try:
                rows = self._process_match_to_dataframe_rows(match)
                all_rows.extend(rows)
            except Exception as e:
                logger.error(f"Error processing match {match.get('id', 'unknown')}: {e}")
                continue
        
        if all_rows:
            new_raw_df = pd.DataFrame(all_rows)
            logger.info(f"Created {len(new_raw_df)} new raw records")
            return new_raw_df
        else:
            logger.info("No new records to add")
            return pd.DataFrame()
    
    def extend_raw_dataset(self, new_raw_df):
        """Extend the raw dataset with new matches."""
        if len(new_raw_df) == 0:
            logger.info("No new data to add to raw dataset")
            return self.raw_df
        
        # Combine with existing data
        if len(self.raw_df) > 0:
            extended_df = pd.concat([self.raw_df, new_raw_df], ignore_index=True)
        else:
            extended_df = new_raw_df.copy()
        
        # Remove duplicates based on match_id, team1, team2, map, pick_ban_action
        extended_df = extended_df.drop_duplicates(
            subset=['match_id', 'team1', 'team2', 'map', 'pick_ban_action'],
            keep='first'
        )
        
        logger.info(f"Extended raw dataset: {len(extended_df)} total records")
        return extended_df
    
    def create_ml_ready_from_raw(self, raw_df):
        """Transform raw dataset to ML-ready format using the same logic as create_ml_dataset.py"""
        logger.info("Creating ML-ready dataset from raw data...")
        
        # Import the ML dataset creation functions
        from create_ml_dataset import create_ml_dataset_from_dataframe
        
        try:
            ml_ready_df = create_ml_dataset_from_dataframe(raw_df)
            logger.info(f"Created ML-ready dataset: {len(ml_ready_df)} records")
            return ml_ready_df
        except Exception as e:
            logger.error(f"Error creating ML-ready dataset: {e}")
            # Fallback: return a basic transformation
            return self._basic_ml_transformation(raw_df)
    
    def _basic_ml_transformation(self, raw_df):
        """Basic ML transformation as fallback."""
        logger.warning("Using basic ML transformation as fallback")
        
        # Filter for played maps only
        played_df = raw_df[raw_df['map_was_played'] == True].copy()
        
        # Create basic ML features
        played_df['team_A'] = played_df['team1']
        played_df['team_B'] = played_df['team2']
        played_df['winner_is_A'] = (played_df['map_winner'] == 'team1').astype(int)
        
        # Add basic time-based splits
        played_df = played_df.sort_values('date')
        n_records = len(played_df)
        train_end = int(0.7 * n_records)
        val_end = int(0.85 * n_records)
        
        played_df['split_row'] = 'test'
        played_df.iloc[:train_end, played_df.columns.get_loc('split_row')] = 'train'
        played_df.iloc[train_end:val_end, played_df.columns.get_loc('split_row')] = 'val'
        
        return played_df
    
    def save_datasets(self, raw_df, ml_df):
        """Save both raw and ML-ready datasets."""
        logger.info("Saving extended datasets...")
        
        # Backup original files
        if self.raw_dataset_path.exists():
            backup_path = self.raw_dataset_path.with_suffix('.csv.backup')
            self.raw_dataset_path.rename(backup_path)
            logger.info(f"Backed up original raw dataset to {backup_path}")
        
        if self.ml_dataset_path.exists():
            backup_path = self.ml_dataset_path.with_suffix('.csv.backup')
            self.ml_dataset_path.rename(backup_path)
            logger.info(f"Backed up original ML dataset to {backup_path}")
        
        # Save new datasets
        raw_df.to_csv(self.raw_dataset_path, index=False)
        ml_df.to_csv(self.ml_dataset_path, index=False)
        
        logger.info(f"Saved extended raw dataset: {len(raw_df)} records")
        logger.info(f"Saved extended ML dataset: {len(ml_df)} records")
    
    async def extend_datasets(self):
        """Main method to extend both datasets."""
        try:
            logger.info("🚀 Starting dataset extension process...")
            
            # Fetch new matches
            new_matches = await self.fetch_new_matches()
            
            if not new_matches:
                logger.info("✅ No new matches found. Datasets are up to date.")
                return
            
            # Process to raw format
            new_raw_df = self.process_new_matches_to_raw_format(new_matches)
            
            if len(new_raw_df) == 0:
                logger.info("✅ No new records to add. Datasets are up to date.")
                return
            
            # Extend raw dataset
            extended_raw_df = self.extend_raw_dataset(new_raw_df)
            
            # Create ML-ready dataset
            extended_ml_df = self.create_ml_ready_from_raw(extended_raw_df)
            
            # Save datasets
            self.save_datasets(extended_raw_df, extended_ml_df)
            
            logger.info("✅ Dataset extension completed successfully!")
            logger.info(f"📊 Added {len(new_raw_df)} new records")
            logger.info(f"📈 Total records - Raw: {len(extended_raw_df)}, ML: {len(extended_ml_df)}")
            
        except Exception as e:
            logger.error(f"❌ Dataset extension failed: {e}")
            raise

def main():
    """Main execution function."""
    extender = DatasetExtender()
    
    try:
        asyncio.run(extender.extend_datasets())
    except KeyboardInterrupt:
        logger.info("Extension process interrupted by user")
    except Exception as e:
        logger.error(f"Extension process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()