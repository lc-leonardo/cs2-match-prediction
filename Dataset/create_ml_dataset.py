#!/usr/bin/env python3
"""
Post-processing script to transform the original CS2 dataset into ML-ready format.
Uses the clean map_picks_last6m_top50.csv with actual match data.
"""

import pandas as pd
import numpy as np
import logging
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_team_name(team_name: str) -> str:
    """Normalize team names for consistency."""
    if not team_name:
        return ""
    
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
    lower_name = team_name.lower().strip()
    if lower_name in name_mapping:
        return name_mapping[lower_name]
    
    # Default to original with proper formatting
    return team_name.strip()



def calculate_winrates_chronologically(df_sorted):
    """Calculate map winrates chronologically to avoid data leakage."""
    logger.info("Calculating map winrates chronologically...")
    
    team_match_history = defaultdict(list)  # team -> [(date, winner), ...]
    team_map_history = defaultdict(lambda: defaultdict(list))  # team -> map -> [(date, winner), ...]
    
    map_winrate_A_list = []
    map_winrate_B_list = []
    recent_form_A_list = []
    recent_form_B_list = []
    
    for idx, row in df_sorted.iterrows():
        date = row['date']
        team_A = row['team_A']
        team_B = row['team_B']
        map_name = row['map']
        winner = row['winner']
        
        # Calculate recent form (last 10 matches)
        recent_A = team_match_history[team_A][-10:] if team_match_history[team_A] else []
        recent_B = team_match_history[team_B][-10:] if team_match_history[team_B] else []
        
        form_A = sum(1 for _, w in recent_A if w == team_A) / len(recent_A) if recent_A else np.nan
        form_B = sum(1 for _, w in recent_B if w == team_B) / len(recent_B) if recent_B else np.nan
        
        recent_form_A_list.append(form_A)
        recent_form_B_list.append(form_B)
        
        # Calculate map-specific winrates
        map_history_A = team_map_history[team_A][map_name]
        map_history_B = team_map_history[team_B][map_name]
        
        winrate_A = sum(1 for _, w in map_history_A if w == team_A) / len(map_history_A) if map_history_A else np.nan
        winrate_B = sum(1 for _, w in map_history_B if w == team_B) / len(map_history_B) if map_history_B else np.nan
        
        map_winrate_A_list.append(winrate_A)
        map_winrate_B_list.append(winrate_B)
        
        # Update history AFTER calculating features (to avoid leakage)
        team_match_history[team_A].append((date, winner))
        team_match_history[team_B].append((date, winner))
        team_map_history[team_A][map_name].append((date, winner))
        team_map_history[team_B][map_name].append((date, winner))
    
    return map_winrate_A_list, map_winrate_B_list, recent_form_A_list, recent_form_B_list

def create_ml_dataset_from_dataframe(df):
    """
    Transform a dataframe into ML-ready format.
    
    Args:
        df: Input pandas DataFrame with raw match data
        
    Returns:
        DataFrame: ML-ready dataset
    """
    logger.info(f"Processing dataframe with {len(df)} records")
    
    try:
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Filter for actually played maps only
        df = df[df['map_was_played'] == True].copy()
        logger.info(f"Filtered to {len(df)} played maps")
        
        # Normalize team names
        logger.info("Normalizing team names...")
        df['team_A'] = df['team1'].apply(normalize_team_name)
        df['team_B'] = df['team2'].apply(normalize_team_name)
        df['winner'] = df['winner'].apply(lambda x: 'team1' if x == 'team1' else 'team2' if x == 'team2' else x)
        
        # Add missing columns with default values
        if 'team_A_rank' not in df.columns:
            df['team_A_rank'] = 50  # Default rank
        if 'team_B_rank' not in df.columns:
            df['team_B_rank'] = 50  # Default rank
        if 'picked_by' not in df.columns:
            # Infer picked_by from pick_ban_action
            df['picked_by'] = df.apply(lambda row: 
                row['team_A'] if row.get('team_action') == 'team1' and row.get('pick_ban_action') == 'pick'
                else row['team_B'] if row.get('team_action') == 'team2' and row.get('pick_ban_action') == 'pick'
                else 'decider' if row.get('pick_ban_action') == 'decider'
                else 'unknown', axis=1)
        
        # Add map_number (sequence within match)
        df['map_number'] = df.groupby('match_id').cumcount() + 1
        
        # Set random seed for reproducible results
        random.seed(42)
        np.random.seed(42)
        
        # Create derived features
        logger.info("Creating derived features...")
        df['rank_diff'] = df['team_A_rank'] - df['team_B_rank']
        df['abs_rank_diff'] = abs(df['rank_diff'])
        
        # Pick indicators
        df['picked_by_is_A'] = (df['picked_by'] == df['team_A']).astype(int)
        df['is_decider'] = (df['picked_by'] == 'decider').astype(int)
        
        # Winner indicators - convert map_winner from team1/team2 to team_A/team_B
        df['winner_is_A'] = df.apply(lambda row: 
            1 if row['map_winner'] == 'team1' else 
            0 if row['map_winner'] == 'team2' else np.nan, axis=1)
        
        # Sort by date for chronological processing
        df_sorted = df.sort_values('date').copy()
        
        # Calculate winrates chronologically
        map_winrate_A, map_winrate_B, recent_form_A, recent_form_B = calculate_winrates_chronologically(df_sorted)
        
        # Add the calculated features
        df_sorted['map_winrate_A'] = map_winrate_A
        df_sorted['map_winrate_B'] = map_winrate_B
        df_sorted['recent_form_A'] = recent_form_A
        df_sorted['recent_form_B'] = recent_form_B
        
        # Create data splits
        logger.info("Creating data splits...")
        
        # Row-level split (70-15-15)
        n_records = len(df_sorted)
        train_end = int(0.7 * n_records)
        val_end = int(0.85 * n_records)
        
        df_sorted['split_row'] = 'test'
        df_sorted.iloc[:train_end, df_sorted.columns.get_loc('split_row')] = 'train'
        df_sorted.iloc[train_end:val_end, df_sorted.columns.get_loc('split_row')] = 'val'
        
        # Match-level split
        unique_matches = df_sorted['match_id'].unique()
        n_matches = len(unique_matches)
        train_matches = unique_matches[:int(0.7 * n_matches)]
        val_matches = unique_matches[int(0.7 * n_matches):int(0.85 * n_matches)]
        
        df_sorted['split_match'] = df_sorted['match_id'].apply(
            lambda x: 'train' if x in train_matches 
            else 'val' if x in val_matches 
            else 'test'
        )
        
        # Select final columns
        final_columns = [
            'match_id', 'date', 'map', 'map_number', 'team_A', 'team_B', 
            'winner', 'team_A_rank', 'team_B_rank', 'rank_diff', 'abs_rank_diff', 
            'picked_by_is_A', 'is_decider',
            'map_winrate_A', 'map_winrate_B', 'recent_form_A', 'recent_form_B', 
            'winner_is_A', 'split_row', 'split_match'
        ]
        
        # Filter to only include columns that exist
        available_columns = [col for col in final_columns if col in df_sorted.columns]
        df_final = df_sorted[available_columns].copy()
        
        logger.info(f"Successfully processed {len(df_final)} records")
        return df_final
        
    except Exception as e:
        logger.error(f"Error processing dataframe: {e}")
        raise

def create_ml_dataset(input_file="map_picks_last6m_top50.csv", 
                     output_file="map_picks_last6m_top50_ml_ready.csv"):
    """
    Transform the original dataset into ML-ready format.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
    """
    logger.info(f"Loading original dataset from {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        
        # Use the dataframe processing function
        df_final = create_ml_dataset_from_dataframe(df)
        
        # Save final dataset
        df_final.to_csv(output_file, index=False)
        logger.info(f"Successfully saved {len(df_final)} records to {output_file}")
        
        # Generate quality report
        generate_quality_report(df_final)
        
        return df_final

        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

def generate_quality_report(df):
    """Generate a comprehensive quality report."""
    logger.info("Generating quality report...")
    
    print("\n" + "="*80)
    print("ML-READY CS2 DATASET QUALITY REPORT")
    print("="*80)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"Total records: {len(df):,}")
    print(f"Unique matches: {df['match_id'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique teams: {pd.concat([df['team_A'], df['team_B']]).nunique()}")
    print(f"Unique maps: {df['map'].nunique()}")
    
    print(f"\n🎯 TARGET DISTRIBUTION:")
    team_a_wins = df['winner_is_A'].sum()
    team_b_wins = len(df) - team_a_wins
    print(f"Team A wins: {team_a_wins} ({team_a_wins/len(df)*100:.1f}%)")
    print(f"Team B wins: {team_b_wins} ({team_b_wins/len(df)*100:.1f}%)")
    

    
    print(f"\n🗺️ MAP DISTRIBUTION:")
    for map_name, count in df['map'].value_counts().items():
        print(f"{map_name}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n🔢 FEATURE STATISTICS:")
    print(f"picked_by_is_A = 1: {df['picked_by_is_A'].sum()} ({df['picked_by_is_A'].mean()*100:.1f}%)")
    print(f"is_decider = 1: {df['is_decider'].sum()} ({df['is_decider'].mean()*100:.1f}%)")
    
    print(f"\n📋 DATA SPLITS:")
    for split_col in ['split_row', 'split_match']:
        print(f"{split_col.upper()} DISTRIBUTION:")
        for split_name, count in df[split_col].value_counts().items():
            split_winners = df[df[split_col] == split_name]['winner_is_A'].mean()
            print(f"  {split_name}: {count} records ({count/len(df)*100:.1f}%) - {split_winners*100:.1f}% Team A wins")
    
    print("="*80)

def main():
    """Main execution function."""
    logger.info("Starting ML dataset creation from original data...")
    
    try:
        df_final = create_ml_dataset()
        logger.info("ML dataset creation completed successfully!")
        
        print("\n🔍 SAMPLE OF FINAL DATA:")
        print(df_final.head(3).to_string())
        
    except Exception as e:
        logger.error(f"ML dataset creation failed: {e}")
        raise

if __name__ == "__main__":
    main()