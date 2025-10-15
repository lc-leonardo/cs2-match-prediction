#!/usr/bin/env python3
"""
Comprehensive CS2 Dataset Profiler
Provides in-depth analysis of the ML-ready dataset including:
- Missing values analysis
- Data distribution analysis
- Data quality checks
- Temporal analysis
- Feature correlations
- Outlier detection
- And much more!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class CS2DatasetProfiler:
    def __init__(self, filepath):
        """Initialize the profiler with dataset."""
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.n_rows = len(self.df)
        self.n_cols = len(self.df.columns)
        
        print(f"📊 Loaded dataset: {self.n_rows:,} rows × {self.n_cols} columns")
        print(f"📁 File: {filepath}")
        print("="*80)
    
    def basic_info(self):
        """Display basic dataset information."""
        print("\n🔍 BASIC DATASET INFORMATION")
        print("-" * 40)
        
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\n📋 Data Types:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Column names and types
        print("\n📝 Column Details:")
        for i, (col, dtype) in enumerate(zip(self.df.columns, self.df.dtypes)):
            non_null = self.df[col].notna().sum()
            print(f"  {i+1:2d}. {col:<20} ({dtype}) - {non_null:,}/{self.n_rows:,} non-null")
    
    def missing_values_analysis(self):
        """Comprehensive missing values analysis."""
        print("\n🔍 MISSING VALUES ANALYSIS")
        print("-" * 40)
        
        # Overall missing statistics
        total_cells = self.n_rows * self.n_cols
        missing_cells = self.df.isnull().sum().sum()
        print(f"Total cells: {total_cells:,}")
        print(f"Missing cells: {missing_cells:,} ({missing_cells/total_cells*100:.2f}%)")
        
        # Per column missing values
        missing_data = []
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / self.n_rows) * 100
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            missing_data.append({
                'Column': col,
                'Missing Count': missing_count,
                'Missing %': missing_pct,
                'Data Type': dtype,
                'Unique Values': unique_count
            })
        
        missing_df = pd.DataFrame(missing_data)
        missing_df = missing_df.sort_values('Missing %', ascending=False)
        
        print(f"\n📊 Missing Values by Column:")
        print(missing_df.to_string(index=False, float_format='%.2f'))
        
        # Columns with missing values
        cols_with_missing = missing_df[missing_df['Missing Count'] > 0]
        if len(cols_with_missing) > 0:
            print(f"\n⚠️  Columns with missing values: {len(cols_with_missing)}")
            for _, row in cols_with_missing.iterrows():
                print(f"   • {row['Column']}: {row['Missing Count']:,} ({row['Missing %']:.1f}%)")
        else:
            print(f"\n✅ No missing values found!")
        
        return missing_df
    
    def data_quality_checks(self):
        """Perform comprehensive data quality checks."""
        print("\n🔍 DATA QUALITY CHECKS")
        print("-" * 40)
        
        issues = []
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate rows: {duplicates}")
        
        # Check date format and consistency
        if 'date' in self.df.columns:
            try:
                self.df['date'] = pd.to_datetime(self.df['date'])
                date_nulls = self.df['date'].isnull().sum()
                if date_nulls > 0:
                    issues.append(f"Invalid dates: {date_nulls}")
            except:
                issues.append("Date column has formatting issues")
        
        # Check for impossible values
        if 'winner_is_A' in self.df.columns:
            invalid_winners = (~self.df['winner_is_A'].isin([0, 1])).sum()
            if invalid_winners > 0:
                issues.append(f"Invalid winner_is_A values: {invalid_winners}")
        
        # Check team name consistency
        if all(col in self.df.columns for col in ['team_A', 'team_B', 'winner']):
            invalid_winners = 0
            for _, row in self.df.iterrows():
                if pd.notna(row['winner']) and row['winner'] not in [row['team_A'], row['team_B']]:
                    invalid_winners += 1
            if invalid_winners > 0:
                issues.append(f"Winner not matching team_A or team_B: {invalid_winners}")
        
        # Check ranking consistency
        for rank_col in ['team_A_rank', 'team_B_rank']:
            if rank_col in self.df.columns:
                negative_ranks = (self.df[rank_col] < 1).sum()
                if negative_ranks > 0:
                    issues.append(f"Invalid {rank_col} (< 1): {negative_ranks}")
        
        # Display results
        if issues:
            print("⚠️  Data Quality Issues Found:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print("✅ No major data quality issues detected!")
        
        return issues
    
    def temporal_analysis(self):
        """Analyze temporal patterns in the data."""
        if 'date' not in self.df.columns:
            print("\n⚠️  No date column found for temporal analysis")
            return
        
        print("\n🔍 TEMPORAL ANALYSIS")
        print("-" * 40)
        
        # Ensure date is datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Date range
        date_range = self.df['date'].max() - self.df['date'].min()
        print(f"Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        print(f"Total span: {date_range.days} days")
        
        # Matches per day
        daily_matches = self.df.groupby('date').size()
        print(f"\nMatches per day:")
        print(f"  Average: {daily_matches.mean():.1f}")
        print(f"  Median: {daily_matches.median():.1f}")
        print(f"  Max: {daily_matches.max()}")
        print(f"  Min: {daily_matches.min()}")
        
        # Monthly distribution
        self.df['year_month'] = self.df['date'].dt.to_period('M')
        monthly_counts = self.df['year_month'].value_counts().sort_index()
        print(f"\n📅 Monthly distribution:")
        for month, count in monthly_counts.items():
            print(f"  {month}: {count:,} matches")
        
        # Day of week patterns
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        dow_counts = self.df['day_of_week'].value_counts()
        print(f"\n📆 Day of week distribution:")
        for day, count in dow_counts.items():
            print(f"  {day}: {count:,} matches ({count/len(self.df)*100:.1f}%)")
    
    def categorical_analysis(self):
        """Analyze categorical variables."""
        print("\n🔍 CATEGORICAL VARIABLES ANALYSIS")
        print("-" * 40)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col == 'date':  # Skip date column
                continue
                
            unique_count = self.df[col].nunique()
            value_counts = self.df[col].value_counts()
            
            print(f"\n📊 {col.upper()}:")
            print(f"   Unique values: {unique_count}")
            print(f"   Most common:")
            
            # Show top 10 most common values
            top_values = value_counts.head(10)
            for value, count in top_values.items():
                percentage = count / len(self.df) * 100
                print(f"     • {value}: {count:,} ({percentage:.1f}%)")
            
            if unique_count > 10:
                print(f"     ... and {unique_count - 10} others")
    
    def numerical_analysis(self):
        """Analyze numerical variables."""
        print("\n🔍 NUMERICAL VARIABLES ANALYSIS")
        print("-" * 40)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            print("No numerical columns found.")
            return
        
        for col in numerical_cols:
            print(f"\n📊 {col.upper()}:")
            
            # Basic statistics
            stats = self.df[col].describe()
            print(f"   Count: {stats['count']:.0f}")
            print(f"   Mean: {stats['mean']:.3f}")
            print(f"   Std: {stats['std']:.3f}")
            print(f"   Min: {stats['min']:.3f}")
            print(f"   25%: {stats['25%']:.3f}")
            print(f"   50%: {stats['50%']:.3f}")
            print(f"   75%: {stats['75%']:.3f}")
            print(f"   Max: {stats['max']:.3f}")
            
            # Check for outliers using IQR method
            Q1 = stats['25%']
            Q3 = stats['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            if len(outliers) > 0:
                print(f"   Outliers: {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)")
    
    def team_analysis(self):
        """Analyze team-specific patterns."""
        if not all(col in self.df.columns for col in ['team_A', 'team_B', 'winner']):
            print("\n⚠️  Team columns not found for team analysis")
            return
        
        print("\n🔍 TEAM ANALYSIS")
        print("-" * 40)
        
        # All teams
        all_teams = pd.concat([self.df['team_A'], self.df['team_B']]).unique()
        print(f"Total unique teams: {len(all_teams)}")
        
        # Team frequency
        team_counts = pd.concat([self.df['team_A'], self.df['team_B']]).value_counts()
        print(f"\n🏆 Most active teams:")
        for team, count in team_counts.head(10).items():
            print(f"   • {team}: {count:,} matches")
        
        # Win rates
        team_stats = {}
        for team in all_teams:
            team_matches = self.df[(self.df['team_A'] == team) | (self.df['team_B'] == team)]
            wins = (team_matches['winner'] == team).sum()
            total = len(team_matches)
            win_rate = wins / total if total > 0 else 0
            team_stats[team] = {'matches': total, 'wins': wins, 'win_rate': win_rate}
        
        # Top win rates (minimum 10 matches)
        qualified_teams = {k: v for k, v in team_stats.items() if v['matches'] >= 10}
        sorted_by_winrate = sorted(qualified_teams.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        
        print(f"\n🥇 Highest win rates (≥10 matches):")
        for team, stats in sorted_by_winrate[:10]:
            print(f"   • {team}: {stats['win_rate']:.1%} ({stats['wins']}/{stats['matches']})")
    
    def map_analysis(self):
        """Analyze map-specific patterns."""
        if 'map' not in self.df.columns:
            print("\n⚠️  Map column not found for map analysis")
            return
        
        print("\n🔍 MAP ANALYSIS")
        print("-" * 40)
        
        # Map frequency
        map_counts = self.df['map'].value_counts()
        print(f"Total unique maps: {len(map_counts)}")
        print(f"\n🗺️  Map frequency:")
        for map_name, count in map_counts.items():
            percentage = count / len(self.df) * 100
            print(f"   • {map_name}: {count:,} ({percentage:.1f}%)")
        
        # Map order analysis
        if 'map_number' in self.df.columns:
            print(f"\n📊 Map order distribution:")
            order_counts = self.df['map_number'].value_counts().sort_index()
            for order, count in order_counts.items():
                percentage = count / len(self.df) * 100
                print(f"   • Map {order}: {count:,} ({percentage:.1f}%)")
    
    def correlation_analysis(self):
        """Analyze correlations between numerical features."""
        print("\n🔍 CORRELATION ANALYSIS")
        print("-" * 40)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            print("Insufficient numerical columns for correlation analysis.")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        # Find strong correlations (> 0.7 or < -0.7)
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        if strong_corr:
            print("🔗 Strong correlations (|r| > 0.7):")
            for col1, col2, corr_val in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                print(f"   • {col1} ↔ {col2}: {corr_val:.3f}")
        else:
            print("No strong correlations found (|r| > 0.7)")
        
        # Show correlation matrix for key features
        key_features = [col for col in ['team_A_rank', 'team_B_rank', 'rank_diff', 'winner_is_A', 
                                       'map_winrate_A', 'map_winrate_B'] if col in numerical_cols]
        
        if len(key_features) > 1:
            print(f"\n📊 Key feature correlations:")
            key_corr = self.df[key_features].corr()
            print(key_corr.round(3).to_string())
    
    def split_analysis(self):
        """Analyze train/test splits if available."""
        split_cols = [col for col in ['split_row', 'split_match'] if col in self.df.columns]
        
        if not split_cols:
            print("\n⚠️  No split columns found")
            return
        
        print("\n🔍 TRAIN/TEST SPLIT ANALYSIS")
        print("-" * 40)
        
        for split_col in split_cols:
            print(f"\n📊 {split_col.upper()} distribution:")
            split_counts = self.df[split_col].value_counts()
            
            for split_name, count in split_counts.items():
                percentage = count / len(self.df) * 100
                
                # Check target balance in each split
                if 'winner_is_A' in self.df.columns:
                    split_data = self.df[self.df[split_col] == split_name]
                    target_balance = split_data['winner_is_A'].mean()
                    print(f"   • {split_name}: {count:,} ({percentage:.1f}%) - Target A wins: {target_balance:.1%}")
                else:
                    print(f"   • {split_name}: {count:,} ({percentage:.1f}%)")
    
    def generate_summary_report(self):
        """Generate executive summary report."""
        print("\n" + "="*80)
        print("📋 DATASET SUMMARY REPORT")
        print("="*80)
        
        # Key metrics
        missing_pct = (self.df.isnull().sum().sum() / (self.n_rows * self.n_cols)) * 100
        
        print(f"📊 Dataset Overview:")
        print(f"   • Shape: {self.n_rows:,} rows × {self.n_cols} columns")
        print(f"   • Missing data: {missing_pct:.2f}% of all cells")
        print(f"   • Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            date_range = (self.df['date'].max() - self.df['date'].min()).days
            print(f"   • Time span: {date_range} days ({self.df['date'].min().date()} to {self.df['date'].max().date()})")
        
        # Data types summary
        dtype_summary = self.df.dtypes.value_counts()
        print(f"\n📋 Data Types:")
        for dtype, count in dtype_summary.items():
            print(f"   • {dtype}: {count} columns")
        
        # Completeness by column type
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0:
            cat_completeness = (self.df[categorical_cols].notna().sum().sum() / 
                              (len(categorical_cols) * self.n_rows)) * 100
            print(f"   • Categorical completeness: {cat_completeness:.1f}%")
        
        if len(numerical_cols) > 0:
            num_completeness = (self.df[numerical_cols].notna().sum().sum() / 
                              (len(numerical_cols) * self.n_rows)) * 100
            print(f"   • Numerical completeness: {num_completeness:.1f}%")
        
        print("\n✅ Analysis complete!")
    
    def run_full_profile(self):
        """Run complete dataset profiling."""
        print("🚀 Starting comprehensive dataset profiling...")
        print("="*80)
        
        try:
            self.basic_info()
            self.missing_values_analysis()
            self.data_quality_checks()
            self.temporal_analysis()
            self.categorical_analysis()
            self.numerical_analysis()
            self.team_analysis()
            self.map_analysis()
            self.correlation_analysis()
            self.split_analysis()
            self.generate_summary_report()
            
        except Exception as e:
            print(f"❌ Error during profiling: {e}")
            raise

def main():
    """Main function to run the profiler."""
    import sys
    
    # Default to the ML-ready dataset
    filepath = "Phase1/map_picks_last6m_top50_ml_ready.csv"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    try:
        profiler = CS2DatasetProfiler(filepath)
        profiler.run_full_profile()
        
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        print("Make sure the dataset exists or provide the correct path.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()