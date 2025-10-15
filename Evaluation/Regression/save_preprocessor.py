"""Fit and persist the ColumnTransformer preprocessor used in the notebook.
This script locates training rows (via 'split_row' if present), fits the ColumnTransformer and saves it as preprocessor.pkl
"""
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_PATH = os.path.join('Phase2','map_picks_last6m_top50_ml_ready.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'map_picks_last6m_top50_ml_ready.csv'
print('Loading data from', DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Features - mirror notebook
feature_cols = [
    'map','map_number','team_A_rank','team_B_rank','rank_diff','abs_rank_diff','picked_by_is_A','is_decider',
    'map_winrate_A','map_winrate_B','recent_form_A','recent_form_B',
    'elo_A','elo_B','map_elo_A','map_elo_B','h2h_rate_A_vs_B','last5_map_rate_A','last5_map_rate_B',
    'elo_diff','map_elo_diff','last5_map_diff','h2h_diff','h2h_raw','map_count_train','map_age_days','new_map_flag','team_A_te','team_B_te',
]
feature_cols = [c for c in feature_cols if c in df.columns]
print('Using features:', feature_cols)

# Construct transformer
categorical_features = ['map'] if 'map' in feature_cols else []
numerical_features = [c for c in feature_cols if c not in categorical_features]
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])

# find training rows
if 'split_row' in df.columns:
    train_mask = df['split_row']=='train'
else:
    # fallback: use 60% of data as train
    train_mask = pd.Series(False, index=df.index)
    train_mask.iloc[:int(0.6*len(df))] = True

X_train = df.loc[train_mask, feature_cols]
print('Fitting preprocessor on', X_train.shape[0], 'rows')
preprocessor.fit(X_train)

out_path = os.path.join('Phase2','Regression','preprocessor.pkl')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
joblib.dump(preprocessor, out_path)
print('Saved preprocessor to', out_path)
